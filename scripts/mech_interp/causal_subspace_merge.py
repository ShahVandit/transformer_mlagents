"""
Causal Subspace Merge (CSM v6, Stage 2) — threshold-free per-direction blend.

Per merged linear (output space, causal Grams from causal_gram.py):

    G      = C_A,PT + C_A,E + C_B,PT + C_B,E     pooled causal Gram
    U      = eigh(G)                             full-rank shared basis
    s_m[d] = u_d^T (C_m,PT + C_m,E) u_d          per-direction causal score
    alpha  = (s_A + eps) / (s_A + s_B + 2 eps)   eps = c_shrink * mean(s)
    W_m    = U ( alpha ⊙ U^T W_A + (1-alpha) ⊙ U^T W_B )

No rel_thresh, no active/null classification, no Gram-Schmidt cascade:
directions causal for A only get A's weights (alpha→1), B only get B's,
both get the score-ratio compromise, neither shrinks to soup (alpha→0.5).

Fused qkv is merged as three independent units (Q/K/V row slices), matching
causal_gram.py's specs. Biases, norms, scales, pos-encoding, action head:
soup. Critic: task2's (fine-tune target regime). Adam state dropped.

--delta-space merges task vectors around pretrain_base instead of raw
weights (ESM-faithful ablation): W = W_base + blend(W_A - W_base, W_B - W_base).
NOTE: the causal Grams are unchanged (they describe each specialist's
activation causality; the task vector is only a different merge coordinate).

Usage
-----
  python causal_subspace_merge.py --save csm_v6
  python causal_subspace_merge.py --nav task1_base --bat task2_base_v2 \\
      --c-shrink 0.05 --save csm_v6
  python causal_subspace_merge.py --delta-space --base pretrain_base --save csm_v6_delta
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from causal_gram import OUT_DIR as GRAM_DIR, linear_specs  # noqa: E402
from patchable_backend import PatchableBackend, BODY       # noqa: E402

PROJECT = r"C:\Users\shahvandit\Downloads\Drone23_curent\drone_battery1"
RESULTS = os.path.join(PROJECT, "results")


def blend_linear(W_A, W_B, C_A, C_B, c_shrink):
    """Per-direction causal soft blend in the pooled eigenbasis.
    Returns (W_merged, stats)."""
    W_A64, W_B64 = W_A.double(), W_B.double()
    G = (C_A + C_B)
    G = 0.5 * (G + G.T)                                   # exact symmetry
    S, U = torch.linalg.eigh(G)                           # ascending
    s_A = torch.einsum("ij,jk,ki->i", U.T, C_A, U).clamp(min=0)
    s_B = torch.einsum("ij,jk,ki->i", U.T, C_B, U).clamp(min=0)
    eps = c_shrink * float((s_A + s_B).mean()) / 2 + 1e-30
    alpha = (s_A + eps) / (s_A + s_B + 2 * eps)           # [d_out]
    Wm = U @ (alpha.unsqueeze(1) * (U.T @ W_A64)
              + (1 - alpha).unsqueeze(1) * (U.T @ W_B64))
    soup = 0.5 * (W_A64 + W_B64)
    stats = {
        "alpha_mean": float(alpha.mean()),
        "alpha_min": float(alpha.min()), "alpha_max": float(alpha.max()),
        "n_A": int((alpha > 0.7).sum()), "n_B": int((alpha < 0.3).sum()),
        "n_soup": int(((alpha - 0.5).abs() < 0.05).sum()), "d": len(alpha),
        "dist_soup": float((Wm - soup).norm()),
    }
    return Wm.to(W_A.dtype), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nav", default="task1_base")
    ap.add_argument("--bat", default="task2_base_v2")
    ap.add_argument("--c-shrink", type=float, default=0.05,
                    help="shrinkage-to-soup strength (eps = c_shrink * mean(s))")
    ap.add_argument("--delta-space", action="store_true",
                    help="blend task vectors around --base instead of raw W")
    ap.add_argument("--base", default="pretrain_base")
    ap.add_argument("--critic", choices=["bat", "nav", "soup"], default="bat")
    ap.add_argument("--save", required=True)
    args = ap.parse_args()

    ck = lambda m: os.path.join(RESULTS, m, "Drone", "checkpoint.pt")
    d_nav = torch.load(ck(args.nav), map_location="cpu")
    d_bat = torch.load(ck(args.bat), map_location="cpu")
    sd_nav, sd_bat = d_nav["Policy"], d_bat["Policy"]
    sd_base = None
    if args.delta_space:
        sd_base = torch.load(ck(args.base), map_location="cpu")["Policy"]

    g_nav = torch.load(os.path.join(GRAM_DIR, f"grams_{args.nav}.pt"),
                       map_location="cpu")
    g_bat = torch.load(os.path.join(GRAM_DIR, f"grams_{args.bat}.pt"),
                       map_location="cpu")
    concepts = list(g_nav["concepts"])

    # backend only for architecture/spec enumeration (no forward)
    backend = PatchableBackend(ck(args.nav))
    specs = linear_specs(backend)

    # ── base: soup everywhere ────────────────────────────────────────────────
    merged = {}
    for k in sd_nav:
        v1, v2 = sd_nav[k], sd_bat[k]
        merged[k] = (0.5 * (v1.float() + v2.float())).to(v1.dtype) \
            if torch.is_tensor(v1) and v1.dtype.is_floating_point \
            and v1.shape == v2.shape else (v2.clone() if torch.is_tensor(v2) else v2)

    print(f"\n{'='*78}\nCausal Subspace Merge (CSM v6)  nav={args.nav}  bat={args.bat}")
    print(f"  c_shrink={args.c_shrink}  delta_space={args.delta_space}"
          + (f"  base={args.base}" if args.delta_space else ""))
    print('='*78)
    hdr = (f"{'linear':<14}{'d':>5}{'a_mean':>8}{'a_min':>7}{'a_max':>7}"
           f"{'nav(>.7)':>9}{'bat(<.3)':>9}{'soup':>6}{'||W-soup||':>11}")
    print(hdr + "\n" + "-" * len(hdr))

    # ── per-linear causal blend; fused qkv reassembled from Q/K/V units ──────
    qkv_parts = {}
    for name, _, wkey, _, osl in specs:
        full_key = BODY + wkey
        W_A_full = sd_nav[full_key].float()
        W_B_full = sd_bat[full_key].float()
        W_A = W_A_full[osl] if osl is not None else W_A_full
        W_B = W_B_full[osl] if osl is not None else W_B_full
        if args.delta_space:
            Wb_full = sd_base[full_key].float()
            Wb = Wb_full[osl] if osl is not None else Wb_full
            W_A, W_B = W_A - Wb, W_B - Wb
        C_A = sum(g_nav["grams"][name][c] for c in concepts)
        C_B = sum(g_bat["grams"][name][c] for c in concepts)
        Wm, st = blend_linear(W_A, W_B, C_A, C_B, args.c_shrink)
        if args.delta_space:
            Wm = (Wm.double() + Wb.double()).to(Wm.dtype)
        print(f"{name:<14}{st['d']:>5}{st['alpha_mean']:>8.3f}{st['alpha_min']:>7.2f}"
              f"{st['alpha_max']:>7.2f}{st['n_A']:>9}{st['n_B']:>9}"
              f"{st['n_soup']:>6}{st['dist_soup']:>11.4f}")
        if osl is not None:
            qkv_parts.setdefault(full_key, {})[name[3]] = Wm  # 'Q'/'K'/'V'
        else:
            merged[full_key] = Wm

    for full_key, parts in qkv_parts.items():
        merged[full_key] = torch.cat([parts["Q"], parts["K"], parts["V"]], dim=0)

    # ── assemble checkpoint (run_soup_sweep.py conventions) ──────────────────
    out = {"Policy": merged}
    if args.critic == "soup":
        c1, c2 = d_nav["Optimizer:critic"], d_bat["Optimizer:critic"]
        out["Optimizer:critic"] = {k: 0.5 * (c1[k] + c2[k])
                                   if torch.is_tensor(c1.get(k))
                                   and c1[k].dtype.is_floating_point
                                   and c1[k].shape == c2[k].shape else c2[k]
                                   for k in c2}
    else:
        out["Optimizer:critic"] = (d_nav if args.critic == "nav"
                                   else d_bat)["Optimizer:critic"]
    for k, v in d_bat.items():
        if k not in ("Policy", "Optimizer:critic", "Optimizer:value_optimizer"):
            out[k] = v

    save_dir = os.path.join(RESULTS, args.save, "Drone")
    os.makedirs(save_dir, exist_ok=True)
    save = os.path.join(save_dir, "checkpoint.pt")
    torch.save(out, save)
    print(f"\n[merge] critic={args.critic}; Adam state dropped")
    print(f"[saved] {save}")

    # sanity: merged model produces finite mu comparable to parents
    be_m = PatchableBackend(out)
    from resid_alignment_check import load_shared_buffer   # noqa: E402
    buf = load_shared_buffer([args.nav, args.bat], 2048, seed=0)
    mu = be_m.forward(buf, capture=[])["mu"]
    mu_n = PatchableBackend(d_nav).forward(buf, capture=[])["mu"]
    print(f"[sanity] mu finite={bool(torch.isfinite(mu).all())}  "
          f"|mu|={float(mu.norm(dim=1).mean()):.3f}  "
          f"(nav parent |mu|={float(mu_n.norm(dim=1).mean()):.3f})")


if __name__ == "__main__":
    main()
