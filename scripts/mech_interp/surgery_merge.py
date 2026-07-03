"""
Ownership surgery merge: repair the soup's task-specific damage unit by unit.

Base = uniform soup (alpha-blend of the two specialists). Then, for every
surgery unit whose ownership was established by ownership_map.py:

  PT-owned      -> overwrite the unit's weights with the NAV specialist's
  energy-owned  -> overwrite with the BAT specialist's
  shared        -> soft blend, w = contrast-based PT share S_PT/(S_PT+S_E)
                   (continuous, no hard threshold — hard cuts killed CSM)
  inert         -> stays soup

Surgery units and their weight-space referents (both-or-neither: a unit's
read and write halves always come from the same donor):

  ffn_hidden.{i} neuron n:
      network_body.ffn_layers.{i}.0.weight  row n      (read)
      network_body.ffn_layers.{i}.0.bias    entry n
      network_body.ffn_layers.{i}.3.weight  column n   (write)
  attention head h of layer i (head_dim = d_model/n_head):
      network_body.qkv_layers.{i}.weight    rows Q/K/V of head h
          Q rows [h*hd,(h+1)*hd), K rows [d+h*hd, d+(h+1)*hd),
          V rows [2d+h*hd, 2d+(h+1)*hd)         (read)
      network_body.qkv_layers.{i}.bias      same rows
      network_body.attn_out_layers.{i}.weight columns [h*hd,(h+1)*hd) (write)

Everything else (input_proj, LayerNorms, attn/ffn scales — per-layer SCALARS,
not splittable — positional encoding, action head) stays soup. Critic:
--critic {bat,nav,soup}, default bat (the fine-tune target regime).
Adam state is dropped (matches run_soup_sweep.py convention).

Cross-model map combination (a unit index is one physical location in the
merged net, but each specialist voted separately):
  either model says shared            -> shared
  models claim OPPOSITE concepts      -> shared (conflict)
  any model claims concept C, none disagrees -> C-owned
  both inert                          -> inert
Alignment gate: if resid_alignment_check.py marked the unit's INPUT basis
boundary (ffn_in.{i} for FFN units, qkv_in.{i} for heads) as not index-
aligned, owned units fall back to soup there (a transplanted row would read
features that don't mean what they meant in the donor).

Usage
-----
  python surgery_merge.py --save surgery_v1
  python surgery_merge.py --nav task1_v8 --bat task2_v3 --alpha 0.5 \\
      --save surgery_v1 --no-align-gate
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

PROJECT = r"C:\Users\shahvandit\Downloads\Drone23_curent\drone_battery1"
RESULTS = os.path.join(PROJECT, "results")
OWN_DIR = os.path.join(RESULTS, "mech_interp", "ownership")
BODY = "network_body."

PT, EN, SHARED, INERT = 0, 1, 2, -1


def combine_labels(la: torch.Tensor, lb: torch.Tensor) -> torch.Tensor:
    """Merge the two specialists' votes for the same physical unit index."""
    out = torch.full_like(la, INERT)
    claims_pt = (la == PT) | (lb == PT)
    claims_en = (la == EN) | (lb == EN)
    any_shared = (la == SHARED) | (lb == SHARED)
    out[claims_pt & ~claims_en & ~any_shared] = PT
    out[claims_en & ~claims_pt & ~any_shared] = EN
    out[any_shared | (claims_pt & claims_en)] = SHARED
    return out


def combine_soft_w(map_a, map_b, bnd):
    """Average the two models' contrast-based PT shares, counting only models
    that have signal (above-null screen for either concept) at that unit."""
    wa, wb = map_a["soft_w"][bnd], map_b["soft_w"][bnd]
    sa = map_a["signal"][bnd].float()
    sb = map_b["signal"][bnd].float()
    den = sa + sb
    num = wa * sa + wb * sb
    return torch.where(den > 0, num / den.clamp(min=1), torch.full_like(wa, 0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nav", default="task1_v8")
    ap.add_argument("--bat", default="task2_v3")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="soup base: alpha*nav + (1-alpha)*bat. NOTE: this is "
                         "the NAV share (run_soup_sweep.py convention); the "
                         "paper's Eq.1 alpha is the BAT share, i.e. "
                         "paper_alpha = 1 - this")
    ap.add_argument("--critic", choices=["bat", "nav", "soup"], default="bat")
    ap.add_argument("--no-align-gate", action="store_true",
                    help="ignore the resid alignment gate (surgery everywhere)")
    ap.add_argument("--save", required=True, help="run name for the merged ckpt")
    args = ap.parse_args()

    ck = lambda m: os.path.join(RESULTS, m, "Drone", "checkpoint.pt")
    d_nav = torch.load(ck(args.nav), map_location="cpu")
    d_bat = torch.load(ck(args.bat), map_location="cpu")
    map_nav = torch.load(os.path.join(OWN_DIR, f"ownership_{args.nav}.pt"),
                         map_location="cpu")
    map_bat = torch.load(os.path.join(OWN_DIR, f"ownership_{args.bat}.pt"),
                         map_location="cpu")
    align_path = os.path.join(OWN_DIR, f"alignment_{args.nav}__{args.bat}.pt")
    aligned = {}
    if os.path.exists(align_path):
        al = torch.load(align_path, map_location="cpu")
        aligned = {b: v["aligned"] for b, v in al["boundaries"].items()}
    elif not args.no_align_gate:
        sys.exit(f"[ABORT] no alignment file {align_path}; run "
                 f"resid_alignment_check.py first or pass --no-align-gate")

    sd_nav, sd_bat = d_nav["Policy"], d_bat["Policy"]
    hd = map_nav["head_dim"]
    n_head = map_nav["n_head"]
    d_model = sd_nav[BODY + "input_proj.weight"].shape[0]
    n_layer = len({k for k in sd_nav if "qkv_layers" in k and "weight" in k})

    # ── base: soup everywhere ────────────────────────────────────────────────
    a = args.alpha
    merged = {}
    for k in sd_nav:
        v1, v2 = sd_nav[k], sd_bat[k]
        merged[k] = (a * v1 + (1 - a) * v2 if torch.is_tensor(v1)
                     and v1.dtype.is_floating_point and v1.shape == v2.shape
                     else v2.clone() if torch.is_tensor(v2) else v2)

    def donor_blend(k, sel, w):
        """merged[k][sel] = w*nav + (1-w)*bat (w scalar or broadcastable)."""
        merged[k] = merged[k].clone()
        merged[k][sel] = w * sd_nav[k][sel] + (1 - w) * sd_bat[k][sel]

    stats = {"PT": 0, "energy": 0, "shared": 0, "inert": 0, "gated": 0}

    # Gate BOTH halves of the unit: the read side (transplanted row reads its
    # input basis by index) AND the write side (transplanted column writes into
    # resid.{i+1} by index) must be aligned for surgery to be valid.
    def gate(read_bnd, i):
        return args.no_align_gate or (aligned.get(read_bnd, False)
                                      and aligned.get(f"resid.{i + 1}", False))

    # ── FFN hidden neurons ───────────────────────────────────────────────────
    for i in range(n_layer):
        bnd = f"ffn_hidden.{i}"
        gate_ok = gate(f"ffn_in.{i}", i)
        lab = combine_labels(map_nav["labels"][bnd], map_bat["labels"][bnd])
        soft = combine_soft_w(map_nav, map_bat, bnd)
        k_in_w = f"{BODY}ffn_layers.{i}.0.weight"
        k_in_b = f"{BODY}ffn_layers.{i}.0.bias"
        k_out = f"{BODY}ffn_layers.{i}.3.weight"
        for n in range(len(lab)):
            l = int(lab[n])
            if l == INERT:
                stats["inert"] += 1
                continue
            if not gate_ok:
                stats["gated"] += 1
                continue
            w = 1.0 if l == PT else 0.0 if l == EN else float(soft[n])
            stats["PT" if l == PT else "energy" if l == EN else "shared"] += 1
            donor_blend(k_in_w, n, w)                       # read row
            if k_in_b in sd_nav:
                donor_blend(k_in_b, n, w)
            donor_blend(k_out, (slice(None), n), w)         # write column

    # ── attention heads ──────────────────────────────────────────────────────
    for i in range(n_layer):
        bnd = f"attn_in.{i}"
        gate_ok = gate(f"qkv_in.{i}", i)
        lab = combine_labels(map_nav["labels"][bnd], map_bat["labels"][bnd])
        soft = combine_soft_w(map_nav, map_bat, bnd)
        k_qkv_w = f"{BODY}qkv_layers.{i}.weight"
        k_qkv_b = f"{BODY}qkv_layers.{i}.bias"
        k_o = f"{BODY}attn_out_layers.{i}.weight"
        for h in range(n_head):
            l = int(lab[h])
            if l == INERT:
                stats["inert"] += 1
                continue
            if not gate_ok:
                stats["gated"] += 1
                continue
            w = 1.0 if l == PT else 0.0 if l == EN else float(soft[h])
            stats["PT" if l == PT else "energy" if l == EN else "shared"] += 1
            rows = torch.cat([torch.arange(q * d_model + h * hd,
                                           q * d_model + (h + 1) * hd)
                              for q in range(3)])
            donor_blend(k_qkv_w, rows, w)
            if k_qkv_b in sd_nav:
                donor_blend(k_qkv_b, rows, w)
            donor_blend(k_o, (slice(None), slice(h * hd, (h + 1) * hd)), w)

    # ── assemble checkpoint (run_soup_sweep.py conventions) ──────────────────
    out = {"Policy": merged}
    if args.critic == "soup":
        c1, c2 = d_nav["Optimizer:critic"], d_bat["Optimizer:critic"]
        out["Optimizer:critic"] = {k: a * c1[k] + (1 - a) * c2[k]
                                   if c1[k].dtype.is_floating_point
                                   and c1[k].shape == c2[k].shape else c2[k]
                                   for k in c2}
    else:
        src = d_nav if args.critic == "nav" else d_bat
        out["Optimizer:critic"] = src["Optimizer:critic"]
    # drop Adam state; keep metadata from bat
    for k, v in d_bat.items():
        if k not in ("Policy", "Optimizer:critic", "Optimizer:value_optimizer"):
            out[k] = v

    save_dir = os.path.join(RESULTS, args.save, "Drone")
    os.makedirs(save_dir, exist_ok=True)
    save = os.path.join(save_dir, "checkpoint.pt")
    torch.save(out, save)

    print(f"[surgery] units: PT-owned={stats['PT']}  energy-owned={stats['energy']}  "
          f"shared(soft-w)={stats['shared']}  inert(soup)={stats['inert']}  "
          f"alignment-gated(soup)={stats['gated']}")
    print(f"[surgery] base soup alpha={a} (nav share); critic={args.critic}; "
          f"Adam state dropped")
    print(f"[saved] {save}")


if __name__ == "__main__":
    main()
