"""
Causal graft merge: theta_merged = theta_nav + M ⊙ (theta_batt - theta_base).

Uses the per-weight causal attribution from causal_weight_attribution.py to build
a sparse mask M of the top-|A| battery weights, grafts ONLY those weight changes
onto the navigator, and sweeps the mask size to trace the merge tradeoff:

  - battery GAIN : mean_x <mu(merged; batt_obs), u_batt>  -> rises toward task_batt
  - nav   COST   : mean_x ||mu(merged; nav_obs) - mu_nav(nav_obs)||  -> rises as we
                   overwrite navigator weights

The shape of (gain vs cost) is the weight-level entanglement, measured causally:
if a small top-k graft captures most battery behavior at near-zero nav cost, the
skills live in distinct weights and the surgical merge works. If nav cost rises in
lockstep with battery gain, the skills share weights and no sparse graft separates
them.

This is the merge operator none of soup/Fisher/AIM/ESM produce: a causal,
intervention-validated, per-weight task graft. Saves a runnable merged checkpoint
at --graft-frac for env evaluation (the final, non-proxy proof).

Usage
-----
  python causal_graft_merge.py                          # defaults below
  python causal_graft_merge.py --nav task1_v8 --graft-frac 0.05 --save merged_causal_v1
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from causal_weight_attribution import (        # noqa: E402
    RESULTS, CAP_DIR, CAPTURE_FILES, BODY, MU_W, MU_B,
    load_weights, model_dims, mu_only, attrib_keys,
)


def load_obs(model, dev, max_tokens, seq_len):
    cap = next((os.path.join(CAP_DIR, model, f) for f in CAPTURE_FILES
                if os.path.exists(os.path.join(CAP_DIR, model, f))), None)
    if cap is None:
        sys.exit(f"[ABORT] no capture in {os.path.join(CAP_DIR, model)}")
    d = torch.load(cap, map_location="cpu")
    obs = d.get("obs_window")
    if obs is None:
        obs = d["obs"].float().unsqueeze(1).repeat(1, seq_len, 1)
    obs = obs.float()
    if len(obs) > max_tokens:
        obs = obs[torch.randperm(len(obs))[:max_tokens]]
    return obs.to(dev)


def build_mask(A, frac, dev):
    """Global top-`frac` of |A| across all attributed weights -> per-key bool."""
    keys = list(A.keys())
    flat = torch.cat([A[k].abs().reshape(-1) for k in keys]).to(dev)
    n = flat.numel()
    k = int(frac * n)
    if k <= 0:
        return {kk: torch.zeros_like(A[kk], dtype=torch.bool, device=dev)
                for kk in keys}, 0.0
    thresh = torch.topk(flat, k).values.min()
    return ({kk: (A[kk].abs().to(dev) >= thresh) for kk in keys},
            float(thresh))


def merged_weights(w_nav, w_base, w_task, mask):
    """theta_nav + M ⊙ (theta_task - theta_base), keys = attributed set."""
    out = {k: w_nav[k].clone() for k in w_nav}
    for k in mask:
        out[k] = w_nav[k] + mask[k].float() * (w_task[k] - w_base[k])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="task1_v7")
    ap.add_argument("--task", default="task2_v3", help="battery specialist")
    ap.add_argument("--nav", default="task1_v7",
                    help="navigator to graft onto (default=base)")
    ap.add_argument("--attrib", default=None,
                    help="attribution .pt (default: weight_attrib_<base>_to_<task>.pt)")
    ap.add_argument("--batt-capture", default=None, help="default=--task")
    ap.add_argument("--nav-capture", default=None, help="default=--nav")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--graft-frac", type=float, default=0.05,
                    help="mask fraction for the saved merged checkpoint")
    ap.add_argument("--save", default=None,
                    help="name -> results/<name>/Drone/checkpoint.pt")
    ap.add_argument("--n-head", type=int, default=4)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    attrib = args.attrib or os.path.join(
        RESULTS, "mech_interp",
        f"weight_attrib_{args.base}_to_{args.task}.pt")
    if not os.path.exists(attrib):
        sys.exit(f"[ABORT] missing attribution {attrib}\n"
                 "Run causal_weight_attribution.py first.")
    AD = torch.load(attrib, map_location="cpu")
    A = AD["A"]
    print(f"[attrib] {attrib}  g_gap={AD['g_gap']:.4f}")

    w_base = load_weights(args.base, dev)
    w_task = load_weights(args.task, dev)
    w_nav = load_weights(args.nav, dev)
    d_model, n_layer, head_dim = model_dims(w_base, args.n_head)
    dims = {"d_model": d_model, "n_layer": n_layer, "n_head": args.n_head,
            "head_dim": head_dim}
    seq_len = w_base["temporal_pos_encoding"].shape[1]

    batt_obs = load_obs(args.batt_capture or args.task, dev,
                        args.max_tokens, seq_len)
    nav_obs = load_obs(args.nav_capture or args.nav, dev,
                       args.max_tokens, seq_len)
    print(f"[obs] battery={tuple(batt_obs.shape)} nav={tuple(nav_obs.shape)}")

    # battery behavior-change direction (base -> task), on battery obs
    mu_b0 = mu_only(w_base, batt_obs, dims)
    mu_bt = mu_only(w_task, batt_obs, dims)
    dmu = mu_bt - mu_b0
    u_batt = dmu / dmu.norm(dim=1, keepdim=True).clamp(min=1e-8)
    g_full = (mu_bt * u_batt).sum().item() / batt_obs.shape[0]   # task2_v3 ceiling
    g_navonly = (mu_b0 * u_batt).sum().item() / batt_obs.shape[0]  # base floor

    # nav reference behavior (the navigator we graft onto), on nav obs
    mu_nav_ref = mu_only(w_nav, nav_obs, dims)

    print(f"\n  battery gain range: floor(base)={g_navonly:.3f} "
          f"ceiling(task2_v3)={g_full:.3f}")
    print("  graft = theta_nav + top-frac|A|(battery delta)\n")
    print(f"  {'frac':>6} {'batt_gain':>10} {'gain_%':>7} {'nav_cost':>9}")

    fracs = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 1.0]
    rows = []
    for f in fracs:
        mask, _ = build_mask(A, f, dev)
        wm = merged_weights(w_nav, w_base, w_task, mask)
        mu_b = mu_only(wm, batt_obs, dims)
        g = (mu_b * u_batt).sum().item() / batt_obs.shape[0]
        gain_pct = (g - g_navonly) / max(g_full - g_navonly, 1e-8)
        mu_n = mu_only(wm, nav_obs, dims)
        nav_cost = (mu_n - mu_nav_ref).norm(dim=1).mean().item()
        rows.append((f, g, gain_pct, nav_cost))
        print(f"  {f:>6.3f} {g:>10.3f} {gain_pct:>6.1%} {nav_cost:>9.4f}")

    # efficiency: battery gain per unit nav cost at small grafts
    print("\n  Read: if a small frac reaches high gain_% at low nav_cost, the")
    print("  battery and nav skills occupy distinct weights (clean surgical merge).")

    if args.save:
        f = args.graft_frac
        mask, thr = build_mask(A, f, dev)
        wm = merged_weights(w_nav, w_base, w_task, mask)
        # write a runnable checkpoint: start from nav's full ckpt, overwrite Policy
        nav_ck = torch.load(os.path.join(RESULTS, args.nav, "Drone",
                                         "checkpoint.pt"), map_location="cpu")
        pol = dict(nav_ck["Policy"])
        for k, v in wm.items():
            if k == "mu.weight":
                pol[MU_W] = v.cpu()
            elif k == "mu.bias":
                pol[MU_B] = v.cpu()
            else:
                pol[BODY + k] = v.cpu()
        nav_ck["Policy"] = pol
        outdir = os.path.join(RESULTS, args.save, "Drone")
        os.makedirs(outdir, exist_ok=True)
        outp = os.path.join(outdir, "checkpoint.pt")
        torch.save(nav_ck, outp)
        n_grafted = int(sum(m.sum() for m in mask.values()))
        print(f"\n[saved] {outp}  (graft_frac={f}, {n_grafted} weights grafted)")
        print("  -> evaluate in env to confirm battery management + navigation")


if __name__ == "__main__":
    main()
