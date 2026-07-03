"""
Cross-model index-alignment precondition check (go/no-go gate for surgery).

Ownership surgery transplants FFN rows / head slices from a specialist into the
soup. A transplanted W_in row reads the residual stream by INDEX, so the cut is
only valid where the two specialists' residual features are index-aligned:
feature k in task1's resid must be (approximately) feature k in task2's resid.
Shared init + staged fine-tuning makes this plausible; 30M divergent PPO steps
can still rotate features. Measure, don't assume.

Method: push the SAME obs buffer (union of both models' captures) through both
checkpoints, and for every boundary compute the per-dimension Pearson
correlation between model A's and model B's activation of that dimension.
High median r at a boundary -> index-aligned there -> surgery valid.
Low r -> a transplanted row would read features that no longer mean what they
meant in the donor -> that boundary falls back to soup / soft-w.

Prior evidence (cka_internal.py): attn.1 CKA=0.23, attn.0=0.35 between the
specialists -> expect partial failure; the per-dim view localizes it.

Usage
-----
  python resid_alignment_check.py                       # task1_v8 vs task2_v3
  python resid_alignment_check.py --model-a task1_v8 --model-b task2_v3
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from patchable_backend import PatchableBackend            # noqa: E402

PROJECT = r"C:\Users\shahvandit\Downloads\Drone23_curent\drone_battery1"
RESULTS = os.path.join(PROJECT, "results")
CAP_DIR = os.path.join(RESULTS, "mech_interp", "captures")
OUT_DIR = os.path.join(RESULTS, "mech_interp", "ownership")
CAPTURE_FILE = "activations_combined.pt"


def load_shared_buffer(models, max_tokens, seed=0):
    """Union of both models' captured obs windows -> one buffer covering both
    task regimes, pushed identically through both checkpoints."""
    parts = []
    for m in models:
        p = os.path.join(CAP_DIR, m, CAPTURE_FILE)
        if not os.path.exists(p):
            sys.exit(f"[ABORT] missing capture: {p}")
        d = torch.load(p, map_location="cpu")
        parts.append(d["obs_window"].float())
    buf = torch.cat(parts)
    g = torch.Generator().manual_seed(seed)
    if len(buf) > max_tokens:
        buf = buf[torch.randperm(len(buf), generator=g)[:max_tokens]]
    return buf


def per_dim_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pearson r per column between two [N, d] activation matrices."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    denom = a.std(0).clamp(min=1e-6) * b.std(0).clamp(min=1e-6)
    return (a * b).mean(0) / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-a", default="task1_v8")
    ap.add_argument("--model-b", default="task2_v3")
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=60_000)
    ap.add_argument("--gate", type=float, default=0.5,
                    help="median |r| below this marks the boundary NOT "
                         "index-aligned (surgery falls back to soup there)")
    args = ap.parse_args()

    ck = lambda m: os.path.join(RESULTS, m, "Drone", "checkpoint.pt")
    ba = PatchableBackend(ck(args.model_a), n_head=args.n_head)
    bb = PatchableBackend(ck(args.model_b), n_head=args.n_head)
    print(ba, "\n", bb, sep="")

    buf = load_shared_buffer([args.model_a, args.model_b], args.max_tokens)
    print(f"[buffer] shared obs windows: {tuple(buf.shape)}")

    caps_a = ba.forward(buf)
    caps_b = bb.forward(buf)

    hdr = f"{'boundary':<16}{'d':>6}{'median|r|':>11}{'q25':>8}{'q75':>8}{'frac<gate':>11}{'aligned?':>10}"
    print("\n" + hdr)
    print("-" * len(hdr))
    out = {"model_a": args.model_a, "model_b": args.model_b,
           "gate": args.gate, "boundaries": {}}
    for bnd in ba.boundaries():
        r = per_dim_corr(caps_a[bnd], caps_b[bnd]).abs()
        med = r.median().item()
        q25, q75 = r.quantile(0.25).item(), r.quantile(0.75).item()
        frac_low = (r < args.gate).float().mean().item()
        ok = med >= args.gate
        print(f"{bnd:<16}{r.numel():>6}{med:>11.3f}{q25:>8.3f}{q75:>8.3f}"
              f"{frac_low:>11.2%}{'YES' if ok else 'NO':>10}")
        out["boundaries"][bnd] = {"r": r, "median": med, "aligned": ok}

    os.makedirs(OUT_DIR, exist_ok=True)
    save = os.path.join(OUT_DIR, f"alignment_{args.model_a}__{args.model_b}.pt")
    torch.save(out, save)
    print(f"\n[saved] {save}")
    print("  surgery is index-valid only at boundaries marked YES; "
          "NO boundaries fall back to soup/soft-w in surgery_merge.py.")


if __name__ == "__main__":
    main()
