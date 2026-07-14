"""
Causal direction census (CSM v6, Stage 3) — THE goal metric.

For each checkpoint, count the activation directions causally related to each
concept's input observations: eigenvalues of the concept's causal Gram
(causal_gram.py) that exceed a rank-matched placebo null (95th percentile of
the sorted spectra of same-size random non-concept dim groups).

Success criterion: the merged model's census ≈ the UNION of the specialists'
censuses (task1_base contributes the PT directions, task2_base_v2 the energy
directions), while plain soup shows fewer surviving causal directions.

The same shared buffer, seed, placebo groups and repeats are used for every
checkpoint, so counts are directly comparable across models.

Usage
-----
  python causal_census.py --runs csm_v6 soup_alpha_base_a0.5 task1_base task2_base_v2
  python causal_census.py --runs csm_v6 --reuse       # reuse saved gram files
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from causal_gram import (OUT_DIR as GRAM_DIR, build_causal_grams,  # noqa: E402
                         make_groups)
from ownership_map import CONCEPTS                                  # noqa: E402
from patchable_backend import PatchableBackend                      # noqa: E402
from resid_alignment_check import load_shared_buffer                # noqa: E402

PROJECT = r"C:\Users\shahvandit\Downloads\Drone23_curent\drone_battery1"
RESULTS = os.path.join(PROJECT, "results")
OUT_DIR = os.environ.get(
    "MECH_CENSUS_DIR",
    os.path.join(RESULTS, "mech_interp", "causal_census"),
)


def census_from_grams(grams: dict, groups: dict):
    """{linear: {concept: (count, spectrum, null_spectrum_q95)}}.
    Null is rank-matched: the q95 across placebo Grams of the k-th largest
    eigenvalue, compared against the concept spectrum's k-th eigenvalue."""
    out = {}
    for name, per_group in grams.items():
        out[name] = {}
        for c in CONCEPTS:
            lam = torch.linalg.eigvalsh(per_group[c]).flip(0)   # descending
            plc = [torch.linalg.eigvalsh(per_group[g]).flip(0)
                   for g in groups if g.startswith(f"plc_{c}_")]
            null = torch.stack(plc).quantile(0.95, dim=0)
            out[name][c] = (int((lam > null).sum()), lam, null)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run names under results/ (checkpoint at <run>/Drone/checkpoint.pt)")
    ap.add_argument("--buffer-models", nargs="+",
                    default=["soup_alpha_v2_a0.5"],
                    help="one or more capture folders whose obs_window tensors form the shared buffer")
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=40_000)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--n-placebo", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reuse", action="store_true",
                    help="load saved grams_<run>.pt when present instead of recomputing")
    args = ap.parse_args()

    buf = load_shared_buffer(args.buffer_models, args.max_tokens, seed=args.seed)
    print(f"[buffer] {tuple(buf.shape)}  runs={args.runs}")

    all_census, all_grams = {}, {}
    for run in args.runs:
        saved = os.path.join(GRAM_DIR, f"grams_{run}.pt")
        if args.reuse and os.path.exists(saved):
            print(f"\n[{run}] reusing {saved}")
            g = torch.load(saved, map_location="cpu")
            grams, groups = g["grams"], g["groups"]
        else:
            ckpt = os.path.join(RESULTS, run, "Drone", "checkpoint.pt")
            backend = PatchableBackend(ckpt, n_head=args.n_head)
            print(f"\n[{run}] building causal grams ({backend})")
            groups = make_groups(backend.obs_dim, args.n_placebo, seed=args.seed)
            g = build_causal_grams(backend, buf, groups, repeats=args.repeats,
                                   chunk=args.chunk, seed=args.seed,
                                   verbose=False)
            grams = g["grams"]
        all_grams[run] = grams
        all_census[run] = census_from_grams(grams, groups)

    # ── the table ────────────────────────────────────────────────────────────
    concepts = list(CONCEPTS)
    linears = list(next(iter(all_census.values())))
    for c in concepts:
        print(f"\n[census] causal directions for concept '{c}' "
              f"(eig > rank-matched placebo q95)")
        hdr = f"{'linear':<14}" + "".join(f"{r[:16]:>18}" for r in args.runs)
        print(hdr + "\n" + "-" * len(hdr))
        totals = {r: 0 for r in args.runs}
        for name in linears:
            row = f"{name:<14}"
            for r in args.runs:
                n = all_census[r][name][c][0]
                totals[r] += n
                row += f"{n:>18}"
            print(row)
        print("-" * len(hdr))
        print(f"{'TOTAL':<14}" + "".join(f"{totals[r]:>18}" for r in args.runs))

    os.makedirs(OUT_DIR, exist_ok=True)
    save = os.path.join(OUT_DIR, "census_" + "__".join(args.runs)[:80] + ".pt")
    torch.save({"runs": args.runs, "census": all_census,
                "buffer_models": args.buffer_models, "seed": args.seed,
                "max_tokens": args.max_tokens, "repeats": args.repeats,
                "n_placebo": args.n_placebo}, save)
    print(f"\n[saved] {save}")


if __name__ == "__main__":
    main()
