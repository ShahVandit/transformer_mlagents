"""
Extract scalar data (rewards etc.) from ML-Agents TensorBoard event files into
paper-ready CSVs.

Why: each run under results/<run>/Drone/ has one or more
`events.out.tfevents.*` files (a --resume appends a NEW event file rather than
continuing the old one), and the data we want for the paper (Environment/
Cumulative Reward vs. step) lives inside them. This script merges all event
files per run, in step order, deduplicates overlapping steps (later file wins),
and writes:

  1. a tidy long CSV   : run,tag,step,value          (one row per point)
  2. a wide reward CSV : step, <run1>, <run2>, ...    (reward aligned by step)
  3. a summary CSV     : run,last_step,final,max,mean_last10pct  (table numbers)

ML-Agents scalar tags of interest (auto-discovered; these are the common ones):
  Environment/Cumulative Reward     <- the reward curve for the paper
  Environment/Episode Length
  Losses/Policy Loss
  Losses/Value Loss
  Policy/Entropy   Policy/Learning Rate

Usage
-----
  # list every scalar tag available in a run (discover names first)
  python extract_tensorboard.py --list-tags --runs soup_alpha_v1_a0.5

  # extract reward for a chosen set of runs
  python extract_tensorboard.py --runs task1_v8 task2_v3 soup_alpha_v1_a0.5 combined_v4

  # extract reward for every run matching a glob
  python extract_tensorboard.py --runs "soup_alpha_v1_*"

  # extract more than just reward
  python extract_tensorboard.py --runs combined_v4 \
      --tags "Environment/Cumulative Reward" "Environment/Episode Length"

Outputs land in results/paper_data/ by default.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import glob
import os
import sys
from collections import defaultdict

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    sys.exit("[ABORT] tensorboard not installed in this env. "
             "`pip install tensorboard` (it ships with the mlagents env).")

PROJECT = r"C:\Users\shahvandit\Downloads\Drone23_curent\drone_battery1"
RESULTS = os.path.join(PROJECT, "results")
DEFAULT_OUT = os.path.join(RESULTS, "paper_data")
DEFAULT_TAGS = ["Environment/Cumulative Reward"]

# EventAccumulator's default size guidance downsamples scalars; 0 = keep all.
SIZE_GUIDANCE = {"scalars": 0, "tensors": 0}


def find_event_files(run_dir: str) -> list:
    """All event files for a run, whether under run/ or run/Drone/, sorted by
    filename (the trailing timestamp orders resume-appended files)."""
    pats = [
        os.path.join(run_dir, "events.out.tfevents.*"),
        os.path.join(run_dir, "Drone", "events.out.tfevents.*"),
        os.path.join(run_dir, "**", "events.out.tfevents.*"),
    ]
    files = []
    for p in pats:
        files.extend(glob.glob(p, recursive=True))
    return sorted(set(files))


def resolve_runs(patterns: list) -> list:
    """Expand run names / globs against results/ (dirs that contain event files)."""
    all_runs = [d for d in os.listdir(RESULTS)
                if os.path.isdir(os.path.join(RESULTS, d))]
    picked = []
    for pat in patterns:
        matches = [r for r in all_runs if fnmatch.fnmatch(r, pat)]
        if not matches and pat in all_runs:
            matches = [pat]
        picked.extend(matches)
    # dedupe, preserve order
    seen, out = set(), []
    for r in picked:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def load_scalars(run_dir: str, tags: list) -> dict:
    """Merge scalars across all event files for the requested tags.
    Returns {tag: {step: value}} with later files overriding earlier on overlap."""
    merged = defaultdict(dict)          # tag -> {step: value}
    available = set()
    for ef in find_event_files(run_dir):
        acc = EventAccumulator(ef, size_guidance=SIZE_GUIDANCE)
        acc.Reload()
        tag_list = acc.Tags().get("scalars", [])
        available.update(tag_list)
        for tag in tags:
            if tag in tag_list:
                for ev in acc.Scalars(tag):
                    merged[tag][ev.step] = ev.value      # later file wins
    return {t: dict(sorted(v.items())) for t, v in merged.items()}, available


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run names or globs under results/ (e.g. soup_alpha_v1_*)")
    ap.add_argument("--tags", nargs="+", default=DEFAULT_TAGS,
                    help="scalar tags to extract (default: reward)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output dir for CSVs")
    ap.add_argument("--list-tags", action="store_true",
                    help="just print the scalar tags available in each run and exit")
    args = ap.parse_args()

    runs = resolve_runs(args.runs)
    if not runs:
        sys.exit(f"[ABORT] no runs matched {args.runs} under {RESULTS}")
    print(f"[runs] {len(runs)}: {', '.join(runs)}")

    if args.list_tags:
        for run in runs:
            _, available = load_scalars(os.path.join(RESULTS, run), [])
            print(f"\n[{run}] {len(available)} scalar tags:")
            for t in sorted(available):
                print(f"    {t}")
        return

    os.makedirs(args.out, exist_ok=True)
    # run -> tag -> {step: value}
    data = {}
    for run in runs:
        scal, available = load_scalars(os.path.join(RESULTS, run), args.tags)
        missing = [t for t in args.tags if t not in scal]
        if missing:
            print(f"  [warn] {run}: tags not found: {missing} "
                  f"(has {len(available)} tags; try --list-tags)")
        n = {t: len(v) for t, v in scal.items()}
        print(f"  [{run}] points per tag: {n}")
        data[run] = scal

    # ---- 1. tidy long CSV --------------------------------------------------
    long_path = os.path.join(args.out, "scalars_long.csv")
    with open(long_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "tag", "step", "value"])
        for run in runs:
            for tag, series in data[run].items():
                for step, val in series.items():
                    w.writerow([run, tag, step, val])
    print(f"\n[saved] {long_path}")

    # ---- 2. wide reward CSV (one column per run) ---------------------------
    for tag in args.tags:
        # union of all steps for this tag across runs
        steps = sorted({s for run in runs for s in data[run].get(tag, {})})
        if not steps:
            continue
        safe = tag.replace("/", "_").replace(" ", "_")
        wide_path = os.path.join(args.out, f"wide_{safe}.csv")
        with open(wide_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step"] + runs)
            for s in steps:
                w.writerow([s] + [data[run].get(tag, {}).get(s, "") for run in runs])
        print(f"[saved] {wide_path}")

    # ---- 3. summary table (paper numbers) ----------------------------------
    sum_path = os.path.join(args.out, "summary.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "tag", "n_points", "last_step",
                    "final", "max", "mean_last10pct"])
        for run in runs:
            for tag, series in data[run].items():
                if not series:
                    continue
                items = list(series.items())
                vals = [v for _, v in items]
                k = max(1, len(vals) // 10)
                last10 = vals[-k:]
                w.writerow([run, tag, len(vals), items[-1][0],
                            f"{vals[-1]:.4f}", f"{max(vals):.4f}",
                            f"{sum(last10) / len(last10):.4f}"])
    print(f"[saved] {sum_path}")
    print("\n[done] long + wide + summary CSVs written to", args.out)


if __name__ == "__main__":
    main()
