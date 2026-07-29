from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import data as D  # noqa: E402


FIXED_INSULIN_EDGES = [0.25, 0.75, 1.50]
FIXED_INSULIN_LABELS = ["insulin_le0p25", "insulin_0p25_0p75", "insulin_0p75_1p5", "insulin_gt1p5"]


def cgm_band(value: float) -> str:
    if not np.isfinite(value):
        return "missing"
    if value < 70:
        return "lt70"
    if value <= 180:
        return "70_180"
    if value <= 250:
        return "181_250"
    return "gt250"


def trend_band(delta: float) -> str:
    if not np.isfinite(delta):
        return "missing"
    if delta <= -15:
        return "fall"
    if delta >= 15:
        return "rise"
    return "flat"


def bin_by_edges(values: pd.Series, edges: list[float]) -> np.ndarray:
    return np.digitize(values.to_numpy(float), np.asarray(edges, dtype=float), right=True)


def action_distribution(frame: pd.DataFrame, action_col: str, labels: list[str]) -> pd.DataFrame:
    rows = []
    for split, group in frame.groupby("split", dropna=False):
        counts = np.bincount(group[action_col].to_numpy(int), minlength=len(labels))
        total = max(int(counts.sum()), 1)
        for action, label in enumerate(labels):
            rows.append(
                {
                    "split": split,
                    "action": action,
                    "label": label,
                    "count": int(counts[action]),
                    "frac": float(counts[action] / total),
                    "percent": float(100 * counts[action] / total),
                }
            )
    return pd.DataFrame(rows)


def split_metrics(dist: pd.DataFrame) -> dict[str, float]:
    out = {}
    for split, group in dist.groupby("split"):
        probs = group.sort_values("action").frac.to_numpy(float)
        out[f"{split}_majority_frac"] = float(probs.max()) if len(probs) else np.nan
        out[f"{split}_min_frac"] = float(probs.min()) if len(probs) else np.nan
        nz = probs[probs > 0]
        out[f"{split}_entropy_norm"] = float(-(nz * np.log(nz)).sum() / np.log(len(probs))) if len(nz) and len(probs) > 1 else 0.0
    if {"train", "test"}.issubset(set(dist.split)):
        train = dist[dist.split == "train"].sort_values("action").frac.to_numpy(float)
        test = dist[dist.split == "test"].sort_values("action").frac.to_numpy(float)
        out["train_test_tvd"] = float(0.5 * np.abs(train - test).sum()) if len(train) == len(test) else np.nan
    return out


def support_metrics(frame: pd.DataFrame, action_col: str, min_examples: int) -> dict[str, float]:
    if frame.empty:
        return {"state_cells": 0, "support_frac": 0.0, "multi_action_cells": 0}
    counts = (
        frame.groupby(["cgm_band", "trend_band", "carbs_recent", action_col], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    cell_totals = counts.groupby(["cgm_band", "trend_band", "carbs_recent"])["n"].sum().rename("cell_n")
    supported = counts[counts.n >= min_examples]
    action_counts = supported.groupby(["cgm_band", "trend_band", "carbs_recent"]).size().rename("supported_actions")
    cells = pd.concat([cell_totals, action_counts], axis=1).fillna({"supported_actions": 0})
    good = cells[cells.supported_actions >= 2]
    return {
        "state_cells": int(len(cells)),
        "multi_action_cells": int(len(good)),
        "support_frac": float(good.cell_n.sum() / max(cells.cell_n.sum(), 1)),
    }


def collect_windows(args) -> pd.DataFrame:
    dataset = ds.dataset(args.parquet, format="parquet")
    columns = sorted(set(D.REQUIRED_COLUMNS) | (set(D.OPTIONAL_COLUMNS) & set(dataset.schema.names)))
    scanner = dataset.scanner(columns=columns, batch_size=args.batch_size, use_threads=True)
    rows = []
    tails: dict[str, pd.DataFrame] = {}
    subject_counts: defaultdict[str, int] = defaultdict(int)
    tail_steps = args.history_steps + args.action_steps
    started = time.perf_counter()

    for batch_idx, record_batch in enumerate(scanner.to_batches(), start=1):
        frame = D._prepare(record_batch.to_pandas())
        for (source, sid), group in frame.groupby(["source_file", "id"], sort=False):
            key = f"{source}::{sid}"
            old_tail = tails.get(key)
            if old_tail is not None:
                group = pd.concat([old_tail, group], ignore_index=True)
                new_mask = np.r_[np.zeros(len(old_tail), dtype=bool), np.ones(len(group) - len(old_tail), dtype=bool)]
            else:
                new_mask = np.ones(len(group), dtype=bool)
            tails[key] = group.tail(tail_steps).copy()
            if bool(group.subject_split_across_traintest.astype(bool).any()):
                continue
            remaining = args.max_subject_windows - subject_counts[key]
            if remaining <= 0:
                continue

            cgm = group.CGM.to_numpy(float)
            basal = group.basal.ffill().fillna(0).clip(lower=0).to_numpy(float)
            bolus = group.bolus.fillna(0).clip(lower=0).to_numpy(float)
            insulin = group.insulin.fillna(0).clip(lower=0).to_numpy(float)
            carbs = group.carbs.fillna(0).clip(lower=0).to_numpy(float)
            dates = group.date.to_numpy()
            split = D.stable_split(key, bool(group.is_test.astype(bool).any()))
            added = 0
            for t in range(args.history_steps - 1, len(group) - args.action_steps, args.stride_steps):
                if not new_mask[t]:
                    continue
                action_end = t + args.action_steps
                minutes = (dates[action_end] - dates[t]) / np.timedelta64(1, "m")
                if not 20 <= float(minutes) <= 45 or not np.isfinite(cgm[t]):
                    continue
                bolus_units = float(np.nansum(bolus[t + 1: action_end + 1]))
                basal_units = float(np.nansum(basal[t + 1: action_end + 1]))
                insulin_units = float(np.nansum(insulin[t + 1: action_end + 1]))
                trend_delta = cgm[t] - cgm[max(0, t - 6)] if np.isfinite(cgm[max(0, t - 6)]) else np.nan
                rows.append(
                    {
                        "source_file": source,
                        "id": sid,
                        "split": split,
                        "date": group.date.iloc[t],
                        "cgm": float(cgm[t]),
                        "cgm_band": cgm_band(float(cgm[t])),
                        "trend_delta_30": float(trend_delta) if np.isfinite(trend_delta) else np.nan,
                        "trend_band": trend_band(float(trend_delta)),
                        "carbs_recent": "carbs" if float(np.nansum(carbs[max(0, t - 5): t + 1])) > 0 else "no_carbs",
                        "bolus_units": bolus_units,
                        "basal_units": basal_units,
                        "insulin_units": insulin_units,
                        "bolus4": D._bolus_bin(bolus_units),
                    }
                )
                subject_counts[key] += 1
                added += 1
                if added >= remaining or len(rows) >= args.max_windows:
                    break
            if len(rows) >= args.max_windows:
                break
        if batch_idx == 1 or batch_idx % 5 == 0:
            print(f"[audit] batch={batch_idx} windows={len(rows):,} elapsed={time.perf_counter() - started:.1f}s", flush=True)
        if len(rows) >= args.max_windows:
            break
    return pd.DataFrame(rows)


def write_reports(windows: pd.DataFrame, out_dir: Path, min_examples: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_values = windows.loc[windows.split == "train", "insulin_units"]
    if len(train_values) < 10:
        train_values = windows.insulin_units
    quantile_edges = np.unique(np.quantile(train_values.to_numpy(float), [0.25, 0.50, 0.75])).tolist()
    while len(quantile_edges) < 3:
        quantile_edges.append(quantile_edges[-1] + 1e-6 if quantile_edges else 1e-6)
    quantile_labels = [
        f"insulin_le{quantile_edges[0]:.3f}",
        f"insulin_{quantile_edges[0]:.3f}_{quantile_edges[1]:.3f}",
        f"insulin_{quantile_edges[1]:.3f}_{quantile_edges[2]:.3f}",
        f"insulin_gt{quantile_edges[2]:.3f}",
    ]
    windows = windows.copy()
    windows["insulin4_fixed"] = bin_by_edges(windows.insulin_units, FIXED_INSULIN_EDGES)
    windows["insulin4_quantile"] = bin_by_edges(windows.insulin_units, quantile_edges)
    windows.to_parquet(out_dir / "insulin_action_windows.parquet", index=False)

    action_specs = {
        "bolus4": D.ACTION_LABELS_BOLUS4,
        "insulin4_fixed": FIXED_INSULIN_LABELS,
        "insulin4_quantile": quantile_labels,
    }
    summary_rows = []
    for name, labels in action_specs.items():
        dist = action_distribution(windows, name, labels)
        dist.to_csv(out_dir / f"{name}_distribution.csv", index=False)
        source_dist = (
            windows.groupby(["source_file", name], dropna=False)
            .size()
            .rename("count")
            .reset_index()
        )
        source_dist["frac"] = source_dist["count"] / source_dist.groupby("source_file")["count"].transform("sum").clip(lower=1)
        source_dist["label"] = source_dist[name].map(lambda x: labels[int(x)] if 0 <= int(x) < len(labels) else "unknown")
        source_dist.to_csv(out_dir / f"{name}_source_distribution.csv", index=False)
        metrics = split_metrics(dist)
        metrics.update(support_metrics(windows, name, min_examples))
        metrics.update({"action_definition": name, "n_windows": int(len(windows)), "labels": json.dumps(labels)})
        summary_rows.append(metrics)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "insulin_action_audit_summary.csv", index=False)
    print("\nINSULIN ACTION AUDIT SUMMARY")
    show_cols = [
        "action_definition",
        "n_windows",
        "train_majority_frac",
        "val_majority_frac",
        "test_majority_frac",
        "train_test_tvd",
        "support_frac",
        "multi_action_cells",
    ]
    print(summary[[c for c in show_cols if c in summary]].round(4).to_string(index=False))
    print("\nTRAIN-BASED QUANTILE INSULIN EDGES")
    print(", ".join(f"{x:.4f}" for x in quantile_edges))
    print(f"\nSaved audit files to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit MetaboNet insulin action definitions for offline RL.")
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "insulin_action_audit")
    parser.add_argument("--batch-size", type=int, default=500_000)
    parser.add_argument("--max-windows", type=int, default=1_000_000)
    parser.add_argument("--max-subject-windows", type=int, default=600)
    parser.add_argument("--history-steps", type=int, default=24)
    parser.add_argument("--action-steps", type=int, default=6)
    parser.add_argument("--stride-steps", type=int, default=6)
    parser.add_argument("--min-support-examples", type=int, default=50)
    args = parser.parse_args()

    windows = collect_windows(args)
    if windows.empty:
        raise RuntimeError("No valid action windows found.")
    write_reports(windows, args.output_dir, args.min_support_examples)


if __name__ == "__main__":
    main()
