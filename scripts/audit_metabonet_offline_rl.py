#!/usr/bin/env python3
"""Audit MetaboNet for a defensible offline-RL/CMDP project.

This script does not train a policy. It determines whether the data support one.
It produces:

* source and trajectory coverage statistics;
* factual, patient-day clinical and operational objectives;
* empirical Pareto fronts (observed care, not counterfactual policy values);
* state-action support diagnostics for offline RL and OPE;
* deployed-controller/treatment-group candidates for real held-out OPE checks;
* explicit pass/warn/fail gates, including a counterfactual-ground-truth gate.

The crucial distinction is preserved throughout: observed glucose outcomes are
ground truth for the actions actually taken. They are not ground truth for a new
policy's unobserved actions. A real OPE benchmark is possible only when a named
policy/controller was deployed on held-out patients, or decisions were randomized.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


REQUIRED_COLUMNS = {
    "CGM",
    "basal",
    "bolus",
    "date",
    "id",
    "is_test",
    "source_file",
}

OPTIONAL_COLUMNS = {
    "carbs",
    "insulin",
    "insulin_delivery_algorithm",
    "insulin_delivery_modality",
    "randomization_date",
    "subject_split_across_traintest",
    "treatment_group",
}

DAY_KEYS = ["source_file", "id", "day"]
SUBJECT_KEYS = ["source_file", "id"]
POLICY_LABELS = [
    "insulin_delivery_algorithm",
    "treatment_group",
    "insulin_delivery_modality",
]


@dataclass
class Gate:
    status: str
    finding: str
    implication: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path, help="Path to metabonet_public.parquet")
    parser.add_argument(
        "--download-url",
        default=None,
        help=(
            "Public URL for metabonet_public.parquet. If PARQUET does not exist, "
            "the script downloads it here. Can also be set via METABONET_PUBLIC_URL."
        ),
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the parquet even if the target file already exists.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/metabonet_audit"),
        help="Directory for CSV, JSON, and plot outputs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500_000,
        help="Rows per streaming batch (reduce if memory is limited)",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional smoke-test limit. Do not use for the final audit.",
    )
    parser.add_argument(
        "--min-cell-count",
        type=int,
        default=50,
        help="Minimum observations for a supported state-action cell",
    )
    parser.add_argument(
        "--min-policy-subjects",
        type=int,
        default=20,
        help="Minimum subjects for a labeled deployed-policy benchmark",
    )
    parser.add_argument(
        "--batch-cache-dir",
        type=Path,
        default=None,
        help="Directory for per-batch aggregate checkpoints. Defaults to OUTPUT_DIR/_batch_cache.",
    )
    parser.add_argument(
        "--reuse-complete-cache",
        action="store_true",
        help="Skip the raw parquet scan if a complete batch cache exists.",
    )
    return parser.parse_args()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".partial")
    print(f"Downloading dataset to: {destination}", flush=True)
    print(f"Source URL: {url}", flush=True)

    def report_progress(block_count: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(block_count * block_size, total_size)
        if block_count == 0 or downloaded == total_size or block_count % 200 == 0:
            percent = downloaded / total_size * 100
            print(
                f"downloaded={downloaded / (1024 ** 3):.2f}GB/{total_size / (1024 ** 3):.2f}GB ({percent:.1f}%)",
                flush=True,
            )

    try:
        urllib.request.urlretrieve(url, temp_path, reporthook=report_progress)
        temp_path.replace(destination)
    finally:
        if temp_path.exists() and not destination.exists():
            temp_path.unlink()


def safe_mode(series: pd.Series) -> object:
    values = series.dropna()
    if values.empty:
        return None
    mode = values.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else values.iloc[0]


def finite_quantiles(series: pd.Series) -> dict[str, float | None]:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {str(q): None for q in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)}
    return {
        str(q): float(value)
        for q, value in clean.quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]).items()
    }


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return math.nan
    return float(np.average(values[valid], weights=weights[valid]))


def pareto_mask(values: np.ndarray, maximize: Sequence[bool]) -> np.ndarray:
    """Return the nondominated mask; NaN rows are excluded."""
    if values.ndim != 2 or values.shape[1] != len(maximize):
        raise ValueError("Pareto value matrix and objective directions do not match")
    finite = np.isfinite(values).all(axis=1)
    transformed = values.copy()
    for column, is_max in enumerate(maximize):
        if not is_max:
            transformed[:, column] *= -1
    result = np.zeros(len(values), dtype=bool)
    valid_indices = np.flatnonzero(finite)
    for index in valid_indices:
        candidate = transformed[index]
        dominates = np.all(transformed[valid_indices] >= candidate, axis=1) & np.any(
            transformed[valid_indices] > candidate, axis=1
        )
        if not dominates.any():
            result[index] = True
    return result


def glucose_band(glucose: pd.Series) -> pd.Categorical:
    return pd.cut(
        glucose,
        bins=[-np.inf, 54, 70, 180, 250, np.inf],
        labels=["lt54", "54_69", "70_180", "181_250", "gt250"],
        right=False,
    )


def trend_band(delta_30m: pd.Series) -> pd.Categorical:
    return pd.cut(
        delta_30m,
        bins=[-np.inf, -30, -10, 10, 30, np.inf],
        labels=["fall_fast", "fall", "flat", "rise", "rise_fast"],
        right=False,
    )


def action_label(basal_delta: pd.Series, bolus: pd.Series) -> pd.Series:
    basal_class = pd.cut(
        basal_delta.fillna(0),
        bins=[-np.inf, -0.01, 0.01, np.inf],
        labels=["basal_down", "basal_same", "basal_up"],
        right=False,
    ).astype("string")
    bolus_class = pd.cut(
        bolus.fillna(0),
        bins=[-np.inf, 1e-12, 2, 5, np.inf],
        labels=["no_bolus", "bolus_le2", "bolus_2_5", "bolus_gt5"],
        right=False,
    ).astype("string")
    return basal_class + "|" + bolus_class


def merge_day_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    raw = pd.concat(parts, ignore_index=True)
    sums = [
        "rows",
        "cgm_n",
        "tir_n",
        "tbr70_n",
        "tbr54_n",
        "tar180_n",
        "tar250_n",
        "glucose_sum",
        "glucose_sq_sum",
        "basal_units",
        "bolus_units",
        "bolus_events",
        "basal_changes",
        "carb_events",
        "valid_transitions",
    ]
    aggregations: dict[str, tuple[str, object]] = {
        name: (name, "sum") for name in sums
    }
    aggregations.update(
        {
            "is_test": ("is_test", "max"),
            "randomized": ("randomized", "max"),
            "first_time": ("first_time", "min"),
            "last_time": ("last_time", "max"),
        }
    )
    for label in POLICY_LABELS:
        aggregations[label] = (label, safe_mode)
    days = raw.groupby(DAY_KEYS, dropna=False).agg(**aggregations).reset_index()
    count = days.cgm_n.replace(0, np.nan)
    days["tir"] = days.tir_n / count
    days["tbr70"] = days.tbr70_n / count
    days["tbr54"] = days.tbr54_n / count
    days["tar180"] = days.tar180_n / count
    days["tar250"] = days.tar250_n / count
    days["mean_glucose"] = days.glucose_sum / count
    variance = days.glucose_sq_sum / count - days.mean_glucose.pow(2)
    days["glucose_sd"] = np.sqrt(variance.clip(lower=0))
    days["glucose_cv"] = days.glucose_sd / days.mean_glucose
    days["total_insulin_units"] = days.basal_units + days.bolus_units
    days["interventions"] = days.bolus_events + days.basal_changes
    days["observed_hours"] = (days.last_time - days.first_time).dt.total_seconds() / 3600
    return days


def merge_subject_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    raw = pd.concat(parts, ignore_index=True)
    sums = ["rows", "cgm_n", "insulin_n", "basal_n", "bolus_n", "carbs_n"]
    aggregations: dict[str, tuple[str, object]] = {
        name: (name, "sum") for name in sums
    }
    aggregations.update(
        {
            "is_test": ("is_test", "max"),
            "split_across_train_test": ("split_across_train_test", "max"),
            "randomized": ("randomized", "max"),
            "start": ("start", "min"),
            "end": ("end", "max"),
        }
    )
    for label in POLICY_LABELS:
        aggregations[label] = (label, safe_mode)
    subjects = raw.groupby(SUBJECT_KEYS, dropna=False).agg(**aggregations).reset_index()
    subjects["span_days"] = (subjects.end - subjects.start).dt.total_seconds() / 86400
    subjects["has_cgm_and_insulin"] = (subjects.cgm_n > 0) & (subjects.insulin_n > 0)
    subjects["has_cgm_and_bolus"] = (subjects.cgm_n > 0) & (subjects.bolus_n > 0)
    return subjects


def aggregate_policy_groups(
    days: pd.DataFrame, min_subjects: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    candidates: list[pd.DataFrame] = []
    metrics = [
        "tir",
        "tbr70",
        "tbr54",
        "tar180",
        "tar250",
        "mean_glucose",
        "glucose_cv",
        "total_insulin_units",
        "interventions",
    ]
    for label in POLICY_LABELS:
        labeled = days[days[label].notna()].copy()
        if labeled.empty:
            continue
        grouped_rows = []
        for (source, value), group in labeled.groupby(["source_file", label], dropna=False):
            row: dict[str, object] = {
                "label_type": label,
                "source_file": source,
                "policy_label": value,
                "subjects": group.id.nunique(),
                "subject_days": len(group),
                "test_subjects": group.loc[group.is_test, "id"].nunique(),
                "randomized_subjects": group.loc[group.randomized, "id"].nunique(),
                "cgm_observations": int(group.cgm_n.sum()),
            }
            for metric in metrics:
                row[metric] = weighted_mean(group[metric], group.cgm_n)
            grouped_rows.append(row)
        frame = pd.DataFrame(grouped_rows)
        frames.append(frame)
        eligible = frame[
            (frame.subjects >= min_subjects)
            & (frame.test_subjects >= max(5, min_subjects // 4))
        ].copy()
        candidates.append(eligible)
    all_groups = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    all_candidates = (
        pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()
    )
    return all_groups, all_candidates


def aggregate_subject_objectives(days: pd.DataFrame) -> pd.DataFrame:
    """Collapse patient-days before Pareto analysis.

    Exact Pareto computation is quadratic in row count. Patient-day output is useful
    for factual objectives, but the audit only needs a representative observed
    frontier to prove tradeoffs exist. Subject-level aggregation keeps the result
    clinically interpretable and small enough for exact nondominated sorting.
    """
    metrics = [
        "tir",
        "tbr70",
        "tbr54",
        "tar180",
        "tar250",
        "mean_glucose",
        "glucose_cv",
        "total_insulin_units",
        "interventions",
    ]
    rows = []
    for keys, group in days.groupby(SUBJECT_KEYS, dropna=False):
        row: dict[str, object] = {
            "source_file": keys[0],
            "id": keys[1],
            "subject_days": len(group),
            "test_days": int(group.is_test.sum()),
            "randomized_days": int(group.randomized.sum()),
            "cgm_observations": int(group.cgm_n.sum()),
        }
        for label in POLICY_LABELS:
            row[label] = safe_mode(group[label])
        for metric in metrics:
            row[metric] = weighted_mean(group[metric], group.cgm_n)
        rows.append(row)
    return pd.DataFrame(rows)


def cache_batch(
    cache_dir: Path,
    batch_index: int,
    day_agg: pd.DataFrame,
    subject_agg: pd.DataFrame,
    support_agg: pd.DataFrame | None,
    summary: dict[str, object],
) -> None:
    batch_dir = cache_dir / f"batch_{batch_index:05d}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    day_agg.to_parquet(batch_dir / "days.parquet", index=False)
    subject_agg.to_parquet(batch_dir / "subjects.parquet", index=False)
    if support_agg is not None and not support_agg.empty:
        support_agg.to_parquet(batch_dir / "support.parquet", index=False)
    with (batch_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)


def load_complete_cache(cache_dir: Path) -> tuple[
    list[pd.DataFrame],
    list[pd.DataFrame],
    list[pd.DataFrame],
    Counter[str],
    Counter[str],
    Counter[str],
    Counter[str],
    defaultdict[str, float],
    int,
] | None:
    marker = cache_dir / "COMPLETE.json"
    if not marker.is_file():
        return None
    day_parts: list[pd.DataFrame] = []
    subject_parts: list[pd.DataFrame] = []
    support_parts: list[pd.DataFrame] = []
    source_counts: Counter[str] = Counter()
    algorithm_counts: Counter[str] = Counter()
    modality_counts: Counter[str] = Counter()
    treatment_counts: Counter[str] = Counter()
    transition_summary: defaultdict[str, float] = defaultdict(float)
    batch_count = 0
    for batch_dir in sorted(cache_dir.glob("batch_*")):
        if not (batch_dir / "days.parquet").is_file() or not (batch_dir / "subjects.parquet").is_file():
            return None
        day_parts.append(pd.read_parquet(batch_dir / "days.parquet"))
        subject_parts.append(pd.read_parquet(batch_dir / "subjects.parquet"))
        support_file = batch_dir / "support.parquet"
        if support_file.is_file():
            support_parts.append(pd.read_parquet(support_file))
        with (batch_dir / "summary.json").open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        source_counts.update(summary.get("source_counts", {}))
        algorithm_counts.update(summary.get("algorithm_counts", {}))
        modality_counts.update(summary.get("modality_counts", {}))
        treatment_counts.update(summary.get("treatment_counts", {}))
        for key, value in summary.get("transition_summary", {}).items():
            transition_summary[key] += value
        batch_count += 1
    return (
        day_parts,
        subject_parts,
        support_parts,
        source_counts,
        algorithm_counts,
        modality_counts,
        treatment_counts,
        transition_summary,
        batch_count,
    )


def add_pareto_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if result.empty:
        result["pareto_clinical"] = pd.Series(dtype=bool)
        result["pareto_qpsi"] = pd.Series(dtype=bool)
        return result
    result["pareto_clinical"] = pareto_mask(
        result[["tir", "tbr54", "tar250"]].to_numpy(float),
        maximize=[True, False, False],
    )
    result["pareto_qpsi"] = pareto_mask(
        result[["tir", "tbr54", "tar250", "interventions"]].to_numpy(float),
        maximize=[True, False, False, False],
    )
    return result


def plot_frontier(frame: pd.DataFrame, output: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; skipping Pareto plot", file=sys.stderr)
        return
    valid = frame.dropna(subset=["tir", "tbr54", "tar250"]).copy()
    if valid.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    dominated = valid[~valid.pareto_clinical]
    frontier = valid[valid.pareto_clinical]
    ax.scatter(
        dominated.tbr54 * 100,
        dominated.tir * 100,
        c=dominated.tar250 * 100,
        cmap="viridis_r",
        alpha=0.25,
        s=16,
        linewidths=0,
        label="Dominated observed points",
    )
    scatter = ax.scatter(
        frontier.tbr54 * 100,
        frontier.tir * 100,
        c=frontier.tar250 * 100,
        cmap="viridis_r",
        edgecolor="black",
        linewidth=0.6,
        s=42,
        label="Empirical clinical Pareto front",
    )
    ax.set_xlabel("Time below 54 mg/dL (%) - lower is better")
    ax.set_ylabel("Time in 70-180 mg/dL (%) - higher is better")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Time above 250 mg/dL (%) - lower is better")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def status_gate(condition: bool, warning: bool, finding: str, implication: str) -> Gate:
    status = "PASS" if condition else ("WARN" if warning else "FAIL")
    return Gate(status=status, finding=finding, implication=implication)


def main() -> int:
    args = parse_args()
    parquet = args.parquet.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    cache_dir = (
        args.batch_cache_dir.expanduser().resolve()
        if args.batch_cache_dir is not None
        else output / "_batch_cache"
    )
    download_url = args.download_url or os.environ.get("METABONET_PUBLIC_URL")
    if args.force_download or not parquet.is_file():
        if not download_url:
            raise FileNotFoundError(
                f"{parquet} does not exist. Provide --download-url or set METABONET_PUBLIC_URL."
            )
        download_file(download_url, parquet)
    if not parquet.is_file():
        raise FileNotFoundError(parquet)
    output.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = ds.dataset(parquet, format="parquet")
    schema_names = set(dataset.schema.names)
    missing = REQUIRED_COLUMNS - schema_names
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    selected = sorted(REQUIRED_COLUMNS | (OPTIONAL_COLUMNS & schema_names))

    day_parts: list[pd.DataFrame] = []
    subject_parts: list[pd.DataFrame] = []
    support_parts: list[pd.DataFrame] = []
    source_counts: Counter[str] = Counter()
    algorithm_counts: Counter[str] = Counter()
    modality_counts: Counter[str] = Counter()
    treatment_counts: Counter[str] = Counter()
    transition_summary: defaultdict[str, float] = defaultdict(float)

    scanner = dataset.scanner(columns=selected, batch_size=args.batch_size, use_threads=True)
    start_time = time.time()
    previous_tail: dict[tuple[str, str], pd.DataFrame] = {}

    for batch_index, record_batch in enumerate(scanner.to_batches(), start=1):
        if args.max_batches is not None and batch_index > args.max_batches:
            break
        frame = record_batch.to_pandas()
        frame["date"] = pd.to_datetime(frame.date, errors="coerce")
        frame = frame.dropna(subset=["source_file", "id", "date"]).copy()
        frame["source_file"] = frame.source_file.astype(str)
        frame["id"] = frame.id.astype(str)
        frame.sort_values(["source_file", "id", "date"], inplace=True)
        frame["day"] = frame.date.dt.floor("D")

        for column in ["CGM", "basal", "bolus", "carbs", "insulin"]:
            if column not in frame:
                frame[column] = np.nan
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for column in POLICY_LABELS:
            if column not in frame:
                frame[column] = None
        if "randomization_date" not in frame:
            frame["randomization_date"] = pd.NaT
        if "subject_split_across_traintest" not in frame:
            frame["subject_split_across_traintest"] = False

        batch_source_counts = Counter(frame.source_file.value_counts().to_dict())
        batch_algorithm_counts = Counter(
            frame.insulin_delivery_algorithm.dropna().astype(str).value_counts().to_dict()
        )
        batch_modality_counts = Counter(
            frame.insulin_delivery_modality.dropna().astype(str).value_counts().to_dict()
        )
        batch_treatment_counts = Counter(
            frame.treatment_group.dropna().astype(str).value_counts().to_dict()
        )
        source_counts.update(batch_source_counts)
        algorithm_counts.update(batch_algorithm_counts)
        modality_counts.update(batch_modality_counts)
        treatment_counts.update(batch_treatment_counts)

        groups = frame.groupby(SUBJECT_KEYS, sort=False, dropna=False)
        frame["dt_hours"] = groups.date.diff().dt.total_seconds().div(3600)
        frame["basal_delta"] = groups.basal.diff()
        frame["cgm_delta_30m"] = groups.CGM.diff(6)
        frame["next_cgm"] = groups.CGM.shift(-1)
        frame["next_date"] = groups.date.shift(-1)
        frame["next_dt_minutes"] = (frame.next_date - frame.date).dt.total_seconds() / 60
        valid_transition = (
            frame.CGM.notna()
            & frame.next_cgm.notna()
            & frame.next_dt_minutes.between(3, 10, inclusive="both")
        )
        frame["valid_transition"] = valid_transition.astype("int32")

        plausible_dt = frame.dt_hours.where(frame.dt_hours.between(0, 0.5), 5 / 60)
        frame["basal_units"] = (frame.basal.clip(lower=0) * plausible_dt).fillna(0)
        frame["bolus_units"] = frame.bolus.clip(lower=0).fillna(0)
        frame["bolus_event"] = frame.bolus.fillna(0).gt(0).astype("int32")
        frame["basal_change"] = frame.basal_delta.abs().gt(0.01).astype("int32")
        frame["carb_event"] = frame.carbs.fillna(0).gt(0).astype("int32")

        cgm = frame.CGM
        frame["cgm_n"] = cgm.notna().astype("int32")
        frame["tir"] = cgm.between(70, 180, inclusive="both").astype("int32")
        frame["tbr70"] = cgm.lt(70).astype("int32")
        frame["tbr54"] = cgm.lt(54).astype("int32")
        frame["tar180"] = cgm.gt(180).astype("int32")
        frame["tar250"] = cgm.gt(250).astype("int32")
        frame["glucose_value"] = cgm.fillna(0)
        frame["glucose_sq"] = cgm.pow(2).fillna(0)

        day_agg = frame.groupby(DAY_KEYS, dropna=False).agg(
            rows=("id", "size"),
            cgm_n=("cgm_n", "sum"),
            tir_n=("tir", "sum"),
            tbr70_n=("tbr70", "sum"),
            tbr54_n=("tbr54", "sum"),
            tar180_n=("tar180", "sum"),
            tar250_n=("tar250", "sum"),
            glucose_sum=("glucose_value", "sum"),
            glucose_sq_sum=("glucose_sq", "sum"),
            basal_units=("basal_units", "sum"),
            bolus_units=("bolus_units", "sum"),
            bolus_events=("bolus_event", "sum"),
            basal_changes=("basal_change", "sum"),
            carb_events=("carb_event", "sum"),
            valid_transitions=("valid_transition", "sum"),
            is_test=("is_test", "max"),
            randomized=("randomization_date", lambda x: x.notna().any()),
            first_time=("date", "min"),
            last_time=("date", "max"),
            insulin_delivery_algorithm=("insulin_delivery_algorithm", safe_mode),
            insulin_delivery_modality=("insulin_delivery_modality", safe_mode),
            treatment_group=("treatment_group", safe_mode),
        ).reset_index()
        day_parts.append(day_agg)

        subject_agg = frame.groupby(SUBJECT_KEYS, dropna=False).agg(
            rows=("id", "size"),
            cgm_n=("CGM", "count"),
            insulin_n=("insulin", "count"),
            basal_n=("basal", "count"),
            bolus_n=("bolus_event", "sum"),
            carbs_n=("carb_event", "sum"),
            is_test=("is_test", "max"),
            split_across_train_test=("subject_split_across_traintest", "max"),
            randomized=("randomization_date", lambda x: x.notna().any()),
            start=("date", "min"),
            end=("date", "max"),
            insulin_delivery_algorithm=("insulin_delivery_algorithm", safe_mode),
            insulin_delivery_modality=("insulin_delivery_modality", safe_mode),
            treatment_group=("treatment_group", safe_mode),
        ).reset_index()
        subject_parts.append(subject_agg)

        decision = frame[
            frame.CGM.notna()
            & frame.cgm_delta_30m.notna()
            & (frame.basal.notna() | frame.bolus.notna())
        ].copy()
        support_agg = None
        if not decision.empty:
            decision["glucose_band"] = glucose_band(decision.CGM).astype("string")
            decision["trend_band"] = trend_band(decision.cgm_delta_30m).astype("string")
            decision["carbs_now"] = np.where(decision.carbs.fillna(0).gt(0), "carbs", "no_carbs")
            decision["action"] = action_label(decision.basal_delta, decision.bolus)
            support_agg = (
                decision.groupby(
                    ["source_file", "glucose_band", "trend_band", "carbs_now", "action"],
                    dropna=False,
                )
                .size()
                .rename("n")
                .reset_index()
            )
            support_parts.append(support_agg)

        batch_transition_summary = {
            "valid_transitions": int(valid_transition.sum()),
            "rows_with_cgm": int(frame.CGM.notna().sum()),
            "rows_with_insulin": int(frame.insulin.notna().sum()),
            "positive_bolus_events": int(frame.bolus_event.sum()),
            "basal_changes": int(frame.basal_change.sum()),
            "carb_events": int(frame.carb_event.sum()),
        }
        for key, value in batch_transition_summary.items():
            transition_summary[key] += value

        cache_batch(
            cache_dir,
            batch_index,
            day_agg,
            subject_agg,
            support_agg,
            {
                "batch_index": batch_index,
                "rows": int(len(frame)),
                "source_counts": dict(batch_source_counts),
                "algorithm_counts": dict(batch_algorithm_counts),
                "modality_counts": dict(batch_modality_counts),
                "treatment_counts": dict(batch_treatment_counts),
                "transition_summary": batch_transition_summary,
            },
        )

        if batch_index == 1 or batch_index % 20 == 0:
            elapsed = time.time() - start_time
            print(
                f"batch={batch_index} rows_seen={sum(source_counts.values()):,} elapsed={elapsed:.1f}s cache={cache_dir}",
                flush=True,
            )

    with (cache_dir / "COMPLETE.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "input": str(parquet),
                "batch_size": args.batch_size,
                "max_batches": args.max_batches,
                "batches": batch_index,
                "rows_scanned": int(sum(source_counts.values())),
                "complete_raw_scan": args.max_batches is None,
            },
            handle,
            indent=2,
        )

    if not day_parts:
        raise RuntimeError("No usable rows were read")

    days = merge_day_parts(day_parts)
    subjects = merge_subject_parts(subject_parts)
    support = pd.concat(support_parts, ignore_index=True)
    support = support.groupby(
        ["source_file", "glucose_band", "trend_band", "carbs_now", "action"],
        dropna=False,
    ).n.sum().reset_index()

    state_keys = ["source_file", "glucose_band", "trend_band", "carbs_now"]
    support["supported_action"] = support.n >= args.min_cell_count
    state_support = support.groupby(state_keys, dropna=False).agg(
        observations=("n", "sum"),
        distinct_actions=("action", "nunique"),
        supported_actions=("supported_action", "sum"),
    ).reset_index()
    state_support["multi_action_supported"] = state_support.supported_actions >= 2
    supported_observations = int(
        state_support.loc[state_support.multi_action_supported, "observations"].sum()
    )
    total_support_observations = int(state_support.observations.sum())
    support_fraction = (
        supported_observations / total_support_observations
        if total_support_observations
        else 0.0
    )

    subject_objectives = add_pareto_columns(aggregate_subject_objectives(days))
    policy_groups, ope_candidates = aggregate_policy_groups(days, args.min_policy_subjects)
    if not policy_groups.empty:
        policy_groups = add_pareto_columns(policy_groups)

    source_summary = subjects.groupby("source_file", dropna=False).agg(
        subjects=("id", "nunique"),
        test_subjects=("is_test", "sum"),
        randomized_subjects=("randomized", "sum"),
        split_subjects=("split_across_train_test", "sum"),
        rows=("rows", "sum"),
        cgm_observations=("cgm_n", "sum"),
        insulin_observations=("insulin_n", "sum"),
        basal_observations=("basal_n", "sum"),
        bolus_events=("bolus_n", "sum"),
        carb_events=("carbs_n", "sum"),
        subjects_cgm_insulin=("has_cgm_and_insulin", "sum"),
        subjects_cgm_bolus=("has_cgm_and_bolus", "sum"),
        median_span_days=("span_days", "median"),
    ).reset_index()

    policy_label_subjects = {}
    for label in POLICY_LABELS:
        policy_label_subjects[label] = int(subjects.loc[subjects[label].notna(), "id"].nunique())

    enough_sequential = int(transition_summary["valid_transitions"]) >= 1_000_000
    enough_action_data = (
        int(transition_summary["positive_bolus_events"] + transition_summary["basal_changes"])
        >= 10_000
    )
    enough_support = support_fraction >= 0.5
    has_objectives = days[["tir", "tbr54", "tar250", "interventions"]].notna().all(axis=1).sum() >= 100
    has_empirical_front = int(subject_objectives.pareto_clinical.sum()) >= 2
    has_policy_benchmark = not ope_candidates.empty
    randomized_subjects = int(subjects.randomized.sum())
    split_leakage = int(subjects.split_across_train_test.sum())

    gates = {
        "sequential_mdp": status_gate(
            enough_sequential,
            int(transition_summary["valid_transitions"]) >= 100_000,
            f"{int(transition_summary['valid_transitions']):,} contiguous CGM transitions were found.",
            "Repeated insulin decisions affect later glucose, so this is a genuine sequential-control problem.",
        ),
        "action_data": status_gate(
            enough_action_data,
            int(transition_summary["positive_bolus_events"] + transition_summary["basal_changes"]) >= 1_000,
            f"Found {int(transition_summary['positive_bolus_events']):,} positive boluses and "
            f"{int(transition_summary['basal_changes']):,} observed basal changes.",
            "Policy learning requires variation in actions, not only dense glucose measurements.",
        ),
        "state_action_support": status_gate(
            enough_support,
            support_fraction >= 0.2,
            f"{support_fraction:.1%} of audited decision observations lie in state cells with at least two actions having "
            f">={args.min_cell_count} examples.",
            "Low support means restrict the policy class or action space; otherwise OPE will extrapolate.",
        ),
        "multiple_objectives": status_gate(
            has_objectives,
            False,
            "Real factual objectives include TIR, severe hypoglycemia, severe hyperglycemia, insulin exposure, and intervention frequency.",
            "Use a CMDP/Pareto formulation instead of hiding clinical tradeoffs in one reward weight.",
        ),
        "empirical_pareto_front": status_gate(
            has_empirical_front,
            False,
            f"{int(subject_objectives.pareto_clinical.sum()):,} observed subjects are clinically nondominated.",
            "This proves observed tradeoffs exist, but the subject-level frontier is not a learned-policy frontier and is confounded by case mix.",
        ),
        "real_policy_value_benchmark": status_gate(
            has_policy_benchmark,
            any(value > 0 for value in policy_label_subjects.values()),
            f"Found {len(ope_candidates)} labeled source/controller groups with enough train/test subjects for held-out factual evaluation.",
            "A held-out deployed controller gives a real on-policy value target for validating OPE estimators.",
        ),
        "counterfactual_ground_truth": Gate(
            status="FAIL",
            finding="No retrospective dataset records outcomes for both taken and untaken insulin actions in the same patient state.",
            implication="Do not call observed test-set TIR the learned policy's performance. Validate OPE against held-out deployed policies, and report that new-policy value remains estimated.",
        ),
        "known_logging_propensity": Gate(
            status="PASS" if randomized_subjects > 0 else "FAIL",
            finding=f"{randomized_subjects:,} subjects have a randomization date; per-decision action propensities are not present in the schema.",
            implication="WIS/DR require an estimated behavior policy unless source documentation supplies exact decision probabilities.",
        ),
        "train_test_integrity": status_gate(
            split_leakage == 0,
            split_leakage <= 5,
            f"{split_leakage:,} subjects are marked as split across train and test.",
            "All final evaluation must split by subject and source, never by row or overlapping window.",
        ),
    }

    recommendations = [
        "Define decisions at clinically meaningful insulin opportunities, not every five-minute CGM row.",
        "Start with discrete conservative actions: basal down/same/up and bolus bins conditioned on glucose trend, carbs, and insulin-on-board proxies.",
        "Train behavior cloning, fitted Q iteration, discrete CQL, and IQL; require each policy to remain inside audited support.",
        "Use cross-fitted FQE and doubly robust OPE; use WIS only where estimated propensities have acceptable effective sample size.",
        "Validate the OPE pipeline by recovering held-out factual values of named deployed controllers before evaluating a learned policy.",
        "Build a learned-policy Pareto set over TIR, TBR<54, TAR>250, and intervention frequency; enforce hypoglycemia as a CMDP constraint.",
        "Report bootstrap confidence regions by patient and source, not only point estimates.",
        "Treat simulator rollouts as stress tests, never as real ground truth.",
    ]

    days.to_csv(output / "patient_day_objectives.csv", index=False)
    subject_objectives.to_csv(output / "subject_objectives.csv", index=False)
    subjects.to_csv(output / "subject_coverage.csv", index=False)
    source_summary.to_csv(output / "source_summary.csv", index=False)
    support.to_csv(output / "state_action_counts.csv", index=False)
    state_support.to_csv(output / "state_action_support.csv", index=False)
    policy_groups.to_csv(output / "observed_policy_group_objectives.csv", index=False)
    ope_candidates.to_csv(output / "real_ope_benchmark_candidates.csv", index=False)
    plot_frontier(
        subject_objectives,
        output / "observed_subject_pareto.png",
        "MetaboNet observed subject-level tradeoffs (not counterfactual policy value)",
    )
    if not policy_groups.empty:
        plot_frontier(
            policy_groups,
            output / "observed_policy_group_pareto.png",
            "MetaboNet deployed-group outcomes (case-mix unadjusted)",
        )

    report = {
        "input": str(parquet),
        "rows_scanned": int(sum(source_counts.values())),
        "batches_scanned": batch_index,
        "complete_scan": args.max_batches is None,
        "schema": dataset.schema.names,
        "sources": source_counts,
        "algorithm_labels": algorithm_counts,
        "modality_labels": modality_counts,
        "treatment_labels": treatment_counts,
        "subjects": int(len(subjects)),
        "patient_days": int(len(days)),
        "subject_objectives": int(len(subject_objectives)),
        "transition_summary": dict(transition_summary),
        "support_fraction_multi_action": support_fraction,
        "patient_day_quantiles": {
            column: finite_quantiles(days[column])
            for column in [
                "tir",
                "tbr54",
                "tar250",
                "total_insulin_units",
                "interventions",
                "glucose_cv",
            ]
        },
        "policy_label_subjects": policy_label_subjects,
        "ope_benchmark_candidates": int(len(ope_candidates)),
        "subject_pareto_points": int(subject_objectives.pareto_clinical.sum()),
        "gates": {name: asdict(gate) for name, gate in gates.items()},
        "recommended_project_design": recommendations,
        "interpretation": {
            "factual_ground_truth": "Observed glucose outcomes under logged insulin actions and deployed controllers.",
            "valid_empirical_pareto": "Nondominated observed patient-days or deployed groups, with case-mix caveats.",
            "not_ground_truth": "Outcomes claimed for a newly learned policy without prospective deployment.",
            "best_available_validation": "OPE estimator recovery of held-out on-policy controller values plus patient-cluster bootstrap.",
        },
        "runtime_seconds": time.time() - start_time,
    }
    with (output / "audit_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)

    print("\nOFFLINE RL / CMDP AUDIT")
    for name, gate in gates.items():
        print(f"[{gate.status}] {name}: {gate.finding}")
        print(f"       {gate.implication}")
    print(f"\nOutputs written to: {output}")
    print("Read audit_report.json first, then real_ope_benchmark_candidates.csv.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
