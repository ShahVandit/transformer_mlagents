from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


NUMERIC_COLUMNS = ["CGM", "basal", "bolus", "carbs", "insulin", "age", "height", "weight"]
LABEL_COLUMNS = ["insulin_delivery_algorithm", "insulin_delivery_modality", "treatment_group"]
REQUIRED_COLUMNS = ["CGM", "basal", "bolus", "date", "id", "is_test", "source_file"]
OPTIONAL_COLUMNS = NUMERIC_COLUMNS + LABEL_COLUMNS + [
    "carbs",
    "insulin",
    "gender",
    "subject_split_across_traintest",
]

SEQ_FEATURES = [
    "cgm_z",
    "basal_scaled",
    "bolus_scaled",
    "carbs_scaled",
    "insulin_scaled",
    "cgm_missing",
    "bolus_event",
    "carb_event",
    "hour_sin",
    "hour_cos",
]
STATIC_FEATURES = ["age_scaled", "height_scaled", "weight_scaled", "gender_male"]
ACTION_LABELS = [
    f"{basal}|{bolus}"
    for basal in ("basal_down", "basal_same", "basal_up")
    for bolus in ("no_bolus", "bolus_le2", "bolus_2_5", "bolus_gt5")
]
N_ACTIONS = len(ACTION_LABELS)


def download_if_needed(parquet: Path, url: str | None) -> None:
    if parquet.exists():
        return
    url = url or os.environ.get("METABONET_PUBLIC_URL")
    if not url:
        raise FileNotFoundError(f"{parquet} missing. Provide --download-url or METABONET_PUBLIC_URL.")
    parquet.parent.mkdir(parents=True, exist_ok=True)
    tmp = parquet.with_suffix(parquet.suffix + ".partial")
    print(f"[download] {url} -> {parquet}", flush=True)
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(parquet)


def stable_split(subject_key: str, is_test: bool) -> str:
    if is_test:
        return "test"
    digest = hashlib.md5(subject_key.encode("utf-8")).hexdigest()
    return "val" if int(digest[:8], 16) % 10 == 0 else "train"


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame.date, errors="coerce")
    frame = frame.dropna(subset=["source_file", "id", "date"]).copy()
    frame["source_file"] = frame.source_file.astype(str)
    frame["id"] = frame.id.astype(str)
    frame.sort_values(["source_file", "id", "date"], inplace=True)
    for col in NUMERIC_COLUMNS:
        if col not in frame:
            frame[col] = np.nan
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in LABEL_COLUMNS:
        if col not in frame:
            frame[col] = None
    if "gender" not in frame:
        frame["gender"] = None
    if "subject_split_across_traintest" not in frame:
        frame["subject_split_across_traintest"] = False
    return frame


def _sequence_features(frame: pd.DataFrame) -> np.ndarray:
    hour = frame.date.dt.hour.to_numpy(float) + frame.date.dt.minute.to_numpy(float) / 60.0
    features = np.column_stack(
        [
            (frame.CGM.to_numpy(float) - 150.0) / 60.0,
            frame.basal.fillna(0).clip(lower=0).to_numpy(float) / 3.0,
            frame.bolus.fillna(0).clip(lower=0).to_numpy(float) / 10.0,
            frame.carbs.fillna(0).clip(lower=0).to_numpy(float) / 100.0,
            frame.insulin.fillna(0).clip(lower=0).to_numpy(float) / 10.0,
            frame.CGM.isna().to_numpy(float),
            frame.bolus.fillna(0).gt(0).to_numpy(float),
            frame.carbs.fillna(0).gt(0).to_numpy(float),
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
        ]
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _static_features(row: pd.Series) -> np.ndarray:
    gender = str(row.get("gender", "")).lower()
    return np.array(
        [
            (float(row.get("age", np.nan)) if pd.notna(row.get("age", np.nan)) else 45.0) / 100.0,
            (float(row.get("height", np.nan)) if pd.notna(row.get("height", np.nan)) else 170.0) / 220.0,
            (float(row.get("weight", np.nan)) if pd.notna(row.get("weight", np.nan)) else 75.0) / 180.0,
            float(gender.startswith("m")),
        ],
        dtype=np.float32,
    )


def _encode_window(features: np.ndarray, t: int, history_steps: int) -> np.ndarray:
    out = np.zeros((history_steps, features.shape[1] + 1), dtype=np.float32)
    out[:, -1] = 1.0
    win = features[max(0, t - history_steps + 1): t + 1]
    out[-len(win):, :-1] = win
    out[-len(win):, -1] = 0.0
    return out


def _basal_bin(delta: float) -> int:
    if delta < -0.01:
        return 0
    if delta > 0.01:
        return 2
    return 1


def _bolus_bin(total: float) -> int:
    if total <= 1e-9:
        return 0
    if total <= 2:
        return 1
    if total <= 5:
        return 2
    return 3


def _build_group(
    group: pd.DataFrame,
    history_steps: int,
    horizon_steps: int,
    stride_steps: int,
    new_row_mask: np.ndarray,
) -> tuple[list, list]:
    features = _sequence_features(group)
    cgm = group.CGM.to_numpy(float)
    basal = group.basal.ffill().fillna(0).to_numpy(float)
    bolus = group.bolus.fillna(0).clip(lower=0).to_numpy(float)
    dates = group.date.to_numpy()
    rows, meta = [], []
    stat = _static_features(group.iloc[0])
    subject = f"{group.source_file.iloc[0]}::{group.id.iloc[0]}"
    split = stable_split(subject, bool(group.is_test.astype(bool).any()))

    for t in range(history_steps - 1, len(group) - horizon_steps, stride_steps):
        if not new_row_mask[t]:
            continue
        nt = t + horizon_steps
        minutes = (dates[nt] - dates[t]) / np.timedelta64(1, "m")
        if not 20 <= float(minutes) <= 45:
            continue
        future_cgm = cgm[t + 1: nt + 1]
        if not np.isfinite(future_cgm).any() or not np.isfinite(cgm[t]):
            continue
        interval_bolus = float(np.nansum(bolus[t + 1: nt + 1]))
        interval_basal_delta = float(np.nanmean(basal[t + 1: nt + 1]) - basal[t])
        action = _basal_bin(interval_basal_delta) * 4 + _bolus_bin(interval_bolus)
        basal_changes = float(np.sum(np.abs(np.diff(basal[t: nt + 1])) > 0.01))
        bolus_events = float(np.sum(bolus[t + 1: nt + 1] > 0))
        burden = bolus_events + basal_changes
        valid = future_cgm[np.isfinite(future_cgm)]
        tir = float(np.mean((valid >= 70) & (valid <= 180)))
        tbr54 = float(np.mean(valid < 54))
        tar250 = float(np.mean(valid > 250))
        rows.append(
            (
                _encode_window(features, t, history_steps),
                stat,
                _encode_window(features, nt, history_steps),
                stat,
                action,
                np.array([-tbr54, -tar250, -burden / max(horizon_steps, 1)], dtype=np.float32),
                np.array([tir, tbr54, tar250, burden], dtype=np.float32),
                0.0,
            )
        )
        meta.append(
            {
                "source_file": group.source_file.iloc[0],
                "id": group.id.iloc[0],
                "date": group.date.iloc[t],
                "split": split,
                "action": action,
                "insulin_delivery_algorithm": group.insulin_delivery_algorithm.iloc[0],
                "insulin_delivery_modality": group.insulin_delivery_modality.iloc[0],
                "treatment_group": group.treatment_group.iloc[0],
            }
        )
    return rows, meta


def balanced_sample_splits(
    split_rows: dict[str, list],
    split_meta: dict[str, list[dict]],
    max_transitions: int | None,
    min_split_transitions: int,
    seed: int = 7,
) -> tuple[dict[str, list], dict[str, list[dict]]]:
    if max_transitions is None:
        return split_rows, split_meta
    rng = np.random.default_rng(seed)
    target = max(min_split_transitions, max(1, max_transitions // 3))
    out_rows: dict[str, list] = {}
    out_meta: dict[str, list[dict]] = {}
    for split in ["train", "val", "test"]:
        rows = split_rows[split]
        metas = split_meta[split]
        if len(rows) <= target:
            out_rows[split] = rows
            out_meta[split] = metas
            continue
        buckets: dict[tuple[str, int], list[int]] = defaultdict(list)
        for idx, meta in enumerate(metas):
            buckets[(str(meta["source_file"]), int(meta["action"]))].append(idx)
        bucket_keys = sorted(buckets)
        chosen: list[int] = []
        while len(chosen) < target and bucket_keys:
            next_keys = []
            for key in bucket_keys:
                bucket = buckets[key]
                if not bucket:
                    continue
                chosen.append(bucket.pop(int(rng.integers(0, len(bucket)))))
                if bucket:
                    next_keys.append(key)
                if len(chosen) >= target:
                    break
            bucket_keys = next_keys
        chosen = sorted(chosen)
        out_rows[split] = [rows[i] for i in chosen]
        out_meta[split] = [metas[i] for i in chosen]
    return out_rows, out_meta


def build_transition_dataset(
    parquet: Path,
    output_dir: Path,
    download_url: str | None = None,
    batch_size: int = 500_000,
    max_transitions: int | None = 300_000,
    min_split_transitions: int = 1_000,
    max_subject_transitions: int = 600,
    history_steps: int = 12,
    horizon_steps: int = 6,
    stride_steps: int = 6,
) -> dict[str, int]:
    download_if_needed(parquet, download_url)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = ds.dataset(parquet, format="parquet")
    missing = set(REQUIRED_COLUMNS) - set(dataset.schema.names)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    columns = sorted(set(REQUIRED_COLUMNS) | (set(OPTIONAL_COLUMNS) & set(dataset.schema.names)))
    scanner = dataset.scanner(columns=columns, batch_size=batch_size, use_threads=True)

    split_rows: dict[str, list] = {"train": [], "val": [], "test": []}
    split_meta: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    tails: dict[str, pd.DataFrame] = {}
    subject_counts: defaultdict[str, int] = defaultdict(int)

    for batch_idx, record_batch in enumerate(scanner.to_batches(), start=1):
        frame = _prepare(record_batch.to_pandas())
        for (source, sid), group in frame.groupby(["source_file", "id"], sort=False):
            key = f"{source}::{sid}"
            old_tail = tails.get(key)
            if old_tail is not None:
                group = pd.concat([old_tail, group], ignore_index=True)
                new_mask = np.r_[np.zeros(len(old_tail), dtype=bool), np.ones(len(group) - len(old_tail), dtype=bool)]
            else:
                new_mask = np.ones(len(group), dtype=bool)
            if bool(group.subject_split_across_traintest.astype(bool).any()):
                tails[key] = group.tail(history_steps + horizon_steps).copy()
                continue
            rows, metas = _build_group(group, history_steps, horizon_steps, stride_steps, new_mask)
            for row, meta in zip(rows, metas):
                split = meta["split"]
                subject = f"{meta['source_file']}::{meta['id']}"
                if subject_counts[subject] >= max_subject_transitions:
                    continue
                split_rows[split].append(row)
                split_meta[split].append(meta)
                subject_counts[subject] += 1
            tails[key] = group.tail(history_steps + horizon_steps).copy()
        if batch_idx == 1 or batch_idx % 20 == 0:
            counts_now = {split: len(rows) for split, rows in split_rows.items()}
            print(f"[data] batch={batch_idx} transitions={counts_now}", flush=True)

    split_rows, split_meta = balanced_sample_splits(
        split_rows,
        split_meta,
        max_transitions=max_transitions,
        min_split_transitions=min_split_transitions,
    )

    counts = {}
    for split, rows in split_rows.items():
        arrays = rows_to_arrays(rows)
        np.savez_compressed(output_dir / f"transitions_{split}.npz", **arrays)
        pd.DataFrame(split_meta[split]).to_parquet(output_dir / f"metadata_{split}.parquet", index=False)
        counts[split] = int(len(arrays["action"]))
    with (output_dir / "mdp_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "seq_features": SEQ_FEATURES,
                "static_features": STATIC_FEATURES,
                "action_labels": ACTION_LABELS,
                "history_steps": history_steps,
                "horizon_steps": horizon_steps,
                "stride_steps": stride_steps,
                "max_subject_transitions": max_subject_transitions,
                "reward_components": ["neg_tbr54", "neg_tar250", "neg_burden"],
                "outcome_components": ["tir", "tbr54", "tar250", "burden"],
                "counts": counts,
            },
            handle,
            indent=2,
        )
    print(f"[data] transitions={counts}", flush=True)
    return counts


def rows_to_arrays(rows: list) -> dict[str, np.ndarray]:
    if not rows:
        return {
            "seq": np.empty((0, 12, len(SEQ_FEATURES) + 1), dtype=np.float32),
            "static": np.empty((0, len(STATIC_FEATURES)), dtype=np.float32),
            "next_seq": np.empty((0, 12, len(SEQ_FEATURES) + 1), dtype=np.float32),
            "next_static": np.empty((0, len(STATIC_FEATURES)), dtype=np.float32),
            "action": np.empty((0,), dtype=np.int64),
            "reward_components": np.empty((0, 3), dtype=np.float32),
            "outcome_components": np.empty((0, 4), dtype=np.float32),
            "done": np.empty((0,), dtype=np.float32),
        }
    seq, static, nseq, nstatic, action, reward, outcome, done = zip(*rows)
    return {
        "seq": np.asarray(seq, dtype=np.float32),
        "static": np.asarray(static, dtype=np.float32),
        "next_seq": np.asarray(nseq, dtype=np.float32),
        "next_static": np.asarray(nstatic, dtype=np.float32),
        "action": np.asarray(action, dtype=np.int64),
        "reward_components": np.asarray(reward, dtype=np.float32),
        "outcome_components": np.asarray(outcome, dtype=np.float32),
        "done": np.asarray(done, dtype=np.float32),
    }


def load_npz(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path)
    return {key: payload[key] for key in payload.files}
