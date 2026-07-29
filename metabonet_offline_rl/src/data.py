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
ACTION_LABELS_12 = [
    f"{basal}|{bolus}"
    for basal in ("basal_down", "basal_same", "basal_up")
    for bolus in ("no_bolus", "bolus_le2", "bolus_2_5", "bolus_gt5")
]
ACTION_LABELS_BOLUS4 = ["no_bolus", "bolus_le2", "bolus_2_5", "bolus_gt5"]
ACTION_LABELS = ACTION_LABELS_BOLUS4
N_ACTIONS = len(ACTION_LABELS)
VECTOR_FEATURES = [
    "cgm_now_z",
    "cgm_delta_15",
    "cgm_delta_30",
    "cgm_delta_60",
    "cgm_delta_120",
    "cgm_mean_30",
    "cgm_mean_60",
    "cgm_mean_120",
    "cgm_std_60",
    "cgm_min_120",
    "cgm_max_120",
    "basal_now_scaled",
    "bolus_sum_30",
    "bolus_sum_60",
    "bolus_sum_120",
    "carbs_sum_30",
    "carbs_sum_60",
    "carbs_sum_120",
    "insulin_sum_120",
    "time_since_bolus",
    "time_since_carbs",
    "iob_proxy",
    "cob_proxy",
    "hour_sin",
    "hour_cos",
] + STATIC_FEATURES


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


def action_labels(action_mode: str) -> list[str]:
    if action_mode == "bolus4":
        return ACTION_LABELS_BOLUS4
    if action_mode == "basal_bolus12":
        return ACTION_LABELS_12
    raise ValueError(f"unknown action_mode={action_mode}")


def encode_action(interval_basal_delta: float, interval_bolus: float, action_mode: str) -> int:
    bolus_action = _bolus_bin(interval_bolus)
    if action_mode == "bolus4":
        return bolus_action
    if action_mode == "basal_bolus12":
        return _basal_bin(interval_basal_delta) * 4 + bolus_action
    raise ValueError(f"unknown action_mode={action_mode}")


def _window(values: np.ndarray, t: int, steps: int) -> np.ndarray:
    return values[max(0, t - steps + 1): t + 1]


def _finite_mean(values: np.ndarray, default: float) -> float:
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else default


def _finite_std(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(values.std()) if len(values) else 0.0


def _finite_min(values: np.ndarray, default: float) -> float:
    values = values[np.isfinite(values)]
    return float(values.min()) if len(values) else default


def _finite_max(values: np.ndarray, default: float) -> float:
    values = values[np.isfinite(values)]
    return float(values.max()) if len(values) else default


def _time_since_event(event: np.ndarray, t: int, max_steps: int) -> float:
    hits = np.flatnonzero(_window(event.astype(float), t, max_steps) > 0)
    if len(hits) == 0:
        return float(max_steps)
    return float(len(_window(event.astype(float), t, max_steps)) - 1 - hits[-1])


def _engineered_vector(group: pd.DataFrame, t: int, stat: np.ndarray) -> np.ndarray:
    cgm = group.CGM.to_numpy(float)
    basal = group.basal.ffill().fillna(0).to_numpy(float)
    bolus = group.bolus.fillna(0).clip(lower=0).to_numpy(float)
    carbs = group.carbs.fillna(0).clip(lower=0).to_numpy(float)
    insulin = group.insulin.fillna(0).clip(lower=0).to_numpy(float)
    hour = float(group.date.iloc[t].hour) + float(group.date.iloc[t].minute) / 60.0
    cgm_now = cgm[t] if np.isfinite(cgm[t]) else 150.0

    def delta(lag: int) -> float:
        idx = max(0, t - lag)
        prior = cgm[idx] if np.isfinite(cgm[idx]) else cgm_now
        return (cgm_now - prior) / 60.0

    recent_bolus = _window(bolus, t, 48)
    recent_carbs = _window(carbs, t, 48)
    decay = np.exp(-np.arange(len(recent_bolus) - 1, -1, -1) / 24.0)
    vector = np.array(
        [
            (cgm_now - 150.0) / 60.0,
            delta(3),
            delta(6),
            delta(12),
            delta(24),
            (_finite_mean(_window(cgm, t, 6), cgm_now) - 150.0) / 60.0,
            (_finite_mean(_window(cgm, t, 12), cgm_now) - 150.0) / 60.0,
            (_finite_mean(_window(cgm, t, 24), cgm_now) - 150.0) / 60.0,
            _finite_std(_window(cgm, t, 12)) / 60.0,
            (_finite_min(_window(cgm, t, 24), cgm_now) - 150.0) / 60.0,
            (_finite_max(_window(cgm, t, 24), cgm_now) - 150.0) / 60.0,
            float(basal[t]) / 3.0,
            float(np.sum(_window(bolus, t, 6))) / 10.0,
            float(np.sum(_window(bolus, t, 12))) / 10.0,
            float(np.sum(_window(bolus, t, 24))) / 10.0,
            float(np.sum(_window(carbs, t, 6))) / 100.0,
            float(np.sum(_window(carbs, t, 12))) / 100.0,
            float(np.sum(_window(carbs, t, 24))) / 100.0,
            float(np.sum(_window(insulin, t, 24))) / 10.0,
            _time_since_event(bolus > 0, t, 48) / 48.0,
            _time_since_event(carbs > 0, t, 48) / 48.0,
            float(np.sum(recent_bolus * decay)) / 10.0,
            float(np.sum(recent_carbs * decay)) / 100.0,
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            *stat.tolist(),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _build_group(
    group: pd.DataFrame,
    history_steps: int,
    action_steps: int,
    reward_delay_steps: int,
    reward_steps: int,
    stride_steps: int,
    new_row_mask: np.ndarray,
    action_mode: str,
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
    max_forward_steps = max(action_steps, reward_delay_steps + reward_steps)

    for t in range(history_steps - 1, len(group) - max_forward_steps, stride_steps):
        if not new_row_mask[t]:
            continue
        action_end = t + action_steps
        reward_start = t + reward_delay_steps + 1
        reward_end = t + reward_delay_steps + reward_steps
        action_minutes = (dates[action_end] - dates[t]) / np.timedelta64(1, "m")
        reward_minutes = (dates[reward_end] - dates[t]) / np.timedelta64(1, "m")
        if not 20 <= float(action_minutes) <= 45:
            continue
        if not 90 <= float(reward_minutes) <= 150:
            continue
        future_cgm = cgm[reward_start: reward_end + 1]
        if not np.isfinite(future_cgm).any() or not np.isfinite(cgm[t]):
            continue
        interval_bolus = float(np.nansum(bolus[t + 1: action_end + 1]))
        interval_basal_delta = float(np.nanmean(basal[t + 1: action_end + 1]) - basal[t])
        action = encode_action(interval_basal_delta, interval_bolus, action_mode)
        bolus_event = float(interval_bolus > 1e-9)
        valid = future_cgm[np.isfinite(future_cgm)]
        tir = float(np.mean((valid >= 70) & (valid <= 180)))
        tbr70 = float(np.mean(valid < 70))
        tbr54 = float(np.mean(valid < 54))
        tar180 = float(np.mean(valid > 180))
        tar250 = float(np.mean(valid > 250))
        glycemic_effectiveness = -float(np.mean(np.maximum(valid - 180.0, 0.0) / 70.0)) - 2.0 * tar250
        low_burden = -bolus_event - interval_bolus / 10.0
        hypo_safety = -(tbr70 + 3.0 * tbr54)
        rows.append(
            (
                _encode_window(features, t, history_steps),
                _engineered_vector(group, t, stat),
                _encode_window(features, action_end, history_steps),
                _engineered_vector(group, action_end, stat),
                action,
                np.array([glycemic_effectiveness, low_burden, hypo_safety], dtype=np.float32),
                np.array(
                    [glycemic_effectiveness, low_burden, hypo_safety, tir, tbr70, tbr54, tar180, tar250, interval_bolus, bolus_event],
                    dtype=np.float32,
                ),
                0.0,
            )
        )
        meta.append(
            {
                "source_file": group.source_file.iloc[0],
                "id": group.id.iloc[0],
                "date": group.date.iloc[t],
                "action_end": group.date.iloc[action_end],
                "reward_start": group.date.iloc[reward_start],
                "reward_end": group.date.iloc[reward_end],
                "split": split,
                "action": action,
                "bolus_units": interval_bolus,
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
    history_steps: int = 24,
    action_steps: int = 6,
    reward_delay_steps: int = 6,
    reward_steps: int = 18,
    stride_steps: int = 6,
    action_mode: str = "bolus4",
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
                tails[key] = group.tail(history_steps + reward_delay_steps + reward_steps).copy()
                continue
            rows, metas = _build_group(
                group,
                history_steps,
                action_steps,
                reward_delay_steps,
                reward_steps,
                stride_steps,
                new_mask,
                action_mode,
            )
            for row, meta in zip(rows, metas):
                split = meta["split"]
                subject = f"{meta['source_file']}::{meta['id']}"
                if subject_counts[subject] >= max_subject_transitions:
                    continue
                split_rows[split].append(row)
                split_meta[split].append(meta)
                subject_counts[subject] += 1
            tails[key] = group.tail(history_steps + reward_delay_steps + reward_steps).copy()
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
        arrays = rows_to_arrays(rows, n_actions=len(action_labels(action_mode)), history_steps=history_steps)
        np.savez_compressed(output_dir / f"transitions_{split}.npz", **arrays)
        pd.DataFrame(split_meta[split]).to_parquet(output_dir / f"metadata_{split}.parquet", index=False)
        counts[split] = int(len(arrays["action"]))
    with (output_dir / "mdp_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "seq_features": SEQ_FEATURES,
                "static_features": VECTOR_FEATURES,
                "action_labels": action_labels(action_mode),
                "action_mode": action_mode,
                "history_steps": history_steps,
                "action_steps": action_steps,
                "reward_delay_steps": reward_delay_steps,
                "reward_steps": reward_steps,
                "stride_steps": stride_steps,
                "max_subject_transitions": max_subject_transitions,
                "reward_components": ["glycemic_effectiveness", "low_burden", "hypo_safety"],
                "outcome_components": [
                    "glycemic_effectiveness",
                    "low_burden",
                    "hypo_safety",
                    "tir",
                    "tbr70",
                    "tbr54",
                    "tar180",
                    "tar250",
                    "bolus_units",
                    "bolus_event",
                ],
                "counts": counts,
            },
            handle,
            indent=2,
        )
    print(f"[data] transitions={counts}", flush=True)
    return counts


def rows_to_arrays(rows: list, n_actions: int = N_ACTIONS, history_steps: int = 24) -> dict[str, np.ndarray]:
    if not rows:
        return {
            "seq": np.empty((0, history_steps, len(SEQ_FEATURES) + 1), dtype=np.float32),
            "static": np.empty((0, len(VECTOR_FEATURES)), dtype=np.float32),
            "next_seq": np.empty((0, history_steps, len(SEQ_FEATURES) + 1), dtype=np.float32),
            "next_static": np.empty((0, len(VECTOR_FEATURES)), dtype=np.float32),
            "action": np.empty((0,), dtype=np.int64),
            "n_actions": np.asarray(n_actions, dtype=np.int64),
            "reward_components": np.empty((0, 3), dtype=np.float32),
            "outcome_components": np.empty((0, 10), dtype=np.float32),
            "done": np.empty((0,), dtype=np.float32),
        }
    seq, static, nseq, nstatic, action, reward, outcome, done = zip(*rows)
    return {
        "seq": np.asarray(seq, dtype=np.float32),
        "static": np.asarray(static, dtype=np.float32),
        "next_seq": np.asarray(nseq, dtype=np.float32),
        "next_static": np.asarray(nstatic, dtype=np.float32),
        "action": np.asarray(action, dtype=np.int64),
        "n_actions": np.asarray(n_actions, dtype=np.int64),
        "reward_components": np.asarray(reward, dtype=np.float32),
        "outcome_components": np.asarray(outcome, dtype=np.float32),
        "done": np.asarray(done, dtype=np.float32),
    }


def load_npz(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path)
    return {key: payload[key] for key in payload.files}
