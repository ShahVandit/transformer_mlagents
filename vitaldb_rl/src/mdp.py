"""Convert VitalDB cases into offline MDP transition arrays."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import data as V

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

CHANNELS = ["map", "sbp", "dbp", "hr", "spo2", "etco2", "bis", "ppf_rate", "rftn_rate"]
RATE_COLS = ["ppf_rate", "rftn_rate"]
STAT_NUM = ["age", "bmi", "asa"]
STAT_BIN = ["sex", "emop", "preop_htn", "preop_dm"]


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Fill action-rate tracks as pump state; keep physiology missingness explicit."""
    out = df.copy()
    for col in RATE_COLS:
        out[col] = out[col].ffill().fillna(0.0)
    return out


def fit_norm(caseids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    s = np.zeros(len(CHANNELS))
    s2 = np.zeros(len(CHANNELS))
    n = np.zeros(len(CHANNELS))
    for cid in caseids:
        df = prepare_frame(V.load_case(cid))
        for j, col in enumerate(CHANNELS):
            vals = df[col].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            s[j] += vals.sum()
            s2[j] += (vals * vals).sum()
            n[j] += vals.size
    mean = np.where(n > 0, s / np.maximum(n, 1), 0.0)
    var = np.where(n > 0, s2 / np.maximum(n, 1) - mean * mean, 1.0)
    return mean.astype(np.float32), np.sqrt(np.maximum(var, 1e-6)).astype(np.float32)


def build_static(meta: pd.DataFrame, train_ids: list[int]) -> pd.DataFrame:
    """Decision-time static features, normalized with train subjects only."""
    meta = meta.set_index("caseid", drop=False)
    train = meta.loc[meta.index.intersection([int(x) for x in train_ids])]
    rows = {}
    med = {c: pd.to_numeric(train.get(c), errors="coerce").median() for c in STAT_NUM if c in meta}
    top_op = train["optype"].value_counts().head(8).index.tolist() if "optype" in train else []
    top_ane = train["ane_type"].value_counts().head(4).index.tolist() if "ane_type" in train else []

    for cid, row in meta.iterrows():
        vec = {}
        for col in STAT_NUM:
            if col not in meta:
                continue
            val = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
            vec[col] = float(med[col] if pd.isna(val) else val)
            vec[f"{col}_miss"] = float(pd.isna(val))
        if "sex" in meta:
            vec["sex_male"] = float(str(row.get("sex")).upper().startswith("M"))
        for col in ["emop", "preop_htn", "preop_dm"]:
            if col in meta:
                vec[col] = float(pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").fillna(0).iloc[0])
        for op in top_op:
            vec[f"op_{op}"] = float(row.get("optype") == op)
        if top_op:
            vec["op_other"] = float(row.get("optype") not in top_op)
        for ane in top_ane:
            vec[f"ane_{ane}"] = float(row.get("ane_type") == ane)
        if top_ane:
            vec["ane_other"] = float(row.get("ane_type") not in top_ane)
        rows[int(cid)] = vec

    X = pd.DataFrame(rows).T.fillna(0.0).astype(np.float32)
    train_idx = [int(x) for x in train_ids if int(x) in X.index]
    for col in [c for c in STAT_NUM if c in X.columns]:
        mu = float(X.loc[train_idx, col].mean())
        sd = float(X.loc[train_idx, col].std()) or 1.0
        X[col] = (X[col] - mu) / sd
    return X.astype(np.float32)


def encode_window(df: pd.DataFrame, t: int, history: int, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    arr = np.stack([df[c].to_numpy(float) for c in CHANNELS], axis=1)
    mask = np.isnan(arr).astype(np.float32)
    z = np.nan_to_num((arr - mean) / std, nan=0.0).astype(np.float32)
    feat = np.concatenate([z, mask], axis=1)
    out = np.zeros((history, feat.shape[1] + 1), dtype=np.float32)
    out[:, -1] = 1.0
    win = feat[max(0, t - history + 1): t + 1]
    out[-len(win):, :-1] = win
    out[-len(win):, -1] = 0.0
    return out


def bin_delta(current: float, future: float, abs_min: float = 1.0, rel_min: float = 0.05) -> int:
    threshold = max(abs_min, rel_min * abs(current))
    delta = future - current
    if delta < -threshold:
        return 0
    if delta > threshold:
        return 2
    return 1


def encode_action(ppf_now: float, ppf_next: float, rftn_now: float, rftn_next: float) -> int:
    ppf = bin_delta(ppf_now, ppf_next)
    rftn = bin_delta(rftn_now, rftn_next)
    return int(ppf * 3 + rftn)


def decode_action(action: int) -> tuple[int, int]:
    return int(action) // 3, int(action) % 3


def reward_components(df: pd.DataFrame, t0: int, t1: int, action: int) -> np.ndarray:
    fut = df.iloc[t0 + 1: t1 + 1]
    mapv = fut["map"].to_numpy(float)
    bisv = fut["bis"].to_numpy(float)

    if np.isfinite(mapv).any():
        low = np.clip((65.0 - mapv[np.isfinite(mapv)]) / 20.0, 0.0, 3.0)
        map_pen = float(low.mean() + 0.5 * np.mean(mapv[np.isfinite(mapv)] < 65.0))
    else:
        map_pen = 0.0

    if np.isfinite(bisv).any():
        b = bisv[np.isfinite(bisv)]
        low = np.clip((40.0 - b) / 20.0, 0.0, 3.0)
        high = np.clip((b - 60.0) / 20.0, 0.0, 3.0)
        bis_pen = float(np.maximum(low, high).mean())
    else:
        bis_pen = 0.0

    ppf_bin, rftn_bin = decode_action(action)
    work_pen = float((ppf_bin != 1) + (rftn_bin != 1)) / 2.0
    return np.array([-map_pen, -bis_pen, -work_pen], dtype=np.float32)


def build_split_arrays(
    caseids: list[int],
    static_x: pd.DataFrame,
    norm: tuple[np.ndarray, np.ndarray],
    history: int = 30,
    step: int = 5,
    horizon: int = 5,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    mean, std = norm
    seq, stat, nseq, nstat, actions, rewards, done, rows = [], [], [], [], [], [], [], []
    for cid in caseids:
        df = prepare_frame(V.load_case(int(cid)))
        n = len(df)
        if n <= history + horizon + 1 or int(cid) not in static_x.index:
            continue
        for t in range(history - 1, n - horizon, step):
            nt = t + horizon
            p0, p1 = float(df.ppf_rate.iloc[t]), float(df.ppf_rate.iloc[nt])
            r0, r1 = float(df.rftn_rate.iloc[t]), float(df.rftn_rate.iloc[nt])
            if max(p0, p1, r0, r1) <= 0.0:
                continue
            fut = df.iloc[t + 1: nt + 1]
            if not fut["map"].notna().any() or not fut["bis"].notna().any():
                continue
            a = encode_action(p0, p1, r0, r1)
            seq.append(encode_window(df, t, history, mean, std))
            nseq.append(encode_window(df, nt, history, mean, std))
            sx = static_x.loc[int(cid)].to_numpy(np.float32)
            stat.append(sx)
            nstat.append(sx)
            actions.append(a)
            rewards.append(reward_components(df, t, nt, a))
            done.append(float(nt + horizon >= n))
            rows.append({"caseid": int(cid), "minute": int(t), "action": int(a), "split": ""})

    arrays = {
        "seq": np.asarray(seq, dtype=np.float32),
        "static": np.asarray(stat, dtype=np.float32),
        "next_seq": np.asarray(nseq, dtype=np.float32),
        "next_static": np.asarray(nstat, dtype=np.float32),
        "action": np.asarray(actions, dtype=np.int64),
        "reward_components": np.asarray(rewards, dtype=np.float32),
        "done": np.asarray(done, dtype=np.float32),
    }
    return arrays, pd.DataFrame(rows)


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def build_transition_dataset(
    limit: int | None = None,
    workers: int = 8,
    refresh: bool = False,
    history: int = 30,
    step: int = 5,
    horizon: int = 5,
) -> dict[str, int]:
    meta = V.build_cache(limit=limit, workers=workers, refresh=refresh)
    splits = V.subject_split(meta)
    norm = fit_norm(splits["train"])
    static_x = build_static(meta, splits["train"])

    np.savez(DATA / "normalization.npz", mean=norm[0], std=norm[1])
    static_x.to_parquet(DATA / "static_features.parquet")
    with open(DATA / "mdp_config.json", "w") as f:
        json.dump({"history": history, "step": step, "horizon": horizon, "channels": CHANNELS}, f, indent=2)

    all_rows = []
    counts = {}
    for split, ids in splits.items():
        arrays, rows = build_split_arrays(ids, static_x, norm, history, step, horizon)
        rows["split"] = split
        save_npz(DATA / f"transitions_{split}.npz", arrays)
        all_rows.append(rows)
        counts[split] = int(len(arrays["action"]))
    pd.concat(all_rows, ignore_index=True).to_parquet(DATA / "transitions.parquet")
    print("[mdp] transitions:", counts)
    return counts

