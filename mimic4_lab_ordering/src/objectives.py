"""Detection and burden objectives for the joint-panel lab-ordering track."""
import json

import numpy as np
import pandas as pd

import config as cfg
import itemids as ids
import panels


def _future_any_by_stay(values, stay_ids, lookahead):
    """For row t, true if any event occurs in (t, t + lookahead]."""
    out = np.zeros(len(values), dtype=bool)
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or stay_ids[i] != stay_ids[start]:
            v = values[start:i].astype(np.int8)
            n = len(v)
            c = np.r_[0, np.cumsum(v)]
            local = np.zeros(n, dtype=bool)
            for j in range(n):
                hi = min(n, j + lookahead + 1)
                local[j] = (c[hi] - c[j + 1]) > 0
            out[start:i] = local
            start = i
    return out


def _recent_any_by_stay(values, stay_ids, lookback):
    """For row t, true if any draw occurred in [t - lookback, t)."""
    out = np.zeros(len(values), dtype=bool)
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or stay_ids[i] != stay_ids[start]:
            v = values[start:i].astype(np.int8)
            c = np.r_[0, np.cumsum(v)]
            n = len(v)
            local = np.zeros(n, dtype=bool)
            for j in range(n):
                lo = max(0, j - lookback)
                local[j] = (c[j] - c[lo]) > 0
            out[start:i] = local
            start = i
    return out


def deterioration_events(df):
    onset_cols = [f"onset_{k}" for k in ids.INTERVENTION_KINDS]
    onset = df[onset_cols].to_numpy().sum(axis=1) > 0
    sofa = df["sofa_delta"].to_numpy() >= cfg.SOFA_DELTA_THRESHOLD
    return onset | sofa


def detection_objective(df, actions, lookahead=cfg.JOINT_DETECTION_LOOKAHEAD_HOURS):
    stay = df["stay_id"].to_numpy()
    event = deterioration_events(df)
    future_event = _future_any_by_stay(event, stay, lookahead)
    any_draw = np.asarray(actions) != 0
    recent_draw = _recent_any_by_stay(any_draw, stay, lookahead)

    r = np.zeros(len(df), dtype=np.float32)
    r[future_event & any_draw] = 1.0
    r[future_event & ~any_draw & ~recent_draw] = -1.0
    return r, future_event.astype(np.int8), event.astype(np.int8)


def burden_objective(df, actions):
    bits = panels.action_bits(actions).astype(np.float32)
    deltas = []
    for lab in panels.LABS:
        d = df[f"delta_{lab}"].to_numpy(dtype=np.float32)
        d = np.where(np.isfinite(d), d, np.inf)
        deltas.append(d)
    deltas = np.stack(deltas, axis=1)
    redundancy = (bits * np.exp(-deltas / cfg.COST_DECAY_GAMMA)).sum(axis=1)
    any_draw = (np.asarray(actions) != 0).astype(np.float32)
    return (any_draw * (1.0 + redundancy)).astype(np.float32)


def normalize_rewards(train_reward, *splits):
    mu = train_reward.mean(axis=0).astype(np.float32)
    sd = train_reward.std(axis=0).astype(np.float32)
    sd[sd < 1e-6] = 1.0
    out = [((x - mu) / sd).astype(np.float32) for x in splits]
    return out, {"reward_mean": mu.tolist(), "reward_sd": sd.tolist()}


def save_meta(path, meta):
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def action_distribution(actions):
    s = pd.Series(actions)
    rows = []
    for a, n in s.value_counts().sort_index().items():
        rows.append({
            "action": int(a),
            "bits": cfg.JOINT_PANEL_BITS[int(a)],
            "panel": cfg.JOINT_PANEL_NAMES[int(a)],
            "count": int(n),
            "frac": float(n / len(actions)),
        })
    return rows
