"""Detection and burden objectives for the joint-panel lab-ordering track."""
import json

import numpy as np
import pandas as pd

import config as cfg
import itemids as ids
import panels


def _col(obj, name):
    if isinstance(obj, dict):
        return obj[name]
    if hasattr(obj, "__getitem__"):
        return obj[name]
    raise TypeError(f"unsupported data container for column {name!r}")


def _to_numpy(x):
    return x.to_numpy() if hasattr(x, "to_numpy") else np.asarray(x)


def _has_col(obj, name):
    if isinstance(obj, dict):
        return name in obj
    return hasattr(obj, "columns") and name in obj.columns


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
    if _has_col(df, "event"):
        return _to_numpy(_col(df, "event")).astype(bool)
    onset_cols = [f"onset_{k}" for k in ids.INTERVENTION_KINDS]
    onset = np.column_stack([_to_numpy(_col(df, c)) for c in onset_cols]).sum(axis=1) > 0
    sofa = _to_numpy(_col(df, "sofa_delta")) >= cfg.SOFA_DELTA_THRESHOLD
    return onset | sofa


def detection_objective(df, actions, lookahead=cfg.JOINT_DETECTION_LOOKAHEAD_HOURS):
    """Score every evaluable deterioration exactly once.

    Consecutive positive hours are treated as one deterioration episode. For an
    episode starting at hour ``t``, a draw in ``[t-lookahead, t)`` covers it. The
    most recent eligible draw receives ``+1``; if there is no eligible draw,
    the last decision before the event receives ``-1``. Events in the first
    hour are excluded because the policy had no earlier decision opportunity.

    This event-level assignment is deliberately symmetric. One event can add
    exactly ``+1`` or ``-1`` to the trajectory, never one reward per hour in
    its lookahead window. Additional draws cannot increase detection reward;
    they are handled only by ``burden_objective``.
    """
    stay = _to_numpy(_col(df, "stay_id"))
    event = deterioration_events(df)
    future_event = _future_any_by_stay(event, stay, lookahead)
    any_draw = np.asarray(actions) != 0
    r = np.zeros(len(stay), dtype=np.float32)

    start = 0
    for i in range(1, len(stay) + 1):
        if i == len(stay) or stay[i] != stay[start]:
            local_draw = any_draw[start:i]
            local_event = event[start:i]
            raw_onsets = np.flatnonzero(
                local_event & ~np.r_[False, local_event[:-1]])
            # Multiple labels close together usually describe one continuing
            # clinical deterioration. Keep only the first onset in each
            # lookahead-sized episode so one draw cannot collect repeated credit
            # for a clustered vasopressor/SOFA/ventilation sequence.
            event_onsets = []
            for onset in raw_onsets:
                if not event_onsets or onset - event_onsets[-1] > lookahead:
                    event_onsets.append(int(onset))
            for event_idx in event_onsets:
                if event_idx == 0:
                    continue
                lo = max(0, event_idx - lookahead)
                candidates = np.flatnonzero(local_draw[lo:event_idx])
                if len(candidates):
                    chosen = lo + int(candidates[-1])
                    r[start + chosen] += 1.0
                else:
                    r[start + event_idx - 1] -= 1.0
            start = i
    return r, future_event.astype(np.int8), event.astype(np.int8)


def burden_objective(df, actions):
    stay = _to_numpy(_col(df, "stay_id"))
    hour = _to_numpy(_col(df, "hour")).astype(np.float32)
    any_draw = np.asarray(actions) != 0
    out = np.zeros(len(stay), dtype=np.float32)

    start = 0
    for i in range(1, len(stay) + 1):
        if i == len(stay) or stay[i] != stay[start]:
            last_seen = np.nan
            for j in range(start, i):
                if not any_draw[j]:
                    continue
                # One physical draw has one base cost. Repeating it soon adds a
                # redundancy surcharge that decays with time since the previous
                # draw; the first draw has no redundancy surcharge.
                delta = hour[j] - last_seen if np.isfinite(last_seen) else np.inf
                out[j] = 1.0 + float(np.exp(-delta / cfg.COST_DECAY_GAMMA))
                last_seen = hour[j]
            start = i
    return out


def normalize_rewards(train_reward, *splits):
    """Scale objectives without centering, preserving zero as neutral.

    Mean-centering sparse event rewards makes every ordinary zero-reward hour
    nonzero. Summing those offsets over stays of different lengths obscures the
    actual event score and creates large, hard-to-interpret episode returns.
    Scale-only normalization keeps ``+1`` and ``-1`` symmetric and leaves a
    neutral transition at exactly zero.
    """
    raw_mu = train_reward.mean(axis=0).astype(np.float32)
    sd = train_reward.std(axis=0).astype(np.float32)
    sd[sd < 1e-6] = 1.0
    out = [(x / sd).astype(np.float32) for x in splits]
    return out, {
        "reward_mean": np.zeros_like(raw_mu).tolist(),
        "reward_sd": sd.tolist(),
        "raw_reward_mean": raw_mu.tolist(),
        "normalization": "scale_only",
    }


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
