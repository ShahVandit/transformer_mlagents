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
    """One credit per deterioration event, one penalty per uncovered one.

        +1  draw, event ahead, nothing drawn recently   (this draw claims the event)
        -1  no draw, event ahead, nothing drawn recently (the event goes uncovered)
         0  otherwise

    The `not recent_draw` gate on the POSITIVE branch is the important part.
    Without it, `+1` fired on every hour with an event inside the lookahead
    window, and since events occur in ~3% of hours but "an event within 12h"
    covers ~32% of them, a policy that drew every single hour collected +1 on a
    third of all hours. One event could pay out twelve times. That is a direct,
    large incentive to draw constantly, and it produced policies ordering ~10
    times a day against a clinician rate of 2.9.

    With the gate, the first draw in a window claims the event and later draws
    in the same window earn nothing, so the payout is capped at roughly one per
    event. Crediting the FIRST draw rather than the last also rewards
    anticipation rather than a last-minute order, which matches the paper's
    time-to-treatment framing.

    Both branches now share the same gate, so the objective is symmetric: within
    a window you are paid once for covering it, or charged once for not. The
    gate stays Markov because `recent_draw` is derivable from the `delta_*`
    terms already in the state.
    """
    stay = _to_numpy(_col(df, "stay_id"))
    event = deterioration_events(df)
    future_event = _future_any_by_stay(event, stay, lookahead)
    any_draw = np.asarray(actions) != 0
    recent_draw = _recent_any_by_stay(any_draw, stay, lookahead)

    claimable = future_event & ~recent_draw
    r = np.zeros(len(stay), dtype=np.float32)
    r[claimable & any_draw] = 1.0
    r[claimable & ~any_draw] = -1.0
    return r, future_event.astype(np.int8), event.astype(np.int8)


def burden_objective(df, actions):
    stay = _to_numpy(_col(df, "stay_id"))
    hour = _to_numpy(_col(df, "hour")).astype(np.float32)
    bits = panels.action_bits(actions).astype(np.float32)
    out = np.zeros(len(stay), dtype=np.float32)

    start = 0
    for i in range(1, len(stay) + 1):
        if i == len(stay) or stay[i] != stay[start]:
            last_seen = np.full(len(panels.LABS), np.nan, dtype=np.float32)
            for j in range(start, i):
                if bits[j].sum() == 0:
                    continue
                # A lab never drawn before carries NO redundancy: infinite
                # elapsed time, so exp(-delta/Gamma) is 0 and the draw costs
                # just the base 1.0. Imputing `hour + 1` here (which is right in
                # the STATE, where Delta stands in for time since admission)
                # charges a first-ever draw as though the lab had been taken an
                # hour earlier: a 3-lab panel at hour 0 came to 3.54 instead of
                # 1.00. That falls hardest on admission labs, biasing every
                # policy in the family against early testing.
                delta = np.where(
                    np.isfinite(last_seen),
                    hour[j] - last_seen,
                    np.inf,
                )
                out[j] = 1.0 + float(
                    (bits[j] * np.exp(-delta / cfg.COST_DECAY_GAMMA)).sum()
                )
                last_seen = np.where(bits[j] > 0, hour[j], last_seen)
            start = i
    return out


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
