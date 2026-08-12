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


def deterioration_episode_onsets(df, lookahead=cfg.JOINT_DETECTION_LOOKAHEAD_HOURS):
    """Collapse nearby deterioration labels into distinct episode onsets."""
    stay = _to_numpy(_col(df, "stay_id"))
    event = deterioration_events(df)
    episode_onset = np.zeros(len(event), dtype=bool)

    start = 0
    for i in range(1, len(stay) + 1):
        if i == len(stay) or stay[i] != stay[start]:
            local_event = event[start:i]
            raw_onsets = np.flatnonzero(
                local_event & ~np.r_[False, local_event[:-1]])
            last_onset = None
            for onset in raw_onsets:
                if last_onset is None or onset - last_onset > lookahead:
                    episode_onset[start + int(onset)] = True
                    last_onset = int(onset)
            start = i
    return episode_onset


def detection_objective(df, actions, lookahead=cfg.JOINT_DETECTION_LOOKAHEAD_HOURS):
    """Score every evaluable deterioration exactly once.

    Consecutive positive hours are treated as one deterioration episode. For an
    episode starting at hour ``t``, a draw in ``[t-lookahead, t)`` covers it. The
    earliest eligible draw receives ``+1``; if there is no eligible draw,
    the last decision before the event receives ``-1``. Events in the first
    hour are excluded because the policy had no earlier decision opportunity.

    This event-level assignment is deliberately symmetric. One event can add
    exactly ``+1`` or ``-1`` to the trajectory, never one reward per hour in
    its lookahead window. Additional draws cannot increase detection reward or
    move credit away from an earlier transition; they are handled only by
    ``burden_objective``.
    """
    stay = _to_numpy(_col(df, "stay_id"))
    event = deterioration_events(df)
    episode_onset = deterioration_episode_onsets(df, lookahead)
    future_event = _future_any_by_stay(episode_onset, stay, lookahead)
    any_draw = np.asarray(actions) != 0
    r = np.zeros(len(stay), dtype=np.float32)

    start = 0
    for i in range(1, len(stay) + 1):
        if i == len(stay) or stay[i] != stay[start]:
            local_draw = any_draw[start:i]
            event_onsets = np.flatnonzero(episode_onset[start:i])
            for event_idx in event_onsets:
                if event_idx == 0:
                    continue
                lo = max(0, event_idx - lookahead)
                candidates = np.flatnonzero(local_draw[lo:event_idx])
                if len(candidates):
                    # Later actions must never rewrite an earlier transition's
                    # reward, so the first qualifying draw keeps the credit.
                    chosen = lo + int(candidates[0])
                    r[start + chosen] += 1.0
                else:
                    r[start + event_idx - 1] -= 1.0
            start = i
    return r, future_event.astype(np.int8), event.astype(np.int8)


def event_coverage(df, actions, lookback=cfg.JOINT_DETECTION_LOOKAHEAD_HOURS):
    """Fraction of evaluable deterioration episodes covered by a prior draw."""
    stay = _to_numpy(_col(df, "stay_id"))
    episode_onset = deterioration_episode_onsets(df, lookback)
    draw = np.asarray(actions) != 0
    covered = []

    start = 0
    for i in range(1, len(stay) + 1):
        if i == len(stay) or stay[i] != stay[start]:
            local_draw = draw[start:i].astype(np.int8)
            cumulative = np.r_[0, np.cumsum(local_draw)]
            for event_idx in np.flatnonzero(episode_onset[start:i]):
                if event_idx == 0:
                    continue
                lo = max(0, event_idx - lookback)
                covered.append((cumulative[event_idx] - cumulative[lo]) > 0)
            start = i
    return float(np.mean(covered)) if covered else np.nan


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


def assert_rewards_current(split, norm_meta=None, name="split", tol=1e-3):
    """Fail loudly if a cached npz predates the current objectives code.

    Both objectives are recomputed under the split's own LOGGED actions and
    compared against the stored `reward` array. They must agree exactly, because
    that is precisely how stage 3b produced them.

    This guard exists because the failure it catches is silent and expensive.
    Editing `detection_objective` or `burden_objective` without re-running stage
    3b leaves the training-derived objective ranges in joint_meta.json describing
    a DIFFERENT reward function than the one `scalar_policy_reward` recomputes at
    evaluation time.
    """
    stored = np.asarray(_col(split, "reward"), dtype=np.float64)
    acts = _to_numpy(_col(split, "action"))
    fresh = np.stack([detection_objective(split, acts)[0],
                      burden_objective(split, acts)], axis=1).astype(np.float64)
    if stored.shape != fresh.shape:
        raise SystemExit(
            f"\n{name}: stored reward has shape {stored.shape} but the current "
            f"objectives produce {fresh.shape}.\nRe-run stage 3b:\n"
            f"    python src/s3b_build_joint_mdp.py\n")

    worst = np.abs(stored - fresh).max(axis=0)
    if (worst > tol).any():
        names = cfg.JOINT_REWARD_DIMS
        detail = "  ".join(f"{n}: max|diff|={w:.4g}" for n, w in zip(names, worst))
        raise SystemExit(
            f"\n{name}: the cached reward does NOT match the current objectives.\n"
            f"  {detail}\n\n"
            f"data/rl/joint_*.npz was built by an older objectives.py, so the\n"
            f"normalization in joint_meta.json describes a different reward.\n"
            f"Every downstream number is measured against the wrong zero point.\n\n"
            f"Re-run stage 3b:\n    python src/s3b_build_joint_mdp.py\n")

    if norm_meta is not None:
        mu = np.asarray(norm_meta["reward_mean"], dtype=np.float64)
        sd = np.asarray(norm_meta["reward_sd"], dtype=np.float64)
        sd = np.where(sd < 1e-6, 1.0, sd)
        if _has_col(split, "reward_norm"):
            got = np.asarray(_col(split, "reward_norm"), dtype=np.float64)
            want = (stored - mu) / sd
            if np.abs(got - want).max() > tol:
                raise SystemExit(
                    f"\n{name}: reward_norm does not match "
                    f"reward divided by the stored objective range.\n"
                    f"Re-run stage 3b:\n    python src/s3b_build_joint_mdp.py\n")
    return True


def clinician_reward_sanity(split, norm_meta, name="split", tol=2.0):
    """Return the clinician's mean scaled per-stay objective values."""
    stay = _to_numpy(_col(split, "stay_id"))
    stored = np.asarray(_col(split, "reward"), dtype=np.float64)
    mu = np.asarray(norm_meta["reward_mean"], dtype=np.float64)
    sd = np.asarray(norm_meta["reward_sd"], dtype=np.float64)
    sd = np.where(sd < 1e-6, 1.0, sd)
    z = (stored - mu) / sd

    out = []
    for d in range(z.shape[1]):
        s = pd.Series(z[:, d]).groupby(stay).sum()
        out.append(float(s.mean()))
    return out


def _mean_stay_returns(values, stay_ids):
    values = np.asarray(values, dtype=np.float64)
    stay_ids = np.asarray(stay_ids)
    _, inverse = np.unique(stay_ids, return_inverse=True)
    totals = np.zeros((inverse.max() + 1, values.shape[1]), dtype=np.float64)
    np.add.at(totals, inverse, values)
    return totals.mean(axis=0)


def normalize_rewards(train_split, *splits):
    """Scale each objective by its training-set per-stay achievable range.

    The anchors are the two extreme constant policies: never draw and always
    draw. This makes a simplex weight describe a fraction of each objective's
    achievable per-stay span instead of mixing objectives with very different
    accumulation rates. Zero remains neutral and no validation/test outcomes
    are used to fit the scale.
    """
    train_reward = np.asarray(_col(train_split, "reward"), dtype=np.float32)
    stay = _to_numpy(_col(train_split, "stay_id"))
    n = len(stay)
    never = np.zeros(n, dtype=np.int64)
    always = np.ones(n, dtype=np.int64)

    never_reward = np.stack([
        detection_objective(train_split, never)[0],
        burden_objective(train_split, never),
    ], axis=1)
    always_reward = np.stack([
        detection_objective(train_split, always)[0],
        burden_objective(train_split, always),
    ], axis=1)
    never_return = _mean_stay_returns(never_reward, stay)
    always_return = _mean_stay_returns(always_reward, stay)
    objective_range = np.abs(always_return - never_return).astype(np.float32)
    objective_range[objective_range < 1e-6] = 1.0

    out = [(np.asarray(x, dtype=np.float32) / objective_range).astype(np.float32)
           for x in splits]
    raw_mu = train_reward.mean(axis=0).astype(np.float32)
    return out, {
        "reward_mean": np.zeros_like(raw_mu).tolist(),
        # Kept for compatibility with downstream readers; this is now a range,
        # not a standard deviation.
        "reward_sd": objective_range.tolist(),
        "reward_scale": objective_range.tolist(),
        "raw_reward_mean": raw_mu.tolist(),
        "never_draw_return": never_return.tolist(),
        "always_draw_return": always_return.tolist(),
        "normalization": "per_stay_extreme_range",
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
