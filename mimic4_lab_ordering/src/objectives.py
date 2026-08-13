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


UTILITY_DEFINITION = "paper_eq3_eq4_clinical_trigger_capped"
EPISODE_START_DEFINITION = "after_all_target_lab_baselines_available"


def post_baseline_mask(df):
    """Rows where every target lab has a value strictly before decision time."""
    available = [
        np.isfinite(_to_numpy(_col(df, f"last_{lab}")).astype(np.float64))
        for lab in panels.LABS
    ]
    return np.logical_and.reduce(available)


def fit_utility_thresholds(train_df):
    """Fit Cheng Eq. 5 cutoffs from decision-time TRAIN state only."""
    if _has_col(train_df, "action"):
        draw = _to_numpy(_col(train_df, "action")) != 0
    else:
        draw = panels.encode_frame(train_df) != 0
    thresholds = {}
    for lab in panels.LABS:
        mean = _to_numpy(_col(train_df, f"mean_{lab}")).astype(np.float64)
        last = _to_numpy(_col(train_df, f"last_{lab}")).astype(np.float64)
        last = np.where(np.isfinite(last), last, mean)
        std = np.maximum(
            _to_numpy(_col(train_df, f"std_{lab}")).astype(np.float64),
            cfg.FORECAST_MIN_STD,
        )
        score = np.abs(mean - last) / std
        eligible = draw & np.isfinite(score)
        thresholds[lab] = float(np.median(score[eligible])) if eligible.any() else 0.0
    return thresholds


def information_potential(df, thresholds):
    """Decision-time information value available for either action.

    This is Cheng Eq. 5's forecast-change proxy, but it uses only values already
    present in the state: predictive mean, last observed value, and uncertainty.
    Summing over the four assays reflects the utility of the joint blood panel.
    """
    values = []
    for lab in panels.LABS:
        mean = _to_numpy(_col(df, f"mean_{lab}")).astype(np.float64)
        last = _to_numpy(_col(df, f"last_{lab}")).astype(np.float64)
        last = np.where(np.isfinite(last), last, mean)
        std = np.maximum(
            _to_numpy(_col(df, f"std_{lab}")).astype(np.float64),
            cfg.FORECAST_MIN_STD,
        )
        score = np.abs(mean - last) / std
        score = np.where(np.isfinite(score), score, 0.0)
        values.append(np.maximum(0.0, score - float(thresholds[lab])))
    return np.sum(np.stack(values, axis=1), axis=1).astype(np.float32)


def _shift_next_by_stay(values, stay_ids, bins=1):
    """values[t + bins] within a stay; 0 past the end. Used for Eq. 4."""
    v = np.asarray(values, dtype=np.float64)
    out = np.zeros(len(v), dtype=np.float64)
    stay_ids = np.asarray(stay_ids)
    start = 0
    for i in range(1, len(v) + 1):
        if i == len(v) or stay_ids[i] != stay_ids[start]:
            n = i - start
            if n > bins:
                out[start:i - bins] = v[start + bins:i]
            start = i
    return out


def clinical_trigger(df, lookahead_bins=cfg.TREAT_LOOKAHEAD_BINS):
    """The paper's Eq. 3 OR Eq. 4, capped at one credit per hour.

        Eq. 3  a SOFA rise of >= 2 between t-1 and t
        Eq. 4  an intervention initiated at t+1

    Why this and not Eq. 5's information term: measured on val, both of these
    trigger a clinician draw at roughly TWICE the base rate (2.07x for the SOFA
    jump, 2.19x for the impending intervention, 2.14x for the capped union),
    while every computable form of Eq. 5 sits at chance -- 0.48 to 0.55 AUC,
    lactate BELOW chance. A low AUC on these two is expected and not evidence
    against them: they fire on under 2% of hours, and a 2x lift on a 2% base
    rate caps AUC near 0.51 by arithmetic alone. Eq. 5 is dense rather than
    rare, so its 0.545 is genuine weakness, and its density is also what made
    drawing profitable at every hour under a small burden weight.

    CAPPED at one, per the double-reward concern: the two triggers co-fire on
    216 val rows (0.107% of hours, 2.98% of trigger rows). Summing them would
    pay twice for one clinical episode.

    Timing: `sofa_delta` is built from the forecaster's past-only predictive
    means plus concurrent ventilation/vasopressor/GCS status -- information a
    clinician deciding at hour t already has, and never anything from the
    future. Eq. 4 deliberately looks one bin ahead: that is the reward's job,
    and it is what forces the policy to ANTICIPATE rather than react. Neither
    trigger is readable off the state (AUC 0.808 and 0.746 from a fitted model,
    not ~1.0), so both remain genuine prediction problems.
    """
    if _has_col(df, "clinical_trigger"):
        return _to_numpy(_col(df, "clinical_trigger")).astype(np.float32)

    sofa_jump = (_to_numpy(_col(df, "sofa_delta")).astype(np.float64)
                 >= cfg.SOFA_DELTA_THRESHOLD)

    onset_cols = [f"onset_{k}" for k in ids.INTERVENTION_KINDS]
    onset_now = np.column_stack(
        [_to_numpy(_col(df, c)) for c in onset_cols]).sum(axis=1) > 0
    onset_next = _shift_next_by_stay(
        onset_now.astype(np.float64), _to_numpy(_col(df, "stay_id")),
        bins=int(lookahead_bins)) > 0

    return (sofa_jump | onset_next).astype(np.float32)


def utility_objective(df, actions, thresholds=None):
    """Eq. 3 + Eq. 4, gated on drawing and capped at one credit per hour.

    Utility is zero unless a draw is taken, exactly as in the paper: all three
    of its positive terms carry the 1[a != 0] gate. Not drawing therefore scores
    zero rather than negative, and the burden objective is the sole counterweight.

    Because the trigger fires on only ~3.6% of hours, utility is SPARSE. That is
    the property that makes the policy selective: with the old dense information
    term every hour carried positive utility, so drawing was profitable
    everywhere and a small burden weight produced draw-every-hour. Here the
    policy only profits where it predicts a trigger, and the preference weight
    sets the threshold on that predicted probability -- so the draw rate still
    varies smoothly across the frontier rather than pinning at 3.6%.
    """
    utility = clinical_trigger(df)
    draw = (np.asarray(actions) != 0).astype(np.float32)
    return utility * draw


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


def burden_potential(df):
    """Cost of drawing now from the current decision-time state.

    The first term is the physical draw. The second penalizes redundancy when
    any target lab was observed recently. Unlike a policy-history replay clock,
    this is the same Markov reward used in both training and evaluation.
    """
    if _has_col(df, "draw_burden"):
        return _to_numpy(_col(df, "draw_burden")).astype(np.float32)
    delta = np.column_stack([
        _to_numpy(_col(df, f"delta_{lab}")).astype(np.float64)
        for lab in panels.LABS
    ])
    seen = np.column_stack([
        _to_numpy(_col(df, f"last_{lab}")).astype(np.float64)
        for lab in panels.LABS
    ])
    seen = np.isfinite(seen)
    delta = np.where(np.isfinite(delta), delta, np.inf)
    delta = np.where(seen, delta, np.inf)
    since_any = np.maximum(np.min(delta, axis=1), 0.0)
    return (1.0 + np.exp(-since_any / cfg.COST_DECAY_GAMMA)).astype(np.float32)


def burden_objective(df, actions):
    draw = (np.asarray(actions) != 0).astype(np.float32)
    if (not _has_col(df, "draw_burden")
            and not _has_col(df, "delta_creatinine")):
        # Legacy standalone tests and the old per-lab path provide only a
        # stay/hour frame. Keep their replay semantics isolated; joint MDP
        # artifacts always use the state-computable draw_burden column.
        stay = _to_numpy(_col(df, "stay_id"))
        hour = _to_numpy(_col(df, "hour")).astype(np.float32)
        out = np.zeros(len(stay), dtype=np.float32)
        start = 0
        for i in range(1, len(stay) + 1):
            if i == len(stay) or stay[i] != stay[start]:
                last_seen = np.nan
                for j in range(start, i):
                    if not draw[j]:
                        continue
                    delta = hour[j] - last_seen if np.isfinite(last_seen) else np.inf
                    out[j] = 1.0 + float(np.exp(-delta / cfg.COST_DECAY_GAMMA))
                    last_seen = hour[j]
                start = i
        return out
    return burden_potential(df) * draw


def assert_rewards_current(split, norm_meta=None, name="split", tol=1e-3):
    """Fail loudly if a cached npz predates the current objectives code.

    Both objectives are recomputed under the split's own LOGGED actions and
    compared against the stored `reward` array. They must agree exactly, because
    that is precisely how stage 3b produced them.

    This guard exists because the failure it catches is silent and expensive.
    Editing `utility_objective` or `burden_objective` without re-running stage
    3b leaves the training-derived objective ranges in joint_meta.json describing
    a DIFFERENT reward function than the one `scalar_policy_reward` recomputes at
    evaluation time.
    """
    stored = np.asarray(_col(split, "reward"), dtype=np.float64)
    acts = _to_numpy(_col(split, "action"))
    fresh = np.stack([utility_objective(split, acts),
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
    """Scale objectives by logged TRAIN mean return per stay.

    Constant-action trajectories are not valid normalization anchors here:
    changing a draw changes later lab-history and forecast state. The logged
    clinician trajectory is factual, train-only, and keeps zero reward neutral.
    """
    train_reward = np.asarray(_col(train_split, "reward"), dtype=np.float32)
    stay = _to_numpy(_col(train_split, "stay_id"))
    logged_return = _mean_stay_returns(train_reward, stay).astype(np.float32)
    objective_scale = np.abs(logged_return)
    objective_scale[objective_scale < 1e-6] = 1.0

    out = [(np.asarray(x, dtype=np.float32) / objective_scale).astype(np.float32)
           for x in splits]
    raw_mu = train_reward.mean(axis=0).astype(np.float32)
    return out, {
        "reward_mean": np.zeros_like(raw_mu).tolist(),
        # Kept for compatibility with downstream readers; this is a factual
        # mean-return scale, not a standard deviation.
        "reward_sd": objective_scale.tolist(),
        "reward_scale": objective_scale.tolist(),
        "raw_reward_mean": raw_mu.tolist(),
        "logged_clinician_mean_return": logged_return.tolist(),
        "normalization": "train_logged_mean_return",
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
