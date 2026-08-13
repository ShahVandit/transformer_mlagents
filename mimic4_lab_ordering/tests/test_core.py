"""
Unit tests for the pieces that are easy to get subtly wrong.

    python tests/test_core.py

These test properties, not golden numbers, so they stay meaningful when the
cohort or the hyperparameters change. Three of them guard leakage boundaries:
the forecaster must not see the future, the split must not share a patient, and
the state must not contain the result of the order being decided.
"""
import json
import pickle
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config as cfg          # noqa: E402
import direct_policy as direct  # noqa: E402
import forecast as fc         # noqa: E402
import mofqi                  # noqa: E402
import objectives             # noqa: E402
import s3_build_mdp as mdp    # noqa: E402
import s5_evaluate_ope as ope  # noqa: E402
import s6_clinical_metrics as clin  # noqa: E402
import s2_hourly_grid as grid  # noqa: E402
import target_lab_forecaster as target_fc  # noqa: E402
import joint_mdp_audit  # noqa: E402
import sofa as sofa_mod       # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if detail and not cond else ""))


# --------------------------------------------------------------- rewards ----
def test_rewards():
    print("\nrewards (Eqs. 3-6)")
    a1 = np.array([1, 1, 0, 0])
    sofa_d = np.array([3.0, 1.0, 3.0, 0.0])
    # Eq. 3: fires only when a lab is ordered AND SOFA rose by >= 2
    r = mdp.reward_sofa(a1, sofa_d)
    check("r_sofa fires on order + SOFA rise >= 2", list(r) == [True, False, False, False])

    # Eq. 4: counts interventions initiated at t+1, gated on ordering
    r = mdp.reward_treat(np.array([1, 0]), np.array([2.0, 2.0]))
    check("r_treat is gated on a != 0", list(r) == [2.0, 0.0])

    # Eq. 5: max(0, |m-y|/sigma - c) * 1[a=1]. |4-1|/1 = 3, minus c=0.5 -> 2.5
    r = mdp.reward_info(np.array([1, 1, 0]), np.array([4.0, 1.2, 4.0]),
                        np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]), 0.5)
    check("r_info = max(0, g - c_l) when ordered", abs(r[0] - 2.5) < 1e-9)
    check("r_info is 0 below the c_l threshold", r[1] == 0.0)
    check("r_info is 0 when not ordered", r[2] == 0.0)

    # Eq. 6: exp(-Delta/Gamma), Gamma = 6. Delta=0 -> 1, Delta=6 -> exp(-1)
    r = mdp.reward_cost(np.array([1, 1, 0]), np.array([0.0, 6.0, 0.0]))
    check("r_cost is 1 at Delta=0", abs(r[0] - 1.0) < 1e-9)
    check("r_cost decays as exp(-Delta/6)", abs(r[1] - np.exp(-1.0)) < 1e-9)
    check("r_cost is 0 when not ordered", r[2] == 0.0)
    check("r_cost is 0 for a never-before-drawn lab",
          mdp.reward_cost(np.array([1]), np.array([np.inf]))[0] == 0.0)

    # The gating property the whole report hinges on.
    for fn, args in [(mdp.reward_sofa, (np.array([0]), np.array([10.0]))),
                     (mdp.reward_treat, (np.array([0]), np.array([4.0]))),
                     (mdp.reward_cost, (np.array([0]), np.array([0.0])))]:
        check(f"not ordering gives 0 from {fn.__name__}", float(fn(*args)[0]) == 0.0)


# ------------------------------------------------------- Pareto / MO-FQI ----
def test_pareto():
    print("\nPareto pruning (Sec. 2.3)")
    # action 1 beats action 0 on every objective -> 0 is dominated
    Q = np.array([[[0.0, 0.0], [1.0, 1.0]]])
    keep = mofqi.pareto_mask(Q)
    check("a dominated action is dropped", list(keep[0]) == [False, True])

    # a trade-off: neither dominates
    Q = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    keep = mofqi.pareto_mask(Q)
    check("mutually non-dominated actions both survive", list(keep[0]) == [True, True])

    # domination must be strict on EVERY objective
    Q = np.array([[[0.0, 1.0], [1.0, 1.0]]])
    check("a tie on one objective is not domination",
          list(mofqi.pareto_mask(Q)[0]) == [True, True])

    # three actions, one dominated by one of the others
    Q = np.array([[[0.0, 0.0], [1.0, 2.0], [2.0, 1.0]]])
    check("pruning generalizes past two actions",
          list(mofqi.pareto_mask(Q)[0]) == [False, True, True])

    # with |A| = 2 the pruned max equals the plain elementwise max
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(200, 2, 4))
    k = mofqi.pareto_mask(Q)
    check("binary-action pruning is a no-op in the backup",
          np.allclose(mofqi.pruned_max(Q, k), Q.max(axis=1)))

    class FixedQ:
        n_dims = 2

        @staticmethod
        def q_all_actions(states):
            return np.tile(
                np.array([[[0.8, 0.1], [0.6, 0.9]]], dtype=np.float32),
                (len(states), 1, 1))

    utility_policy = mofqi.WeightedQPolicy(FixedQ(), (0.9, 0.1))
    burden_policy = mofqi.WeightedQPolicy(FixedQ(), (0.1, 0.9))
    dummy = np.zeros((3, 1), dtype=np.float32)
    check("MO-FQI preference extraction selects different trade-offs",
          np.all(utility_policy.predict(dummy) == 0)
          and np.all(burden_policy.predict(dummy) == 1))


def test_budget():
    print("\nbudget rule (Sec. 3)")
    stay = np.zeros(50, dtype=int)
    hours = np.arange(50)
    a = np.zeros(50, dtype=int)
    out = mofqi.apply_budget(a, stay, hours, budget_hours=24)
    check("a silent 48h stay gets exactly 2 forced orders", int(out.sum()) == 2)
    check("forced orders land at the end of each window",
          list(np.flatnonzero(out)) == [23, 47])

    a = np.zeros(50, dtype=int)
    a[10] = 1
    out = mofqi.apply_budget(a, stay, hours, budget_hours=24)
    check("a real order resets the budget clock",
          list(np.flatnonzero(out)) == [10, 34])

    stay2 = np.array([0] * 25 + [1] * 25)
    out = mofqi.apply_budget(np.zeros(50, dtype=int), stay2, hours, budget_hours=24)
    check("the budget clock does not cross stays", int(out.sum()) == 2)


def test_joint_mdp_audit_rejects_bad_transition():
    stay = np.array([1, 1, 2, 2], dtype=np.int64)
    subject = np.array([10, 10, 20, 20], dtype=np.int64)
    action = np.array([0, 1, 0, 1], dtype=np.int64)
    potential = np.array([0.0, 2.0, 0.5, 1.0], dtype=np.float32)
    draw_burden = np.array([1.0, 1.5, 1.0, 1.5], dtype=np.float32)
    base = {
        "stay_id": stay,
        "subject_id": subject,
        "hour": np.array([0, 1, 0, 1], dtype=np.int64),
        "action": action,
        "information_potential": potential,
        "draw_burden": draw_burden,
    }
    utility = objectives.utility_objective(base, action)
    burden = objectives.burden_objective(base, action)
    reward = np.stack([utility, burden], axis=1).astype(np.float32)
    split = {
        **base,
        "state": np.arange(8, dtype=np.float32).reshape(4, 2),
        "reward": reward,
        "done": np.array([0, 1, 0, 1], dtype=np.float32),
        "event": np.zeros(4, dtype=np.int8),
        "future_event": np.zeros(4, dtype=np.int8),
        "n_labs": np.array([0, 1, 0, 1], dtype=np.int64),
    }
    split["next_state"] = split["state"][[1, 1, 3, 3]]
    normed, norm = objectives.normalize_rewards(split, reward)
    split["reward_norm"] = normed[0]
    meta = {
        "state_dim": 2,
        "reward_dims": cfg.JOINT_REWARD_DIMS,
        "panel_bits": cfg.JOINT_PANEL_BITS,
        "reward_normalization": norm,
    }
    joint_mdp_audit.audit_split("synthetic", split, meta)
    broken = dict(split)
    broken["next_state"] = split["next_state"].copy()
    broken["next_state"][0] = -1
    try:
        joint_mdp_audit.audit_split("broken", broken, meta)
    except ValueError as exc:
        check("joint MDP audit rejects bad next-state linkage",
              "next-state linkage" in str(exc))
    else:
        check("joint MDP audit rejects bad next-state linkage", False)


# ------------------------------------------------------------- forecaster ----
def test_forecaster_no_leakage():
    print("\nforecaster leakage guard (Sec. 2.1)")
    rng = np.random.default_rng(0)
    obs = np.full((4, 40, 1), np.nan)
    for i in range(4):
        idx = rng.choice(40, 12, replace=False)
        obs[i, idx, 0] = rng.normal(size=12)

    f = fc.LocalTrendForecaster(["x"], fit_max_stays=4).fit(obs)
    m_full, s_full = f.filter(obs)

    truncated = obs.copy()
    truncated[:, 20:, :] = np.nan
    m_trunc, s_trunc = f.filter(truncated)

    check("filter() at hour t ignores observations after t",
          np.allclose(m_full[:, :20], m_trunc[:, :20], atol=1e-6) and
          np.allclose(s_full[:, :20], s_trunc[:, :20], atol=1e-6))

    ms_full, _ = f.smooth(obs)
    ms_trunc, _ = f.smooth(truncated)
    check("smooth() DOES use later observations (it is evaluation-only)",
          not np.allclose(ms_full[:, :20], ms_trunc[:, :20], atol=1e-6))

    # uncertainty must grow while nothing is measured
    one = np.full((1, 10, 1), np.nan)
    one[0, 0, 0] = 0.0
    _, s = f.filter(one)
    check("sigma_t grows with time since the last measurement",
          bool(np.all(np.diff(s[0, 1:, 0]) > -1e-9) and s[0, 9, 0] > s[0, 1, 0]))

    # a trend model must extrapolate, or Eq. 5 is identically zero
    ramp = np.full((1, 12, 1), np.nan)
    ramp[0, :6, 0] = np.arange(6, dtype=float)
    m, _ = f.filter(ramp)
    check("the forecast extrapolates a trend (keeps r_info alive)",
          float(m[0, 8, 0]) > float(m[0, 6, 0]))

    # Sparse-trait regression. The RTS covariance recursion diverges across long
    # unobserved stretches (measured: inf for bilirubin at 3% hourly coverage),
    # which overflowed the float32 cast. The smoothed MEAN is unaffected because
    # its recursion never reads that covariance, and the mean is all this
    # project consumes -- so the covariance is not computed by default.
    sparse = np.full((2, 400, 1), np.nan)
    sparse[0, [5, 180, 390], 0] = [1.0, 2.0, 1.5]   # 3 obs in 400 hours
    sparse[1, ::40, 0] = rng.normal(size=10)
    lengths = np.array([400, 400])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        m_s, s_s = f.smooth(sparse, lengths)
    check("smooth() on a sparse trait raises no overflow warning", True)
    check("smoothed mean is finite on a sparse trait", bool(np.isfinite(m_s).all()))
    check("smoothed std is not computed unless asked", s_s is None)

    _, s_req = f.smooth(sparse, lengths, return_std=True)
    check("an explicitly requested smoothed std stays in float32 range",
          bool(np.isfinite(s_req).all()
               and float(np.max(s_req)) <= np.finfo(np.float32).max))

    # The smoothed mean feeds the information-gain metric, so an excursion far
    # outside the training range would read as spurious information.
    check("smoothed mean is bounded by the training range",
          bool((m_s >= f.lo_[0] - 1e-6).all() and (m_s <= f.hi_[0] + 1e-6).all()))

    # Batch composition must not change a stay's own result.
    alone = np.full((1, 400, 1), np.nan)
    alone[0, [5, 180, 390], 0] = [1.0, 2.0, 1.5]
    m_alone, _ = f.smooth(alone, np.array([400]))
    check("a stay's smoothed mean is independent of its batch",
          np.allclose(m_s[0], m_alone[0], atol=1e-4))

    check("filtered output is finite on a sparse trait",
          bool(np.isfinite(f.filter(sparse, lengths)[0]).all()))


def test_target_forecaster_selection_and_calibration():
    actual = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    last = np.zeros(5)
    predicted_delta = actual * 1.5
    rmse, shrinkage = target_fc._select_shrinkage(
        actual, last, predicted_delta, candidates=[0.0, 0.5, 1.0])
    locf_rmse = float(np.sqrt(np.mean((actual - last) ** 2)))
    assert rmse <= locf_rmse
    assert shrinkage == 0.5

    rng = np.random.default_rng(7)
    final_mean = rng.normal(size=20_000)
    raw_std = rng.uniform(0.5, 2.0, size=20_000)
    actual = final_mean + raw_std * rng.normal(size=20_000)
    scale = target_fc._calibrate_interval(actual, final_mean, raw_std)
    coverage = np.mean(
        np.abs(actual - final_mean) <= 1.6448536269514722 * raw_std * scale)
    assert abs(coverage - 0.90) < 0.01


# ------------------------------------------------------------------ state ----
def test_state_and_delta():
    print("\nstate construction (Sec. 2.2)")
    check("joint state adds four explicit lab-history flags to the 21 paper features",
          len(mdp.state_columns(include_poe=False)) == 25,
          str(len(mdp.state_columns(include_poe=False))))
    check("POE extension adds exactly four past-only workflow features",
          len(mdp.state_columns(include_poe=True)) == 29,
          str(len(mdp.state_columns(include_poe=True))))

    obs = np.full((1, 6, 1), np.nan)
    obs[0, 2, 0] = 5.0
    last, delta = grid.last_and_delta(obs)
    check("y_t excludes the measurement taken in hour t itself",
          np.isnan(last[0, 2, 0]))
    check("y_t is available from the next hour on", last[0, 3, 0] == 5.0)
    check("Delta_t is 1 an hour after the draw", delta[0, 3, 0] == 1.0)
    check("Delta_t increments hourly", delta[0, 5, 0] == 3.0)
    check("Delta_t is undefined before the first draw", np.isnan(delta[0, 1, 0]))


def test_poe_features_are_past_only():
    print("\nPOE workflow features")
    stays = np.array([7])
    lengths = pd.Series([5], index=stays)
    intimes = {7: pd.Timestamp("2020-01-01 00:00:00")}
    orders = pd.DataFrame({
        "stay_id": [7, 7, 7],
        "ordertime": [pd.Timestamp("2020-01-01 01:10:00"),
                      pd.Timestamp("2020-01-01 01:10:00"),
                      pd.Timestamp("2020-01-01 03:00:00")],
    })
    f = grid.poe_feature_grids(orders, stays, lengths, intimes, 5)
    check("simultaneous POE rows form one order group",
          int(f["poe_order_groups_current_hour"][0, 1]) == 1
          and int(f["poe_order_rows_current_hour"][0, 1]) == 2)
    check("current-hour POE is absent from current policy state",
          int(f["poe_order_groups_6h"][0, 1]) == 0)
    check("POE enters state in the following hour",
          int(f["poe_order_groups_6h"][0, 2]) == 1
          and int(f["poe_order_rows_6h"][0, 2]) == 2)
    check("time since POE is also strictly lagged",
          float(f["poe_hours_since_lab_order"][0, 1]) == 2.0
          and float(f["poe_hours_since_lab_order"][0, 2]) == 1.0)


# ------------------------------------------------------------------- SOFA ----
def test_batch_slicing():
    print("\nstay batching (stage 2)")
    import pandas as pd
    ev = pd.DataFrame({
        "stay_id": np.array([1, 1, 2, 3, 3, 3, 5], dtype=np.int32),
        "hour": np.arange(7),
        "trait": ["hr"] * 7,
        "value": np.arange(7, dtype=float),
    }).sort_values("stay_id", kind="stable")

    got = grid._slice_by_stay(ev, np.array([1, 3], dtype=np.int32))
    check("slicing picks exactly the requested stays",
          sorted(got["stay_id"].tolist()) == [1, 1, 3, 3, 3])
    check("slicing keeps every row of a selected stay", len(got) == 5)

    got = grid._slice_by_stay(ev, np.array([4], dtype=np.int32))
    check("a stay with no events yields no rows", len(got) == 0)

    got = grid._slice_by_stay(ev, np.array([1, 2, 3, 5], dtype=np.int32))
    check("slicing every stay returns every row", len(got) == len(ev))

    # equivalence with the obvious-but-heavy isin implementation
    rng = np.random.default_rng(0)
    big = pd.DataFrame({"stay_id": np.sort(rng.integers(0, 50, 500).astype(np.int32)),
                        "value": rng.normal(size=500)})
    ids_ = np.array([3, 7, 11, 42], dtype=np.int32)
    a = grid._slice_by_stay(big, ids_)["value"].to_numpy()
    b = big[big["stay_id"].isin(ids_)]["value"].to_numpy()
    check("slicing matches an isin filter", np.array_equal(np.sort(a), np.sort(b)))


def test_joint_panels():
    print("\njoint draw encoding (Pareto track)")
    import panels
    import objectives
    import pandas as pd

    all_combinations = np.array([
        [int(c) for c in format(i, "04b")] for i in range(16)
    ], dtype=np.int8)
    encoded = panels.encode_bits(all_combinations)
    check("the empty combination maps to `none`",
          int(encoded[0]) == 0)
    check("every non-empty lab combination maps to one blood draw",
          bool((encoded[1:] == 1).all()))

    # burden: zero for no draw, larger for a repeat than for a stale draw.
    # Deltas are derived from stay_id/hour, not read from delta_* columns.
    n = 30
    d1 = {"stay_id": np.zeros(n, dtype=int), "hour": np.arange(n, dtype=float)}
    a1 = np.zeros(n, dtype=int)
    a1[0] = 1          # first blood draw
    a1[1] = 1          # repeated one hour later
    a1[25] = 1         # repeated a day later
    bb = objectives.burden_objective(d1, a1)
    check("burden is 0 for the empty panel", float(bb[5]) == 0.0)
    check("burden penalises a repeat draw more than a stale one", bb[1] > bb[25])
    b = np.array([bb[1], bb[25]])
    check("burden is at least 1 whenever blood is drawn", bool((b >= 1.0).all()))

    # A first-ever draw carries no redundancy at any hour. Imputing `hour + 1`
    # for a never-drawn lab charged a 3-lab panel at hour 0 as 3.54 instead of
    # 1.00, which falls hardest on admission labs.
    for h0 in (0, 2, 6, 24):
        n = h0 + 2
        d2 = {"stay_id": np.zeros(n, dtype=int), "hour": np.arange(n, dtype=float)}
        a2 = np.zeros(n, dtype=int)
        a2[h0] = 1                                   # first blood draw
        got = float(objectives.burden_objective(d2, a2)[h0])
        if abs(got - 1.0) > 1e-6:
            check(f"a first-ever draw at hour {h0} costs exactly the base 1.0",
                  False, f"got {got:.4f}")
            break
    else:
        check("a first-ever draw costs exactly the base 1.0 at any hour", True)

    # redundancy decays with elapsed time between repeats
    n = 12
    d3 = {"stay_id": np.zeros(n, dtype=int), "hour": np.arange(n, dtype=float)}
    a3 = np.zeros(n, dtype=int)
    a3[5] = 1
    a3[6] = 1
    a3[10] = 1
    b3 = objectives.burden_objective(d3, a3)
    check("redundancy decays as the gap between repeats grows",
          b3[5] < b3[10] < b3[6])

    # the redundancy clock does not leak across stays
    d4 = {"stay_id": np.array([0, 0, 1, 1]), "hour": np.array([0.0, 1.0, 0.0, 1.0])}
    a4 = np.array([1, 0, 1, 0])
    b4 = objectives.burden_objective(d4, a4)
    check("a new stay starts with a clean redundancy clock",
          abs(float(b4[2]) - 1.0) < 1e-6)

    # Joint utility is available before either action and follows the paper's
    # thresholded forecast-change proxy.
    n = 5
    d = {"stay_id": np.zeros(n, dtype=int),
         "hour": np.arange(n, dtype=float)}
    for lab in __import__("itemids").TARGET_LABS:
        d[f"mean_{lab}"] = np.array([0, 1, 4, 1, 0], dtype=float)
        d[f"last_{lab}"] = np.zeros(n)
        d[f"std_{lab}"] = np.ones(n)
        d[f"obs_{lab}"] = np.array([np.nan, 2, np.nan, 2, np.nan])
        d[f"delta_{lab}"] = np.array([np.nan, 1, 1, 2, 3], dtype=float)
    d["action"] = np.array([0, 1, 0, 1, 0], dtype=int)
    thresholds = objectives.fit_utility_thresholds(d)
    potential = objectives.information_potential(d, thresholds)
    check("training thresholds are medians of clinician-ordered scores",
          all(v == 1.0 for v in thresholds.values()))
    check("threshold-level draws have no information utility", potential[1] == 0.0)
    check("larger forecast changes have more information utility", potential[2] > 0.0)

    d["information_potential"] = potential
    d["draw_burden"] = objectives.burden_potential(d)
    none = np.zeros(n, dtype=int)
    informative = none.copy()
    informative[2] = 1
    uninformative = none.copy()
    uninformative[1] = 1
    u_none = objectives.utility_objective(d, none)
    u_info = objectives.utility_objective(d, informative)
    u_low = objectives.utility_objective(d, uninformative)
    b_low = objectives.burden_objective(d, uninformative)
    check("missing informative draws receives negative utility",
          float(u_none.sum()) < 0.0)
    check("an informative draw receives positive utility", float(u_info.sum()) > 0.0)
    check("drawing at zero potential does not alter utility",
          np.allclose(u_low, u_none))
    check("an uninformative draw still incurs burden", float(b_low.sum()) > 0.0)

    train_r = np.stack([u_info, objectives.burden_objective(d, informative)], axis=1)
    train_split = {**d, "action": informative, "reward": train_r}
    normed, norm_meta = objectives.normalize_rewards(train_split, train_r)
    check("normalization keeps neutral rewards at exactly zero",
          bool(np.array_equal(normed[0][0], np.zeros(2, dtype=np.float32))))
    check("normalization records per-stay range semantics",
          norm_meta.get("normalization") == "per_stay_extreme_range")


def test_direct_policy():
    print("\ndirect constrained policy")
    state_cols = mdp.state_columns()
    n = 6
    state = np.zeros((n, len(state_cols)), dtype=np.float32)
    thresholds = {lab: 0.5 for lab in __import__("itemids").TARGET_LABS}
    for lab in __import__("itemids").TARGET_LABS:
        state[:, state_cols.index(f"mean_{lab}")] = np.arange(n)
        state[:, state_cols.index(f"last_{lab}")] = 0.0
        state[:, state_cols.index(f"std_{lab}")] = 1.0
        state[:, state_cols.index(f"delta_{lab}")] = np.arange(n) + 1.0

    info = direct.information_utility(state, state_cols, thresholds)
    burden = direct.draw_burden(state, state_cols)
    check("direct information is non-negative", bool((info >= 0).all()))
    check("direct burden includes the base draw cost", bool((burden >= 1).all()))
    scales = direct.fit_utility_scales(
        state, np.array([0, 0, 0, 1, 1, 1]), state_cols, thresholds)
    g_info = direct.draw_utility(
        state, state_cols, thresholds, scales, (0.9, 0.1))
    g_burden = direct.draw_utility(
        state, state_cols, thresholds, scales, (0.1, 0.9))
    check("burden-dominant utility can make a draw negative",
          bool((g_burden < 0).any()))
    check("information weight raises high-surprise draw value",
          float(g_info[-1]) > float(g_info[0]))

    policy_p = np.array([0.0, 0.25, 0.75, 1.0], dtype=np.float32)
    behavior_p = np.array([0.01, 0.3, 0.7, 0.99], dtype=np.float32)
    overlap = direct.expected_overlap(policy_p, behavior_p)
    unsupported = direct.unsupported_action_mass(policy_p, behavior_p, 0.05)
    check("expected propensity stays within probability bounds",
          bool(((overlap >= 0) & (overlap <= 1)).all()))
    check("unsupported-action mass is non-negative",
          bool((unsupported >= 0).all()))
    check("supported deterministic actions have zero violation",
          float(unsupported[[0, 3]].sum()) == 0.0)

    # Without a support penalty, exact expected utility is linear in p, so its
    # optimizer is the deterministic sign threshold on g(s, 1).
    candidate = np.linspace(0.0, 1.0, 101)
    chosen = np.array([candidate[np.argmax(candidate * g)] for g in g_info])
    target = (g_info > 0).astype(float)
    check("unconstrained direct utility is optimized by the sign threshold",
          bool(np.array_equal(chosen, target)))

    net = direct.PolicyNet(len(state_cols), hidden=8)
    policy = direct.DirectPolicy(net, state.mean(axis=0), state.std(axis=0))
    restored = pickle.loads(pickle.dumps(policy))
    check("DirectPolicy survives a pickle round trip",
          np.array_equal(policy.predict(state), restored.predict(state)))


def test_sofa():
    print("\nSOFA")
    z = np.zeros(1)
    total = sofa_mod.sofa_score(
        pao2=np.array([500.0]), fio2=np.array([0.21]), ventilated=np.array([False]),
        platelets=np.array([300.0]), bilirubin=np.array([0.5]),
        mbp=np.array([90.0]), vaso_class=np.array([0]), vaso_rate=np.array([np.nan]),
        gcs_total=np.array([15.0]), creatinine=np.array([0.8]))
    check("a healthy physiology scores 0", float(total[0]) == 0.0)

    total = sofa_mod.sofa_score(
        pao2=np.array([50.0]), fio2=np.array([1.0]), ventilated=np.array([True]),
        platelets=np.array([10.0]), bilirubin=np.array([15.0]),
        mbp=np.array([50.0]), vaso_class=np.array([3]), vaso_rate=np.array([0.5]),
        gcs_total=np.array([3.0]), creatinine=np.array([6.0]))
    check("maximal organ failure scores 24", float(total[0]) == 24.0)

    hi = sofa_mod.respiration(np.array([50.0]), np.array([1.0]), np.array([True]))
    lo = sofa_mod.respiration(np.array([50.0]), np.array([1.0]), np.array([False]))
    check("respiratory grades 3-4 require ventilation",
          float(hi[0]) == 4.0 and float(lo[0]) == 2.0)
    check("missing GCS is read as no CNS dysfunction",
          float(sofa_mod.cns(np.array([np.nan]))[0]) == 0.0)


# -------------------------------------------------------------------- OPE ----
def test_ope_helpers():
    print("\nOPE helpers")
    trajs = [np.arange(5) for _ in range(10)]
    uniform = [np.zeros(5) for _ in range(10)]
    ess = ope.effective_sample_size(trajs, uniform)
    check("ESS equals n under uniform weights", abs(ess["ess_final"] - 10) < 1e-6)

    point = [np.concatenate([np.zeros(4), [50.0]]) for _ in range(10)]
    for i in range(1, 10):
        point[i][-1] = -50.0
    ess = ope.effective_sample_size(trajs, point)
    check("ESS collapses toward 1 under a point-mass weight",
          ess["ess_final"] < 1.01)

    p = ope.epsilon_greedy_probs(np.array([1, 0]), 0.1)
    check("epsilon-greedy probabilities sum to 1",
          np.allclose(p.sum(axis=1), 1.0))
    check("epsilon-greedy puts 1 - eps + eps/A on the chosen action",
          abs(p[0, 1] - 0.95) < 1e-9 and abs(p[0, 0] - 0.05) < 1e-9)

    split = {"stay_id": np.array([1, 1, 1, 2, 2])}
    t = ope.to_trajectories(split)
    check("trajectories split on stay boundaries",
          len(t) == 2 and len(t[0]) == 3 and len(t[1]) == 2)


def test_run_filter():
    print("\norder-count run filter (Sec. 3.1)")
    stay = np.zeros(6, dtype=int)
    rec = np.array([1, 1, 1, 0, 1, 1])
    clin = np.zeros(6, dtype=int)
    out = clin.copy()
    out = clin_collapse = clin_run(rec, clin, stay)
    check("a run with no clinician order collapses to its first",
          int(clin_collapse.sum()) == 1 and clin_collapse[0] == 1)

    clin = np.array([0, 0, 0, 1, 0, 0])
    out = clin_run(rec, clin, stay)
    check("a clinician order re-arms the counter", int(out.sum()) == 2)

    stay = np.array([0, 0, 0, 1, 1, 1])
    out = clin_run(np.ones(6, dtype=int), np.zeros(6, dtype=int), stay)
    check("the run filter does not cross stays", int(out.sum()) == 2)


def clin_run(rec, clinician, stay):
    return clin.collapse_runs(rec, clinician, stay)


# ------------------------------------------------------------- artefacts ----
def test_artifacts():
    print("\ngenerated artefacts (skipped if the pipeline has not been run)")
    if not cfg.SPLITS_JSON.exists():
        print("  SKIP  no data/splits.json; run stage 1 first")
        return
    splits = json.loads(cfg.SPLITS_JSON.read_text())
    tr, va, te = (set(splits[k]) for k in ("train", "val", "test"))
    check("no patient appears in two splits",
          not (tr & va) and not (tr & te) and not (va & te))

    meta_p = cfg.RL_DIR / "meta.json"
    if not meta_p.exists():
        print("  SKIP  no rl/meta.json; run stage 3 first")
        return
    meta = json.loads(meta_p.read_text())
    expected_dim = 25 if meta.get("poe_state_included") else 21
    check("meta records the configured state dimension",
          meta["state_dim"] == expected_dim,
          f"expected {expected_dim}, got {meta['state_dim']}")

    p = cfg.RL_DIR / "wbc_test.npz"
    if not p.exists():
        print("  SKIP  no wbc_test.npz; run stage 3 first")
        return
    d = np.load(p)
    check("state is finite everywhere", bool(np.isfinite(d["state"]).all()))
    check("r_sofa, r_treat and r_info are non-negative",
          bool((d["reward"][:, :3] >= 0).all()))
    check("neg_r_cost is non-positive", bool((d["reward"][:, 3] <= 0).all()))
    check("every stay is terminated exactly once",
          int(d["done"].sum()) == len(np.unique(d["stay_id"])))
    check("the action is the observed-lab indicator",
          bool((d["action"] == (~np.isnan(d["obs_lab"])).astype(int)).all()))


def main():
    print("=" * 62)
    test_rewards()
    test_pareto()
    test_budget()
    test_joint_mdp_audit_rejects_bad_transition()
    test_forecaster_no_leakage()
    test_state_and_delta()
    test_poe_features_are_past_only()
    test_batch_slicing()
    test_joint_panels()
    test_direct_policy()
    test_sofa()
    test_ope_helpers()
    test_run_filter()
    test_artifacts()
    print("=" * 62)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
