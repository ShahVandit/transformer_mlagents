"""
Unit tests for the pieces that are easy to get subtly wrong.

    python tests/test_core.py

These test properties, not golden numbers, so they stay meaningful when the
cohort or the hyperparameters change. Three of them guard leakage boundaries:
the forecaster must not see the future, the split must not share a patient, and
the state must not contain the result of the order being decided.
"""
import json
import warnings
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config as cfg          # noqa: E402
import forecast as fc         # noqa: E402
import mofqi                  # noqa: E402
import s3_build_mdp as mdp    # noqa: E402
import s5_evaluate_ope as ope  # noqa: E402
import s6_clinical_metrics as clin  # noqa: E402
import s2_hourly_grid as grid  # noqa: E402
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


# ------------------------------------------------------------------ state ----
def test_state_and_delta():
    print("\nstate construction (Sec. 2.2)")
    check("state dimension is 21 as the paper reports",
          len(mdp.state_columns()) == 21, str(len(mdp.state_columns())))

    obs = np.full((1, 6, 1), np.nan)
    obs[0, 2, 0] = 5.0
    last, delta = grid.last_and_delta(obs)
    check("y_t excludes the measurement taken in hour t itself",
          np.isnan(last[0, 2, 0]))
    check("y_t is available from the next hour on", last[0, 3, 0] == 5.0)
    check("Delta_t is 1 an hour after the draw", delta[0, 3, 0] == 1.0)
    check("Delta_t increments hourly", delta[0, 5, 0] == 3.0)
    check("Delta_t is undefined before the first draw", np.isnan(delta[0, 1, 0]))


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
    print("\njoint panel encoding (Pareto track)")
    import panels
    import objectives
    import pandas as pd

    for i in range(16):
        bits = np.array([int(c) for c in format(i, "04b")], dtype=np.int8)
        a = int(panels.encode_bits(bits[None, :])[0])
        if bits.sum() > 0 and a == 0:
            check(f"combination {format(i,'04b')} is not collapsed to `none`", False)
            break
    else:
        check("no real draw is ever encoded as the empty panel", True)

    check("the empty combination maps to `none`",
          int(panels.encode_bits(np.zeros((1, 4), np.int8))[0]) == 0)
    check("creatinine alone maps to the panel it rides on",
          panels.PANEL_BITS[int(panels.encode_bits(
              np.array([[1, 0, 0, 0]], np.int8))[0])] == "1100")

    # every retained panel round-trips
    ok = all(int(panels.encode_bits(panels.PANEL_ARRAY[i][None, :])[0]) == i
             for i in range(len(panels.PANEL_BITS)))
    check("every retained panel encodes to its own id", ok)

    # rare combinations land within Hamming distance 2
    worst = 0
    for i in range(16):
        bits = np.array([int(c) for c in format(i, "04b")], dtype=np.int8)
        a = int(panels.encode_bits(bits[None, :])[0])
        worst = max(worst, int(np.abs(panels.PANEL_ARRAY[a] - bits).sum()))
    check("every combination maps within Hamming distance 2", worst <= 2, str(worst))

    # burden: zero for no draw, larger for a repeat than for a stale draw.
    # Deltas are derived from stay_id/hour, not read from delta_* columns.
    n = 30
    d1 = {"stay_id": np.zeros(n, dtype=int), "hour": np.arange(n, dtype=float)}
    a1 = np.zeros(n, dtype=int)
    a1[0] = 4          # creatinine+bun, first ever
    a1[1] = 4          # repeated one hour later
    a1[25] = 4         # repeated a day later
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
        a2[h0] = 1                                   # creatinine+bun+wbc
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

    # detection: +1 only on a draw before an event, -1 only on an uncovered miss
    n = 30
    d = pd.DataFrame({
        "stay_id": np.zeros(n, dtype=int),
        "sofa_delta": np.zeros(n),
        **{f"onset_{k}": np.zeros(n, dtype=int) for k in
           __import__("itemids").INTERVENTION_KINDS},
    })
    d.loc[20, "onset_vasopressor"] = 1          # event at hour 20
    acts = np.zeros(n, dtype=int)
    acts[15] = 1                                 # a draw 5h before the event
    r, fut, ev = objectives.detection_objective(d, acts, lookahead=12)
    check("a draw inside the window before an event scores +1", r[15] == 1.0)
    check("the event hour itself is flagged", ev[20] == 1)
    check("hours with no upcoming event score 0", r[0] == 0.0)
    # hour 10: the event at 20 is inside the (10, 22] window, and the draw at 15
    # has not happened yet, so nothing covers it
    check("an uncovered hour before an event is penalised", r[10] == -1.0)
    # hour 19: also inside the window, but the draw at 15 is recent, so no penalty
    check("a recent draw suppresses the miss penalty", r[19] == 0.0)
    check("an hour covered by a recent draw is not penalised", r[16] == 0.0)

    # One credit per event. Before the gate on the positive branch, +1 fired on
    # EVERY hour with an event in the lookahead window, so a policy drawing every
    # hour collected +1 on ~32% of all hours and one event could pay out twelve
    # times. That is what drove policies to ~10 draws/day.
    acts2 = np.zeros(n, dtype=int)
    acts2[15] = 1
    acts2[16] = 1                       # second draw inside the same window
    acts2[17] = 1
    r2, _, _ = objectives.detection_objective(d, acts2, lookahead=12)
    check("the first draw in a window claims the event", r2[15] == 1.0)
    check("a second draw in the same window earns nothing", r2[16] == 0.0)
    check("a third draw in the same window earns nothing", r2[17] == 0.0)

    # The credit is capped at one per event, but the PENALTY accrues for every
    # hour spent uncovered while an event is approaching. That asymmetry is
    # deliberate: being blind for seven hours before a deterioration is worse
    # than being blind for one. Its consequence is that detection alone is
    # maximised by always drawing, which is exactly why burden has to be the
    # counterweight. Before the cap, always-draw also accumulated unbounded
    # POSITIVE credit, so detection rose without limit as the policy drew more.
    all_draw = np.ones(n, dtype=int)
    r_all, _, _ = objectives.detection_objective(d, all_draw, lookahead=12)
    r_one, _, _ = objectives.detection_objective(d, acts, lookahead=12)
    r_none, _, _ = objectives.detection_objective(d, np.zeros(n, dtype=int),
                                                  lookahead=12)
    check("always-draw maximises detection (burden is the counterweight)",
          float(r_all.sum()) >= float(r_one.sum()) >= float(r_none.sum()),
          f"all={r_all.sum():.1f} one={r_one.sum():.1f} none={r_none.sum():.1f}")
    check("total positive credit never exceeds the number of events",
          float(r_all[r_all > 0].sum()) <= float(np.asarray(ev).sum()) + 1e-6)
    check("an uncovered stretch is penalised once per hour of exposure",
          float(r_none.sum()) < -1.0)


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
    check("meta records a 21-dim state", meta["state_dim"] == 21)

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
    test_forecaster_no_leakage()
    test_state_and_delta()
    test_batch_slicing()
    test_joint_panels()
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
