"""
Stage 8: evaluate a joint policy family and build the Pareto frontier.

Uses d3rlpy for policy loading and FQE; WIS/WDR remain custom.

Outputs:
  reports/joint_frontier.csv
  reports/joint_frontier.json
  reports/joint_frontier.md
  reports/joint_frontier_ope.png, if matplotlib is available
  reports/joint_frontier_replay.png, if matplotlib is available
"""
import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "d3rlpy"))

import config as cfg
import mofqi
import objectives
import s5_evaluate_ope as ope
import s4c_train_family as tf
from s4c_train_family import clean_tag, lam_slug, pref_slug


N_ACTIONS = len(cfg.JOINT_PANEL_BITS)


def load_split(split):
    p = cfg.RL_DIR / f"joint_{split}.npz"
    if not p.exists():
        raise SystemExit(f"{p} not found; run stage 3b first")
    d = np.load(p)
    return {k: d[k] for k in d.files}


def load_policy(pref, tag="", device="cpu", family="cql"):
    if family == "mofqi":
        p = (cfg.MODELS_DIR /
             f"joint_mofqi_{pref_slug(pref)}{clean_tag(tag)}.pkl")
        if not p.exists():
            raise SystemExit(f"{p} not found; run joint MO-FQI training first")
        with p.open("rb") as f:
            payload = pickle.load(f)
        if not np.allclose(payload.get("preference", pref), pref):
            raise SystemExit(f"{p} was trained for a different preference")
        return mofqi.WeightedQPolicy(payload["model"], pref)
    suffix = ".pkl" if family == "direct" else ".d3"
    p = cfg.MODELS_DIR / f"joint_{family}_{pref_slug(pref)}{clean_tag(tag)}{suffix}"
    if not p.exists():
        raise SystemExit(f"{p} not found; run stage 4 first")
    if family == "direct":
        with p.open("rb") as f:
            return pickle.load(f)
    import d3rlpy
    return d3rlpy.load_learnable(str(p), device=device)


def make_dataset(split, reward):
    from d3rlpy.constants import ActionSpace
    from d3rlpy.dataset import MDPDataset
    return MDPDataset(
        observations=split["state"].astype(np.float32),
        actions=split["action"].astype(np.int64),
        rewards=np.asarray(reward, dtype=np.float32).reshape(-1, 1),
        terminals=split["done"].astype(np.float32),
        action_space=ActionSpace.DISCRETE,
        action_size=N_ACTIONS,
    )


def patient_days(split):
    return max(1e-6, len(split["action"]) / 24.0)


def event_coverage(split, actions, lookback=cfg.JOINT_DETECTION_LOOKAHEAD_HOURS):
    return objectives.event_coverage(split, actions, lookback)


def discounted_policy_objectives(split, actions, gamma=cfg.GAMMA):
    """Replay mean discounted utility and burden per stay."""
    actions = np.asarray(actions, dtype=np.int64)
    values = np.stack([
        objectives.utility_objective(split, actions),
        objectives.burden_objective(split, actions),
    ], axis=1).astype(np.float64)
    stay = np.asarray(split["stay_id"])
    returns = []
    start = 0
    for i in range(1, len(stay) + 1):
        if i == len(stay) or stay[i] != stay[start]:
            discount = gamma ** np.arange(i - start, dtype=np.float64)
            returns.append((values[start:i] * discount[:, None]).sum(axis=0))
            start = i
    if not returns:
        return 0.0, 0.0
    mean = np.asarray(returns, dtype=np.float64).mean(axis=0)
    return float(mean[0]), float(mean[1])


def constant_objective_candidates(split):
    """Constant policies used to validate an epsilon-constrained selection."""
    n = len(split["action"])
    out = []
    for action, name in ((0, "never_draw"), (1, "always_draw")):
        acts = np.full(n, action, dtype=np.int64)
        utility, burden = discounted_policy_objectives(split, acts)
        out.append({"name": name, "utility": utility, "burden": burden})
    return out


def select_epsilon_policies(prefs, tag, val, burden_fractions):
    """Select learned MO-FQI candidates using validation data only."""
    clinician_actions = np.asarray(val["action"], dtype=np.int64)
    _, clinician_burden = discounted_policy_objectives(val, clinician_actions)
    if clinician_burden <= 0.0:
        raise SystemExit("clinician validation burden is not positive")

    candidates = []
    policies = {}
    for pref in prefs:
        name = f"mofqi_{pref_slug(pref)}{clean_tag(tag)}"
        policy = load_policy(pref, tag=tag, device="cpu", family="mofqi")
        actions = policy.predict(val["state"].astype(np.float32)).astype(np.int64)
        utility, burden = discounted_policy_objectives(val, actions)
        candidates.append({
            "name": name,
            "utility": utility,
            "burden": burden,
            "preference": tuple(pref),
            "draws_per_patient_day": float(
                (actions != 0).sum() / patient_days(val)),
        })
        policies[name] = policy

    fractions = [float(x) for x in burden_fractions]
    if any(not np.isfinite(x) or x <= 0.0 for x in fractions):
        raise SystemExit("--burden-fractions must contain positive finite values")
    budgets = [x * clinician_burden for x in fractions]
    selections = mofqi.epsilon_constraint_select(candidates, budgets)
    constants = constant_objective_candidates(val)

    rows = []
    for fraction, result in zip(fractions, selections):
        budget = result["budget"]
        feasible_constants = [x for x in constants if x["burden"] <= budget]
        best_constant = max(feasible_constants, key=lambda x: x["utility"])
        selected = result["selected"]
        valid = bool(
            selected is not None
            and selected["utility"] > best_constant["utility"] + cfg.VALIDITY_MARGIN
        )
        rows.append({
            "burden_fraction": fraction,
            "burden_limit": budget,
            "clinician_burden": clinician_burden,
            "selected_policy": None if selected is None else selected["name"],
            "selected_preference": (None if selected is None
                                    else list(selected["preference"])),
            "val_utility": None if selected is None else selected["utility"],
            "val_burden": None if selected is None else selected["burden"],
            "val_draws_per_patient_day": (None if selected is None else
                                           selected["draws_per_patient_day"]),
            "best_feasible_constant": best_constant["name"],
            "best_feasible_constant_utility": best_constant["utility"],
            "valid": valid,
        })
    return rows, policies


def non_dominated(df, utility_col="wdr_utility", bur_col="wdr_burden"):
    vals = df[[utility_col, bur_col]].to_numpy(dtype=float)
    keep = np.ones(len(vals), dtype=bool)
    for i, (det_i, bur_i) in enumerate(vals):
        for j, (det_j, bur_j) in enumerate(vals):
            if i == j:
                continue
            dominates = (det_j >= det_i and bur_j <= bur_i
                         and (det_j > det_i or bur_j < bur_i))
            if dominates:
                keep[i] = False
                break
    return keep


def policy_beats_trivial(split, actions, pref, norm_meta, cache=None):
    """Did this policy beat the best CONSTANT policy on its own objective?

    A policy that loses to a fixed rule has not found a trade-off, it has
    diverged, and plotting it makes a training failure indistinguishable from a
    preference that genuinely wants more testing.

    `split` must be VALIDATION data. Deciding validity on test would let the
    held-out set select which policies get reported.
    """
    stay = split["stay_id"]
    mine = float(tf.per_stay_sum(
        tf.scalar_policy_reward(split, np.asarray(actions), pref, norm_meta),
        stay).mean())
    base = tf.trivial_baselines(split, pref, norm_meta, cache)
    return bool(mine > max(base.values()) + cfg.VALIDITY_MARGIN)


def validity_from_sidecar(pref, tag, val, norm_meta, cache, family="cql"):
    """Read stage 4c's val verdict; recompute on val only if it is missing."""
    meta_path = (cfg.MODELS_DIR /
                 f"joint_{family}_{pref_slug(pref)}{clean_tag(tag)}.json")
    if meta_path.exists():
        payload = json.loads(meta_path.read_text())
        if family == "cql" and payload.get("reward_dims") != cfg.JOINT_REWARD_DIMS:
            raise SystemExit(
                f"{meta_path} was trained on the old reward; rerun stage 4c")
        verdict = payload.get("beats_trivial")
        if verdict is not None:
            return bool(verdict), f"stage4_{family}_val"
    if family in ("direct", "mofqi"):
        raise SystemExit(f"{meta_path} missing {family} policy validity verdict")
    policy = load_policy(pref, tag=tag, device="cpu", family=family)
    acts = policy.predict(val["state"].astype(np.float32)).astype(np.int64)
    return policy_beats_trivial(val, acts, pref, norm_meta, cache), "recomputed_val"


def fqe_calibration_check(train, test, beh, pi_b_test, trajs, subj_of_traj,
                          clin_returns, device="cpu"):
    """Run FQE with pi_e := pi_b and see whether it recovers the known answer.

    FQE uses no importance weights, so ESS says nothing about it: this is the
    ONLY diagnostic it has. It depends only on the behaviour policy, so one run
    serves every policy in the family rather than one per policy.
    """
    print("\n[FQE calibration check: pi_e := pi_b]")
    out = {}

    for dim, name in enumerate(cfg.JOINT_REWARD_DIMS):
        try:
            train_dim = dict(train)
            train_dim["reward"] = train["reward"][:, [dim]]
            pi_b_next = ope.behavior_probs(beh, train["next_state"])
            fqe = ope.train_fqe(train_dim, pi_b_next, gamma=cfg.GAMMA)
            qv = ope.q_values(fqe, test["state"])
            v0 = ope.fqe_initial_values(qv, pi_b_test, trajs)
            est = float(np.mean(v0[:, 0]))
        except Exception as exc:                       # noqa: BLE001
            print(f"  {name}: FQE on pi_b failed ({exc}); skipping")
            out[name] = {"fqe_on_pi_b": None, "inside_ci": None}
            continue

        v = clin_returns[name]
        idx_of = {id(tr): k for k, tr in enumerate(trajs)}

        def factual(sample, vv=v):
            return float(np.mean([vv[idx_of[id(tr)]] for tr in sample]))

        pt, lo, hi = ope.subject_bootstrap(factual, trajs, subj_of_traj)
        inside = bool(lo <= est <= hi)
        out[name] = {"fqe_on_pi_b": est, "factual": pt,
                     "factual_lo": lo, "factual_hi": hi, "inside_ci": inside}
        print(f"  {name}: FQE {est:+.4f}  factual {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
              f"{'OK' if inside else 'MISMATCH -> do not trust the FQE column'}")
    return out


def hypervolume_2d(points, ref):
    """Dominated area for (maximize utility, minimize burden).

    Burden is negated so both axes maximize, then the standard 2-D sweep
    applies. `ref` is (utility_ref, burden_ref) in the original orientation
    and must be worse than every point on both axes.
    """
    pts = [(d, -b) for d, b in points]
    r = (ref[0], -ref[1])
    pts = [p for p in pts if p[0] >= r[0] and p[1] >= r[1]]
    if not pts:
        return 0.0
    pts.sort(key=lambda p: (-p[0], -p[1]))
    hv, prev_y = 0.0, r[1]
    front = []
    for x, y in pts:                      # keep only non-dominated
        if not front or y > front[-1][1]:
            front.append((x, y))
    for x, y in front:
        hv += (x - r[0]) * (y - prev_y)
        prev_y = y
    return float(hv)


def sparsity_2d(points):
    """Mean squared gap between adjacent frontier points, sorted by burden.

    Low sparsity means the frontier is densely and evenly covered; a large value
    means the points cluster and leave the trade-off unexplored between them.
    """
    if len(points) < 2:
        return float("nan")
    pts = sorted(points, key=lambda p: p[1])
    gaps = [((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
            for a, b in zip(pts[:-1], pts[1:])]
    return float(np.mean(gaps))


def lambda_from_name(name):
    """Recover lambda from a policy name like `cql_lam0p3` or `cql_lam0p3_a0p1`.

    Splitting on "lam" and replacing every "p" with "." breaks the moment a
    --tag is present: `cql_lam0p3_a0p1` becomes "0.3_a0.1", which is not a
    float. The tag exists precisely so runs do not overwrite each other, so the
    parser has to tolerate it.
    """
    m = re.search(r"lam(\d+)p(\d+)", str(name))
    if not m:
        return np.nan
    return float(f"{m.group(1)}.{m.group(2)}")


def epsilon_greedy_probs(actions, epsilon):
    p = np.full((len(actions), N_ACTIONS), epsilon / N_ACTIONS, dtype=np.float32)
    p[np.arange(len(actions)), actions] += 1.0 - epsilon
    return p


def fqe_q_values(fqe, states):
    out = []
    for a in range(N_ACTIONS):
        aa = np.full(len(states), a, dtype=np.int64)
        q = np.asarray(fqe.predict_value(states.astype(np.float32), aa)).reshape(-1)
        out.append(q)
    return np.stack(out, axis=1)


def train_fqe(train_dataset, policy, n_steps, device="cpu", gamma=cfg.GAMMA):
    from d3rlpy.ope import DiscreteFQE, FQEConfig
    from d3rlpy.preprocessing import StandardObservationScaler
    cfg_fqe = FQEConfig(
        learning_rate=cfg.FQE_LR,
        batch_size=cfg.FQE_BATCH,
        gamma=gamma,
        target_update_interval=100,
        observation_scaler=StandardObservationScaler(),
    )
    fqe = DiscreteFQE(algo=policy, config=cfg_fqe, device=device)
    capped_steps = min(n_steps, cfg.FQE_STEPS)
    print(f"  FQE budget: {capped_steps:,} steps "
          f"({cfg.FQE_STEPS_PER_EPOCH:,} steps/epoch cap)")
    fqe.fit(
        train_dataset,
        n_steps=capped_steps,
        n_steps_per_epoch=cfg.FQE_STEPS_PER_EPOCH,
        experiment_name="joint_fqe",
        with_timestamp=False,
        show_progress=False,
    )
    return fqe


def evaluate_policy(name, policy, train, test, beh, pi_b_test, trajs, subj_of_traj,
                    epsilon, reward_dim, device="cpu", ope_mode="all",
                    family="cql"):
    policy_actions = policy.predict(test["state"].astype(np.float32)).astype(int)
    pi_e_test = epsilon_greedy_probs(policy_actions, epsilon)

    logw = ope.per_step_log_weights(test, trajs, pi_e_test, pi_b_test)
    idx_of = {id(tr): k for k, tr in enumerate(trajs)}

    row = {
        "policy": name,
        "lambda": lambda_from_name(name),
        "draws_per_patient_day": float((policy_actions != 0).sum() / patient_days(test)),
        "draw_rate": float((policy_actions != 0).mean()),
        "event_coverage_replay": event_coverage(test, policy_actions),
    }
    ess = ope.effective_sample_size(trajs, logw)
    row["ess_final"] = ess["ess_final"]
    row["ess_mean"] = ess["ess_mean_over_steps"]

    dim_name = cfg.JOINT_REWARD_DIMS[reward_dim]
    clin = ope.factual_return(test, trajs, reward_dim, cfg.GAMMA)

    def clin_est(sample, clin=clin):
        return float(np.mean([clin[idx_of[id(tr)]] for tr in sample]))

    def wis_est(sample):
        lw = [logw[idx_of[id(tr)]] for tr in sample]
        # Explicit gamma. ps_wis defaults to cfg.WIS_GAMMA = 1.0, which is the
        # paper-faithful setting for the per-lab arm; using it here would put an
        # undiscounted policy estimate next to a discounted clinician value.
        return ope.ps_wis(test, sample, lw, reward_dim,
                          gamma=cfg.JOINT_WIS_GAMMA)

    estimators = [("factual", clin_est), ("wis", wis_est)]
    if ope_mode != "wis":
        if family in ("direct", "mofqi"):
            pi_e_train = epsilon_greedy_probs(
                policy.predict(train["next_state"].astype(np.float32)).astype(int),
                epsilon)
            train_dim = dict(train)
            train_dim["reward"] = train["reward"][:, [reward_dim]]
            fqe = ope.train_fqe(train_dim, pi_e_train, gamma=cfg.GAMMA)
            qv_test_3d = ope.q_values(fqe, test["state"])
        else:
            fqe = train_fqe(
                make_dataset(train, train["reward"][:, reward_dim]),
                policy,
                n_steps=max(cfg.FQE_MIN_STEPS, len(train["action"]) // 8),
                device=device,
                gamma=cfg.GAMMA,
            )
            qv_test = fqe_q_values(fqe, test["state"])
            qv_test_3d = qv_test[:, :, None]
        fqe_v0 = ope.fqe_initial_values(qv_test_3d, pi_e_test, trajs)

        def fqe_est(sample):
            return float(np.mean([fqe_v0[idx_of[id(tr)], 0] for tr in sample]))

        # ope.wdr reads its rewards as split["reward"][:, dim]. `qv_test_3d` has
        # a single column, so dim must be 0 there, which means the split handed
        # to wdr must ALSO carry only this objective's reward. Passing the full
        # two-column reward with dim=0 pairs this objective's Q-function with
        # the OTHER objective's rewards, silently corrupting every burden row.
        test_dim = dict(test)
        test_dim["reward"] = test["reward"][:, [reward_dim]]

        def wdr_est(sample):
            lw = [logw[idx_of[id(tr)]] for tr in sample]
            return ope.wdr(test_dim, sample, lw, qv_test_3d, pi_e_test, 0)

        estimators += [("fqe", fqe_est), ("wdr", wdr_est)]

    for prefix, est in estimators:
        pt, lo, hi = ope.subject_bootstrap(est, trajs, subj_of_traj)
        row[f"{prefix}_{dim_name}"] = pt
        row[f"{prefix}_{dim_name}_lo"] = lo
        row[f"{prefix}_{dim_name}_hi"] = hi
    return row


def output_stem(family, selection="weighted"):
    base = "joint_frontier" if family == "cql" else f"joint_{family}_frontier"
    return base if selection == "weighted" else f"joint_{family}_epsilon_frontier"


def maybe_plot(df, family="cql", selection="weighted"):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable; skipping plots ({exc})")
        return

    if "wdr_utility" in df.columns and "wdr_burden" in df.columns:
        prefix = "wdr"
        label = "WDR"
    elif "wis_utility" in df.columns and "wis_burden" in df.columns:
        prefix = "wis"
        label = "WIS"
    else:
        print("WIS/WDR columns unavailable; skipping OPE plot")
        return

    utility_col = f"{prefix}_utility"
    bur_col = f"{prefix}_burden"
    utility_lo_col = f"{utility_col}_lo"
    utility_hi_col = f"{utility_col}_hi"
    bur_lo_col = f"{bur_col}_lo"
    bur_hi_col = f"{bur_col}_hi"

    pol = df[df["policy"] != "clinician"].copy()
    clin = df[df["policy"] == "clinician"].iloc[0]

    fig, ax = plt.subplots(figsize=(7, 5))
    xerr = None
    yerr = None
    if all(c in pol.columns for c in [bur_lo_col, bur_hi_col]):
        xerr = [pol[bur_col] - pol[bur_lo_col],
                pol[bur_hi_col] - pol[bur_col]]
    if all(c in pol.columns for c in [utility_lo_col, utility_hi_col]):
        yerr = [pol[utility_col] - pol[utility_lo_col],
                pol[utility_hi_col] - pol[utility_col]]
    ax.errorbar(pol[bur_col], pol[utility_col], xerr=xerr, yerr=yerr,
                fmt="o", label=f"{family.upper()} policies ({label})")
    ax.scatter([clin[bur_col]], [clin[utility_col]],
               marker="x", s=80, label="clinician")
    ax.set_xlabel("Burden return, lower is better")
    ax.set_ylabel("Information utility return, higher is better")
    ax.legend()
    fig.tight_layout()
    prefix_name = output_stem(family, selection)
    fig.savefig(cfg.REPORTS_DIR / f"{prefix_name}_ope.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(pol["draws_per_patient_day"], pol["event_coverage_replay"],
               label=f"{family.upper()} policies")
    ax.scatter([clin["draws_per_patient_day"]], [clin["event_coverage_replay"]],
               marker="x", s=80, label="clinician")
    ax.set_xlabel("Draws per patient-day")
    ax.set_ylabel("Replay event coverage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.REPORTS_DIR / f"{prefix_name}_replay.png", dpi=160)
    plt.close(fig)


def write_report(df, metrics=None, family="cql", selection="weighted"):
    utility_col = "wdr_utility" if "wdr_utility" in df.columns else "wis_utility"
    bur_col = "wdr_burden" if "wdr_burden" in df.columns else "wis_burden"
    label = "WDR" if utility_col.startswith("wdr") else "WIS"
    cols = ["policy", "w_utility", "w_burden", "epsilon_burden_fractions",
            "draws_per_patient_day",
            "event_coverage_replay", utility_col, bur_col, "ess_final",
            "beats_trivial", "non_dominated"]
    cols = [c for c in cols if c in df.columns]
    L = [f"# Joint-panel Pareto frontier ({family})\n\n",
         "Information utility is better higher. Burden is better lower. "
         "Non-dominated means no other learned policy has both higher utility "
         f"and lower burden under {label} point estimates.\n\n",
         f"| policy | selection | draws/day | replay coverage | {label} utility "
         f"| {label} burden | ESS final | valid | non-dominated |\n",
         "|---|---|---:|---:|---:|---:|---:|---|---|\n"]
    for _, r in df[cols].iterrows():
        if selection == "epsilon" and r.get("policy") != "clinician":
            w = f"epsilon={r.get('epsilon_burden_fractions', '')}x clinician"
        else:
            w = ("" if pd.isna(r.get("w_utility", np.nan))
                 else f"{r['w_utility']:.1f}/{r['w_burden']:.1f}")
        nd = "yes" if bool(r.get("non_dominated")) else ""
        bt = r.get("beats_trivial")
        valid = "" if bt is None or pd.isna(bt) else ("yes" if bt else "**NO**")
        L.append(f"| {r['policy']} | {w} | {r['draws_per_patient_day']:.3f} | "
                 f"{r['event_coverage_replay']:.3f} | {r[utility_col]:+.4f} | "
                 f"{r[bur_col]:+.4f} | {r['ess_final']:.1f} | {valid} | {nd} |\n")

    if metrics:
        L.append("\n## Frontier metrics\n\n")
        L.append(f"- valid policies: {metrics.get('n_valid')}\n")
        L.append(f"- rejected by the validity gate: {metrics.get('n_rejected')}\n")
        L.append(f"- non-dominated: {metrics.get('n_non_dominated')}\n")
        if "hypervolume" in metrics:
            L.append(f"- hypervolume: {metrics['hypervolume']:.4f} "
                     f"(reference {metrics['reference_point']})\n")
            L.append(f"- sparsity: {metrics['sparsity']:.4f}\n")
        cal = metrics.get("fqe_calibration") or {}
        if cal:
            L.append("\n## FQE calibration (pi_e := pi_b)\n\n"
                     "FQE uses no importance weights, so ESS says nothing about "
                     "it. This is its only diagnostic: run it on the behaviour "
                     "policy, where the answer is known.\n\n"
                     "| objective | FQE on pi_b | factual | inside CI |\n"
                     "|---|---:|---:|---|\n")
            for name, c in cal.items():
                if c.get("fqe_on_pi_b") is None:
                    L.append(f"| {name} | failed | | |\n")
                    continue
                L.append(f"| {name} | {c['fqe_on_pi_b']:+.4f} | "
                         f"{c['factual']:+.4f} [{c['factual_lo']:+.4f}, "
                         f"{c['factual_hi']:+.4f}] | "
                         f"{'yes' if c['inside_ci'] else '**NO**'} |\n")

        epsilon_rows = metrics.get("epsilon_selection") or []
        if epsilon_rows:
            L.append("\n## Epsilon-constraint selection on validation\n\n"
                     "Each limit is a fraction of the clinician's validation "
                     "burden. The feasible candidate with maximum validation "
                     "utility is selected before the test set is evaluated.\n\n"
                     "| burden fraction | burden limit | selected policy | "
                     "val utility | val burden | valid |\n"
                     "|---:|---:|---|---:|---:|---|\n")
            for item in epsilon_rows:
                utility = item.get("val_utility")
                burden = item.get("val_burden")
                L.append(
                    f"| {item['burden_fraction']:.3f} | "
                    f"{item['burden_limit']:.4f} | "
                    f"{item.get('selected_policy') or 'none'} | "
                    f"{'' if utility is None else f'{utility:+.4f}'} | "
                    f"{'' if burden is None else f'{burden:.4f}'} | "
                    f"{'yes' if item.get('valid') else '**NO**'} |\n")

    L.append(f"\nAll returns use `gamma = {cfg.GAMMA}`, the clinician row "
             f"included. Mixing a discounted clinician value with an "
             f"undiscounted policy estimate introduces a fixed scale factor of "
             f"roughly (stay length) / (1/(1-gamma)), about 15x on this cohort, "
             f"which reads as the policies beating the clinician 15-fold on "
             f"both objectives when their per-step values are in fact "
             f"comparable.\n")
    stem = output_stem(family, selection)
    out = cfg.REPORTS_DIR / f"{stem}.md"
    out.write_text("".join(L), encoding="utf-8")
    print(f"wrote report -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefs", nargs="+", type=float, default=None,
                    help="flat list of w_utility w_burden pairs, e.g. 0.9 0.1 0.5 0.5")
    ap.add_argument("--tag", default="",
                    help="optional model filename suffix used during training")
    ap.add_argument("--ope", choices=["all", "wis"], default="all",
                    help="all = FQE/WIS/WDR; wis = skip FQE/WDR for fast check")
    ap.add_argument("--epsilon", type=float, default=cfg.OPE_EPSILON)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--family", choices=["cql", "direct", "mofqi"], default="cql")
    ap.add_argument("--selection", choices=["weighted", "epsilon"],
                    default="weighted",
                    help="weighted evaluates every trained preference; epsilon "
                         "maximizes validation utility under burden limits")
    ap.add_argument("--burden-fractions", nargs="+", type=float, default=None,
                    help="epsilon burden limits as fractions of clinician "
                         "validation burden")
    args = ap.parse_args()

    if args.selection == "epsilon" and args.family != "mofqi":
        raise SystemExit("epsilon selection currently requires --family mofqi")

    cfg.ensure_dirs()
    if args.family == "cql":
        import d3rlpy
        d3rlpy.seed(cfg.SEED)
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    norm_meta = json.loads(
        (cfg.RL_DIR / "joint_meta.json").read_text())["reward_normalization"]
    meta = json.loads((cfg.RL_DIR / "joint_meta.json").read_text())
    tf.validate_joint_artifacts(meta, train, val, test)

    if args.prefs:
        flat = list(args.prefs)
        if len(flat) % 2:
            raise SystemExit("--prefs needs an even count: w_utility w_burden pairs")
        prefs = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
    else:
        prefs = [tuple(x) for x in cfg.JOINT_PREFERENCES]

    ope.N_ACTIONS = N_ACTIONS
    ope.D = len(cfg.JOINT_REWARD_DIMS)
    ope.DIM_NAMES = list(cfg.JOINT_REWARD_DIMS)

    for nm, sp in (("train", train), ("val", val), ("test", test)):
        objectives.assert_rewards_current(sp, norm_meta, name=f"joint_{nm}.npz")
    print("reward cache matches the current objectives")

    epsilon_selection = []
    selected_policies = {}
    if args.selection == "epsilon":
        fractions = (args.burden_fractions
                     or cfg.JOINT_EPSILON_BURDEN_FRACTIONS)
        print("selecting epsilon-constrained policies on validation only")
        epsilon_selection, selected_policies = select_epsilon_policies(
            prefs, args.tag, val, fractions)
        for item in epsilon_selection:
            selected = item["selected_policy"] or "NONE"
            status = "VALID" if item["valid"] else "REJECTED"
            print(f"  epsilon={item['burden_fraction']:.3f}x clinician: "
                  f"limit={item['burden_limit']:.4f} selected={selected} "
                  f"val_utility={item['val_utility']} "
                  f"val_burden={item['val_burden']} -> {status}")

    print(f"joint frontier: {len(test['action']):,} test transitions, "
          f"{len(np.unique(test['stay_id'])):,} test stays")
    beh = ope.fit_behavior_policy(train)
    pi_b_test = ope.behavior_probs(beh, test["state"])
    trajs = ope.to_trajectories(test)
    subj_of_traj = [int(test["subject_id"][tr[0]]) for tr in trajs]

    rows = []
    print("\n[clinician]")
    # Every column in this table must use the SAME discount. Reporting the
    # clinician as a gamma=0.9 discounted return next to policy rows estimated
    # with gamma_WIS=1.0 compares a ~10-step effective horizon against a
    # ~150-step one: a fixed 15x scale factor that looked like the policies
    # beating the clinician 15-fold on both objectives when in fact their
    # per-step burden was identical.
    clin = {d: ope.factual_return(test, trajs, i, cfg.GAMMA)
            for i, d in enumerate(cfg.JOINT_REWARD_DIMS)}
    clin_row = {
        "policy": "clinician",
        "w_utility": np.nan, "w_burden": np.nan,
        "draws_per_patient_day": float((test["action"] != 0).sum() / patient_days(test)),
        "draw_rate": float((test["action"] != 0).mean()),
        "event_coverage_replay": event_coverage(test, test["action"]),
        "utility_per_draw_replay": float(
            test["reward"][test["action"] != 0, 0].mean()
            if (test["action"] != 0).any() else 0.0),
        "ess_final": np.nan,
        "beats_trivial": True,
    }
    idx_of_traj = {id(tr): k for k, tr in enumerate(trajs)}
    for d, v in clin.items():
        def est(sample, vv=v):
            return float(np.mean([vv[idx_of_traj[id(tr)]] for tr in sample]))

        pt, lo, hi = ope.subject_bootstrap(est, trajs, subj_of_traj)
        # The factual return IS the ground truth for the behaviour policy, so it
        # fills every estimator column rather than pretending to be an estimate.
        for prefix in ("factual", "wis", "fqe", "wdr"):
            clin_row[f"{prefix}_{d}"] = pt
            clin_row[f"{prefix}_{d}_lo"] = lo
            clin_row[f"{prefix}_{d}_hi"] = hi
        print(f"  factual {d}: {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]  (gamma={cfg.GAMMA})")
    rows.append(clin_row)

    print("\ncaching constant-policy baselines on VAL for the validity gate")
    base_cache = tf.constant_policy_returns(val, norm_meta)

    if args.ope != "wis":
        calibration = fqe_calibration_check(train, test, beh, pi_b_test, trajs,
                                            subj_of_traj, clin, args.device)
    else:
        calibration = {}

    if args.selection == "weighted":
        evaluation_specs = [{
            "name": f"{args.family}_{pref_slug(pref)}{clean_tag(args.tag)}",
            "preference": tuple(pref),
            "policy": None,
            "epsilon_rows": [],
        } for pref in prefs]
    else:
        grouped = {}
        for item in epsilon_selection:
            name = item["selected_policy"]
            if name is None:
                continue
            spec = grouped.setdefault(name, {
                "name": name,
                "preference": tuple(item["selected_preference"]),
                "policy": selected_policies[name],
                "epsilon_rows": [],
            })
            spec["epsilon_rows"].append(item)
        evaluation_specs = list(grouped.values())

    for spec in evaluation_specs:
        pref = spec["preference"]
        print(f"\n[policy={spec['name']}]")
        policy = (spec["policy"] if spec["policy"] is not None else
                  load_policy(pref, tag=args.tag, device=args.device,
                              family=args.family))
        row = {"policy": spec["name"],
               "w_utility": float(pref[0]), "w_burden": float(pref[1])}
        if args.selection == "epsilon":
            row["epsilon_burden_fractions"] = ",".join(
                f"{x['burden_fraction']:.3g}" for x in spec["epsilon_rows"])
            row["epsilon_burden_limits"] = ",".join(
                f"{x['burden_limit']:.6g}" for x in spec["epsilon_rows"])
        policy_actions = policy.predict(test["state"].astype(np.float32)).astype(int)
        draw = policy_actions != 0
        utility = objectives.utility_objective(test, policy_actions)
        row["draws_per_patient_day"] = float(draw.sum() / patient_days(test))
        row["draw_rate"] = float(draw.mean())
        row["utility_per_draw_replay"] = float(
            utility[draw].mean() if draw.any() else 0.0)
        row["event_coverage_replay"] = event_coverage(test, policy_actions)
        # The gate is a VALIDATION decision. Recomputing it on test would let the
        # held-out set choose which policies are reportable, which is exactly the
        # selection test data must never make. Stage 4c already evaluated it on
        # val and stored the verdict beside the model.
        if args.selection == "epsilon":
            row["beats_trivial"] = any(x["valid"] for x in spec["epsilon_rows"])
            row["gate_source"] = "epsilon_constraint_val"
        else:
            row["beats_trivial"], row["gate_source"] = validity_from_sidecar(
                pref, args.tag, val, norm_meta, base_cache, family=args.family)
        if not row["beats_trivial"]:
            print("  WARNING: below the best constant policy on val; "
                  "excluded from the frontier")

        for reward_dim in range(len(cfg.JOINT_REWARD_DIMS)):
            est = evaluate_policy(
                row["policy"], policy, train, test, beh, pi_b_test, trajs,
                subj_of_traj, args.epsilon, reward_dim, device=args.device,
                ope_mode=args.ope, family=args.family
            )
            for k, v in est.items():
                if k not in row:
                    row[k] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    utility_col = "wis_utility" if args.ope == "wis" else "wdr_utility"
    bur_col = "wis_burden" if args.ope == "wis" else "wdr_burden"

    # Only VALID learned policies compete for the frontier. A diverged run that
    # loses to never-draw can still be non-dominated by accident, which would
    # put a training failure on the plot as though it were a trade-off.
    eligible = (df["policy"] != "clinician") & df["beats_trivial"].fillna(False)
    df["non_dominated"] = False
    if eligible.any():
        df.loc[eligible, "non_dominated"] = non_dominated(
            df[eligible], utility_col=utility_col, bur_col=bur_col)
    n_rejected = int(((df["policy"] != "clinician") & ~eligible).sum())
    if n_rejected:
        print(f"\n{n_rejected} policy(s) rejected by the validity gate and "
              f"excluded from the frontier")

    # Frontier-quality metrics, on the non-dominated set. The reference point is
    # the worst value seen on each axis, so hypervolume is comparable across
    # runs of this same pipeline (it is not comparable across datasets).
    front = df[df["non_dominated"]]
    metrics = {"n_valid": int(eligible.sum()), "n_rejected": n_rejected,
               "n_non_dominated": int(len(front))}
    if len(front):
        pts = list(zip(front[utility_col].astype(float), front[bur_col].astype(float)))
        ref = (float(df[utility_col].min()), float(df[bur_col].max()))
        metrics["hypervolume"] = hypervolume_2d(pts, ref)
        metrics["sparsity"] = sparsity_2d(pts)
        metrics["reference_point"] = list(ref)
        print(f"frontier: {len(front)} non-dominated, "
              f"hypervolume={metrics['hypervolume']:.4f}, "
              f"sparsity={metrics['sparsity']:.4f}")
    metrics["fqe_calibration"] = calibration
    metrics["gamma"] = cfg.GAMMA
    metrics["estimator_used_for_frontier"] = utility_col.split("_")[0]
    metrics["family"] = args.family
    metrics["selection"] = args.selection
    if args.selection == "epsilon":
        metrics["epsilon_selection"] = epsilon_selection

    stem = output_stem(args.family, args.selection)
    csv_out = cfg.REPORTS_DIR / f"{stem}.csv"
    json_out = cfg.REPORTS_DIR / f"{stem}.json"
    df.to_csv(csv_out, index=False)
    json_out.write_text(json.dumps(
        {"metrics": metrics, "rows": json.loads(df.to_json(orient="records"))},
        indent=2), encoding="utf-8")
    write_report(df, metrics, family=args.family, selection=args.selection)
    maybe_plot(df, family=args.family, selection=args.selection)
    print(f"wrote -> {csv_out}")
    print(f"wrote -> {json_out}")


if __name__ == "__main__":
    main()
