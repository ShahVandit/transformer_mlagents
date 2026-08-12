"""
Stage 8: evaluate the joint CQL policy family and build the Pareto frontier.

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
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "d3rlpy"))

import config as cfg
import d3rlpy
from d3rlpy.constants import ActionSpace
from d3rlpy.dataset import MDPDataset
from d3rlpy.ope import DiscreteFQE, FQEConfig
from d3rlpy.preprocessing import StandardObservationScaler
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


def load_policy(pref, tag="", device="cpu"):
    p = cfg.MODELS_DIR / f"joint_cql_{pref_slug(pref)}{clean_tag(tag)}.d3"
    if not p.exists():
        raise SystemExit(f"{p} not found; run stage 4c first")
    return d3rlpy.load_learnable(str(p), device=device)


def make_dataset(split, reward):
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
    event = split["event"].astype(bool)
    draw = np.asarray(actions) != 0
    stay = split["stay_id"]
    covered = np.zeros(len(event), dtype=bool)
    start = 0
    for i in range(1, len(event) + 1):
        if i == len(event) or stay[i] != stay[start]:
            e = event[start:i]
            d = draw[start:i].astype(np.int8)
            c = np.r_[0, np.cumsum(d)]
            for j in np.flatnonzero(e):
                lo = max(0, j - lookback)
                covered[start + j] = (c[j] - c[lo]) > 0
            start = i
    n_events = int(event.sum())
    return float(covered[event].mean()) if n_events else np.nan


def non_dominated(df, det_col="wdr_detection", bur_col="wdr_burden"):
    vals = df[[det_col, bur_col]].to_numpy(dtype=float)
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


def validity_from_sidecar(pref, tag, val, norm_meta, cache):
    """Read stage 4c's val verdict; recompute on val only if it is missing."""
    meta_path = (cfg.MODELS_DIR /
                 f"joint_cql_{pref_slug(pref)}{clean_tag(tag)}.json")
    if meta_path.exists():
        payload = json.loads(meta_path.read_text())
        verdict = payload.get("beats_trivial")
        if verdict is not None:
            return bool(verdict), "stage4c_val"
    policy = load_policy(pref, tag=tag, device="cpu")
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

    class _BehaviourPolicy:
        """Minimal shim so DiscreteFQE can treat pi_b as the target policy."""

        def __init__(self, clf):
            self.clf = clf

        def predict(self, states):
            return self.clf.predict(np.asarray(states)).astype(np.int64)

    for dim, name in enumerate(cfg.JOINT_REWARD_DIMS):
        try:
            fqe = train_fqe(
                make_dataset(train, train["reward"][:, dim]),
                _BehaviourPolicy(beh),
                n_steps=max(cfg.FQE_MIN_STEPS, len(train["action"]) // 8),
                device=device, gamma=cfg.GAMMA,
            )
            qv = fqe_q_values(fqe, test["state"])[:, :, None]
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
    """Dominated area for (maximize detection, minimize burden).

    Burden is negated so both axes maximize, then the standard 2-D sweep
    applies. `ref` is (detection_ref, burden_ref) in the original orientation
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
                    epsilon, reward_dim, device="cpu", ope_mode="all"):
    det_actions = policy.predict(test["state"].astype(np.float32)).astype(int)
    pi_e_test = epsilon_greedy_probs(det_actions, epsilon)

    logw = ope.per_step_log_weights(test, trajs, pi_e_test, pi_b_test)
    idx_of = {id(tr): k for k, tr in enumerate(trajs)}

    row = {
        "policy": name,
        "lambda": lambda_from_name(name),
        "draws_per_patient_day": float((det_actions != 0).sum() / patient_days(test)),
        "draw_rate": float((det_actions != 0).mean()),
        "event_coverage_replay": event_coverage(test, det_actions),
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


def maybe_plot(df):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable; skipping plots ({exc})")
        return

    if "wdr_detection" in df.columns and "wdr_burden" in df.columns:
        prefix = "wdr"
        label = "WDR"
    elif "wis_detection" in df.columns and "wis_burden" in df.columns:
        prefix = "wis"
        label = "WIS"
    else:
        print("WIS/WDR columns unavailable; skipping OPE plot")
        return

    det_col = f"{prefix}_detection"
    bur_col = f"{prefix}_burden"
    det_lo_col = f"{det_col}_lo"
    det_hi_col = f"{det_col}_hi"
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
    if all(c in pol.columns for c in [det_lo_col, det_hi_col]):
        yerr = [pol[det_col] - pol[det_lo_col],
                pol[det_hi_col] - pol[det_col]]
    ax.errorbar(pol[bur_col], pol[det_col], xerr=xerr, yerr=yerr,
                fmt="o", label=f"CQL policies ({label})")
    ax.scatter([clin[bur_col]], [clin[det_col]],
               marker="x", s=80, label="clinician")
    ax.set_xlabel("Burden return, lower is better")
    ax.set_ylabel("Detection return, higher is better")
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.REPORTS_DIR / "joint_frontier_ope.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(pol["draws_per_patient_day"], pol["event_coverage_replay"], label="CQL policies")
    ax.scatter([clin["draws_per_patient_day"]], [clin["event_coverage_replay"]],
               marker="x", s=80, label="clinician")
    ax.set_xlabel("Draws per patient-day")
    ax.set_ylabel("Replay event coverage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.REPORTS_DIR / "joint_frontier_replay.png", dpi=160)
    plt.close(fig)


def write_report(df, metrics=None):
    det_col = "wdr_detection" if "wdr_detection" in df.columns else "wis_detection"
    bur_col = "wdr_burden" if "wdr_burden" in df.columns else "wis_burden"
    label = "WDR" if det_col.startswith("wdr") else "WIS"
    cols = ["policy", "w_detection", "w_burden", "draws_per_patient_day",
            "event_coverage_replay", det_col, bur_col, "ess_final",
            "beats_trivial", "non_dominated"]
    cols = [c for c in cols if c in df.columns]
    L = ["# Joint-panel Pareto frontier\n\n",
         "Detection is better higher. Burden is better lower. "
         "Non-dominated means no other learned policy has both higher detection "
         f"and lower burden under {label} point estimates.\n\n",
         f"| policy | w_det/w_bur | draws/day | replay coverage | {label} detection "
         f"| {label} burden | ESS final | valid | non-dominated |\n",
         "|---|---|---:|---:|---:|---:|---:|---|---|\n"]
    for _, r in df[cols].iterrows():
        w = ("" if pd.isna(r.get("w_detection", np.nan))
             else f"{r['w_detection']:.1f}/{r['w_burden']:.1f}")
        nd = "yes" if bool(r.get("non_dominated")) else ""
        bt = r.get("beats_trivial")
        valid = "" if bt is None or pd.isna(bt) else ("yes" if bt else "**NO**")
        L.append(f"| {r['policy']} | {w} | {r['draws_per_patient_day']:.3f} | "
                 f"{r['event_coverage_replay']:.3f} | {r[det_col]:+.4f} | "
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

    L.append(f"\nAll returns use `gamma = {cfg.GAMMA}`, the clinician row "
             f"included. Mixing a discounted clinician value with an "
             f"undiscounted policy estimate introduces a fixed scale factor of "
             f"roughly (stay length) / (1/(1-gamma)), about 15x on this cohort, "
             f"which reads as the policies beating the clinician 15-fold on "
             f"both objectives when their per-step values are in fact "
             f"comparable.\n")
    out = cfg.REPORTS_DIR / "joint_frontier.md"
    out.write_text("".join(L), encoding="utf-8")
    print(f"wrote report -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefs", nargs="+", type=float, default=None,
                    help="flat list of w_detection w_burden pairs, e.g. 0.9 0.1 0.5 0.5")
    ap.add_argument("--tag", default="",
                    help="optional model filename suffix used during training")
    ap.add_argument("--ope", choices=["all", "wis"], default="all",
                    help="all = FQE/WIS/WDR; wis = skip FQE/WDR for fast check")
    ap.add_argument("--epsilon", type=float, default=cfg.OPE_EPSILON)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg.ensure_dirs()
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
            raise SystemExit("--prefs needs an even count: w_detection w_burden pairs")
        prefs = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
    else:
        prefs = [tuple(x) for x in cfg.JOINT_PREFERENCES]

    ope.N_ACTIONS = N_ACTIONS
    ope.D = len(cfg.JOINT_REWARD_DIMS)
    ope.DIM_NAMES = list(cfg.JOINT_REWARD_DIMS)

    for nm, sp in (("train", train), ("val", val), ("test", test)):
        objectives.assert_rewards_current(sp, norm_meta, name=f"joint_{nm}.npz")
    print("reward cache matches the current objectives")

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
        "w_detection": np.nan, "w_burden": np.nan,
        "draws_per_patient_day": float((test["action"] != 0).sum() / patient_days(test)),
        "draw_rate": float((test["action"] != 0).mean()),
        "event_coverage_replay": event_coverage(test, test["action"]),
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

    for pref in prefs:
        print(f"\n[w_det={pref[0]}, w_bur={pref[1]}]")
        policy = load_policy(pref, tag=args.tag, device=args.device)
        row = {"policy": f"cql_{pref_slug(pref)}{clean_tag(args.tag)}",
               "w_detection": float(pref[0]), "w_burden": float(pref[1])}
        det_actions = policy.predict(test["state"].astype(np.float32)).astype(int)
        row["draws_per_patient_day"] = float((det_actions != 0).sum() / patient_days(test))
        row["draw_rate"] = float((det_actions != 0).mean())
        row["event_coverage_replay"] = event_coverage(test, det_actions)
        # The gate is a VALIDATION decision. Recomputing it on test would let the
        # held-out set choose which policies are reportable, which is exactly the
        # selection test data must never make. Stage 4c already evaluated it on
        # val and stored the verdict beside the model.
        row["beats_trivial"], row["gate_source"] = validity_from_sidecar(
            pref, args.tag, val, norm_meta, base_cache)
        if not row["beats_trivial"]:
            print("  WARNING: below the best constant policy on val; "
                  "excluded from the frontier")

        for reward_dim in range(len(cfg.JOINT_REWARD_DIMS)):
            est = evaluate_policy(
                row["policy"], policy, train, test, beh, pi_b_test, trajs,
                subj_of_traj, args.epsilon, reward_dim, device=args.device,
                ope_mode=args.ope
            )
            for k, v in est.items():
                if k not in row:
                    row[k] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    det_col = "wis_detection" if args.ope == "wis" else "wdr_detection"
    bur_col = "wis_burden" if args.ope == "wis" else "wdr_burden"

    # Only VALID learned policies compete for the frontier. A diverged run that
    # loses to never-draw can still be non-dominated by accident, which would
    # put a training failure on the plot as though it were a trade-off.
    eligible = (df["policy"] != "clinician") & df["beats_trivial"].fillna(False)
    df["non_dominated"] = False
    if eligible.any():
        df.loc[eligible, "non_dominated"] = non_dominated(
            df[eligible], det_col=det_col, bur_col=bur_col)
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
        pts = list(zip(front[det_col].astype(float), front[bur_col].astype(float)))
        ref = (float(df[det_col].min()), float(df[bur_col].max()))
        metrics["hypervolume"] = hypervolume_2d(pts, ref)
        metrics["sparsity"] = sparsity_2d(pts)
        metrics["reference_point"] = list(ref)
        print(f"frontier: {len(front)} non-dominated, "
              f"hypervolume={metrics['hypervolume']:.4f}, "
              f"sparsity={metrics['sparsity']:.4f}")
    metrics["fqe_calibration"] = calibration
    metrics["gamma"] = cfg.GAMMA
    metrics["estimator_used_for_frontier"] = det_col.split("_")[0]

    csv_out = cfg.REPORTS_DIR / "joint_frontier.csv"
    json_out = cfg.REPORTS_DIR / "joint_frontier.json"
    df.to_csv(csv_out, index=False)
    json_out.write_text(json.dumps(
        {"metrics": metrics, "rows": json.loads(df.to_json(orient="records"))},
        indent=2), encoding="utf-8")
    write_report(df, metrics)
    maybe_plot(df)
    print(f"wrote -> {csv_out}")
    print(f"wrote -> {json_out}")


if __name__ == "__main__":
    main()
