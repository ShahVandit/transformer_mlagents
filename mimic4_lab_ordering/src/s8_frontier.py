"""
Stage 8: evaluate the joint CQL policy family and build the Pareto frontier.

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
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
import s5_evaluate_ope as ope
from s4c_train_family import lam_slug


N_ACTIONS = len(cfg.JOINT_PANEL_BITS)


def load_split(split):
    p = cfg.RL_DIR / f"joint_{split}.npz"
    if not p.exists():
        raise SystemExit(f"{p} not found; run stage 3b first")
    d = np.load(p)
    return {k: d[k] for k in d.files}


def load_bundle(lam):
    p = cfg.MODELS_DIR / f"joint_cql_lam{lam_slug(lam)}.pkl"
    if not p.exists():
        raise SystemExit(f"{p} not found; run stage 4c first")
    with open(p, "rb") as fh:
        return pickle.load(fh)


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


def non_dominated(df):
    vals = df[["wdr_detection", "wdr_burden"]].to_numpy(dtype=float)
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


def make_policy_probs(actions, epsilon):
    return ope.epsilon_greedy_probs(actions, epsilon)


def evaluate_policy(name, bundle, train, test, beh, pi_b_test, trajs, subj_of_traj,
                    epsilon):
    if bundle is None:
        det_actions = test["action"].astype(int)
        pi_e_test = pi_b_test
        pi_e_next = ope.behavior_probs(beh, train["next_state"])
    else:
        det_actions = bundle["policy"].predict(test["state"]).astype(int)
        pi_e_test = make_policy_probs(det_actions, epsilon)
        pi_e_next = make_policy_probs(
            bundle["policy"].predict(train["next_state"]).astype(int), epsilon)

    logw = ope.per_step_log_weights(test, trajs, pi_e_test, pi_b_test)
    qnet = ope.train_fqe(train, pi_e_next)
    qv_test = ope.q_values(qnet, test["state"])
    fqe_v0 = ope.fqe_initial_values(qv_test, pi_e_test, trajs)
    idx_of = {id(tr): k for k, tr in enumerate(trajs)}

    row = {
        "policy": name,
        "lambda": np.nan if bundle is None else float(bundle["lambda"]),
        "draws_per_patient_day": float((det_actions != 0).sum() / patient_days(test)),
        "draw_rate": float((det_actions != 0).mean()),
        "event_coverage_replay": event_coverage(test, det_actions),
    }
    ess = ope.effective_sample_size(trajs, logw)
    row["ess_final"] = ess["ess_final"]
    row["ess_mean"] = ess["ess_mean_over_steps"]

    for d, dim_name in enumerate(cfg.JOINT_REWARD_DIMS):
        clin = ope.factual_return(test, trajs, d, cfg.GAMMA)

        def clin_est(sample, clin=clin):
            return float(np.mean([clin[idx_of[id(tr)]] for tr in sample]))

        def fqe_est(sample, d=d):
            return float(np.mean([fqe_v0[idx_of[id(tr)], d] for tr in sample]))

        def wis_est(sample, d=d):
            lw = [logw[idx_of[id(tr)]] for tr in sample]
            return ope.ps_wis(test, sample, lw, d)

        def wdr_est(sample, d=d):
            lw = [logw[idx_of[id(tr)]] for tr in sample]
            return ope.wdr(test, sample, lw, qv_test, pi_e_test, d)

        for prefix, est in [
            ("clinician", clin_est),
            ("fqe", fqe_est),
            ("wis", wis_est),
            ("wdr", wdr_est),
        ]:
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

    pol = df[df["policy"] != "clinician"].copy()
    clin = df[df["policy"] == "clinician"].iloc[0]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(pol["wdr_burden"], pol["wdr_detection"],
                xerr=[pol["wdr_burden"] - pol["wdr_burden_lo"],
                      pol["wdr_burden_hi"] - pol["wdr_burden"]],
                yerr=[pol["wdr_detection"] - pol["wdr_detection_lo"],
                      pol["wdr_detection_hi"] - pol["wdr_detection"]],
                fmt="o", label="CQL policies")
    ax.scatter([clin["clinician_burden"]], [clin["clinician_detection"]],
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


def write_report(df):
    cols = ["policy", "lambda", "draws_per_patient_day", "event_coverage_replay",
            "wdr_detection", "wdr_burden", "ess_final", "non_dominated"]
    L = ["# Joint-panel Pareto frontier\n\n",
         "Detection is better higher. Burden is better lower. "
         "Non-dominated means no other learned policy has both higher detection "
         "and lower burden under WDR point estimates.\n\n",
         "| policy | lambda | draws/day | replay coverage | WDR detection | WDR burden | ESS final | non-dominated |\n",
         "|---|---:|---:|---:|---:|---:|---:|---|\n"]
    for _, r in df[cols].iterrows():
        lam = "" if pd.isna(r["lambda"]) else f"{r['lambda']:.2f}"
        nd = "yes" if bool(r["non_dominated"]) else ""
        L.append(f"| {r['policy']} | {lam} | {r['draws_per_patient_day']:.3f} | "
                 f"{r['event_coverage_replay']:.3f} | {r['wdr_detection']:+.4f} | "
                 f"{r['wdr_burden']:+.4f} | {r['ess_final']:.1f} | {nd} |\n")
    out = cfg.REPORTS_DIR / "joint_frontier.md"
    out.write_text("".join(L), encoding="utf-8")
    print(f"wrote report -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", nargs="+", type=float, default=cfg.JOINT_LAMBDAS)
    ap.add_argument("--epsilon", type=float, default=cfg.OPE_EPSILON)
    args = ap.parse_args()

    cfg.ensure_dirs()
    train = load_split("train")
    test = load_split("test")

    ope.N_ACTIONS = N_ACTIONS
    ope.D = len(cfg.JOINT_REWARD_DIMS)
    ope.DIM_NAMES = list(cfg.JOINT_REWARD_DIMS)

    print(f"joint frontier: {len(test['action']):,} test transitions, "
          f"{len(np.unique(test['stay_id'])):,} test stays")
    beh = ope.fit_behavior_policy(train)
    pi_b_test = ope.behavior_probs(beh, test["state"])
    trajs = ope.to_trajectories(test)
    subj_of_traj = [int(test["subject_id"][tr[0]]) for tr in trajs]

    rows = []
    print("\n[clinician]")
    rows.append(evaluate_policy("clinician", None, train, test, beh, pi_b_test,
                                trajs, subj_of_traj, args.epsilon))

    for lam in args.lambdas:
        print(f"\n[lambda={lam}]")
        bundle = load_bundle(lam)
        rows.append(evaluate_policy(f"cql_lam{lam_slug(lam)}", bundle, train, test,
                                    beh, pi_b_test, trajs, subj_of_traj, args.epsilon))

    df = pd.DataFrame(rows)
    learned = df["policy"] != "clinician"
    df["non_dominated"] = False
    df.loc[learned, "non_dominated"] = non_dominated(df[learned])

    csv_out = cfg.REPORTS_DIR / "joint_frontier.csv"
    json_out = cfg.REPORTS_DIR / "joint_frontier.json"
    df.to_csv(csv_out, index=False)
    json_out.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    write_report(df)
    maybe_plot(df)
    print(f"wrote -> {csv_out}")
    print(f"wrote -> {json_out}")


if __name__ == "__main__":
    main()
