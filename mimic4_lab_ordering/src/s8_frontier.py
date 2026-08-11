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
import s5_evaluate_ope as ope
from s4c_train_family import lam_slug


N_ACTIONS = len(cfg.JOINT_PANEL_BITS)


def load_split(split):
    p = cfg.RL_DIR / f"joint_{split}.npz"
    if not p.exists():
        raise SystemExit(f"{p} not found; run stage 3b first")
    d = np.load(p)
    return {k: d[k] for k in d.files}


def load_policy(lam, device="cpu"):
    p = cfg.MODELS_DIR / f"joint_cql_lam{lam_slug(lam)}.d3"
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
                    epsilon, reward_dim, device="cpu"):
    det_actions = policy.predict(test["state"].astype(np.float32)).astype(int)
    pi_e_test = epsilon_greedy_probs(det_actions, epsilon)

    logw = ope.per_step_log_weights(test, trajs, pi_e_test, pi_b_test)
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
    idx_of = {id(tr): k for k, tr in enumerate(trajs)}

    row = {
        "policy": name,
        "lambda": np.nan if policy is None else float(name.split("lam")[-1].replace("p", ".")) if "lam" in name else np.nan,
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

    def fqe_est(sample):
        return float(np.mean([fqe_v0[idx_of[id(tr)], 0] for tr in sample]))

    def wis_est(sample):
        lw = [logw[idx_of[id(tr)]] for tr in sample]
        return ope.ps_wis(test, sample, lw, reward_dim)

    def wdr_est(sample):
        lw = [logw[idx_of[id(tr)]] for tr in sample]
        return ope.wdr(test, sample, lw, qv_test_3d, pi_e_test, 0)

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
    ax.scatter([clin["wdr_burden"]], [clin["wdr_detection"]],
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
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg.ensure_dirs()
    d3rlpy.seed(cfg.SEED)
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
    clin_det = ope.factual_return(test, trajs, 0, cfg.GAMMA).mean()
    clin_bur = ope.factual_return(test, trajs, 1, cfg.GAMMA).mean()
    rows.append({
        "policy": "clinician",
        "lambda": np.nan,
        "draws_per_patient_day": float((test["action"] != 0).sum() / patient_days(test)),
        "draw_rate": float((test["action"] != 0).mean()),
        "event_coverage_replay": event_coverage(test, test["action"]),
        "wdr_detection": float(clin_det),
        "wdr_burden": float(clin_bur),
        "ess_final": np.nan,
    })

    for lam in args.lambdas:
        print(f"\n[lambda={lam}]")
        policy = load_policy(lam, device=args.device)
        row = {"policy": f"cql_lam{lam_slug(lam)}", "lambda": float(lam)}
        det_actions = policy.predict(test["state"].astype(np.float32)).astype(int)
        row["draws_per_patient_day"] = float((det_actions != 0).sum() / patient_days(test))
        row["draw_rate"] = float((det_actions != 0).mean())
        row["event_coverage_replay"] = event_coverage(test, det_actions)

        for reward_dim in range(len(cfg.JOINT_REWARD_DIMS)):
            est = evaluate_policy(
                row["policy"], policy, train, test, beh, pi_b_test, trajs,
                subj_of_traj, args.epsilon, reward_dim, device=args.device
            )
            for k, v in est.items():
                if k not in row:
                    row[k] = v
        rows.append(row)

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
