"""Stage 4d: train an ExOSITO-style constrained direct policy family."""
import argparse
import copy
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
import direct_policy as dp
import objectives
from s4c_train_family import clean_tag, load_split, pref_slug


def parse_preferences(flat):
    if flat is None:
        return [tuple(x) for x in cfg.JOINT_PREFERENCES]
    if len(flat) % 2:
        raise SystemExit("--prefs needs an even count: w_utility w_burden pairs")
    return [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]


def fit_behavior(train):
    """Fit on one subject-disjoint train partition and calibrate on another."""
    subjects = np.unique(train["subject_id"])
    rng = np.random.default_rng(cfg.SEED)
    subjects = rng.permutation(subjects)
    n_cal = max(1, int(round(len(subjects) * cfg.DIRECT_CALIBRATION_FRAC)))
    cal_subjects = set(subjects[:n_cal].tolist())
    cal_mask = np.fromiter(
        (int(x) in cal_subjects for x in train["subject_id"]),
        dtype=bool, count=len(train["subject_id"]),
    )
    fit_mask = ~cal_mask
    base = HistGradientBoostingClassifier(
        random_state=cfg.SEED, max_iter=300, learning_rate=0.1)
    base.fit(train["state"][fit_mask], train["action"][fit_mask])
    clf = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    clf.fit(train["state"][cal_mask], train["action"][cal_mask])
    p = clf.predict_proba(train["state"])[:, 1]
    auc = roc_auc_score(train["action"], p)
    brier = brier_score_loss(train["action"], p)
    return clf, p.astype(np.float32), {
        "auc_train": float(auc),
        "brier_train": float(brier),
        "fit_rows": int(fit_mask.sum()),
        "calibration_rows": int(cal_mask.sum()),
        "calibration": "subject_disjoint_sigmoid",
    }


def logged_propensity(draw_prob, actions):
    actions = np.asarray(actions, dtype=np.int64)
    return np.where(actions == 1, draw_prob, 1.0 - draw_prob)


def direct_baselines(g1, stay_ids, behavior_draw, epsilon):
    """Constant policies after projection into the supported action set."""
    behavior_draw = np.asarray(behavior_draw, dtype=np.float32)
    never_actions = np.zeros(len(g1), dtype=np.float32)
    always_actions = np.ones(len(g1), dtype=np.float32)
    no_draw_unsupported = (1.0 - behavior_draw) < epsilon
    draw_unsupported = behavior_draw < epsilon
    never_actions[no_draw_unsupported & ~draw_unsupported] = 1.0
    always_actions[draw_unsupported & ~no_draw_unsupported] = 0.0
    return {
        "never_draw": float(dp._mean_stay_sum(never_actions * g1, stay_ids)),
        "always_draw": float(dp._mean_stay_sum(always_actions * g1, stay_ids)),
    }


def summarize(policy, split, g1, behavior_draw, epsilon, baselines):
    prob = policy.predict_proba(split["state"])[:, 1]
    actions = policy.predict(split["state"])
    expected_value = prob * g1
    hard_value = actions * g1
    overlap = dp.expected_overlap(prob, behavior_draw)
    violation = dp.unsupported_action_mass(prob, behavior_draw, epsilon)
    chosen_propensity = np.where(actions == 1, behavior_draw, 1.0 - behavior_draw)
    best_name = max(baselines, key=baselines.get)
    hard_stay = dp._mean_stay_sum(hard_value, split["stay_id"])
    return {
        "expected_value_mean": float(expected_value.mean()),
        "expected_value_per_stay": dp._mean_stay_sum(expected_value, split["stay_id"]),
        "hard_value_per_stay": hard_stay,
        "draw_rate": float(actions.mean()),
        "draws_per_patient_day": float(actions.sum() / max(len(actions) / 24.0, 1e-6)),
        "clin_draws_per_patient_day": float(
            (split["action"] != 0).sum() / max(len(actions) / 24.0, 1e-6)),
        "event_coverage": objectives.event_coverage(split, actions),
        "mean_expected_overlap": float(overlap.mean()),
        "unsupported_action_mass": float(violation.mean()),
        "support_pass_rate": float((overlap >= epsilon).mean()),
        "hard_support_pass_rate": float((chosen_propensity >= epsilon).mean()),
        "never_draw_rew": baselines["never_draw"],
        "always_draw_rew": baselines["always_draw"],
        "best_constant_name": best_name,
        "best_constant_rew": baselines[best_name],
        "beats_trivial": bool(
            hard_stay >= baselines[best_name] + cfg.VALIDITY_MARGIN - 1e-6),
        "action_counts": {"0": int((actions == 0).sum()), "1": int(actions.sum())},
    }


def train_one(train, val, g1_train, g1_val, b_train, b_val, epsilon,
              baselines, behavior_model, device, epochs):
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    mean = train["state"].mean(axis=0).astype(np.float32)
    std = train["state"].std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    net = dp.PolicyNet(train["state"].shape[1], cfg.DIRECT_HIDDEN).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.DIRECT_LR)
    lambda_value = float(cfg.DIRECT_LAMBDA_INIT)

    states = np.asarray(train["state"], dtype=np.float32)
    n = len(states)
    rng = np.random.default_rng(cfg.SEED)
    best_state = copy.deepcopy(net.state_dict())
    best_score = -np.inf
    stale = 0
    history = []

    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        train_value = train_violation = 0.0
        seen = 0
        net.train()
        for start in range(0, n, cfg.DIRECT_BATCH):
            idx = order[start:start + cfg.DIRECT_BATCH]
            x = torch.as_tensor((states[idx] - mean) / std,
                                dtype=torch.float32, device=device)
            value = torch.as_tensor(g1_train[idx], dtype=torch.float32, device=device)
            behavior = torch.as_tensor(b_train[idx], dtype=torch.float32, device=device)
            prob = net(x)
            utility = (prob * value).mean()
            violation = dp.unsupported_action_mass_torch(
                prob, behavior, epsilon).mean()
            loss = -utility + lambda_value * violation
            opt.zero_grad()
            loss.backward()
            opt.step()
            batch_n = len(idx)
            train_value += float(utility.detach().cpu()) * batch_n
            train_violation += float(violation.detach().cpu()) * batch_n
            seen += batch_n
            lambda_value = max(
                0.0, lambda_value + cfg.DIRECT_LAMBDA_LR
                * float(violation.detach().cpu()))

        policy = dp.DirectPolicy(
            net, mean, std, behavior_model=behavior_model,
            support_epsilon=epsilon)
        summary = summarize(policy, val, g1_val, b_val, epsilon, baselines)
        score = (summary["expected_value_mean"]
                 - lambda_value * summary["unsupported_action_mass"])
        rec = {
            "epoch": epoch,
            "train_expected_value": train_value / seen,
            "train_unsupported_mass": train_violation / seen,
            "lambda": lambda_value,
            "constrained_val_objective": score,
            **summary,
        }
        history.append(rec)
        print(
            f"  epoch={epoch:02d} value/stay={score:+.4f} "
            f"draws/day={summary['draws_per_patient_day']:.3f} "
            f"coverage={summary['event_coverage']:.3f} "
            f"unsupported={summary['unsupported_action_mass']:.6f} "
            f"lambda={lambda_value:.3f}", flush=True)
        if score > best_score + 1e-8:
            best_score = score
            best_state = copy.deepcopy(net.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= cfg.DIRECT_PATIENCE:
                break

    net.load_state_dict(best_state)
    return dp.DirectPolicy(
        net, mean, std, behavior_model=behavior_model,
        support_epsilon=epsilon), history, lambda_value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefs", nargs="+", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=cfg.DIRECT_EPOCHS)
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    cfg.ensure_dirs()
    device = torch.device(args.device)

    meta = json.loads((cfg.RL_DIR / "joint_meta.json").read_text())
    train, val = load_split("train"), load_split("val")
    state_cols = meta["state_cols"]
    thresholds = dp.fit_information_thresholds(
        train["state"], train["action"], state_cols)
    scales = dp.fit_utility_scales(
        train["state"], train["stay_id"], state_cols, thresholds)

    print("fitting calibrated behavior policy")
    behavior, b_train, behavior_meta = fit_behavior(train)
    b_val = behavior.predict_proba(val["state"])[:, 1].astype(np.float32)
    factual = logged_propensity(b_train, train["action"])
    epsilon = float(np.quantile(factual, cfg.DIRECT_EPS_QUANTILE))
    print(f"  AUC={behavior_meta['auc_train']:.4f} "
          f"Brier={behavior_meta['brier_train']:.4f} epsilon={epsilon:.6f}")

    rows = []
    for pref in parse_preferences(args.prefs):
        print(f"\n[w_utility={pref[0]}, w_burden={pref[1]}]")
        g1_train = dp.draw_utility(
            train["state"], state_cols, thresholds, scales, pref)
        g1_val = dp.draw_utility(
            val["state"], state_cols, thresholds, scales, pref)
        baselines = direct_baselines(
            g1_val, val["stay_id"], b_val, epsilon)
        t0 = time.time()
        policy, history, lambda_value = train_one(
            train, val, g1_train, g1_val, b_train, b_val, epsilon,
            baselines, behavior, device, args.epochs)
        summary = summarize(policy, val, g1_val, b_val, epsilon, baselines)
        verdict = "VALID" if summary["beats_trivial"] else "REJECTED"
        print(f"  {verdict}: value={summary['hard_value_per_stay']:+.4f} "
              f"best constant={summary['best_constant_rew']:+.4f} "
              f"elapsed={time.time() - t0:.1f}s")

        stem = cfg.MODELS_DIR / f"joint_direct_{pref_slug(pref)}{clean_tag(args.tag)}"
        model_path = stem.with_suffix(".pkl")
        sidecar_path = stem.with_suffix(".json")
        with model_path.open("wb") as f:
            pickle.dump(policy, f)
        payload = {
            "track": "joint",
            "family": "direct",
            "learner": "DirectConstrainedPolicy",
            "model_path": str(model_path),
            "preference": [float(pref[0]), float(pref[1])],
            "state_cols": state_cols,
            "utility_definition": "past_only_forecast_information_minus_draw_burden",
            "utility_thresholds": thresholds,
            "utility_threshold_source": "train_logged_draw_decision_state",
            "utility_scales": scales,
            "behavior_model": behavior_meta,
            "epsilon": epsilon,
            "epsilon_quantile": cfg.DIRECT_EPS_QUANTILE,
            "lambda_final": lambda_value,
            "epochs": len(history),
            "history": history,
            "constant_baselines": baselines,
            "best_constant_name": summary["best_constant_name"],
            "beats_trivial": summary["beats_trivial"],
            "val_summary": summary,
        }
        sidecar_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        rows.append({"w_utility": pref[0], "w_burden": pref[1], **summary})
        print(f"  saved -> {model_path}")

    report = [
        "# Direct constrained policy family\n\n",
        "Training utility is computed entirely from decision-time state. Event "
        "coverage is evaluation-only.\n\n",
        "| w_utility | w_burden | draws/day | coverage | value/stay | best constant | valid |\n",
        "|---:|---:|---:|---:|---:|---:|---|\n",
    ]
    for row in rows:
        report.append(
            f"| {row['w_utility']:.1f} | {row['w_burden']:.1f} | "
            f"{row['draws_per_patient_day']:.3f} | {row['event_coverage']:.3f} | "
            f"{row['hard_value_per_stay']:+.4f} | "
            f"{row['best_constant_rew']:+.4f} ({row['best_constant_name']}) | "
            f"{'yes' if row['beats_trivial'] else '**NO**'} |\n")
    out = cfg.REPORTS_DIR / "train_joint_direct_family.md"
    out.write_text("".join(report), encoding="utf-8")
    print(f"\nwrote report -> {out}")


if __name__ == "__main__":
    main()
