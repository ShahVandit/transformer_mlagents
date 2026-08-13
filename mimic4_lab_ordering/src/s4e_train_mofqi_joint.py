"""Train one joint two-objective MO-FQI model and extract a policy family."""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
import mofqi
import objectives
import joint_mdp_audit
import s4c_train_family as tf


def parse_preferences(flat):
    if flat is None:
        return [tuple(x) for x in cfg.JOINT_PREFERENCES]
    if len(flat) % 2:
        raise SystemExit("--prefs needs an even count: w_utility w_burden pairs")
    return [(float(flat[i]), float(flat[i + 1]))
            for i in range(0, len(flat), 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefs", nargs="+", type=float, default=None)
    ap.add_argument("--burden-fractions", nargs="+", type=float, default=None,
                    help="validation burden targets relative to clinicians")
    ap.add_argument("--lambda-grid", nargs="+", type=float,
                    default=[0.0, 0.1, 0.3, 1.0, 3.0, 10.0],
                    help="policy-consistent burden multipliers to train")
    ap.add_argument("--iterations", type=int, default=cfg.FQI_ITERATIONS)
    ap.add_argument("--sample-per-iter", type=int, default=cfg.FQI_SAMPLE_PER_ITER)
    ap.add_argument("--tag", default="")
    ap.add_argument("--per-preference-backup", action="store_true",
                    help="fit one preference-consistent model per preference")
    args = ap.parse_args()

    cfg.ensure_dirs()
    meta = json.loads((cfg.RL_DIR / "joint_meta.json").read_text())
    norm_meta = meta["reward_normalization"]
    train = tf.load_split("train")
    val = tf.load_split("val")
    test = tf.load_split("test")
    tf.validate_joint_artifacts(meta, train, val)
    for name, split in (("train", train), ("val", val)):
        objectives.assert_rewards_current(
            split, norm_meta, name=f"joint_{name}.npz")
    print("running mandatory joint MDP audit before training")
    joint_mdp_audit.audit_all(
        meta, {"train": train, "val": val, "test": test})
    print("joint MDP audit: PASS")

    prefs = parse_preferences(args.prefs)
    burden_fractions = (args.burden_fractions
                        or cfg.JOINT_EPSILON_BURDEN_FRACTIONS)
    train_for_fit = dict(train)
    train_for_fit["reward"] = train["reward_norm"].astype(np.float32).copy()
    train_for_fit["reward"][:, 1] *= -1.0

    tag = tf.clean_tag(args.tag)
    print(f"joint MO-FQI: {len(train['action']):,} train transitions, "
          f"state dim={train['state'].shape[1]}, objectives=[utility, -burden]")
    mode = (f"{len(prefs)} preference-consistent backups"
            if args.per_preference_backup else
            f"{len(args.lambda_grid)} policy-consistent lambda backups")
    print(f"training with {mode}, {args.iterations} iterations")
    base_cache = tf.constant_policy_returns(val, norm_meta)
    rows = []
    shared_model = None
    if not args.per_preference_backup:
        lambdas = sorted({float(x) for x in args.lambda_grid
                          if np.isfinite(x) and x >= 0.0})
        if not lambdas:
            raise SystemExit("--lambda-grid must contain a non-negative finite value")
        models = {}
        candidates = []
        for lam in lambdas:
            print(f"\n[lambda_burden={lam:g}]")
            started = time.time()
            model = mofqi.MOFittedQ(
                n_actions=len(cfg.JOINT_PANEL_BITS),
                n_dims=len(cfg.JOINT_REWARD_DIMS),
            )
            model.fit(
                train_for_fit,
                iterations=args.iterations,
                gamma=cfg.GAMMA,
                sample_per_iter=args.sample_per_iter,
                preference=(1.0, lam),
            )
            key = str(lam).replace(".", "p")
            models[key] = model
            policy = mofqi.WeightedQPolicy(model, (1.0, lam))
            acts = policy.predict(val["state"])
            utility = objectives.utility_objective(val, acts)
            burden = objectives.burden_objective(val, acts)
            candidates.append({
                "lambda_burden": lam,
                "model_key": key,
                "utility": float(tf.per_stay_sum(utility, val["stay_id"]).mean()),
                "burden": float(tf.per_stay_sum(burden, val["stay_id"]).mean()),
                "draws_per_patient_day": float(acts.sum() / max(len(acts) / 24.0, 1e-6)),
            })
            print(f"  completed in {time.time() - started:.1f}s; "
                  f"draws/day={candidates[-1]['draws_per_patient_day']:.3f}")

        clinician_burden = float(tf.per_stay_sum(
            objectives.burden_objective(val, val["action"]), val["stay_id"]).mean())
        calibrated = []
        for fraction in burden_fractions:
            limit = float(fraction) * clinician_burden
            feasible = [row for row in candidates if row["burden"] <= limit]
            best = max(feasible, key=lambda row: (row["utility"], -row["burden"])) if feasible else None
            row = {"burden_fraction": float(fraction), "burden_limit": limit,
                   "clinician_burden": clinician_burden, "valid": best is not None}
            if best:
                row.update(best)
                row["achieved_burden_fraction"] = best["burden"] / clinician_burden
            calibrated.append(row)

        shared_path = cfg.MODELS_DIR / f"joint_mofqi_lambda_bundle{tag}.pkl"
        with shared_path.open("wb") as f:
            pickle.dump({"models": models, "candidates": candidates,
                         "state_cols": meta["state_cols"],
                         "reward_dims": ["utility", "neg_burden"],
                         "iterations": args.iterations,
                         "sample_per_iter": args.sample_per_iter,
                         "backup": "preference_consistent_lambda"}, f)
        print(f"lambda bundle saved -> {shared_path}")
        manifest_path = (cfg.MODELS_DIR /
                         f"joint_mofqi_burden_policies{tag}.json")
        manifest_path.write_text(json.dumps({
            "track": "joint",
            "learner": "MO-FQI",
            "model_path": str(shared_path),
            "q_reward_dims": ["utility", "neg_burden"],
            "backup": "preference_consistent_lambda",
            "policy_rule": "argmax weighted Q with preference=(1, lambda)",
            "selection_split": "validation",
            "burden_reference": "clinician_discounted_mean_return",
            "gamma": cfg.GAMMA,
            "policies": calibrated,
        }, indent=2), encoding="utf-8")
        print("\nvalidation-calibrated policies")
        for row in calibrated:
            print(f"  {row['burden_fraction']:.2f}x clinician: "
                  f"lambda={row.get('lambda_burden', float('nan')):.6g} "
                  f"achieved={row.get('achieved_burden_fraction', float('nan')):.3f}x "
                  f"draws/day={row.get('draws_per_patient_day', float('nan')):.3f} "
                  f"utility={row.get('utility', float('nan')):.4f}")
        report = [
            "# Shared MO-FQI burden-calibrated policies\n\n",
            "One preference-independent vector MO-FQI model is trained on "
            "`[utility, -burden]`. Each policy uses a Lagrange multiplier "
            "selected on validation data to maximize utility without exceeding "
            "a stated fraction of clinician burden.\n\n",
            "| burden target | lambda | achieved burden | draws/day | utility |\n",
            "|---:|---:|---:|---:|---:|\n",
        ]
        for row in calibrated:
            report.append(
                f"| {row['burden_fraction']:.2f}x | "
                f"{row.get('lambda_burden', float('nan')):.6g} | "
                f"{row.get('achieved_burden_fraction', float('nan')):.3f}x | "
                f"{row.get('draws_per_patient_day', float('nan')):.3f} | "
                f"{row.get('utility', float('nan')):.4f} |\n")
        out = cfg.REPORTS_DIR / "train_joint_mofqi_family.md"
        out.write_text("".join(report), encoding="utf-8")
        print(f"wrote calibration manifest -> {manifest_path}")
        print(f"wrote report -> {out}")
        return

    for pref in prefs:
        print(f"\n[preference={pref}]")
        model = shared_model
        backup = "vector_pareto_max"
        if args.per_preference_backup:
            started = time.time()
            model = mofqi.MOFittedQ(
                n_actions=len(cfg.JOINT_PANEL_BITS),
                n_dims=len(cfg.JOINT_REWARD_DIMS),
            )
            model.fit(
                train_for_fit,
                iterations=args.iterations,
                gamma=cfg.GAMMA,
                sample_per_iter=args.sample_per_iter,
                preference=pref,
            )
            backup = "preference_consistent"
            print(f"  preference-specific training completed in {time.time() - started:.1f}s")
        model_path = (cfg.MODELS_DIR /
                      (f"joint_mofqi_{tf.pref_slug(pref)}{tag}.pkl"
                       if args.per_preference_backup else
                       f"joint_mofqi_vector{tag}.pkl"))
        if args.per_preference_backup:
            with model_path.open("wb") as f:
                pickle.dump({
                    "model": model,
                    "preference": list(pref),
                    "state_cols": meta["state_cols"],
                    "reward_dims": ["utility", "neg_burden"],
                    "iterations": args.iterations,
                    "sample_per_iter": args.sample_per_iter,
                    "backup": backup,
                }, f)
            print(f"  saved vector-Q model -> {model_path}")

        policy = mofqi.WeightedQPolicy(model, pref)
        base = tf.trivial_baselines(val, pref, norm_meta, base_cache)
        summary = tf.summarize_policy_with_pref(
            policy, val, pref, norm_meta, base)
        status = "VALID" if summary["beats_trivial"] else "REJECTED"
        print(f"  pref={pref}: draws/day={summary['draws_per_patient_day']:.3f} "
              f"coverage={summary['event_coverage']:.3f} "
              f"reward={summary['ep_rew_mean']:+.3f} -> {status}")

        sidecar = cfg.MODELS_DIR / f"joint_mofqi_{tf.pref_slug(pref)}{tag}.json"
        sidecar.write_text(json.dumps({
            "track": "joint",
            "learner": "MO-FQI",
            "model_path": str(model_path),
            "preference": list(pref),
            "reward_dims": cfg.JOINT_REWARD_DIMS,
            "q_reward_dims": ["utility", "neg_burden"],
            "backup": backup,
            "policy_rule": "weighted_draw_advantage_gt_zero",
            "shared_model": not args.per_preference_backup,
            "beats_trivial": summary["beats_trivial"],
            "constant_baselines": base,
            "val_summary": summary,
        }, indent=2), encoding="utf-8")
        rows.append({"preference": list(pref), **summary})

    report = [
        "# Joint MO-FQI policy family\n\n",
        ("Each policy is extracted from one shared MO-FQI vector-Q model "
         "trained on `[utility, -burden]`; preferences weight the learned "
         "action advantages.\n\n" if not args.per_preference_backup else
         "Each vector-Q model was trained on `[utility, -burden]` with its "
         "preference used inside the Bellman backup.\n\n"),
        "| utility weight | burden weight | draws/day | coverage | reward | valid |\n",
        "|---:|---:|---:|---:|---:|---|\n",
    ]
    for row in rows:
        report.append(
            f"| {row['preference'][0]:.1f} | {row['preference'][1]:.1f} | "
            f"{row['draws_per_patient_day']:.3f} | {row['event_coverage']:.3f} | "
            f"{row['ep_rew_mean']:+.3f} | "
            f"{'yes' if row['beats_trivial'] else '**NO**'} |\n")
    out = cfg.REPORTS_DIR / "train_joint_mofqi_family.md"
    out.write_text("".join(report), encoding="utf-8")
    print(f"wrote report -> {out}")


if __name__ == "__main__":
    main()
