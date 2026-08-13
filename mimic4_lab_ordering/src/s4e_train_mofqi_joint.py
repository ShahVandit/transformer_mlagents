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
    ap.add_argument("--iterations", type=int, default=cfg.FQI_ITERATIONS)
    ap.add_argument("--sample-per-iter", type=int, default=cfg.FQI_SAMPLE_PER_ITER)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg.ensure_dirs()
    meta = json.loads((cfg.RL_DIR / "joint_meta.json").read_text())
    norm_meta = meta["reward_normalization"]
    train = tf.load_split("train")
    val = tf.load_split("val")
    tf.validate_joint_artifacts(meta, train, val)
    for name, split in (("train", train), ("val", val)):
        objectives.assert_rewards_current(
            split, norm_meta, name=f"joint_{name}.npz")

    prefs = parse_preferences(args.prefs)
    train_for_fit = dict(train)
    train_for_fit["reward"] = train["reward_norm"].astype(np.float32).copy()
    train_for_fit["reward"][:, 1] *= -1.0

    print(f"joint MO-FQI: {len(train['action']):,} train transitions, "
          f"state dim={train['state'].shape[1]}, objectives=[utility, -burden]")
    print(f"training one vector-Q model for {args.iterations} iterations")
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
    )
    print(f"training completed in {time.time() - started:.1f}s")

    tag = tf.clean_tag(args.tag)
    model_path = cfg.MODELS_DIR / f"joint_mofqi{tag}.pkl"
    with model_path.open("wb") as f:
        pickle.dump({
            "model": model,
            "state_cols": meta["state_cols"],
            "reward_dims": ["utility", "neg_burden"],
            "iterations": args.iterations,
            "sample_per_iter": args.sample_per_iter,
        }, f)
    print(f"saved shared vector-Q model -> {model_path}")

    base_cache = tf.constant_policy_returns(val, norm_meta)
    rows = []
    for pref in prefs:
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
            "shared_model_path": str(model_path),
            "preference": list(pref),
            "reward_dims": cfg.JOINT_REWARD_DIMS,
            "q_reward_dims": ["utility", "neg_burden"],
            "beats_trivial": summary["beats_trivial"],
            "constant_baselines": base,
            "val_summary": summary,
        }, indent=2), encoding="utf-8")
        rows.append({"preference": list(pref), **summary})

    report = [
        "# Joint MO-FQI policy family\n\n",
        "One vector-Q model was trained on `[utility, -burden]`. The preference "
        "is applied only during action selection.\n\n",
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
