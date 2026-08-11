"""
Stage 4c: train a family of joint-panel CQL policies with d3rlpy.

Each policy uses the same two raw objectives and the same normalized reward:

    r_lambda = z_detection - lambda * z_burden

Output:
  models/joint_cql_lam{lambda}.d3
  models/joint_cql_lam{lambda}.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "d3rlpy"))

import config as cfg

import d3rlpy
from d3rlpy.constants import ActionSpace
from d3rlpy.dataset import MDPDataset
from d3rlpy.preprocessing import StandardObservationScaler


N_ACTIONS = len(cfg.JOINT_PANEL_BITS)


def lam_slug(lam):
    return str(lam).replace(".", "p")


def load_split(split):
    p = cfg.RL_DIR / f"joint_{split}.npz"
    if not p.exists():
        raise SystemExit(f"{p} not found; run stage 3b first")
    d = np.load(p)
    return {k: d[k] for k in d.files}


def scalar_reward(split, lam):
    r = split["reward_norm"].astype(np.float32)
    return (r[:, 0] - float(lam) * r[:, 1]).astype(np.float32)


def make_dataset(split, reward):
    return MDPDataset(
        observations=split["state"].astype(np.float32),
        actions=split["action"].astype(np.int64),
        rewards=np.asarray(reward, dtype=np.float32).reshape(-1, 1),
        terminals=split["done"].astype(np.float32),
        action_space=ActionSpace.DISCRETE,
        action_size=N_ACTIONS,
    )


def make_cql(device, alpha):
    return d3rlpy.algos.DiscreteCQLConfig(
        learning_rate=cfg.CQL_LR,
        batch_size=cfg.CQL_BATCH,
        gamma=cfg.GAMMA,
        alpha=alpha,
        target_update_interval=cfg.CQL_EVAL_EVERY,
        observation_scaler=StandardObservationScaler(),
    ).create(device=device)


def summarize_policy(algo, split):
    actions = algo.predict(split["state"].astype(np.float32)).astype(np.int64)
    any_draw = actions != 0
    days = max(1e-6, len(actions) / 24.0)
    return {
        "draw_rate": float(any_draw.mean()),
        "draws_per_patient_day": float(any_draw.sum() / days),
        "action_counts": {str(i): int((actions == i).sum()) for i in range(N_ACTIONS)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", nargs="+", type=float, default=cfg.JOINT_LAMBDAS)
    ap.add_argument("--steps", type=int, default=cfg.CQL_STEPS)
    ap.add_argument("--alpha", type=float, default=cfg.CQL_ALPHA)
    ap.add_argument("--device", default=False,
                    help="d3rlpy device argument: False, cpu, cuda:0, or GPU id")
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()

    cfg.ensure_dirs()
    d3rlpy.seed(cfg.SEED)

    meta = json.loads((cfg.RL_DIR / "joint_meta.json").read_text())
    train = load_split("train")
    val = load_split("val")

    print(f"joint d3rlpy CQL family: train n={len(train['action']):,}, "
          f"state dim={train['state'].shape[1]}, actions={N_ACTIONS}")
    print("reward: z_detection - lambda * z_burden")

    rows = []
    for lam in args.lambdas:
        print(f"\n[lambda={lam}] d3rlpy DiscreteCQL, "
              f"{args.steps:,} steps, alpha={args.alpha}")
        reward = scalar_reward(train, lam)
        dataset = make_dataset(train, reward)
        algo = make_cql(args.device, args.alpha)

        t0 = time.time()
        history = algo.fit(
            dataset,
            n_steps=args.steps,
            n_steps_per_epoch=cfg.CQL_EVAL_EVERY,
            experiment_name=f"joint_cql_lam{lam_slug(lam)}",
            with_timestamp=False,
            show_progress=args.progress,
            save_interval=max(1, args.steps // max(1, cfg.CQL_EVAL_EVERY)),
        )
        elapsed = time.time() - t0

        val_summary = summarize_policy(algo, val)
        print(f"  done in {elapsed:.1f}s")
        print(f"  val draws/patient-day={val_summary['draws_per_patient_day']:.3f}  "
              f"draw rate={val_summary['draw_rate']:.4f}")

        stem = cfg.MODELS_DIR / f"joint_cql_lam{lam_slug(lam)}"
        model_path = stem.with_suffix(".d3")
        meta_path = stem.with_suffix(".json")
        algo.save(str(model_path))
        payload = {
            "track": "joint",
            "library": "d3rlpy",
            "learner": "DiscreteCQL",
            "model_path": str(model_path),
            "lambda": float(lam),
            "n_actions": N_ACTIONS,
            "panel_bits": cfg.JOINT_PANEL_BITS,
            "panel_names": cfg.JOINT_PANEL_NAMES,
            "reward_dims": cfg.JOINT_REWARD_DIMS,
            "state_cols": meta["state_cols"],
            "alpha": args.alpha,
            "steps": args.steps,
            "history": [(int(e), {k: float(v) for k, v in m.items()})
                        for e, m in history],
            "val_summary": val_summary,
        }
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  saved -> {model_path}")
        rows.append({"lambda": float(lam), **val_summary})

    report = ["# Joint d3rlpy CQL policy family\n\n",
              "| lambda | val draws/patient-day | val draw rate |\n",
              "|---|---:|---:|\n"]
    for row in rows:
        report.append(f"| {row['lambda']} | {row['draws_per_patient_day']:.3f} | "
                      f"{row['draw_rate']:.4f} |\n")
    out_report = cfg.REPORTS_DIR / "train_joint_cql_family.md"
    out_report.write_text("".join(report), encoding="utf-8")
    print(f"\nwrote report -> {out_report}")


if __name__ == "__main__":
    main()
