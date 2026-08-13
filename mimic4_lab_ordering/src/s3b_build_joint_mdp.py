"""
Stage 3b: joint lab-panel MDP for policy-level Pareto frontier experiments.

Output: data/rl/joint_<split>.npz and data/rl/joint_meta.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
import objectives
import panels
from s3_build_mdp import add_sofa, build_state, state_columns


def make_done_and_next_state(df, state):
    stay = df["stay_id"].to_numpy()
    done = np.zeros(len(df), dtype=np.float32)
    done[np.r_[np.flatnonzero(stay[1:] != stay[:-1]), len(df) - 1]] = 1.0
    nxt = np.arange(len(df)) + 1
    nxt[done == 1.0] = np.flatnonzero(done == 1.0)
    return done, state[nxt]


def build_split(df, utility_thresholds, include_poe=True):
    df = add_sofa(df.sort_values(["stay_id", "hour"]).reset_index(drop=True))
    state, cols = build_state(df, include_poe=include_poe)
    action = panels.encode_frame(df)
    realized_utility = objectives.realized_information(df, utility_thresholds)
    utility = objectives.utility_objective(
        {
            "realized_utility": realized_utility,
            "stay_id": df["stay_id"].to_numpy(),
            "hour": df["hour"].to_numpy(),
        },
        action,
    )
    burden = objectives.burden_objective(df, action)
    reward = np.stack([utility, burden], axis=1).astype(np.float32)
    event = objectives.deterioration_events(df)
    episode_onset = objectives.deterioration_episode_onsets(df)
    future_event = objectives._future_any_by_stay(
        episode_onset, df["stay_id"].to_numpy(),
        cfg.JOINT_DETECTION_LOOKAHEAD_HOURS)
    done, next_state = make_done_and_next_state(df, state)
    return {
        "state": state,
        "action": action.astype(np.int64),
        "reward": reward,
        "next_state": next_state,
        "done": done,
        "stay_id": df["stay_id"].to_numpy(dtype=np.int64),
        "subject_id": df["subject_id"].to_numpy(dtype=np.int64),
        "hour": df["hour"].to_numpy(dtype=np.int64),
        "event": event.astype(np.int8),
        "future_event": future_event.astype(np.int8),
        "realized_utility": realized_utility.astype(np.float32),
        "n_labs": panels.panel_n_labs(action),
    }, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude-poe-state", action="store_true",
                    help="ablation: build the original physiology-only state")
    args = ap.parse_args()

    cfg.ensure_dirs()
    paths = {split: cfg.HOURLY_DIR / f"{split}.parquet"
             for split in ("train", "val", "test")}
    for p in paths.values():
        if not p.exists():
            raise SystemExit(f"{p} not found; run stage 2 first")

    train_frame = pd.read_parquet(paths["train"])
    utility_thresholds = objectives.fit_utility_thresholds(train_frame)
    print("information thresholds fitted on clinician-ordered TRAIN rows")
    for lab, threshold in utility_thresholds.items():
        print(f"  {lab}: {threshold:.4f}")

    built = {}
    state_cols = None
    for split in ("train", "val", "test"):
        frame = train_frame if split == "train" else pd.read_parquet(paths[split])
        d, state_cols = build_split(
            frame, utility_thresholds, include_poe=not args.exclude_poe_state)
        built[split] = d
        print(f"{split}: {len(d['action']):,} hours, "
              f"{len(np.unique(d['stay_id'])):,} stays, "
              f"draw_rate={(d['action'] != 0).mean():.4f}, "
              f"future_event_rate={d['future_event'].mean():.4f}")
        del frame
    del train_frame

    normed, norm_meta = objectives.normalize_rewards(
        built["train"],
        built["train"]["reward"],
        built["val"]["reward"],
        built["test"]["reward"],
    )
    for split, r_norm in zip(("train", "val", "test"), normed):
        built[split]["reward_norm"] = r_norm

    meta = {
        "track": "joint",
        "state_cols": state_cols or state_columns(),
        "state_dim": len(state_cols or state_columns()),
        "poe_state_included": not args.exclude_poe_state,
        "reward_dims": cfg.JOINT_REWARD_DIMS,
        "reward_normalization": norm_meta,
        "panel_bits": cfg.JOINT_PANEL_BITS,
        "panel_names": cfg.JOINT_PANEL_NAMES,
        "lookahead_hours": cfg.JOINT_DETECTION_LOOKAHEAD_HOURS,
        "utility_definition": "clinician_conditioned_signed_realized_information",
        "utility_thresholds": utility_thresholds,
        "gamma": cfg.GAMMA,
        "action_distribution": {
            split: objectives.action_distribution(d["action"])
            for split, d in built.items()
        },
    }
    objectives.save_meta(cfg.RL_DIR / "joint_meta.json", meta)

    print("\nwriting joint transitions")
    for split, d in built.items():
        out = cfg.RL_DIR / f"joint_{split}.npz"
        np.savez_compressed(out, **d)
        r = d["reward"].mean(axis=0)
        print(f"  {split:5s} -> {out}  "
              f"mean utility={r[0]:+.4f}  mean burden={r[1]:+.4f}")

    print("\naction distribution, train")
    for row in meta["action_distribution"]["train"]:
        print(f"  {row['action']}: {row['bits']} {row['panel']:28s} "
              f"{row['frac']:.4%}")


if __name__ == "__main__":
    main()
