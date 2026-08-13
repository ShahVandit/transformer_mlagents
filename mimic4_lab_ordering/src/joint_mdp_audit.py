"""Fail-fast structural and arithmetic audit for the joint offline-RL MDP."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import config as cfg
import objectives


REQUIRED = {
    "state", "action", "reward", "reward_norm", "next_state", "done",
    "stay_id", "subject_id", "hour", "event", "future_event",
    "information_potential", "draw_burden", "n_labs",
}


def load_split(name):
    path = cfg.RL_DIR / f"joint_{name}.npz"
    if not path.exists():
        raise SystemExit(f"{path} not found; run stage 3b first")
    data = np.load(path)
    return {key: data[key] for key in data.files}


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def audit_split(name, split, meta):
    missing = sorted(REQUIRED - set(split))
    _require(not missing, f"{name}: missing arrays {missing}")

    n = len(split["action"])
    _require(n > 0, f"{name}: no transitions")
    for key in REQUIRED - {"state", "next_state", "reward", "reward_norm"}:
        _require(len(split[key]) == n,
                 f"{name}: {key} has {len(split[key])} rows, expected {n}")
    state = np.asarray(split["state"])
    next_state = np.asarray(split["next_state"])
    reward = np.asarray(split["reward"])
    reward_norm = np.asarray(split["reward_norm"])
    _require(state.shape == (n, meta["state_dim"]),
             f"{name}: state shape {state.shape} != {(n, meta['state_dim'])}")
    _require(next_state.shape == state.shape,
             f"{name}: next_state shape does not match state")
    expected_reward_shape = (n, len(meta["reward_dims"]))
    _require(reward.shape == expected_reward_shape,
             f"{name}: reward shape {reward.shape} != {expected_reward_shape}")
    _require(reward_norm.shape == reward.shape,
             f"{name}: reward_norm shape does not match reward")
    for key in ("state", "next_state", "reward", "reward_norm"):
        _require(np.isfinite(split[key]).all(), f"{name}: {key} is non-finite")

    action = np.asarray(split["action"])
    _require(np.isin(action, np.arange(len(meta["panel_bits"]))).all(),
             f"{name}: action outside configured action space")
    _require(np.array_equal(split["n_labs"],
                            (action != 0).astype(split["n_labs"].dtype)),
             f"{name}: n_labs does not match binary action mapping")

    stay = np.asarray(split["stay_id"])
    subject = np.asarray(split["subject_id"])
    hour = np.asarray(split["hour"])
    done = np.asarray(split["done"])
    _require(np.isin(done, [0.0, 1.0]).all(), f"{name}: done is not binary")
    _require(np.all(stay[1:] >= stay[:-1]), f"{name}: stays are not sorted")

    boundaries = np.r_[0, np.flatnonzero(stay[1:] != stay[:-1]) + 1, n]
    _require(int(done.sum()) == len(boundaries) - 1,
             f"{name}: expected one terminal per stay")
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        _require(np.array_equal(hour[lo:hi], np.arange(hi - lo)),
                 f"{name}: stay {stay[lo]} hours are not contiguous from zero")
        _require((done[lo:hi - 1] == 0).all() and done[hi - 1] == 1,
                 f"{name}: stay {stay[lo]} terminal placement is wrong")
        _require((subject[lo:hi] == subject[lo]).all(),
                 f"{name}: stay {stay[lo]} maps to multiple subjects")
        if hi - lo > 1:
            _require(np.array_equal(next_state[lo:hi - 1], state[lo + 1:hi]),
                     f"{name}: stay {stay[lo]} next-state linkage is wrong")
        _require(np.array_equal(next_state[hi - 1], state[hi - 1]),
                 f"{name}: terminal transition is not a self-loop")

    norm = meta["reward_normalization"]
    objectives.assert_rewards_current(split, norm, name=f"joint_{name}.npz")
    utility = reward[:, 0]
    burden = reward[:, 1]
    potential = np.asarray(split["information_potential"])
    draw_burden = np.asarray(split["draw_burden"])
    draw = action != 0
    _require((potential >= 0).all(), f"{name}: information potential is negative")
    _require(np.allclose(utility[potential == 0], 0.0),
             f"{name}: zero-information rows have nonzero utility")
    _require(np.allclose(utility[draw], potential[draw]),
             f"{name}: logged draws do not receive +information utility")
    _require(np.allclose(utility[~draw], -potential[~draw]),
             f"{name}: logged omissions do not receive -information utility")
    _require((utility > 0).any() and (utility < 0).any(),
             f"{name}: utility has no usable signed factual signal")
    _require((potential[draw] > 0).any() and (potential[~draw] > 0).any(),
             f"{name}: information potential is confounded with logged action")
    _require((draw_burden >= 1.0).all(),
             f"{name}: draw burden potential is below base cost")
    _require((burden >= 0).all(), f"{name}: burden is negative")
    _require(np.allclose(burden[~draw], 0.0),
             f"{name}: no-draw rows carry burden")
    _require(np.allclose(burden[draw], draw_burden[draw]),
             f"{name}: draw rows do not receive state-based burden")
    return {
        "transitions": int(n),
        "stays": int(len(boundaries) - 1),
        "subjects": int(len(np.unique(subject))),
        "draw_rate": float(draw.mean()),
        "mean_utility": float(utility.mean()),
        "mean_information_potential": float(potential.mean()),
        "mean_burden": float(burden.mean()),
    }


def audit_all(meta=None, splits=None):
    if meta is None:
        path = cfg.RL_DIR / "joint_meta.json"
        if not path.exists():
            raise SystemExit(f"{path} not found; run stage 3b first")
        meta = json.loads(path.read_text())
    if splits is None:
        splits = {name: load_split(name) for name in ("train", "val", "test")}

    _require(meta.get("track") == "joint", "metadata track is not joint")
    _require(meta.get("panel_bits") == cfg.JOINT_PANEL_BITS,
             "metadata action mapping is stale")
    _require(meta.get("reward_dims") == cfg.JOINT_REWARD_DIMS,
             "metadata reward dimensions are stale")
    _require(meta.get("reward_normalization", {}).get("normalization")
             == "per_stay_extreme_range", "reward normalization is stale")
    _require(len(meta.get("state_cols", [])) == meta.get("state_dim"),
             "metadata state columns do not match state_dim")
    _require(len(set(meta.get("state_cols", []))) == meta.get("state_dim"),
             "metadata state columns contain duplicates")

    # Independently recompute the scale from TRAIN only. This catches metadata
    # produced from validation/test outcomes or from an older reward function.
    _, expected_norm = objectives.normalize_rewards(
        splits["train"], splits["train"]["reward"])
    got_scale = np.asarray(meta["reward_normalization"]["reward_scale"])
    expected_scale = np.asarray(expected_norm["reward_scale"])
    _require(np.allclose(got_scale, expected_scale, rtol=1e-5, atol=1e-6),
             "reward scale does not match train-only constant-policy range")

    reports = {name: audit_split(name, split, meta)
               for name, split in splits.items()}
    subjects = {name: set(np.asarray(split["subject_id"]).tolist())
                for name, split in splits.items()}
    _require(not (subjects["train"] & subjects["val"]),
             "subject leakage between train and val")
    _require(not (subjects["train"] & subjects["test"]),
             "subject leakage between train and test")
    _require(not (subjects["val"] & subjects["test"]),
             "subject leakage between val and test")
    for name, split in splits.items():
        expected_actions = objectives.action_distribution(split["action"])
        _require(meta.get("action_distribution", {}).get(name) == expected_actions,
                 f"{name}: metadata action distribution is stale")
    return reports


def main():
    reports = audit_all()
    print("joint MDP audit: PASS")
    for name, row in reports.items():
        print(f"  {name}: {row['transitions']:,} transitions, "
              f"{row['stays']:,} stays, {row['subjects']:,} subjects, "
              f"draw_rate={row['draw_rate']:.4f}, "
              f"mean_utility={row['mean_utility']:+.4f}, "
              f"mean_burden={row['mean_burden']:+.4f}")


if __name__ == "__main__":
    main()
