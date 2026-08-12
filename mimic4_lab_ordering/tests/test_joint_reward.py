"""Standalone truth-table checks for joint information utility and burden."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import itemids as ids  # noqa: E402
import objectives  # noqa: E402
import s4c_train_family as train_family  # noqa: E402


def information_frame():
    n = 6
    frame = {
        "stay_id": np.zeros(n, dtype=np.int64),
        "hour": np.arange(n, dtype=np.float32),
    }
    for lab in ids.TARGET_LABS:
        frame[f"mean_{lab}"] = np.array([0, 1, 4, 4, 1, 0], dtype=np.float32)
        frame[f"last_{lab}"] = np.zeros(n, dtype=np.float32)
        frame[f"std_{lab}"] = np.ones(n, dtype=np.float32)
        frame[f"obs_{lab}"] = np.array(
            [np.nan, 1, 4, np.nan, 1, np.nan], dtype=np.float32)
    return frame


def main():
    frame = information_frame()
    thresholds = objectives.fit_utility_thresholds(frame)
    assert all(value == 1.0 for value in thresholds.values()), thresholds

    potential = objectives.information_potential(frame, thresholds)
    # The binary action uses the strongest supported lab signal. Row 1 is at
    # the threshold and row 2 carries 3 units, regardless of lab count.
    assert potential[1] == 0.0
    assert potential[2] == 3.0

    split = {**frame, "utility_potential": potential}
    never = np.zeros(len(potential), dtype=np.int64)
    low = never.copy()
    low[1] = 1
    high = never.copy()
    high[2] = 1
    repeated = never.copy()
    repeated[[2, 3]] = 1
    always = np.ones(len(potential), dtype=np.int64)

    rows = {}
    for name, actions in {
        "never": never,
        "low_information": low,
        "high_information": high,
        "repeated": repeated,
        "always": always,
    }.items():
        utility = objectives.utility_objective(split, actions)
        burden = objectives.burden_objective(split, actions)
        rows[name] = (float(utility.sum()), float(burden.sum()))

    print("case                 utility  burden")
    for name, (utility, burden) in rows.items():
        print(f"{name:20s} {utility:8.2f} {burden:7.2f}")

    assert rows["never"] == (0.0, 0.0)
    assert rows["low_information"][0] == 0.0
    assert rows["low_information"][1] > 0.0
    assert rows["high_information"][0] > rows["low_information"][0]
    assert rows["repeated"][0] > rows["high_information"][0]
    assert rows["repeated"][1] > rows["high_information"][1]
    assert rows["always"][1] > rows["repeated"][1]

    # No-draw is zero rather than negative. It still loses the positive utility
    # that an informative draw would have collected.
    missed = objectives.utility_objective(split, never)
    taken = objectives.utility_objective(split, high)
    assert missed[2] == 0.0
    assert taken[2] == potential[2] > 0.0

    clinician_actions = high
    clinician_utility = objectives.utility_objective(split, clinician_actions)
    clinician_burden = objectives.burden_objective(split, clinician_actions)
    clinician_raw = np.stack([clinician_utility, clinician_burden], axis=1)
    split["action"] = clinician_actions
    split["reward"] = clinician_raw
    normed, norm_meta = objectives.normalize_rewards(split, clinician_raw)
    split["reward_norm"] = normed[0]

    assert norm_meta["normalization"] == "per_stay_extreme_range"
    assert np.allclose(
        np.asarray(norm_meta["reward_scale"]),
        np.abs(np.asarray(norm_meta["always_draw_return"]) -
               np.asarray(norm_meta["never_draw_return"])),
    )
    for pref in ((0.9, 0.1), (0.5, 0.5), (0.1, 0.9)):
        stored = train_family.scalar_reward(split, pref)
        replayed = train_family.scalar_policy_reward(
            split, clinician_actions, pref, norm_meta)
        assert np.allclose(stored, replayed), pref

    print("PASS: information utility is action-gated, low-information draws "
          "earn no utility, and every draw incurs burden")


if __name__ == "__main__":
    main()
