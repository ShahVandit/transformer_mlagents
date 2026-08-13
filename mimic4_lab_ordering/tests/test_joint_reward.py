"""Standalone truth-table checks for joint clinical utility and burden."""
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
        frame[f"mean_{lab}"] = np.array(
            [0, 1, 4, 0, 1, 0], dtype=np.float32)
        frame[f"last_{lab}"] = np.zeros(n, dtype=np.float32)
        frame[f"std_{lab}"] = np.ones(n, dtype=np.float32)
        frame[f"obs_{lab}"] = np.array(
            [np.nan, 1, 4, np.nan, 1, np.nan], dtype=np.float32)
        frame[f"delta_{lab}"] = np.array(
            [np.nan, 1, 1, 2, 1, 2], dtype=np.float32)
    frame["action"] = np.array([0, 1, 0, 0, 1, 0], dtype=np.int64)
    return frame


def main():
    frame = information_frame()
    thresholds = objectives.fit_utility_thresholds(frame)
    assert all(value == 1.0 for value in thresholds.values()), thresholds

    potential = objectives.information_potential(frame, thresholds)
    # The binary action is one joint panel, so information is summed over labs.
    assert potential[1] == 0.0
    assert potential[2] == 12.0

    clinician = frame["action"].copy()
    trigger = np.array([0, 0, 1, 0, 1, 0], dtype=np.float32)
    split = {
        **frame,
        "information_potential": potential,
        "clinical_trigger": trigger,
        "draw_burden": objectives.burden_potential(frame),
        "action": clinician,
    }
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
        "off_trigger": low,
        "on_trigger": high,
        "repeated": repeated,
        "always": always,
    }.items():
        utility = objectives.utility_objective(split, actions)
        burden = objectives.burden_objective(split, actions)
        rows[name] = (float(utility.sum()), float(burden.sum()))

    print("case                 utility  burden")
    for name, (utility, burden) in rows.items():
        print(f"{name:20s} {utility:8.2f} {burden:7.2f}")

    assert rows["never"][0] == 0.0
    assert rows["never"][1] == 0.0
    assert rows["off_trigger"][0] == rows["never"][0]
    assert rows["off_trigger"][1] > 0.0
    assert rows["on_trigger"][0] > rows["off_trigger"][0]
    assert rows["repeated"][0] == rows["on_trigger"][0]
    assert rows["repeated"][1] > rows["on_trigger"][1]
    assert rows["always"][0] > rows["on_trigger"][0]
    assert rows["always"][1] > rows["repeated"][1]

    # At a clinical trigger, drawing receives +1. A no-draw decision scores 0.
    missed = objectives.utility_objective(split, never)
    taken = objectives.utility_objective(split, high)
    assert missed[2] == 0.0
    assert taken[2] == trigger[2] == 1.0
    assert objectives.utility_objective(split, low)[1] == missed[1] == 0.0

    # Full clinician-conditioned action table at one trigger hour
    # and one logged no-draw hour.
    informative_hour = 2
    no_draw_hour = 3
    matched_draw = objectives.utility_objective(split, high)
    missed_draw = objectives.utility_objective(split, never)
    extra_draw = objectives.utility_objective(split, repeated)
    matched_no_draw = objectives.utility_objective(split, never)
    assert matched_draw[informative_hour] == trigger[informative_hour]
    assert missed_draw[informative_hour] == 0.0
    assert matched_no_draw[no_draw_hour] == 0.0
    assert extra_draw[no_draw_hour] == 0.0
    assert objectives.burden_objective(split, repeated)[no_draw_hour] > 0.0

    clinician_actions = clinician
    clinician_utility = objectives.utility_objective(split, clinician_actions)
    clinician_burden = objectives.burden_objective(split, clinician_actions)
    clinician_raw = np.stack([clinician_utility, clinician_burden], axis=1)
    split["action"] = clinician_actions
    split["reward"] = clinician_raw
    normed, norm_meta = objectives.normalize_rewards(split, clinician_raw)
    split["reward_norm"] = normed[0]

    assert norm_meta["normalization"] == "train_logged_mean_return"
    expected_scale = np.abs(
        np.asarray(norm_meta["logged_clinician_mean_return"]))
    expected_scale[expected_scale < 1e-6] = 1.0
    assert np.allclose(np.asarray(norm_meta["reward_scale"]), expected_scale)
    for pref in ((0.9, 0.1), (0.5, 0.5), (0.1, 0.9)):
        stored = train_family.scalar_reward(split, pref)
        replayed = train_family.scalar_policy_reward(
            split, clinician_actions, pref, norm_meta)
        assert np.allclose(stored, replayed), pref

    print("PASS: trigger-aligned draws earn +1, no-draw actions earn zero "
          "utility, off-trigger draws earn no utility, and every draw incurs burden")


if __name__ == "__main__":
    main()
