"""Standalone truth-table checks for the joint reward; requires no torch."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import itemids as ids  # noqa: E402
import objectives  # noqa: E402
import s4c_train_family as train_family  # noqa: E402


def synthetic_stay(event_hours=(20,), n=30):
    data = {
        "stay_id": np.zeros(n, dtype=np.int64),
        "hour": np.arange(n, dtype=np.float32),
        "sofa_delta": np.zeros(n, dtype=np.float32),
    }
    for kind in ids.INTERVENTION_KINDS:
        data[f"onset_{kind}"] = np.zeros(n, dtype=np.int8)
    frame = pd.DataFrame(data)
    for hour in event_hours:
        frame.loc[hour, "onset_vasopressor"] = 1
    return frame


def score(frame, actions, w_det=0.9, w_bur=0.1):
    detection = objectives.detection_objective(frame, actions, lookahead=12)[0]
    burden = objectives.burden_objective(frame, actions)
    return detection, burden, w_det * detection - w_bur * burden


def main():
    frame = synthetic_stay()
    n = len(frame)
    cases = {}
    cases["never"] = np.zeros(n, dtype=np.int64)
    cases["timely_once"] = np.zeros(n, dtype=np.int64)
    cases["timely_once"][15] = 1
    cases["timely_repeated"] = np.zeros(n, dtype=np.int64)
    cases["timely_repeated"][[15, 16, 17]] = 1
    cases["late_only"] = np.zeros(n, dtype=np.int64)
    cases["late_only"][21] = 1
    cases["always"] = np.ones(n, dtype=np.int64)

    expected_detection = {
        "never": -1.0,
        "timely_once": 1.0,
        "timely_repeated": 1.0,
        "late_only": -1.0,
        "always": 1.0,
    }

    print("case              detection  burden  scalar(0.9/0.1)")
    totals = {}
    for name, actions in cases.items():
        detection, burden, scalar = score(frame, actions)
        totals[name] = (float(detection.sum()), float(burden.sum()), float(scalar.sum()))
        print(f"{name:18s} {totals[name][0]:+9.2f} "
              f"{totals[name][1]:7.2f} {totals[name][2]:+16.2f}")
        assert totals[name][0] == expected_detection[name], (name, totals[name])

    assert totals["timely_repeated"][1] > totals["timely_once"][1]
    assert totals["timely_repeated"][2] < totals["timely_once"][2]
    assert totals["late_only"][1] > totals["never"][1]
    assert totals["late_only"][2] < totals["never"][2]
    assert totals["always"][1] > totals["timely_repeated"][1]
    assert totals["always"][2] < totals["timely_once"][2]

    # The epoch callback recomputes rewards under policy actions. When those
    # actions equal the logged clinician actions, both paths must be identical.
    clinician_actions = cases["timely_once"]
    clinician_det, clinician_bur, _ = score(frame, clinician_actions)
    clinician_raw = np.stack([clinician_det, clinician_bur], axis=1)
    split = {
        "stay_id": frame["stay_id"].to_numpy(),
        "hour": frame["hour"].to_numpy(),
        "event": objectives.deterioration_events(frame).astype(np.int8),
        "action": clinician_actions,
        "reward": clinician_raw,
    }
    clinician_normed, clinician_meta = objectives.normalize_rewards(
        split, clinician_raw)
    split["reward_norm"] = clinician_normed[0]
    assert np.array_equal(split["reward_norm"][0], np.zeros(2, dtype=np.float32))
    assert clinician_meta["normalization"] == "per_stay_extreme_range"
    assert np.allclose(
        np.asarray(clinician_meta["reward_scale"]),
        np.abs(np.asarray(clinician_meta["always_draw_return"]) -
               np.asarray(clinician_meta["never_draw_return"])))
    for pref in ((0.9, 0.1), (0.5, 0.5), (0.1, 0.9)):
        stored = train_family.scalar_reward(split, pref)
        replayed = train_family.scalar_policy_reward(
            split, clinician_actions, pref, clinician_meta)
        assert np.allclose(stored, replayed), pref

    # Clustered positive labels are one deterioration episode, so a persistent
    # decline cannot pay repeatedly. Onsets beyond the lookahead are separate.
    clustered = synthetic_stay(event_hours=(20, 21))
    one_draw = np.zeros(len(clustered), dtype=np.int64)
    one_draw[15] = 1
    detection, _, _ = score(clustered, one_draw)
    assert float(detection.sum()) == 1.0

    clustered = synthetic_stay(event_hours=(20, 22))
    one_draw = np.zeros(len(clustered), dtype=np.int64)
    one_draw[15] = 1
    detection, _, _ = score(clustered, one_draw)
    assert float(detection.sum()) == 1.0

    separated = synthetic_stay(event_hours=(10, 25))
    two_draws = np.zeros(len(separated), dtype=np.int64)
    two_draws[[5, 20]] = 1
    detection, _, _ = score(separated, two_draws)
    assert float(detection.sum()) == 2.0

    print("PASS: symmetric event outcomes, no repeated detection credit, "
          "extra draws only increase burden, neutral reward stays zero, "
          "policy and clinician arithmetic are identical")


if __name__ == "__main__":
    main()
