from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import data  # noqa: E402
import evaluate  # noqa: E402
import ope  # noqa: E402
import run_d3rlpy_pipeline as d3pipe  # noqa: E402


def test_action_space_has_expected_bins():
    assert data.N_ACTIONS == 4
    assert data.ACTION_LABELS == ["no_bolus", "bolus_le2", "bolus_2_5", "bolus_gt5"]
    assert data.action_labels("basal_bolus12")[4] == "basal_same|no_bolus"


def test_pareto_mask_uses_higher_is_better_values():
    rows = pd.DataFrame(
        [
            {"policy": "dominated", "a": 1.0, "b": 1.0},
            {"policy": "best_a", "a": 2.0, "b": 1.0},
            {"policy": "best_b", "a": 1.0, "b": 2.0},
        ]
    )
    mask = evaluate.pareto_mask(rows, ["a", "b"])
    assert rows.loc[mask, "policy"].tolist() == ["best_a", "best_b"]


def test_d3rlpy_flat_observation_shape():
    seq = np.zeros((24, 11), dtype="float32")
    static = np.zeros((29,), dtype="float32")
    assert d3pipe._flat_observation(seq, static).shape == (293,)


def test_d3rlpy_quantile_action_encoding():
    edges = np.asarray([0.0, 0.3, 0.7], dtype=np.float32)
    actions = d3pipe._encode_quantile_actions(np.asarray([0.0, 0.2, 0.5, 1.0], dtype=np.float32), edges)
    assert actions.tolist() == [0, 1, 2, 3]


def test_d3rlpy_transition_caps_prioritize_training():
    assert d3pipe.transition_caps(300_000) == {
        "train": 240_000,
        "val": 30_000,
        "test": 30_000,
    }


def test_reference_reward_variants_follow_clinical_ranges():
    rewards = d3pipe.glucose_reward_variants(np.asarray([50.0, 100.0, 220.0]))
    assert rewards.shape == (4,)
    assert np.isclose(rewards[1], 1.0 / 3.0)
    assert rewards[2] < rewards[0]


def test_ope_logged_returns_and_identical_policy_wis_match():
    rewards = np.asarray([1.0, 2.0, 3.0, 4.0])
    actions = np.asarray([0, 1, 0, 1])
    trajectories = [np.asarray([0, 1]), np.asarray([2, 3])]
    probs = np.full((4, 2), 0.5)
    log_weights = ope.cumulative_log_ratios(actions, probs, probs, trajectories)
    assert np.allclose(ope.discounted_clinician_returns(rewards, trajectories, 1.0), [3.0, 7.0])
    assert np.isclose(ope.wis_value(rewards, trajectories, log_weights, 1.0), 5.0)


def test_ope_episode_indices_do_not_cross_patients():
    metadata = pd.DataFrame({
        "source_file": ["a", "a", "a"],
        "id": [2, 1, 1],
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01"]),
    })
    trajectories = ope.episode_indices(metadata, 3)
    assert [indices.tolist() for indices in trajectories] == [[2, 1], [0]]


def test_ope_comparison_labels_wdr_against_logged_value():
    frame = pd.DataFrame([{
        "policy": "p", "clinician_return_scalarized": 1.0,
        "clinician_return_scalarized_ci_low": 0.8, "clinician_return_scalarized_ci_high": 1.2,
        "fqe_bootstrap_scalarized": 1.1, "fqe_scalarized_ci_low": 0.9,
        "fqe_scalarized_ci_high": 1.3, "wis_scalarized": 1.4,
        "wis_scalarized_ci_low": 0.7, "wis_scalarized_ci_high": 2.0,
        "wdr_scalarized": 1.5, "wdr_scalarized_ci_low": 1.1,
        "wdr_scalarized_ci_high": 1.7, "fqe_bellman_mse_val_scalarized": 0.1,
        "fqe_bellman_mae_val_scalarized": 0.2, "fqe_bellman_mse_test_scalarized": 0.3,
        "fqe_bellman_mae_test_scalarized": 0.4, "behavior_support_mean": 0.5,
        "behavior_support_p10": 0.2, "behavior_support_frac_ge_0p05": 0.9,
    }])
    out = d3pipe.ope_comparison_frame(frame)
    assert out.loc[0, "wdr_vs_clinician"] == "ABOVE"
