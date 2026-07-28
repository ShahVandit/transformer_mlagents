import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import evaluate  # noqa: E402
import mdp  # noqa: E402


def test_action_encoding_hold_and_changes():
    assert mdp.encode_action(100, 100, 50, 50) == 4
    assert mdp.encode_action(100, 80, 50, 50) == 1
    assert mdp.encode_action(100, 100, 50, 70) == 5
    assert mdp.decode_action(8) == (2, 2)


def test_reward_components_penalize_bad_future_state():
    df = pd.DataFrame(
        {
            "map": [80, 60, 55, 70],
            "bis": [50, 35, 75, 50],
            "ppf_rate": [100, 100, 100, 100],
            "rftn_rate": [50, 50, 50, 50],
        }
    )
    r = mdp.reward_components(df, 0, 2, 4)
    assert r[0] < 0
    assert r[1] < 0
    assert r[2] == 0


def test_window_has_values_masks_and_padding_channel():
    df = pd.DataFrame({c: [np.nan, 1.0, 2.0] for c in mdp.CHANNELS})
    df = mdp.prepare_frame(df)
    mean = np.zeros(len(mdp.CHANNELS), dtype=np.float32)
    std = np.ones(len(mdp.CHANNELS), dtype=np.float32)
    x = mdp.encode_window(df, 1, history=4, mean=mean, std=std)
    assert x.shape == (4, len(mdp.CHANNELS) * 2 + 1)
    assert x[0, -1] == 1.0
    assert x[-1, -1] == 0.0


def test_pareto_indices():
    rows = [
        {"map_value": 1, "bis_value": 1, "work_value": -1},
        {"map_value": 2, "bis_value": 1, "work_value": -1},
        {"map_value": 1, "bis_value": 2, "work_value": -0.5},
    ]
    assert evaluate.pareto_indices(rows, ["map_value", "bis_value", "work_value"]) == [1, 2]

