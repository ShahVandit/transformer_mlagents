from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import data  # noqa: E402
import evaluate  # noqa: E402


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
