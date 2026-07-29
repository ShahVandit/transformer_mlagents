from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path)
    return {key: payload[key] for key in payload.files}


def action_distribution(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with (data_dir / "mdp_config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    labels = config["action_labels"]
    rows = []
    for split in ["train", "val", "test"]:
        data = load_npz(data_dir / f"transitions_{split}.npz")
        counts = np.bincount(data["action"], minlength=len(labels))
        total = max(int(counts.sum()), 1)
        for action, label in enumerate(labels):
            rows.append(
                {
                    "split": split,
                    "action": action,
                    "label": label,
                    "count": int(counts[action]),
                    "frac": float(counts[action] / total),
                }
            )
    long = pd.DataFrame(rows)
    counts = long.pivot(index=["action", "label"], columns="split", values="count").reset_index()
    frac = long.pivot(index=["action", "label"], columns="split", values="frac").reset_index()
    return counts, frac


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--save-csv", action="store_true")
    args = parser.parse_args()

    counts, frac = action_distribution(args.data_dir)
    print("\nACTION DISTRIBUTION - COUNTS")
    print(counts.to_string(index=False))
    print("\nACTION DISTRIBUTION - FRACTIONS")
    print(frac.round(4).to_string(index=False))
    if args.save_csv:
        counts.to_csv(args.data_dir / "action_distribution_counts.csv", index=False)
        frac.to_csv(args.data_dir / "action_distribution_frac.csv", index=False)


if __name__ == "__main__":
    main()
