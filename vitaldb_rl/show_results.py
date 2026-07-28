"""Print compact summaries from the VitalDB RL result files."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def show_csv(name: str, n: int | None = None) -> None:
    path = RESULTS / name
    if not path.exists():
        print(f"[missing] {name}")
        return
    df = pd.read_csv(path)
    print(f"\n{name}")
    print("=" * len(name))
    print((df.head(n) if n else df).round(4).to_string(index=False))


def main() -> None:
    show_csv("action_distribution.csv")
    show_csv("bc_metrics.csv")
    show_csv("cql_metrics.csv")
    show_csv("pareto_frontier.csv")
    show_csv("action_support.csv")
    show_csv("pareto_frontier_gru.csv")
    show_csv("action_support_gru.csv")
    show_csv("pareto_frontier_moment.csv")
    show_csv("action_support_moment.csv")
    plot = RESULTS / "pareto_frontier.png"
    if plot.exists():
        print(f"\nplot: {plot}")


if __name__ == "__main__":
    main()
