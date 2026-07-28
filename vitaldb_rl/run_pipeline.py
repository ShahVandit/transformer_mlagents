"""End-to-end pipeline for VitalDB anesthetic infusion offline RL."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
sys.path.insert(0, str(SRC))

import evaluate  # noqa: E402
import mdp  # noqa: E402
import models  # noqa: E402

WEIGHT_GRID = [
    (0.70, 0.20, 0.10),
    (0.60, 0.30, 0.10),
    (0.50, 0.40, 0.10),
    (0.40, 0.50, 0.10),
    (0.30, 0.60, 0.10),
    (0.45, 0.45, 0.10),
    (0.40, 0.40, 0.20),
]


def load_split(split: str) -> dict[str, np.ndarray]:
    return mdp.load_npz(DATA / f"transitions_{split}.npz")


def ensure_data_exists() -> None:
    missing = [split for split in ("train", "val", "test") if not (DATA / f"transitions_{split}.npz").exists()]
    if missing:
        raise FileNotFoundError(f"missing transition files for {missing}; run --stage data first")


def save_metrics(path: Path, metrics: dict[str, float]) -> None:
    pd.DataFrame([metrics]).to_csv(path, index=False)


def run_data(args) -> None:
    mdp.build_transition_dataset(
        limit=args.limit,
        workers=args.workers,
        refresh=args.refresh,
        history=args.history,
        step=args.step,
        horizon=args.horizon,
    )


def run_diagnostics() -> None:
    ensure_data_exists()
    frames = []
    for split in ("train", "val", "test"):
        df = evaluate.action_distribution(load_split(split))
        df.insert(0, "split", split)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(RESULTS / "action_distribution.csv", index=False)
    print(out.to_string(index=False))


def run_bc(args):
    ensure_data_exists()
    train, val = load_split("train"), load_split("val")
    model, metrics = models.train_bc(train, val, epochs=args.bc_epochs, batch_size=args.batch_size)
    save_metrics(RESULTS / "bc_metrics.csv", metrics)
    seq_dim, static_dim = train["seq"].shape[2], train["static"].shape[1]
    models.save_model(RESULTS / "bc_model.pt", model, "bc", {"seq_dim": int(seq_dim), "static_dim": int(static_dim)})
    print("[bc]", metrics)
    return model


def load_or_train_bc(args):
    train = load_split("train")
    path = RESULTS / "bc_model.pt"
    if path.exists():
        return models.load_model(path, "bc", train["seq"].shape[2], train["static"].shape[1])
    return run_bc(args)


def run_cql(args):
    ensure_data_exists()
    train, val = load_split("train"), load_split("val")
    rows = []
    for i, weights in enumerate(WEIGHT_GRID):
        q, metrics = models.train_cql(
            train,
            val,
            weights,
            epochs=args.cql_epochs,
            batch_size=args.batch_size,
            cql_alpha=args.cql_alpha,
        )
        name = f"cql_w{i}"
        row = {"policy": name, "w_map": weights[0], "w_bis": weights[1], "w_work": weights[2]}
        row.update(metrics)
        rows.append(row)
        models.save_model(RESULTS / f"{name}.pt", q, "q", {"weights": weights})
        print("[cql]", row)
    pd.DataFrame(rows).to_csv(RESULTS / "cql_metrics.csv", index=False)


def run_evaluate(args):
    ensure_data_exists()
    train, test = load_split("train"), load_split("test")
    bc = load_or_train_bc(args)
    policies = [("bc", (0.45, 0.45, 0.10), bc, "bc")]
    for i, weights in enumerate(WEIGHT_GRID):
        path = RESULTS / f"cql_w{i}.pt"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; run --stage cql first")
        q = models.load_model(path, "q", train["seq"].shape[2], train["static"].shape[1])
        policies.append((f"cql_w{i}", weights, q, "q"))

    rows = evaluate.evaluate_policy_rows(train, test, bc, policies, fqe_epochs=args.fqe_epochs)
    rows.to_csv(RESULTS / "pareto_frontier.csv", index=False)
    support_cols = ["policy", "action_match_logged", "support_prob_mean", "support_prob_p10", "support_frac_ge_0p05"]
    rows[support_cols].to_csv(RESULTS / "action_support.csv", index=False)
    evaluate.plot_pareto(rows, RESULTS / "pareto_frontier.png")
    print(rows.round(4).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["data", "diagnostics", "bc", "cql", "evaluate", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--history", type=int, default=30)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--bc-epochs", type=int, default=10)
    ap.add_argument("--cql-epochs", type=int, default=20)
    ap.add_argument("--fqe-epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--cql-alpha", type=float, default=0.5)
    args = ap.parse_args()

    if args.stage in ("data", "all"):
        run_data(args)
    if args.stage in ("diagnostics", "all"):
        run_diagnostics()
    if args.stage in ("bc", "all"):
        run_bc(args)
    if args.stage in ("cql", "all"):
        run_cql(args)
    if args.stage in ("evaluate", "all"):
        run_evaluate(args)


if __name__ == "__main__":
    main()

