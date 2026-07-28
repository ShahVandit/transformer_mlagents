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
    model, metrics = models.train_bc(
        train,
        val,
        epochs=args.bc_epochs,
        batch_size=args.batch_size,
        encoder=args.encoder,
        moment_model=args.moment_model,
    )
    save_metrics(RESULTS / "bc_metrics.csv", metrics)
    seq_dim, static_dim = train["seq"].shape[2], train["static"].shape[1]
    models.save_model(
        RESULTS / f"bc_model_{args.encoder}.pt",
        model,
        "bc",
        {
            "seq_dim": int(seq_dim),
            "static_dim": int(static_dim),
            "encoder": args.encoder,
            "moment_model": args.moment_model,
        },
    )
    print("[bc]", metrics)
    return model


def load_or_train_bc(args):
    train = load_split("train")
    path = RESULTS / f"bc_model_{args.encoder}.pt"
    if path.exists():
        return models.load_model(
            path,
            "bc",
            train["seq"].shape[2],
            train["static"].shape[1],
            encoder=args.encoder,
            moment_model=args.moment_model,
        )
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
            encoder=args.encoder,
            moment_model=args.moment_model,
        )
        name = f"cql_{args.encoder}_w{i}"
        row = {"policy": name, "w_map": weights[0], "w_bis": weights[1], "w_work": weights[2]}
        row.update(metrics)
        rows.append(row)
        models.save_model(
            RESULTS / f"{name}.pt",
            q,
            "q",
            {"weights": weights, "encoder": args.encoder, "moment_model": args.moment_model},
        )
        print("[cql]", row)
    pd.DataFrame(rows).to_csv(RESULTS / "cql_metrics.csv", index=False)


def load_cql_policies(args, train):
    policies = []
    for i, weights in enumerate(WEIGHT_GRID):
        path = RESULTS / f"cql_{args.encoder}_w{i}.pt"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; run --stage cql first")
        q = models.load_model(
            path,
            "q",
            train["seq"].shape[2],
            train["static"].shape[1],
            encoder=args.encoder,
            moment_model=args.moment_model,
        )
        policies.append((f"cql_{args.encoder}_w{i}", weights, q, "q"))
    return policies


def run_select(args):
    ensure_data_exists()
    train, val = load_split("train"), load_split("val")
    bc = load_or_train_bc(args)
    rows = [
        evaluate.policy_row(
            train,
            val,
            bc,
            "bc",
            evaluate.DEFAULT_SELECTION_WEIGHTS,
            bc,
            "bc",
            args.fqe_epochs,
            encoder=args.encoder,
            moment_model=args.moment_model,
        )
    ]
    for name, weights, q, ptype in load_cql_policies(args, train):
        rows.append(
            evaluate.policy_row(
                train,
                val,
                bc,
                name,
                weights,
                q,
                ptype,
                args.fqe_epochs,
                encoder=args.encoder,
                moment_model=args.moment_model,
            )
        )
    selected = evaluate.select_policy(
        pd.DataFrame(rows),
        support_prob_p10_min=args.support_prob_p10_min,
        support_frac_ge_0p05_min=args.support_frac_ge_0p05_min,
        action_match_min=args.action_match_min,
        max_hold_frac=args.max_hold_frac,
    )
    selected["split"] = "val"
    for k, v in evaluate.observed_value(val, evaluate.DEFAULT_SELECTION_WEIGHTS).items():
        selected[f"clinician_{k}"] = v
    selected.to_csv(RESULTS / f"policy_selection_{args.encoder}.csv", index=False)
    print(selected.round(4).to_string(index=False))


def run_evaluate(args):
    ensure_data_exists()
    train, test = load_split("train"), load_split("test")
    bc = load_or_train_bc(args)
    selection_path = RESULTS / f"policy_selection_{args.encoder}.csv"
    if not selection_path.exists():
        raise FileNotFoundError(f"missing {selection_path}; run --stage select first")
    selection = pd.read_csv(selection_path)
    selected_names = selection.loc[
        (selection["selected"] == True) & (selection["policy"] != "bc"), "policy"
    ].tolist()
    policies = [("bc", evaluate.DEFAULT_SELECTION_WEIGHTS, bc, "bc")]
    for policy in load_cql_policies(args, train):
        if policy[0] in selected_names:
            policies.append(policy)
    if len(policies) == 1:
        print("[warn] no feasible selected CQL policy; final report includes BC only")

    rows = pd.DataFrame(
        [
            evaluate.policy_row(
                train,
                test,
                bc,
                name,
                weights,
                policy,
                ptype,
                args.fqe_epochs,
                encoder=args.encoder,
                moment_model=args.moment_model,
            )
            for name, weights, policy, ptype in policies
        ]
    )
    rows["split"] = "test"
    for k, v in evaluate.observed_value(test, evaluate.DEFAULT_SELECTION_WEIGHTS).items():
        rows[f"clinician_{k}"] = v
    rows.to_csv(RESULTS / f"final_test_report_{args.encoder}.csv", index=False)
    support_cols = ["policy", "action_match_logged", "support_prob_mean", "support_prob_p10", "support_frac_ge_0p05"]
    rows[support_cols].to_csv(RESULTS / f"action_support_{args.encoder}.csv", index=False)
    print(rows.round(4).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["data", "diagnostics", "bc", "cql", "select", "evaluate", "all"], default="all")
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
    ap.add_argument("--encoder", choices=["gru", "transformer", "moment"], default="gru")
    ap.add_argument("--moment-model", default="AutonLab/MOMENT-1-small")
    ap.add_argument("--support-prob-p10-min", type=float, default=0.01)
    ap.add_argument("--support-frac-ge-0p05-min", type=float, default=0.50)
    ap.add_argument("--action-match-min", type=float, default=0.45)
    ap.add_argument("--max-hold-frac", type=float, default=0.95)
    args = ap.parse_args()

    if args.stage in ("data", "all"):
        run_data(args)
    if args.stage in ("diagnostics", "all"):
        run_diagnostics()
    if args.stage in ("bc", "all"):
        run_bc(args)
    if args.stage in ("cql", "all"):
        run_cql(args)
    if args.stage in ("select", "all"):
        run_select(args)
    if args.stage in ("evaluate", "all"):
        run_evaluate(args)


if __name__ == "__main__":
    main()
