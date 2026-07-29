from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
DATA = ROOT / "data"
RESULTS = ROOT / "results"
sys.path.insert(0, str(SRC))

import data as D  # noqa: E402
import evaluate  # noqa: E402
import models  # noqa: E402


def load_split(split: str) -> dict[str, np.ndarray]:
    return D.load_npz(DATA / f"transitions_{split}.npz")


def ensure_data() -> None:
    missing = [s for s in ["train", "val", "test"] if not (DATA / f"transitions_{s}.npz").exists()]
    if missing:
        raise FileNotFoundError(f"Missing transition files for {missing}. Run --stage data first.")


def load_config() -> dict:
    with (DATA / "mdp_config.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_data(args) -> None:
    D.build_transition_dataset(
        parquet=args.parquet,
        output_dir=DATA,
        download_url=args.download_url,
        batch_size=args.batch_size,
        max_transitions=args.max_transitions,
        min_split_transitions=args.min_split_transitions,
        max_subject_transitions=args.max_subject_transitions,
        history_steps=args.history_steps,
        horizon_steps=args.horizon_steps,
        stride_steps=args.stride_steps,
    )


def run_diagnostics() -> None:
    ensure_data()
    RESULTS.mkdir(parents=True, exist_ok=True)
    labels = load_config()["action_labels"]
    frames = []
    source_frames = []
    for split in ["train", "val", "test"]:
        frame = evaluate.action_distribution(load_split(split), labels)
        frame.insert(0, "split", split)
        frames.append(frame)
        meta_path = DATA / f"metadata_{split}.parquet"
        if meta_path.exists():
            source_frame = evaluate.source_action_distribution(pd.read_parquet(meta_path), labels)
            source_frame.insert(0, "split", split)
            source_frames.append(source_frame)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(RESULTS / "action_distribution.csv", index=False)
    pivot = out.pivot(index=["action", "label"], columns="split", values="frac").reset_index()
    counts = out.pivot(index=["action", "label"], columns="split", values="count").reset_index()
    pivot.to_csv(RESULTS / "action_distribution_pivot_frac.csv", index=False)
    counts.to_csv(RESULTS / "action_distribution_pivot_count.csv", index=False)
    print("\nACTION DISTRIBUTION BY SPLIT - COUNTS")
    print(counts.to_string(index=False))
    print("\nACTION DISTRIBUTION BY SPLIT - FRACTIONS")
    print(pivot.round(4).to_string(index=False))
    if source_frames:
        source_out = pd.concat(source_frames, ignore_index=True)
        source_out.to_csv(RESULTS / "source_action_distribution.csv", index=False)
        source_summary = (
            source_out.sort_values(["split", "source_file", "count"], ascending=[True, True, False])
            .groupby(["split", "source_file"], as_index=False)
            .head(3)
        )
        print("\nTOP ACTIONS BY SPLIT AND SOURCE")
        print(source_summary.round(4).to_string(index=False))


def run_bc(args):
    ensure_data()
    RESULTS.mkdir(parents=True, exist_ok=True)
    train, val = load_split("train"), load_split("val")
    model, metrics = models.train_bc(train, val, epochs=args.bc_epochs, batch_size=args.train_batch_size)
    pd.DataFrame([metrics]).to_csv(RESULTS / "bc_metrics.csv", index=False)
    seq_dim, static_dim, n_actions = train["seq"].shape[2], train["static"].shape[1], int(train["action"].max() + 1)
    models.save_model(RESULTS / "bc_transformer.pt", model, "bc", {"seq_dim": seq_dim, "static_dim": static_dim, "n_actions": n_actions})
    print("[bc]", metrics)
    return model


def load_bc(args):
    train = load_split("train")
    path = RESULTS / "bc_transformer.pt"
    if path.exists():
        return models.load_model(path, "bc", train["seq"].shape[2], train["static"].shape[1], int(train["action"].max() + 1))
    return run_bc(args)


def run_cql(args) -> None:
    ensure_data()
    RESULTS.mkdir(parents=True, exist_ok=True)
    train, val = load_split("train"), load_split("val")
    rows = []
    for name, weights in evaluate.OBJECTIVE_WEIGHTS.items():
        q, metrics = models.train_cql(
            train,
            val,
            weights,
            epochs=args.cql_epochs,
            batch_size=args.train_batch_size,
            cql_alpha=args.cql_alpha,
        )
        row = {"policy": f"cql_{name}", "w_hypo": weights[0], "w_hyper": weights[1], "w_burden": weights[2]}
        row.update(metrics)
        rows.append(row)
        models.save_model(RESULTS / f"cql_{name}.pt", q, "q", {"weights": weights})
        print("[cql]", row)
    pd.DataFrame(rows).to_csv(RESULTS / "cql_metrics.csv", index=False)


def load_cql_policies(train) -> list[tuple[str, object, str]]:
    policies = []
    n_actions = int(train["action"].max() + 1)
    for name in evaluate.OBJECTIVE_WEIGHTS:
        path = RESULTS / f"cql_{name}.pt"
        if path.exists():
            policies.append((f"cql_{name}", models.load_model(path, "q", train["seq"].shape[2], train["static"].shape[1], n_actions), "q"))
    return policies


def plot_pareto(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    colors = np.where(frame["pareto"], "tab:red", "tab:blue")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(frame.fqe_hypo_safety, frame.fqe_hyper_control, s=80, c=colors, alpha=0.85)
    for _, row in frame.iterrows():
        ax.annotate(row.policy, (row.fqe_hypo_safety, row.fqe_hyper_control), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("FQE hypoglycemia safety value (higher better)")
    ax.set_ylabel("FQE hyperglycemia control value (higher better)")
    ax.set_title("Policy tradeoffs; low-burden value in CSV")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_evaluate(args) -> None:
    ensure_data()
    RESULTS.mkdir(parents=True, exist_ok=True)
    train, test = load_split("train"), load_split("test")
    if len(test["action"]) == 0:
        raise RuntimeError("test split is empty; rerun --stage data with a larger --max-transitions")
    bc = load_bc(args)
    policies = [("bc", bc, "bc")] + load_cql_policies(train)
    rows = [evaluate.policy_value_row(train, test, bc, name, policy, ptype, args.fqe_epochs) for name, policy, ptype in policies]
    values = pd.DataFrame(rows)
    values["observed_test_tir"] = evaluate.observed_metrics(test)["tir"]
    values["pareto"] = evaluate.pareto_mask(values, ["fqe_hypo_safety", "fqe_hyper_control", "fqe_low_burden"])
    values.to_csv(RESULTS / "policy_values.csv", index=False)
    values[values.pareto].to_csv(RESULTS / "pareto_policies.csv", index=False)
    plot_pareto(values, RESULTS / "pareto_frontier.png")

    meta_path = DATA / "metadata_test.parquet"
    if meta_path.exists():
        group_values = evaluate.observed_group_values(pd.read_parquet(meta_path), test)
        group_values.to_csv(RESULTS / "observed_group_values.csv", index=False)
    write_report(values, test)
    print(values.round(4).to_string(index=False))


def write_report(values: pd.DataFrame, test: dict[str, np.ndarray]) -> None:
    observed = evaluate.observed_metrics(test)
    lines = [
        "# MetaboNet Offline MORL Report",
        "",
        "This run trains Transformer behavior-cloning and CQL policies from logged MetaboNet trajectories.",
        "Learned-policy values are FQE estimates, not direct ground truth.",
        "",
        "## Observed Logged Test Outcomes",
        f"- TIR: {observed['tir']:.4f}",
        f"- TBR<54: {observed['tbr54']:.4f}",
        f"- TAR>250: {observed['tar250']:.4f}",
        f"- Burden: {observed['burden']:.4f}",
        "",
        "## Pareto Policies",
    ]
    for _, row in values[values.pareto].iterrows():
        lines.append(
            f"- {row.policy}: hypo_safety={row.fqe_hypo_safety:.4f}, "
            f"hyper_control={row.fqe_hyper_control:.4f}, low_burden={row.fqe_low_burden:.4f}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "Ground truth exists for logged decisions and deployed groups. New learned policies require OPE and uncertainty-aware interpretation.",
        ]
    )
    (RESULTS / "final_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["data", "diagnostics", "bc", "cql", "evaluate", "all"], default="all")
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--download-url", default=None)
    parser.add_argument("--batch-size", type=int, default=500_000)
    parser.add_argument("--max-transitions", type=int, default=300_000)
    parser.add_argument("--min-split-transitions", type=int, default=1000)
    parser.add_argument("--max-subject-transitions", type=int, default=600)
    parser.add_argument("--history-steps", type=int, default=12)
    parser.add_argument("--horizon-steps", type=int, default=6)
    parser.add_argument("--stride-steps", type=int, default=6)
    parser.add_argument("--bc-epochs", type=int, default=5)
    parser.add_argument("--cql-epochs", type=int, default=8)
    parser.add_argument("--fqe-epochs", type=int, default=5)
    parser.add_argument("--train-batch-size", type=int, default=1024)
    parser.add_argument("--cql-alpha", type=float, default=0.5)
    args = parser.parse_args()

    if args.stage in ["data", "all"]:
        run_data(args)
    if args.stage in ["diagnostics", "all"]:
        run_diagnostics()
    if args.stage in ["bc", "all"]:
        run_bc(args)
    if args.stage in ["cql", "all"]:
        run_cql(args)
    if args.stage in ["evaluate", "all"]:
        run_evaluate(args)


if __name__ == "__main__":
    main()
