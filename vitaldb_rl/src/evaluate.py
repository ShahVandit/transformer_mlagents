"""Evaluation and Pareto utilities for VitalDB offline RL."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mdp
import models

ACTION_LABELS = {
    0: "ppf_dec+rftn_dec",
    1: "ppf_dec+rftn_hold",
    2: "ppf_dec+rftn_inc",
    3: "ppf_hold+rftn_dec",
    4: "ppf_hold+rftn_hold",
    5: "ppf_hold+rftn_inc",
    6: "ppf_inc+rftn_dec",
    7: "ppf_inc+rftn_hold",
    8: "ppf_inc+rftn_inc",
}


def action_distribution(data: dict[str, np.ndarray]) -> pd.DataFrame:
    counts = np.bincount(data["action"], minlength=9)
    total = max(int(counts.sum()), 1)
    return pd.DataFrame(
        {
            "action": np.arange(9),
            "label": [ACTION_LABELS[i] for i in range(9)],
            "count": counts,
            "frac": counts / total,
        }
    )


def support_metrics(policy_actions: np.ndarray, bc_probs: np.ndarray, logged_actions: np.ndarray) -> dict[str, float]:
    chosen_prob = bc_probs[np.arange(len(policy_actions)), policy_actions]
    return {
        "action_match_logged": float(np.mean(policy_actions == logged_actions)),
        "support_prob_mean": float(np.mean(chosen_prob)),
        "support_prob_p10": float(np.percentile(chosen_prob, 10)),
        "support_frac_ge_0p05": float(np.mean(chosen_prob >= 0.05)),
    }


def pareto_indices(rows: list[dict], objectives: list[str]) -> list[int]:
    vals = np.asarray([[r[o] for o in objectives] for r in rows], dtype=float)
    keep = []
    for i in range(len(vals)):
        dominated = False
        for j in range(len(vals)):
            if i == j:
                continue
            if np.all(vals[j] >= vals[i]) and np.any(vals[j] > vals[i]):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def evaluate_policy_rows(
    train: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    bc_model,
    policies: list[tuple[str, tuple[float, float, float], object, str]],
    fqe_epochs: int,
) -> pd.DataFrame:
    rows = []
    bc_probs = models.policy_probs(bc_model, test)
    component_weights = {
        "map_value": (1.0, 0.0, 0.0),
        "bis_value": (0.0, 1.0, 0.0),
        "work_value": (0.0, 0.0, 1.0),
    }
    for name, weights, policy, ptype in policies:
        scalar_fqe = models.train_fqe(train, weights, policy, ptype, epochs=fqe_epochs)
        row = {
            "policy": name,
            "w_map": weights[0],
            "w_bis": weights[1],
            "w_work": weights[2],
            "fqe_scalar_value": models.fqe_value(scalar_fqe, test, policy, ptype),
        }
        for metric, cw in component_weights.items():
            fqe = models.train_fqe(train, cw, policy, ptype, epochs=fqe_epochs)
            row[metric] = models.fqe_value(fqe, test, policy, ptype)
        if ptype == "bc":
            acts = models.greedy_actions(policy, test)
        else:
            acts = models.greedy_actions(policy, test)
        row.update(support_metrics(acts, bc_probs, test["action"]))
        rows.append(row)

    nd = pareto_indices(rows, ["map_value", "bis_value", "work_value"])
    for i, row in enumerate(rows):
        row["pareto"] = i in nd
    return pd.DataFrame(rows)


def plot_pareto(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(exist_ok=True)
    plt.figure(figsize=(7, 5))
    colors = np.where(df["pareto"], "tab:red", "tab:blue")
    plt.scatter(-df["work_value"], df["map_value"], c=colors, s=70)
    for _, r in df.iterrows():
        plt.text(-r["work_value"], r["map_value"], r["policy"], fontsize=8)
    plt.xlabel("Estimated intervention burden penalty")
    plt.ylabel("Estimated MAP stability value")
    plt.title("Offline RL Pareto View")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()

