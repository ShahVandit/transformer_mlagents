from __future__ import annotations

import numpy as np
import pandas as pd

import models


OBJECTIVE_WEIGHTS = {
    "safety_heavy": (0.70, 0.20, 0.10),
    "hyper_heavy": (0.20, 0.70, 0.10),
    "balanced": (0.45, 0.45, 0.10),
    "low_burden": (0.30, 0.30, 0.40),
}


def action_distribution(data: dict[str, np.ndarray], labels: list[str]) -> pd.DataFrame:
    counts = np.bincount(data["action"], minlength=len(labels))
    total = max(int(counts.sum()), 1)
    return pd.DataFrame({"action": range(len(labels)), "label": labels, "count": counts, "frac": counts / total})


def source_action_distribution(metadata: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    if metadata.empty:
        return pd.DataFrame(columns=["source_file", "action", "label", "count", "frac"])
    frame = metadata[["source_file", "action"]].copy()
    frame["action"] = pd.to_numeric(frame["action"], errors="coerce").fillna(-1).astype(int)
    counts = (
        frame.groupby(["source_file", "action"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = counts.groupby("source_file")["count"].transform("sum").clip(lower=1)
    counts["frac"] = counts["count"] / totals
    counts["label"] = counts["action"].map(lambda x: labels[x] if 0 <= x < len(labels) else "unknown")
    return counts[["source_file", "action", "label", "count", "frac"]].sort_values(
        ["source_file", "action"]
    )


def support_metrics(policy_actions: np.ndarray, bc_probs: np.ndarray, logged_actions: np.ndarray) -> dict[str, float]:
    if len(policy_actions) == 0:
        return {"action_match_logged": 0.0, "support_prob_mean": 0.0, "support_prob_p10": 0.0, "support_frac_ge_0p05": 0.0}
    chosen_prob = bc_probs[np.arange(len(policy_actions)), policy_actions]
    return {
        "action_match_logged": float(np.mean(policy_actions == logged_actions)),
        "support_prob_mean": float(np.mean(chosen_prob)),
        "support_prob_p10": float(np.percentile(chosen_prob, 10)),
        "support_frac_ge_0p05": float(np.mean(chosen_prob >= 0.05)),
    }


def observed_metrics(data: dict[str, np.ndarray]) -> dict[str, float]:
    out = data["outcome_components"]
    if len(out) == 0:
        return {"tir": np.nan, "tbr54": np.nan, "tar250": np.nan, "burden": np.nan}
    return {
        "tir": float(np.mean(out[:, 0])),
        "tbr54": float(np.mean(out[:, 1])),
        "tar250": float(np.mean(out[:, 2])),
        "burden": float(np.mean(out[:, 3])),
    }


def policy_value_row(train, eval_data, bc_model, name: str, policy, policy_type: str, fqe_epochs: int) -> dict[str, float | str]:
    row: dict[str, float | str] = {"policy": name}
    reward_components = {
        "fqe_hypo_safety": train["reward_components"][:, 0],
        "fqe_hyper_control": train["reward_components"][:, 1],
        "fqe_low_burden": train["reward_components"][:, 2],
    }
    for out_name, reward in reward_components.items():
        fqe = models.train_fqe(train, reward, policy, policy_type, epochs=fqe_epochs)
        row[out_name] = models.fqe_value(fqe, eval_data, policy, policy_type)
    actions = models.greedy_actions(policy, eval_data) if policy_type != "bc" else models.policy_probs(policy, eval_data).argmax(1)
    row.update(support_metrics(actions, models.policy_probs(bc_model, eval_data), eval_data["action"]))
    return row


def pareto_mask(frame: pd.DataFrame, objectives: list[str]) -> np.ndarray:
    values = frame[objectives].to_numpy(float)
    finite = np.isfinite(values).all(axis=1)
    mask = np.zeros(len(frame), dtype=bool)
    valid = np.flatnonzero(finite)
    for i in valid:
        dominated = np.all(values[valid] >= values[i], axis=1) & np.any(values[valid] > values[i], axis=1)
        if not dominated.any():
            mask[i] = True
    return mask


def observed_group_values(metadata: pd.DataFrame, data: dict[str, np.ndarray]) -> pd.DataFrame:
    if metadata.empty or len(data["outcome_components"]) == 0:
        return pd.DataFrame()
    out = metadata.reset_index(drop=True).copy()
    vals = data["outcome_components"]
    for i, col in enumerate(["tir", "tbr54", "tar250", "burden"]):
        out[col] = vals[:, i]
    rows = []
    for label in ["insulin_delivery_algorithm", "insulin_delivery_modality", "treatment_group"]:
        if label not in out:
            continue
        labeled = out[out[label].notna()]
        for (source, value), group in labeled.groupby(["source_file", label], dropna=False):
            rows.append(
                {
                    "label_type": label,
                    "source_file": source,
                    "policy_label": value,
                    "n": len(group),
                    "subjects": group["id"].nunique(),
                    "tir": group.tir.mean(),
                    "tbr54": group.tbr54.mean(),
                    "tar250": group.tar250.mean(),
                    "burden": group.burden.mean(),
                }
            )
    return pd.DataFrame(rows)
