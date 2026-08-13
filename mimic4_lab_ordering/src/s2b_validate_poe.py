"""Validate whether past POE lab-order features add useful workflow signal.

This stage never matches a POE row to a specimen. It asks two narrower questions:

1. Are POE lab-order groups temporally associated with later target specimens?
2. Do strictly past-only POE features improve prediction of the next hourly
   specimen beyond the existing physiology state?

Output: reports/poe_feature_validation.md and .json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
import panels
from s3_build_mdp import add_sofa, build_state, poe_state_columns


def future_draw_within(draw, stay, hours):
    """Whether a draw occurs in [t, t + hours], without crossing a stay."""
    out = np.zeros(len(draw), dtype=bool)
    start = 0
    for i in range(1, len(draw) + 1):
        if i == len(draw) or stay[i] != stay[start]:
            v = draw[start:i].astype(np.int8)
            c = np.r_[0, np.cumsum(v)]
            for j in range(len(v)):
                hi = min(len(v), j + hours + 1)
                out[start + j] = (c[hi] - c[j]) > 0
            start = i
    return out


def metric_row(name, y, p):
    return {
        "model": name,
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def sample_indices(y, max_rows, seed):
    if len(y) <= max_rows:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(len(y), max_rows, replace=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train-rows", type=int, default=400_000)
    args = ap.parse_args()

    paths = {s: cfg.HOURLY_DIR / f"{s}.parquet" for s in ("train", "val")}
    for p in paths.values():
        if not p.exists():
            raise SystemExit(f"{p} not found; run stage 2 first")

    frames = {}
    for split, path in paths.items():
        df = pd.read_parquet(path).sort_values(["stay_id", "hour"]).reset_index(drop=True)
        required = set(poe_state_columns()) | {
            "poe_order_groups_current_hour", "poe_order_rows_current_hour"
        }
        missing = sorted(required - set(df.columns))
        if missing:
            raise SystemExit(f"{path} lacks POE columns: {missing}; rebuild stage 2")
        frames[split] = add_sofa(df)

    arrays = {}
    for split, df in frames.items():
        base, base_cols = build_state(df, include_poe=False)
        full, full_cols = build_state(df, include_poe=True)
        y = (panels.encode_frame(df) != 0).astype(np.int8)
        arrays[split] = (base, full, y)

    Xb, Xf, ytr = arrays["train"]
    fit_idx = sample_indices(ytr, args.max_train_rows, cfg.SEED)
    Xb_fit, Xf_fit, y_fit = Xb[fit_idx], Xf[fit_idx], ytr[fit_idx]
    base_model = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.08, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=cfg.SEED)
    full_model = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.08, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=cfg.SEED)
    print(f"fitting physiology-only model on {len(y_fit):,} rows")
    base_model.fit(Xb_fit, y_fit)
    print(f"fitting physiology + past POE model on {len(y_fit):,} rows")
    full_model.fit(Xf_fit, y_fit)

    Xb_val, Xf_val, yval = arrays["val"]
    metrics = [
        metric_row("physiology_only", yval, base_model.predict_proba(Xb_val)[:, 1]),
        metric_row("physiology_plus_past_poe", yval,
                   full_model.predict_proba(Xf_val)[:, 1]),
    ]
    delta = {
        k: metrics[1][k] - metrics[0][k]
        for k in ("roc_auc", "pr_auc", "brier")
    }

    val = frames["val"]
    draw = (panels.encode_frame(val) != 0)
    stay = val["stay_id"].to_numpy()
    current_groups = val["poe_order_groups_current_hour"].to_numpy() > 0
    temporal = []
    for hours in (0, 1, 2, 4, 6, 12):
        future = future_draw_within(draw, stay, hours)
        temporal.append({
            "window_hours": hours,
            "poe_group_hours": int(current_groups.sum()),
            "future_draw_rate_after_poe": float(future[current_groups].mean())
                if current_groups.any() else float("nan"),
            "future_draw_rate_without_poe": float(future[~current_groups].mean()),
        })

    coverage = {
        "val_hours": int(len(val)),
        "val_stays": int(val["stay_id"].nunique()),
        "draw_hour_rate": float(draw.mean()),
        "poe_group_hour_rate": float(current_groups.mean()),
        "stays_with_poe": float(
            val.groupby("stay_id")["poe_order_groups_current_hour"]
            .max().gt(0).mean()),
        "mean_poe_rows_per_group_hour": float(
            val.loc[current_groups, "poe_order_rows_current_hour"].mean())
            if current_groups.any() else float("nan"),
    }
    result = {
        "interpretation": (
            "POE rows are independent workflow context, not linked orders. "
            "Only strictly prior-hour POE features enter the MDP state."),
        "base_state_dim": int(Xb.shape[1]),
        "poe_state_dim": int(Xf.shape[1]),
        "poe_state_columns": poe_state_columns(),
        "coverage": coverage,
        "temporal_association": temporal,
        "predictive_metrics": metrics,
        "incremental_change": delta,
    }
    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = cfg.REPORTS_DIR / "poe_feature_validation.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# POE feature validation\n\n",
        "POE is treated as an independent workflow stream. No POE row is "
        "claimed to match a particular specimen. Only orders from hours before "
        "the current decision enter the MDP state.\n\n",
        "## Coverage\n\n",
        f"- Validation hours: {coverage['val_hours']:,}\n",
        f"- Validation stays: {coverage['val_stays']:,}\n",
        f"- Target-draw hour rate: {coverage['draw_hour_rate']:.2%}\n",
        f"- POE lab-order group hour rate: {coverage['poe_group_hour_rate']:.2%}\n",
        f"- Stays with at least one POE lab order: {coverage['stays_with_poe']:.2%}\n\n",
        "## Temporal association\n\n",
        "| window after POE hour | draw rate after POE | draw rate without POE |\n",
        "|---:|---:|---:|\n",
    ]
    for row in temporal:
        lines.append(
            f"| {row['window_hours']}h | {row['future_draw_rate_after_poe']:.2%} | "
            f"{row['future_draw_rate_without_poe']:.2%} |\n")
    lines += [
        "\n## Incremental prediction of the current-hour target specimen\n\n",
        "| model | ROC-AUC | PR-AUC | Brier |\n|---|---:|---:|---:|\n",
    ]
    for row in metrics:
        lines.append(
            f"| {row['model']} | {row['roc_auc']:.4f} | {row['pr_auc']:.4f} | "
            f"{row['brier']:.4f} |\n")
    lines.append(
        "\nInterpret cautiously: improvement shows POE captures clinician workflow, "
        "not that it improves autonomous clinical utility.\n")
    md_path = cfg.REPORTS_DIR / "poe_feature_validation.md"
    md_path.write_text("".join(lines), encoding="utf-8")
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
