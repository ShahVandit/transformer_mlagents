"""Held-out evaluation and quality gates for the hourly forecaster."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config as cfg
import itemids as ids


def _metrics(actual, mean, std, last, population_mean):
    keep = np.isfinite(actual) & np.isfinite(mean) & np.isfinite(std)
    actual, mean = actual[keep], mean[keep]
    std = np.maximum(std[keep], cfg.FORECAST_MIN_STD)
    last = last[keep]
    if len(actual) == 0:
        return {"n": 0}

    err = actual - mean
    z = err / std
    out = {
        "n": int(len(actual)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "nll": float(np.mean(0.5 * (np.log(2 * np.pi * std * std) + z * z))),
        "correlation": float(np.corrcoef(actual, mean)[0, 1])
            if len(actual) > 1 and np.std(actual) > 0 and np.std(mean) > 0 else float("nan"),
        "standardized_residual_mean": float(np.mean(z)),
        "standardized_residual_sd": float(np.std(z)),
        "coverage_50": float(np.mean(np.abs(z) <= 0.67449)),
        "coverage_80": float(np.mean(np.abs(z) <= 1.28155)),
        "coverage_90": float(np.mean(np.abs(z) <= 1.64485)),
        "coverage_95": float(np.mean(np.abs(z) <= 1.95996)),
    }

    repeat = np.isfinite(last)
    out["n_repeat"] = int(repeat.sum())
    if repeat.any():
        locf_err = actual[repeat] - last[repeat]
        model_repeat_err = actual[repeat] - mean[repeat]
        locf_rmse = float(np.sqrt(np.mean(locf_err * locf_err)))
        model_rmse = float(np.sqrt(np.mean(model_repeat_err * model_repeat_err)))
        out["repeat_rmse"] = model_rmse
        out["locf_rmse"] = locf_rmse
        out["rmse_skill_vs_locf"] = 1.0 - model_rmse / max(locf_rmse, 1e-12)
        out["locf_mae"] = float(np.mean(np.abs(locf_err)))

        predicted_utility = np.abs(mean[repeat] - last[repeat]) / std[repeat]
        realized_change = np.abs(actual[repeat] - last[repeat])
        corr = spearmanr(predicted_utility, realized_change, nan_policy="omit")
        rho = getattr(corr, "statistic", getattr(corr, "correlation", corr[0]))
        out["utility_spearman"] = float(rho) if np.isfinite(rho) else 0.0
        threshold = np.quantile(realized_change, 0.75)
        high = realized_change >= threshold
        k = max(1, int(high.sum()))
        top = np.argpartition(predicted_utility, -k)[-k:]
        precision = float(high[top].mean())
        out["high_change_prevalence"] = float(high.mean())
        out["high_change_topk_precision"] = precision
        out["high_change_lift"] = precision / max(float(high.mean()), 1e-12)
    else:
        out.update({"repeat_rmse": float("nan"), "locf_rmse": float("nan"),
                    "rmse_skill_vs_locf": float("nan"), "locf_mae": float("nan"),
                    "utility_spearman": float("nan"), "high_change_lift": float("nan")})

    pop_err = actual - population_mean
    out["population_rmse"] = float(np.sqrt(np.mean(pop_err * pop_err)))
    out["population_mae"] = float(np.mean(np.abs(pop_err)))
    return out


def evaluate_hourly(path, population_means):
    cols = []
    for lab in ids.TARGET_LABS:
        cols += [f"obs_{lab}", f"mean_{lab}", f"std_{lab}", f"last_{lab}"]
    df = pd.read_parquet(path, columns=cols)
    return {
        lab: _metrics(
            df[f"obs_{lab}"].to_numpy(dtype=float),
            df[f"mean_{lab}"].to_numpy(dtype=float),
            df[f"std_{lab}"].to_numpy(dtype=float),
            df[f"last_{lab}"].to_numpy(dtype=float),
            float(population_means[lab]),
        )
        for lab in ids.TARGET_LABS
    }


def validation_gate(metrics):
    weighted_model = weighted_locf = 0.0
    total = 0
    beating = 0
    utility_corr = []
    coverages = []
    reasons = []
    for lab, row in metrics.items():
        n = row.get("n_repeat", 0)
        if n <= 0:
            reasons.append(f"{lab}: no repeat observations")
            continue
        weighted_model += n * row["repeat_rmse"] ** 2
        weighted_locf += n * row["locf_rmse"] ** 2
        total += n
        beating += int(row["rmse_skill_vs_locf"] > 0)
        utility_corr.append(row["utility_spearman"])
        coverages.append(row["coverage_90"])
    aggregate_skill = 1.0 - np.sqrt(weighted_model / max(total, 1)) \
        / max(np.sqrt(weighted_locf / max(total, 1)), 1e-12)
    median_utility = float(np.median(utility_corr)) if utility_corr else float("nan")
    mean_coverage = float(np.mean(coverages)) if coverages else float("nan")
    if aggregate_skill < cfg.FORECAST_MIN_AGGREGATE_RMSE_SKILL:
        reasons.append(f"aggregate RMSE skill {aggregate_skill:.3f} is below "
                       f"{cfg.FORECAST_MIN_AGGREGATE_RMSE_SKILL:.3f}")
    if beating < cfg.FORECAST_MIN_TARGETS_BEATING_LOCF:
        reasons.append(f"only {beating}/4 target labs beat LOCF")
    if median_utility < cfg.FORECAST_MIN_UTILITY_SPEARMAN:
        reasons.append(f"median utility Spearman {median_utility:.3f} is below "
                       f"{cfg.FORECAST_MIN_UTILITY_SPEARMAN:.3f}")
    if not (cfg.FORECAST_COVERAGE90_MIN <= mean_coverage <= cfg.FORECAST_COVERAGE90_MAX):
        reasons.append(f"mean 90% coverage {mean_coverage:.3f} outside "
                       f"[{cfg.FORECAST_COVERAGE90_MIN:.2f}, "
                       f"{cfg.FORECAST_COVERAGE90_MAX:.2f}]")
    return {
        "passed": not reasons,
        "aggregate_rmse_skill_vs_locf": float(aggregate_skill),
        "target_labs_beating_locf": int(beating),
        "median_utility_spearman": median_utility,
        "mean_coverage_90": mean_coverage,
        "reasons": reasons,
    }


def write_report(validation, test, tuning, output_stem):
    result = {"tuning": tuning, "validation": validation, "test": test}
    json_path = Path(str(output_stem) + ".json")
    md_path = Path(str(output_stem) + ".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = ["# Forecaster validation\n\n",
             "Predictions at hour `t` are generated before the observation at "
             "hour `t` is incorporated. Validation selected the forecaster; test "
             "was evaluated once after the validation gate passed.\n\n"]
    for split_name, block in (("Validation", validation), ("Locked test", test)):
        lines += [f"## {split_name}\n\n"]
        gate = block.get("gate")
        if gate:
            lines.append(f"Gate: **{'PASS' if gate['passed'] else 'FAIL'}**. "
                         f"Aggregate RMSE skill vs LOCF: "
                         f"{gate['aggregate_rmse_skill_vs_locf']:+.3f}; "
                         f"median utility Spearman: {gate['median_utility_spearman']:.3f}; "
                         f"mean 90% coverage: {gate['mean_coverage_90']:.3f}.\n\n")
        lines += ["| lab | n | RMSE | LOCF RMSE | skill | MAE | corr | 90% cov | utility rho | lift |\n",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"]
        for lab, row in block["metrics"].items():
            lines.append(
                f"| {lab} | {row.get('n', 0):,} | {row.get('repeat_rmse', float('nan')):.4f} | "
                f"{row.get('locf_rmse', float('nan')):.4f} | "
                f"{row.get('rmse_skill_vs_locf', float('nan')):+.3f} | "
                f"{row.get('mae', float('nan')):.4f} | "
                f"{row.get('correlation', float('nan')):.3f} | "
                f"{row.get('coverage_90', float('nan')):.3f} | "
                f"{row.get('utility_spearman', float('nan')):.3f} | "
                f"{row.get('high_change_lift', float('nan')):.2f}x |\n")
        lines.append("\n")
    md_path.write_text("".join(lines), encoding="utf-8")
    return md_path, json_path
