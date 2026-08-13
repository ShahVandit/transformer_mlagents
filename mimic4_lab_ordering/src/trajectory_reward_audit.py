"""Audit joint-MDP rewards on representative patient trajectories.

This script deliberately distinguishes factual rewards from fixed-state replay.
The clinician row follows the logged trajectory and is factual. Changing an
action without regenerating subsequent forecast and lab-history state is only a
one-step diagnostic, not a counterfactual policy return.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
import objectives
import panels


def load_split(name, data_dir):
    path = data_dir / f"joint_{name}.npz"
    if not path.exists():
        raise SystemExit(f"{path} not found; run stage 3 first")
    data = np.load(path)
    return {key: data[key] for key in data.files}


def stay_slices(stay_ids):
    stay_ids = np.asarray(stay_ids)
    start = 0
    for i in range(1, len(stay_ids) + 1):
        if i == len(stay_ids) or stay_ids[i] != stay_ids[start]:
            yield slice(start, i)
            start = i


def discounted_sum(values, gamma=cfg.GAMMA):
    values = np.asarray(values, dtype=np.float64)
    return float(np.dot(values, gamma ** np.arange(len(values))))


def policy_history_burden(actions, hours, gamma=cfg.COST_DECAY_GAMMA):
    """Burden under the candidate policy's own prior draw history."""
    actions = np.asarray(actions) != 0
    hours = np.asarray(hours, dtype=np.float64)
    out = np.zeros(len(actions), dtype=np.float64)
    last_draw_hour = None
    for i, draw in enumerate(actions):
        if not draw:
            continue
        if last_draw_hour is None:
            out[i] = 1.0
        else:
            elapsed = max(1.0, hours[i] - last_draw_hour)
            out[i] = 1.0 + np.exp(-elapsed / gamma)
        last_draw_hour = hours[i]
    return out


def stay_summary(split):
    rows = []
    for sl in stay_slices(split["stay_id"]):
        action = split["action"][sl]
        draw = action != 0
        trigger = split["clinical_trigger"][sl]
        potential = split["information_potential"][sl]
        burden = split["draw_burden"][sl]
        n = len(action)
        rows.append({
            "stay_id": int(split["stay_id"][sl][0]),
            "hours": n,
            "clinician_draws": int(draw.sum()),
            "clinician_draws_per_day": float(draw.sum() / max(n / 24.0, 1e-6)),
            "clinician_utility": discounted_sum(trigger * draw),
            "always_draw_frozen_utility": discounted_sum(trigger),
            "clinician_burden": discounted_sum(burden * draw),
            "always_draw_frozen_burden": discounted_sum(burden),
            "events": int(split["event"][sl].sum()),
            "clinical_trigger_hours": int(trigger.sum()),
            "mean_information_potential": float(potential.mean()),
        })
    return pd.DataFrame(rows)


def choose_stays(summary):
    typical_score = (
        (summary["hours"] - summary["hours"].median()).abs()
        + 20.0 * (summary["clinician_draws_per_day"]
                  - summary["clinician_draws_per_day"].median()).abs()
    )
    selected = {"typical": int(summary.loc[typical_score.idxmin(), "stay_id"])}
    selected["high_information"] = int(
        summary.nlargest(1, "clinician_utility").iloc[0]["stay_id"])
    selected["high_testing"] = int(
        summary.nlargest(1, "clinician_draws_per_day").iloc[0]["stay_id"])
    long_enough = summary[summary["hours"] >= 24]
    selected["low_testing"] = int(
        long_enough.nsmallest(1, "clinician_draws_per_day").iloc[0]["stay_id"])
    with_events = summary[summary["events"] > 0]
    if len(with_events):
        selected["deterioration"] = int(
            with_events.sort_values(
                ["events", "clinician_utility"], ascending=False
            ).iloc[0]["stay_id"])
    return selected


def utility_components(states, state_cols, thresholds):
    lookup = {name: i for i, name in enumerate(state_cols)}
    out = {}
    for lab in panels.LABS:
        mean = states[:, lookup[f"mean_{lab}"]]
        last = states[:, lookup[f"last_{lab}"]]
        std = np.maximum(states[:, lookup[f"std_{lab}"]], cfg.FORECAST_MIN_STD)
        out[lab] = np.maximum(
            0.0, np.abs(mean - last) / std - float(thresholds[lab]))
    return out


def trajectory_rows(split, meta, labels):
    selected = set(labels.values())
    mask = np.isin(split["stay_id"], list(selected))
    indices = np.flatnonzero(mask)
    components = utility_components(
        split["state"][indices], meta["state_cols"], meta["utility_thresholds"])
    inverse_labels = {}
    for label, stay_id in labels.items():
        inverse_labels.setdefault(stay_id, []).append(label)

    rows = []
    for local, idx in enumerate(indices):
        row = {
            "selection": ",".join(inverse_labels[int(split["stay_id"][idx])]),
            "stay_id": int(split["stay_id"][idx]),
            "hour": int(split["hour"][idx]),
            "clinician_draw": int(split["action"][idx] != 0),
            "information_potential": float(split["information_potential"][idx]),
            "clinical_trigger": float(split["clinical_trigger"][idx]),
            "draw_burden_from_logged_state": float(split["draw_burden"][idx]),
            "factual_utility": float(split["reward"][idx, 0]),
            "factual_burden": float(split["reward"][idx, 1]),
            "event": int(split["event"][idx]),
            "future_event": int(split["future_event"][idx]),
        }
        for lab in panels.LABS:
            row[f"potential_{lab}"] = float(components[lab][local])
        rows.append(row)
    return pd.DataFrame(rows)


def policy_rows(split, selected):
    rows = []
    for sl in stay_slices(split["stay_id"]):
        stay_id = int(split["stay_id"][sl][0])
        if stay_id not in selected:
            continue
        hours = split["hour"][sl]
        trigger = split["clinical_trigger"][sl]
        logged_burden = split["draw_burden"][sl]
        policies = {
            "clinician": split["action"][sl] != 0,
            "never_draw": np.zeros(len(hours), dtype=bool),
            "always_draw": np.ones(len(hours), dtype=bool),
        }
        for name, draw in policies.items():
            recursive_burden = policy_history_burden(draw, hours)
            rows.append({
                "stay_id": stay_id,
                "policy": name,
                "draws": int(draw.sum()),
                "draws_per_day": float(draw.sum() / max(len(draw) / 24.0, 1e-6)),
                "frozen_state_utility": discounted_sum(trigger * draw),
                "frozen_state_burden": discounted_sum(logged_burden * draw),
                "policy_history_burden": discounted_sum(recursive_burden),
                "utility_is_counterfactual_return": name == "clinician",
            })
    return pd.DataFrame(rows)


def write_report(summary, trajectories, policies, labels, split_name):
    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_out = cfg.REPORTS_DIR / f"trajectory_reward_summary_{split_name}.csv"
    detail_out = cfg.REPORTS_DIR / f"trajectory_reward_hours_{split_name}.csv"
    policy_out = cfg.REPORTS_DIR / f"trajectory_reward_policies_{split_name}.csv"
    report_out = cfg.REPORTS_DIR / f"trajectory_reward_audit_{split_name}.md"
    summary.to_csv(summary_out, index=False)
    trajectories.to_csv(detail_out, index=False)
    policies.to_csv(policy_out, index=False)

    selected_summary = summary[summary["stay_id"].isin(labels.values())].copy()
    lines = [
        f"# Trajectory reward audit ({split_name})\n\n",
        "The clinician row is factual. `always_draw` and `never_draw` utility "
        "reuse logged clinical triggers, so they are frozen-trajectory "
        "diagnostics, not counterfactual trajectory returns. Altering a draw "
        "would change later `last_*`, `delta_*`, forecasts, and information "
        "potential. Burden can be replayed exactly from each candidate policy's "
        "own draw clock and is reported separately.\n\n",
        "## Selected stays\n\n",
        "| selection | stay | hours | clinician draws/day | events | trigger hours | clinician utility | frozen always utility |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    label_of = {}
    for label, stay_id in labels.items():
        label_of.setdefault(stay_id, []).append(label)
    for _, row in selected_summary.sort_values("stay_id").iterrows():
        label = ", ".join(label_of[int(row["stay_id"])])
        lines.append(
            f"| {label} | {int(row['stay_id'])} | {int(row['hours'])} | "
            f"{row['clinician_draws_per_day']:.3f} | {int(row['events'])} | "
            f"{int(row['clinical_trigger_hours'])} | "
            f"{row['clinician_utility']:.3f} | "
            f"{row['always_draw_frozen_utility']:.3f} |\n"
        )
    lines.extend([
        "\n## Policy diagnostics\n\n",
        "| stay | policy | draws/day | frozen utility | frozen burden | policy-history burden | valid trajectory utility? |\n",
        "|---:|---|---:|---:|---:|---:|---|\n",
    ])
    for _, row in policies.sort_values(["stay_id", "policy"]).iterrows():
        lines.append(
            f"| {int(row['stay_id'])} | {row['policy']} | "
            f"{row['draws_per_day']:.3f} | {row['frozen_state_utility']:.3f} | "
            f"{row['frozen_state_burden']:.3f} | "
            f"{row['policy_history_burden']:.3f} | "
            f"{'yes' if row['utility_is_counterfactual_return'] else '**no**'} |\n"
        )
    lines.append(
        "\nThe hour-level CSV contains the clinical trigger, the diagnostic "
        "information potential, the logged action, factual reward, and "
        "deterioration flags.\n"
    )
    report_out.write_text("".join(lines), encoding="utf-8")
    return report_out, summary_out, detail_out, policy_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--stay-ids", nargs="+", type=int, default=None)
    parser.add_argument(
        "--data-dir", type=Path, default=cfg.RL_DIR,
        help="directory containing joint_meta.json and joint_<split>.npz",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    split = load_split(args.split, data_dir)
    meta_path = data_dir / "joint_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} not found")
    meta = json.loads(meta_path.read_text())
    # The trajectory audit consumes raw reward only. Validate that contract
    # independently so a stale normalization sidecar cannot hide or block the
    # immediate-reward diagnosis.
    objectives.assert_rewards_current(
        split, norm_meta=None, name=f"joint_{args.split}.npz")
    scale = np.asarray(meta["reward_normalization"]["reward_scale"], dtype=float)
    expected_norm = split["reward"] / scale
    norm_error = float(np.max(np.abs(split["reward_norm"] - expected_norm)))
    if norm_error > 1e-3:
        print(
            "WARNING: reward_norm does not match joint_meta.json "
            f"(max abs diff={norm_error:.4g}); raw rewards remain auditable."
        )
    summary = stay_summary(split)
    if args.stay_ids:
        missing = sorted(set(args.stay_ids) - set(summary["stay_id"]))
        if missing:
            raise SystemExit(f"stay IDs not found in {args.split}: {missing}")
        labels = {f"requested_{i + 1}": stay_id
                  for i, stay_id in enumerate(args.stay_ids)}
    else:
        labels = choose_stays(summary)
    details = trajectory_rows(split, meta, labels)
    policies = policy_rows(split, set(labels.values()))
    outputs = write_report(summary, details, policies, labels, args.split)
    print("selected stays:")
    for label, stay_id in labels.items():
        print(f"  {label:18s} {stay_id}")
    for output in outputs:
        print(f"wrote -> {output}")


if __name__ == "__main__":
    main()
