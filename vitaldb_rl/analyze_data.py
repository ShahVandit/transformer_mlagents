"""Data audit for deciding the VitalDB offline RL formulation.

Run after:
    python run_pipeline.py --stage data --workers 8
"""
from __future__ import annotations

import json
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

import data as V  # noqa: E402
import evaluate  # noqa: E402
import mdp  # noqa: E402


def _save(df: pd.DataFrame, name: str) -> None:
    path = RESULTS / name
    df.to_csv(path, index=False)
    print(f"[write] {path}")


def _q(x, qs=(0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)) -> dict:
    x = pd.Series(x).dropna()
    if x.empty:
        return {f"p{int(q * 100)}": np.nan for q in qs}
    return {f"p{int(q * 100)}": float(x.quantile(q)) for q in qs}


def cohort_track_summary() -> pd.DataFrame:
    trks = V.get_trks()
    sets = V.track_case_sets(trks)
    rows = []
    for key, ids in sets.items():
        rows.append({"track": key, "case_count": len(ids), "vitaldb_name": V.TRACKS[key][0]})
    required = set.intersection(*(sets[k] for k in V.REQUIRED))
    all_channels = set.intersection(*(sets[k] for k in mdp.CHANNELS))
    rows.append({"track": "required_intersection", "case_count": len(required), "vitaldb_name": "+".join(V.REQUIRED)})
    rows.append({"track": "all_channel_intersection", "case_count": len(all_channels), "vitaldb_name": "+".join(mdp.CHANNELS)})
    return pd.DataFrame(rows)


def load_cached_caseids() -> list[int]:
    meta_path = DATA / "cohort.parquet"
    if not meta_path.exists():
        raise FileNotFoundError("Missing data/cohort.parquet. Run `python run_pipeline.py --stage data` first.")
    meta = pd.read_parquet(meta_path)
    return meta["caseid"].astype(int).tolist()


def trajectory_and_missingness(caseids: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    traj_rows = []
    miss_rows = []
    case_miss_rows = []

    total_values = {c: 0 for c in mdp.CHANNELS}
    total_missing = {c: 0 for c in mdp.CHANNELS}

    for cid in caseids:
        df = V.load_case(cid)
        traj_rows.append(
            {
                "caseid": cid,
                "minutes": len(df),
                "has_any_ppf": float(df["ppf_rate"].notna().any()),
                "has_any_rftn": float(df["rftn_rate"].notna().any()),
                "has_any_bis": float(df["bis"].notna().any()),
                "has_any_map": float(df["map"].notna().any()),
            }
        )
        row = {"caseid": cid, "minutes": len(df)}
        for col in mdp.CHANNELS:
            miss_frac = float(df[col].isna().mean())
            row[f"{col}_missing_frac"] = miss_frac
            row[f"{col}_present_frac"] = 1.0 - miss_frac
            total_values[col] += len(df)
            total_missing[col] += int(df[col].isna().sum())
        case_miss_rows.append(row)

    for col in mdp.CHANNELS:
        per_case = [r[f"{col}_missing_frac"] for r in case_miss_rows]
        miss_rows.append(
            {
                "signal": col,
                "overall_missing_frac": total_missing[col] / max(total_values[col], 1),
                "case_median_missing_frac": float(np.median(per_case)),
                "cases_gt_50pct_missing": int(np.sum(np.asarray(per_case) > 0.5)),
                "cases_gt_90pct_missing": int(np.sum(np.asarray(per_case) > 0.9)),
                "n_cases": len(caseids),
            }
        )

    traj = pd.DataFrame(traj_rows)
    traj_summary = pd.DataFrame([{**{"n_cases": len(traj)}, **_q(traj["minutes"])}])
    return traj_summary, pd.DataFrame(miss_rows), pd.DataFrame(case_miss_rows)


def load_transitions() -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    arrays = {}
    frames = []
    for split in ("train", "val", "test"):
        fp = DATA / f"transitions_{split}.npz"
        if not fp.exists():
            continue
        arr = mdp.load_npz(fp)
        arrays[split] = arr
        df = pd.DataFrame(
            {
                "split": split,
                "action": arr["action"],
                "map_reward": arr["reward_components"][:, 0],
                "bis_reward": arr["reward_components"][:, 1],
                "work_reward": arr["reward_components"][:, 2],
                "done": arr["done"],
            }
        )
        df["ppf_bin"] = df["action"] // 3
        df["rftn_bin"] = df["action"] % 3
        frames.append(df)
    if not frames:
        raise FileNotFoundError("Missing transition npz files. Run `python run_pipeline.py --stage data` first.")
    return pd.concat(frames, ignore_index=True), arrays


def transition_summaries(transitions: pd.DataFrame, arrays: dict[str, dict[str, np.ndarray]]) -> dict[str, pd.DataFrame]:
    action_dist = []
    for split, arr in arrays.items():
        df = evaluate.action_distribution(arr)
        df.insert(0, "split", split)
        action_dist.append(df)

    ppf = (
        transitions.groupby(["split", "ppf_bin"])
        .size()
        .reset_index(name="count")
        .assign(action_type=lambda d: d["ppf_bin"].map({0: "decrease", 1: "hold", 2: "increase"}))
    )
    rftn = (
        transitions.groupby(["split", "rftn_bin"])
        .size()
        .reset_index(name="count")
        .assign(action_type=lambda d: d["rftn_bin"].map({0: "decrease", 1: "hold", 2: "increase"}))
    )

    reward_summary = []
    for split, sub in transitions.groupby("split"):
        row = {"split": split, "n_transitions": len(sub)}
        for col in ["map_reward", "bis_reward", "work_reward"]:
            row.update({f"{col}_{k}": v for k, v in _q(sub[col]).items()})
            row[f"{col}_mean"] = float(sub[col].mean())
        reward_summary.append(row)

    corr = transitions[["map_reward", "bis_reward", "work_reward"]].corr().reset_index().rename(columns={"index": "reward"})
    return {
        "action_distribution.csv": pd.concat(action_dist, ignore_index=True),
        "propofol_action_distribution.csv": ppf,
        "remifentanil_action_distribution.csv": rftn,
        "reward_summary.csv": pd.DataFrame(reward_summary),
        "reward_correlations.csv": corr,
    }


def case_action_summary(transitions_path: Path = DATA / "transitions.parquet") -> pd.DataFrame:
    if not transitions_path.exists():
        return pd.DataFrame()
    t = pd.read_parquet(transitions_path)
    if t.empty:
        return pd.DataFrame()
    t["ppf_bin"] = t["action"] // 3
    t["rftn_bin"] = t["action"] % 3
    rows = []
    for (split, cid), sub in t.groupby(["split", "caseid"]):
        rows.append(
            {
                "split": split,
                "caseid": cid,
                "n_transitions": len(sub),
                "n_unique_actions": sub["action"].nunique(),
                "frac_hold_hold": float((sub["action"] == 4).mean()),
                "has_any_ppf_change": bool((sub["ppf_bin"] != 1).any()),
                "has_any_rftn_change": bool((sub["rftn_bin"] != 1).any()),
                "has_any_combined_change": bool((sub["action"] != 4).any()),
            }
        )
    return pd.DataFrame(rows)


def decision_recommendations(
    signal_missingness: pd.DataFrame,
    action_dist: pd.DataFrame,
    reward_corr: pd.DataFrame,
    case_actions: pd.DataFrame,
) -> pd.DataFrame:
    recs = []
    high_missing = signal_missingness.loc[signal_missingness["overall_missing_frac"] > 0.25, "signal"].tolist()
    if high_missing:
        recs.append(
            {
                "area": "encoder",
                "finding": f"Signals with >25% overall missingness: {', '.join(high_missing)}",
                "recommendation": "Use explicit masks; consider missingness-aware Transformer/GRU-D if missingness is also case-concentrated.",
            }
        )
    else:
        recs.append(
            {
                "area": "encoder",
                "finding": "Overall missingness is not high across selected signals.",
                "recommendation": "Start with TCN/Transformer-with-masks; GRU-D is not necessary unless per-case gaps are large.",
            }
        )

    all_actions = action_dist.groupby("action")["count"].sum()
    rare = all_actions[all_actions < 500].index.tolist()
    hold_frac = float(all_actions.get(4, 0) / max(all_actions.sum(), 1))
    if hold_frac > 0.8 or rare:
        recs.append(
            {
                "area": "action_space",
                "finding": f"hold/hold fraction={hold_frac:.3f}; rare actions={rare}",
                "recommendation": "Keep 9 actions only if rare actions are clinically important and supported; otherwise collapse to fewer actions or separate propofol/remifentanil policies.",
            }
        )
    else:
        recs.append(
            {
                "area": "action_space",
                "finding": f"9-action support looks usable; hold/hold fraction={hold_frac:.3f}.",
                "recommendation": "Keep the 3x3 action space.",
            }
        )

    if not case_actions.empty:
        changing_cases = float(case_actions["has_any_combined_change"].mean())
        recs.append(
            {
                "area": "rl_viability",
                "finding": f"{changing_cases:.3f} of cases have at least one non-hold action.",
                "recommendation": "Offline RL is more defensible if many cases contain changes; otherwise treat BC/action-support analysis as the main result.",
            }
        )

    corr = reward_corr.set_index("reward")
    vals = []
    for a, b in [("map_reward", "bis_reward"), ("map_reward", "work_reward"), ("bis_reward", "work_reward")]:
        if a in corr.index and b in corr.columns:
            vals.append(abs(float(corr.loc[a, b])))
    if vals and max(vals) > 0.8:
        recs.append(
            {
                "area": "pareto",
                "finding": f"Max absolute reward correlation={max(vals):.3f}.",
                "recommendation": "Pareto story may be weak if objectives move together; inspect scatter plots before claiming tradeoff.",
            }
        )
    else:
        recs.append(
            {
                "area": "pareto",
                "finding": "Reward components are not near-perfectly correlated.",
                "recommendation": "Pareto analysis is plausible, pending policy results.",
            }
        )
    return pd.DataFrame(recs)


def main() -> None:
    _save(cohort_track_summary(), "cohort_track_summary.csv")

    caseids = load_cached_caseids()
    traj, signal_missing, case_missing = trajectory_and_missingness(caseids)
    _save(traj, "trajectory_lengths.csv")
    _save(signal_missing, "signal_missingness.csv")
    _save(case_missing, "case_missingness.csv")

    transitions, arrays = load_transitions()
    summaries = transition_summaries(transitions, arrays)
    for name, df in summaries.items():
        _save(df, name)

    case_actions = case_action_summary()
    if not case_actions.empty:
        _save(case_actions, "action_by_case.csv")

    recs = decision_recommendations(
        signal_missing,
        summaries["action_distribution.csv"],
        summaries["reward_correlations.csv"],
        case_actions,
    )
    _save(recs, "data_audit_recommendations.csv")

    print("\nDATA AUDIT RECOMMENDATIONS")
    print(recs.to_string(index=False))


if __name__ == "__main__":
    main()

