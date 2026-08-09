from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
DATA = ROOT / "data_d3rlpy"
RESULTS = ROOT / "results" / "d3rlpy"
MODELS = RESULTS / "models"
sys.path.insert(0, str(SRC))

import data as D  # noqa: E402
import evaluate  # noqa: E402
import ope  # noqa: E402


ACTION_LABELS = ["bolus_q1", "bolus_q2", "bolus_q3", "bolus_q4"]
REWARD_COMPONENTS = ["glycemic_effectiveness", "low_burden", "hypo_safety"]
CLINICAL_POLICY_NAME = "cql_clinical"
CLINICAL_REWARD_NAME = "piecewise_cgm"
REWARD_VARIANTS = ["current", "piecewise", "tir_binary", "asymmetric", "smooth"]
TRANSITION_CAP_FRACTIONS = {"train": 0.80, "val": 0.10, "test": 0.10}
OUTCOME_COMPONENTS = [
    "glycemic_effectiveness",
    "low_burden",
    "hypo_safety",
    "tir",
    "tbr70",
    "tbr54",
    "tar180",
    "tar250",
    "bolus_units",
]


def _import_d3rlpy():
    try:
        import d3rlpy
        from d3rlpy.algos import DiscreteCQLConfig
        from d3rlpy.constants import ActionSpace
        from d3rlpy.dataset import MDPDataset
        from d3rlpy.metrics import (
            AverageValueEstimationEvaluator,
            DiscreteActionMatchEvaluator,
            TDErrorEvaluator,
        )
    except ImportError as exc:
        raise ImportError(
            "d3rlpy is not installed in this Python environment. "
            "Create/activate the d3rlpy env and run `python -m pip install -r metabonet_offline_rl/requirements.txt`."
        ) from exc
    return (
        d3rlpy,
        DiscreteCQLConfig,
        ActionSpace,
        MDPDataset,
        TDErrorEvaluator,
        AverageValueEstimationEvaluator,
        DiscreteActionMatchEvaluator,
    )


def _flat_observation(seq: np.ndarray, static: np.ndarray) -> np.ndarray:
    return np.concatenate([seq.reshape(-1), static], dtype=np.float32)


def _encode_quantile_actions(interval_dose: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(interval_dose, edges, right=True).astype(np.int64)


def _labels_from_edges(edges: np.ndarray) -> list[str]:
    if len(edges) != 3:
        return ACTION_LABELS
    return [
        f"bolus_le{edges[0]:.3f}",
        f"bolus_{edges[0]:.3f}_{edges[1]:.3f}",
        f"bolus_{edges[1]:.3f}_{edges[2]:.3f}",
        f"bolus_gt{edges[2]:.3f}",
    ]


def transition_caps(max_transitions: int | None) -> dict[str, int] | None:
    """Allocate a total transition budget without wasting training capacity."""
    if max_transitions is None:
        return None
    train_cap = int(max_transitions * TRANSITION_CAP_FRACTIONS["train"])
    val_cap = int(max_transitions * TRANSITION_CAP_FRACTIONS["val"])
    return {
        "train": train_cap,
        "val": val_cap,
        "test": max_transitions - train_cap - val_cap,
    }


def glucose_reward_variants(valid: np.ndarray) -> np.ndarray:
    """Return reference-style glucose rewards averaged over the future window."""
    piecewise = np.select(
        [valid < 40, valid < 70, valid <= 180, valid <= 300],
        [-1.0, -0.4, 1.0, -0.3], default=-0.6,
    )
    tir_binary = ((valid >= 70) & (valid <= 180)).astype(np.float32)
    asymmetric = np.select(
        [valid < 40, valid < 70, valid <= 180, valid <= 300],
        [-2.0, -1.0, 1.0, -0.3], default=-0.6,
    )
    smooth = 2.0 * np.exp(-((valid - 110.0) ** 2) / (2 * 45.0 ** 2)) - 1.0
    return np.asarray([
        float(np.mean(piecewise)),
        float(np.mean(tir_binary)),
        float(np.mean(asymmetric)),
        float(np.mean(smooth)),
    ], dtype=np.float32)


def _build_group_rows(
    group: pd.DataFrame,
    new_row_mask: np.ndarray,
    history_steps: int,
    action_steps: int,
    reward_delay_steps: int,
    reward_steps: int,
    stride_steps: int,
) -> list[dict]:
    cgm_raw = group.CGM.to_numpy(float)
    cgm_missing = ~np.isfinite(cgm_raw)
    cgm = np.where(cgm_missing, np.nan, cgm_raw)
    basal = group.basal.ffill().fillna(0).to_numpy(float)
    bolus = group.bolus.fillna(0).clip(lower=0).to_numpy(float)
    carbs = group.carbs.fillna(0).clip(lower=0).to_numpy(float)
    insulin = group.insulin.fillna(0).clip(lower=0).to_numpy(float)
    hour = group.date.dt.hour.to_numpy(float) + group.date.dt.minute.to_numpy(float) / 60.0
    features = D._sequence_features_from_arrays(np.where(cgm_missing, 150.0, cgm), basal, bolus, carbs, insulin, hour, cgm_missing)
    stat = D._static_features(group.iloc[0])
    vectors = D._engineered_matrix(
        cgm,
        basal,
        bolus,
        carbs,
        insulin,
        hour,
        (bolus > 0).astype(float),
        (carbs > 0).astype(float),
        stat,
    )
    dates = group.date.to_numpy()
    source_file = group.source_file.iloc[0]
    subject_id = group.id.iloc[0]
    subject = f"{source_file}::{subject_id}"
    split = D.stable_split(subject, bool(group.is_test.astype(bool).any()))
    max_forward_steps = max(action_steps, reward_delay_steps + reward_steps)
    rows: list[dict] = []

    for t in range(history_steps - 1, len(group) - max_forward_steps, stride_steps):
        if not new_row_mask[t]:
            continue
        action_end = t + action_steps
        reward_start = t + reward_delay_steps + 1
        reward_end = t + reward_delay_steps + reward_steps
        action_minutes = (dates[action_end] - dates[t]) / np.timedelta64(1, "m")
        reward_minutes = (dates[reward_end] - dates[t]) / np.timedelta64(1, "m")
        if not 20 <= float(action_minutes) <= 45:
            continue
        if not 90 <= float(reward_minutes) <= 150:
            continue
        future_cgm = cgm[reward_start: reward_end + 1]
        if not np.isfinite(future_cgm).any() or not np.isfinite(cgm[t]):
            continue

        valid = future_cgm[np.isfinite(future_cgm)]
        tir = float(np.mean((valid >= 70) & (valid <= 180)))
        tbr70 = float(np.mean(valid < 70))
        tbr54 = float(np.mean(valid < 54))
        tar180 = float(np.mean(valid > 180))
        tar250 = float(np.mean(valid > 250))
        interval_bolus = float(np.nansum(bolus[t + 1: action_end + 1]))

        glycemic_effectiveness = -float(np.mean(np.maximum(valid - 180.0, 0.0) / 70.0)) - 2.0 * tar250
        low_burden = -interval_bolus
        hypo_safety = -(tbr70 + 3.0 * tbr54)
        reference_rewards = glucose_reward_variants(valid)
        rows.append(
            {
                "split": split,
                "source_file": source_file,
                "id": subject_id,
                "date": group.date.iloc[t],
                "observation": _flat_observation(D._encode_window(features, t, history_steps), vectors[t]),
                "interval_bolus": interval_bolus,
                "reward_components": np.asarray([glycemic_effectiveness, low_burden, hypo_safety], dtype=np.float32),
                "reward_variants": reference_rewards,
                "outcome_components": np.asarray(
                    [glycemic_effectiveness, low_burden, hypo_safety, tir, tbr70, tbr54, tar180, tar250, interval_bolus],
                    dtype=np.float32,
                ),
            }
        )
    return rows


def build_ordered_dataset(
    parquet: Path,
    output_dir: Path,
    download_url: str | None,
    batch_size: int,
    max_transitions: int | None,
    max_subject_transitions: int,
    history_steps: int,
    action_steps: int,
    reward_delay_steps: int,
    reward_steps: int,
    stride_steps: int,
) -> dict[str, int]:
    D.download_if_needed(parquet, download_url)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = ds.dataset(parquet, format="parquet")
    missing = set(D.REQUIRED_COLUMNS) - set(dataset.schema.names)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    columns = sorted(set(D.REQUIRED_COLUMNS) | (set(D.OPTIONAL_COLUMNS) & set(dataset.schema.names)))
    scanner = dataset.scanner(columns=columns, batch_size=batch_size, use_threads=True)

    split_caps = transition_caps(max_transitions)
    split_rows: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    subject_counts: dict[str, int] = {}
    tails: dict[str, pd.DataFrame] = {}
    tail_steps = history_steps + max(action_steps, reward_delay_steps + reward_steps)
    started = time.perf_counter()

    for batch_idx, record_batch in enumerate(scanner.to_batches(), start=1):
        frame = D._prepare(record_batch.to_pandas())
        for (source, sid), group in frame.groupby(["source_file", "id"], sort=False):
            key = f"{source}::{sid}"
            old_tail = tails.get(key)
            if old_tail is not None:
                group = pd.concat([old_tail, group], ignore_index=True)
                new_mask = np.r_[np.zeros(len(old_tail), dtype=bool), np.ones(len(group) - len(old_tail), dtype=bool)]
            else:
                new_mask = np.ones(len(group), dtype=bool)
            tails[key] = group.tail(tail_steps).copy()
            if bool(group.subject_split_across_traintest.astype(bool).any()):
                continue
            remaining_subject = max_subject_transitions - subject_counts.get(key, 0)
            if remaining_subject <= 0:
                continue
            rows = _build_group_rows(
                group,
                new_mask,
                history_steps,
                action_steps,
                reward_delay_steps,
                reward_steps,
                stride_steps,
            )
            if not rows:
                continue
            split = rows[0]["split"]
            if split_caps is not None:
                remaining_split = split_caps[split] - len(split_rows[split])
                if remaining_split <= 0:
                    continue
                rows = rows[:remaining_split]
            rows = rows[:remaining_subject]
            if rows:
                rows[-1]["terminal"] = 1.0
                for row in rows[:-1]:
                    row["terminal"] = 0.0
                split_rows[split].extend(rows)
                subject_counts[key] = subject_counts.get(key, 0) + len(rows)
        if batch_idx == 1 or batch_idx % 20 == 0:
            counts_now = {split: len(rows) for split, rows in split_rows.items()}
            print(f"[d3rlpy:data] batch={batch_idx} transitions={counts_now} elapsed={time.perf_counter() - started:.1f}s", flush=True)
        if split_caps is not None and all(len(split_rows[s]) >= split_caps[s] for s in ["train", "val", "test"]):
            break

    train_bolus = np.asarray([row["interval_bolus"] for row in split_rows["train"]], dtype=np.float32)
    if len(train_bolus) == 0:
        raise RuntimeError("No train rows were built. Increase --max-transitions or check parquet columns.")
    edges = np.quantile(train_bolus, [0.25, 0.50, 0.75]).astype(np.float32)
    labels = _labels_from_edges(edges)

    counts: dict[str, int] = {}
    observation_dim = int(history_steps * (len(D.SEQ_FEATURES) + 1) + len(D.VECTOR_FEATURES))
    for split, rows in split_rows.items():
        observations = np.asarray([row["observation"] for row in rows], dtype=np.float32)
        if observations.size == 0:
            observations = np.empty((0, observation_dim), dtype=np.float32)
        interval_bolus = np.asarray([row["interval_bolus"] for row in rows], dtype=np.float32)
        actions = _encode_quantile_actions(interval_bolus, edges)
        reward_components = np.asarray([row["reward_components"] for row in rows], dtype=np.float32)
        if reward_components.size == 0:
            reward_components = np.empty((0, len(REWARD_COMPONENTS)), dtype=np.float32)
        reward_variants = np.asarray([row["reward_variants"] for row in rows], dtype=np.float32)
        if reward_variants.size == 0:
            reward_variants = np.empty((0, 4), dtype=np.float32)
        outcome_components = np.asarray([row["outcome_components"] for row in rows], dtype=np.float32)
        if outcome_components.size == 0:
            outcome_components = np.empty((0, len(OUTCOME_COMPONENTS)), dtype=np.float32)
        terminals = np.asarray([row.get("terminal", 0.0) for row in rows], dtype=np.float32)
        if len(terminals) and terminals[-1] == 0.0:
            terminals[-1] = 1.0
        np.savez_compressed(
            output_dir / f"d3rlpy_{split}.npz",
            observations=observations,
            actions=actions,
            reward_components=reward_components,
            reward_variants=reward_variants,
            outcome_components=outcome_components,
            terminals=terminals,
            interval_bolus=interval_bolus,
            n_actions=np.asarray(4, dtype=np.int64),
        )
        meta = pd.DataFrame([{k: row[k] for k in ["source_file", "id", "date", "split", "interval_bolus"]} for row in rows])
        if len(meta):
            meta["action"] = actions
            meta["label"] = [labels[int(a)] for a in actions]
        meta.to_parquet(output_dir / f"d3rlpy_metadata_{split}.parquet", index=False)
        counts[split] = int(len(rows))

    config = {
        "action_mode": "bolus4_quantile",
        "action_labels": labels,
        "bolus_quantile_edges": edges.tolist(),
        "observation_dim": observation_dim,
        "seq_features": D.SEQ_FEATURES,
        "static_features": D.VECTOR_FEATURES,
        "history_steps": history_steps,
        "action_steps": action_steps,
        "reward_delay_steps": reward_delay_steps,
        "reward_steps": reward_steps,
        "stride_steps": stride_steps,
        "reward_components": REWARD_COMPONENTS,
        "reward_variants": ["piecewise", "tir_binary", "asymmetric", "smooth"],
        "outcome_components": OUTCOME_COMPONENTS,
        "counts": counts,
    }
    (output_dir / "d3rlpy_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[d3rlpy:data] transitions={counts} caps={split_caps}", flush=True)
    print(f"[d3rlpy:data] train bolus quantile edges={edges.round(4).tolist()}", flush=True)
    return counts


def load_arrays(split: str) -> dict[str, np.ndarray]:
    path = DATA / f"d3rlpy_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run --stage data first.")
    payload = np.load(path)
    return {key: payload[key] for key in payload.files}


def load_config() -> dict:
    path = DATA / "d3rlpy_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run --stage data first.")
    return json.loads(path.read_text(encoding="utf-8"))


def scalar_reward(arrays: dict[str, np.ndarray], weights: tuple[float, float, float]) -> np.ndarray:
    return (arrays["reward_components"] @ np.asarray(weights, dtype=np.float32)).astype(np.float32)


def scalar_reward_variant(
    arrays: dict[str, np.ndarray],
    weights: tuple[float, float, float],
    variant: str,
) -> np.ndarray:
    """Scalarize the normal components, optionally replacing glycemic reward."""
    if variant == "current":
        return scalar_reward(arrays, weights)
    if "reward_variants" not in arrays:
        raise ValueError("Dataset lacks reward_variants; rerun --stage data.")
    variant_index = {"piecewise": 0, "tir_binary": 1, "asymmetric": 2, "smooth": 3}[variant]
    components = arrays["reward_components"].copy()
    components[:, 0] = arrays["reward_variants"][:, variant_index]
    return (components @ np.asarray(weights, dtype=np.float32)).astype(np.float32)


def clinical_piecewise_reward(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """ICU-reference-style CGM reward for the delayed outcome window.

    The data builder stores this as the first reference reward variant:
    <40: -1.0, 40-70: -0.4, 70-180: +1.0, 180-300: -0.3, >300: -0.6.
    """
    if "reward_variants" not in arrays:
        raise ValueError("Dataset lacks reference rewards; rerun --stage data.")
    return arrays["reward_variants"][:, 0].astype(np.float32)


def logged_reward_summary(
    train_reward: np.ndarray,
    val_reward: np.ndarray,
    train_dataset,
    val_dataset,
) -> dict[str, float]:
    train_returns = [episode.compute_return() for episode in train_dataset.episodes]
    val_returns = [episode.compute_return() for episode in val_dataset.episodes]
    return {
        "logged_train_avg_step_reward": float(np.mean(train_reward)) if len(train_reward) else float("nan"),
        "logged_val_avg_step_reward": float(np.mean(val_reward)) if len(val_reward) else float("nan"),
        "logged_train_avg_episode_return": float(np.mean(train_returns)) if train_returns else float("nan"),
        "logged_val_avg_episode_return": float(np.mean(val_returns)) if val_returns else float("nan"),
    }


def make_mdp_dataset(arrays: dict[str, np.ndarray], reward: np.ndarray):
    _, _, ActionSpace, MDPDataset, *_ = _import_d3rlpy()
    return MDPDataset(
        observations=arrays["observations"].astype(np.float32),
        actions=arrays["actions"].astype(np.int64),
        rewards=reward.astype(np.float32),
        terminals=arrays["terminals"].astype(np.float32),
        action_space=ActionSpace.DISCRETE,
        action_size=4,
    )


def action_distribution(actions: np.ndarray, labels: list[str]) -> pd.DataFrame:
    counts = np.bincount(actions.astype(int), minlength=len(labels))
    total = max(int(counts.sum()), 1)
    return pd.DataFrame(
        {
            "action": np.arange(len(labels)),
            "label": labels,
            "count": counts,
            "frac": counts / total,
            "percent": 100.0 * counts / total,
        }
    )


def run_diagnostics() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    labels = load_config()["action_labels"]
    rows = []
    for split in ["train", "val", "test"]:
        dist = action_distribution(load_arrays(split)["actions"], labels)
        dist.insert(0, "split", split)
        rows.append(dist)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(RESULTS / "action_distribution.csv", index=False)
    print("\nD3RLPY ACTION DISTRIBUTION - PERCENT")
    print(out.pivot(index=["action", "label"], columns="split", values="percent").reset_index().round(2).to_string(index=False))
    print("\nD3RLPY ACTION DISTRIBUTION - COUNTS")
    print(out.pivot(index=["action", "label"], columns="split", values="count").reset_index().to_string(index=False))


def train_policies(args) -> None:
    (
        d3rlpy,
        DiscreteCQLConfig,
        _,
        _,
        TDErrorEvaluator,
        AverageValueEstimationEvaluator,
        DiscreteActionMatchEvaluator,
    ) = _import_d3rlpy()
    train = load_arrays("train")
    val = load_arrays("val")
    MODELS.mkdir(parents=True, exist_ok=True)
    model_path = MODELS / f"{CLINICAL_POLICY_NAME}.d3"
    if args.skip_existing and model_path.exists():
        print(f"\n[d3rlpy:train] skip existing policy={CLINICAL_POLICY_NAME} path={model_path}", flush=True)
        return
    train_reward = clinical_piecewise_reward(train)
    val_reward = clinical_piecewise_reward(val)
    dataset = make_mdp_dataset(train, train_reward)
    val_dataset = make_mdp_dataset(val, val_reward)
    steps_per_epoch = max(1, min(args.n_steps_per_epoch, args.n_steps))
    algo = DiscreteCQLConfig(
        learning_rate=args.learning_rate,
        batch_size=args.train_batch_size,
        gamma=args.gamma,
        alpha=args.cql_alpha,
        target_update_interval=args.target_update_interval,
    ).create(device=args.device)
    reward_summary = logged_reward_summary(train_reward, val_reward, dataset, val_dataset)
    print(f"\n[d3rlpy:train] policy={CLINICAL_POLICY_NAME} reward={CLINICAL_REWARD_NAME} steps={args.n_steps} device={args.device}", flush=True)
    print(
        "[d3rlpy:train] logged reward "
        f"train_step={reward_summary['logged_train_avg_step_reward']:.4f} "
        f"val_step={reward_summary['logged_val_avg_step_reward']:.4f} "
        f"val_episode_return={reward_summary['logged_val_avg_episode_return']:.4f}",
        flush=True,
    )
    history = algo.fit(
        dataset,
        n_steps=args.n_steps,
        n_steps_per_epoch=steps_per_epoch,
        experiment_name=f"d3rlpy_{CLINICAL_POLICY_NAME}",
        with_timestamp=False,
        show_progress=True,
        save_interval=max(1, args.n_steps // steps_per_epoch + 1),
        evaluators={
            "td_error_val": TDErrorEvaluator(val_dataset.episodes),
            "avg_value_val": AverageValueEstimationEvaluator(val_dataset.episodes),
            "action_match_val": DiscreteActionMatchEvaluator(val_dataset.episodes),
        },
    )
    algo.save(str(model_path))
    row = {"policy": CLINICAL_POLICY_NAME, "reward": CLINICAL_REWARD_NAME, "model_path": str(model_path)}
    row.update(reward_summary)
    if history:
        row.update({f"last_{k}": float(v) for k, v in history[-1][1].items()})
    pd.DataFrame([row]).to_csv(RESULTS / "training_metrics.csv", index=False)


def _parse_grid(text: str, cast):
    return [cast(value.strip()) for value in text.split(",") if value.strip()]


def tune_hyperparameters(args) -> None:
    """Run short validation-only CQL pilots before expensive full training."""
    (
        _,
        DiscreteCQLConfig,
        _,
        _,
        TDErrorEvaluator,
        AverageValueEstimationEvaluator,
        DiscreteActionMatchEvaluator,
    ) = _import_d3rlpy()
    train = load_arrays("train")
    val = load_arrays("val")
    train_reward = clinical_piecewise_reward(train)
    val_reward = clinical_piecewise_reward(val)
    train_dataset = make_mdp_dataset(train, train_reward)
    val_dataset = make_mdp_dataset(val, val_reward)

    grid = itertools.product(
        _parse_grid(args.tune_gammas, float),
        _parse_grid(args.tune_learning_rates, float),
        _parse_grid(args.tune_cql_alphas, float),
        _parse_grid(args.tune_target_update_intervals, int),
    )
    rows = []
    for trial, (gamma, learning_rate, alpha, target_update_interval) in enumerate(grid, 1):
        print(
            f"[tune] trial={trial} policy={CLINICAL_POLICY_NAME} "
            f"gamma={gamma} lr={learning_rate} alpha={alpha} "
            f"target_update_interval={target_update_interval}",
            flush=True,
        )
        algo = DiscreteCQLConfig(
            learning_rate=learning_rate,
            batch_size=args.tune_batch_size,
            gamma=gamma,
            alpha=alpha,
            target_update_interval=target_update_interval,
        ).create(device=args.device)
        steps_per_epoch = max(1, min(args.tune_steps_per_epoch, args.tune_steps))
        history = algo.fit(
            train_dataset,
            n_steps=args.tune_steps,
            n_steps_per_epoch=steps_per_epoch,
            experiment_name=f"d3rlpy_tune_{CLINICAL_POLICY_NAME}_{trial}",
            with_timestamp=False,
            show_progress=True,
            save_interval=max(1, args.tune_steps // steps_per_epoch + 1),
            evaluators={
                "td_error_val": TDErrorEvaluator(val_dataset.episodes),
                "avg_value_val": AverageValueEstimationEvaluator(val_dataset.episodes),
                "action_match_val": DiscreteActionMatchEvaluator(val_dataset.episodes),
            },
        )
        trial_rows = []
        for epoch, metrics in history:
            row = {
                "trial": trial,
                "epoch": int(epoch),
                "step": int(epoch) * steps_per_epoch,
                "policy": CLINICAL_POLICY_NAME,
                "gamma": gamma,
                "learning_rate": learning_rate,
                "cql_alpha": alpha,
                "target_update_interval": target_update_interval,
                "batch_size": args.tune_batch_size,
            }
            row.update({key: float(value) for key, value in metrics.items()})
            trial_rows.append(row)
        rows.extend(trial_rows)
        last_row = trial_rows[-1] if trial_rows else {}
        print(
            f"[tune] result trial={trial} "
            f"last_td_error_val={last_row.get('td_error_val', float('nan')):.4f} "
            f"last_avg_value_val={last_row.get('avg_value_val', float('nan')):.4f} "
            f"last_action_match_val={last_row.get('action_match_val', float('nan')):.4f}",
            flush=True,
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    output = RESULTS / "hyperparameter_tuning_clinical.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    if not frame.empty and "td_error_val" in frame:
        best = frame.loc[frame["td_error_val"].idxmin()]
        print(
            f"[tune] best_by_td_error trial={int(best.trial)} "
            f"epoch={int(best.epoch)} step={int(best.step)} "
            f"gamma={best.gamma} lr={best.learning_rate} "
            f"alpha={best.cql_alpha} target_update_interval={int(best.target_update_interval)} "
            f"td_error_val={best.td_error_val:.4f}",
            flush=True,
        )
    print(f"[tune] wrote {output}", flush=True)


def saved_policy_paths() -> list[tuple[str, Path]]:
    return [(CLINICAL_POLICY_NAME, MODELS / f"{CLINICAL_POLICY_NAME}.d3")]


def existing_policy_paths() -> list[tuple[str, Path]]:
    paths = [(name, path) for name, path in saved_policy_paths() if path.exists()]
    if not paths:
        raise FileNotFoundError(f"No d3rlpy models found in {MODELS}. Run --stage train first.")
    return paths


def load_policy(path: Path, device: str):
    d3rlpy, *_ = _import_d3rlpy()
    if not path.exists():
        raise FileNotFoundError(f"Missing model {path}. Run --stage train first.")
    return d3rlpy.load_learnable(str(path), device=device)


def run_infer(args) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    config = load_config()
    labels = config["action_labels"]
    test = load_arrays("test")
    logged = test["actions"].astype(int)
    logged_dist = action_distribution(logged, labels)
    logged_dist.to_csv(RESULTS / "test_logged_action_distribution.csv", index=False)
    summaries = []
    pred_dists = []
    cm_rows = []
    print("\nTEST LOGGED ACTION DISTRIBUTION")
    print(logged_dist.round(4).to_string(index=False))
    for policy_name, path in existing_policy_paths():
        algo = load_policy(path, args.device)
        pred = algo.predict(test["observations"].astype(np.float32)).astype(int).reshape(-1)
        pred_dist = action_distribution(pred, labels)
        pred_dist.insert(0, "policy", policy_name)
        pred_dists.append(pred_dist)
        cm = confusion_matrix(logged, pred, labels=list(range(len(labels))))
        for true_idx, label in enumerate(labels):
            row = {"policy": policy_name, "true_action": true_idx, "true_label": label}
            row.update({f"pred_{labels[pred_idx]}": int(cm[true_idx, pred_idx]) for pred_idx in range(len(labels))})
            cm_rows.append(row)
        counts = np.bincount(pred, minlength=len(labels))
        total = max(int(counts.sum()), 1)
        summary = {
            "policy": policy_name,
            "accuracy_logged": float(accuracy_score(logged, pred)),
            "balanced_accuracy_logged": float(balanced_accuracy_score(logged, pred)),
            "macro_f1_logged": float(f1_score(logged, pred, average="macro", zero_division=0)),
            "unique_actions": int(np.count_nonzero(counts)),
            "max_action_frac": float(counts.max() / total),
        }
        for idx, label in enumerate(labels):
            summary[f"pred_{label}_count"] = int(counts[idx])
            summary[f"pred_{label}_percent"] = float(100 * counts[idx] / total)
        summaries.append(summary)
    summary_frame = pd.DataFrame(summaries)
    pred_frame = pd.concat(pred_dists, ignore_index=True)
    cm_frame = pd.DataFrame(cm_rows)
    summary_frame.to_csv(RESULTS / "policy_test_inference.csv", index=False)
    pred_frame.to_csv(RESULTS / "policy_predicted_action_distribution.csv", index=False)
    cm_frame.to_csv(RESULTS / "policy_confusion_matrices.csv", index=False)
    print("\nPOLICY TEST INFERENCE SUMMARY")
    print(summary_frame.round(4).to_string(index=False))
    print("\nPOLICY PREDICTED ACTION DISTRIBUTION")
    print(pred_frame.round(4).to_string(index=False))


def next_observations(observations: np.ndarray, metadata: pd.DataFrame, terminals: np.ndarray) -> np.ndarray:
    """Construct next observations without crossing patient trajectories."""
    result = np.zeros_like(observations, dtype=np.float32)
    for indices in ope.episode_indices(metadata, len(observations), terminals):
        if len(indices) > 1:
            result[indices[:-1]] = observations[indices[1:]]
    return result


def run_fqe(args) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    config = load_config()
    labels = config["action_labels"]
    train = load_arrays("train")
    val = load_arrays("val")
    test = load_arrays("test")
    train_meta = pd.read_parquet(DATA / "d3rlpy_metadata_train.parquet")
    val_meta = pd.read_parquet(DATA / "d3rlpy_metadata_val.parquet")
    test_meta = pd.read_parquet(DATA / "d3rlpy_metadata_test.parquet")
    n_actions = len(labels)
    test_trajs = ope.episode_indices(test_meta, len(test["observations"]), test["terminals"])
    behavior_test, _ = ope.behavior_probabilities(
        train["observations"], train["actions"], test["observations"], n_actions
    )
    train_next = next_observations(train["observations"], train_meta, train["terminals"])
    test_next = next_observations(test["observations"], test_meta, test["terminals"])
    val_next = next_observations(val["observations"], val_meta, val["terminals"])
    policy_name, path = existing_policy_paths()[0]
    algo = load_policy(path, args.device)
    train_reward = clinical_piecewise_reward(train)
    val_reward = clinical_piecewise_reward(val)
    test_reward = clinical_piecewise_reward(test)
    cql_train_next_q = ope.cql_q_values(algo, train_next, n_actions)
    cql_val_next_q = ope.cql_q_values(algo, val_next, n_actions)
    cql_test_q = ope.cql_q_values(algo, test["observations"], n_actions)
    cql_test_next_q = ope.cql_q_values(algo, test_next, n_actions)
    target_train_next = ope.softmax_probabilities(cql_train_next_q, args.ope_temperature)
    target_val_next = ope.softmax_probabilities(cql_val_next_q, args.ope_temperature)
    target_test = ope.softmax_probabilities(cql_test_q, args.ope_temperature)
    target_test_next = ope.softmax_probabilities(cql_test_next_q, args.ope_temperature)
    pred_test = cql_test_q.argmax(axis=1).astype(int)

    steps_per_epoch = max(1, min(args.fqe_steps_per_epoch, args.fqe_steps))
    print(
        f"\n[fqe] policy={policy_name} reward={CLINICAL_REWARD_NAME} "
        f"steps={args.fqe_steps} target=softmax_q(tau={args.ope_temperature})",
        flush=True,
    )
    fqe, history = ope.fit_softmax_fqe(
        train["observations"],
        train["actions"],
        train_reward,
        train_next,
        train["terminals"],
        target_train_next,
        val["observations"],
        val["actions"],
        val_reward,
        val_next,
        val["terminals"],
        target_val_next,
        n_actions=n_actions,
        device=args.device,
        n_steps=args.fqe_steps,
        batch_size=args.fqe_batch_size,
        gamma=args.gamma,
        learning_rate=args.fqe_learning_rate,
        target_tau=args.fqe_target_tau,
        steps_per_epoch=steps_per_epoch,
    )
    metric_rows = []
    for epoch, metrics in enumerate(history, 1):
        row = {
            "policy": policy_name,
            "reward": CLINICAL_REWARD_NAME,
            "epoch": epoch,
            "gamma": args.gamma,
            "ope_temperature": args.ope_temperature,
            "fqe_batch_size": args.fqe_batch_size,
            "fqe_learning_rate": args.fqe_learning_rate,
            "fqe_target_tau": args.fqe_target_tau,
        }
        row.update({key: float(value) for key, value in metrics.items()})
        metric_rows.append(row)

    q_test = ope.fqe_q_values(fqe, test["observations"], args.device)
    q_val = ope.fqe_q_values(fqe, val["observations"], args.device)
    q_test_next = ope.fqe_q_values(fqe, test_next, args.device)
    q_val_next = ope.fqe_q_values(fqe, val_next, args.device)
    test_log_weights = ope.cumulative_log_ratios(test["actions"].astype(int), target_test, behavior_test, test_trajs, args.ope_ratio_clip)
    clinician_returns = ope.discounted_clinician_returns(test_reward, test_trajs, args.gamma)
    fqe_returns = np.asarray([float(np.sum(target_test[idx[0]] * q_test[idx[0]])) for idx in test_trajs if len(idx)])
    trajectory_positions = {id(trajectory): i for i, trajectory in enumerate(test_trajs)}

    def wis_estimate(sample):
        return ope.wis_value(test_reward, sample, [test_log_weights[trajectory_positions[id(x)]] for x in sample], args.gamma)

    def wdr_estimate(sample):
        return ope.wdr_value(test_reward, test["actions"], sample, [test_log_weights[trajectory_positions[id(x)]] for x in sample], q_test, target_test, args.gamma)

    clinician_point, clinician_lo, clinician_hi = ope.bootstrap_mean(clinician_returns, args.ope_bootstrap)
    fqe_point, fqe_lo, fqe_hi = ope.bootstrap_mean(fqe_returns, args.ope_bootstrap)
    wis_point, wis_lo, wis_hi = ope.bootstrap_estimator(wis_estimate, test_trajs, args.ope_bootstrap)
    wdr_point, wdr_lo, wdr_hi = ope.bootstrap_estimator(wdr_estimate, test_trajs, args.ope_bootstrap)
    val_residual = ope.fqe_bellman_residual(q_val, q_val_next, val["actions"], val_reward, val["terminals"], target_val_next, args.gamma)
    test_residual = ope.fqe_bellman_residual(q_test, q_test_next, test["actions"], test_reward, test["terminals"], target_test_next, args.gamma)
    counts = np.bincount(pred_test, minlength=n_actions)
    chosen_support = behavior_test[np.arange(len(pred_test)), pred_test]
    values = pd.DataFrame([{
        "policy": policy_name, "reward": CLINICAL_REWARD_NAME, "target_policy": "softmax_cql_q",
        "ope_temperature": args.ope_temperature,
        "fqe_value": fqe_point, "fqe_ci_low": fqe_lo, "fqe_ci_high": fqe_hi,
        "clinician_value": clinician_point, "clinician_ci_low": clinician_lo, "clinician_ci_high": clinician_hi,
        "wis_value": wis_point, "wis_ci_low": wis_lo, "wis_ci_high": wis_hi,
        "wdr_value": wdr_point, "wdr_ci_low": wdr_lo, "wdr_ci_high": wdr_hi,
        "fqe_bellman_mse_val": val_residual["bellman_mse"], "fqe_bellman_mae_val": val_residual["bellman_mae"],
        "fqe_bellman_mse_test": test_residual["bellman_mse"], "fqe_bellman_mae_test": test_residual["bellman_mae"],
        "behavior_support_mean": float(np.mean(chosen_support)), "behavior_support_p10": float(np.percentile(chosen_support, 10)),
        "behavior_support_frac_ge_0p05": float(np.mean(chosen_support >= 0.05)),
        "pred_mean_action_bin": float(np.mean(pred_test)), "pred_highest_insulin_percent": float(100 * counts[-1] / max(int(counts.sum()), 1)),
    }])
    values["wdr_vs_clinician"] = np.where(values.wdr_ci_low > values.clinician_value, "ABOVE", np.where(values.wdr_ci_high < values.clinician_value, "BELOW", "OVERLAPS"))
    values.to_csv(RESULTS / "ope_policy_comparison.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(RESULTS / "fqe_training_metrics.csv", index=False)
    print("\nLEARNED POLICY VS LOGGED CLINICIAN OPE")
    print(values.round(4).to_string(index=False))
    print(f"\n[fqe] trained 1 softmax-policy FQE model; wrote {RESULTS / 'fqe_training_metrics.csv'}", flush=True)


def add_pareto_analysis(values: pd.DataFrame) -> pd.DataFrame:
    values = values.copy()
    objectives = ["fqe_glycemic_effectiveness", "fqe_low_burden"]
    values["pareto"] = evaluate.pareto_mask(values, objectives)
    obj = values[objectives].to_numpy(float)
    dominated_by = []
    for i in range(len(values)):
        dominators = []
        if np.isfinite(obj[i]).all():
            for j in range(len(values)):
                if i == j or not np.isfinite(obj[j]).all():
                    continue
                if np.all(obj[j] >= obj[i]) and np.any(obj[j] > obj[i]):
                    dominators.append(str(values.iloc[j].policy))
        dominated_by.append(";".join(dominators))
    values["dominated_by"] = dominated_by
    values["frontier_order"] = np.nan
    values["delta_glycemic_from_prev_frontier"] = np.nan
    values["delta_low_burden_from_prev_frontier"] = np.nan
    values["burden_cost_per_control_gain"] = np.nan
    frontier = values[values["pareto"]].sort_values("fqe_glycemic_effectiveness")
    previous_idx = None
    for order, idx in enumerate(frontier.index, 1):
        values.loc[idx, "frontier_order"] = order
        if previous_idx is not None:
            dx = values.loc[idx, "fqe_glycemic_effectiveness"] - values.loc[previous_idx, "fqe_glycemic_effectiveness"]
            dy = values.loc[idx, "fqe_low_burden"] - values.loc[previous_idx, "fqe_low_burden"]
            values.loc[idx, "delta_glycemic_from_prev_frontier"] = dx
            values.loc[idx, "delta_low_burden_from_prev_frontier"] = dy
            values.loc[idx, "burden_cost_per_control_gain"] = -dy / dx if dx != 0 else np.nan
        previous_idx = idx
    return values


def observed_metrics(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    out = arrays["outcome_components"]
    if len(out) == 0:
        return {name: float("nan") for name in OUTCOME_COMPONENTS}
    return {name: float(np.mean(out[:, idx])) for idx, name in enumerate(OUTCOME_COMPONENTS)}


def ope_comparison_frame(values: pd.DataFrame) -> pd.DataFrame:
    """Readable policy-vs-logged-clinician OPE table for the scalarized reward."""
    required = {
        "policy", "clinician_return_scalarized", "clinician_return_scalarized_ci_low",
        "clinician_return_scalarized_ci_high", "fqe_bootstrap_scalarized",
        "fqe_scalarized_ci_low", "fqe_scalarized_ci_high", "wis_scalarized",
        "wis_scalarized_ci_low", "wis_scalarized_ci_high", "wdr_scalarized",
        "wdr_scalarized_ci_low", "wdr_scalarized_ci_high",
    }
    if not required.issubset(values.columns):
        return pd.DataFrame()
    frame = values[[
        "policy", "clinician_return_scalarized", "clinician_return_scalarized_ci_low",
        "clinician_return_scalarized_ci_high", "fqe_bootstrap_scalarized",
        "fqe_scalarized_ci_low", "fqe_scalarized_ci_high", "wis_scalarized",
        "wis_scalarized_ci_low", "wis_scalarized_ci_high", "wdr_scalarized",
        "wdr_scalarized_ci_low", "wdr_scalarized_ci_high", "fqe_bellman_mse_val_scalarized",
        "fqe_bellman_mae_val_scalarized", "fqe_bellman_mse_test_scalarized",
        "fqe_bellman_mae_test_scalarized", "behavior_support_mean", "behavior_support_p10",
        "behavior_support_frac_ge_0p05",
    ]].copy()
    frame.columns = [
        "policy", "clinician_value", "clinician_ci_low", "clinician_ci_high",
        "fqe_value", "fqe_ci_low", "fqe_ci_high", "wis_value", "wis_ci_low",
        "wis_ci_high", "wdr_value", "wdr_ci_low", "wdr_ci_high",
        "fqe_bellman_mse_val", "fqe_bellman_mae_val", "fqe_bellman_mse_test",
        "fqe_bellman_mae_test", "behavior_support_mean", "behavior_support_p10",
        "behavior_support_frac_ge_0p05",
    ]
    frame["wdr_vs_clinician"] = np.where(
        frame.wdr_ci_low > frame.clinician_value,
        "ABOVE",
        np.where(frame.wdr_ci_high < frame.clinician_value, "BELOW", "OVERLAPS"),
    )
    return frame


def plot_pareto(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    colors = np.where(frame["pareto"], "tab:red", "tab:blue")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(frame.fqe_glycemic_effectiveness, frame.fqe_low_burden, s=80, c=colors, alpha=0.85)
    for _, row in frame.iterrows():
        ax.annotate(row.policy, (row.fqe_glycemic_effectiveness, row.fqe_low_burden), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("FQE glycemic effectiveness (higher better)")
    ax.set_ylabel("FQE low treatment burden (higher better)")
    ax.set_title("d3rlpy CQL Pareto tradeoff")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_pareto_analysis(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    pareto = frame["pareto"].to_numpy(bool)
    scatter = ax.scatter(
        frame.fqe_glycemic_effectiveness,
        frame.fqe_low_burden,
        s=np.where(pareto, 120, 70),
        c=frame.w_glycemic_effectiveness,
        cmap="viridis",
        marker="o",
        edgecolors=np.where(pareto, "black", "none"),
        alpha=0.9,
    )
    frontier = frame[frame["pareto"]].sort_values("fqe_glycemic_effectiveness")
    if len(frontier) >= 2:
        ax.plot(frontier.fqe_glycemic_effectiveness, frontier.fqe_low_burden, color="black", linewidth=1.2, alpha=0.8)
    for _, row in frame.iterrows():
        label = f"{row.policy}\nhigh={row.pred_highest_insulin_percent:.1f}%"
        ax.annotate(label, (row.fqe_glycemic_effectiveness, row.fqe_low_burden), fontsize=7, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("FQE glycemic effectiveness (higher better)")
    ax.set_ylabel("FQE low insulin burden (higher better)")
    ax.set_title("Estimated Pareto frontier: glucose control vs insulin burden")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Glycemic-control reward weight")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_data(args) -> None:
    build_ordered_dataset(
        parquet=args.parquet,
        output_dir=DATA,
        download_url=args.download_url,
        batch_size=args.batch_size,
        max_transitions=args.max_transitions,
        max_subject_transitions=args.max_subject_transitions,
        history_steps=args.history_steps,
        action_steps=args.action_steps,
        reward_delay_steps=args.reward_delay_steps,
        reward_steps=args.reward_steps,
        stride_steps=args.stride_steps,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["data", "diagnostics", "tune", "train", "infer", "fqe", "all"], default="all")
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--download-url", default=None)
    parser.add_argument("--batch-size", type=int, default=500_000)
    parser.add_argument("--max-transitions", type=int, default=300_000)
    parser.add_argument("--max-subject-transitions", type=int, default=600)
    parser.add_argument("--history-steps", type=int, default=24)
    parser.add_argument("--action-steps", type=int, default=6)
    parser.add_argument("--reward-delay-steps", type=int, default=6)
    parser.add_argument("--reward-steps", type=int, default=18)
    parser.add_argument("--stride-steps", type=int, default=6)
    parser.add_argument("--n-steps", type=int, default=50_000)
    parser.add_argument("--n-steps-per-epoch", type=int, default=10_000)
    parser.add_argument("--fqe-steps", type=int, default=30_000)
    parser.add_argument("--fqe-steps-per-epoch", type=int, default=10_000)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--fqe-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--cql-alpha", type=float, default=1.0)
    parser.add_argument("--target-update-interval", type=int, default=1000)
    parser.add_argument("--tune-steps", type=int, default=2_000)
    parser.add_argument("--tune-steps-per-epoch", type=int, default=500)
    parser.add_argument("--tune-batch-size", type=int, default=1024)
    parser.add_argument("--tune-gammas", default="0.95,0.98,0.99")
    parser.add_argument("--tune-learning-rates", default="1e-4")
    parser.add_argument("--tune-cql-alphas", default="2.0")
    parser.add_argument("--tune-target-update-intervals", default="1000")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ope-temperature", type=float, default=0.5,
                        help="softmax temperature for the frozen CQL target policy used by FQE, WIS, and WDR")
    parser.add_argument("--ope-ratio-clip", type=float, default=5.0,
                        help="clip per-step log importance ratios")
    parser.add_argument("--ope-bootstrap", type=int, default=200,
                        help="patient-level bootstrap replicates for OPE intervals")
    parser.add_argument("--fqe-learning-rate", type=float, default=1e-3)
    parser.add_argument("--fqe-target-tau", type=float, default=0.005)
    parser.add_argument("--skip-existing", action="store_true", help="During --stage train/all, do not retrain policies whose .d3 file already exists.")
    args = parser.parse_args()

    if args.stage in ["data", "all"]:
        run_data(args)
    if args.stage in ["diagnostics", "all"]:
        run_diagnostics()
    if args.stage == "tune":
        tune_hyperparameters(args)
    if args.stage in ["train", "all"]:
        train_policies(args)
    if args.stage in ["infer", "all"]:
        run_infer(args)
    if args.stage == "fqe":
        run_fqe(args)


if __name__ == "__main__":
    main()
