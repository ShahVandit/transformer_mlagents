"""Lightweight off-policy evaluation utilities for the d3rlpy pipeline.

The estimators follow the ICU reference implementation, but operate on the
ordered MetaboNet arrays and use an epsilon-soft version of a d3rlpy policy.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import torch
from torch import nn


class SoftmaxFQE(nn.Module):
    """Small fitted-Q network for one fixed stochastic target policy."""

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, observations):
        return self.network(observations)


def episode_indices(metadata, n_rows: int, terminals=None) -> list[np.ndarray]:
    """Return patient trajectories, ordered by patient and timestamp."""
    if len(metadata) != n_rows:
        raise ValueError("metadata and transition arrays have different lengths")
    frame = metadata.reset_index(drop=True).copy()
    frame["_row"] = np.arange(len(frame))
    frame = frame.sort_values(["source_file", "id", "date", "_row"], kind="stable")
    terminal_flags = None if terminals is None else np.asarray(terminals).astype(bool)
    groups = []
    for _, group in frame.groupby(["source_file", "id"], sort=False, dropna=False):
        ordered = group["_row"].to_numpy(dtype=np.int64)
        if terminal_flags is None:
            groups.append(ordered)
            continue
        start = 0
        for position, row in enumerate(ordered):
            if terminal_flags[row]:
                groups.append(ordered[start:position + 1])
                start = position + 1
        if start < len(ordered):
            groups.append(ordered[start:])
    return groups


def behavior_probabilities(train_obs, train_actions, eval_obs, n_actions: int,
                           seed: int = 0, probability_floor: float = 1e-3):
    model = HistGradientBoostingClassifier(random_state=seed, max_iter=300, learning_rate=0.1)
    model.fit(train_obs, train_actions)
    predicted = model.predict_proba(eval_obs)
    probs = np.full((len(eval_obs), n_actions), probability_floor, dtype=np.float64)
    for col, action in enumerate(model.classes_.astype(int)):
        probs[:, action] = np.maximum(predicted[:, col], probability_floor)
    return probs / probs.sum(axis=1, keepdims=True), model


def softmax_probabilities(q_values, temperature: float = 0.5):
    """Target policy used consistently by FQE, WIS, and WDR."""
    temperature = max(float(temperature), 1e-6)
    logits = np.asarray(q_values, dtype=np.float64) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    return probs / probs.sum(axis=1, keepdims=True)


def cql_q_values(algo, observations, n_actions: int, batch_size: int = 8192):
    """Read all discrete CQL action-values from a frozen d3rlpy policy."""
    values = np.empty((len(observations), n_actions), dtype=np.float32)
    for start in range(0, len(observations), batch_size):
        obs = observations[start:start + batch_size].astype(np.float32)
        for action in range(n_actions):
            actions = np.full(len(obs), action, dtype=np.int64)
            values[start:start + len(obs), action] = algo.predict_value(obs, actions).reshape(-1)
    return values


def fqe_q_values(fqe, observations, device: str, batch_size: int = 8192):
    """Read all action-values from a trained SoftmaxFQE."""
    fqe.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            batch = torch.as_tensor(observations[start:start + batch_size], dtype=torch.float32, device=device)
            values.append(fqe(batch).cpu().numpy())
    return np.concatenate(values, axis=0) if values else np.empty((0, 0), dtype=np.float32)


def fit_softmax_fqe(
    train_observations,
    train_actions,
    train_rewards,
    train_next_observations,
    train_terminals,
    train_next_target_probs,
    val_observations,
    val_actions,
    val_rewards,
    val_next_observations,
    val_terminals,
    val_next_target_probs,
    n_actions: int,
    device: str,
    n_steps: int,
    batch_size: int,
    gamma: float,
    learning_rate: float,
    target_tau: float,
    steps_per_epoch: int,
    seed: int = 0,
):
    """Reference-style FQE using the same soft target policy as WIS and WDR."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    state_dim = int(train_observations.shape[1])
    fqe = SoftmaxFQE(state_dim, n_actions).to(device)
    target = SoftmaxFQE(state_dim, n_actions).to(device)
    target.load_state_dict(fqe.state_dict())
    optimizer = torch.optim.Adam(fqe.parameters(), lr=learning_rate)

    train_obs = torch.as_tensor(train_observations, dtype=torch.float32, device=device)
    train_actions_t = torch.as_tensor(train_actions, dtype=torch.long, device=device)
    train_rewards_t = torch.as_tensor(train_rewards, dtype=torch.float32, device=device)
    train_next_obs = torch.as_tensor(train_next_observations, dtype=torch.float32, device=device)
    train_terminals_t = torch.as_tensor(train_terminals, dtype=torch.float32, device=device)
    train_next_pi = torch.as_tensor(train_next_target_probs, dtype=torch.float32, device=device)

    val_obs = torch.as_tensor(val_observations, dtype=torch.float32, device=device)
    val_actions_t = torch.as_tensor(val_actions, dtype=torch.long, device=device)
    val_rewards_t = torch.as_tensor(val_rewards, dtype=torch.float32, device=device)
    val_next_obs = torch.as_tensor(val_next_observations, dtype=torch.float32, device=device)
    val_terminals_t = torch.as_tensor(val_terminals, dtype=torch.float32, device=device)
    val_next_pi = torch.as_tensor(val_next_target_probs, dtype=torch.float32, device=device)

    history = []
    total_loss = 0.0
    n_updates = 0
    for step in range(1, n_steps + 1):
        indices = torch.as_tensor(
            rng.integers(0, len(train_observations), size=min(batch_size, len(train_observations))),
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            next_q = target(train_next_obs[indices])
            next_v = (train_next_pi[indices] * next_q).sum(dim=1)
            td_target = train_rewards_t[indices] + gamma * (1.0 - train_terminals_t[indices]) * next_v
        q_sa = fqe(train_obs[indices]).gather(1, train_actions_t[indices].unsqueeze(1)).squeeze(1)
        loss = torch.mean((q_sa - td_target) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for online, delayed in zip(fqe.parameters(), target.parameters()):
                delayed.mul_(1.0 - target_tau).add_(online, alpha=target_tau)
        total_loss += float(loss.detach().cpu())
        n_updates += 1

        if step % steps_per_epoch == 0 or step == n_steps:
            with torch.no_grad():
                val_q = fqe(val_obs)
                val_q_sa = val_q.gather(1, val_actions_t.unsqueeze(1)).squeeze(1)
                val_next_q = fqe(val_next_obs)
                val_next_v = (val_next_pi * val_next_q).sum(dim=1)
                val_target = val_rewards_t + gamma * (1.0 - val_terminals_t) * val_next_v
                residual = val_q_sa - val_target
            history.append(
                {
                    "step": step,
                    "fqe_train_td_loss": total_loss / max(n_updates, 1),
                    "fqe_bellman_mse_val": float(torch.mean(residual ** 2).cpu()),
                    "fqe_bellman_mae_val": float(torch.mean(torch.abs(residual)).cpu()),
                    "fqe_avg_value_val": float(torch.mean((val_next_pi * val_q).sum(dim=1)).cpu()),
                }
            )
            total_loss = 0.0
            n_updates = 0
    return fqe, history


def discounted_clinician_returns(rewards, trajectories, gamma: float):
    return np.asarray([
        float(np.sum(rewards[idx] * (gamma ** np.arange(len(idx)))))
        for idx in trajectories
    ], dtype=np.float64)


def cumulative_log_ratios(actions, target_probs, behavior_probs, trajectories,
                          ratio_clip: float = 5.0):
    chosen_target = np.maximum(target_probs[np.arange(len(actions)), actions], 1e-12)
    chosen_behavior = np.maximum(behavior_probs[np.arange(len(actions)), actions], 1e-12)
    log_ratio = np.clip(np.log(chosen_target) - np.log(chosen_behavior),
                        -ratio_clip, ratio_clip)
    return [np.cumsum(log_ratio[idx]) for idx in trajectories]


def wis_value(rewards, trajectories, trajectory_log_weights, gamma: float):
    if not trajectories:
        return float("nan")
    value = 0.0
    max_len = max(len(idx) for idx in trajectories)
    for step in range(max_len):
        rows = [(idx, weights) for idx, weights in zip(trajectories, trajectory_log_weights)
                if step < len(idx)]
        if not rows:
            continue
        logw = np.asarray([weights[step] for _, weights in rows])
        reward = np.asarray([rewards[idx[step]] for idx, _ in rows])
        weights = np.exp(logw - np.max(logw))
        weights /= np.sum(weights)
        value += (gamma ** step) * float(np.sum(weights * reward))
    return float(value)


def wdr_value(rewards, actions, trajectories, trajectory_log_weights, q_values,
              target_probs, gamma: float):
    q_sa = q_values[np.arange(len(actions)), actions]
    v = np.sum(target_probs * q_values, axis=1)
    baseline = [v[idx[0]] for idx in trajectories if len(idx)]
    correction = 0.0
    max_len = max((len(idx) for idx in trajectories), default=0)
    for step in range(max_len):
        rows = [(idx, weights) for idx, weights in zip(trajectories, trajectory_log_weights)
                if step < len(idx)]
        if not rows:
            continue
        logw = np.asarray([weights[step] for _, weights in rows])
        weights = np.exp(logw - np.max(logw))
        weights /= np.sum(weights)
        residual = []
        for idx, _ in rows:
            pos = idx[step]
            next_v = v[idx[step + 1]] if step + 1 < len(idx) else 0.0
            residual.append(rewards[pos] + gamma * next_v - q_sa[pos])
        correction += (gamma ** step) * float(np.sum(weights * np.asarray(residual)))
    return float(np.mean(baseline) + correction) if baseline else float("nan")


def bootstrap_mean(values, n_bootstrap: int = 200, seed: int = 0):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    estimates = values[samples].mean(axis=1)
    return float(values.mean()), float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def bootstrap_estimator(estimator, trajectories, n_bootstrap: int = 200, seed: int = 0):
    """Patient-level bootstrap for an estimator that accepts trajectory lists."""
    if not trajectories:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    point = float(estimator(trajectories))
    estimates = []
    for _ in range(n_bootstrap):
        sample = [trajectories[i] for i in rng.integers(0, len(trajectories), len(trajectories))]
        estimates.append(float(estimator(sample)))
    return point, float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def fqe_bellman_residual(q_values, next_q_values, actions, rewards, terminals,
                         next_target_probs, gamma: float):
    """MSE of the FQE Bellman equation on held-out transitions."""
    next_v = np.sum(next_target_probs * next_q_values, axis=1)
    target = rewards + gamma * (1.0 - terminals) * next_v
    chosen = q_values[np.arange(len(rewards)), actions.astype(int)]
    residual = chosen - target
    return {
        "bellman_mse": float(np.mean(residual ** 2)) if len(residual) else float("nan"),
        "bellman_mae": float(np.mean(np.abs(residual))) if len(residual) else float("nan"),
        "q_mean": float(np.mean(chosen)) if len(chosen) else float("nan"),
    }
