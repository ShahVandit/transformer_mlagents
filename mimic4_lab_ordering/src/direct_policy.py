"""Picklable direct policy and state-computable blood-draw utility helpers."""
import numpy as np
import torch
import torch.nn as nn

import config as cfg
import itemids as ids


def _indices(state_cols, prefix):
    lookup = {name: i for i, name in enumerate(state_cols)}
    return np.asarray([lookup[f"{prefix}_{lab}"] for lab in ids.TARGET_LABS])


def fit_information_thresholds(states, actions, state_cols):
    """Fit Cheng Eq. 5 cutoffs from decision-time state on logged draw rows."""
    x = np.asarray(states, dtype=np.float32)
    draw = np.asarray(actions, dtype=np.int64) == 1
    mean = x[:, _indices(state_cols, "mean")]
    last = x[:, _indices(state_cols, "last")]
    std = np.maximum(x[:, _indices(state_cols, "std")], cfg.FORECAST_MIN_STD)
    score = np.abs(mean - last) / std
    out = {}
    for j, lab in enumerate(ids.TARGET_LABS):
        values = score[draw, j]
        values = values[np.isfinite(values)]
        out[lab] = float(np.median(values)) if len(values) else 0.0
    return out


def information_utility(states, state_cols, thresholds):
    """Forecast-based information available before the current decision."""
    x = np.asarray(states, dtype=np.float32)
    mean = x[:, _indices(state_cols, "mean")]
    last = x[:, _indices(state_cols, "last")]
    std = np.maximum(x[:, _indices(state_cols, "std")], cfg.FORECAST_MIN_STD)
    cut = np.asarray([thresholds[lab] for lab in ids.TARGET_LABS], dtype=np.float32)
    score = np.maximum(0.0, np.abs(mean - last) / std - cut)
    return score.sum(axis=1).astype(np.float32)


def draw_burden(states, state_cols):
    """Base draw cost plus state-computable redundancy cost."""
    x = np.asarray(states, dtype=np.float32)
    delta = np.maximum(x[:, _indices(state_cols, "delta")], 0.0)
    since_any_lab = np.min(delta, axis=1)
    return (1.0 + np.exp(-since_any_lab / cfg.COST_DECAY_GAMMA)).astype(np.float32)


def _mean_stay_sum(values, stay_ids):
    values = np.asarray(values, dtype=np.float64)
    stay_ids = np.asarray(stay_ids)
    _, inverse = np.unique(stay_ids, return_inverse=True)
    totals = np.zeros(int(inverse.max()) + 1, dtype=np.float64)
    np.add.at(totals, inverse, values)
    return float(totals.mean())


def fit_utility_scales(states, stay_ids, state_cols, thresholds):
    """Fit per-decision objective scales on training rows only.

    The optimizer averages its loss over decisions, so its utility scales must
    be per-decision too. Per-stay scales would divide the utility gradient by
    the mean stay length while leaving the overlap penalty unchanged.
    """
    info = information_utility(states, state_cols, thresholds)
    burden = draw_burden(states, state_cols)
    info_scale = max(float(info.mean()), 1e-6)
    burden_scale = max(float(burden.mean()), 1e-6)
    return {
        "information_scale": info_scale,
        "burden_scale": burden_scale,
        "normalization": "train_mean_per_decision_always_draw",
    }


def utility_components(states, state_cols, thresholds, scales):
    info = information_utility(states, state_cols, thresholds)
    burden = draw_burden(states, state_cols)
    return (
        info / float(scales["information_scale"]),
        burden / float(scales["burden_scale"]),
    )


def draw_utility(states, state_cols, thresholds, scales, preference):
    """g(s, 1); g(s, 0) is exactly zero."""
    info, burden = utility_components(states, state_cols, thresholds, scales)
    w_info, w_burden = map(float, preference)
    return (w_info * info - w_burden * burden).astype(np.float32)


def expected_overlap(draw_prob, behavior_draw_prob):
    """Probability that independent policy and behavior actions agree."""
    p = np.asarray(draw_prob, dtype=np.float32)
    b1 = np.asarray(behavior_draw_prob, dtype=np.float32)
    return (p * b1 + (1.0 - p) * (1.0 - b1)).astype(np.float32)


def unsupported_action_mass(draw_prob, behavior_draw_prob, epsilon):
    """Expected mass placed on discrete actions below the support floor."""
    p = np.asarray(draw_prob, dtype=np.float32)
    b1 = np.asarray(behavior_draw_prob, dtype=np.float32)
    b0 = 1.0 - b1
    return (
        p * np.maximum(float(epsilon) - b1, 0.0)
        + (1.0 - p) * np.maximum(float(epsilon) - b0, 0.0)
    ).astype(np.float32)


def unsupported_action_mass_torch(draw_prob, behavior_draw_prob, epsilon):
    b0 = 1.0 - behavior_draw_prob
    return (
        draw_prob * torch.relu(float(epsilon) - behavior_draw_prob)
        + (1.0 - draw_prob) * torch.relu(float(epsilon) - b0)
    )


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden=cfg.DIRECT_HIDDEN):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, states):
        return torch.sigmoid(self.layers(states)).squeeze(-1)


class DirectPolicy:
    """Standardized stochastic policy with the pipeline's predict interface."""
    def __init__(self, net, mean, std, threshold=0.5,
                 behavior_model=None, support_epsilon=None):
        self.net = net.cpu()
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.std = np.where(self.std < 1e-6, 1.0, self.std).astype(np.float32)
        self.threshold = float(threshold)
        self.behavior_model = behavior_model
        self.support_epsilon = (None if support_epsilon is None
                                else float(support_epsilon))

    def predict_proba(self, states):
        x = (np.asarray(states, dtype=np.float32) - self.mean) / self.std
        self.net.eval()
        with torch.no_grad():
            p1 = self.net(torch.as_tensor(x, dtype=torch.float32)).numpy()
        return np.column_stack([1.0 - p1, p1]).astype(np.float32)

    def predict(self, states):
        p1 = self.predict_proba(states)[:, 1]
        actions = (p1 > self.threshold).astype(np.int64)
        if self.behavior_model is None or self.support_epsilon is None:
            return actions
        behavior_p1 = self.behavior_model.predict_proba(
            np.asarray(states, dtype=np.float32))[:, 1]
        draw_unsupported = behavior_p1 < self.support_epsilon
        no_draw_unsupported = (1.0 - behavior_p1) < self.support_epsilon
        actions[draw_unsupported & ~no_draw_unsupported] = 0
        actions[no_draw_unsupported & ~draw_unsupported] = 1
        return actions
