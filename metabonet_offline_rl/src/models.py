from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_ACTIONS = 12


class TransformerEncoder(nn.Module):
    def __init__(self, seq_dim: int, static_dim: int, hidden: int = 96, layers: int = 2, heads: int = 4):
        super().__init__()
        self.input = nn.Linear(seq_dim, hidden)
        self.pos = nn.Parameter(torch.zeros(1, 256, hidden))
        block = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.seq = nn.TransformerEncoder(block, num_layers=layers)
        self.static = nn.Sequential(nn.Linear(static_dim, hidden), nn.ReLU())
        self.out_dim = hidden * 2

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        pad_mask = seq[:, :, -1] > 0.5
        x = self.input(seq)
        x = x + self.pos[:, : x.shape[1]]
        h = self.seq(x, src_key_padding_mask=pad_mask)
        valid = (~pad_mask).float().unsqueeze(-1)
        pooled = (h * valid).sum(1) / valid.sum(1).clamp_min(1.0)
        return torch.cat([pooled, self.static(static)], dim=1)


class PolicyNet(nn.Module):
    def __init__(self, seq_dim: int, static_dim: int, n_actions: int, hidden: int = 96):
        super().__init__()
        self.encoder = TransformerEncoder(seq_dim, static_dim, hidden)
        self.head = nn.Sequential(nn.Linear(self.encoder.out_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(seq, static))


class QNet(PolicyNet):
    pass


def _dims(data: dict[str, np.ndarray]) -> tuple[int, int, int]:
    observed = int(data["action"].max() + 1) if len(data["action"]) else N_ACTIONS
    return int(data["seq"].shape[2]), int(data["static"].shape[1]), max(N_ACTIONS, observed)


def _loader(data: dict[str, np.ndarray], keys: list[str], batch_size: int, shuffle: bool = True) -> DataLoader:
    tensors = [torch.tensor(data[key]) for key in keys]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def scalar_reward(data: dict[str, np.ndarray], weights: tuple[float, float, float]) -> np.ndarray:
    return data["reward_components"] @ np.asarray(weights, dtype=np.float32)


def predict(model: nn.Module, data: dict[str, np.ndarray], batch_size: int = 8192) -> np.ndarray:
    model.eval()
    rows = []
    with torch.no_grad():
        for seq, static in _loader(data, ["seq", "static"], batch_size, shuffle=False):
            rows.append(model(seq.to(DEVICE), static.to(DEVICE)).cpu().numpy())
    return np.concatenate(rows) if rows else np.empty((0, 0), dtype=np.float32)


def greedy_actions(model: nn.Module, data: dict[str, np.ndarray], batch_size: int = 8192) -> np.ndarray:
    logits = predict(model, data, batch_size)
    return logits.argmax(1).astype(np.int64) if len(logits) else np.empty((0,), dtype=np.int64)


def policy_probs(model: nn.Module, data: dict[str, np.ndarray], batch_size: int = 8192) -> np.ndarray:
    logits = predict(model, data, batch_size)
    logits = logits - logits.max(1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(1, keepdims=True)


def train_bc(train, val, epochs: int = 5, batch_size: int = 1024, lr: float = 1e-3, seed: int = 0):
    torch.manual_seed(seed)
    seq_dim, static_dim, n_actions = _dims(train)
    model = PolicyNet(seq_dim, static_dim, n_actions).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_f1, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        for seq, static, action in _loader(train, ["seq", "static", "action"], batch_size):
            loss = F.cross_entropy(model(seq.to(DEVICE), static.to(DEVICE)), action.to(DEVICE).long())
            opt.zero_grad()
            loss.backward()
            opt.step()
        pred = greedy_actions(model, val)
        f1 = f1_score(val["action"], pred, average="macro", zero_division=0) if len(pred) else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    pred = greedy_actions(model, val)
    counts = np.bincount(val["action"], minlength=n_actions) if len(val["action"]) else np.zeros(n_actions)
    return model, {
        "val_accuracy": float(accuracy_score(val["action"], pred)) if len(pred) else 0.0,
        "val_macro_f1": float(f1_score(val["action"], pred, average="macro", zero_division=0)) if len(pred) else 0.0,
        "val_majority_accuracy": float(counts.max() / max(counts.sum(), 1)),
        "val_majority_action": int(counts.argmax()) if len(counts) else -1,
    }


def train_cql(
    train,
    val,
    weights: tuple[float, float, float],
    epochs: int = 8,
    batch_size: int = 1024,
    lr: float = 3e-4,
    gamma: float = 0.99,
    cql_alpha: float = 0.5,
    seed: int = 0,
):
    torch.manual_seed(seed)
    seq_dim, static_dim, n_actions = _dims(train)
    q = QNet(seq_dim, static_dim, n_actions).to(DEVICE)
    target = QNet(seq_dim, static_dim, n_actions).to(DEVICE)
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    local = dict(train)
    local["reward"] = scalar_reward(train, weights).astype(np.float32)
    best_loss, best_state = float("inf"), None
    for _ in range(epochs):
        losses = []
        q.train()
        for seq, static, action, reward, nseq, nstatic, done in _loader(
            local, ["seq", "static", "action", "reward", "next_seq", "next_static", "done"], batch_size
        ):
            seq, static, action = seq.to(DEVICE), static.to(DEVICE), action.to(DEVICE).long()
            reward, done = reward.to(DEVICE), done.to(DEVICE)
            nseq, nstatic = nseq.to(DEVICE), nstatic.to(DEVICE)
            with torch.no_grad():
                y = reward + gamma * (1.0 - done) * target(nseq, nstatic).max(1).values
            q_all = q(seq, static)
            q_a = q_all.gather(1, action[:, None]).squeeze(1)
            bellman = F.mse_loss(q_a, y)
            conservative = torch.logsumexp(q_all, dim=1).mean() - q_a.mean()
            loss = bellman + cql_alpha * conservative
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        target.load_state_dict(q.state_dict())
        mean_loss = float(np.mean(losses)) if losses else float("inf")
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {k: v.detach().cpu().clone() for k, v in q.state_dict().items()}
    if best_state:
        q.load_state_dict(best_state)
    return q, {"train_loss": best_loss, "val_action_match": float(np.mean(greedy_actions(q, val) == val["action"]))}


def train_fqe(
    train,
    reward: np.ndarray,
    policy,
    policy_type: str,
    epochs: int = 5,
    batch_size: int = 1024,
    lr: float = 3e-4,
    gamma: float = 0.99,
    seed: int = 0,
):
    torch.manual_seed(seed)
    seq_dim, static_dim, n_actions = _dims(train)
    q = QNet(seq_dim, static_dim, n_actions).to(DEVICE)
    target = QNet(seq_dim, static_dim, n_actions).to(DEVICE)
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    local = dict(train)
    local["reward"] = reward.astype(np.float32)
    for _ in range(epochs):
        q.train()
        for seq, static, action, rew, nseq, nstatic, done in _loader(
            local, ["seq", "static", "action", "reward", "next_seq", "next_static", "done"], batch_size
        ):
            seq, static, action = seq.to(DEVICE), static.to(DEVICE), action.to(DEVICE).long()
            rew, done = rew.to(DEVICE), done.to(DEVICE)
            nseq, nstatic = nseq.to(DEVICE), nstatic.to(DEVICE)
            with torch.no_grad():
                next_q = target(nseq, nstatic)
                if policy_type == "bc":
                    probs = torch.softmax(policy(nseq, nstatic), dim=1)
                    next_v = (probs * next_q).sum(1)
                else:
                    next_a = policy(nseq, nstatic).argmax(1)
                    next_v = next_q.gather(1, next_a[:, None]).squeeze(1)
                y = rew + gamma * (1.0 - done) * next_v
            pred = q(seq, static).gather(1, action[:, None]).squeeze(1)
            loss = F.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        target.load_state_dict(q.state_dict())
    return q


def fqe_value(fqe: nn.Module, data, policy, policy_type: str) -> float:
    values = []
    fqe.eval()
    with torch.no_grad():
        for seq, static in _loader(data, ["seq", "static"], 8192, shuffle=False):
            seq, static = seq.to(DEVICE), static.to(DEVICE)
            q = fqe(seq, static)
            if policy_type == "bc":
                probs = torch.softmax(policy(seq, static), dim=1)
                values.append((probs * q).sum(1).cpu().numpy())
            else:
                action = policy(seq, static).argmax(1)
                values.append(q.gather(1, action[:, None]).squeeze(1).cpu().numpy())
    return float(np.concatenate(values).mean()) if values else float("nan")


def save_model(path: Path, model: nn.Module, kind: str, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": kind, "state_dict": model.state_dict(), "extra": extra or {}}, path)


def load_model(path: Path, kind: str, seq_dim: int, static_dim: int, n_actions: int):
    payload = torch.load(path, map_location=DEVICE)
    model = PolicyNet(seq_dim, static_dim, n_actions) if kind == "bc" else QNet(seq_dim, static_dim, n_actions)
    model.load_state_dict(payload["state_dict"])
    return model.to(DEVICE)
