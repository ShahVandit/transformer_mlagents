"""PyTorch behavior cloning, CQL-DQN, FQE, and optional MOMENT encoders."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_ACTIONS = 9


class GRUEncoder(nn.Module):
    def __init__(self, seq_dim: int, static_dim: int, hidden: int = 96):
        super().__init__()
        self.gru = nn.GRU(seq_dim, hidden, batch_first=True)
        self.static = nn.Sequential(nn.Linear(static_dim, hidden), nn.ReLU())
        self.out_dim = hidden * 2

    def forward(self, seq, static):
        h = self.gru(seq)[0][:, -1]
        s = self.static(static)
        return torch.cat([h, s], dim=1)


class MomentEncoder(nn.Module):
    """Frozen MOMENT embedding encoder with a static-feature branch.

    MOMENT is optional because it requires the external `momentfm` package and
    downloads Hugging Face weights. The time-series input is [B, T, C]; MOMENT
    expects [B, C, T]. We pad short VitalDB windows to 512 samples because the
    pretrained checkpoints are patch-based and commonly configured for longer
    contexts than our 30-minute windows.
    """

    def __init__(
        self,
        seq_dim: int,
        static_dim: int,
        hidden: int = 96,
        model_name: str = "AutonLab/MOMENT-1-small",
        target_len: int = 512,
        freeze: bool = True,
    ):
        super().__init__()
        try:
            from momentfm import MOMENTPipeline
        except ImportError as exc:
            raise ImportError(
                "MOMENT encoder requested but `momentfm` is not installed. "
                "Install with `pip install -r requirements-moment.txt`."
            ) from exc

        self.target_len = target_len
        self.n_value_channels = (seq_dim - 1) // 2
        self.moment = MOMENTPipeline.from_pretrained(
            model_name,
            model_kwargs={"task_name": "embedding"},
        )
        self.moment.init()
        if freeze:
            for p in self.moment.parameters():
                p.requires_grad = False
            self.moment.eval()

        self.static = nn.Sequential(
            nn.Linear(static_dim + self.n_value_channels, hidden),
            nn.ReLU(),
        )
        self.proj = nn.LazyLinear(hidden)
        self.out_dim = hidden * 2

    def _pad_or_crop(self, seq):
        # seq: [B, T, values+masks+pad_flag]. MOMENT should see only values.
        values = seq[:, :, :self.n_value_channels]
        missing = seq[:, :, self.n_value_channels: self.n_value_channels * 2]
        observed = 1.0 - seq[:, :, -1]

        # MOMENT only supports a time-level mask. Per-signal missingness is kept
        # as a compact side feature rather than turning mostly-complete minutes
        # into fully masked patches.
        denom = observed.sum(dim=1, keepdim=True).clamp_min(1.0)
        missing_summary = (missing * observed.unsqueeze(-1)).sum(dim=1) / denom

        x = values.transpose(1, 2)
        mask = observed
        length = x.shape[-1]
        if length < self.target_len:
            pad = self.target_len - length
            x = F.pad(x, (pad, 0))
            mask = F.pad(mask, (pad, 0))
        elif length > self.target_len:
            x = x[..., -self.target_len:]
            mask = mask[..., -self.target_len:]
        return x, mask, missing_summary

    @staticmethod
    def _extract_embedding(out):
        if torch.is_tensor(out):
            emb = out
        elif hasattr(out, "embeddings"):
            emb = out.embeddings
        elif hasattr(out, "embedding"):
            emb = out.embedding
        elif isinstance(out, dict):
            emb = out.get("embeddings", out.get("embedding"))
        else:
            raise TypeError(f"Unsupported MOMENT output type: {type(out)!r}")
        if emb.ndim > 2:
            emb = emb.mean(dim=tuple(range(1, emb.ndim - 1)))
        return emb

    def forward(self, seq, static):
        x, input_mask, missing_summary = self._pad_or_crop(seq)
        if not any(p.requires_grad for p in self.moment.parameters()):
            with torch.no_grad():
                out = self.moment(x_enc=x, input_mask=input_mask)
        else:
            out = self.moment(x_enc=x, input_mask=input_mask)
        emb = self._extract_embedding(out)
        static_aug = torch.cat([static, missing_summary], dim=1)
        return torch.cat([self.proj(emb), self.static(static_aug)], dim=1)


def make_encoder(
    encoder: str,
    seq_dim: int,
    static_dim: int,
    hidden: int = 96,
    moment_model: str = "AutonLab/MOMENT-1-small",
) -> nn.Module:
    if encoder == "gru":
        return GRUEncoder(seq_dim, static_dim, hidden)
    if encoder == "moment":
        return MomentEncoder(seq_dim, static_dim, hidden, model_name=moment_model)
    raise ValueError(f"unknown encoder: {encoder}")


class PolicyNet(nn.Module):
    def __init__(
        self,
        seq_dim: int,
        static_dim: int,
        hidden: int = 96,
        n_actions: int = N_ACTIONS,
        encoder: str = "gru",
        moment_model: str = "AutonLab/MOMENT-1-small",
    ):
        super().__init__()
        self.encoder_name = encoder
        self.moment_model = moment_model
        self.encoder = make_encoder(encoder, seq_dim, static_dim, hidden, moment_model)
        self.head = nn.Sequential(nn.Linear(self.encoder.out_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))

    def forward(self, seq, static):
        return self.head(self.encoder(seq, static))


class QNet(nn.Module):
    def __init__(
        self,
        seq_dim: int,
        static_dim: int,
        hidden: int = 96,
        n_actions: int = N_ACTIONS,
        encoder: str = "gru",
        moment_model: str = "AutonLab/MOMENT-1-small",
    ):
        super().__init__()
        self.encoder_name = encoder
        self.moment_model = moment_model
        self.encoder = make_encoder(encoder, seq_dim, static_dim, hidden, moment_model)
        self.head = nn.Sequential(nn.Linear(self.encoder.out_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))

    def forward(self, seq, static):
        return self.head(self.encoder(seq, static))


def _loader(data: dict[str, np.ndarray], keys: list[str], batch_size: int, shuffle: bool = True):
    tensors = [torch.tensor(data[k]) for k in keys]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def _dims(data: dict[str, np.ndarray]) -> tuple[int, int]:
    return int(data["seq"].shape[2]), int(data["static"].shape[1])


def _predict_batch_size(model: nn.Module, batch_size: int) -> int:
    uses_moment = getattr(model, "encoder_name", None) == "moment"
    return min(batch_size, 32) if uses_moment else batch_size


def predict_logits(model: nn.Module, data: dict[str, np.ndarray], batch_size: int = 8192) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for seq, static in _loader(data, ["seq", "static"], _predict_batch_size(model, batch_size), shuffle=False):
            outs.append(model(seq.to(DEVICE), static.to(DEVICE)).cpu().numpy())
    return np.concatenate(outs) if outs else np.empty((0, N_ACTIONS))


def policy_probs(model: PolicyNet, data: dict[str, np.ndarray], batch_size: int = 8192) -> np.ndarray:
    logits = predict_logits(model, data, batch_size)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def greedy_actions(model: nn.Module, data: dict[str, np.ndarray], batch_size: int = 8192) -> np.ndarray:
    logits = predict_logits(model, data, batch_size)
    return logits.argmax(axis=1).astype(np.int64)


def train_bc(
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    epochs: int = 10,
    batch_size: int = 1024,
    lr: float = 1e-3,
    seed: int = 0,
    encoder: str = "gru",
    moment_model: str = "AutonLab/MOMENT-1-small",
) -> tuple[PolicyNet, dict[str, float]]:
    torch.manual_seed(seed)
    seq_dim, static_dim = _dims(train)
    model = PolicyNet(seq_dim, static_dim, encoder=encoder, moment_model=moment_model).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr)
    best, best_state = -1.0, None

    for _epoch in range(epochs):
        model.train()
        for seq, static, action in _loader(train, ["seq", "static", "action"], batch_size):
            opt.zero_grad()
            loss = F.cross_entropy(model(seq.to(DEVICE), static.to(DEVICE)), action.to(DEVICE))
            loss.backward()
            opt.step()
        pred = greedy_actions(model, val)
        score = f1_score(val["action"], pred, average="macro", zero_division=0)
        if score > best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    pred = greedy_actions(model, val)
    val_counts = np.bincount(val["action"], minlength=N_ACTIONS)
    pred_counts = np.bincount(pred, minlength=N_ACTIONS)
    majority = int(val_counts.argmax())
    metrics = {
        "val_accuracy": float(accuracy_score(val["action"], pred)),
        "val_macro_f1": float(f1_score(val["action"], pred, average="macro", zero_division=0)),
        "val_majority_accuracy": float(val_counts.max() / max(val_counts.sum(), 1)),
        "val_majority_action": majority,
        "pred_majority_action": int(pred_counts.argmax()),
        "pred_hold_hold_frac": float(pred_counts[4] / max(pred_counts.sum(), 1)),
    }
    return model, metrics


def scalar_reward(data: dict[str, np.ndarray], weights: tuple[float, float, float]) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float32)
    return data["reward_components"] @ w


def train_cql(
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    weights: tuple[float, float, float],
    epochs: int = 20,
    batch_size: int = 1024,
    lr: float = 3e-4,
    gamma: float = 0.99,
    cql_alpha: float = 0.5,
    seed: int = 0,
    encoder: str = "gru",
    moment_model: str = "AutonLab/MOMENT-1-small",
) -> tuple[QNet, dict[str, float]]:
    torch.manual_seed(seed)
    seq_dim, static_dim = _dims(train)
    q = QNet(seq_dim, static_dim, encoder=encoder, moment_model=moment_model).to(DEVICE)
    target = QNet(seq_dim, static_dim, encoder=encoder, moment_model=moment_model).to(DEVICE)
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr)

    train_local = dict(train)
    train_local["reward"] = scalar_reward(train, weights).astype(np.float32)
    best_loss, best_state = float("inf"), None

    for epoch in range(epochs):
        q.train()
        losses = []
        for seq, static, action, reward, nseq, nstatic, done in _loader(
            train_local,
            ["seq", "static", "action", "reward", "next_seq", "next_static", "done"],
            batch_size,
        ):
            seq, static = seq.to(DEVICE), static.to(DEVICE)
            action = action.to(DEVICE).long()
            reward, done = reward.to(DEVICE), done.to(DEVICE)
            nseq, nstatic = nseq.to(DEVICE), nstatic.to(DEVICE)
            with torch.no_grad():
                next_q = target(nseq, nstatic).max(dim=1).values
                y = reward + gamma * (1.0 - done) * next_q
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
    val_actions = greedy_actions(q, val)
    support = float(np.mean(val_actions == val["action"])) if len(val_actions) else 0.0
    return q, {"train_loss": best_loss, "val_action_match": support}


def train_fqe(
    train: dict[str, np.ndarray],
    weights: tuple[float, float, float],
    policy: nn.Module,
    policy_type: str,
    epochs: int = 15,
    batch_size: int = 1024,
    lr: float = 3e-4,
    gamma: float = 0.99,
    seed: int = 0,
    encoder: str = "gru",
    moment_model: str = "AutonLab/MOMENT-1-small",
) -> QNet:
    torch.manual_seed(seed)
    seq_dim, static_dim = _dims(train)
    q = QNet(seq_dim, static_dim, encoder=encoder, moment_model=moment_model).to(DEVICE)
    target = QNet(seq_dim, static_dim, encoder=encoder, moment_model=moment_model).to(DEVICE)
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr)
    train_local = dict(train)
    train_local["reward"] = scalar_reward(train, weights).astype(np.float32)

    for _epoch in range(epochs):
        q.train()
        for seq, static, action, reward, nseq, nstatic, done in _loader(
            train_local,
            ["seq", "static", "action", "reward", "next_seq", "next_static", "done"],
            batch_size,
        ):
            seq, static = seq.to(DEVICE), static.to(DEVICE)
            action = action.to(DEVICE).long()
            reward, done = reward.to(DEVICE), done.to(DEVICE)
            nseq, nstatic = nseq.to(DEVICE), nstatic.to(DEVICE)
            with torch.no_grad():
                nq = target(nseq, nstatic)
                if policy_type == "bc":
                    probs = torch.softmax(policy(nseq, nstatic), dim=1)
                    next_v = (probs * nq).sum(dim=1)
                else:
                    pa = policy(nseq, nstatic).argmax(dim=1)
                    next_v = nq.gather(1, pa[:, None]).squeeze(1)
                y = reward + gamma * (1.0 - done) * next_v
            pred = q(seq, static).gather(1, action[:, None]).squeeze(1)
            loss = F.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        target.load_state_dict(q.state_dict())
    return q


def fqe_value(fqe: QNet, data: dict[str, np.ndarray], policy: nn.Module, policy_type: str) -> float:
    fqe.eval()
    vals = []
    with torch.no_grad():
        for seq, static in _loader(data, ["seq", "static"], 8192, shuffle=False):
            seq, static = seq.to(DEVICE), static.to(DEVICE)
            qv = fqe(seq, static)
            if policy_type == "bc":
                probs = torch.softmax(policy(seq, static), dim=1)
                vals.append((probs * qv).sum(dim=1).cpu().numpy())
            else:
                action = policy(seq, static).argmax(dim=1)
                vals.append(qv.gather(1, action[:, None]).squeeze(1).cpu().numpy())
    return float(np.concatenate(vals).mean()) if vals else float("nan")


def save_model(path: Path, model: nn.Module, kind: str, extra: dict | None = None) -> None:
    path.parent.mkdir(exist_ok=True)
    torch.save({"kind": kind, "state_dict": model.state_dict(), "extra": extra or {}}, path)


def load_model(
    path: Path,
    kind: str,
    seq_dim: int,
    static_dim: int,
    encoder: str = "gru",
    moment_model: str = "AutonLab/MOMENT-1-small",
) -> nn.Module:
    payload = torch.load(path, map_location=DEVICE)
    extra = payload.get("extra", {})
    encoder = extra.get("encoder", encoder)
    moment_model = extra.get("moment_model", moment_model)
    model = (
        PolicyNet(seq_dim, static_dim, encoder=encoder, moment_model=moment_model)
        if kind == "bc"
        else QNet(seq_dim, static_dim, encoder=encoder, moment_model=moment_model)
    )
    model.load_state_dict(payload["state_dict"])
    return model.to(DEVICE)
