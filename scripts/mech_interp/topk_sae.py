"""
TopK Sparse Autoencoder — supporting (per-model) track.

Trimmed port of the EEG-foundation-model repo's SAE (mecheeg/sae.py): TopK
sparsity (structural L0, no L1 shrinkage), pre-encoder bias, unit-norm decoder
columns, Anthropic-style dead-neuron resampling. Used to give human-legible
meaning to a single model's features (max-activating obs, then TCAV / probes);
the crosscoder handles the cross-model diff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEConfig:
    d_in: int
    expansion: int = 8            # dict_size = d_in * expansion
    k: int = 32
    lr: float = 3e-4
    epochs: int = 30
    batch_size: int = 2048
    resample_every: int = 5
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 0


class TopKSAE(nn.Module):
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.dict_size = cfg.d_in * cfg.expansion
        self.b_pre = nn.Parameter(torch.zeros(cfg.d_in))
        self.encoder = nn.Linear(cfg.d_in, self.dict_size)
        self.decoder = nn.Linear(self.dict_size, cfg.d_in)
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.kaiming_uniform_(self.decoder.weight)
        self._normalise_decoder()
        self.to(cfg.device)

    @torch.no_grad()
    def _normalise_decoder(self):
        n = self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.decoder.weight.div_(n)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.encoder(x - self.b_pre))
        k = self.cfg.k
        if k < z.shape[-1]:
            vals, idx = z.topk(k, dim=-1)
            mask = torch.zeros_like(z)
            mask.scatter_(-1, idx, 1.0)
            z = z * mask
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z) + self.b_pre

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


def train_sae(acts: torch.Tensor, cfg: SAEConfig, verbose: bool = True
              ) -> Tuple[TopKSAE, Dict[str, float], torch.Tensor]:
    """acts: [N, d_in] raw activations. Normalises (z-score) internally; stores
    mean/std on the returned model as buffers for later encoding."""
    dev = cfg.device
    mean = acts.mean(0)
    std = acts.std(0).clamp(min=1e-8)
    X = ((acts - mean) / std).float()

    sae = TopKSAE(cfg)
    sae.register_buffer("act_mean", mean)
    sae.register_buffer("act_std", std)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    N = X.shape[0]
    fire = torch.zeros(sae.dict_size, device=dev)

    for ep in range(cfg.epochs):
        perm = torch.randperm(N)
        ep_loss = 0.0
        ep_fire = torch.zeros(sae.dict_size, device=dev)
        nb = 0
        for i in range(0, N, cfg.batch_size):
            xb = X[perm[i:i + cfg.batch_size]].to(dev)
            xh, z = sae(xb)
            loss = F.mse_loss(xh, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sae._normalise_decoder()
            ep_loss += loss.item()
            ep_fire += (z.detach() > 0).float().sum(0)
            nb += 1
        fire += ep_fire
        if verbose:
            with torch.no_grad():
                xb = X[:min(N, 8192)].to(dev)
                xh, _ = sae(xb)
                ev = 1 - (xb - xh).pow(2).sum().item() / (
                    (xb - xb.mean(0)).pow(2).sum().item() + 1e-8)
            print(f"  ep {ep + 1:3d}/{cfg.epochs} loss={ep_loss / nb:.4f} "
                  f"R2={ev:.4f} dead={int((ep_fire == 0).sum())}/{sae.dict_size}",
                  flush=True)
        if (cfg.resample_every and (ep + 1) % cfg.resample_every == 0
                and ep + 1 < cfg.epochs):
            _resample(sae, X, ep_fire, opt)

    alive = (fire > 0)
    with torch.no_grad():
        xb = X.to(dev)
        xh, z = sae(xb)
        metrics = {
            "r2": 1 - (xb - xh).pow(2).sum().item() / (
                (xb - xb.mean(0)).pow(2).sum().item() + 1e-8),
            "l0": (z > 0).float().sum(-1).mean().item(),
            "dead_frac": (~alive).float().mean().item(),
        }
    return sae.cpu(), metrics, alive.cpu()


@torch.no_grad()
def _resample(sae: TopKSAE, X: torch.Tensor, fire: torch.Tensor,
              opt: torch.optim.Optimizer) -> int:
    dead = (fire == 0)
    n_dead = int(dead.sum().item())
    if n_dead == 0:
        return 0
    dev = sae.cfg.device
    idx = torch.randperm(len(X))[:min(len(X), 8192)]
    xb = X[idx].to(dev)
    xh, _ = sae(xb)
    err = (xb - xh).pow(2).sum(-1)
    pick = torch.multinomial(err / (err.sum() + 1e-8), n_dead, replacement=True)
    dirs = (xb[pick] - sae.b_pre)
    dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)
    di = dead.nonzero(as_tuple=True)[0]
    alive_norm = sae.encoder.weight[~dead].norm(dim=-1).mean()
    sae.encoder.weight[di] = dirs * alive_norm * 0.8
    sae.encoder.bias[di] = 0.0
    sae.decoder.weight[:, di] = dirs.T
    sae._normalise_decoder()
    return n_dead
