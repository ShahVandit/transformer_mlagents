"""
Two-model TopK crosscoder for model diffing (task1 vs task2).

Adapted from the Kissane/Nanda crosscoder core (crosscoder-model-diff-replication),
with the key upgrade flagged by the model-diffing literature: replace the L1
sparsity with **TopK** (structural L0). Naive L1 crosscoders induce "Complete
Shrinkage" and "Latent Decoupling" — they manufacture false model-specific
features, which is fatal for our narrow-fine-tune diff. TopK creates competition
between latents and removes the decoder-norm-shrinkage pressure.
  refs: arXiv:2504.02922 (Overcoming Sparsity Artifacts), arXiv:2603.04426
        (Delta-Crosscoder, narrow fine-tuning).

A shared encoder reads BOTH models' activations to produce one set of sparse
latents; per-model decoders reconstruct each model's activation. Each latent's
(decoder_norm_model0, decoder_norm_model1) pair classifies it as shared vs
task-specific — the literal "what's shared / what's task-specific" decomposition
that the merge question asks for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CrossCoderConfig:
    d_in: int                     # activation dim (d_model)
    dict_size: int = 2048         # latent dictionary size
    k: int = 32                   # TopK active latents per token
    n_models: int = 2
    dec_init_norm: float = 0.1
    lr: float = 5e-4
    epochs: int = 60
    batch_size: int = 4096
    resample_every: int = 10      # dead-latent resample cadence (epochs); 0=off
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 0


def _topk_mask(z: torch.Tensor, k: int) -> torch.Tensor:
    """Zero all but the top-k per row (per token)."""
    if k >= z.shape[-1]:
        return z
    vals, idx = z.topk(k, dim=-1)
    mask = torch.zeros_like(z)
    mask.scatter_(-1, idx, 1.0)
    return z * mask


class TopKCrossCoder(nn.Module):
    def __init__(self, cfg: CrossCoderConfig):
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        M, d, h = cfg.n_models, cfg.d_in, cfg.dict_size
        # Encoder: per-model projection into shared latent space.
        self.W_enc = nn.Parameter(torch.empty(M, d, h))
        # Decoder: per-model reconstruction from shared latents.
        self.W_dec = nn.Parameter(torch.empty(h, M, d))
        nn.init.normal_(self.W_dec)
        # Unit-ish decoder columns scaled to dec_init_norm (per model).
        self.W_dec.data = (
            self.W_dec.data
            / self.W_dec.data.norm(dim=-1, keepdim=True)
            * cfg.dec_init_norm
        )
        # Init encoder as the decoder transpose (standard SAE/crosscoder init).
        self.W_enc.data = self.W_dec.data.clone().permute(1, 2, 0)
        self.b_enc = nn.Parameter(torch.zeros(h))
        self.b_dec = nn.Parameter(torch.zeros(M, d))
        self.to(cfg.device)

    # x: [batch, n_models, d_in]
    def encode(self, x: torch.Tensor, apply_topk: bool = True) -> torch.Tensor:
        # sum over models: shared latents see both models' activations
        pre = torch.einsum("bmd,mdh->bh", x, self.W_enc) + self.b_enc
        z = F.relu(pre)
        if apply_topk:
            z = _topk_mask(z, self.cfg.k)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bh,hmd->bmd", z, self.W_dec) + self.b_dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z

    def loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_hat, z = self(x)
        recon = (x_hat - x).pow(2).sum(-1).mean()   # sum over (model,d), mean over batch
        return recon, z

    # ── decoder-norm diffing ────────────────────────────────────────────────
    @torch.no_grad()
    def decoder_norms(self) -> torch.Tensor:
        """[dict_size, n_models] L2 norm of each latent's per-model decoder."""
        return self.W_dec.norm(dim=-1)

    @torch.no_grad()
    def relative_norms(self) -> torch.Tensor:
        """[dict_size] in [0,1]: norm_model1 / (norm_model0 + norm_model1).
        ~0  -> model0 (task1) specific
        ~1  -> model1 (task2) specific
        ~.5 -> shared
        """
        n = self.decoder_norms()
        return n[:, 1] / (n.sum(dim=-1) + 1e-8)

    @torch.no_grad()
    def shared_specific_split(self, lo: float = 0.3, hi: float = 0.7,
                              alive: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        """Boolean masks over latents. `alive` optionally restricts to latents
        that ever fired (pass from training/eval)."""
        r = self.relative_norms()
        a = torch.ones_like(r, dtype=torch.bool) if alive is None else alive
        return {
            "task1_specific": (r < lo) & a,   # model0
            "shared": (r >= lo) & (r <= hi) & a,
            "task2_specific": (r > hi) & a,   # model1
            "relative_norms": r,
            "alive": a,
        }


# ── normalisation: scale each model's acts to mean-norm sqrt(d) ──────────────
def estimate_scale(acts: torch.Tensor) -> float:
    import math
    mean_norm = acts.norm(dim=-1).mean().item()
    return math.sqrt(acts.shape[-1]) / (mean_norm + 1e-8)


def train_crosscoder(
    acts0: torch.Tensor,    # [N, d_in] model0 (task1)
    acts1: torch.Tensor,    # [N, d_in] model1 (task2), matched inputs
    cfg: CrossCoderConfig,
    verbose: bool = True,
) -> Tuple[TopKCrossCoder, Dict[str, float], torch.Tensor]:
    """Returns (trained crosscoder, final metrics, alive-latent mask)."""
    assert acts0.shape == acts1.shape, "matched-input activations required"
    dev = cfg.device
    s0, s1 = estimate_scale(acts0), estimate_scale(acts1)
    x0, x1 = acts0.float() * s0, acts1.float() * s1
    X = torch.stack([x0, x1], dim=1)               # [N, 2, d_in]

    cc = TopKCrossCoder(cfg)
    opt = torch.optim.Adam(cc.parameters(), lr=cfg.lr)
    N = X.shape[0]
    fire_count = torch.zeros(cfg.dict_size, device=dev)

    for ep in range(cfg.epochs):
        perm = torch.randperm(N)
        ep_recon = 0.0
        ep_fire = torch.zeros(cfg.dict_size, device=dev)
        nb = 0
        for i in range(0, N, cfg.batch_size):
            xb = X[perm[i:i + cfg.batch_size]].to(dev)
            recon, z = cc.loss(xb)
            opt.zero_grad()
            recon.backward()
            opt.step()
            ep_recon += recon.item()
            ep_fire += (z.detach() > 0).float().sum(0)
            nb += 1
        fire_count += ep_fire
        n_dead = int((ep_fire == 0).sum().item())
        if verbose:
            # variance-explained on a held-in batch
            with torch.no_grad():
                xb = X[:min(N, 8192)].to(dev)
                xh, _ = cc(xb)
                ss_res = (xb - xh).pow(2).sum().item()
                ss_tot = (xb - xb.mean(0)).pow(2).sum().item()
                ev = 1 - ss_res / (ss_tot + 1e-8)
            print(f"  ep {ep + 1:3d}/{cfg.epochs} recon={ep_recon / nb:.4f} "
                  f"EV={ev:.4f} dead={n_dead}/{cfg.dict_size}", flush=True)
        if (cfg.resample_every and (ep + 1) % cfg.resample_every == 0
                and ep + 1 < cfg.epochs):
            _resample_dead(cc, X, ep_fire, opt)

    alive = (fire_count > 0)
    with torch.no_grad():
        xb = X.to(dev)
        xh, z = cc(xb)
        ss_res = (xb - xh).pow(2).sum().item()
        ss_tot = (xb - xb.mean(0)).pow(2).sum().item()
        metrics = {
            "explained_variance": 1 - ss_res / (ss_tot + 1e-8),
            "l0": (z > 0).float().sum(-1).mean().item(),
            "dead_frac": (~alive).float().mean().item(),
            "scale0": s0, "scale1": s1,
        }
    return cc.cpu(), metrics, alive.cpu()


@torch.no_grad()
def _resample_dead(cc: TopKCrossCoder, X: torch.Tensor,
                   fire: torch.Tensor, opt: torch.optim.Optimizer) -> int:
    """Reinit dead latents toward high-error inputs (Anthropic-style)."""
    dead = (fire == 0)
    n_dead = int(dead.sum().item())
    if n_dead == 0:
        return 0
    dev = cc.cfg.device
    idx = torch.randperm(len(X))[:min(len(X), 8192)]
    xb = X[idx].to(dev)
    xh, _ = cc(xb)
    err = (xb - xh).pow(2).sum(dim=(1, 2))          # per-token error
    probs = err / (err.sum() + 1e-8)
    pick = torch.multinomial(probs, n_dead, replacement=True)
    # use model-averaged activation direction as the new decoder/encoder dir
    dirs = xb[pick].mean(dim=1)                      # [n_dead, d_in]
    dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)
    di = dead.nonzero(as_tuple=True)[0]
    for m in range(cc.cfg.n_models):
        cc.W_dec.data[di, m, :] = dirs * cc.cfg.dec_init_norm
        cc.W_enc.data[m, :, di] = dirs.T
    cc.b_enc.data[di] = 0.0
    for p in (cc.W_enc, cc.W_dec, cc.b_enc):
        st = opt.state.get(p)
        if st:
            st.get("exp_avg", torch.zeros(0)).zero_()
            st.get("exp_avg_sq", torch.zeros(0)).zero_()
    return n_dead
