"""
Standalone inference + activation-capture backend for the ML-Agents
TemporalTransformerBody policy, with ZERO dependency on mlagents.

Why standalone: for a fair crosscoder / CKA diff we must push the *same*
observation batch through two different checkpoints and read their internal
activations. Instantiating the full mlagents actor (ObservationSpec,
NetworkSettings, ActionSpec, env) is heavy and brittle. Instead we replicate
the body's forward exactly (verified against transformer_actor.py) and load
weights from the checkpoint's "Policy" state_dict by key.

Static-replay semantics
-----------------------
The live policy runs over a temporal sequence (length L=seq_len) built from
agent memory. For a controlled, matched-input comparison we use the
`memories=None` path, which the real forward takes at episode start: the single
observation is repeated across the L sequence positions and run through the
transformer. This is deterministic and identical in structure for both models,
so any activation difference is attributable to the *weights*, not to differing
histories. The encoding is the last-token residual after final_norm — exactly
the vector that feeds the action head.

Inferred from the checkpoint (no config needed):
    obs_dim   = input_proj.weight.shape[1]
    d_model   = input_proj.weight.shape[0]
    seq_len   = temporal_pos_encoding.shape[1]
    n_layer   = count of qkv_layers.*
Only `n_head` must be supplied (default 4, matching cbm.yaml).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

PREFIX = "network_body."


class TransformerPolicyBackend:
    """Loads one checkpoint and replays observations, capturing residual-stream
    activations at each layer boundary (last token = the policy's decision vector).

    Capture keys (each [N, d_model], last token):
        resid.0  : residual stream entering layer 0 (input_proj output + pos enc)
        resid.1  : residual stream entering layer 1 (after layer 0)
        ...
        resid.L  : residual stream after the last layer (pre final_norm)
        encoding : final_norm(resid.L) last token — feeds the action head
    """

    def __init__(self, checkpoint_path: str, n_head: int = 4,
                 device: Optional[str] = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(checkpoint_path, map_location="cpu")
        if "Policy" not in ck:
            raise KeyError(
                f"{checkpoint_path}: no 'Policy' section "
                f"(keys: {list(ck.keys())[:6]})")
        sd = ck["Policy"]
        # Keep only body tensors, strip the prefix for convenience.
        self.w: Dict[str, torch.Tensor] = {
            k[len(PREFIX):]: v.to(self.device).float()
            for k, v in sd.items() if k.startswith(PREFIX)
        }
        if "input_proj.weight" not in self.w:
            raise KeyError(
                f"{checkpoint_path}: missing 'network_body.input_proj.weight'. "
                f"Sample keys: {list(sd.keys())[:6]}")

        self.obs_dim = self.w["input_proj.weight"].shape[1]
        self.d_model = self.w["input_proj.weight"].shape[0]
        self.seq_len = self.w["temporal_pos_encoding"].shape[1]
        self.n_head = n_head
        assert self.d_model % n_head == 0, \
            f"d_model {self.d_model} not divisible by n_head {n_head}"
        self.head_dim = self.d_model // n_head
        self.n_layer = self._count_layers()
        self.ckpt = checkpoint_path

    def _count_layers(self) -> int:
        idxs = set()
        for k in self.w:
            m = re.match(r"qkv_layers\.(\d+)\.weight", k)
            if m:
                idxs.add(int(m.group(1)))
        return len(idxs)

    # ── functional building blocks ──────────────────────────────────────────
    def _ln(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return F.layer_norm(x, (self.d_model,),
                            self.w[f"{name}.weight"], self.w[f"{name}.bias"],
                            eps=1e-5)

    def _lin(self, x: torch.Tensor, name: str) -> torch.Tensor:
        b = self.w.get(f"{name}.bias")
        return F.linear(x, self.w[f"{name}.weight"], b)

    @torch.no_grad()
    def _forward_capture(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """obs: either [B, obs_dim] (single obs -> tiled, static-replay) or
        [B, L, obs_dim] (the REAL temporal window the policy consumed). Returns
        last-token captures. Real windows make the analyzed activation match the
        behavior-generating computation (temporal attention + pos-encoding)."""
        B = obs.shape[0]
        if obs.dim() == 3:
            # Real-window replay: use the genuine sliding history as-is.
            assert obs.shape[1] == self.seq_len, (
                f"window len {obs.shape[1]} != seq_len {self.seq_len}")
            L = self.seq_len
            obs_seq = obs                                    # [B, L, obs_dim]
        else:
            # Static replay: repeat the single obs across the sequence.
            L = self.seq_len
            obs_seq = obs.unsqueeze(1).repeat(1, L, 1)       # [B, L, obs_dim]
        x = self._lin(obs_seq, "input_proj")                 # [B, L, d_model]
        x = x + self.w["temporal_pos_encoding"]              # [1, L, d_model]

        cap: Dict[str, torch.Tensor] = {}
        cap["resid.0"] = x[:, -1, :].clone()

        for i in range(self.n_layer):
            x_norm = self._ln(x, f"norm1_layers.{i}")
            qkv = self._lin(x_norm, f"qkv_layers.{i}")       # [B, L, 3*d_model]
            qkv = qkv.reshape(B, L, 3, self.n_head, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            scale = self.head_dim ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = torch.softmax(attn, dim=-1)               # dropout off (eval)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).reshape(B, L, self.d_model)
            out = self._lin(out, f"attn_out_layers.{i}")
            x = x + self.w[f"attn_scales.{i}"] * out

            x_norm2 = self._ln(x, f"norm2_layers.{i}")
            ffn = self._lin(x_norm2, f"ffn_layers.{i}.0")
            ffn = F.gelu(ffn)
            ffn = self._lin(ffn, f"ffn_layers.{i}.3")
            x = x + self.w[f"ffn_scales.{i}"] * ffn

            cap[f"resid.{i + 1}"] = x[:, -1, :].clone()

        x = self._ln(x, "final_norm")
        cap["encoding"] = x[:, -1, :].clone()
        return cap

    # ── public API ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def capture(self, obs: torch.Tensor, batch_size: int = 4096
                ) -> Dict[str, torch.Tensor]:
        """Run all observations through the policy, returning {key: [N, d_model]}
        on CPU. obs: [N, obs_dim] (cpu or device)."""
        obs = obs.float()
        chunks: Dict[str, List[torch.Tensor]] = {}
        for i in range(0, len(obs), batch_size):
            cap = self._forward_capture(obs[i:i + batch_size].to(self.device))
            for kk, vv in cap.items():
                chunks.setdefault(kk, []).append(vv.cpu())
        return {kk: torch.cat(vv, dim=0) for kk, vv in chunks.items()}

    @torch.no_grad()
    def validate_input_proj(self, obs: torch.Tensor,
                            obs_out_ref: torch.Tensor) -> float:
        """Sanity check: input_proj is applied per-token, so input_proj(obs) for
        the current (last) obs must equal the captured 'obs_out' from the live
        run. Returns max abs error; should be ~1e-4 or less (fp32)."""
        pred = self._lin(obs.float().to(self.device), "input_proj").cpu()
        return (pred - obs_out_ref.float()).abs().max().item()

    def __repr__(self) -> str:
        return (f"TransformerPolicyBackend(obs_dim={self.obs_dim}, "
                f"d_model={self.d_model}, seq_len={self.seq_len}, "
                f"n_layer={self.n_layer}, n_head={self.n_head}, "
                f"ckpt={self.ckpt})")
