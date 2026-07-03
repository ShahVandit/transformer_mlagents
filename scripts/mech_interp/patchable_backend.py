"""
Patchable inference backend for the TemporalTransformerBody policy.

Extends the read-only policy_backend.py idea with the two capabilities the
ownership-map pipeline needs and that capture-only code cannot provide:

  1. ACTION HEAD: the forward returns mu (the deterministic action mean), so
     causal effects can be scored on the policy OUTPUT, not just on internal
     activations ("carries the concept to the action" vs "wiggles with input").
  2. ACTIVATION PATCHING: any boundary tensor can be overwritten mid-forward,
     at ALL sequence positions (attention is bidirectional over the 8-step
     window, so last-token-only patching would leak the unpatched signal
     through the other positions at the next layer).

Boundary names (matching policy_backend.py capture keys):
    resid.0, qkv_in.i, attn_in.i, attn.i, ffn_in.i, ffn_hidden.i, ffn.i,
    resid.{i+1}, encoding

forward() semantics
-------------------
    out = backend.forward(obs_window, patches=..., capture=[...], full_seq=False)
    out["mu"]            [B, act_dim]   action mean (post final_norm -> mu head)
    out[boundary]        [B, d] (last token) or [B, L, d] if full_seq

patches = {boundary: (idx, values)} where idx is a 1-D LongTensor of neuron
indices and values is [B, L, len(idx)] — the tensor at that boundary gets
values written into those channels at every position, and the forward
continues from the modified tensor (downstream effects propagate).

Weight loading matches the checkpoint layout used everywhere in this project:
    "Policy" state_dict, body under "network_body.", action head at
    "action_model._continuous_distribution.mu.{weight,bias}".
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

BODY = "network_body."
MU_W = "action_model._continuous_distribution.mu.weight"
MU_B = "action_model._continuous_distribution.mu.bias"


class PatchableBackend:
    def __init__(self, checkpoint_path: str, n_head: int = 4,
                 device: Optional[str] = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # checkpoint_path may be a path OR an already-loaded checkpoint dict
        # (lets callers build e.g. an in-memory soup without touching disk).
        ck = (torch.load(checkpoint_path, map_location="cpu")
              if isinstance(checkpoint_path, str) else checkpoint_path)
        sd = ck["Policy"] if "Policy" in ck else ck
        self.w: Dict[str, torch.Tensor] = {
            k[len(BODY):]: v.to(self.device).float()
            for k, v in sd.items() if k.startswith(BODY)}
        if "input_proj.weight" not in self.w:
            raise KeyError(f"{checkpoint_path}: missing network_body.input_proj")
        if MU_W not in sd:
            raise KeyError(f"{checkpoint_path}: missing action head {MU_W}")
        self.mu_w = sd[MU_W].to(self.device).float()
        self.mu_b = sd[MU_B].to(self.device).float()

        self.obs_dim = self.w["input_proj.weight"].shape[1]
        self.d_model = self.w["input_proj.weight"].shape[0]
        self.seq_len = self.w["temporal_pos_encoding"].shape[1]
        self.n_head = n_head
        assert self.d_model % n_head == 0
        self.head_dim = self.d_model // n_head
        self.n_layer = len({int(m.group(1)) for k in self.w
                            if (m := re.match(r"qkv_layers\.(\d+)\.weight", k))})
        self.ckpt = (checkpoint_path if isinstance(checkpoint_path, str)
                     else "<in-memory>")

    # ── functional pieces ────────────────────────────────────────────────────
    def _ln(self, x, name):
        return F.layer_norm(x, (self.d_model,),
                            self.w[f"{name}.weight"], self.w[f"{name}.bias"],
                            eps=1e-5)

    def _lin(self, x, name):
        return F.linear(x, self.w[f"{name}.weight"], self.w.get(f"{name}.bias"))

    @staticmethod
    def _apply_patch(x, boundary, patches):
        """Overwrite channels of the full-seq tensor at this boundary, if asked."""
        if patches and boundary in patches:
            idx, vals = patches[boundary]
            x = x.clone()
            x[:, :, idx] = vals
        return x

    # ── forward ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _forward(self, obs, patches, capture, full_seq):
        B = obs.shape[0]
        L = self.seq_len
        if obs.dim() == 2:
            obs = obs.unsqueeze(1).repeat(1, L, 1)
        assert obs.shape[1] == L, f"window len {obs.shape[1]} != seq_len {L}"

        out: Dict[str, torch.Tensor] = {}

        def keep(name, x):
            if capture is None or name in capture:
                out[name] = (x if full_seq else x[:, -1, :]).clone()

        x = self._lin(obs, "input_proj") + self.w["temporal_pos_encoding"]
        x = self._apply_patch(x, "resid.0", patches)
        keep("resid.0", x)

        for i in range(self.n_layer):
            xn = self._ln(x, f"norm1_layers.{i}")
            xn = self._apply_patch(xn, f"qkv_in.{i}", patches)
            keep(f"qkv_in.{i}", xn)
            qkv = self._lin(xn, f"qkv_layers.{i}")
            qkv = qkv.reshape(B, L, 3, self.n_head, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = torch.softmax(
                q @ k.transpose(-2, -1) * self.head_dim ** -0.5, dim=-1)
            a = (attn @ v).transpose(1, 2).reshape(B, L, self.d_model)
            a = self._apply_patch(a, f"attn_in.{i}", patches)
            keep(f"attn_in.{i}", a)
            a = self._lin(a, f"attn_out_layers.{i}")
            a = self._apply_patch(a, f"attn.{i}", patches)
            keep(f"attn.{i}", a)
            x = x + self.w[f"attn_scales.{i}"] * a

            xn = self._ln(x, f"norm2_layers.{i}")
            xn = self._apply_patch(xn, f"ffn_in.{i}", patches)
            keep(f"ffn_in.{i}", xn)
            h = F.gelu(self._lin(xn, f"ffn_layers.{i}.0"))
            h = self._apply_patch(h, f"ffn_hidden.{i}", patches)
            keep(f"ffn_hidden.{i}", h)
            h = self._lin(h, f"ffn_layers.{i}.3")
            h = self._apply_patch(h, f"ffn.{i}", patches)
            keep(f"ffn.{i}", h)
            x = x + self.w[f"ffn_scales.{i}"] * h

            x = self._apply_patch(x, f"resid.{i + 1}", patches)
            keep(f"resid.{i + 1}", x)

        enc = self._ln(x, "final_norm")
        enc = self._apply_patch(enc, "encoding", patches)
        keep("encoding", enc)
        out["mu"] = F.linear(enc[:, -1, :], self.mu_w, self.mu_b)
        return out

    @torch.no_grad()
    def forward(self, obs: torch.Tensor,
                patches: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
                capture: Optional[List[str]] = None,
                full_seq: bool = False,
                batch_size: int = 4096) -> Dict[str, torch.Tensor]:
        """Batched forward. patches values must be [N, L, k] aligned with obs;
        they are sliced along the batch dim together with obs."""
        obs = obs.float()
        chunks: Dict[str, List[torch.Tensor]] = {}
        for s in range(0, len(obs), batch_size):
            e = min(s + batch_size, len(obs))
            p = None
            if patches:
                p = {b: (idx.to(self.device), vals[s:e].to(self.device))
                     for b, (idx, vals) in patches.items()}
            o = self._forward(obs[s:e].to(self.device), p, capture, full_seq)
            for kk, vv in o.items():
                chunks.setdefault(kk, []).append(vv.cpu())
        return {kk: torch.cat(vv) for kk, vv in chunks.items()}

    def boundaries(self) -> List[str]:
        names = ["resid.0"]
        for i in range(self.n_layer):
            names += [f"qkv_in.{i}", f"attn_in.{i}", f"attn.{i}",
                      f"ffn_in.{i}", f"ffn_hidden.{i}", f"ffn.{i}",
                      f"resid.{i + 1}"]
        return names + ["encoding"]

    def __repr__(self):
        return (f"PatchableBackend(obs_dim={self.obs_dim}, d_model={self.d_model}, "
                f"seq_len={self.seq_len}, n_layer={self.n_layer}, "
                f"n_head={self.n_head}, ckpt={self.ckpt})")
