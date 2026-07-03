"""
Causal labeling of SAE features by INPUT feature group, via Integrated Gradients.

Why this exists
---------------
`label_features.py` and `neuron_attribution.py` are both correlational:

  - neuron_attribution.py : Pearson corr(neuron, obs). PT and BS are spatially
    correlated in the arena, so a PT neuron also correlates with BS (spurious).
  - label_features.py     : trains a probe activation->concept then ablates. This
    measures how *decodable* an input group is from the residual stream. Because
    input_proj linearly embeds all 99 obs dims into the stream, BS is ~perfectly
    recoverable even in a nav-only model -> the impossible result BS > PT. It
    measures presence, not use.

This script implements the CaFE method (arXiv:2509.00749, "Causal Interpretation
of Sparse Autoencoder Features") adapted to a tabular RL observation: attribute
each SAE latent BACK TO THE RAW OBSERVATION with Integrated Gradients (the causal
input->feature direction), group-pool into the 6 obs groups, and score
monosemanticity by attribution concentration. Optionally validate causally with
insertion/deletion tests (the proof that distinguishes this from correlation).

The IG path runs through the entire trained body (input_proj -> attention -> FFN
-> SAE encoder), so an input the policy ignores (e.g. BS in task1) gets ~0
gradient and correctly scores ~0 -- unlike the probe.

Target scalar attributed: the PRE-TopK, post-ReLU feature activation
    pre_f = relu( W_enc[f] . (acts_norm - b_pre) + b_enc[f] )
so the TopK mask discontinuity does not break the path integral.

Usage
-----
  python causal_label_features.py --model task1_v8
  python causal_label_features.py --model task1_v8 --validate --plot
  python causal_label_features.py --model task2_v3 --layers resid.2 \
      --ig-steps 64 --baseline mean --max-tokens 20000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from policy_backend import TransformerPolicyBackend          # noqa: E402
from topk_sae import TopKSAE, SAEConfig                       # noqa: E402

PROJECT = r"C:\Users\shahvandit\Downloads\Drone23_curent\drone_battery1"
RESULTS = os.path.join(PROJECT, "results")
CAP_DIR = os.path.join(RESULTS, "mech_interp", "captures")
SAE_DIR = os.path.join(RESULTS, "mech_interp", "sae")
CAPTURE_FILES = ("activations_combined.pt", "activations.pt")

# obs-column groups (v2 layout, obs_dim=99). Verified identical to
# label_features.py:43-50 and separability_check.py:51-60.
GROUPS = {
    "PT":       [86, 87, 94],     # PTdirX, PTdirY, PTdist
    "BS":       [89, 90, 95],     # BSdirX, BSdirY, BSdist
    "battery":  [88],
    "gate":     [96],
    "velocity": [84, 85],
    "rays":     list(range(84)),
}
GROUP_ORDER = ["PT", "BS", "battery", "gate", "velocity", "rays"]
# Task-relevant (non-background) groups for the monosemantic-keep summary.
TASK_GROUPS = ["PT", "BS", "battery", "gate"]

# Per-dim layout: the 15 individual vector obs dims (global indices 84..98),
# named exactly per SimpleTask1.cs CollectObservations. We attribute each of
# these INDIVIDUALLY (no grouping) so a 1-dim concept (battery) and a 2-dim
# concept (PT dir) are never compared via a size-biased pool. The 84 ray dims
# are kept only as one background magnitude to flag obstacle-avoidance atoms.
VEC_DIMS = list(range(84, 99))   # 15 dims
DIM_NAMES = ["velX", "velZ", "PTdirX", "PTdirZ", "battery", "BSdirX", "BSdirZ",
             "BS2dirX", "BS2dirZ", "BS2dist", "PTdist", "BSdist", "gate",
             "ptCount", "stepFrac"]
RAYS = list(range(84))


# ─────────────────────────────────────────────────────────────────────────────
# Differentiable forward to a chosen layer (last token), built from backend.w
# and the backend's UNDECORATED _lin/_ln blocks (no torch.no_grad, so autograd
# flows). Mirrors policy_backend._forward_capture math exactly.
# ─────────────────────────────────────────────────────────────────────────────

def fwd_layer_single(backend: TransformerPolicyBackend, w: torch.Tensor,
                     layer: str) -> torch.Tensor:
    """w: [L, 99] -> last-token activation [d_model] at `layer`.
    Single-sample, fully differentiable (used inside jacrev)."""
    L = backend.seq_len
    x = backend._lin(w, "input_proj")                       # [L, d_model]
    x = x + backend.w["temporal_pos_encoding"][0]           # [L, d_model]
    if layer == "resid.0":
        return x[-1]
    nh, hd = backend.n_head, backend.head_dim
    for i in range(backend.n_layer):
        xn = backend._ln(x, f"norm1_layers.{i}")
        qkv = backend._lin(xn, f"qkv_layers.{i}")           # [L, 3*d_model]
        qkv = qkv.reshape(L, 3, nh, hd).permute(1, 2, 0, 3)  # [3, nh, L, hd]
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
        attn = attn.softmax(dim=-1)
        o = attn @ v                                        # [nh, L, hd]
        o = o.transpose(0, 1).reshape(L, backend.d_model)
        o = backend._lin(o, f"attn_out_layers.{i}")
        x = x + backend.w[f"attn_scales.{i}"] * o
        xn2 = backend._ln(x, f"norm2_layers.{i}")
        ffn = backend._lin(xn2, f"ffn_layers.{i}.0")
        ffn = F.gelu(ffn)
        ffn = backend._lin(ffn, f"ffn_layers.{i}.3")
        x = x + backend.w[f"ffn_scales.{i}"] * ffn
        if layer == f"resid.{i + 1}":
            return x[-1]
    x = backend._ln(x, "final_norm")
    return x[-1]                                            # "encoding"


def fwd_layer_batched(backend: TransformerPolicyBackend, w: torch.Tensor,
                      layer: str) -> torch.Tensor:
    """w: [B, L, 99] -> [B, d_model] last-token activation at `layer`."""
    B, L, _ = w.shape
    x = backend._lin(w, "input_proj")
    x = x + backend.w["temporal_pos_encoding"]              # [1, L, d_model]
    if layer == "resid.0":
        return x[:, -1, :]
    nh, hd = backend.n_head, backend.head_dim
    for i in range(backend.n_layer):
        xn = backend._ln(x, f"norm1_layers.{i}")
        qkv = backend._lin(xn, f"qkv_layers.{i}")
        qkv = qkv.reshape(B, L, 3, nh, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
        attn = attn.softmax(dim=-1)
        o = (attn @ v).transpose(1, 2).reshape(B, L, backend.d_model)
        o = backend._lin(o, f"attn_out_layers.{i}")
        x = x + backend.w[f"attn_scales.{i}"] * o
        xn2 = backend._ln(x, f"norm2_layers.{i}")
        ffn = backend._lin(xn2, f"ffn_layers.{i}.0")
        ffn = F.gelu(ffn)
        ffn = backend._lin(ffn, f"ffn_layers.{i}.3")
        x = x + backend.w[f"ffn_scales.{i}"] * ffn
        if layer == f"resid.{i + 1}":
            return x[:, -1, :]
    x = backend._ln(x, "final_norm")
    return x[:, -1, :]


def compute_jacobian(backend, x_last, hist, layer):
    """d(layer act)/d(last-token obs). x_last:[B,99] hist:[B,L-1,99] -> [B,d,99].

    Tries torch.func vmap(jacrev) (fast); falls back to a per-output autograd
    loop if torch.func is unavailable."""
    try:
        from torch.func import vmap, jacrev

        def single(xl, h):                                  # xl:[99] h:[L-1,99]
            w = torch.cat([h, xl.unsqueeze(0)], dim=0)      # [L,99]
            return fwd_layer_single(backend, w, layer)      # [d]
        return vmap(jacrev(single))(x_last, hist)           # [B,d,99]
    except Exception:
        xl = x_last.clone().requires_grad_(True)
        w = torch.cat([hist, xl.unsqueeze(1)], dim=1)       # [B,L,99]
        acts = fwd_layer_batched(backend, w, layer)         # [B,d]
        cols = []
        d = acts.shape[1]
        for j in range(d):
            g, = torch.autograd.grad(acts[:, j].sum(), xl, retain_graph=True)
            cols.append(g)
        return torch.stack(cols, dim=1)                     # [B,d,99]


# ─────────────────────────────────────────────────────────────────────────────
# SAE encode helpers (match topk_sae.TopKSAE.encode exactly)
# ─────────────────────────────────────────────────────────────────────────────

def load_sae(sf):
    ck = torch.load(sf, map_location="cpu")
    cfg = SAEConfig(d_in=ck["d_in"], expansion=ck["expansion"], k=ck["k"],
                    device="cpu")
    sae = TopKSAE(cfg)
    sae.load_state_dict(ck["sae_state_dict"], strict=False)
    sae.eval()
    return sae, ck


def pre_topk(sae, acts_norm):
    """relu(W_enc.(acts_norm - b_pre) + b_enc) -- the pre-TopK feature value."""
    return F.relu(sae.encoder(acts_norm - sae.b_pre))


# ─────────────────────────────────────────────────────────────────────────────
# Integrated Gradients: each INDIVIDUAL input dim -> SAE feature (no grouping)
# ─────────────────────────────────────────────────────────────────────────────

def ig_dim_attribution(backend, sae, ck, window, baseline_last, layer,
                       steps, chunk, device):
    """Return (vec_attrib [F,15], rays_attrib [F]).

    vec_attrib  = mean over N of |IG| for each of the 15 INDIVIDUAL vector dims
                  (no pooling, so 1-dim vs 2-dim concepts compete fairly).
    rays_attrib = mean over N of the total |IG| across the 84 ray dims, kept only
                  as a background magnitude to flag obstacle-avoidance atoms.

    IG_f,j = (x_j - x'_j) * mean_a d pre_f( x' + a (x-x') ) / d x_j , last token.
    """
    mean = ck["act_mean"].float().to(device)
    std = ck["act_std"].float().to(device)
    W_enc = sae.encoder.weight.detach().to(device)          # [F, d]
    F_ = W_enc.shape[0]
    n = window.shape[0]
    vec_sum = torch.zeros(F_, len(VEC_DIMS))
    rays_sum = torch.zeros(F_)
    seen = 0
    for s0 in range(0, n, chunk):
        wb = window[s0:s0 + chunk].to(device)               # [B,L,99]
        B = wb.shape[0]
        x = wb[:, -1, :]                                    # [B,99]
        hist = wb[:, :-1, :]
        xb = baseline_last.to(device).expand(B, -1)         # [B,99]
        diff = x - xb
        ig = torch.zeros(B, F_, 99, device=device)
        for m in range(1, steps + 1):
            a = m / steps
            xa = xb + a * diff                              # [B,99]
            J = compute_jacobian(backend, xa, hist, layer)  # [B,d,99]
            with torch.no_grad():
                wa = torch.cat([hist, xa.unsqueeze(1)], dim=1)
                acts = fwd_layer_batched(backend, wa, layer)  # [B,d]
                acts_norm = (acts - mean) / std
                s = pre_topk(sae, acts_norm)               # [B,F]
                mask = (s > 0).float()
            dnorm = J / std[None, :, None]                 # d acts_norm / d x
            ds = torch.einsum("fc,bcj->bfj", W_enc, dnorm)  # [B,F,99]
            ig += mask[:, :, None] * ds
        ig = ig * diff[:, None, :] / steps                 # [B,F,99]
        vec_sum += ig[:, :, VEC_DIMS].abs().sum(dim=0).cpu()        # [F,15]
        rays_sum += ig[:, :, RAYS].abs().sum(dim=2).sum(dim=0).cpu()  # [F]
        seen += B
        print(f"    IG {seen}/{n}", flush=True)
    return vec_sum / max(seen, 1), rays_sum / max(seen, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Causal validation: per-dim occlusion (ablate one input dim, measure drop)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def dim_occlusion(backend, sae, ck, window, baseline_last, layer, device,
                  n_val=512):
    """Per-dim ablation. For each of the 15 vector dims:
        | pre_f(full) - pre_f(full, dim j -> baseline) |  averaged over n_val.
    Returns [F,15]. Cheap (16 forwards). Used to confirm IG's per-dim ranking
    causally (argmax agreement = the proof this is causal, not correlational)."""
    mean = ck["act_mean"].float().to(device)
    std = ck["act_std"].float().to(device)
    idx = torch.randperm(window.shape[0])[:n_val]
    wv = window[idx].to(device)
    hist = wv[:, :-1, :]
    x = wv[:, -1, :]
    base = baseline_last.to(device).expand(x.shape[0], -1)

    def pre_of(last):
        w = torch.cat([hist, last.unsqueeze(1)], dim=1)
        return pre_topk(sae, (fwd_layer_batched(backend, w, layer) - mean) / std)

    pre_full = pre_of(x)                                    # [B,F]
    occ = torch.zeros(pre_full.shape[1], len(VEC_DIMS))
    for di, j in enumerate(VEC_DIMS):
        last = x.clone()
        last[:, j] = base[:, j]
        occ[:, di] = (pre_full - pre_of(last)).abs().mean(dim=0).cpu()
    return occ                                             # [F,15]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", default="resid.2,resid.1",
                    help="comma list; lead with resid.2 (action-readout)")
    ap.add_argument("--expansion", type=int, default=8,
                    help="only use SAEs with this expansion (None=any)")
    ap.add_argument("--k", type=int, default=16,
                    help="only use SAEs with this k (None=any)")
    ap.add_argument("--baseline", choices=["mean", "zero"], default="mean")
    ap.add_argument("--ig-steps", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=5_000)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--mono-share", type=float, default=0.50,
                    help="dominant single-dim share (of the 15) for a mono label")
    ap.add_argument("--share-cutoff", type=float, default=0.20,
                    help="per-dim share needed to count toward breadth")
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = os.path.join(RESULTS, args.model, "Drone", "checkpoint.pt")
    if not os.path.exists(ckpt):
        sys.exit(f"[ABORT] missing checkpoint: {ckpt}")

    cap = next((os.path.join(CAP_DIR, args.model, f) for f in CAPTURE_FILES
                if os.path.exists(os.path.join(CAP_DIR, args.model, f))), None)
    if cap is None:
        sys.exit(f"[ABORT] no capture in {os.path.join(CAP_DIR, args.model)}")
    sae_files = sorted(glob.glob(os.path.join(SAE_DIR, args.model, "sae_*.pt")))
    if not sae_files:
        sys.exit(f"[ABORT] no SAEs in {os.path.join(SAE_DIR, args.model)}")

    d = torch.load(cap, map_location="cpu")
    window = d.get("obs_window")
    if window is None:
        window = d["obs"].float().unsqueeze(1).repeat(1, 8, 1)
    window = window.float()
    if len(window) > args.max_tokens:
        window = window[torch.randperm(len(window))[:args.max_tokens]]
    print(f"[model] {args.model}  windows={tuple(window.shape)}  device={dev}")

    backend = TransformerPolicyBackend(ckpt, n_head=args.n_head, device=dev)
    print(f"[backend] {backend}")

    # baseline last-token obs
    if args.baseline == "mean":
        baseline_last = window[:, -1, :].mean(dim=0)
    else:
        baseline_last = torch.zeros(window.shape[2])

    layers = [l.strip() for l in args.layers.split(",") if l.strip()]
    sae_by_layer = {}
    for sf in sae_files:
        ck = torch.load(sf, map_location="cpu")
        if args.expansion is not None and ck["expansion"] != args.expansion:
            continue
        if args.k is not None and ck["k"] != args.k:
            continue
        sae_by_layer.setdefault(ck["layer"], sf)
    if not sae_by_layer:
        sys.exit(f"[ABORT] no SAE matched expansion={args.expansion} k={args.k} "
                 f"in {os.path.join(SAE_DIR, args.model)}")

    summary = {}
    for layer in layers:
        if layer not in sae_by_layer:
            print(f"[skip] no SAE for layer {layer} (have {sorted(sae_by_layer)})")
            continue
        sf = sae_by_layer[layer]
        sae, ck = load_sae(sf)
        sae.to(dev)
        F_ = sae.encoder.weight.shape[0]
        print(f"\n{'='*64}\n{layer}  E={ck['expansion']} k={ck['k']} F={F_}\n{'='*64}")

        # fire rate over real windows (post-TopK)
        with torch.no_grad():
            acts = backend.capture(window)[layer].float()
            acts_norm = (acts - ck["act_mean"].float()) / ck["act_std"].float()
            z = sae.encode(acts_norm.to(dev)).cpu()        # [N,F]
        fire_rate = (z > 0).float().mean(0)                # [F]
        alive = fire_rate > 1e-4

        vec_attrib, rays_attrib = ig_dim_attribution(
            backend, sae, ck, window, baseline_last, layer,
            args.ig_steps, args.chunk, dev)                # [F,15], [F]
        total = vec_attrib.sum(1).clamp(min=1e-12)
        shares = vec_attrib / total.unsqueeze(1)           # [F,15]
        dom = vec_attrib.argmax(1)                         # 0..14
        breadth = (shares >= args.share_cutoff).sum(1)
        dom_share = shares.max(1).values
        # atom is obstacle-avoidance (background) if its total ray attribution
        # exceeds its total vector-dim attribution
        rays_dominated = rays_attrib > vec_attrib.sum(1)
        mono = (alive & (dom_share >= args.mono_share) & (breadth == 1)
                & ~rays_dominated)

        occ = occ_dom = None
        if args.validate:
            print("  [validate] per-dim occlusion ...", flush=True)
            occ = dim_occlusion(backend, sae, ck, window, baseline_last,
                                layer, dev)                # [F,15]
            occ_dom = occ.argmax(1)

        feats = []
        for f in range(F_):
            feats.append({
                "feature": f,
                "fire_rate": float(fire_rate[f]),
                "dominant_dim": DIM_NAMES[int(dom[f])],
                "dominant_share": float(dom_share[f]),
                "breadth": int(breadth[f]),
                "monosemantic": bool(mono[f]),
                "rays_dominated": bool(rays_dominated[f]),
                "rays_attrib": float(rays_attrib[f]),
                "dim_attrib": {DIM_NAMES[di]: float(vec_attrib[f, di])
                               for di in range(len(VEC_DIMS))},
                "dim_shares": {DIM_NAMES[di]: float(shares[f, di])
                               for di in range(len(VEC_DIMS))},
                **({"occlusion_dim": DIM_NAMES[int(occ_dom[f])]}
                   if occ_dom is not None else {}),
            })

        n_alive = int(alive.sum())
        print(f"  alive={n_alive}/{F_}  monosemantic={int(mono.sum())}  "
              f"rays_dominated={int((alive & rays_dominated).sum())}")
        print("  per-dim histogram (alive features):")
        for di, name in enumerate(DIM_NAMES):
            n_dom = int((alive & (dom == di)).sum())
            n_mono = int((mono & (dom == di)).sum())
            print(f"    {name:<9} dom={n_dom:>4}  mono={n_mono:>4}")
        if occ_dom is not None:
            agree = int((alive & (dom == occ_dom)).sum())
            print(f"  IG-vs-occlusion argmax agreement (alive): "
                  f"{agree}/{n_alive} = {agree / max(n_alive, 1):.2f}  "
                  f"(high => IG ranking is causal)")

        summary[layer] = {
            "expansion": ck["expansion"], "k": ck["k"], "n_features": F_,
            "n_alive": n_alive, "n_monosemantic": int(mono.sum()),
            "baseline": args.baseline, "ig_steps": args.ig_steps,
            "mono_share": args.mono_share, "share_cutoff": args.share_cutoff,
            "dim_names": DIM_NAMES, "features": feats,
        }

    out = os.path.join(SAE_DIR, args.model, "feature_labels_causal_ig.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[saved] {out}")

    if args.plot:
        try:
            _plot(summary, args.model)
        except ImportError:
            print("  [WARN] matplotlib missing, skipping plot")


def _plot(summary, model):
    import matplotlib.pyplot as plt
    import numpy as np
    for layer, data in summary.items():
        feats = data["features"]
        alive = [f for f in feats if f["fire_rate"] > 1e-4]
        if not alive:
            continue
        M = np.array([[f["dim_shares"][n] for n in DIM_NAMES] for f in alive])
        order = np.argsort([f["dominant_dim"] for f in alive])
        fig, ax = plt.subplots(figsize=(9, max(4, len(alive) // 12)))
        im = ax.imshow(M[order], aspect="auto", cmap="magma", vmin=0, vmax=1)
        ax.set_xticks(range(len(DIM_NAMES)))
        ax.set_xticklabels(DIM_NAMES, rotation=45, ha="right")
        ax.set_ylabel("alive SAE feature (sorted by dominant dim)")
        ax.set_title(f"{model} / {layer}  causal IG per-dim shares")
        plt.colorbar(im, ax=ax, label="share of |IG|")
        plt.tight_layout()
        p = os.path.join(SAE_DIR, model,
                         f"causal_ig_{layer.replace('.', '')}.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"  [plot] {p}")


if __name__ == "__main__":
    main()
