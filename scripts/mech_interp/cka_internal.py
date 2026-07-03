"""
Per-internal-boundary CKA screen: how much do task1 and task2 diverge INSIDE
the network (not just at the residual output).

Linear CKA (Kornblith et al., ICML 2019, "Similarity of Neural Network
Representations Revisited"):

    CKA(X, Y) = ||Xc^T Yc||_F^2 / ( ||Xc^T Xc||_F * ||Yc^T Yc||_F )

computed on centered activations Xc, Yc that the two checkpoints produce for the
SAME observation batch. CKA in [0, 1]; 1 = identical representation (shared),
lower = model-specific divergence. This is the cheap, no-training pre-screen:
run it first, and only bother training a crosscoder where CKA shows real
divergence.

Boundaries screened (inputs/outputs of every mergeable weight, from the extended
policy_backend captures):
    qkv_in.i     input to qkv_layers.i          (norm1 out)
    attn_in.i    input to attn_out_layers.i     (concat heads)
    attn.i       output of attn_out_layers.i
    ffn_in.i     input to ffn_layers.i.0        (norm2 out)
    ffn_hidden.i input to ffn_layers.i.3        (post-gelu)
    ffn.i        output of ffn_layers.i.3
    resid.i      residual stream after layer i

Usage
-----
  python cka_internal.py --model-a task1_v8 --model-b task2_v3
  python cka_internal.py --model-a task1_v11 --model-b task2_v6 --n-head 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from policy_backend import TransformerPolicyBackend          # noqa: E402

PROJECT = r"C:\Users\shahvandit\Downloads\Drone23_curent\drone_battery1"
RESULTS = os.path.join(PROJECT, "results")
CAP_DIR = os.path.join(RESULTS, "mech_interp", "captures")
SAE_DIR = os.path.join(RESULTS, "mech_interp", "sae")
CAPTURE_FILE = "activations_combined.pt"


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between [N, d1] and [N, d2] activations (same N rows)."""
    Xc = X - X.mean(0, keepdim=True)
    Yc = Y - Y.mean(0, keepdim=True)
    xy = (Xc.t() @ Yc).pow(2).sum()               # ||Xc^T Yc||_F^2
    xx = (Xc.t() @ Xc).pow(2).sum().sqrt()        # ||Xc^T Xc||_F
    yy = (Yc.t() @ Yc).pow(2).sum().sqrt()        # ||Yc^T Yc||_F
    return float(xy / (xx * yy + 1e-12))


def load_shared_window(model_a: str, model_b: str, max_tokens: int) -> torch.Tensor:
    """Load the obs window from model-a's capture (both models are pushed through
    the SAME obs for a fair CKA)."""
    path = os.path.join(CAP_DIR, model_a, CAPTURE_FILE)
    if not os.path.exists(path):
        # fall back to model-b's capture
        path = os.path.join(CAP_DIR, model_b, CAPTURE_FILE)
    if not os.path.exists(path):
        sys.exit(f"[ABORT] no capture found for {model_a} or {model_b} in {CAP_DIR}")
    d = torch.load(path, map_location="cpu")
    window = d["obs_window"].float() if "obs_window" in d else \
        d["obs"].float().unsqueeze(1).repeat(1, 8, 1)
    if len(window) > max_tokens:
        idx = torch.randperm(len(window))[:max_tokens]
        window = window[idx]
    return window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-a", required=True, help="run-id, e.g. task1_v8 (nav)")
    ap.add_argument("--model-b", required=True, help="run-id, e.g. task2_v3 (battery)")
    ap.add_argument("--n-head", type=int, default=4,
                    help="4 for small (d_model=128), 12 for big (d_model=768)")
    ap.add_argument("--max-tokens", type=int, default=100_000)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck_a = os.path.join(RESULTS, args.model_a, "Drone", "checkpoint.pt")
    ck_b = os.path.join(RESULTS, args.model_b, "Drone", "checkpoint.pt")
    for p in (ck_a, ck_b):
        if not os.path.exists(p):
            sys.exit(f"[ABORT] missing checkpoint: {p}")

    print(f"[device] {dev}")
    window = load_shared_window(args.model_a, args.model_b, args.max_tokens)
    print(f"[window] {tuple(window.shape)}  (same obs pushed through both models)")

    ba = TransformerPolicyBackend(ck_a, n_head=args.n_head, device=dev)
    bb = TransformerPolicyBackend(ck_b, n_head=args.n_head, device=dev)
    print(f"[model-a] {args.model_a}: {ba}")
    print(f"[model-b] {args.model_b}: {bb}")
    if ba.n_layer != bb.n_layer or ba.d_model != bb.d_model:
        sys.exit("[ABORT] architectures differ; CKA screen assumes same arch "
                 "(task2 should be initialized from task1)")

    caps_a = ba.capture(window)
    caps_b = bb.capture(window)
    keys = [k for k in caps_a if k in caps_b]

    # order: resid.0, then per-layer internal boundaries in forward order, resid.i
    def sort_key(k: str):
        if k == "encoding":
            return (99, 9)
        name, _, idx = k.partition(".")
        layer = int(idx) if idx.isdigit() else 0
        order = {"resid": 0, "qkv_in": 1, "attn_in": 2, "attn": 3,
                 "ffn_in": 4, "ffn_hidden": 5, "ffn": 6}.get(name, 7)
        return (layer, order)
    keys = sorted(keys, key=sort_key)

    print(f"\n{'boundary':<16}{'d':>6}{'CKA':>10}   (1.0 = shared, lower = model-specific)")
    print("-" * 46)
    report = {}
    for k in keys:
        Xa = caps_a[k].to(dev)
        Xb = caps_b[k].to(dev)
        cka = linear_cka(Xa, Xb)
        report[k] = cka
        bar = "#" * int((1 - cka) * 40)      # longer bar = more divergence
        print(f"{k:<16}{Xa.shape[1]:>6}{cka:>10.4f}   {bar}")

    # summary: which boundaries diverge most
    ranked = sorted(report.items(), key=lambda kv: kv[1])
    print("\n[most divergent boundaries] (lowest CKA = most model-specific)")
    for k, v in ranked[:8]:
        print(f"  {k:<16} CKA={v:.4f}")

    mean_cka = sum(report.values()) / len(report)
    print(f"\n[overall] mean CKA across {len(report)} boundaries = {mean_cka:.4f}")
    if mean_cka > 0.95:
        print("  => models are ~mostly SHARED internally. Little model-specific "
              "structure to decompose; a crosscoder would likely come out all-shared. "
              "Merge target = the sparse weight-delta, not an activation subspace.")
    else:
        print("  => real internal divergence exists. Worth training a crosscoder at "
              "the low-CKA boundaries to name the model-specific features.")

    out = os.path.join(SAE_DIR, f"cka_internal_{args.model_a}_vs_{args.model_b}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"model_a": args.model_a, "model_b": args.model_b,
                   "n_tokens": len(window), "cka": report,
                   "mean_cka": mean_cka}, f, indent=2)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
