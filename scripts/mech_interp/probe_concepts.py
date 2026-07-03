"""
Align internal activations with model-relevant inputs (concept probing).

Goal: at EVERY internal weight boundary, find the low-rank subspace of activation
space that encodes a given input concept, so the merge can preserve each model's
concept-aligned computation.

Validated method (linear probing / concept directions):
  Alain & Bengio 2016 (linear classifier probes); Gurnee et al. 2023 (sparse
  probing); Kim et al. 2018 (TCAV — the probe weight IS the concept direction).

For each boundary activation A [N, d] and concept target C [N, c] (the raw input
dims that define the concept), we fit a ridge probe  C ~= A @ W  (+bias). The
columns of W span the concept-aligned subspace in activation space; we
orthonormalize them (SVD) to get U [d, r], r = rank(concept) (PT=3, energy=5).
The projector onto the concept is  P = U U^T.

Why R2=1 is fine here: we are not discriminating concepts or proving
monosemanticity — we only need the concept DIRECTION. A high R2 means the
direction is pinned cleanly. The subspace is low-rank by construction (bounded
by the number of concept input dims), so it is well-defined regardless.

Boundaries probed (from the extended policy_backend):
  qkv_in.i attn_in.i attn.i ffn_in.i ffn_hidden.i ffn.i resid.i encoding

Usage
-----
  python probe_concepts.py --model task1_v8
  python probe_concepts.py --model task2_v3 --n-head 4
  python probe_concepts.py --model task1_v11 --n-head 12
"""

from __future__ import annotations

import argparse
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

# input dims that define each concept (the "model-relevant input")
CONCEPTS = {
    "PT":     [86, 87, 94],              # target position  -> task1 (navigation)
    "energy": [88, 89, 90, 95, 96],      # battery + BS pos + recharge-urgent gate -> task2
}


def standardize(x: torch.Tensor):
    mean = x.mean(0, keepdim=True)
    std = x.std(0, keepdim=True).clamp(min=1e-8)
    return (x - mean) / std, mean, std


def ridge_fit(X: torch.Tensor, Y: torch.Tensor, ridge: float) -> torch.Tensor:
    """Fit Y ~= [X, 1] @ W. Returns W [d+1, c]."""
    ones = torch.ones(X.shape[0], 1, dtype=X.dtype)
    Xb = torch.cat([X, ones], dim=1)
    eye = torch.eye(Xb.shape[1], dtype=X.dtype)
    eye[-1, -1] = 0.0                                   # don't penalize bias
    return torch.linalg.solve(Xb.t() @ Xb + ridge * eye, Xb.t() @ Y)


def r2_score(y: torch.Tensor, pred: torch.Tensor) -> float:
    sse = (y - pred).pow(2).sum()
    sst = (y - y.mean(0, keepdim=True)).pow(2).sum().clamp(min=1e-8)
    return float(1.0 - sse / sst)


def concept_subspace(A: torch.Tensor, C: torch.Tensor, ridge: float,
                     val_frac: float, energy_keep: float):
    """Fit a probe A->C and return (orthonormal basis U [d, r], val R2, rank).

    U spans the concept-aligned subspace; r <= n concept dims. energy_keep drops
    trailing singular directions of the probe weight that carry <(1-keep) of its
    Frobenius energy (so a rank-deficient concept map yields a tight basis).
    """
    An, _, _ = standardize(A)
    Cn, _, _ = standardize(C)
    n = An.shape[0]
    n_val = max(1, int(n * val_frac))
    perm = torch.randperm(n)
    va, tr = perm[:n_val], perm[n_val:]

    W = ridge_fit(An[tr], Cn[tr], ridge)                # [d+1, c]
    xvb = torch.cat([An[va], torch.ones(len(va), 1)], dim=1)
    r2 = r2_score(Cn[va], xvb @ W)

    Wc = W[:-1]                                         # [d, c]  drop bias row
    # orthonormal basis of the concept-aligned directions (column space of Wc)
    U, S, _ = torch.linalg.svd(Wc, full_matrices=False)  # U [d, c], S [c]
    if S.numel() > 1:
        cum = torch.cumsum(S.pow(2), 0) / S.pow(2).sum().clamp(min=1e-12)
        rank = int((cum < energy_keep).sum().item()) + 1
    else:
        rank = 1
    return U[:, :rank].contiguous(), r2, rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="run-id, e.g. task1_v8 / task2_v3")
    ap.add_argument("--concepts", default="PT,energy",
                    help="comma-separated subset of " + ",".join(CONCEPTS))
    ap.add_argument("--n-head", type=int, default=4,
                    help="4 for d_model=128, 12 for d_model=768")
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--energy-keep", type=float, default=0.99,
                    help="fraction of probe-weight energy retained -> subspace rank")
    ap.add_argument("--max-tokens", type=int, default=120_000)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = os.path.join(RESULTS, args.model, "Drone", "checkpoint.pt")
    cap = os.path.join(CAP_DIR, args.model, CAPTURE_FILE)
    for p in (ckpt, cap):
        if not os.path.exists(p):
            sys.exit(f"[ABORT] missing: {p}")
    concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]

    d = torch.load(cap, map_location="cpu")
    window = d["obs_window"].float()
    obs = d["obs"].float()
    if len(window) > args.max_tokens:
        idx = torch.randperm(len(window))[:args.max_tokens]
        window, obs = window[idx], obs[idx]
    print(f"[model] {args.model}  tokens={len(window)}  concepts={concepts}")

    backend = TransformerPolicyBackend(ckpt, n_head=args.n_head, device=dev)
    caps = backend.capture(window)                      # {boundary: [N, d]}

    def sort_key(k: str):
        if k == "encoding":
            return (99, 9)
        name, _, idx = k.partition(".")
        layer = int(idx) if idx.isdigit() else 0
        order = {"resid": 0, "qkv_in": 1, "attn_in": 2, "attn": 3,
                 "ffn_in": 4, "ffn_hidden": 5, "ffn": 6}.get(name, 7)
        return (layer, order)
    boundaries = sorted(caps.keys(), key=sort_key)

    targets = {c: obs[:, CONCEPTS[c]] for c in concepts}

    header = f"{'boundary':<16}{'d':>6}" + "".join(
        f"{c + ' R2':>12}{c + ' rank':>9}" for c in concepts)
    print("\n" + header)
    print("-" * len(header))

    out = {"model": args.model, "concepts": concepts, "boundaries": {}}
    for b in boundaries:
        A = caps[b].float()
        row = f"{b:<16}{A.shape[1]:>6}"
        out["boundaries"][b] = {}
        for c in concepts:
            U, r2, rank = concept_subspace(A, targets[c], args.ridge,
                                           args.val_frac, args.energy_keep)
            out["boundaries"][b][c] = {"basis": U.cpu(), "r2": r2, "rank": rank}
            row += f"{r2:>12.4f}{rank:>9d}"
        print(row)

    save = os.path.join(SAE_DIR, args.model, "concept_subspaces.pt")
    os.makedirs(os.path.dirname(save), exist_ok=True)
    torch.save(out, save)
    print(f"\n[saved] {save}")
    print("  each boundary/concept holds an orthonormal basis U [d, rank]; "
          "projector P = U @ U.T feeds the CSM merge.")


if __name__ == "__main__":
    main()
