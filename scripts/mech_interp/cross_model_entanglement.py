"""
Cross-model SAE entanglement analysis for structure-aware model merging.

Why this exists
---------------
Task1 (PT pursuit) and Task2 (battery management) each have a TopK SAE trained
on their own activation space (same 128-dim ℝ¹²⁸, different policy weights).
The SAE decoder columns W_dec[:, f] ∈ ℝ¹²⁸ are unit-normed directions in
activation space that latent f writes to. Two concepts from DIFFERENT models are
"entangled" if their decoder columns span overlapping subspaces in ℝ¹²⁸ — meaning
when you merge the two policies you cannot keep one task's computation without
disturbing the other's.

Example: if model1's PT-localized latents and model2's BS-localized latents
both live in the same subspace of ℝ¹²⁸, a naive merge will smear PT and BS
information onto the same activation directions → the merged agent confuses them.

Metric: subspace similarity = ||U1ᵀ U2||_F² / min(r1, r2)
  - 0 = orthogonal (no entanglement, merge is clean)
  - 1 = identical subspaces (maximum entanglement)
where U1, U2 are orthonormal bases of the two concept subspaces (via thin SVD).

Requires
--------
causal_label_features - Copy.py must have been run on BOTH models first:
  python "causal_label_features - Copy.py" --model task1_v8
  python "causal_label_features - Copy.py" --model task2_v3
This produces feature_labels_causal_ig.json with per-latent concept + purity.

Usage
-----
  python cross_model_entanglement.py --model1 task1_v8 --model2 task2_v3
  python cross_model_entanglement.py --model1 task1_v8 --model2 task2_v3 \\
      --layer resid.2 --purity-cut 0.5 --plot
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from topk_sae import TopKSAE, SAEConfig   # noqa: E402

PROJECT  = r"C:\Users\shahvandit\Downloads\Drone23_curent\drone_battery1"
RESULTS  = os.path.join(PROJECT, "results")
SAE_DIR  = os.path.join(RESULTS, "mech_interp", "sae")

CONCEPT_ORDER = ["PT", "BS", "battery", "gate", "velocity", "temporal", "BS2"]
TASK1_CONCEPTS = {"PT"}
TASK2_CONCEPTS = {"battery", "gate", "BS"}


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_labels(model: str, layer: str) -> list[dict]:
    """Load per-feature label dicts from the IG JSON for (model, layer)."""
    p = os.path.join(SAE_DIR, model, "feature_labels_causal_ig.json")
    if not os.path.exists(p):
        sys.exit(f"[ABORT] missing IG labels: {p}\n"
                 "Run causal_label_features - Copy.py first.")
    with open(p) as fh:
        j = json.load(fh)
    if layer not in j:
        sys.exit(f"[ABORT] layer '{layer}' not in {p}. "
                 f"Available: {list(j.keys())}")
    return j[layer]["features"]


def load_sae(model: str, layer: str, expansion: int, k: int) -> TopKSAE:
    """Load the matching SAE checkpoint (expansion + k filter)."""
    pattern = os.path.join(SAE_DIR, model, "sae_*.pt")
    for sf in sorted(glob.glob(pattern)):
        ck = torch.load(sf, map_location="cpu")
        if ck.get("layer") != layer:
            continue
        if expansion is not None and ck.get("expansion") != expansion:
            continue
        if k is not None and ck.get("k") != k:
            continue
        cfg = SAEConfig(d_in=ck["d_in"], expansion=ck["expansion"],
                        k=ck["k"], device="cpu")
        sae = TopKSAE(cfg)
        sae.load_state_dict(ck["sae_state_dict"], strict=False)
        sae.eval()
        return sae
    sys.exit(f"[ABORT] no SAE matched model={model} layer={layer} "
             f"expansion={expansion} k={k}\n  searched: {pattern}")


# ─────────────────────────────────────────────────────────────────────────────
# Subspace helpers
# ─────────────────────────────────────────────────────────────────────────────

def subspace_basis(vecs: torch.Tensor, eps_ratio: float = 1e-3
                   ) -> torch.Tensor | None:
    """Orthonormal basis for span(vecs).  vecs: [n, d] → U: [d, rank] or None."""
    if vecs.shape[0] == 0:
        return None
    U, S, _ = torch.linalg.svd(vecs.float(), full_matrices=False)
    # U is [n, r], S is [r] — but we need column space of vecs, so use V:
    _, S, Vt = torch.linalg.svd(vecs.float(), full_matrices=False)
    rank = int((S > eps_ratio * S[0]).sum())
    if rank == 0:
        return None
    return Vt[:rank].t()   # [d, rank]  (columns = basis vectors)


def subspace_similarity(U1: torch.Tensor, U2: torch.Tensor) -> float:
    """||U1ᵀ U2||_F² / min(r1, r2). 0=orthogonal, 1=identical."""
    M = U1.t() @ U2                                    # [r1, r2]
    return float((M * M).sum()) / min(U1.shape[1], U2.shape[1])


def principal_cosines(U1: torch.Tensor, U2: torch.Tensor) -> torch.Tensor:
    """Cosines of principal angles between subspaces (singular values of U1ᵀU2)."""
    M = U1.t() @ U2
    return torch.linalg.svdvals(M)


# ─────────────────────────────────────────────────────────────────────────────
# Concept → decoder column set
# ─────────────────────────────────────────────────────────────────────────────

def concept_decoder_vecs(feats: list[dict], sae: TopKSAE,
                         concept: str, purity_cut: float) -> torch.Tensor:
    """Decoder columns [d, F_concept] for latents with ig_dominant_concept==concept
    and concept_purity >= purity_cut (falls back to ig_dominant_share if
    concept_purity not yet in JSON)."""
    # decoder.weight is [d_in, F] in nn.Linear(F, d_in), columns = latent dirs
    W_dec = sae.decoder.weight.detach()   # [d_in, F]

    idx = []
    for f in feats:
        if f.get("ig_dominant_concept") != concept:
            continue
        # prefer concept_purity (concept-level) over ig_dominant_share (dim-level)
        purity = f.get("concept_purity", f.get("ig_dominant_share", 0.0))
        if purity < purity_cut:
            continue
        idx.append(f["feature"])
    if not idx:
        return torch.zeros(0, W_dec.shape[0])
    cols = W_dec[:, idx]   # [d_in, n_latents]
    return cols.t()        # [n_latents, d_in]  (rows = individual directions)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model1", required=True,
                    help="navigation/task1 model name, e.g. task1_v8")
    ap.add_argument("--model2", required=True,
                    help="battery/task2 model name, e.g. task2_v3")
    ap.add_argument("--layer", default="resid.2")
    ap.add_argument("--expansion", type=int, default=8)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--purity-cut", type=float, default=0.50,
                    help="min concept_purity to include a latent in its concept "
                         "subspace (lower = more latents, noisier subspace)")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    print(f"\n{'═'*66}")
    print(f"  CROSS-MODEL ENTANGLEMENT:  {args.model1}  ×  {args.model2}"
          f"  /  {args.layer}")
    print(f"  purity_cut={args.purity_cut}  expansion={args.expansion}  k={args.k}")
    print(f"{'═'*66}")

    feats1 = load_labels(args.model1, args.layer)
    feats2 = load_labels(args.model2, args.layer)
    sae1   = load_sae(args.model1, args.layer, args.expansion, args.k)
    sae2   = load_sae(args.model2, args.layer, args.expansion, args.k)

    d = sae1.decoder.weight.shape[0]
    print(f"\n  activation dim d={d}  |  "
          f"F1={sae1.dict_size}  F2={sae2.dict_size}")

    # ── build per-concept subspace bases ────────────────────────────────────
    bases1, sizes1 = {}, {}
    bases2, sizes2 = {}, {}
    for c in CONCEPT_ORDER:
        vecs1 = concept_decoder_vecs(feats1, sae1, c, args.purity_cut)
        vecs2 = concept_decoder_vecs(feats2, sae2, c, args.purity_cut)
        sizes1[c] = vecs1.shape[0]
        sizes2[c] = vecs2.shape[0]
        bases1[c] = subspace_basis(vecs1)
        bases2[c] = subspace_basis(vecs2)

    print(f"\n  Latents per concept (purity>={args.purity_cut}):")
    print(f"  {'concept':<10} {'model1':>7} {'model2':>7}")
    for c in CONCEPT_ORDER:
        r1 = bases1[c].shape[1] if bases1[c] is not None else 0
        r2 = bases2[c].shape[1] if bases2[c] is not None else 0
        print(f"  {c:<10} {sizes1[c]:>5} (r={r1:>3})  {sizes2[c]:>5} (r={r2:>3})")

    # ── concept × concept entanglement matrix (model1 row, model2 col) ──────
    sim = {}
    for c1 in CONCEPT_ORDER:
        sim[c1] = {}
        for c2 in CONCEPT_ORDER:
            U1 = bases1[c1]
            U2 = bases2[c2]
            if U1 is None or U2 is None:
                sim[c1][c2] = float("nan")
            else:
                sim[c1][c2] = subspace_similarity(U1, U2)

    print(f"\n  ENTANGLEMENT MATRIX  (subspace_similarity, 0=clean, 1=identical)")
    print(f"  model1↓ / model2→")
    hdr = "  " + " " * 10 + "".join(f"{c[:4]:>7}" for c in CONCEPT_ORDER)
    print(hdr)
    for c1 in CONCEPT_ORDER:
        if bases1[c1] is None:
            continue
        row = f"  {c1:<10}"
        for c2 in CONCEPT_ORDER:
            v = sim[c1][c2]
            row += f"  {v:5.3f}" if v == v else "    nan"
        print(row)

    # ── task-level summary ───────────────────────────────────────────────────
    print(f"\n  ── TASK-LEVEL SUMMARY ────────────────────────────────────────")
    t1_vecs, t2_vecs = [], []
    for c in TASK1_CONCEPTS:
        v = concept_decoder_vecs(feats1, sae1, c, args.purity_cut)
        if v.shape[0]:
            t1_vecs.append(v)
    for c in TASK2_CONCEPTS:
        v = concept_decoder_vecs(feats2, sae2, c, args.purity_cut)
        if v.shape[0]:
            t2_vecs.append(v)

    if t1_vecs and t2_vecs:
        T1 = torch.cat(t1_vecs, dim=0)   # all task1 concept dirs
        T2 = torch.cat(t2_vecs, dim=0)   # all task2 concept dirs
        U_T1 = subspace_basis(T1)
        U_T2 = subspace_basis(T2)
        task_sim = (subspace_similarity(U_T1, U_T2)
                    if U_T1 is not None and U_T2 is not None else float("nan"))
        print(f"  task1 subspace (PT dirs, r={U_T1.shape[1] if U_T1 is not None else 0})"
              f"  vs  task2 subspace (battery/gate/BS, "
              f"r={U_T2.shape[1] if U_T2 is not None else 0})")
        print(f"  similarity = {task_sim:.4f}")
        if task_sim == task_sim:
            if task_sim < 0.15:
                verdict = "LOW → tasks are nearly orthogonal, merge is clean"
            elif task_sim < 0.35:
                verdict = "MODERATE → partial overlap, CSM/QKTM needed"
            else:
                verdict = "HIGH → substantial entanglement, simple soup will conflate tasks"
            print(f"  verdict: {verdict}")
        if U_T1 is not None and U_T2 is not None:
            cosines = principal_cosines(U_T1, U_T2)
            top5 = cosines[:min(5, len(cosines))]
            print(f"  principal cosines (top {len(top5)}): "
                  + " ".join(f"{v:.3f}" for v in top5.tolist()))
    else:
        print("  (insufficient latents for task-level comparison)")

    # ── danger pairs ─────────────────────────────────────────────────────────
    print(f"\n  ── DANGER PAIRS (sim > 0.25, cross-task) ────────────────────")
    danger = []
    for c1 in CONCEPT_ORDER:
        for c2 in CONCEPT_ORDER:
            v = sim[c1][c2]
            if v != v:
                continue
            # flag cross-task pairs (one task1 concept, one task2 concept)
            is_cross = ((c1 in TASK1_CONCEPTS and c2 in TASK2_CONCEPTS) or
                        (c1 in TASK2_CONCEPTS and c2 in TASK1_CONCEPTS))
            if is_cross and v > 0.25:
                danger.append((v, c1, c2))
    danger.sort(reverse=True)
    if danger:
        for v, c1, c2 in danger:
            print(f"  model1.{c1:<10} ↔  model2.{c2:<10}  sim={v:.3f}"
                  f"  ← ENTANGLED")
    else:
        print("  none above threshold — cross-task subspaces are separable")

    # ── same-concept shared dirs ─────────────────────────────────────────────
    print(f"\n  ── SAME-CONCEPT OVERLAP (shared dirs expected here) ──────────")
    for c in CONCEPT_ORDER:
        v = sim[c][c]
        if v != v or bases1[c] is None or bases2[c] is None:
            continue
        print(f"  {c:<10}  model1 ↔ model2  sim={v:.3f}"
              + ("  (both use same dirs — keep both)" if v > 0.3 else ""))

    # ── plot ─────────────────────────────────────────────────────────────────
    if args.plot:
        try:
            _plot(sim, args.model1, args.model2, args.layer, args.purity_cut)
        except ImportError:
            print("\n  [WARN] matplotlib not available, skipping plot")


def _plot(sim, model1, model2, layer, purity_cut):
    import matplotlib.pyplot as plt
    import numpy as np
    concepts = [c for c in CONCEPT_ORDER
                if not all(sim[c][c2] != sim[c][c2] for c2 in CONCEPT_ORDER)]
    n = len(concepts)
    mat = np.full((n, n), float("nan"))
    for i, c1 in enumerate(concepts):
        for j, c2 in enumerate(concepts):
            v = sim[c1][c2]
            if v == v:
                mat[i, j] = v

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="Reds")
    ax.set_xticks(range(n)); ax.set_xticklabels(concepts, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(concepts)
    ax.set_xlabel(f"model2 ({model2}) concept")
    ax.set_ylabel(f"model1 ({model1}) concept")
    ax.set_title(f"SAE subspace entanglement  {layer}  (purity≥{purity_cut})")
    plt.colorbar(im, ax=ax, label="subspace similarity")
    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if mat[i, j] > 0.5 else "black")
    plt.tight_layout()
    out = os.path.join(SAE_DIR, model1,
                       f"entanglement_{model1}_x_{model2}_{layer.replace('.','')}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  [plot] {out}")


if __name__ == "__main__":
    main()
