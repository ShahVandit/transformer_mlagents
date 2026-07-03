"""
Input -> neuron causal effect (permutation importance targeting neurons).

For each internal neuron, how much does changing an input concept change that
neuron's activation? Method (Breiman 2001 permutation importance / occlusion,
here targeting the neuron instead of the model output):

  1. baseline: A0 = neuron activations on the real obs window.
  2. do(concept): resample ONLY that concept's input dims across the batch
     (empirical-marginal shuffle -> in-distribution counterfactual), re-run,
     giving A_c.
  3. effect[neuron, concept] = mean_n |A_c - A0| / std(A0)   (relative change).

A neuron whose concept-effect is high is causally driven by that input concept.
Run per model; keep task1's PT-driven neurons and task2's energy-driven neurons
for the merge.

Usage
-----
  python input_neuron_effect.py --model task1_v8
  python input_neuron_effect.py --model task2_v3 --n-head 4
  python input_neuron_effect.py --model task1_v11 --n-head 12
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

CONCEPTS = {
    "PT":     [86, 87, 94],              # target position  -> task1 (navigation)
    "energy": [88, 89, 90, 95, 96],      # battery + BS pos + recharge gate -> task2
}


def resample_concept(window: torch.Tensor, dims: list) -> torch.Tensor:
    """do(concept): replace the concept's input dims with another random sample's
    values (same permutation across the whole L-window -> coherent, in-distribution)."""
    perm = torch.randperm(window.shape[0])
    w = window.clone()
    w[:, :, dims] = window[perm][:, :, dims]
    return w


def pick_control_dims(obs_dim: int, concept_dims: set, k: int, seed: int = 0) -> list:
    """A negative control: k random dims that are NOT part of any real concept.
    Permuting these gives the noise floor of the permutation-importance effect,
    so a concept effect is only meaningful when it exceeds this floor."""
    g = torch.Generator().manual_seed(seed)
    pool = [i for i in range(obs_dim) if i not in concept_dims]
    idx = torch.randperm(len(pool), generator=g)[:k]
    return sorted(pool[i] for i in idx.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--concepts", default="PT,energy")
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=4,
                    help="average the effect over this many random resamples")
    ap.add_argument("--dom-ratio", type=float, default=1.5,
                    help="a neuron is concept-dominant if its top effect exceeds "
                         "the runner-up by this ratio")
    ap.add_argument("--ctrl-margin", type=float, default=1.0,
                    help="a concept effect must exceed (control floor * this) to "
                         "count as real; >1.0 is stricter")
    ap.add_argument("--max-tokens", type=int, default=60_000)
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
    if len(window) > args.max_tokens:
        window = window[torch.randperm(len(window))[:args.max_tokens]]
    print(f"[model] {args.model}  tokens={len(window)}  concepts={concepts}  "
          f"repeats={args.repeats}")

    backend = TransformerPolicyBackend(ckpt, n_head=args.n_head, device=dev)
    caps0 = backend.capture(window)                          # baseline {b: [N, d]}
    base_std = {b: caps0[b].std(0).clamp(min=1e-6) for b in caps0}

    # negative control: random non-concept dims of the same size as the largest
    # concept -> the noise floor of the permutation effect.
    concept_all = {d for c in concepts for d in CONCEPTS[c]}
    ctrl_k = max(len(CONCEPTS[c]) for c in concepts)
    ctrl_dims = pick_control_dims(window.shape[2], concept_all, ctrl_k)
    print(f"[control] permuting {ctrl_k} random non-concept dims {ctrl_dims} "
          f"as the noise floor")
    series = concepts + ["_ctrl"]
    concept_dims = {**{c: CONCEPTS[c] for c in concepts}, "_ctrl": ctrl_dims}

    # accumulate |A_c - A0| per concept (and the control), averaged over repeats
    eff = {b: {c: torch.zeros(caps0[b].shape[1]) for c in series} for b in caps0}
    for c in series:
        for _ in range(args.repeats):
            caps_c = backend.capture(resample_concept(window, concept_dims[c]))
            for b in caps0:
                eff[b][c] += (caps_c[b] - caps0[b]).abs().mean(0) / args.repeats
    # relative effect (per-neuron, normalized by baseline scale)
    for b in caps0:
        for c in series:
            eff[b][c] = eff[b][c] / base_std[b]

    def sort_key(k: str):
        if k == "encoding":
            return (99, 9)
        name, _, idx = k.partition(".")
        layer = int(idx) if idx.isdigit() else 0
        order = {"resid": 0, "qkv_in": 1, "attn_in": 2, "attn": 3,
                 "ffn_in": 4, "ffn_hidden": 5, "ffn": 6}.get(name, 7)
        return (layer, order)
    boundaries = sorted(caps0.keys(), key=sort_key)

    # A neuron is genuinely concept-driven only when its concept effect exceeds
    # the control noise floor. floor = per-neuron control effect * this margin.
    ctrl_margin = args.ctrl_margin

    # per boundary: mean effect per concept, control floor, and dominance counts
    hdr = (f"{'boundary':<16}{'d':>6}"
           + "".join(f"{c + ' mean':>11}{c + ' dom#':>10}" for c in concepts)
           + f"{'ctrl':>9}{'shared#':>9}")
    print("\n" + hdr)
    print("-" * len(hdr))
    out = {"model": args.model, "concepts": concepts,
           "control_dims": ctrl_dims, "ctrl_margin": ctrl_margin, "boundaries": {}}
    for b in boundaries:
        E = torch.stack([eff[b][c] for c in concepts], dim=1)   # [d, n_concept]
        floor = eff[b]["_ctrl"] * ctrl_margin                    # [d] per-neuron floor
        top, order = E.sort(dim=1, descending=True)
        # dominant: top concept beats runner-up by dom_ratio AND clears the
        # control floor (i.e. the input concept causally moves this neuron more
        # than shuffling random unrelated inputs does).
        sig = top[:, 0] > floor
        dominant = torch.where(
            sig & (top[:, 0] > args.dom_ratio * top[:, 1].clamp(min=1e-9)),
            order[:, 0], torch.full_like(order[:, 0], -1))       # -1 = shared/none
        row = f"{b:<16}{E.shape[0]:>6}"
        dom_counts = {}
        for ci, c in enumerate(concepts):
            row += f"{eff[b][c].mean():>11.4f}{int((dominant == ci).sum()):>10}"
            dom_counts[c] = int((dominant == ci).sum())
        shared = int((dominant == -1).sum())
        row += f"{eff[b]['_ctrl'].mean():>9.4f}{shared:>9}"
        print(row)
        out["boundaries"][b] = {
            "effect": {c: eff[b][c] for c in concepts},         # [d] per concept
            "control": eff[b]["_ctrl"],                          # [d] noise floor
            "dominant": dominant,                                # [d]  ci or -1
            "dom_counts": dom_counts, "shared": shared,
        }

    save = os.path.join(SAE_DIR, args.model, "input_neuron_effect.pt")
    os.makedirs(os.path.dirname(save), exist_ok=True)
    torch.save(out, save)
    print(f"\n[saved] {save}")
    print("  per boundary: effect[concept] = [d] relative sensitivity; "
          "control = [d] random-dim noise floor; dominant = [d] (concept that "
          "drives each neuron above the floor, -1=shared).")
    print("  VALIDATION: a real input->neuron effect must exceed the 'ctrl' column. "
          "If concept means <= ctrl, the effect is noise.")
    print("  merge: keep task1's PT-dominant neurons, task2's energy-dominant, soup shared.")


if __name__ == "__main__":
    main()
