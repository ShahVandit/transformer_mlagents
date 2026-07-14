"""
Causal Gram extraction (CSM v6, Stage 1).

For each concept (PT dims [86,87,94], energy dims [88,89,90,95,96]) and each
merged linear layer, build the CAUSAL GRAM: the second moment of the layer's
output-space activation shift under do(concept) input-resampling, weighted by
the action-gradient (AtP) so that shifts which actually reach the policy
output count more than shifts that die downstream:

    dX   = X_clean - X_do(c)          captured at the linear's INPUT boundary
    dY   = dX @ W^T                   exact output-space shift of THIS linear
    dY~  = dY * gbar                  gbar[j] = E[ ||d mu / d y_j|| ]  (AtP)
    C_c  = E[ dY~^T dY~ ]             [d_out, d_out]

Eigendirections of C_c are the activation directions causally driven by the
concept (upstream propagation is included: dX already contains everything the
input intervention changed at this depth). Placebo Grams (random same-size
non-concept dim groups) provide the null for the census.

Positions are pooled (full_seq): attention mixes the 8-step window, so Q/K/V
shifts at non-last positions still reach the last-token action.

Reuses: PatchableBackend (forward + captures), ownership_map's donor-segment
resampling + placebo groups, resid_alignment_check's shared buffer.

Known-answer check printed at the end: task1_base (pure nav) must show
~zero causal energy for the energy concept.

v6.1 calibration upgrades (both on by default, disable with --no-* flags):
  1. VARIANCE-MATCHED PLACEBOS (default 20 groups): placebo groups are chosen
     so their injected input variance matches the concept's, instead of random
     groups whose perturbations measured 2-8x larger (a false-negative bias)
     with an ~8x group-to-group spread (a noisy null).
  2. CONCEPT-CONDITIONED GBAR: each concept's action-gradient weights are
     measured on that concept's behaviorally relevant windows (energy: urgency
     gate on), so context-gated circuits are not diluted by windows where
     their gradient to the action is ~0. Placebos share their concept's gbar.

Usage
-----
  python causal_gram.py --model task1_base
  python causal_gram.py --model task2_base_v2 --max-tokens 40000 --repeats 2
  python causal_gram.py --model X --no-variance-match --no-gbar-condition  # legacy
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from patchable_backend import PatchableBackend             # noqa: E402
from resid_alignment_check import load_shared_buffer       # noqa: E402
from ownership_map import (resample_dims, placebo_groups,  # noqa: E402
                           CONCEPTS)

PROJECT = r"C:\Users\shahvandit\Downloads\Drone23_curent\drone_battery1"
RESULTS = os.path.join(PROJECT, "results")
SOUP_DIR = os.path.join(PROJECT, "combined_models")
OUT_DIR = os.environ.get(
    "MECH_GRAM_DIR",
    os.path.join(RESULTS, "mech_interp", "causal_grams"),
)

URGENCY_DIM = 96      # charging-urgency gate (last position of the window)

# Concept-conditioned gbar: the action-gradient through a context-gated circuit
# is ~0 on windows where the gate is off, so averaging gbar over the whole
# buffer dilutes gated skills (energy fires on ~20% of combined-mode windows ->
# ~5x understated) — a systematic false-negative source. Each concept's gbar is
# therefore measured on the windows where that concept is behaviorally relevant;
# a concept's placebo groups use the same filter (fair null). None = all windows.
GBAR_FILTERS = {
    "energy": lambda buf: buf[:, -1, URGENCY_DIM] > 0.5,
    "PT":     None,
}


def concept_of(group_name: str) -> str:
    """'energy' -> 'energy';  'plc_energy_3' -> 'energy'."""
    return group_name if group_name in CONCEPTS else group_name.split("_")[1]


def checkpoint_for_model(model: str) -> str:
    """Resolve model names without introducing alias run directories.

    For soup_alpha_v2_a* the intended 0-step merge lives in combined_models.
    The same name may also exist under results/ after fine-tuning, so prefer
    combined_models here to avoid accidentally analyzing a resumed checkpoint.
    Fine-tuned staged checkpoints should use explicit names such as ft_v2_a0.5_12M.
    """
    if model.startswith("soup_alpha_v2_a"):
        soup = os.path.join(SOUP_DIR, f"{model}.pt")
        if os.path.exists(soup):
            return soup
    return os.path.join(RESULTS, model, "Drone", "checkpoint.pt")


# ─────────────────────────────────────────────────────────────────────────────
# Which linears get merged, and where their input/output live
# ─────────────────────────────────────────────────────────────────────────────

def linear_specs(backend):
    """One spec per merged linear:
        (name, input_boundary, weight_key, grad_node, out_slice)
    input_boundary "obs" means the raw observation window (input_proj's input).
    grad_node is the key in the autograd forward's node dict; out_slice cuts
    the fused-qkv output into Q/K/V (each its own merge unit, as in CSM v3)."""
    d = backend.d_model
    specs = [("input_proj", "obs", "input_proj.weight", "input_proj", None)]
    for i in range(backend.n_layer):
        for nm, sl in (("Q", slice(0, d)), ("K", slice(d, 2 * d)),
                       ("V", slice(2 * d, 3 * d))):
            specs.append((f"qkv{nm}.{i}", f"qkv_in.{i}",
                          f"qkv_layers.{i}.weight", f"qkv.{i}", sl))
        specs.append((f"o.{i}", f"attn_in.{i}",
                      f"attn_out_layers.{i}.weight", f"o.{i}", None))
        specs.append((f"ffn1.{i}", f"ffn_in.{i}",
                      f"ffn_layers.{i}.0.weight", f"ffn1.{i}", None))
        specs.append((f"ffn2.{i}", f"ffn_hidden.{i}",
                      f"ffn_layers.{i}.3.weight", f"ffn2.{i}", None))
    return specs


# ─────────────────────────────────────────────────────────────────────────────
# AtP weights: per-channel action-gradient norms at each linear's output
# ─────────────────────────────────────────────────────────────────────────────

def _grad_nodes(be: PatchableBackend, obs: torch.Tensor):
    """Autograd-enabled replica of PatchableBackend._forward that exposes each
    merged linear's raw output as a graph node. obs must require grad so the
    graph exists; we never use d/d obs itself."""
    w = be.w
    B, L = obs.shape[0], be.seq_len

    def ln(x, name):
        return F.layer_norm(x, (be.d_model,), w[f"{name}.weight"],
                            w[f"{name}.bias"], eps=1e-5)

    def lin(x, name):
        return F.linear(x, w[f"{name}.weight"], w.get(f"{name}.bias"))

    nodes = {}
    y = lin(obs, "input_proj")
    nodes["input_proj"] = y
    x = y + w["temporal_pos_encoding"]
    for i in range(be.n_layer):
        xn = ln(x, f"norm1_layers.{i}")
        qkv = lin(xn, f"qkv_layers.{i}")
        nodes[f"qkv.{i}"] = qkv
        qkv_r = qkv.reshape(B, L, 3, be.n_head, be.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv_r[0], qkv_r[1], qkv_r[2]
        attn = torch.softmax(q @ k.transpose(-2, -1) * be.head_dim ** -0.5, dim=-1)
        a = (attn @ v).transpose(1, 2).reshape(B, L, be.d_model)
        o = lin(a, f"attn_out_layers.{i}")
        nodes[f"o.{i}"] = o
        x = x + w[f"attn_scales.{i}"] * o
        xn = ln(x, f"norm2_layers.{i}")
        f1 = lin(xn, f"ffn_layers.{i}.0")
        nodes[f"ffn1.{i}"] = f1
        f2 = lin(F.gelu(f1), f"ffn_layers.{i}.3")
        nodes[f"ffn2.{i}"] = f2
        x = x + w[f"ffn_scales.{i}"] * f2
    mu = F.linear(ln(x, "final_norm")[:, -1, :], be.mu_w, be.mu_b)
    return mu, nodes


def grad_colnorms(be: PatchableBackend, buf: torch.Tensor,
                  n_tokens: int = 4096, chunk: int = 512, seed: int = 0):
    """gbar[node][j] = sqrt(E_{samples,positions}[ sum_k (d mu_k / d y_j)^2 ]).
    Post-tanh action space (matches ownership_map's mediation metric)."""
    g = torch.Generator().manual_seed(seed)
    sub = buf[torch.randperm(len(buf), generator=g)[:n_tokens]].float()
    acc, cnt = {}, 0
    for s in range(0, len(sub), chunk):
        obs = sub[s:s + chunk].to(be.device).requires_grad_(True)
        mu, nodes = _grad_nodes(be, obs)
        mu = torch.tanh(mu)
        names = list(nodes)
        tensors = [nodes[n] for n in names]
        act_dim = mu.shape[1]
        for k in range(act_dim):
            grads = torch.autograd.grad(mu[:, k].sum(), tensors,
                                        retain_graph=(k < act_dim - 1))
            for n, gr in zip(names, grads):
                acc[n] = acc.get(n, 0) + gr.detach().pow(2).sum(dim=(0, 1)).cpu()
        cnt += obs.shape[0] * be.seq_len
    return {n: (a / cnt).sqrt().clamp(min=1e-9) for n, a in acc.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Causal Gram accumulation
# ─────────────────────────────────────────────────────────────────────────────

def build_causal_grams(backend: PatchableBackend, buf: torch.Tensor,
                       groups: dict, repeats: int = 2, chunk: int = 2048,
                       grad_tokens: int = 4096, seed: int = 0,
                       gbar_condition: bool = True, verbose: bool = True):
    """groups = {name: [obs dims]} (concepts + placebos). Returns
    {"grams": {linear: {group: C [d,d] float64}}, "gbar": ..., "specs": ...}.
    gbar_condition: measure each concept's action-gradient weights on that
    concept's relevant windows (GBAR_FILTERS) instead of the whole buffer."""
    specs = linear_specs(backend)
    gen = torch.Generator().manual_seed(seed)
    dev = backend.device

    gbars = {}
    for c in CONCEPTS:
        filt = GBAR_FILTERS.get(c) if gbar_condition else None
        sub = buf
        if filt is not None:
            mask = filt(buf)
            if int(mask.sum()) >= 512:
                sub = buf[mask]
            elif verbose:
                print(f"[gbar] {c}: only {int(mask.sum())} filtered windows; "
                      f"falling back to full buffer")
        gbars[c] = grad_colnorms(backend, sub, n_tokens=grad_tokens, seed=seed)
        if verbose:
            print(f"[gbar:{c}] windows={len(sub)}  channel norms per node: "
                  + "  ".join(f"{n}:{float(v.mean()):.2e}"
                              for n, v in gbars[c].items()))

    # donor permutations, one per (group, repeat) — deterministic given seed
    perms = {(g, r): torch.randperm(len(buf), generator=gen)
             for g in groups for r in range(repeats)}

    W = {name: backend.w[wkey][osl] if osl is not None else backend.w[wkey]
         for name, _, wkey, _, osl in specs}          # already on device
    gb = {c: {name: (gbars[c][node][osl] if osl is not None
                     else gbars[c][node]).to(dev)
              for name, _, _, node, osl in specs} for c in CONCEPTS}

    C = {name: {g: torch.zeros(W[name].shape[0], W[name].shape[0],
                               dtype=torch.float64)
                for g in groups} for name, *_ in specs}
    cap_bnds = sorted({inb for _, inb, *_ in specs if inb != "obs"})

    n_rows = 0
    buf = buf.float()
    for s in range(0, len(buf), chunk):
        ob_c = buf[s:s + chunk]
        caps_c = backend.forward(ob_c, capture=cap_bnds, full_seq=True)
        for gname, dims in groups.items():
            for r in range(repeats):
                ob_p = ob_c.clone()
                donor = buf[perms[(gname, r)][s:s + chunk]]
                ob_p[:, :, dims] = donor[:, :, dims]
                caps_p = backend.forward(ob_p, capture=cap_bnds, full_seq=True)
                gc = concept_of(gname)
                for name, inb, _, _, _ in specs:
                    dX = ((ob_c - ob_p) if inb == "obs"
                          else (caps_c[inb] - caps_p[inb])).to(dev)
                    dY = (dX @ W[name].T) * gb[gc][name]
                    dY = dY.reshape(-1, dY.shape[-1]).double()
                    C[name][gname] += (dY.T @ dY).cpu()
        n_rows += ob_c.shape[0] * backend.seq_len
        if verbose:
            print(f"  [gram] {min(s + chunk, len(buf))}/{len(buf)} windows")

    norm = n_rows * repeats
    for name in C:
        for g in C[name]:
            C[name][g] /= norm
    return {"grams": C,
            "gbar": {c: {k: v.cpu() for k, v in gbars[c].items()}
                     for c in gbars},                 # per-concept since v6.1
            "gbar_condition": gbar_condition,
            "specs": [(n, i, w, nd) for n, i, w, nd, _ in specs],
            "n_rows": n_rows, "repeats": repeats}


def make_groups(obs_dim: int, n_placebo: int, seed: int = 0,
                buf: torch.Tensor | None = None, verbose: bool = True):
    """Concepts + per-concept placebo groups (same size, non-concept dims).

    With buf given, placebo groups are VARIANCE-MATCHED: a donor swap injects
    2*sum(Var(dim)) of input variance, and unmatched random groups span an
    ~8x spread (rays and high-variance counters vs near-constant dims), which
    makes the rank-matched null both noisy and, on average, far stricter than
    the concept's own perturbation (measured 2-8x larger -> false negatives).
    Here we sample many candidate groups and keep the n_placebo whose injected
    variance is closest to the concept's, so the null is calibrated."""
    groups = dict(CONCEPTS)
    if buf is None:                                   # legacy unmatched null
        for c in CONCEPTS:
            for j, grp in enumerate(placebo_groups(obs_dim, len(CONCEPTS[c]),
                                                   n_placebo, seed=seed)):
                groups[f"plc_{c}_{j}"] = grp
        return groups

    var = buf.reshape(-1, obs_dim).float().var(dim=0)
    concept_dims = sorted({d for v in CONCEPTS.values() for d in v})
    pool = [i for i in range(obs_dim)
            if i not in concept_dims and float(var[i]) > 1e-8]
    g = torch.Generator().manual_seed(seed)
    for c, dims in CONCEPTS.items():
        target = float(var[dims].sum())
        cands = {}
        for _ in range(max(500, 50 * n_placebo)):
            grp = tuple(sorted(torch.tensor(pool)[
                torch.randperm(len(pool), generator=g)[:len(dims)]].tolist()))
            if grp not in cands:
                inj = float(var[list(grp)].sum())
                cands[grp] = abs(torch.log(torch.tensor(inj / target)).item())
        best = sorted(cands.items(), key=lambda kv: kv[1])[:n_placebo]
        ratios = []
        for j, (grp, _) in enumerate(best):
            groups[f"plc_{c}_{j}"] = list(grp)
            ratios.append(float(var[list(grp)].sum()) / target)
        if verbose:
            print(f"[placebo] {c}: {len(best)} variance-matched groups, "
                  f"injected-variance ratio to concept "
                  f"{min(ratios):.2f}-{max(ratios):.2f}")
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--buffer-models", nargs="+",
                    default=["soup_alpha_v2_a0.5"],
                    help="one or more capture folders whose obs_window tensors form the shared buffer")
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=40_000)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--n-placebo", type=int, default=20)
    ap.add_argument("--grad-tokens", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-variance-match", action="store_true",
                    help="legacy null: random placebo groups, not variance-matched")
    ap.add_argument("--no-gbar-condition", action="store_true",
                    help="legacy gbar: average action-gradient over all windows")
    args = ap.parse_args()

    ckpt = checkpoint_for_model(args.model)
    backend = PatchableBackend(ckpt, n_head=args.n_head)
    print(backend)

    buf = load_shared_buffer(args.buffer_models, args.max_tokens, seed=args.seed)
    print(f"[buffer] {tuple(buf.shape)}  concepts={list(CONCEPTS)}  "
          f"placebo={args.n_placebo}/concept  repeats={args.repeats}")

    groups = make_groups(backend.obs_dim, args.n_placebo, seed=args.seed,
                         buf=None if args.no_variance_match else buf)
    out = build_causal_grams(backend, buf, groups, repeats=args.repeats,
                             chunk=args.chunk, grad_tokens=args.grad_tokens,
                             seed=args.seed,
                             gbar_condition=not args.no_gbar_condition)
    out.update({"model": args.model, "concepts": CONCEPTS, "groups": groups,
                "buffer_models": args.buffer_models,
                "max_tokens": args.max_tokens, "seed": args.seed})

    # summary: total causal energy (trace) per linear per concept, vs placebo q95
    print(f"\n[causal energy] trace(C) per linear (placebo q95 in parens)")
    hdr = f"{'linear':<14}" + "".join(f"{c:>24}" for c in CONCEPTS)
    print(hdr + "\n" + "-" * len(hdr))
    for name in out["grams"]:
        row = f"{name:<14}"
        for c in CONCEPTS:
            tr = float(torch.diagonal(out["grams"][name][c]).sum())
            plc = torch.tensor([float(torch.diagonal(out["grams"][name][g]).sum())
                                for g in out["groups"] if g.startswith(f"plc_{c}_")])
            row += f"{tr:>13.3e} ({plc.quantile(0.95):.1e})"
        print(row)

    os.makedirs(OUT_DIR, exist_ok=True)
    save = os.path.join(OUT_DIR, f"grams_{args.model}.pt")
    torch.save(out, save)
    print(f"\n[saved] {save}")
    csv_path = os.path.join(OUT_DIR, f"causal_energy_{args.model}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "linear", "concept", "trace", "placebo_q95"])
        for name in out["grams"]:
            for c in CONCEPTS:
                tr = float(torch.diagonal(out["grams"][name][c]).sum())
                plc = torch.tensor([
                    float(torch.diagonal(out["grams"][name][g]).sum())
                    for g in out["groups"] if g.startswith(f"plc_{c}_")
                ])
                w.writerow([args.model, name, c, tr, float(plc.quantile(0.95))])
    print(f"[csv saved] {csv_path}")


if __name__ == "__main__":
    main()
