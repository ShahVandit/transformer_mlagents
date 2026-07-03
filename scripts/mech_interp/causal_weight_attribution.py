"""
Causal per-weight attribution of a TASK VECTOR to policy behavior, + ablation.

The thesis
----------
Existing merge methods weight parameters by statistics that are NOT causal:
soup (uniform), TIES (task-vector magnitude/sign), RegMean (input covariance),
AIM (activation magnitude), ESM (PCA variance of activation shift). Fisher is the
closest — E[(d log pi / d theta)^2] — but it is a single-point, squared-gradient
sensitivity of the LOSS that never references the base model, i.e. it is not about
the parameter CHANGE that produced the new behavior.

This script measures, for every weight, how much its CHANGE from base->task
causally contributed to the change in the policy's behavior (action mean mu),
via Integrated Gradients along the weight path:

    Delta            = theta_task - theta_base                       (task vector)
    u(x)             = normalize( mu_task(x) - mu_base(x) )          (per-state
                                                          behavior-change dir)
    g(theta)         = mean_x < mu(theta; x), u(x) >                 (scalar)
    A_ij             = Delta_ij * (1/M) sum_m  d g / d theta_ij |_{theta_base + a_m Delta}

By IG completeness, sum_ij A_ij ~= g(theta_task) - g(theta_base) = mean_x ||Delta mu(x)||,
so A_ij is a signed, path-integrated, base-referenced attribution of the behavior
change to each weight's change. |A_ij| is the per-weight causal importance for the
task. Computed per task vector; the merge keeps each task's high-|A| changes and
resolves weights both tasks claim (the weight-level analogue of the SAE
entanglement analysis).

Difference from Fisher (the reviewer question): A_ij is (a) about Delta, not the
endpoint; (b) path-integrated base->task, not single-point; (c) signed/attributive
via completeness, not a magnitude; (d) on the action mean mu (behavior), not the
PPO loss.

Honest caveats
--------------
1. IG is gradient-based: a path-integrated linearization, not a true do()-
   intervention. The ablation test below is the actual causal check.
2. "Important for behavior" != "give to one model": where both tasks have high
   |A| on the same weight, attribution alone cannot separate them.
3. O(n_params * N * M); viable only because this policy is ~279K params.

Ablation (the proof)
--------------------
Revert the top-k highest-|A| weights of theta_task back to theta_base and measure
the drop in g (projected behavior). If reverting high-|A| weights collapses g far
faster than reverting low-|A| or random weights, the attribution is causal. A flat
curve falsifies the whole direction. Pure forward passes — no env, no grad.

Usage
-----
  python causal_weight_attribution.py --base task1_v7 --task task2_v3 \\
      --capture task2_v3 --ig-steps 32 --max-tokens 4096
  python causal_weight_attribution.py --base task1_v7 --task task2_v3 --plot
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import torch
import torch.nn.functional as F

PROJECT = r"C:\Users\shahvandit\Downloads\Drone23_curent\drone_battery1"
RESULTS = os.path.join(PROJECT, "results")
CAP_DIR = os.path.join(RESULTS, "mech_interp", "captures")
CAPTURE_FILES = ("activations_combined.pt", "activations.pt")

MU_W = "action_model._continuous_distribution.mu.weight"
MU_B = "action_model._continuous_distribution.mu.bias"
BODY = "network_body."


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint -> flat differentiable weight dict (body + action-mean head)
# ─────────────────────────────────────────────────────────────────────────────

def load_weights(model: str, device) -> dict:
    ck = torch.load(os.path.join(RESULTS, model, "Drone", "checkpoint.pt"),
                    map_location="cpu")
    sd = ck["Policy"]
    w = {}
    for k, v in sd.items():
        if k.startswith(BODY):
            w[k[len(BODY):]] = v.to(device).float()
    if MU_W not in sd:
        sys.exit(f"[ABORT] {model}: missing action head {MU_W}")
    w["mu.weight"] = sd[MU_W].to(device).float()
    w["mu.bias"] = sd[MU_B].to(device).float()
    return w


def model_dims(w: dict, n_head: int):
    d_model = w["input_proj.weight"].shape[0]
    n_layer = len({int(m.group(1)) for k in w
                   if (m := re.match(r"qkv_layers\.(\d+)\.weight", k))})
    return d_model, n_layer, d_model // n_head


# ─────────────────────────────────────────────────────────────────────────────
# Functional forward to the action mean (differentiable in w)
# ─────────────────────────────────────────────────────────────────────────────

def forward_mu(w: dict, obs: torch.Tensor, d_model, n_layer, n_head, head_dim):
    """w: weight dict (leaf tensors). obs: [B, L, 99] -> mu [B, 2]."""
    B, L, _ = obs.shape

    def ln(x, name):
        return F.layer_norm(x, (d_model,), w[f"{name}.weight"],
                            w[f"{name}.bias"], eps=1e-5)

    x = F.linear(obs, w["input_proj.weight"], w["input_proj.bias"])
    x = x + w["temporal_pos_encoding"]
    for i in range(n_layer):
        xn = ln(x, f"norm1_layers.{i}")
        qkv = F.linear(xn, w[f"qkv_layers.{i}.weight"], w[f"qkv_layers.{i}.bias"])
        qkv = qkv.reshape(B, L, 3, n_head, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        o = (attn @ v).transpose(1, 2).reshape(B, L, d_model)
        o = F.linear(o, w[f"attn_out_layers.{i}.weight"],
                     w[f"attn_out_layers.{i}.bias"])
        x = x + w[f"attn_scales.{i}"] * o
        xn2 = ln(x, f"norm2_layers.{i}")
        ffn = F.linear(xn2, w[f"ffn_layers.{i}.0.weight"],
                       w[f"ffn_layers.{i}.0.bias"])
        ffn = F.gelu(ffn)
        ffn = F.linear(ffn, w[f"ffn_layers.{i}.3.weight"],
                       w[f"ffn_layers.{i}.3.bias"])
        x = x + w[f"ffn_scales.{i}"] * ffn
    x = ln(x, "final_norm")
    return F.linear(x[:, -1, :], w["mu.weight"], w["mu.bias"])   # [B, 2]


@torch.no_grad()
def mu_only(w, obs, dims, chunk=4096):
    d_model, n_layer, head_dim = dims["d_model"], dims["n_layer"], dims["head_dim"]
    out = []
    for s in range(0, obs.shape[0], chunk):
        ob = obs[s:s + chunk]
        out.append(forward_mu(w, ob, d_model, n_layer, dims["n_head"], head_dim))
    return torch.cat(out, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Attribution keys: only real, differentiable parameter tensors
# ─────────────────────────────────────────────────────────────────────────────

def attrib_keys(w: dict):
    return [k for k, v in w.items() if v.dtype.is_floating_point]


# ─────────────────────────────────────────────────────────────────────────────
# Integrated Gradients over the weight path
# ─────────────────────────────────────────────────────────────────────────────

def attribute(w_base, w_task, obs, u, dims, steps, chunk):
    """Returns A: dict of per-weight attribution tensors (same shapes as w)."""
    keys = attrib_keys(w_base)
    delta = {k: (w_task[k] - w_base[k]) for k in keys}
    grad_acc = {k: torch.zeros_like(w_base[k]) for k in keys}
    d_model, n_layer = dims["d_model"], dims["n_layer"]
    n_head, head_dim = dims["n_head"], dims["head_dim"]
    N = obs.shape[0]

    for m in range(1, steps + 1):
        a = m / steps
        # fresh leaf tensors at the interpolated point theta_base + a*delta
        wa = {}
        for k in w_base:
            if k in delta:
                t = (w_base[k] + a * delta[k]).detach().requires_grad_(True)
            else:
                t = w_base[k]
            wa[k] = t
        # g = mean_x <mu, u>; accumulate grad over chunks (.grad sums)
        for s in range(0, N, chunk):
            ob = obs[s:s + chunk]
            uu = u[s:s + chunk]
            mu = forward_mu(wa, ob, d_model, n_layer, n_head, head_dim)
            g = (mu * uu).sum() / N
            g.backward()
        for k in keys:
            grad_acc[k] += wa[k].grad.detach()
        print(f"    IG weight-path {m}/{steps}", flush=True)

    return {k: delta[k] * grad_acc[k] / steps for k in keys}


# ─────────────────────────────────────────────────────────────────────────────
# Ablation: revert top-|A| weights of task -> base, measure g collapse
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def ablation_curve(w_base, w_task, A, obs, u, dims, fracs, order):
    """For each fraction f, revert that fraction of task weights (chosen by
    `order`) back to base, return mean g = mean_x <mu, u>. order in
    {top, bottom, random}."""
    keys = attrib_keys(w_base)
    # flatten |A| across all attributed weights with an index map
    flat, idxmap = [], []
    for k in keys:
        a = A[k].abs().reshape(-1)
        flat.append(a)
        idxmap += [(k, i) for i in range(a.numel())]
    flat = torch.cat(flat)
    n = flat.numel()
    if order == "top":
        rank = torch.argsort(flat, descending=True)
    elif order == "bottom":
        rank = torch.argsort(flat, descending=False)
    else:
        rank = torch.randperm(n)

    N = obs.shape[0]
    gs = []
    for f in fracs:
        krev = int(f * n)
        revert = rank[:krev]
        # build a reverted weight dict (task, with selected entries set to base)
        w = {k: w_task[k].clone() for k in w_task}
        # group reverts by key for vectorized assignment
        bykey = {}
        for ridx in revert.tolist():
            k, i = idxmap[ridx]
            bykey.setdefault(k, []).append(i)
        for k, ii in bykey.items():
            flatk = w[k].reshape(-1)
            base_flat = w_base[k].reshape(-1)
            ii_t = torch.tensor(ii, device=flatk.device)
            flatk[ii_t] = base_flat[ii_t]
            w[k] = flatk.reshape(w_task[k].shape)
        mu = mu_only(w, obs, dims)
        g = (mu * u).sum().item() / N
        gs.append(g)
    return gs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="task1_v7")
    ap.add_argument("--task", default="task2_v3")
    ap.add_argument("--capture", default=None,
                    help="model whose obs capture to use as inputs "
                         "(default = --task; battery-task distribution)")
    ap.add_argument("--ig-steps", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cap_model = args.capture or args.task
    cap = next((os.path.join(CAP_DIR, cap_model, f) for f in CAPTURE_FILES
                if os.path.exists(os.path.join(CAP_DIR, cap_model, f))), None)
    if cap is None:
        sys.exit(f"[ABORT] no capture in {os.path.join(CAP_DIR, cap_model)}")

    d = torch.load(cap, map_location="cpu")
    obs = d.get("obs_window")
    if obs is None:
        obs = d["obs"].float().unsqueeze(1)
    obs = obs.float()
    if len(obs) > args.max_tokens:
        obs = obs[torch.randperm(len(obs))[:args.max_tokens]]
    obs = obs.to(dev)
    print(f"[inputs] {cap_model} obs={tuple(obs.shape)}  device={dev}")

    w_base = load_weights(args.base, dev)
    w_task = load_weights(args.task, dev)
    d_model, n_layer, head_dim = model_dims(w_base, args.n_head)
    dims = {"d_model": d_model, "n_layer": n_layer, "n_head": args.n_head,
            "head_dim": head_dim}
    if obs.shape[1] == 1:                      # tile single obs to seq_len
        L = w_base["temporal_pos_encoding"].shape[1]
        obs = obs.repeat(1, L, 1)
    print(f"[model] base={args.base} task={args.task}  d_model={d_model} "
          f"n_layer={n_layer} n_head={args.n_head}")

    # behavior-change direction u(x), fixed
    mu_base = mu_only(w_base, obs, dims)
    mu_task = mu_only(w_task, obs, dims)
    dmu = mu_task - mu_base
    g_gap = dmu.norm(dim=1).mean().item()
    u = dmu / dmu.norm(dim=1, keepdim=True).clamp(min=1e-8)
    print(f"[behavior] mean ||mu_task - mu_base|| = {g_gap:.4f}  "
          f"(this is g(task)-g(base), the total to be attributed)")

    # ── attribution ──────────────────────────────────────────────────────────
    A = attribute(w_base, w_task, obs, u, dims, args.ig_steps, args.chunk)

    total = sum(v.sum().item() for v in A.values())
    print(f"\n[completeness] sum(A) = {total:.4f}  vs  g_gap = {g_gap:.4f}  "
          f"(ratio {total / max(g_gap, 1e-8):.2f}; ~1.0 = IG path well resolved)")

    # per-tensor causal mass
    print("\n  per-parameter causal mass (sum |A|), descending:")
    masses = sorted(((float(v.abs().sum()), k) for k, v in A.items()),
                    reverse=True)
    tot_mass = sum(m for m, _ in masses) + 1e-12
    for mass, k in masses:
        print(f"    {k:<32} {mass:10.4f}  {mass / tot_mass:6.1%}")

    out = os.path.join(RESULTS, "mech_interp",
                       f"weight_attrib_{args.base}_to_{args.task}.pt")
    torch.save({"A": {k: v.cpu() for k, v in A.items()},
                "base": args.base, "task": args.task, "g_gap": g_gap,
                "capture": cap_model, "ig_steps": args.ig_steps}, out)
    print(f"\n[saved] {out}")

    # ── ablation (the causal proof) ───────────────────────────────────────────
    fracs = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 1.0]
    print("\n  ── ABLATION: revert task->base weights, measure g collapse ──")
    print("  (g starts high; reverting TOP-|A| should drop it fastest)")
    g_top = ablation_curve(w_base, w_task, A, obs, u, dims, fracs, "top")
    g_bot = ablation_curve(w_base, w_task, A, obs, u, dims, fracs, "bottom")
    g_rnd = ablation_curve(w_base, w_task, A, obs, u, dims, fracs, "random")
    print(f"    {'frac':>6} {'top-|A|':>9} {'bottom-|A|':>11} {'random':>9}")
    for i, f in enumerate(fracs):
        print(f"    {f:>6.2f} {g_top[i]:>9.4f} {g_bot[i]:>11.4f} {g_rnd[i]:>9.4f}")
    # area between top and random as a scalar causal-quality score
    import statistics
    auc_gap = statistics.mean(g_rnd[i] - g_top[i] for i in range(len(fracs)))
    print(f"\n  causal score (mean[g_random - g_top]) = {auc_gap:.4f}  "
          f"(>0 => high-|A| weights are causally responsible; ~0 => falsified)")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 4))
            plt.plot(fracs, g_top, "o-", label="revert top-|A|")
            plt.plot(fracs, g_bot, "s-", label="revert bottom-|A|")
            plt.plot(fracs, g_rnd, "^-", label="revert random")
            plt.xlabel("fraction of task weights reverted to base")
            plt.ylabel("g = mean <mu, behavior-change dir>")
            plt.title(f"Causal ablation  {args.base} -> {args.task}")
            plt.legend(); plt.tight_layout()
            p = os.path.join(RESULTS, "mech_interp",
                             f"ablation_{args.base}_to_{args.task}.png")
            plt.savefig(p, dpi=150); plt.close()
            print(f"  [plot] {p}")
        except ImportError:
            print("  [WARN] matplotlib missing, skipping plot")


if __name__ == "__main__":
    main()
