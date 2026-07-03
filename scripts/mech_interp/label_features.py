"""
Label SAE features by their *causal contribution* to concept readouts.

This version avoids the old raw-correlation rule. It first trains a small
linear probe from the layer activation to each state-derived concept. By
default, each concept target is residualized against the other concepts first,
so "PT" means PT information not already explained by BS/battery/gate/velocity.

For each SAE feature f, ablation removes:

  z_f * decoder[:, f]

from the normalized residual stream. We pass that ablation through the concept
probe weights and score the resulting change in each concept readout. This is
an intervention score in SAE feature space, not a Pearson correlation.

Output: per-feature {fire_rate, dominant_group, concept_breadth,
group_shares, group_scores} JSON, a printed summary, and task-relevant
monosemantic features. Velocity/rays are included as competing background
groups, so shared/background features are not forced into PT/BS/battery/gate.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from policy_backend import TransformerPolicyBackend          # noqa: E402
from topk_sae import TopKSAE, SAEConfig                       # noqa: E402

PROJECT = r"C:\Users\shahvandit\Downloads\Drone23_curent\drone_battery1"
RESULTS = os.path.join(PROJECT, "results")
CAP_DIR = os.path.join(RESULTS, "mech_interp", "captures")
SAE_DIR = os.path.join(RESULTS, "mech_interp", "sae")
CAPTURE_FILE = "activations_combined.pt"

GROUPS = {
    "PT":       [86, 87, 94],
    "BS":       [89, 90, 95],
    "battery":  [88],
    "gate":     [96],
    "velocity": [84, 85],
    "rays":     list(range(84)),
}
TASK_GROUP = {"task1": "PT", "task2": "BS"}   # primary task-relevant group hint

# Concepts that define a task vs shared background. Breadth is computed over
# CONCEPT_GROUPS only: rays (obstacle avoidance) and velocity are used by both
# tasks, so a "PT + rays" feature is still treated as PT-relevant.
TASK_CONCEPT_GROUPS = ["PT", "BS", "battery", "gate"]
BACKGROUND_GROUPS = ["velocity", "rays"]
READOUT_GROUPS = TASK_CONCEPT_GROUPS + BACKGROUND_GROUPS
CONTROL_GROUPS = ["PT", "BS", "battery", "gate", "velocity"]


def _standardize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = x.mean(0, keepdim=True)
    std = x.std(0, keepdim=True).clamp(min=1e-8)
    return (x - mean) / std, mean, std


def _ridge_fit(X: torch.Tensor, Y: torch.Tensor, ridge: float) -> torch.Tensor:
    """Fit Y ~= [X, 1] @ W. Returns W [D+1, O]."""
    ones = torch.ones(X.shape[0], 1, dtype=X.dtype)
    Xb = torch.cat([X, ones], dim=1)
    eye = torch.eye(Xb.shape[1], dtype=X.dtype)
    eye[-1, -1] = 0.0  # do not penalize bias
    return torch.linalg.solve(Xb.t() @ Xb + ridge * eye, Xb.t() @ Y)


def _r2_score(y: torch.Tensor, pred: torch.Tensor) -> float:
    sse = (y - pred).pow(2).sum()
    sst = (y - y.mean(0, keepdim=True)).pow(2).sum().clamp(min=1e-8)
    return float(1.0 - sse / sst)


def _obs_group(obs: torch.Tensor, group: str) -> torch.Tensor:
    return obs[:, GROUPS[group]].float()


def _controls_for(obs: torch.Tensor, group: str) -> torch.Tensor:
    cols = []
    for g in CONTROL_GROUPS:
        if g != group:
            cols.extend(GROUPS[g])
    return obs[:, cols].float()


def make_concept_targets(obs: torch.Tensor, residualize: bool,
                         ridge: float, min_target_std: float) -> tuple[dict[str, torch.Tensor], dict]:
    """Build standardized concept targets.

    If residualize=True, remove the linear component predictable from the other
    concept groups first. This is the main guard against PT/BS spatial confounds.
    """
    targets = {}
    meta = {}
    for g in READOUT_GROUPS:
        raw = _obs_group(obs, g)
        raw_std = raw.std(0).mean().item()
        y, _, _ = _standardize(raw)
        if residualize and g in TASK_CONCEPT_GROUPS:
            c_raw = _controls_for(obs, g)
            c, _, _ = _standardize(c_raw)
            wc = _ridge_fit(c, y, ridge)
            cb = torch.cat([c, torch.ones(c.shape[0], 1)], dim=1)
            y = y - cb @ wc
            resid_std_before_norm = y.std(0).mean().item()
            y, _, _ = _standardize(y)
        else:
            resid_std_before_norm = raw_std
        valid = raw_std >= min_target_std and resid_std_before_norm >= min_target_std
        targets[g] = y
        meta[g] = {
            "raw_std": raw_std,
            "residual_std": resid_std_before_norm,
            "valid": valid,
        }
    return targets, meta


def train_concept_probes(acts_norm: torch.Tensor, targets: dict[str, torch.Tensor],
                         ridge: float, val_frac: float) -> tuple[dict, dict]:
    """Train layer-activation -> concept probes and return weights + val R2."""
    n = acts_norm.shape[0]
    n_val = max(1, int(n * val_frac))
    perm = torch.randperm(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    probes = {}
    metrics = {}
    for g, y in targets.items():
        w = _ridge_fit(acts_norm[train_idx], y[train_idx], ridge)
        xvb = torch.cat([
            acts_norm[val_idx],
            torch.ones(len(val_idx), 1, dtype=acts_norm.dtype),
        ], dim=1)
        pred = xvb @ w
        probes[g] = w[:-1]  # [d_model, concept_dim], bias has no ablation effect
        metrics[g] = _r2_score(y[val_idx], pred)
    return probes, metrics


def probe_causal_scores(z: torch.Tensor, sae: TopKSAE,
                        probes: dict[str, torch.Tensor]) -> torch.Tensor:
    """Effect of ablating each SAE feature on each concept probe.

    SAE decoder weight has shape [d_model, F]. Ablating feature f changes the
    normalized activation by -z_f * decoder[:, f]. A linear probe's output
    change is therefore -z_f * decoder[:, f]^T W_probe.
    """
    z_abs = z.abs().mean(dim=0)                         # [F]
    dec = sae.decoder.weight.detach()                   # [d_model, F]
    scores = []
    for g in READOUT_GROUPS:
        effect = dec.t() @ probes[g]                    # [F, concept_dim]
        score = z_abs * effect.pow(2).mean(dim=1).sqrt()
        scores.append(score)
    return torch.stack(scores, dim=1)                   # [F, n_readouts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--thresh", type=float, default=None,
                    help="alias for --mono-share")
    ap.add_argument("--share-cutoff", type=float, default=0.20,
                    help="concept share needed for a group to count toward breadth")
    ap.add_argument("--mono-share", type=float, default=0.65,
                    help="dominant concept share required for monosemantic label")
    ap.add_argument("--effect-cutoff", type=float, default=1e-6,
                    help="minimum total concept-effect needed for a feature to count")
    ap.add_argument("--effect-percentile", type=float, default=25.0,
                    help="percentile of alive total readout effect used as cutoff")
    ap.add_argument("--min-probe-r2", type=float, default=0.05,
                    help="minimum validation R2 for a concept label to be trusted")
    ap.add_argument("--min-target-std", type=float, default=1e-5,
                    help="minimum raw/residual target std for a concept to be valid")
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.20)
    ap.add_argument("--no-residualize", action="store_true",
                    help="do not remove other concept groups from each target")
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=150_000)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = os.path.join(RESULTS, args.model, "Drone", "checkpoint.pt")
    cap = os.path.join(CAP_DIR, args.model, CAPTURE_FILE)
    sae_files = sorted(glob.glob(os.path.join(SAE_DIR, args.model, "sae_*.pt")))
    if not sae_files:
        sys.exit(f"[ABORT] no SAEs in {os.path.join(SAE_DIR, args.model)} "
                 f"(run train_sae.py first)")

    d = torch.load(cap, map_location="cpu")
    window = d["obs_window"].float()
    obs = d["obs"].float()
    if len(window) > args.max_tokens:
        idx = torch.randperm(len(window))[:args.max_tokens]
        window, obs = window[idx], obs[idx]
    print(f"[model] {args.model}  tokens={len(window)}")

    backend = TransformerPolicyBackend(ckpt, n_head=args.n_head, device=dev)
    caps = backend.capture(window)

    targets, target_meta = make_concept_targets(
        obs, residualize=not args.no_residualize, ridge=args.ridge,
        min_target_std=args.min_target_std)
    print("[labels] probe-mediated causal labels; "
          f"residualize={not args.no_residualize}")
    print("[targets] " + "  ".join(
        f"{g}:raw={target_meta[g]['raw_std']:.2e},resid={target_meta[g]['residual_std']:.2e}"
        f"{'' if target_meta[g]['valid'] else '*'}"
        for g in READOUT_GROUPS))

    summary = {}
    for sf in sae_files:
        ck = torch.load(sf, map_location="cpu")
        layer = ck["layer"]
        acts = caps[layer].float()                           # [N, 128]
        mean, std = ck["act_mean"].float(), ck["act_std"].float()
        cfg = SAEConfig(d_in=ck["d_in"], expansion=ck["expansion"],
                        k=ck["k"], device="cpu")
        sae = TopKSAE(cfg)
        # state_dict carries act_mean/act_std buffers (read separately above)
        sae.load_state_dict(ck["sae_state_dict"], strict=False); sae.eval()
        with torch.no_grad():
            acts_norm = (acts - mean) / std
            z = sae.encode(acts_norm)                       # [N, F]

        F_ = z.shape[1]
        fire_rate = (z > 0).float().mean(0)                  # [F]
        probes, probe_r2 = train_concept_probes(
            acts_norm, targets, ridge=args.ridge, val_frac=args.val_frac)
        readout_score = probe_causal_scores(z, sae, probes)  # [F, G]
        readout_total = readout_score.sum(1).clamp(min=1e-8)
        readout_share = readout_score / readout_total.unsqueeze(1)
        mono_share = float(args.thresh) if args.thresh is not None else args.mono_share
        share_cutoff = args.share_cutoff
        alive = fire_rate > 1e-4
        if int(alive.sum()) > 0:
            pct_cutoff = torch.quantile(
                readout_total[alive], args.effect_percentile / 100.0).item()
        else:
            pct_cutoff = 0.0
        effect_cutoff = max(args.effect_cutoff, pct_cutoff)

        task_idx = [READOUT_GROUPS.index(g) for g in TASK_CONCEPT_GROUPS]
        task_share = readout_share[:, task_idx]
        task_score = readout_score[:, task_idx]
        breadth = (task_share >= share_cutoff).sum(1)        # # task groups hit
        dom_readout = readout_score.argmax(1)                # arg over all groups
        dom_task = task_score.argmax(1)                      # arg over task groups
        dom_share = readout_share.max(1).values
        task_dom_share = task_share.max(1).values

        probe_ok = torch.tensor(
            [
                (probe_r2[g] >= args.min_probe_r2) and target_meta[g]["valid"]
                for g in READOUT_GROUPS
            ],
            dtype=torch.bool,
        )
        dom_probe_ok = probe_ok[dom_readout]
        dominant_is_task = torch.tensor(
            [READOUT_GROUPS[int(i)] in TASK_CONCEPT_GROUPS for i in dom_readout],
            dtype=torch.bool,
        )
        mono = (alive & (readout_total >= effect_cutoff) & dominant_is_task &
                dom_probe_ok & (task_dom_share >= mono_share) & (breadth == 1))

        feats = []
        for f in range(F_):
            feats.append({
                "feature": f,
                "fire_rate": float(fire_rate[f]),
                "dominant_group": READOUT_GROUPS[int(dom_readout[f])],
                "dominant_concept": (
                    READOUT_GROUPS[int(dom_readout[f])]
                    if READOUT_GROUPS[int(dom_readout[f])] in TASK_CONCEPT_GROUPS
                    else None
                ),
                "concept_breadth": int(breadth[f]),
                "dominant_share": float(dom_share[f]),
                "task_dominant_share": float(task_dom_share[f]),
                "total_readout_effect": float(readout_total[f]),
                "label_confident": bool(dom_probe_ok[f]),
                "monosemantic": bool(mono[f]),
                "group_scores": {g: float(readout_score[f, gi])
                                 for gi, g in enumerate(READOUT_GROUPS)},
                "group_shares": {g: float(readout_share[f, gi])
                                 for gi, g in enumerate(READOUT_GROUPS)},
            })

        n_alive = int(alive.sum())
        print(f"\n{'='*60}\n{layer}  (E={ck['expansion']}, k={ck['k']}, F={F_}, alive={n_alive})\n{'='*60}")
        print("  concept probe val R2: " +
              "  ".join(
                  f"{g}={probe_r2[g]:.2f}{'' if probe_ok[READOUT_GROUPS.index(g)] else '*'}"
                  for g in READOUT_GROUPS))
        print(f"  * = below min_probe_r2={args.min_probe_r2:.2f} or "
              f"target std < {args.min_target_std:.1e}; labels gated")
        print("  max causal effect per group: " +
              "  ".join(f"{g}={readout_score[:, gi].max():.2f}"
                        for gi, g in enumerate(READOUT_GROUPS)))
        print(f"  monosemantic over task concepts {TASK_CONCEPT_GROUPS}: "
              f"{int(mono.sum())}/{n_alive}")
        print(f"  breadth-share cutoff={share_cutoff:.2f}  "
              f"mono-share cutoff={mono_share:.2f}  effect-cutoff={effect_cutoff:.2e} "
              f"(p{args.effect_percentile:.0f})")
        print(f"  {'concept':<10}{'mono#':>7}{'uses#':>7}  "
              f"(mono=breadth-1 here; uses=alive & above on it)")
        for ci, g in enumerate(TASK_CONCEPT_GROUPS):
            ri = READOUT_GROUPS.index(g)
            mono_g = int((mono & (dom_readout == ri)).sum())
            uses_g = int((alive & (readout_score[:, ri] >= effect_cutoff)).sum())
            print(f"  {g:<10}{mono_g:>7}{uses_g:>7}")

        task = "task1" if args.model.startswith("task1") else "task2"
        relevant = {"PT"} if task == "task1" else {"BS", "battery", "gate"}
        keep = [f["feature"] for f in feats
                if f["monosemantic"] and f["dominant_concept"] in relevant]
        print(f"  >> task-relevant monosemantic features ({sorted(relevant)}): "
              f"{len(keep)}  e.g. {keep[:12]}")

        summary[layer] = {
            "expansion": ck["expansion"], "k": ck["k"], "n_features": F_, "n_alive": n_alive,
            "n_monosemantic": int(mono.sum()),
            "probe_r2": probe_r2,
            "target_meta": target_meta,
            "min_probe_r2": args.min_probe_r2,
            "min_target_std": args.min_target_std,
            "share_cutoff": share_cutoff,
            "mono_share": mono_share,
            "effect_cutoff": effect_cutoff,
            "effect_percentile": args.effect_percentile,
            "task_relevant_keep": keep, "features": feats,
        }

    out = os.path.join(SAE_DIR, args.model, "feature_labels_segregate.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
