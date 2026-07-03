"""
Separability / superposition check for the merge operator (parameter-free).

Question it answers: in the 128-dim encoding, does task1's NAVIGATION subspace
sit in (near-)orthogonal directions to task2's BATTERY subspace? If yes ->
the skills are separable -> a clean transplant is possible. If they overlap
(small principal angles) -> superposition -> no clean merge, fine-tuning is
unavoidable. This is the go/no-go before building any operator (and the
empirical justification for a wider model if it fails).

Method (forward-only, no training, no gradients):
  For each known input column j (battery/gate/BS = "battery features";
  PT-dir/PT-dist = "nav features"), PERTURB that column in the real window and
  measure the change in the 128-dim encoding: dE_j = enc(perturbed) - enc(base).
  The directions a feature drives = top singular vectors of dE_j [N,128].

  - NAV subspace (task1)   = SVD of stacked dE over {PTdirX, PTdirY, PTdist}
  - BATT subspace (task2)  = SVD of stacked dE over {battery, gate, BSdirX/Y, BSdist}
  - Separability            = principal angles between those two subspaces.
                              ~90deg => orthogonal/separable; ~0deg => entangled.

The perturbation is SYNTHESISED (e.g. gate 0->1), so it works even if the
borrowed eval-states have the gate column dead -- it measures each model's
learned RESPONSE to that input, which is what matters.

Behavioural relevance: each dE is also projected through the action head
(mu [2,128]) to report how much of the response actually reaches the action.

Run:
  python scripts\mech_interp\separability_check.py
  python scripts\mech_interp\separability_check.py --model1 task1_v7 --model2 task2_v3 \
      --states results\mech_interp\captures\task1_v9\activations.pt --n 8000
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from policy_backend import TransformerPolicyBackend  # noqa: E402

PROJECT = r"C:\Users\shahvandit\Downloads\Drone23_curent\drone_battery1"
RESULTS = os.path.join(PROJECT, "results")

# obs-column groups (v2 layout: 15-dim vector sensor at offset 84)
FEATURES = {
    "PTdirX":   (86, "nav"),
    "PTdirY":   (87, "nav"),
    "PTdist":   (94, "nav"),
    "battery":  (88, "batt"),
    "BSdirX":   (89, "batt"),
    "BSdirY":   (90, "batt"),
    "BSdist":   (95, "batt"),
    "gate":     (96, "batt"),
}
GATE_IDX = 96


def ckpt(run_id):
    return os.path.join(RESULTS, run_id, "Drone", "checkpoint.pt")


def load_states(path, n):
    d = torch.load(path, map_location="cpu")
    w = d.get("obs_window")
    if w is None:                       # fall back: tile static obs into a window
        obs = d["obs"]
        w = obs.unsqueeze(1).repeat(1, 8, 1)
    w = w.float()
    if len(w) > n:
        idx = torch.randperm(len(w))[:n]
        w = w[idx]
    return w                             # [n, 8, 99]


def load_head(run_id, device):
    pol = torch.load(ckpt(run_id), map_location="cpu")["Policy"]
    mw = pol["action_model._continuous_distribution.mu.weight"].to(device).float()
    mb = pol["action_model._continuous_distribution.mu.bias"].to(device).float()
    return mw, mb                        # [2,128], [2]


@torch.no_grad()
def response(backend, windows, j, delta):
    """dE = enc(perturb col j at last token by delta) - enc(base). [N,128]."""
    base = backend.capture(windows)["encoding"]          # [N,128]
    pert = windows.clone()
    if j == GATE_IDX:
        pert[:, -1, j] = 1.0                              # flip gate 0/.. -> 1
    else:
        pert[:, -1, j] = pert[:, -1, j] + delta          # +1 std nudge
    enc = backend.capture(pert)["encoding"]
    return (enc - base)


def orthobasis(dE_list, energy=0.90):
    """Stack response matrices, SVD, keep top dirs capturing `energy` of variance.
    Returns (basis [128,k], singular_values, k)."""
    M = torch.cat(dE_list, dim=0)                         # [sum_N, 128]
    M = M - M.mean(0, keepdim=True)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    var = (S ** 2)
    cum = torch.cumsum(var, 0) / var.sum()
    k = int((cum < energy).sum().item()) + 1
    return Vh[:k].T.contiguous(), S, k                    # cols = directions


def principal_angles(Q1, Q2):
    """Q1 [128,k1], Q2 [128,k2] orthonormal -> principal angles (deg)."""
    Q1, _ = torch.linalg.qr(Q1)
    Q2, _ = torch.linalg.qr(Q2)
    s = torch.linalg.svdvals(Q1.T @ Q2).clamp(-1, 1)
    return torch.rad2deg(torch.arccos(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model1", default="task1_v7", help="navigation specialist")
    ap.add_argument("--model2", default="task2_v3", help="battery specialist")
    ap.add_argument("--states",
                    default=os.path.join(RESULTS, "mech_interp", "captures",
                                         "task1_v9", "activations.pt"),
                    help="capture .pt to borrow obs windows from (eval states only)")
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--energy", type=float, default=0.90,
                    help="variance fraction kept when forming each subspace")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {dev}")
    print(f"[models] nav={args.model1}  batt={args.model2}")
    print(f"[states] {args.states}")

    windows = load_states(args.states, args.n).to(dev)
    print(f"[states] using {tuple(windows.shape)}")

    # per-column 1-std perturbation magnitude (gate handled as 0->1 flip)
    last = windows[:, -1, :]
    std = last.std(0)

    b1 = TransformerPolicyBackend(ckpt(args.model1), n_head=4, device=dev)
    b2 = TransformerPolicyBackend(ckpt(args.model2), n_head=4, device=dev)
    # backend.capture() returns CPU tensors, so keep head + algebra on CPU
    mw1, _ = load_head(args.model1, "cpu")
    mw2, _ = load_head(args.model2, "cpu")

    # ---- responses dE for every feature, in BOTH models -------------------
    print(f"\n{'feature':9s} | {'||dE|| t1':>9s} {'->act t1':>9s} | "
          f"{'||dE|| t2':>9s} {'->act t2':>9s}")
    print("-" * 56)
    dE1, dE2 = {}, {}
    for name, (j, grp) in FEATURES.items():
        delta = float(std[j].item()) if j != GATE_IDX else 0.0
        r1 = response(b1, windows, j, delta)
        r2 = response(b2, windows, j, delta)
        dE1[name], dE2[name] = r1, r2
        # magnitude of encoding response and how much reaches the 2-d action
        m1 = r1.norm(dim=1).mean().item()
        m2 = r2.norm(dim=1).mean().item()
        a1 = (r1 @ mw1.T).norm(dim=1).mean().item()
        a2 = (r2 @ mw2.T).norm(dim=1).mean().item()
        print(f"{name:9s} | {m1:9.4f} {a1:9.4f} | {m2:9.4f} {a2:9.4f}")

    # ---- build subspaces ---------------------------------------------------
    nav_feats  = [n for n, (_, g) in FEATURES.items() if g == "nav"]
    batt_feats = [n for n, (_, g) in FEATURES.items() if g == "batt"]

    Q_nav_t1, S_nav, k_nav = orthobasis([dE1[n] for n in nav_feats], args.energy)
    Q_bat_t2, S_bat, k_bat = orthobasis([dE2[n] for n in batt_feats], args.energy)
    # cross controls: does the OTHER model also use these directions?
    Q_bat_t1, _, _ = orthobasis([dE1[n] for n in batt_feats], args.energy)
    Q_nav_t2, _, _ = orthobasis([dE2[n] for n in nav_feats], args.energy)

    print(f"\n[subspaces] NAV(task1) rank={k_nav}   BATT(task2) rank={k_bat}  "
          f"(energy={args.energy})")

    ang = principal_angles(Q_nav_t1, Q_bat_t2)
    print(f"\n=== RAW PRINCIPAL ANGLES  NAV(task1)  vs  BATT(task2) ===")
    print("  angles(deg):", np.round(ang.cpu().numpy(), 1))
    print(f"  min={ang.min():.1f}  mean={ang.mean():.1f}  "
          f"(includes behaviorally-inert directions)")

    # how much battery energy escapes the nav subspace (separable fraction)
    proj = Q_nav_t1 @ (Q_nav_t1.T @ Q_bat_t2)             # battery dirs projected onto nav
    overlap = proj.norm() / Q_bat_t2.norm()
    print(f"  battery-subspace fraction lying inside nav-subspace: {overlap.item():.3f}")

    # ---- behavior-weighted (action-relevant) separability ------------------
    # weight each encoding dim by how much EITHER model's action reads it, so
    # directions that don't drive any action are suppressed before comparing.
    s = torch.sqrt(mw1.pow(2).sum(0) + mw2.pow(2).sum(0))    # [128] action-sensitivity
    s = s / s.mean()
    def w(dE):                                              # apply behavioral metric
        return dE * s
    Qn = orthobasis([w(dE1[n]) for n in nav_feats],  args.energy)[0]
    Qb = orthobasis([w(dE2[n]) for n in batt_feats], args.energy)[0]
    angw = principal_angles(Qn, Qb)
    projw = Qn @ (Qn.T @ Qb)
    overlapw = projw.norm() / Qb.norm()
    print(f"\n=== BEHAVIOR-WEIGHTED ANGLES  NAV(task1) vs BATT(task2)  (action-relevant) ===")
    print("  angles(deg):", np.round(angw.cpu().numpy(), 1))
    print(f"  min={angw.min():.1f}  mean={angw.mean():.1f}")
    print(f"  battery fraction inside nav (action-relevant): {overlapw.item():.3f} "
          f"(low = separable where it matters)")

    print(f"\n=== VERDICT (behavior-weighted) ===")
    mn = angw.min().item()
    if mn > 60:
        print(f"  min angle {mn:.0f}deg > 60  -> SEPARABLE. Clean transplant is "
              "plausible; build the operator on the current model.")
    elif mn > 35:
        print(f"  min angle {mn:.0f}deg in [35,60] -> PARTIAL overlap. Transplant "
              "needs care; expect some fine-tuning.")
    else:
        print(f"  min angle {mn:.0f}deg < 35 -> ENTANGLED (superposition). No clean "
              "merge at d_model=128; this is the justification to go wider.")


if __name__ == "__main__":
    main()
