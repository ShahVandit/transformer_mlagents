"""
task1 vs task2 model-diffing crosscoder — end to end.

Pipeline:
  1. Load a SHARED observation set (from one or more capture .pt files; the
     CaptureAccumulator stores an "obs" tensor). Using the same obs for both
     models is the fair-diff requirement.
  2. Replay those obs through BOTH checkpoints (static memories=None path) and
     read the residual stream at the chosen layer  ->  matched activations.
  3. Train a TopK crosscoder; report explained variance, L0, dead%.
  4. Classify latents shared / task1-specific / task2-specific via relative
     decoder norm; save everything (+ histogram data) for plotting.

Example:
  python run_crosscoder_diff.py \
    --task1 results/task1_v8/Drone/checkpoint.pt \
    --task2 results/task2_v5/Drone/checkpoint.pt \
    --obs   results/mech_interp/captures/task1_v8/activations.pt,results/mech_interp/captures/task2_v5/activations.pt \
    --layer encoding --dict-size 2048 --k 32 --out results/mech_interp/crosscoder
"""

from __future__ import annotations

import argparse
import os

import torch

from policy_backend import TransformerPolicyBackend
from topk_crosscoder import CrossCoderConfig, train_crosscoder


def load_obs(paths: str, max_obs: int | None) -> torch.Tensor:
    """Concatenate observations from one or more capture .pt files.

    Prefers 'obs_window' [N, L, obs_dim] (the REAL temporal history the policy
    consumed) so the replayed activation matches the behavior-generating
    computation. Falls back to 'obs' [N, obs_dim] (static-replay) for older
    captures, or a raw tensor saved directly."""
    obs_list = []
    for p in paths.split(","):
        p = p.strip()
        if not p:
            continue
        d = torch.load(p, map_location="cpu")
        if isinstance(d, dict):
            if "obs_window" in d:
                obs_list.append(d["obs_window"].float())
            elif "obs" in d:
                obs_list.append(d["obs"].float())
            else:
                raise KeyError(f"{p}: no 'obs_window'/'obs' key (keys: {list(d)[:8]})")
        else:
            obs_list.append(d.float())
        print(f"  [obs] {p}: {tuple(obs_list[-1].shape)}")
    obs = torch.cat(obs_list, dim=0)
    if max_obs and len(obs) > max_obs:
        idx = torch.randperm(len(obs))[:max_obs]
        obs = obs[idx]
    print(f"  [obs] shared set: {tuple(obs.shape)}")
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task1", required=True, help="task1 (model0) checkpoint.pt")
    ap.add_argument("--task2", required=True, help="task2 (model1) checkpoint.pt")
    ap.add_argument("--obs", required=True,
                    help="comma-separated capture .pt files (use their 'obs')")
    ap.add_argument("--layer", default="encoding",
                    help="capture key to diff: encoding | resid.0 | resid.1 | ...")
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--max-obs", type=int, default=100_000)
    ap.add_argument("--dict-size", type=int, default=2048)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--out", default="results/mech_interp/crosscoder")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {dev}")

    print("\n=== Load shared observations ===")
    obs = load_obs(args.obs, args.max_obs)

    print("\n=== Backends ===")
    b1 = TransformerPolicyBackend(args.task1, n_head=args.n_head, device=dev)
    b2 = TransformerPolicyBackend(args.task2, n_head=args.n_head, device=dev)
    print(f"  task1: {b1}")
    print(f"  task2: {b2}")
    assert b1.obs_dim == b2.obs_dim == obs.shape[-1], (
        f"obs_dim mismatch: task1={b1.obs_dim} task2={b2.obs_dim} "
        f"obs={obs.shape[-1]} (obs shape {tuple(obs.shape)})")

    # Optional backend sanity check against a capture's obs_out (task1's own file).
    first = torch.load(args.obs.split(",")[0].strip(), map_location="cpu")
    if isinstance(first, dict) and "obs_out" in first and "obs" in first:
        err = b1.validate_input_proj(first["obs"][: len(first["obs_out"])],
                                     first["obs_out"])
        print(f"  [sanity] input_proj replay max-abs-err vs capture: {err:.2e} "
              f"({'OK' if err < 1e-3 else 'CHECK n_head/keys!'})")

    print("\n=== Capture matched activations ===")
    cap1 = b1.capture(obs)[args.layer]
    cap2 = b2.capture(obs)[args.layer]
    print(f"  task1[{args.layer}]={tuple(cap1.shape)}  "
          f"task2[{args.layer}]={tuple(cap2.shape)}")

    print("\n=== Train TopK crosscoder ===")
    cfg = CrossCoderConfig(
        d_in=cap1.shape[1], dict_size=args.dict_size, k=args.k,
        epochs=args.epochs, batch_size=args.batch_size, device=dev)
    cc, metrics, alive = train_crosscoder(cap1, cap2, cfg)
    print(f"\n[metrics] {metrics}")

    split = cc.shared_specific_split(alive=alive)
    n_t1 = int(split["task1_specific"].sum())
    n_sh = int(split["shared"].sum())
    n_t2 = int(split["task2_specific"].sum())
    n_alive = int(alive.sum())
    print(f"\n=== Latent decomposition (alive={n_alive}/{cfg.dict_size}) ===")
    print(f"  task1-specific : {n_t1}")
    print(f"  shared         : {n_sh}")
    print(f"  task2-specific : {n_t2}")

    out_pt = os.path.join(args.out, f"crosscoder_{args.layer.replace('.', '_')}.pt")
    torch.save({
        "state_dict": cc.state_dict(),
        "cfg": vars(cfg),
        "metrics": metrics,
        "relative_norms": split["relative_norms"],
        "alive": alive,
        "masks": {k: split[k] for k in
                  ("task1_specific", "shared", "task2_specific")},
        "layer": args.layer,
        "task1": args.task1, "task2": args.task2,
    }, out_pt)
    print(f"\n[saved] {out_pt}")
    print("  plot: relative_norms histogram -> shared(~0.5)/task1(~0)/task2(~1)")


if __name__ == "__main__":
    main()
