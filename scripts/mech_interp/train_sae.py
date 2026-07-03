"""
Train one TopK SAE per transformer-layer output, mirroring the eeg repo
(mecheeg) recipe: hook each transformer LAYER's residual-stream output,
z-score, fit a TopK SAE, save with mean/std + diagnostics.

What "layer output" means here
------------------------------
The eeg repo hooks each nn.TransformerEncoderLayer's OUTPUT (the whole-block
residual-stream representation), one SAE per layer. Our capture stores only
sub-components, so we regenerate the exact per-layer residual stream from the
backend over the captured REAL window (obs_window):

    resid.1   = output of transformer layer 0
    resid.2   = output of transformer layer 1
    encoding  = final_norm(resid.2)  (behaviour-readout; optional extra)

Hyperparameters match the original training scripts:
    k = 8, expansion = 8 (sweep 2,4,8,16), epochs = 20, batch = 256,
    resample_every = 2, inputs z-scored.

Usage
-----
  python train_sae.py --model task1_v8
  python train_sae.py --model task2_v3 --layers resid.1,resid.2,encoding \
      --expansions 2,4,8,16 --k 8
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from policy_backend import TransformerPolicyBackend          # noqa: E402
from topk_sae import SAEConfig, train_sae                    # noqa: E402

PROJECT = r"C:\Users\shahvandit\Downloads\Drone23_curent\drone_battery1"
RESULTS = os.path.join(PROJECT, "results")
CAP_DIR = os.path.join(RESULTS, "mech_interp", "captures")
SAE_DIR = os.path.join(RESULTS, "mech_interp", "sae")
CAPTURE_FILES = ("activations_combined.pt", "activations.pt")


def load_window(model: str) -> torch.Tensor:
    """Return the real 8-step window [N, L, obs_dim] from the model's capture."""
    paths = [os.path.join(CAP_DIR, model, name) for name in CAPTURE_FILES]
    path = next((p for p in paths if os.path.exists(p)), None)
    if path is None:
        searched = "\n  ".join(paths)
        sys.exit(
            "[ABORT] no capture found. Expected one of:\n  "
            f"{searched}\n(run scripts\\run_capture.py first)"
        )
    print(f"[capture] {path}")
    d = torch.load(path, map_location="cpu")
    if "obs_window" in d:
        return d["obs_window"].float()
    # fall back: tile static obs into a window (less faithful)
    return d["obs"].float().unsqueeze(1).repeat(1, 8, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="run-id, e.g. task1_v8 / task2_v3")
    ap.add_argument("--layers", default="resid.1,resid.2",
                    help="backend capture keys to train SAEs on "
                         "(resid.1=layer0 out, resid.2=layer1 out, encoding)")
    ap.add_argument("--expansions", default="8",
                    help="comma-separated expansion factors (e.g. 2,4,8,16)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)        # was 20 (still improving)
    ap.add_argument("--batch-size", type=int, default=2048)  # was 256 (steadier grads)
    ap.add_argument("--lr", type=float, default=1e-4)        # was 3e-4 (avoid collapse)
    ap.add_argument("--resample-every", type=int, default=5) # was 2 (gentler)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=300_000)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = os.path.join(RESULTS, args.model, "Drone", "checkpoint.pt")
    if not os.path.exists(ckpt):
        sys.exit(f"[ABORT] missing checkpoint: {ckpt}")

    print(f"[device] {dev}")
    print(f"[model]  {args.model}")
    window = load_window(args.model)
    if len(window) > args.max_tokens:
        idx = torch.randperm(len(window))[:args.max_tokens]
        window = window[idx]
    print(f"[window] {tuple(window.shape)}")

    backend = TransformerPolicyBackend(ckpt, n_head=args.n_head, device=dev)
    print(f"[backend] {backend}")

    # regenerate per-layer residual-stream outputs (the SAE targets)
    print("[capture] regenerating layer activations through backend...")
    caps = backend.capture(window)                            # {key: [N, d_model]}
    layers = [l.strip() for l in args.layers.split(",") if l.strip()]
    Es = [int(e) for e in args.expansions.split(",")]
    for l in layers:
        if l not in caps:
            sys.exit(f"[ABORT] layer key '{l}' not produced by backend "
                     f"(available: {sorted(caps)})")

    out_root = os.path.join(SAE_DIR, args.model)
    os.makedirs(out_root, exist_ok=True)

    for layer in layers:
        acts = caps[layer].float()                            # [N, 128]
        print(f"\n{'='*64}\nLAYER {layer}  acts={tuple(acts.shape)}\n{'='*64}")
        for E in Es:
            print(f"\n--- expansion={E}  (n_features={acts.shape[1]*E})  k={args.k} ---")
            cfg = SAEConfig(d_in=acts.shape[1], expansion=E, k=args.k, lr=args.lr,
                            epochs=args.epochs, batch_size=args.batch_size,
                            resample_every=args.resample_every, device=dev)
            sae, metrics, alive = train_sae(acts, cfg, verbose=True)
            print(f"  [metrics] EV={metrics['r2']:.4f}  L0={metrics['l0']:.1f}  "
                  f"dead={metrics['dead_frac']:.1%}")

            out = os.path.join(out_root,
                               f"sae_{layer.replace('.', '')}_exp{E}_k{args.k}.pt")
            torch.save({
                "sae_state_dict": sae.state_dict(),
                "act_mean": sae.act_mean, "act_std": sae.act_std,
                "d_in": acts.shape[1], "expansion": E, "k": args.k,
                "layer": layer, "model": args.model,
                "metrics": metrics, "alive": alive,
            }, out)
            print(f"  [saved] {out}")

    print(f"\n[DONE] SAEs for {args.model} -> {out_root}")


if __name__ == "__main__":
    main()
