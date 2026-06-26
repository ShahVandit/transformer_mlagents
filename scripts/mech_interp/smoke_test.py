"""
Quick backend sanity check (run in the mlagents_transformers conda env).

  python smoke_test.py --ckpt results/task1_v7/Drone/checkpoint.pt \
                       --capture results/mech_interp/captures/task1_v7/activations.pt

- Loads the checkpoint, prints inferred architecture.
- If --capture is given, validates the standalone forward: input_proj(obs) must
  equal the captured 'obs_out' (input_proj is per-token, so the static-replay
  last-token must match the live last-token exactly). max-abs-err should be ~1e-5.
- Captures residual activations and prints shapes + norms (finite check).
"""
import argparse
import torch
from policy_backend import TransformerPolicyBackend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--capture", default=None,
                    help="optional capture .pt with 'obs'/'obs_out' for validation")
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-obs", type=int, default=2048)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    b = TransformerPolicyBackend(args.ckpt, n_head=args.n_head, device=dev)
    print(b)

    if args.capture:
        d = torch.load(args.capture, map_location="cpu")
        obs = d["obs"].float()[: args.n_obs]
        if "obs_out" in d:
            err = b.validate_input_proj(d["obs"][: len(d["obs_out"])], d["obs_out"])
            print(f"[validate] input_proj max-abs-err = {err:.2e} "
                  f"({'PASS' if err < 1e-3 else 'FAIL — check n_head / key prefix'})")
    else:
        obs = torch.randn(args.n_obs, b.obs_dim)
        print("[warn] no --capture: using RANDOM obs (off-distribution; shapes only)")

    cap = b.capture(obs)
    print("[capture] keys -> shape (mean L2 norm):")
    for k, v in cap.items():
        assert torch.isfinite(v).all(), f"{k} has non-finite values!"
        print(f"  {k:10s} {tuple(v.shape)}  ||.||={v.norm(dim=-1).mean():.3f}")
    print("OK")


if __name__ == "__main__":
    main()
