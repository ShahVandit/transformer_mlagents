"""
stage_fisher_init.py

Stages a Fisher-merged checkpoint (from combine_fisher.py) as an
--initialize-from directory for mlagents-learn, dropping the Adam optimizer
state so the fine-tune starts with a fresh optimizer.

This mirrors the soup pipeline (run_soup_sweep.make_soup), which keeps the
critic weights (Optimizer:critic) and policy/metadata but drops
Optimizer:value_optimizer. Matching that convention keeps the Fisher fine-tune
an apples-to-apples comparison against the soup blends.

Usage:
    python scripts/stage_fisher_init.py <fisher_merged.pt> <init_dir>
    e.g. python scripts/stage_fisher_init.py combined_models/fisher_merged_v2.pt results/fisher_init/Drone
"""

import os
import sys
import torch


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python stage_fisher_init.py <fisher_merged.pt> <init_dir>")
    src, init_dir = sys.argv[1], sys.argv[2]

    ck = torch.load(src, map_location="cpu")
    if not (isinstance(ck, dict) and "Policy" in ck):
        sys.exit(f"[ABORT] {src} is not a nested checkpoint with a 'Policy' key.")

    # Drop fresh-optimizer state (keep Policy, Optimizer:critic, global_step, etc.)
    dropped = [k for k in ("Optimizer:value_optimizer",) if k in ck]
    for k in dropped:
        ck.pop(k, None)

    os.makedirs(init_dir, exist_ok=True)
    dest = os.path.join(init_dir, "checkpoint.pt")
    torch.save(ck, dest)

    reloaded = torch.load(dest, map_location="cpu")
    assert "Optimizer:value_optimizer" not in reloaded, "Adam state not stripped!"
    print(f"[stage] {src} -> {dest}")
    print(f"        dropped: {dropped or 'nothing'}")
    print(f"        kept keys: {sorted(reloaded.keys())}")


if __name__ == "__main__":
    main()
