# Mech-interp: task1 vs task2 model diffing

Weight/feature-level analysis of the drone_battery1 specialist policies, built to
explain and inform the soup/Fisher **merge** (not to invent a new merge operator).

## Why this design
- **Standalone backend** (`policy_backend.py`): replays observations through a
  checkpoint with *zero* mlagents dependency, so the SAME obs can go through two
  models offline → matched-input activations (the fair-diff requirement). Forward
  is replicated exactly from `transformer_actor.py` (static `memories=None` path);
  architecture is inferred from the checkpoint.
- **TopK crosscoder** (`topk_crosscoder.py`): the cross-model diff. Uses TopK
  (structural L0), NOT L1 — the model-diffing literature shows naive L1 crosscoders
  manufacture false model-specific features (Complete Shrinkage / Latent
  Decoupling; arXiv:2504.02922, 2603.04426). Shared encoder + per-model decoders;
  the relative decoder norm of each latent classifies it shared / task1 / task2.
- **TopK SAE** (`topk_sae.py`): supporting per-model track for giving features
  human-legible meaning (max-activating obs → probes/TCAV later).

## Prerequisites
A capture `.pt` (from `CaptureAccumulator`, i.e. the capture run) for at least
one model — used both as the shared observation set (its `obs` tensor) and to
validate the backend (its `obs_out`). The obs should cover battery-varying states;
`task2_v5` and `combined` captures are good sources.

## Steps

1. **Smoke-test the backend** (verifies weight loading + forward):
   ```
   python smoke_test.py \
     --ckpt results/task1_v7/Drone/checkpoint.pt \
     --capture results/mech_interp/captures/task1_v7/activations.pt
   ```
   `input_proj max-abs-err` must be ~1e-5 (PASS). If it FAILs, `--n-head` or the
   state_dict key prefix is wrong.

2. **Run the diff** (task1_v8 once it has converged; task2_v5):
   ```
   python run_crosscoder_diff.py \
     --task1 results/task1_v8/Drone/checkpoint.pt \
     --task2 results/task2_v5/Drone/checkpoint.pt \
     --obs   results/mech_interp/captures/task2_v5/activations.pt \
     --layer encoding --dict-size 2048 --k 32 \
     --out results/mech_interp/crosscoder
   ```
   `--layer` can be `encoding` (decision vector, default), `resid.0`, `resid.1`, …
   Output `.pt` has `relative_norms` (histogram → shared ~0.5 / task1 ~0 / task2 ~1)
   and the three latent masks.

3. **Next (payoff):** run the trained crosscoder's dictionary on the MERGED models
   (soup-α, Fisher) on the same obs and measure which task-specific latents survive
   → feature-level account of the task2-leaning bias and the convergence asymmetry.

## Notes
- Static replay (`memories=None`) ignores temporal history by design: identical
  deterministic computation for both models, so activation differences are
  attributable to weights. This is the controlled choice for diffing.
- `n_head` is the only value not inferable from the checkpoint (default 4 =
  cbm.yaml).
