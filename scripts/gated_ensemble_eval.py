"""
Battery-gated ensemble of the two specialists — zero-training test of:
"can both behaviours be restored by combining the two models per-step?"

At each step, run BOTH specialists on the same observation window and pick:
    action = model1 (PT navigation)  if battery > threshold
             model2 (BS / survival)  otherwise

Run in Combined mode (both pursuit + battery active) and measure cumulative
episode reward. Sweeps several thresholds plus the two pure-model baselines
(threshold 0.0 = model1 only, 1.01 = model2 only). If a gated setting clearly
beats both pure baselines, both behaviours ARE present and combinable per
situation, and the only remaining work is distilling into one net + tuning the
threshold. If even the best gate is poor, arbitration is not minor.

Faithful to training:
  * real 8-step sliding window per agent (episode start = tiled, matches
    transformer_actor._update_observation_buffer)
  * deterministic action = mu(encoding) (the policy mean), clipped to [-1,1]
    (ML-Agents continuous-action clip)

Run via the conda env:
  python scripts\gated_ensemble_eval.py
  python scripts\gated_ensemble_eval.py --episodes 150 --model1 task1_v8 --model2 task2_v3
"""

import argparse
import json
import os
import sys
from collections import defaultdict, deque

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mech_interp"))
from policy_backend import TransformerPolicyBackend  # noqa: E402

from mlagents_envs.environment import UnityEnvironment  # noqa: E402
from mlagents_envs.base_env import ActionTuple  # noqa: E402
from mlagents_envs.side_channel.engine_configuration_channel import (  # noqa: E402
    EngineConfigurationChannel,
)

PROJECT = r"C:\Users\shahvandit\Downloads\Drone23_curent\drone_battery1"
RESULTS_DIR = os.path.join(PROJECT, "results")
RUN_CONFIG = os.path.join(PROJECT, r"config\current_run_config.json")
ENV_EXE = os.path.join(PROJECT, r"config\drone_battery1.exe")
SEQ_LEN = 8
BATTERY_VEC_IDX = 4          # battery is index 4 of the 15-dim vector sensor

# Combined-mode env config — wide battery so the trade-off is exercised.
ENV_CONFIG = {
    "max_steps": 10000,
    "training_mode": "Combined",
    "charge_reward_mode": "urgency",
    "k_charge": 2.5,
    "idle_penalty": 0.005,
    "battery_drain_rate": "1/3000",
    "recharge_rate": "1/100",
    "initial_battery_min": 0.4,
    "initial_battery_max": 1.0,
    "obstacle_penalty": 50,
    "bs_spawn_range": 50,
    "max_targets": 5,
}


class ActorWithHead:
    """policy_backend body + the mu action head from the checkpoint."""

    def __init__(self, run_id, device):
        ck_path = os.path.join(RESULTS_DIR, run_id, "Drone", "checkpoint.pt")
        self.backend = TransformerPolicyBackend(ck_path, n_head=4, device=device)
        pol = torch.load(ck_path, map_location="cpu")["Policy"]
        mw = "action_model._continuous_distribution.mu.weight"
        mb = "action_model._continuous_distribution.mu.bias"
        self.mu_w = pol[mw].to(device).float()       # [2, 128]
        self.mu_b = pol[mb].to(device).float()       # [2]
        self.device = device

    @torch.no_grad()
    def act(self, windows):
        """windows: [n, 8, obs_dim] real history -> [n, 2] deterministic action."""
        w = torch.as_tensor(windows, dtype=torch.float32, device=self.device)
        enc = self.backend.capture(w)["encoding"].to(self.device)   # [n, 128]
        a = enc @ self.mu_w.t() + self.mu_b                          # [n, 2]
        return torch.clamp(a, -1.0, 1.0).cpu().numpy()


def write_combined_config():
    cfg = dict(ENV_CONFIG)
    cfg["log_path"] = os.path.join(RESULTS_DIR, "gated_ensemble",
                                   "train_log.txt").replace("\\", "/")
    os.makedirs(os.path.dirname(cfg["log_path"]), exist_ok=True)
    with open(RUN_CONFIG, "w") as f:
        json.dump(cfg, f, indent=4)


def find_vector_offset(obs_specs_dims):
    """Global column offset of the 15-dim vector sensor within the concat, and
    assert battery lands at global index 88 (= training layout)."""
    offset = 0
    for d in obs_specs_dims:
        if d == 15:
            bat_global = offset + BATTERY_VEC_IDX
            assert bat_global == 88, (
                f"battery at global {bat_global}, expected 88 — obs concat order "
                f"differs from training (dims {obs_specs_dims})")
            return offset
        offset += d
    raise RuntimeError(f"no 15-dim vector sensor found in obs dims {obs_specs_dims}")


def run_threshold(env, behavior, m1, m2, threshold, n_episodes, vec_offset, dev):
    """Roll out the gated ensemble until n_episodes finish; return reward stats."""
    windows = {}                       # agent_id -> deque(maxlen=8) of [obs_dim]
    cum = defaultdict(float)           # agent_id -> running episode reward
    ep_rewards = []

    def ensure_window(aid, obs_row):
        if aid not in windows:
            windows[aid] = deque([obs_row.copy() for _ in range(SEQ_LEN)],
                                 maxlen=SEQ_LEN)
        else:
            windows[aid].append(obs_row.copy())
        return np.stack(windows[aid], axis=0)        # [8, obs_dim]

    while len(ep_rewards) < n_episodes:
        dec, term = env.get_steps(behavior)

        # finished episodes
        for i, aid in enumerate(term.agent_id):
            cum[aid] += float(term.reward[i])
            ep_rewards.append(cum[aid])
            cum.pop(aid, None)
            windows.pop(aid, None)

        if len(dec) == 0:
            env.step()
            continue

        obs_concat = np.concatenate([o.astype(np.float32) for o in dec.obs], axis=1)
        battery = obs_concat[:, vec_offset + BATTERY_VEC_IDX]      # [n]

        wins = np.stack([ensure_window(aid, obs_concat[i])
                         for i, aid in enumerate(dec.agent_id)], axis=0)  # [n,8,D]

        a1 = m1.act(wins)
        a2 = m2.act(wins)
        use_m1 = (battery > threshold)[:, None]
        actions = np.where(use_m1, a1, a2).astype(np.float32)

        for i, aid in enumerate(dec.agent_id):
            cum[aid] += float(dec.reward[i])

        env.set_actions(behavior, ActionTuple(continuous=actions))
        env.step()

    r = np.array(ep_rewards[:n_episodes])
    return dict(mean=float(r.mean()), std=float(r.std()),
                median=float(np.median(r)), n=len(r),
                frac_m1=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model1", default="task1_v8", help="PT navigation specialist")
    ap.add_argument("--model2", default="task2_v3", help="battery specialist")
    ap.add_argument("--episodes", type=int, default=120, help="episodes per threshold")
    ap.add_argument("--thresholds", default="0.0,0.2,0.3,0.4,0.5,1.01",
                    help="battery thresholds; 0.0=model1-only, 1.01=model2-only")
    ap.add_argument("--base-port", type=int, default=5010)
    ap.add_argument("--time-scale", type=float, default=20.0)
    ap.add_argument("--editor", action="store_true",
                    help="connect to the Unity Editor (watch the agent) instead "
                         "of the headless exe; press Play in the Editor after launch")
    args = ap.parse_args()
    if args.editor:
        args.time_scale = min(args.time_scale, 1.0)  # watchable speed in the Editor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {dev}")
    print(f"[models] model1={args.model1} (PT)  model2={args.model2} (battery)")

    m1 = ActorWithHead(args.model1, dev)
    m2 = ActorWithHead(args.model2, dev)
    assert m1.backend.obs_dim == m2.backend.obs_dim
    print(f"[backend] obs_dim={m1.backend.obs_dim} seq_len={m1.backend.seq_len}")

    write_combined_config()
    eng = EngineConfigurationChannel()
    if args.editor:
        # file_name=None -> wait for the Unity Editor to connect on the editor
        # port (5004). Press Play in the Editor; the scene renders so you watch.
        print("[editor] waiting for the Unity Editor — press PLAY now "
              "(editor port 5004)...")
        env = UnityEnvironment(file_name=None, base_port=5004,
                               side_channels=[eng])
    else:
        env = UnityEnvironment(file_name=ENV_EXE, base_port=args.base_port,
                               side_channels=[eng], no_graphics=True)
    eng.set_configuration_parameters(time_scale=args.time_scale)
    env.reset()
    behavior = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior]
    obs_dims = [int(np.prod(o.shape)) for o in spec.observation_specs]
    print(f"[env] behavior={behavior}  obs dims={obs_dims}  "
          f"sum={sum(obs_dims)}")
    vec_offset = find_vector_offset(obs_dims)
    print(f"[env] vector offset={vec_offset}, battery global idx={vec_offset + BATTERY_VEC_IDX}\n")

    thresholds = [float(t) for t in args.thresholds.split(",")]
    rows = []
    try:
        for thr in thresholds:
            tag = ("model1-only" if thr <= 0.0 else
                   "model2-only" if thr >= 1.01 else f"gate@{thr:.2f}")
            stats = run_threshold(env, behavior, m1, m2, thr,
                                  args.episodes, vec_offset, dev)
            rows.append((tag, thr, stats))
            print(f"  {tag:14s} reward mean={stats['mean']:8.2f} "
                  f"std={stats['std']:7.2f} median={stats['median']:8.2f} "
                  f"(n={stats['n']})")
    finally:
        env.close()

    print(f"\n{'='*60}\nSUMMARY (combined-task cumulative episode reward)\n{'='*60}")
    base_m1 = next(s['mean'] for t, thr, s in rows if thr <= 0.0)
    base_m2 = next(s['mean'] for t, thr, s in rows if thr >= 1.01)
    best = max((s['mean'] for t, thr, s in rows
                if 0.0 < thr < 1.01), default=float("-inf"))
    print(f"  model1-only : {base_m1:8.2f}")
    print(f"  model2-only : {base_m2:8.2f}")
    print(f"  best gate   : {best:8.2f}")
    if best > max(base_m1, base_m2):
        print("  => gated ensemble BEATS both pure baselines: both behaviours are "
              "present and combinable. Distill + tune threshold (minor).")
    else:
        print("  => no gate beats the pure baselines: arbitration is NOT minor; "
              "the behaviours conflict beyond a simple battery switch.")


if __name__ == "__main__":
    main()
