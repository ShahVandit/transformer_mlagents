"""
Capture real temporal-window activations for the merge-operator analysis.

Each model is captured in its OWN training mode so episodes run long enough
to expose the full range of states the model actually learned to handle:
  task1_v7 -> Task1 mode (pure navigation, no battery death, full PT coverage)
  task2_v3 -> Task2 mode (pure battery management, full drain/recharge cycles)

The CaptureAccumulator now saves both:
  "obs"        [N, 99]    -- last-token obs (for smoke_test / backward compat)
  "obs_window" [N, 8, 99] -- REAL 8-step sliding history the policy consumed

policy_backend.py feeds obs_window through both checkpoints offline (matched
inputs), so the activation comparison is fair even though the captures came
from different modes.

Run via the conda env:
  python scripts\run_capture.py            # capture both models
  python scripts\run_capture.py --dry-run  # print commands only
"""

import argparse
import json
import os
import re
import subprocess
import sys

# ======== CONFIG ========
PROJECT = r"C:\Users\shahvandit\Downloads\Drone23_curent\drone_battery1"
CBM_YAML = os.path.join(PROJECT, r"config\cbm.yaml")
CAPTURE_YAML = os.path.join(PROJECT, r"config\cbm_capture.yaml")
RUN_CONFIG = os.path.join(PROJECT, r"config\current_run_config.json")
ENV_EXE = r"config\drone_battery1.exe"
RESULTS_DIR = os.path.join(PROJECT, "results")
CAPTURE_DIR = os.path.join(RESULTS_DIR, "mech_interp", "captures")

N_OBS = 300_000
ALL_TOKENS = False           # last-token only; obs_window stores the full history
CAPTURE_MAX_STEPS = 1_000_000
BASE_PORT = 5010

# Per-model env configs — each model captured in its own training mode.
MODEL_CONFIGS = {
    "task1_v7": {
        "training_mode": "task1",         # nav only: battery drains to 0 but does NOT end episode
        "max_steps": 10000,
        "charge_reward_mode": "none",
        "k_charge": 2.5,
        "idle_penalty": 0.005,
        "battery_drain_rate": "1/3000",
        "recharge_rate": "1/100",
        "initial_battery_min": 0.2,       # vary battery obs even if it doesn't drain
        "initial_battery_max": 1.0,
        "obstacle_penalty": 50,
        "bs_spawn_range": 50,
        "max_targets": 5,
    },
    "task2_v3": {
        "training_mode": "task2",         # battery only: full drain/recharge cycles
        "max_steps": 10000,
        "charge_reward_mode": "none",
        "k_charge": 2.5,
        "idle_penalty": 0.005,
        "battery_drain_rate": "1/3000",
        "recharge_rate": "1/100",
        "initial_battery_min": 0.2,       # sample low-battery urgency states
        "initial_battery_max": 1.0,
        "obstacle_penalty": 50,
        "bs_spawn_range": 50,
        "max_targets": 5,
    },
}

MODELS = list(MODEL_CONFIGS.keys())   # ["task1_v7", "task2_v3"]
# ========================


def write_env_config(model, run_id):
    cfg = dict(MODEL_CONFIGS[model])
    cfg["log_path"] = os.path.join(RESULTS_DIR, run_id, "train_log.txt").replace("\\", "/")
    os.makedirs(os.path.join(RESULTS_DIR, run_id), exist_ok=True)
    with open(RUN_CONFIG, "w") as f:
        json.dump(cfg, f, indent=4)
    print(f"  [config] mode={cfg['training_mode']}, "
          f"battery[{cfg['initial_battery_min']}-{cfg['initial_battery_max']}] "
          f"-> {cfg['log_path']}")


def write_capture_yaml(out_path):
    """Generate cbm_capture.yaml from cbm.yaml: lr=0, capture_* on."""
    with open(CBM_YAML, "r") as f:
        text = f.read()
    text = re.sub(r"#\s*capture_all_tokens:.*",
                  f"capture_all_tokens: {'true' if ALL_TOKENS else 'false'}", text)
    text = re.sub(r"#\s*capture_n_obs:.*", f"capture_n_obs: {N_OBS}", text)
    text = re.sub(r'#\s*capture_output_path:.*',
                  f'capture_output_path: "{out_path}"', text)
    text = re.sub(r"learning_rate:\s*[0-9.eE+-]+", "learning_rate: 0.0", text)
    text = re.sub(r"max_steps:\s*\d+", f"max_steps: {CAPTURE_MAX_STEPS}", text, count=1)
    with open(CAPTURE_YAML, "w") as f:
        f.write(text)
    print(f"  [yaml] cbm_capture.yaml: lr=0, capture_n_obs={N_OBS}, "
          f"all_tokens={ALL_TOKENS}, max_steps={CAPTURE_MAX_STEPS}")


def capture_one(model, dry):
    run_id = f"{model}_capture"
    out_path = os.path.join(CAPTURE_DIR, model, "activations.pt").replace("\\", "/")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = [sys.executable, "-m", "mlagents.trainers.learn", CAPTURE_YAML,
           f"--run-id={run_id}", f"--initialize-from={model}", "--force",
           "--env", ENV_EXE, "--num-envs", "1", f"--base-port={BASE_PORT}",
           "--timeout-wait", "300"]
    print(f"\n{'='*64}\nCAPTURE {model}  ->  {out_path}\n  CMD: {' '.join(cmd)}\n{'='*64}")
    if dry:
        return True
    write_env_config(model, run_id)
    write_capture_yaml(out_path)
    out_log = os.path.join(RESULTS_DIR, run_id, "capture_output.log")
    with open(out_log, "w", encoding="utf-8") as f:
        child_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.Popen(cmd, cwd=PROJECT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, encoding="utf-8",
                                errors="replace", env=child_env)
        for line in proc.stdout:
            f.write(line); f.flush()
            print(line, end="", flush=True)
        proc.wait()
    ok = proc.returncode == 0 and os.path.exists(out_path)
    print(f"  [{'DONE' if ok else 'FAILED'}] {model} "
          f"({'saved ' + out_path if os.path.exists(out_path) else 'NO capture file!'})")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--models", default=",".join(MODELS),
                    help=f"comma-separated run-ids to capture (default: {','.join(MODELS)})")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in MODEL_CONFIGS:
            sys.exit(f"[ABORT] no config for model '{m}'. Add it to MODEL_CONFIGS.")
        ck = os.path.join(RESULTS_DIR, m, "Drone", "checkpoint.pt")
        if not os.path.exists(ck):
            sys.exit(f"[ABORT] missing checkpoint: {ck}")

    os.makedirs(CAPTURE_DIR, exist_ok=True)
    failures = [m for m in models if not capture_one(m, args.dry_run)]
    print(f"\n[CAPTURE COMPLETE] {len(models) - len(failures)}/{len(models)} ok"
          + (f"; failed: {failures}" if failures else ""))
    if not failures and not args.dry_run:
        files = ",".join(os.path.join(CAPTURE_DIR, m, "activations.pt") for m in models)
        print("\nNext — smoke test then diff:")
        print(f"  cd scripts\\mech_interp")
        print(f"  python smoke_test.py "
              f"--ckpt ..\\..\\results\\{models[0]}\\Drone\\checkpoint.pt "
              f"--capture {os.path.join(CAPTURE_DIR, models[0], 'activations.pt')}")
        print(f"  python run_crosscoder_diff.py "
              f"--task1 ..\\..\\results\\task1_v7\\Drone\\checkpoint.pt "
              f"--task2 ..\\..\\results\\task2_v3\\Drone\\checkpoint.pt "
              f"--obs {files} "
              f"--layer encoding --dict-size 2048 --k 32 "
              f"--out ..\\..\\results\\mech_interp\\crosscoder")


if __name__ == "__main__":
    main()
