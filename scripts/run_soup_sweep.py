"""
Soup alpha-sweep for drone_battery1: task1_v10 x task2_v6 -> Combined.

Waits for the currently running combined_v4 training to finish (polls for an
mlagents process whose command line mentions WAIT_FOR_RUN_ID), then for each
alpha: blends the two specialist checkpoints, stages results/soup_init, and
fine-tunes in Combined mode for SWEEP_STEPS.

Zero-shot performance = window 0 of each run's train_log.txt
(scripts/analyze_combined_training.py <log> --windows N).

Run via run_soup_sweep.bat (activates the conda env), or directly:
  python scripts\run_soup_sweep.py            # wait + run everything
  python scripts\run_soup_sweep.py --no-wait  # skip the wait gate
  python scripts\run_soup_sweep.py --dry-run  # print commands only
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

import torch

# ======== CONFIG ========
PROJECT = r"C:\Users\shahvandit\Downloads\Drone23_curent\drone_battery1"
TASK1_MODEL = os.path.join(PROJECT, r"results\task1_v10\Drone\checkpoint.pt")
TASK2_MODEL = os.path.join(PROJECT, r"results\task2_v6\Drone\checkpoint.pt")
CBM_YAML = os.path.join(PROJECT, r"config\cbm_bigger.yaml")
RUN_CONFIG = os.path.join(PROJECT, r"config\current_run_config.json")
ENV_EXE = r"config\drone_battery1.exe"
RESULTS_DIR = os.path.join(PROJECT, "results")
BEHAVIOR = "Drone"
INIT_RUN_ID = "soup_init"          # staging dir, overwritten per alpha
SOUP_DIR = os.path.join(PROJECT, "combined_models")
RUN_TAG = "soup_alpha_v6"          # prefix for run dirs and soup .pt files (task1_v10 x task2_v6 lineage)

# alpha = task1 weight: alpha=1.0 -> pure task1, alpha=0.0 -> pure task2.
# Endpoints get 70M; interior blends get 30M.
ALPHAS = [0.0, 0.5, 1.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
DEFAULT_STEPS = 30_000_000
ENDPOINT_STEPS = 70_000_000
WAIT_FOR_RUN_ID = "task2_v6"    # poll until no mlagents process mentions this
POLL_SECONDS = 300

# Unity env config written per run — keep in sync with run_sequence.bat
ENV_CONFIG = {
    "max_steps": 10000,
    "training_mode": "Combined",
    "charge_reward_mode": "none",   # V3: dim96=0, proportional charge reward
    "k_charge": 2.5,                # keep <3.0 (= step_penalty/drainRate) to avoid farming
    "idle_penalty": 0.005,          # escalating "leave the BS" penalty
    "battery_drain_rate": "1/3000",
    "recharge_rate": "1/100",
    "initial_battery_min": 0.4,
    "initial_battery_max": 1.0,
    "obstacle_penalty": 50,
    "bs_spawn_range": 50,
    "max_targets": 7,
}
# ========================


def blend_state_dicts(sd1, sd2, alpha):
    """alpha*task1 + (1-alpha)*task2 for all matching tensor keys.

    Returns (blended, blended_keys, copied_keys) where blended_keys were mixed
    and copied_keys were taken from task2 verbatim (missing in task1, non-tensor,
    or shape mismatch).
    """
    blended = dict(sd2)
    blended_keys, copied_keys = [], []
    for k, v2 in sd2.items():
        v1 = sd1.get(k)
        if (v1 is not None and isinstance(v1, torch.Tensor)
                and isinstance(v2, torch.Tensor) and v1.shape == v2.shape):
            blended[k] = alpha * v1 + (1 - alpha) * v2
            blended_keys.append((k, tuple(v2.shape)))
        else:
            if isinstance(v2, torch.Tensor):
                reason = "missing in task1" if v1 is None else (
                    "shape mismatch" if isinstance(v1, torch.Tensor) else "non-tensor in task1")
                copied_keys.append((k, tuple(v2.shape), reason))
    return blended, blended_keys, copied_keys


def make_soup(alpha, save_path):
    d1 = torch.load(TASK1_MODEL, map_location="cpu")
    d2 = torch.load(TASK2_MODEL, map_location="cpu")
    out = {}
    total = 0
    print(f"  [blend] W_merged = {alpha:.2f}*task1 + {1 - alpha:.2f}*task2  (per layer)")
    for section in ("Policy", "Optimizer:critic"):
        sd1, sd2 = d1.get(section, {}), d2.get(section, {})
        if sd1 and sd2:
            out[section], blended_keys, copied_keys = blend_state_dicts(sd1, sd2, alpha)
            total += len(blended_keys)
            print(f"  [blend] section '{section}': "
                  f"{len(blended_keys)} blended, {len(copied_keys)} copied-from-task2")
            for k, shape in blended_keys:
                print(f"           BLEND  {k:<55} {str(shape):<18} "
                      f"{alpha:.2f}*t1 + {1 - alpha:.2f}*t2")
            for k, shape, reason in copied_keys:
                print(f"           COPY   {k:<55} {str(shape):<18} (task2 only: {reason})")
        elif sd2:
            out[section] = sd2
            print(f"  [blend] section '{section}': copied wholesale from task2 "
                  f"(task1 section absent)")
    # Drop Adam state; keep metadata (global_step etc.) from task2
    for k, v in d2.items():
        if k not in ("Policy", "Optimizer:critic", "Optimizer:value_optimizer"):
            out[k] = v
    torch.save(out, save_path)
    return total


def stage_init(soup_pt):
    init_dir = os.path.join(RESULTS_DIR, INIT_RUN_ID, BEHAVIOR)
    os.makedirs(init_dir, exist_ok=True)
    dest = os.path.join(init_dir, "checkpoint.pt")
    shutil.copy2(soup_pt, dest)
    ck = torch.load(dest, map_location="cpu")
    assert "Optimizer:value_optimizer" not in ck, "stale Adam state in soup!"
    print(f"  [init] staged {os.path.basename(soup_pt)} -> {dest}")


def write_env_config(run_id):
    cfg = dict(ENV_CONFIG)
    cfg["log_path"] = os.path.join(RESULTS_DIR, run_id, "train_log.txt").replace("\\", "/")
    os.makedirs(os.path.join(RESULTS_DIR, run_id), exist_ok=True)
    with open(RUN_CONFIG, "w") as f:
        json.dump(cfg, f, indent=4)
    print(f"  [config] log_path -> {cfg['log_path']}")


def patch_trainer_steps(steps):
    with open(CBM_YAML, "r") as f:
        text = f.read()
    text = re.sub(r"max_steps:\s*\d+", f"max_steps: {steps}", text, count=1)
    with open(CBM_YAML, "w") as f:
        f.write(text)
    print(f"  [yaml] trainer max_steps -> {steps}")


def mlagents_running(tag):
    """True if any process command line mentions both mlagents and the tag."""
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*mlagents*' "
          f"-and $_.CommandLine -like '*{tag}*' " + "} | Measure-Object | "
          "Select-Object -ExpandProperty Count")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           capture_output=True, text=True, timeout=60)
        return int(r.stdout.strip() or "0") > 0
    except Exception as e:
        print(f"  [wait] poll error ({e}); assuming still running")
        return True


def wait_for_training():
    print(f"[WAIT] polling every {POLL_SECONDS}s for '{WAIT_FOR_RUN_ID}' to finish...")
    while mlagents_running(WAIT_FOR_RUN_ID):
        print(f"  [{time.strftime('%H:%M:%S')}] {WAIT_FOR_RUN_ID} still training")
        time.sleep(POLL_SECONDS)
    print(f"[WAIT] {WAIT_FOR_RUN_ID} finished. Starting sweep in 60s "
          "(let the env process shut down cleanly).")
    time.sleep(60)


def run_one(run_id, soup_pt, dry):
    cmd = [sys.executable, "-m", "mlagents.trainers.learn", CBM_YAML,
           f"--run-id={run_id}", f"--initialize-from={INIT_RUN_ID}",
           "--force", "--env", ENV_EXE, "--num-envs", "1", "--timeout-wait", "300"]
    print(f"\n{'='*60}\nRUN {run_id}\n  CMD: {' '.join(cmd)}")
    print("  ENV_CONFIG:")
    for k, v in ENV_CONFIG.items():
        print(f"    {k}: {v}")
    print('='*60)
    if dry:
        return True
    write_env_config(run_id)
    stage_init(soup_pt)
    out_log = os.path.join(RESULTS_DIR, run_id, "train_output.log")
    with open(out_log, "w", encoding="utf-8") as f:
        child_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.Popen(cmd, cwd=PROJECT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, encoding="utf-8",
                                errors="replace", env=child_env)
        for line in proc.stdout:
            f.write(line)
            f.flush()
            print(line, end="", flush=True)
        proc.wait()
    ok = proc.returncode == 0
    print(f"  [{'DONE' if ok else 'FAILED rc=' + str(proc.returncode)}] {run_id}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stats-only", action="store_true",
                    help="Print blend stats for each alpha and exit (no training)")
    args = ap.parse_args()

    for p in (TASK1_MODEL, TASK2_MODEL):
        if not os.path.exists(p):
            sys.exit(f"[ABORT] missing checkpoint: {p}")

    if args.stats_only:
        os.makedirs(SOUP_DIR, exist_ok=True)
        for alpha in ALPHAS:
            a = f"{alpha:.1f}"
            soup_pt = os.path.join(SOUP_DIR, f"{RUN_TAG}_a{a}.pt")
            print(f"\n{'='*60}\nalpha={a}\n{'='*60}")
            make_soup(alpha, soup_pt)
        return

    if not args.no_wait and not args.dry_run:
        wait_for_training()

    os.makedirs(SOUP_DIR, exist_ok=True)

    failures = []
    for alpha in ALPHAS:
        a = f"{alpha:.1f}"
        run_id = f"{RUN_TAG}_a{a}"
        soup_pt = os.path.join(SOUP_DIR, f"{RUN_TAG}_a{a}.pt")
        steps = ENDPOINT_STEPS if alpha in (0.0, 1.0) else DEFAULT_STEPS
        if not args.dry_run:
            n = make_soup(alpha, soup_pt)
            print(f"[SOUP] alpha={a}: {n} tensors blended -> {soup_pt}")
            patch_trainer_steps(steps)
        if not run_one(run_id, soup_pt, args.dry_run):
            failures.append(run_id)

    print(f"\n[SWEEP COMPLETE] {len(ALPHAS) - len(failures)}/{len(ALPHAS)} ok"
          + (f"; failed: {failures}" if failures else ""))
    print("Analyze each with: python scripts\\analyze_combined_training.py "
          "results\\soup_aX\\train_log.txt --windows 6")


if __name__ == "__main__":
    main()
