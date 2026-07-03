"""
Calibrate steps-per-unit-distance ratio k from task1_v7 inference log.

Only completed legs (agent reached the PT) are used — partial legs from
obstacle/timeout episodes are automatically excluded because ptReachSteps
and ptLegDists only contain entries for legs that ended in a PT reach.

Usage:
    python scripts/calibrate_nav_ratio.py
    python scripts/calibrate_nav_ratio.py --log results/task1_v7_inference/inference_log.txt
    python scripts/calibrate_nav_ratio.py --plot
"""

import argparse
import re
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("--log", default=r"results\task1_v7_inference\inference_log.txt")
ap.add_argument("--plot", action="store_true")
args = ap.parse_args()

log_path = args.log

def parse_list(line, field):
    """Extract [v1,v2,...] field, return list of floats or None."""
    m = re.search(rf'{re.escape(field)}=\[([^\]]*)\]', line)
    if not m or m.group(1) == "NA":
        return None
    try:
        return [float(x) for x in m.group(1).split(",")]
    except ValueError:
        return None

UNITS_PER_STEP = 0.4   # maxSpeed(20) * fixedDt(0.02) * 1; physical max travel/step

all_dists  = []   # per-leg center-to-center distance (ptLegDists)
all_steps  = []   # per-leg steps taken
per_leg_k  = []   # per-leg ratio steps/dist

n_episodes = 0
n_with_legs = 0

with open(log_path, "r", errors="replace") as f:
    for line in f:
        if "[EPISODE END]" not in line:
            continue
        n_episodes += 1
        reach_steps = parse_list(line, "ptReachSteps")
        leg_dists   = parse_list(line, "ptLegDists")
        if reach_steps is None or leg_dists is None:
            continue
        if len(reach_steps) != len(leg_dists) or len(reach_steps) == 0:
            continue
        n_with_legs += 1
        prev_step = 0
        for step, dist in zip(reach_steps, leg_dists):
            leg_steps = step - prev_step
            prev_step = step
            if dist > 0 and leg_steps > 0:
                all_dists.append(dist)
                all_steps.append(leg_steps)
                per_leg_k.append(leg_steps / dist)

if not all_dists:
    print("No completed legs found.")
    raise SystemExit

n = len(all_dists)

# --- Model 1: pooled ratio (proportional, steps = k*dist) ---
k_pooled = sum(all_steps) / sum(all_dists)

# --- Model 2: linear fit (steps = slope*dist + intercept), ordinary least squares ---
# A negative intercept is expected: the PT trigger radius means the agent stops
# ~r units short of center, so center-to-center distance overstates travel.
sx  = sum(all_dists)
sy  = sum(all_steps)
sxx = sum(d * d for d in all_dists)
sxy = sum(d * s for d, s in zip(all_dists, all_steps))
denom = n * sxx - sx * sx
slope = (n * sxy - sx * sy) / denom
intercept = (sy - slope * sx) / n
# R^2
ybar = sy / n
ss_tot = sum((s - ybar) ** 2 for s in all_steps)
ss_res = sum((s - (slope * d + intercept)) ** 2 for d, s in zip(all_dists, all_steps))
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
implied_trigger_r = -intercept / slope if slope > 0 else float("nan")  # dist where steps->0

# --- Per-leg ratio spread (for necessity bands) ---
k_mean   = statistics.mean(per_leg_k)
k_median = statistics.median(per_leg_k)
sp = sorted(per_leg_k)
k_p10 = sp[int(0.10 * n)]
k_p90 = sp[int(0.90 * n)]
k_stdev = statistics.stdev(per_leg_k)

print(f"\n{'='*64}")
print(f"  task1_v7 inference calibration  ({log_path})")
print(f"{'='*64}")
print(f"  Episodes parsed : {n_episodes}")
print(f"  With leg data   : {n_with_legs}  ({100*n_with_legs/n_episodes:.1f}%)")
print(f"  Total legs      : {n}")
print(f"  Leg dist range  : {min(all_dists):.1f} .. {max(all_dists):.1f} units")

print(f"\n  Model 1 — proportional  (steps = k * dist)")
print(f"    k_pooled (Σsteps/Σdist) : {k_pooled:.3f}")
print(f"    per-leg k: mean {k_mean:.3f}, median {k_median:.3f}, sd {k_stdev:.3f}")
print(f"    p10 {k_p10:.3f}  ..  p90 {k_p90:.3f}")

print(f"\n  Model 2 — linear  (steps = slope * dist + intercept)   [recommended]")
print(f"    slope     : {slope:.3f}  steps/unit")
print(f"    intercept : {intercept:.2f}  steps   (negative = PT trigger radius)")
print(f"    R^2       : {r2:.4f}")
print(f"    implied trigger radius : {implied_trigger_r:.1f} units")

print(f"\n  Note: max travel is {UNITS_PER_STEP} units/step, so the pure straight-line")
print(f"  rate is {1/UNITS_PER_STEP:.2f} steps/unit. Short legs read BELOW this because")
print(f"  the agent stops ~{implied_trigger_r:.0f} units short (trigger radius), not a bug.")

print(f"\n  Necessity (taskDist = sum of ptLegDists, per episode):")
print(f"    expected  : batteryAtStart < (slope*taskDist + nLegs*intercept) / batteryLife")
print(f"    or simple : batteryAtStart < k_pooled*taskDist / batteryLife")
print(f"    'definitely needed' uses LOW k (k_p10): even efficient nav can't make it")
print(f"    'possibly needed'   uses HIGH k (k_p90): needs charge even on a bad detour")
print(f"{'='*64}\n")

if args.plot:
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Scatter: leg dist vs leg steps
        axes[0].scatter(all_dists, all_steps, alpha=0.3, s=10, color="steelblue")
        x = np.linspace(0, max(all_dists), 100)
        axes[0].plot(x, x / UNITS_PER_STEP, "g--", label=f"max speed ({1/UNITS_PER_STEP:.2f}/u)")
        axes[0].plot(x, k_pooled * x, "r-",  label=f"proportional k={k_pooled:.2f}")
        axes[0].plot(x, slope * x + intercept, "k-", label=f"linear: {slope:.2f}d{intercept:+.0f}")
        axes[0].set_xlabel("Leg distance (units, center-to-center)")
        axes[0].set_ylabel("Leg steps")
        axes[0].set_title("Steps vs Distance per leg (task1_v7 inference)")
        axes[0].legend()

        # Histogram of per-leg k
        axes[1].hist(per_leg_k, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
        axes[1].axvline(1/UNITS_PER_STEP, color="g", linestyle="--", label=f"max speed {1/UNITS_PER_STEP:.2f}")
        axes[1].axvline(k_pooled, color="r", linestyle="-",  label=f"pooled {k_pooled:.2f}")
        axes[1].axvline(k_p10,    color="orange", linestyle=":", label=f"p10 {k_p10:.2f}")
        axes[1].axvline(k_p90,    color="purple", linestyle=":", label=f"p90 {k_p90:.2f}")
        axes[1].set_xlabel("k = steps / dist")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Per-leg k distribution")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig("scripts/calibrate_nav_ratio.png", dpi=150)
        print("Plot saved to scripts/calibrate_nav_ratio.png")
        plt.show()
    except ImportError:
        print("matplotlib not available — skipping plot")
