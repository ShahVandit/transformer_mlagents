"""
Analyze average steps for success episodes from train_log.txt.
Usage: python analyze_success_steps.py [log_path] [--under N]
  --under N   print percent of success episodes under N steps
Default log path: results/task1_v5/train_log.txt
"""
import re
import sys
import os
from collections import Counter

DEFAULT_LOG = os.path.join(os.path.dirname(__file__), "..", "results", "task1_v7_inference", "train_log.txt")

args = sys.argv[1:]
under_thresholds = []
top_k = None
filtered_args = []
i = 0
while i < len(args):
    if args[i] == "--under" and i + 1 < len(args):
        under_thresholds.append(int(args[i + 1]))
        i += 2
    elif args[i] == "--top-k" and i + 1 < len(args):
        top_k = int(args[i + 1])
        i += 2
    else:
        filtered_args.append(args[i])
        i += 1

log_path = filtered_args[0] if filtered_args else DEFAULT_LOG

success_steps = []
pt_deltas = []
episode_reasons = Counter()
total_steps_all = 0
total_pt_reaches_all = 0

ep_end_re = re.compile(r'\[EPISODE END\] reason=(\w+).*?totalSteps=(\d+).*?ptReaches=(\d+)')
pt_steps_re = re.compile(r'PT(\d+)@\d+\(\+(\d+)\)')

with open(log_path, "r") as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    m = ep_end_re.search(lines[i])
    if m:
        reason = m.group(1)
        total = int(m.group(2))
        pt_reaches = int(m.group(3))
        episode_reasons[reason] += 1
        total_steps_all += total
        total_pt_reaches_all += pt_reaches
        if reason == "success":
            success_steps.append(total)
            if i + 2 < len(lines) and "[PT STEPS]" in lines[i + 2]:
                deltas = [int(d) for _, d in pt_steps_re.findall(lines[i + 2])]
                if deltas:
                    pt_deltas.append(deltas)
    i += 1

n = len(success_steps)
if n == 0:
    total_episodes = sum(episode_reasons.values())
    if total_episodes:
        print("=" * 105)
        print(f"  EPISODE TERMINATION SUMMARY   ({log_path})")
        print("=" * 105)
        for reason, count in episode_reasons.most_common():
            print(f"  {reason:<14}  n={count:>6}  ({count / total_episodes * 100:>5.1f}%)")
        print("-" * 105)
        print(f"  {'battery_dead':<14}  n={episode_reasons.get('battery_dead', 0):>6}")
        print("=" * 105)
    print("No success episodes found.")
    sys.exit(0)

def percentile(sorted_data, p):
    idx = max(0, min(len(sorted_data) - 1, int(p / 100 * len(sorted_data))))
    return sorted_data[idx]

def stats(vals):
    s = sorted(vals)
    avg = sum(s) / len(s)
    std = (sum((x - avg) ** 2 for x in s) / len(s)) ** 0.5
    return {
        "n"  : len(s),
        "avg": avg,
        "std": std,
        "min": s[0],
        "p10": percentile(s, 10),
        "p25": percentile(s, 25),
        "p50": percentile(s, 50),
        "p75": percentile(s, 75),
        "p90": percentile(s, 90),
        "max": s[-1],
    }

def print_stats(label, vals):
    st = stats(vals)
    print(f"  {label:<14}  n={st['n']:>6}  avg={st['avg']:>7.1f}  std={st['std']:>7.1f}  "
          f"min={st['min']:>5}  p10={st['p10']:>5}  p25={st['p25']:>5}  "
          f"p50={st['p50']:>5}  p75={st['p75']:>5}  p90={st['p90']:>5}  max={st['max']:>5}")

SEP = "=" * 105

print(SEP)
total_episodes = sum(episode_reasons.values())
if total_episodes:
    print("  EPISODE TERMINATION SUMMARY")
    print(SEP)
    for reason, count in episode_reasons.most_common():
        print(f"  {reason:<14}  n={count:>6}  ({count / total_episodes * 100:>5.1f}%)")
    print("-" * 105)
    print(f"  {'battery_dead':<14}  n={episode_reasons.get('battery_dead', 0):>6}")
    print(SEP)
    print()

n_obstacle  = episode_reasons.get("obstacle", 0)
n_success   = episode_reasons.get("success", 0)
n_timeout   = episode_reasons.get("timeout", 0)
n_bat_dead  = episode_reasons.get("battery_dead", 0)

print(SEP)
print("  COLLISION STATS")
print(SEP)
if total_steps_all > 0:
    print(f"  Collisions per 1000 steps:        {n_obstacle / total_steps_all * 1000:.3f}")
if total_pt_reaches_all > 0:
    print(f"  Collisions per target reached:    {n_obstacle / total_pt_reaches_all:.4f}")
denom_excl_timeout = n_success + n_obstacle + n_bat_dead
if denom_excl_timeout > 0:
    print(f"  Collision rate (excl. timeout):   {n_obstacle / denom_excl_timeout * 100:.2f}%  ({n_obstacle}/{denom_excl_timeout})")
denom_before_final = n_success + n_obstacle
if denom_before_final > 0:
    print(f"  Collision rate before final PT:   {n_obstacle / denom_before_final * 100:.2f}%  ({n_obstacle}/{denom_before_final})")
print(SEP)
print()

analysis_steps = sorted(success_steps)
if top_k is not None:
    analysis_steps = analysis_steps[:top_k]
    top_k_label = f"TOP-{top_k} fastest"
else:
    top_k_label = "all"

print(SEP)
print(f"  SUCCESS EPISODE ANALYSIS   ({log_path})   [{top_k_label} episodes]")
print(SEP)
print(f"  {'Metric':<14}  {'n':>6}  {'avg':>7}  {'std':>7}  "
      f"{'min':>5}  {'p10':>5}  {'p25':>5}  {'p50':>5}  {'p75':>5}  {'p90':>5}  {'max':>5}")
print("-" * 105)
print_stats("Total steps", analysis_steps)

if pt_deltas:
    max_pts = max(len(d) for d in pt_deltas)
    print()
    print("  Steps per segment (Start->PT1, PT1->PT2, ...):")
    print("-" * 105)
    for k in range(max_pts):
        vals = [d[k] for d in pt_deltas if len(d) > k]
        label = f"Start->PT{k+1}" if k == 0 else f"PT{k}->PT{k+1}"
        print_stats(label, vals)

p10 = percentile(analysis_steps, 10)
p50 = percentile(analysis_steps, 50)

print()
print(SEP)
print("  BATTERY LIFE RECOMMENDATION")
print(SEP)
print(f"  p10={p10} steps  |  p50={p50} steps")
print()
print(f"  Force >=1 recharge even in fastest episodes  ->  1f / {int(p10 * 0.8)}f  (~80% of p10)")
print(f"  Force >=1 recharge in typical episodes       ->  1f / {int(p50 * 0.8)}f  (~80% of p50)")
print()
print("  Current drain rate: 1f / 4000f  ->  drains in 4000 steps")
print()
for threshold in sorted(under_thresholds):
    under_n = sum(1 for s in success_steps if s < threshold)
    pct_under_n = under_n / n * 100
    print(f"  Episodes under {threshold} steps: {under_n} / {n}  ({pct_under_n:.1f}%)")
print(SEP)
