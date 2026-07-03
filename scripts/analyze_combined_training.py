"""
Analyze combined-model training progression from train_log.txt.

Tracks the co-emergence of navigation (PT reaches) and battery management
(recharge adoption) across training, binned into windows.

Usage:
    python analyze_combined_training.py [log_path] [--windows N] [--last N]
                                        [--per-session] [--plot]
  --windows N     number of bins across training (default 40)
  --last N        only analyze the last N episodes
  --per-session   bin by '=== Session started ===' boundaries instead of fixed windows
  --plot          render recharge-adoption + termination-mix curves (needs matplotlib)

Default log: results/combined_v3/train_log.txt
"""
import re
import sys
import os

DEFAULT_LOG = os.path.join(os.path.dirname(__file__), "..", "results", "fisher_merge_v2", "train_log.txt")

# ---- arg parsing -----------------------------------------------------------
args = sys.argv[1:]
log_path = None
n_windows = 40
last_n = None
per_session = False
do_plot = False
i = 0
while i < len(args):
    a = args[i]
    if a == "--windows" and i + 1 < len(args):
        n_windows = int(args[i + 1]); i += 2
    elif a == "--last" and i + 1 < len(args):
        last_n = int(args[i + 1]); i += 2
    elif a == "--per-session":
        per_session = True; i += 1
    elif a == "--plot":
        do_plot = True; i += 1
    else:
        log_path = a; i += 1
if log_path is None:
    log_path = DEFAULT_LOG

# ---- parsing ---------------------------------------------------------------
# One regex per field; tolerant of the early ("recharges") vs late
# ("bonusRecharges") schema and NA-valued charge fields.
session_re = re.compile(r'=== Session started')
end_re = re.compile(r'\[EPISODE END\]')

def fnum(line, key, default=None):
    """Pull a numeric field; returns float or default (for NA / missing)."""
    m = re.search(rf'{key}=(-?[\d.]+)', line)
    return float(m.group(1)) if m else default

def sstr(line, key):
    m = re.search(rf'{key}=([^\s|]+)', line)
    return m.group(1) if m else None

episodes = []   # list of dicts in file order
session_idx = -1

with open(log_path, "r", errors="replace") as f:
    for line in f:
        if session_re.search(line):
            session_idx += 1
            continue
        if not end_re.search(line):
            continue
        reason = sstr(line, "reason")
        ep = {
            "session": session_idx,
            "reason": reason,
            "steps": fnum(line, "totalSteps", 0),
            "reward": fnum(line, "reward", 0.0),
            "pt": fnum(line, "ptReaches", 0),
            # accept both schema names for the recharge-count field
            "bonus": fnum(line, "bonusRecharges", fnum(line, "recharges", 0)),
            "chargeStarts": fnum(line, "chargeStarts", 0),
            "chargingSteps": fnum(line, "chargingSteps", 0),
            "idleSteps": fnum(line, "stationIdleSteps", 0),
            "minChargeStart": fnum(line, "minChargeStart"),   # None if NA
            "maxChargeExit": fnum(line, "maxChargeExit"),
            "chargeGained": fnum(line, "chargeGained"),
            "firstPtStep": fnum(line, "firstPtStep", -1),
            "batteryAtStart": fnum(line, "batteryAtStart"),   # None if not logged yet
        }
        # charging was mandatory if the starting battery couldn't cover the episode
        # drain = 1/3000 per step; batteryLife field captures the actual drain rate
        battery_life = fnum(line, "batteryLife", 3000)  # steps at full battery
        bat_start = ep["batteryAtStart"]
        if bat_start is not None:
            ep["chargingRequired"] = (bat_start * battery_life) < ep["steps"]
        else:
            ep["chargingRequired"] = None   # old log without batteryAtStart
        episodes.append(ep)

if not episodes:
    print(f"No [EPISODE END] lines found in {log_path}")
    sys.exit(0)

if last_n is not None:
    episodes = episodes[-last_n:]

N = len(episodes)

# ---- windowing -------------------------------------------------------------
def make_windows():
    if per_session:
        groups = {}
        for idx, ep in enumerate(episodes):
            groups.setdefault(ep["session"], []).append(idx)
        return [("S%d" % s, idxs) for s, idxs in sorted(groups.items())]
    # fixed equal-count windows
    w = max(1, N // n_windows)
    out = []
    start = 0
    bin_id = 0
    while start < N:
        end = min(N, start + w)
        out.append((str(bin_id), list(range(start, end))))
        start = end
        bin_id += 1
    return out

windows = make_windows()

# ---- aggregation -----------------------------------------------------------
def agg(idxs):
    eps = [episodes[k] for k in idxs]
    n = len(eps)
    cnt = {"success": 0, "battery_dead": 0, "obstacle": 0, "timeout": 0}
    for e in eps:
        if e["reason"] in cnt:
            cnt[e["reason"]] += 1
    succ = [e for e in eps if e["reason"] == "success"]
    dead = [e for e in eps if e["reason"] == "battery_dead"]
    used_charge = [e for e in eps if e["chargeStarts"] > 0]
    mcs = [e["minChargeStart"] for e in eps if e["minChargeStart"] is not None]
    mce = [e["maxChargeExit"] for e in eps if e["maxChargeExit"] is not None]
    cg = [e["chargeGained"] for e in eps if e["chargeGained"] is not None]
    succ_recharged = [e for e in succ if e["chargeStarts"] > 0]

    # Behavioral taxonomy: outcome x battery engagement. The learning
    # trajectory we want is succ+chg up, dead-raw down; succ-raw up alone
    # means it is only getting faster at racing the battery, not managing it.
    n_succ_chg = len(succ_recharged)
    n_succ_raw = len(succ) - n_succ_chg
    n_dead_chg = sum(1 for e in dead if e["chargeStarts"] > 0)
    n_dead_raw = len(dead) - n_dead_chg

    # charging-required breakdown (only episodes where batteryAtStart was logged)
    req_eps = [e for e in eps if e["chargingRequired"] is not None]
    n_req = sum(1 for e in req_eps if e["chargingRequired"])
    n_req_succ_chg  = sum(1 for e in req_eps if e["chargingRequired"] and e["reason"] == "success" and e["chargeStarts"] > 0)
    n_req_succ_raw  = sum(1 for e in req_eps if e["chargingRequired"] and e["reason"] == "success" and e["chargeStarts"] == 0)
    n_req_dead      = sum(1 for e in req_eps if e["chargingRequired"] and e["reason"] == "battery_dead")
    n_free_succ     = sum(1 for e in req_eps if not e["chargingRequired"] and e["reason"] == "success")
    n_free_dead     = sum(1 for e in req_eps if not e["chargingRequired"] and e["reason"] == "battery_dead")
    total_starts = sum(e["chargeStarts"] for e in eps)
    total_bonus = sum(e["bonus"] for e in eps)

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "n": n,
        "n_succ": len(succ),
        "n_succ_nochg": len(succ) - len(succ_recharged),   # success without any charge
        "p_succ": cnt["success"] / n * 100,
        "p_dead": cnt["battery_dead"] / n * 100,
        "p_obs": cnt["obstacle"] / n * 100,
        "p_tout": cnt["timeout"] / n * 100,
        "reward": mean([e["reward"] for e in eps]),
        "pt": mean([e["pt"] for e in eps]),
        "adopt": len(used_charge) / n * 100,             # % episodes that charged
        "bonus": mean([e["bonus"] for e in eps]),
        "chgSteps": mean([e["chargingSteps"] for e in eps]),
        "idleSteps": mean([e["idleSteps"] for e in eps]),
        "minStart": mean(mcs) if mcs else float("nan"),  # how low before charging
        "maxExit": mean(mce) if mce else float("nan"),   # how full it tops up
        "gained": mean(cg) if cg else float("nan"),
        "deadPt": mean([e["pt"] for e in dead]),         # PTs before death
        "deadStep": mean([e["steps"] for e in dead]),    # steps before death
        "succRechgPct": (len(succ_recharged) / len(succ) * 100) if succ else float("nan"),
        # taxonomy (% of window)
        "t_succ_chg": n_succ_chg / n * 100,
        "t_succ_raw": n_succ_raw / n * 100,
        "t_dead_chg": n_dead_chg / n * 100,
        "t_dead_raw": n_dead_raw / n * 100,
        # fraction of started charge cycles that completed to the bonus level
        "cycleDone": (total_bonus / total_starts * 100) if total_starts > 0 else float("nan"),
        # charging-required breakdown (None when batteryAtStart not in log)
        "has_req_data": len(req_eps) > 0,
        "pct_req": n_req / len(req_eps) * 100 if req_eps else float("nan"),
        "req_succ_chg": n_req_succ_chg / n * 100,   # mandatory + succeeded + charged
        "req_succ_raw": n_req_succ_raw / n * 100,   # mandatory + succeeded WITHOUT charging (shouldn't happen)
        "req_dead": n_req_dead / n * 100,            # mandatory + died (genuine failure)
        "free_succ": n_free_succ / n * 100,          # optional + succeeded
        "free_dead": n_free_dead / n * 100,          # optional + died (suspicious)
    }

rows = [(label, agg(idxs)) for label, idxs in windows]

# ---- output ----------------------------------------------------------------
SEP = "=" * 122

print(SEP)
print(f"  COMBINED TRAINING PROGRESSION   ({log_path})")
print(f"  episodes={N}   windows={len(rows)}   mode={'per-session' if per_session else 'fixed-bin'}"
      + (f"   (last {last_n})" if last_n else ""))
print(SEP)

# Block 1: outcomes + reward + navigation
print("  OUTCOMES & NAVIGATION")
print(f"  {'win':>4} {'n':>6} {'%succ':>6} {'%dead':>6} {'%obs':>5} {'%tout':>6} "
      f"{'reward':>8} {'avgPT':>6}")
print("-" * 122)
for label, r in rows:
    print(f"  {label:>4} {r['n']:>6} {r['p_succ']:>6.1f} {r['p_dead']:>6.1f} "
          f"{r['p_obs']:>5.1f} {r['p_tout']:>6.1f} {r['reward']:>8.1f} {r['pt']:>6.2f}")

# Block 2: battery management emergence
print()
print("  BATTERY MANAGEMENT EMERGENCE")
print(f"  {'win':>4} {'%charged':>9} {'bonus':>6} {'chgStp':>7} {'idleStp':>8} "
      f"{'minStart':>9} {'maxExit':>8} {'gained':>7} {'%succ+chg':>10} {'succ_nochg':>11}")
print("-" * 122)
for label, r in rows:
    def f(x, w, p):
        return (f"{x:>{w}.{p}f}" if x == x else f"{'NA':>{w}}")  # NaN check
    print(f"  {label:>4} {r['adopt']:>9.1f} {r['bonus']:>6.2f} {r['chgSteps']:>7.0f} "
          f"{r['idleSteps']:>8.0f} {f(r['minStart'],9,3)} {f(r['maxExit'],8,3)} "
          f"{f(r['gained'],7,3)} {f(r['succRechgPct'],10,1)} {r['n_succ_nochg']:>11}")

# Block 3: behavioral taxonomy — outcome x battery engagement, % of window
print()
print("  EPISODE TAXONOMY  (% of window: how each outcome was achieved)")
print(f"  {'win':>4} {'succ+chg':>9} {'succ-raw':>9} {'dead+chg':>9} {'dead-raw':>9} "
      f"{'%obs':>5} {'%tout':>6} {'cycleDone%':>11}")
print("-" * 122)
for label, r in rows:
    cd = f"{r['cycleDone']:>11.1f}" if r['cycleDone'] == r['cycleDone'] else f"{'NA':>11}"
    print(f"  {label:>4} {r['t_succ_chg']:>9.1f} {r['t_succ_raw']:>9.1f} "
          f"{r['t_dead_chg']:>9.1f} {r['t_dead_raw']:>9.1f} "
          f"{r['p_obs']:>5.1f} {r['p_tout']:>6.1f} {cd}")

# Block 4: charging-required breakdown (only shown when batteryAtStart is logged)
overall_check = agg(list(range(N)))
if overall_check["has_req_data"]:
    print()
    print("  CHARGING-REQUIRED BREAKDOWN  (episodes where battery couldn't last without recharge)")
    print(f"  {'win':>4} {'%req':>6} {'req+succ+chg':>13} {'req+succ-raw':>13} {'req+dead':>9} {'free+succ':>10} {'free+dead':>10}")
    print("-" * 122)
    def fn(x, w):
        return f"{x:>{w}.1f}" if x == x else f"{'NA':>{w}}"
    for label, r in rows:
        print(f"  {label:>4} {fn(r['pct_req'],6)} {fn(r['req_succ_chg'],13)} "
              f"{fn(r['req_succ_raw'],13)} {fn(r['req_dead'],9)} "
              f"{fn(r['free_succ'],10)} {fn(r['free_dead'],10)}")

# Block 5: battery-death diagnostics
print()
print("  BATTERY-DEATH DIAGNOSTICS  (how far it got before dying)")
print(f"  {'win':>4} {'%dead':>6} {'avgPT@death':>12} {'avgStep@death':>14}")
print("-" * 122)
for label, r in rows:
    dp = f"{r['deadPt']:>12.2f}" if r['p_dead'] > 0 else f"{'-':>12}"
    ds = f"{r['deadStep']:>14.0f}" if r['p_dead'] > 0 else f"{'-':>14}"
    print(f"  {label:>4} {r['p_dead']:>6.1f} {dp} {ds}")

# Overall summary
overall = agg(list(range(N)))
print()
print(SEP)
print("  OVERALL")
print(SEP)
print(f"  episodes            {N}")
print(f"  success             {overall['p_succ']:.1f}%")
print(f"  battery_dead        {overall['p_dead']:.1f}%")
print(f"  obstacle            {overall['p_obs']:.1f}%")
print(f"  timeout             {overall['p_tout']:.1f}%")
print(f"  avg reward          {overall['reward']:.1f}")
print(f"  avg PT reaches      {overall['pt']:.2f}")
print(f"  recharge adoption   {overall['adopt']:.1f}%  (episodes with >=1 charge start)")
print(f"  success w/ recharge {overall['succRechgPct']:.1f}%  (of success episodes)")
print(f"  success no charge   {overall['n_succ_nochg']}  (of {overall['n_succ']} success episodes — never started a charge)")
print(f"  taxonomy            succ+chg {overall['t_succ_chg']:.1f}% | succ-raw {overall['t_succ_raw']:.1f}% | "
      f"dead+chg {overall['t_dead_chg']:.1f}% | dead-raw {overall['t_dead_raw']:.1f}%")
print(f"  cycle completion    {overall['cycleDone']:.1f}%  (charge starts that reached the bonus level)")
if overall["has_req_data"]:
    print(f"  charging required   {overall['pct_req']:.1f}%  (episodes where battery couldn't last without recharge)")
    print(f"  req+succ+chg        {overall['req_succ_chg']:.1f}%  (mandatory charging, succeeded — correct behavior)")
    print(f"  req+succ-raw        {overall['req_succ_raw']:.1f}%  (mandatory charging, succeeded without charging — impossible/fast ep)")
    print(f"  req+dead            {overall['req_dead']:.1f}%  (mandatory charging, died — genuine failure)")
    print(f"  free+succ           {overall['free_succ']:.1f}%  (charging optional, succeeded)")
    print(f"  free+dead           {overall['free_dead']:.1f}%  (charging optional, died — suspicious)")
print(SEP)

# ---- optional plot ---------------------------------------------------------
if do_plot:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[plot skipped] matplotlib not available")
        sys.exit(0)

    x = list(range(len(rows)))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(x, [r["p_succ"] for _, r in rows], label="success %", color="green")
    ax1.plot(x, [r["p_dead"] for _, r in rows], label="battery_dead %", color="red")
    ax1.plot(x, [r["p_obs"] for _, r in rows], label="obstacle %", color="orange")
    ax1.plot(x, [r["p_tout"] for _, r in rows], label="timeout %", color="gray")
    ax1.set_ylabel("termination mix (%)")
    ax1.legend(loc="center right"); ax1.grid(alpha=0.3)
    ax1.set_title(f"Combined training progression  ({os.path.basename(log_path)})")

    ax2.plot(x, [r["adopt"] for _, r in rows], label="recharge adoption %", color="blue")
    ax2.plot(x, [r["succRechgPct"] if r["succRechgPct"] == r["succRechgPct"] else 0
                 for _, r in rows], label="success-with-recharge %", color="purple")
    ax2b = ax2.twinx()
    ax2b.plot(x, [r["pt"] for _, r in rows], label="avg PT reaches", color="black", ls="--")
    ax2b.set_ylabel("avg PT reaches")
    ax2.set_ylabel("recharge (%)"); ax2.set_xlabel("training window")
    ax2.legend(loc="upper left"); ax2b.legend(loc="upper right"); ax2.grid(alpha=0.3)

    out_png = os.path.join(os.path.dirname(os.path.abspath(log_path)),
                           "combined_training_progression.png")
    plt.tight_layout(); plt.savefig(out_png, dpi=120)
    print(f"\n[plot saved] {out_png}")
