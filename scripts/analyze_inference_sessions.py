"""
Per-session inference analysis for soup_alpha inference logs.

Splits by "=== Session started ===" lines, extracts config (drain/recharge),
and reports per-session: episode counts by reason, timeout PT-reach distribution,
steps stats, charging behavior, and reward stats.

Usage:
    python scripts/analyze_inference_sessions.py
    python scripts/analyze_inference_sessions.py --log results/soup_alpha_v1_a0.5/inference_log.txt
    python scripts/analyze_inference_sessions.py --plot
"""

import argparse
import json
import re
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("--log", default=r"results\dummy_runs\inference_log.txt")
ap.add_argument("--plot", action="store_true")
args = ap.parse_args()

# ── helpers ──────────────────────────────────────────────────────────────────

def fval(line, key):
    """Extract scalar field like key=123.4 from EPISODE END line."""
    m = re.search(rf'\b{re.escape(key)}=([^\s|]+)', line)
    return m.group(1) if m else None

def fnum(line, key):
    v = fval(line, key)
    try: return float(v)
    except: return None

def fint(line, key):
    v = fval(line, key)
    try: return int(v)
    except: return None

def parse_fraction(s):
    """'1/3000' → 3000 (returns denominator); plain number → float."""
    if s and '/' in s:
        parts = s.split('/')
        try: return float(parts[1])
        except: pass
    try: return float(s)
    except: return None

# ── load file and split into raw sessions ───────────────────────────────────

with open(args.log, "r", errors="replace") as f:
    lines = f.readlines()

SESSION_RE = re.compile(r'=== Session started (.+?) ===')

# Build list of (start_line_idx, timestamp) for each session header
session_starts = []
for i, line in enumerate(lines):
    m = SESSION_RE.match(line.strip())
    if m:
        session_starts.append((i, m.group(1).strip()))

if not session_starts:
    print("No sessions found.")
    raise SystemExit

# Slice lines belonging to each session
sessions_raw = []
for idx, (start, ts) in enumerate(session_starts):
    end = session_starts[idx + 1][0] if idx + 1 < len(session_starts) else len(lines)
    sessions_raw.append((ts, lines[start:end]))

# ── parse each session ───────────────────────────────────────────────────────

def parse_session(ts, slines):
    # Extract config JSON (first {...} block)
    config = {}
    json_buf = []
    in_json = False
    for line in slines:
        stripped = line.strip()
        if stripped == '{':
            in_json = True
            json_buf = [stripped]
        elif in_json:
            json_buf.append(stripped)
            if stripped == '}':
                try:
                    config = json.loads('\n'.join(json_buf))
                except json.JSONDecodeError:
                    pass
                break

    drain_denom  = parse_fraction(str(config.get("battery_drain_rate", "")))
    recharge_denom = parse_fraction(str(config.get("recharge_rate", "")))
    max_steps    = config.get("max_steps", "?")
    drain_label  = f"1/{int(drain_denom)}" if drain_denom else "?"
    recharge_label = f"1/{int(recharge_denom)}" if recharge_denom else "?"

    episodes = []
    for line in slines:
        if "[EPISODE END]" not in line:
            continue
        reason  = fval(line, "reason") or "unknown"
        steps   = fint(line,  "totalSteps")
        pt      = fint(line,  "ptReaches")
        reward  = fnum(line,  "reward")
        charges = fint(line,  "chargeStarts")
        chsteps = fint(line,  "chargingSteps")
        bonus   = fint(line,  "bonusRecharges")
        pt_rate = fnum(line,  "ptRate")
        first_pt = fint(line, "firstPtStep")
        bat_start = fnum(line,"batteryAtStart")
        bat_end   = fnum(line,"batteryAtEnd")
        episodes.append(dict(
            reason=reason, steps=steps, pt=pt, reward=reward,
            charges=charges, chsteps=chsteps, bonus=bonus,
            pt_rate=pt_rate, first_pt=first_pt,
            bat_start=bat_start, bat_end=bat_end,
        ))

    return dict(
        ts=ts, drain=drain_label, recharge=recharge_label,
        max_steps=max_steps, config=config, episodes=episodes
    )

sessions = [parse_session(ts, sl) for ts, sl in sessions_raw]
# Drop sessions with no episodes
sessions = [s for s in sessions if s["episodes"]]

# ── print report ─────────────────────────────────────────────────────────────

REASONS = ["success", "timeout", "battery_dead", "obstacle"]
PT_BUCKETS = list(range(6))   # 0-5

def pct(n, total):
    return f"{100*n/total:.1f}%" if total else "—"

def stats_str(vals):
    vals = [v for v in vals if v is not None]
    if not vals: return "—"
    vals.sort()
    mean = sum(vals) / len(vals)
    med  = statistics.median(vals)
    p10  = vals[max(0, int(0.10*len(vals)))]
    p90  = vals[min(len(vals)-1, int(0.90*len(vals)))]
    return f"mean={mean:.1f}  med={med:.1f}  p10={p10:.1f}  p90={p90:.1f}"

print()
for sidx, s in enumerate(sessions):
    eps   = s["episodes"]
    N     = len(eps)
    label = f"drain={s['drain']}  recharge={s['recharge']}  max_steps={s['max_steps']}"
    print(f"{'='*68}")
    print(f"  Session {sidx+1}  [{s['ts']}]")
    print(f"  {label}")
    print(f"  Total episodes: {N}")
    print(f"{'='*68}")

    # ── reason breakdown
    by_reason = {r: [e for e in eps if e["reason"] == r] for r in REASONS}
    other = [e for e in eps if e["reason"] not in REASONS]
    # accuracy = success / (total - timeouts): "when the episode terminates, did the agent succeed?"
    n_terminated = N - len(by_reason["timeout"])
    accuracy = 100 * len(by_reason["success"]) / n_terminated if n_terminated else 0
    print(f"\n  Episode outcomes:")
    for r in REASONS:
        n = len(by_reason[r])
        print(f"    {r:<14} {n:>4}  ({pct(n, N)})")
    if other:
        print(f"    other          {len(other):>4}  ({pct(len(other), N)})")
    print(f"    ── accuracy (success / terminated): {accuracy:.1f}%  "
          f"[terminated={n_terminated}, timeouts excluded]")

    # ── timeout breakdown: how many PTs reached?
    timeouts = by_reason["timeout"]
    if timeouts:
        print(f"\n  Timeout episodes ({len(timeouts)}) — ptReaches distribution:")
        bucket = {i: 0 for i in PT_BUCKETS}
        for e in timeouts:
            k = e["pt"] if e["pt"] is not None else -1
            bucket[k if k in bucket else -1] = bucket.get(k if k in bucket else -1, 0) + 1
        for k in sorted(bucket):
            cnt = bucket[k]
            if cnt == 0: continue
            bar = "█" * cnt + " " * max(0, 20 - cnt)
            print(f"    ptReaches={k}: {cnt:>4}  {bar[:20]}  ({pct(cnt, len(timeouts))})")
        # median ptReaches in timeouts
        pt_vals = [e["pt"] for e in timeouts if e["pt"] is not None]
        if pt_vals:
            print(f"    median ptReaches in timeouts: {statistics.median(pt_vals):.1f}")
    else:
        print(f"\n  No timeout episodes.")

    # ── steps stats
    print(f"\n  Steps per episode:")
    succ = by_reason["success"]
    if succ:
        print(f"    success  ({len(succ):>3}): {stats_str([e['steps'] for e in succ])}")
    if timeouts:
        print(f"    timeout  ({len(timeouts):>3}): {stats_str([e['steps'] for e in timeouts])}")
    dead = by_reason["battery_dead"]
    if dead:
        print(f"    batt_dead({len(dead):>3}): {stats_str([e['steps'] for e in dead])}")

    # ── charging behavior (success episodes only)
    if succ:
        print(f"\n  Charging behavior (success episodes):")
        print(f"    avg chargeStarts : {sum(e['charges'] or 0 for e in succ)/len(succ):.2f}")
        print(f"    avg chargingSteps: {sum(e['chsteps'] or 0 for e in succ)/len(succ):.1f}")
        print(f"    avg bonusRecharges:{sum(e['bonus'] or 0 for e in succ)/len(succ):.2f}")
        # fraction of steps spent charging
        frac_charge = [
            e['chsteps'] / e['steps']
            for e in succ if e['chsteps'] is not None and e['steps']
        ]
        if frac_charge:
            print(f"    chargingSteps/totalSteps: {100*sum(frac_charge)/len(frac_charge):.1f}% avg")

    # ── reward & navigation speed (success)
    if succ:
        rewards = [e['reward'] for e in succ if e['reward'] is not None]
        pt_rates = [e['pt_rate'] for e in succ if e['pt_rate'] is not None]
        first_pts = [e['first_pt'] for e in succ if e['first_pt'] is not None and e['first_pt'] >= 0]
        print(f"\n  Reward & nav speed (success episodes):")
        if rewards:
            print(f"    reward       : {stats_str(rewards)}")
        if pt_rates:
            print(f"    ptRate(PT/kst): {stats_str(pt_rates)}")
        if first_pts:
            print(f"    firstPtStep  : {stats_str(first_pts)}")

    print()

# ── cross-session comparison table ───────────────────────────────────────────
print(f"{'='*68}")
print("  Cross-session summary")
print(f"{'='*68}")
hdr = f"  {'Sess':>4}  {'drain':>6}  {'rech':>6}  {'N':>4}  {'succ%':>6}  {'tout%':>6}  {'dead%':>6}  {'accuracy':>9}  {'med_steps_succ':>14}  {'avg_ch_steps':>12}"
print(hdr)
print(f"  {'-'*len(hdr.rstrip())}")
for sidx, s in enumerate(sessions):
    eps = s["episodes"]
    N   = len(eps)
    if not N: continue
    by_reason = {r: [e for e in eps if e["reason"] == r] for r in REASONS}
    succ = by_reason["success"]
    tout = by_reason["timeout"]
    dead = by_reason["battery_dead"]
    n_term   = N - len(tout)
    acc      = 100 * len(succ) / n_term if n_term else float('nan')
    med_succ = statistics.median([e['steps'] for e in succ if e['steps']]) if succ else float('nan')
    avg_ch   = (sum(e['chsteps'] or 0 for e in succ)/len(succ)) if succ else float('nan')
    print(f"  {sidx+1:>4}  {s['drain']:>6}  {s['recharge']:>6}  {N:>4}  "
          f"{pct(len(succ),N):>6}  {pct(len(tout),N):>6}  {pct(len(dead),N):>6}  "
          f"{acc:>8.1f}%  {med_succ:>14.0f}  {avg_ch:>12.1f}")
print()

# ── optional plot ─────────────────────────────────────────────────────────────
if args.plot:
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        n_sess = len(sessions)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        labels = [f"S{i+1}\n{s['drain']}\n{s['recharge']}" for i, s in enumerate(sessions)]

        # 1) success/timeout/dead % stacked bar
        succ_pct = []
        tout_pct = []
        dead_pct = []
        for s in sessions:
            eps = s["episodes"]
            N   = len(eps)
            by_reason = {r: [e for e in eps if e["reason"] == r] for r in REASONS}
            succ_pct.append(100*len(by_reason["success"])/N if N else 0)
            tout_pct.append(100*len(by_reason["timeout"])/N if N else 0)
            dead_pct.append(100*len(by_reason["battery_dead"])/N if N else 0)

        x = np.arange(n_sess)
        axes[0].bar(x, succ_pct, label="success",      color="steelblue")
        axes[0].bar(x, tout_pct, bottom=succ_pct,      label="timeout",   color="orange")
        bot2 = [s+t for s,t in zip(succ_pct, tout_pct)]
        axes[0].bar(x, dead_pct, bottom=bot2,          label="batt_dead", color="salmon")
        axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=8)
        axes[0].set_ylabel("% episodes"); axes[0].set_title("Episode outcomes by session")
        axes[0].legend(fontsize=7)

        # 2) ptReaches distribution for timeout episodes (all sessions combined, color-coded)
        colors_s = plt.cm.tab10(np.linspace(0, 1, n_sess))
        for sidx, s in enumerate(sessions):
            tout = [e for e in s["episodes"] if e["reason"] == "timeout"]
            if not tout: continue
            pt_vals = [e["pt"] if e["pt"] is not None else -1 for e in tout]
            buckets = [pt_vals.count(k) for k in range(6)]
            axes[1].plot(range(6), buckets, marker='o', label=labels[sidx].replace('\n',' '),
                         color=colors_s[sidx])
        axes[1].set_xlabel("ptReaches at timeout")
        axes[1].set_ylabel("Episode count")
        axes[1].set_title("PT reached at timeout (per session)")
        axes[1].legend(fontsize=7)
        axes[1].set_xticks(range(6))

        # 3) median steps for success episodes + avg chargingSteps
        med_steps = []
        avg_ch    = []
        for s in sessions:
            succ = [e for e in s["episodes"] if e["reason"] == "success"]
            med_steps.append(statistics.median([e['steps'] for e in succ if e['steps']]) if succ else 0)
            avg_ch.append(sum(e['chsteps'] or 0 for e in succ)/len(succ) if succ else 0)

        ax2 = axes[2]
        ax3 = ax2.twinx()
        ax2.bar(x - 0.2, med_steps, width=0.4, label="median steps (success)", color="steelblue", alpha=0.8)
        ax3.bar(x + 0.2, avg_ch,    width=0.4, label="avg chargingSteps",      color="orange",    alpha=0.8)
        ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
        ax2.set_ylabel("Steps"); ax3.set_ylabel("Charging steps")
        ax2.set_title("Episode length vs charging overhead")
        ax2.legend(loc="upper left", fontsize=7)
        ax3.legend(loc="upper right", fontsize=7)

        plt.tight_layout()
        out = args.log.replace("inference_log.txt", "inference_sessions.png")
        plt.savefig(out, dpi=150)
        print(f"Plot saved to {out}")
        plt.show()
    except ImportError:
        print("matplotlib not available — skipping plot")
