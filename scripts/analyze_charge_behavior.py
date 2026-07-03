"""
Detect three station-related failure patterns in inference logs.

Reads the last N episodes (default 500, across all sessions in file order) andf
separates:

  A. Boundary chatter  — charge just above threshold, leave, return.
       Signature: charge cycles that exit low (< early_exit) and/or gain little
       (< tiny_gain); consecutive CH with no PT between them in the timeline.

  B. Station camping    — sit at the BS doing nothing useful.
       Signature: high stationIdleSteps fraction; TU (top-up) spam at near-full
       battery (start > camp_bat).

  C. Far-PT -> BS retreat (Task2 reversion) — timeout while NOT stranded.
       Signature: reason=timeout AND batteryAtEnd high (> ok_bat) AND high
       station dwell ((idle+charging)/total) AND long stall after the last PT.
       The agent had charge but loitered near the BS instead of traversing.

Usage:
    python scripts/analyze_charge_behavior.py
    python scripts/analyze_charge_behavior.py --log results/soup_alpha_v1_a0.5/inference_log.txt
    python scripts/analyze_charge_behavior.py --last 500
    python scripts/analyze_charge_behavior.py --worst 15
"""

import argparse
import re
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("--log", default=r"results\soup_05\inference_log.txt")
ap.add_argument("--last", type=int, default=1000, help="analyze the last N episodes")
ap.add_argument("--worst", type=int, default=10, help="list N worst offenders per pattern")
# tunable thresholds
ap.add_argument("--early-exit", type=float, default=0.70, help="exit battery below this = left too early")
ap.add_argument("--tiny-gain",  type=float, default=0.10, help="charge gain below this = barely charged")
ap.add_argument("--camp-bat",   type=float, default=0.90, help="TU starting above this = near-full top-up (camping)")
ap.add_argument("--idle-frac",  type=float, default=0.15, help="stationIdleSteps/total above this = camping")
ap.add_argument("--ok-bat",     type=float, default=0.60, help="batteryAtEnd above this in a timeout = not stranded")
ap.add_argument("--dwell-frac", type=float, default=0.10, help="(idle+charging)/total above this = high station dwell")
args = ap.parse_args()

# ── field helpers ─────────────────────────────────────────────────────────────

def fval(line, key):
    m = re.search(rf'\b{re.escape(key)}=([^\s|]+)', line)
    return m.group(1) if m else None

def fnum(line, key):
    v = fval(line, key)
    try: return float(v)
    except (TypeError, ValueError): return None

def fint(line, key):
    v = fval(line, key)
    try: return int(v)
    except (TypeError, ValueError): return None

def flist(line, key):
    """ptReachSteps=[1,2,3] -> [1.0, 2.0, 3.0]"""
    m = re.search(rf'{re.escape(key)}=\[([^\]]*)\]', line)
    if not m or not m.group(1).strip():
        return []
    out = []
    for tok in m.group(1).split(","):
        tok = tok.strip()
        try: out.append(float(tok))
        except ValueError: pass
    return out

CYCLE_RE = re.compile(r'([\d.]+)->([\d.]+)\(\+([\d.]+)\)')   # start->exit(+gain)
PT_TOK_RE = re.compile(r'PT\d+@(\d+)\(b=([\d.]+)\)')
TU_TOK_RE = re.compile(r'TU@([\d.]+)->([\d.]+)')
CH_TOK_RE = re.compile(r'CH\d+@([\d.]+)->([\d.]+)\(\+([\d.]+)\)')

def parse_cycles(end_line):
    """Charge cycles from the chargeEvents field: list of (start, exit, gain)."""
    m = re.search(r'chargeEvents=(.*)$', end_line)
    if not m:
        return []
    return [(float(a), float(b), float(g)) for a, b, g in CYCLE_RE.findall(m.group(1))]

def parse_timeline(events_line):
    """Ordered token types from [EVENTS]: list of (kind, payload).
    kind in {CH, PT, TU, END}; payload is a dict with parsed numbers."""
    body = events_line.split("[EVENTS]", 1)[-1]
    toks = [t.strip() for t in body.split("|")]
    seq = []
    for t in toks:
        if t.startswith("CH"):
            m = CH_TOK_RE.search(t)
            if m: seq.append(("CH", dict(start=float(m.group(1)), exit=float(m.group(2)), gain=float(m.group(3)))))
        elif t.startswith("PT"):
            m = PT_TOK_RE.search(t)
            if m: seq.append(("PT", dict(step=int(m.group(1)), bat=float(m.group(2)))))
        elif t.startswith("TU"):
            m = TU_TOK_RE.search(t)
            if m: seq.append(("TU", dict(start=float(m.group(1)), exit=float(m.group(2)))))
        elif t.startswith("END"):
            seq.append(("END", {}))
    return seq

# ── load + collect episodes (END paired with EVENTS) ──────────────────────────

with open(args.log, "r", errors="replace") as f:
    lines = f.readlines()

episodes = []
for i, line in enumerate(lines):
    if "[EPISODE END]" not in line:
        continue
    if "mode=Combined" not in line:
        continue
    # find the EVENTS line before the next EPISODE END (usually i+2)
    events_line = ""
    for j in range(i + 1, min(i + 5, len(lines))):
        if "[EPISODE END]" in lines[j]:
            break
        if "[EVENTS]" in lines[j]:
            events_line = lines[j]
            break
    episodes.append((line, events_line))

episodes = episodes[-args.last:]
if not episodes:
    print("No episodes found.")
    raise SystemExit

# ── analyze each episode ──────────────────────────────────────────────────────

records = []
for end_line, events_line in episodes:
    reason   = fval(end_line, "reason") or "unknown"
    total    = fint(end_line, "totalSteps") or 0
    pt       = fint(end_line, "ptReaches") or 0
    bat_end  = fnum(end_line, "batteryAtEnd")
    idle     = fint(end_line, "stationIdleSteps") or 0
    charging = fint(end_line, "chargingSteps") or 0
    top_ups  = fint(end_line, "topUps") or 0
    starts   = fint(end_line, "chargeStarts") or 0
    pt_steps = flist(end_line, "ptReachSteps")
    cycles   = parse_cycles(end_line)
    seq      = parse_timeline(events_line)

    # ---- Pattern A: boundary chatter -----------------------------------------
    early_exits = [c for c in cycles if c[1] < args.early_exit]
    tiny_gains  = [c for c in cycles if c[2] < args.tiny_gain]
    # charges per PT-segment from the timeline: count CH between consecutive PTs
    seg_charges, cur = [], 0
    consec_ch = 0
    prev_kind = None
    for kind, _ in seq:
        if kind == "CH":
            cur += 1
            if prev_kind == "CH":
                consec_ch += 1
        elif kind == "PT":
            seg_charges.append(cur)
            cur = 0
        prev_kind = kind
    seg_charges.append(cur)  # trailing segment after last PT
    chatter_segments = sum(1 for s in seg_charges if s >= 2)
    is_chatter = chatter_segments > 0 or len(early_exits) >= 2

    # ---- Pattern B: station camping ------------------------------------------
    idle_frac = idle / total if total else 0.0
    near_full_tu = sum(1 for k, p in seq if k == "TU" and p["start"] > args.camp_bat)
    is_camping = idle_frac > args.idle_frac or near_full_tu >= 10

    # ---- Pattern C: far-PT -> BS retreat (camping timeout) -------------------
    dwell_frac = (idle + charging) / total if total else 0.0
    last_pt    = pt_steps[-1] if pt_steps else 0
    stall      = total - last_pt
    is_camp_timeout = (reason == "timeout"
                       and bat_end is not None and bat_end > args.ok_bat
                       and dwell_frac > args.dwell_frac)

    records.append(dict(
        reason=reason, total=total, pt=pt, bat_end=bat_end,
        idle=idle, charging=charging, top_ups=top_ups, starts=starts,
        n_cycles=len(cycles), early_exits=len(early_exits), tiny_gains=len(tiny_gains),
        chatter_segments=chatter_segments, consec_ch=consec_ch, is_chatter=is_chatter,
        idle_frac=idle_frac, near_full_tu=near_full_tu, is_camping=is_camping,
        dwell_frac=dwell_frac, stall=stall, is_camp_timeout=is_camp_timeout,
        cycles=cycles,
    ))

N = len(records)
SEP = "=" * 92

def pct(n): return f"{100 * n / N:.1f}%" if N else "—"

# ── report ────────────────────────────────────────────────────────────────────

print()
print(SEP)
print(f"  CHARGE-BEHAVIOR ANALYSIS   ({args.log})")
print(f"  Episodes analyzed: {N} (last {args.last})")
print(SEP)

# outcome breakdown
from collections import Counter
outcomes = Counter(r["reason"] for r in records)
print("\n  Outcomes:")
for reason, c in outcomes.most_common():
    print(f"    {reason:<14} {c:>4}  ({pct(c)})")

# all charge cycles pooled
all_cycles = [c for r in records for c in r["cycles"]]
n_cyc = len(all_cycles)
def cyc_pct(n): return f"{100 * n / n_cyc:.1f}%" if n_cyc else "—"

# ---- Pattern A ----
chatter_eps = [r for r in records if r["is_chatter"]]
early = sum(1 for c in all_cycles if c[1] < args.early_exit)
tiny  = sum(1 for c in all_cycles if c[2] < args.tiny_gain)
exits = [c[1] for c in all_cycles]
print(f"\n{SEP}\n  A. BOUNDARY CHATTER\n{SEP}")
print(f"    Episodes flagged           : {len(chatter_eps):>4}  ({pct(len(chatter_eps))})")
print(f"    Total charge cycles        : {n_cyc:>4}")
print(f"    Early exits (<{args.early_exit:.2f})         : {early:>4}  ({cyc_pct(early)})")
print(f"    Tiny gains (<{args.tiny_gain:.2f})          : {tiny:>4}  ({cyc_pct(tiny)})")
if exits:
    print(f"    Exit battery  median={statistics.median(exits):.2f}  mean={statistics.mean(exits):.2f}  min={min(exits):.2f}")
print(f"    Chatter segments (>=2 CH between PTs), total: {sum(r['chatter_segments'] for r in records)}")
print(f"    Consecutive-CH events (return w/o progress) : {sum(r['consec_ch'] for r in records)}")

# ---- Pattern B ----
camp_eps = [r for r in records if r["is_camping"]]
print(f"\n{SEP}\n  B. STATION CAMPING\n{SEP}")
print(f"    Episodes flagged           : {len(camp_eps):>4}  ({pct(len(camp_eps))})")
print(f"    Idle-frac  mean={statistics.mean(r['idle_frac'] for r in records):.3f}  "
      f"max={max(r['idle_frac'] for r in records):.3f}")
print(f"    Top-ups    mean={statistics.mean(r['top_ups'] for r in records):.1f}  "
      f"max={max(r['top_ups'] for r in records)}")
print(f"    Near-full TU spam total    : {sum(r['near_full_tu'] for r in records)}")

# ---- Pattern C ----
timeouts = [r for r in records if r["reason"] == "timeout"]
camp_to  = [r for r in timeouts if r["is_camp_timeout"]]
print(f"\n{SEP}\n  C. FAR-PT -> BS RETREAT  (camping timeouts)\n{SEP}")
print(f"    Timeout episodes           : {len(timeouts):>4}  ({pct(len(timeouts))})")
if timeouts:
    print(f"    Of those, 'not stranded' (high battery + high dwell): "
          f"{len(camp_to)}  ({100*len(camp_to)/len(timeouts):.1f}% of timeouts)")
    print(f"    Timeout batteryAtEnd  mean={statistics.mean(r['bat_end'] for r in timeouts if r['bat_end'] is not None):.2f}")
    print(f"    Timeout station dwell mean={statistics.mean(r['dwell_frac'] for r in timeouts):.3f}")
    print(f"    Timeout stall steps   mean={statistics.mean(r['stall'] for r in timeouts):.0f}  "
          f"(steps between last PT and timeout)")
    print(f"    --> {len(camp_to)} timeouts ended with battery>{args.ok_bat} while loitering at BS "
          f"(NOT battery-limited)")

# ---- worst offenders ----
def show_worst(title, key, fmt):
    ranked = sorted(records, key=key, reverse=True)[:args.worst]
    print(f"\n  {title}")
    print(f"    {'reason':<10} {'steps':>6} {'pt':>3} {'batEnd':>7} {'starts':>7} {'idle':>6} {'idleF':>6} {'eExit':>6} {'segCh':>6}")
    for r in ranked:
        be = f"{r['bat_end']:.2f}" if r['bat_end'] is not None else "—"
        print(f"    {r['reason']:<10} {r['total']:>6} {r['pt']:>3} {be:>7} {r['starts']:>7} "
              f"{r['idle']:>6} {r['idle_frac']:>6.2f} {r['early_exits']:>6} {r['chatter_segments']:>6}")

print(f"\n{SEP}\n  WORST OFFENDERS\n{SEP}")
show_worst("By charge starts (chatter + camping):", lambda r: r["starts"], None)
show_worst("By station idle steps (camping):",      lambda r: r["idle"], None)

print(SEP)
