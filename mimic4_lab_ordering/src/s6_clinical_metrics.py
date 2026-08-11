"""
Stage 6: the paper's non-OPE evaluation of the learned policy (Sec. 3.1).

Three metrics, all on the held-out test split:

  1. ORDER-COUNT REDUCTION versus the clinician. The paper filters "a sequence of
     recommended orders to just the first (onset) of recommendations if there are
     no clinician orders between them", on the grounds that later
     recommendations in a run are made without counterfactual state estimation:
     the state was never updated with the result of the first recommended draw,
     so continuing to recommend is not a real second decision. Reported figures:
     WBC -44%, lactate -27%.

  2. INFORMATION GAIN per order. The approximate true value of the lab is imputed
     with the forecaster's SMOOTHER, which sees all observations including future
     ones, and compared against the filter-only forecast the policy actually had.
     This is the one place a future-looking quantity is legitimate, because it is
     scoring a decision after the fact rather than making one. Reported means
     (clinician vs MO-FQI): 0.69/1.53 WBC, 0.09/0.18 creatinine, 1.63/3.39 BUN,
     0.19/0.38 lactate.

  3. TIME TO TREATMENT ONSET. For each initiation of vasopressors, antibiotics,
     ventilation or dialysis, trace back to the earliest order in the preceding
     48 hours and take the gap. Reported means (clinician vs MO-FQI): 9.1/13.2
     WBC, 7.9/12.5 creatinine, 8.0/12.5 BUN, 14.4/15.9 lactate.

The 24-hour budget rule is applied here and its effect on order counts is broken
out separately, since it is not part of the Markov policy that stage 5 evaluates.

Output: reports/clinical_<lab>.md
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
import mofqi


def load_split(lab, split):
    d = np.load(cfg.RL_DIR / f"{lab}_{split}.npz")
    return {k: d[k] for k in d.files}


def deterministic_actions(bundle, states):
    if bundle["policy"] is not None:
        return bundle["policy"].predict(states).astype(int)
    return bundle["model"].collapse(states, bundle["eps"]).astype(int)


# ---------------------------------------------------------- order counting ----
def collapse_runs(rec, clinician, stay_ids):
    """Keep only the first recommendation in a run with no clinician order in it.

    Paper Sec. 3.1. Without this the count is inflated by repeat recommendations
    made from a state that was never updated with the first draw's result.
    """
    out = np.zeros_like(rec)
    n = len(rec)
    start = 0
    for i in range(1, n + 1):
        if i == n or stay_ids[i] != stay_ids[start]:
            armed = True          # a fresh recommendation is allowed
            for t in range(start, i):
                if clinician[t] == 1:
                    armed = True  # the state genuinely refreshed
                if rec[t] == 1 and armed:
                    out[t] = 1
                    armed = False
            start = i
    return out


# -------------------------------------------------------- information gain ----
def information_gain(split, mask, normalize=False):
    """|approximate true value - forecast| at the hours flagged by `mask`.

    The "approximate true value" is the smoother's posterior mean, which uses all
    observations in the stay; the forecast is the filter's past-only mean, which
    is what the decision was actually made on.
    """
    g = np.abs(split["smooth_lab"] - split["mean_lab"])
    if normalize:
        g = g / np.maximum(split["std_lab"], cfg.FORECAST_MIN_STD)
    v = g[mask.astype(bool)]
    return v[np.isfinite(v)]


# ------------------------------------------------------ time to treatment ----
def time_to_treatment(split, order_mask, lookback=cfg.TREATMENT_LOOKBACK_HOURS):
    """Hours from the earliest order in the preceding window to each onset."""
    stay = split["stay_id"]
    hour = split["hour"]
    onset = split["onset_any"].astype(bool)
    orders = order_mask.astype(bool)

    gaps = []
    for s in np.unique(stay):
        sel = stay == s
        h = hour[sel]
        o_h = h[orders[sel]]
        if len(o_h) == 0:
            continue
        for t in h[onset[sel]]:
            window = o_h[(o_h <= t) & (o_h >= t - lookback)]
            if len(window):
                gaps.append(float(t - window.min()))
    return np.array(gaps)


def describe(v):
    if len(v) == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan")}
    return {"n": int(len(v)), "mean": float(np.mean(v)),
            "median": float(np.median(v))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", required=True)
    ap.add_argument("--learner", default="mofqi", choices=["mofqi", "cql"],
                    help="which stage-4 bundle to evaluate")
    args = ap.parse_args()

    cfg.ensure_dirs()
    lab = args.lab
    with open(cfg.MODELS_DIR / f"{lab}_{args.learner}.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    test = load_split(lab, "test")

    clinician = test["action"].astype(int)
    raw = deterministic_actions(bundle, test["state"])
    filtered = collapse_runs(raw, clinician, test["stay_id"])
    budgeted = mofqi.apply_budget(filtered, test["stay_id"], test["hour"])

    n_clin = int(clinician.sum())
    counts = {"clinician": n_clin,
              "policy (raw)": int(raw.sum()),
              "policy (run-filtered)": int(filtered.sum()),
              "policy (+ 24h budget)": int(budgeted.sum())}
    print(f"{lab}: order counts on {len(np.unique(test['stay_id'])):,} test stays")
    for k, v in counts.items():
        pct = "" if k == "clinician" else f"   ({(v - n_clin) / max(n_clin, 1):+.1%} vs clinician)"
        print(f"  {k:24s} {v:7,}{pct}")

    ig_clin = information_gain(test, clinician)
    ig_pol = information_gain(test, budgeted)
    ig_clin_n = information_gain(test, clinician, normalize=True)
    ig_pol_n = information_gain(test, budgeted, normalize=True)
    print(f"\n  information gain (raw units)        clinician "
          f"{describe(ig_clin)['mean']:.4f}  policy {describe(ig_pol)['mean']:.4f}")
    print(f"  information gain (sigma-normalized) clinician "
          f"{describe(ig_clin_n)['mean']:.4f}  policy {describe(ig_pol_n)['mean']:.4f}")

    tt_clin = time_to_treatment(test, clinician)
    tt_pol = time_to_treatment(test, budgeted)
    print(f"\n  time to treatment onset (h)  clinician "
          f"{describe(tt_clin)['mean']:.2f} (n={len(tt_clin)})  policy "
          f"{describe(tt_pol)['mean']:.2f} (n={len(tt_pol)})")

    sfx = "" if args.learner == "mofqi" else f"_{args.learner}"
    np.savez_compressed(cfg.RL_DIR / f"{lab}_clinical{sfx}.npz",
                        ig_clin=ig_clin, ig_pol=ig_pol,
                        tt_clin=tt_clin, tt_pol=tt_pol,
                        counts=np.array(list(counts.values())))

    L = [
        f"# Clinical metrics: {lab}"
        + (f" ({args.learner.upper()} arm)" if args.learner != "mofqi" else "")
        + "\n\n",
        f"Test split, {len(np.unique(test['stay_id'])):,} ICU stays, "
        f"{len(clinician):,} hourly decisions.\n\n",
        "## 1. Order counts (paper Sec. 3.1)\n\n",
        "| policy | orders | vs clinician |\n|---|---|---|\n",
    ]
    for k, v in counts.items():
        pct = "-" if k == "clinician" else f"{(v - n_clin) / max(n_clin, 1):+.1%}"
        L.append(f"| {k} | {v:,} | {pct} |\n")
    L.append(
        "\n`run-filtered` applies the paper's rule of counting only the first "
        "recommendation in a run with no clinician order in between: later "
        "recommendations in a run are made from a state that was never updated "
        "with the first draw's result, so they are not independent decisions. "
        "`+ 24h budget` then forces one order per 24-hour window the policy "
        "leaves silent. The headline reduction the paper quotes corresponds to "
        "the run-filtered row.\n")

    L += ["\n## 2. Information gain per order (paper Fig. 5)\n\n",
          "| | n orders | mean | median |\n|---|---|---|---|\n"]
    for name, v in [("clinician (raw)", ig_clin), ("MO-FQI (raw)", ig_pol),
                    ("clinician (sigma-normalized)", ig_clin_n),
                    ("MO-FQI (sigma-normalized)", ig_pol_n)]:
        d = describe(v)
        L.append(f"| {name} | {d['n']:,} | {d['mean']:.4f} | {d['median']:.4f} |\n")
    L.append(
        "\nThe approximate true value comes from the forecaster's SMOOTHER, which "
        "uses every observation in the stay including future ones. That is "
        "legitimate here and only here: this metric scores a decision after the "
        "fact rather than informing one. The same quantity never enters the "
        "state vector.\n")

    L += ["\n## 3. Time to treatment onset (paper Fig. 6)\n\n",
          f"Lookback window {cfg.TREATMENT_LOOKBACK_HOURS}h, over initiations of "
          f"vasopressors, antibiotics, ventilation or dialysis.\n\n",
          "| | n onsets matched | mean hours | median hours |\n|---|---|---|---|\n"]
    for name, v in [("clinician", tt_clin), ("MO-FQI", tt_pol)]:
        d = describe(v)
        L.append(f"| {name} | {d['n']:,} | {d['mean']:.2f} | {d['median']:.2f} |\n")
    L.append(
        "\nA larger number means the order preceded the intervention by longer, "
        "which the paper reads as the lab having been available earlier to inform "
        "it. Note this is a lead-time comparison between two order schedules, not "
        "evidence that the intervention would actually have started sooner: "
        "nothing here is counterfactual, and a policy that simply orders more "
        "often has more chances to land early in the window.\n")

    out = cfg.REPORTS_DIR / f"clinical_{lab}{sfx}.md"
    out.write_text("".join(L), encoding="utf-8")
    print(f"\nwrote report -> {out}")


if __name__ == "__main__":
    main()
