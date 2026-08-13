"""
Stage 2: resample every stay onto a one-hour grid and forecast it (paper Sec. 2.1).

The paper resamples the raw, irregularly sampled traces with a multi-output
Gaussian process to obtain hourly predictive means and standard deviations. Here
that job is done by whatever `config.FORECASTER` selects, behind the interface in
`forecast.py`. The forecaster is fit on TRAIN subjects only.

What each output column is, and which ones the policy is allowed to see:

    mean_<trait>, std_<trait>   one-step-ahead predictive moments, PAST-ONLY.
                                These are m_t and sigma_t in the state (Sec. 2.2)
                                and in the information reward (Eq. 5).
    obs_<lab>                   the raw value measured in hour t, NaN if none.
                                Its not-NaN pattern IS the clinician's action.
    last_<lab>                  y_t, the last observed value carried forward from
                                strictly earlier hours (Sec. 2.2).
    delta_<lab>                 Delta_t, hours since that lab was last ordered.
    smooth_<lab>                posterior mean using ALL observations. EVALUATION
                                ONLY (Sec. 3.1 information gain). Stage 3 must
                                never read these into the state.
    poe_*                       lab-order workflow features built from POE rows
                                strictly before hour t. POE is not linked to a
                                particular specimen; current-hour counts are
                                retained for audit only and never enter state.
    gcs_total                   last-value-imputed GCS sum, for SOFA.
    ventilated, vaso_class, vaso_rate   per-hour intervention status, for SOFA.

Output: data/processed/hourly/<split>.parquet, one row per (stay_id, hour).
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
import itemids as ids
import forecast as fc
import forecaster_validation as fv
import sofa as sofa_mod
import target_lab_forecaster as tlf

TRAITS = ids.FORECAST_TRAITS
TRAIT_IDX = {t: i for i, t in enumerate(TRAITS)}
GCS_TRAITS = list(ids.GCS_ITEMS)
GCS_IDX = {t: i for i, t in enumerate(GCS_TRAITS)}
BATCH_STAYS = 1000
FORECASTER_PKL = cfg.MODELS_DIR / "forecaster.pkl"
TARGET_FORECASTER_PKL = cfg.MODELS_DIR / "target_lab_forecaster.pkl"


# ------------------------------------------------------------ grid assembly ----
def load_events():
    frames = []
    for path in (cfg.CHART_PARQUET, cfg.LAB_PARQUET):
        frames.append(pd.read_parquet(path))
    ev = pd.concat(frames, ignore_index=True)
    return ev


def assign_hours(ev, stays):
    """Hour index of each event, relative to ICU intime, clipped to the stay."""
    lengths = stay_lengths(stays)
    ev = ev.merge(stays[["stay_id", "intime"]], on="stay_id", how="inner")
    delta = (ev["charttime"] - ev["intime"]).dt.total_seconds() / 3600.0
    ev = ev.assign(hour=np.floor(delta.to_numpy()).astype("int64"))
    ev = ev[ev["hour"] >= 0]
    n_hours = ev["stay_id"].map(lengths)
    return ev[ev["hour"] < n_hours.to_numpy()]


def stay_lengths(stays):
    """Hours on the grid per stay, from the recorded ICU length of stay."""
    n = np.ceil(stays["los"].to_numpy() * 24.0).astype("int64")
    return pd.Series(np.maximum(n, 2), index=stays["stay_id"].to_numpy())


def poe_feature_grids(poe_orders, stay_ids, lengths, intimes, tmax):
    """Past-only POE workflow features for one batch of stays.

    Multiple POE rows at exactly the same timestamp are one order group, while
    ``rows`` preserves how many component order records were entered. Features
    at hour t only use order groups from hours < t.
    """
    b = len(stay_ids)
    row = {s: i for i, s in enumerate(stay_ids)}
    groups = np.zeros((b, tmax), dtype=np.int16)
    rows = np.zeros((b, tmax), dtype=np.int16)

    sub = poe_orders[poe_orders["stay_id"].isin(row)].copy()
    if len(sub):
        grouped = (sub.groupby(["stay_id", "ordertime"], as_index=False)
                   .size().rename(columns={"size": "n_rows"}))
        for r in grouped.itertuples(index=False):
            i = row[r.stay_id]
            h = int(np.floor(
                (r.ordertime - intimes[r.stay_id]).total_seconds() / 3600.0))
            if 0 <= h < int(lengths[r.stay_id]):
                groups[i, h] += 1
                rows[i, h] += int(r.n_rows)

    prior_groups = np.zeros_like(groups, dtype=np.int32)
    prior_rows = np.zeros_like(rows, dtype=np.int32)
    prior_groups[:, 1:] = groups[:, :-1]
    prior_rows[:, 1:] = rows[:, :-1]

    def rolling_sum(x, window):
        c = np.cumsum(x, axis=1, dtype=np.int32)
        out = c.copy()
        if window < x.shape[1]:
            out[:, window:] -= c[:, :-window]
        return out

    groups_6h = rolling_sum(prior_groups, cfg.POE_RECENT_HOURS)
    rows_6h = rolling_sum(prior_rows, cfg.POE_RECENT_HOURS)
    groups_24h = rolling_sum(prior_groups, cfg.POE_LONG_HOURS)

    since = np.empty((b, tmax), dtype=np.float32)
    for i in range(b):
        last = -1
        for t in range(tmax):
            if t > 0 and groups[i, t - 1] > 0:
                last = t - 1
            elapsed = t - last if last >= 0 else t + 1
            since[i, t] = min(elapsed, cfg.POE_MAX_SINCE_HOURS)

    return {
        "poe_order_groups_current_hour": groups,
        "poe_order_rows_current_hour": rows,
        "poe_order_groups_6h": groups_6h,
        "poe_order_rows_6h": rows_6h,
        "poe_order_groups_24h": groups_24h,
        "poe_hours_since_lab_order": since,
    }


def dense_batch(ev_batch, stay_ids, lengths, trait_idx, n_traits):
    """[B, Tmax, K] observation array, NaN where nothing was measured.

    Several values of the same trait can land in one hour (a repeated blood gas,
    a re-checked pressure); they are averaged.
    """
    b = len(stay_ids)
    tmax = int(max(lengths[s] for s in stay_ids))
    row = {s: i for i, s in enumerate(stay_ids)}

    total = np.zeros((b, tmax, n_traits), dtype=np.float64)
    count = np.zeros((b, tmax, n_traits), dtype=np.int32)
    if len(ev_batch):
        r = ev_batch["stay_id"].map(row).to_numpy()
        h = ev_batch["hour"].to_numpy()
        k = ev_batch["trait"].map(trait_idx).to_numpy()
        keep = ~pd.isna(k)
        r, h, k = r[keep].astype(int), h[keep].astype(int), k[keep].astype(int)
        v = ev_batch["value"].to_numpy(dtype="float64")[keep]
        np.add.at(total, (r, h, k), v)
        np.add.at(count, (r, h, k), 1)

    with np.errstate(invalid="ignore"):
        obs = np.where(count > 0, total / count, np.nan)
    return obs.astype(np.float64), tmax


def forward_fill(a):
    """LOCF along axis 1 of a [B, T, K] array."""
    seen = ~np.isnan(a)
    idx = np.where(seen, np.arange(a.shape[1])[None, :, None], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = np.take_along_axis(a, idx, axis=1)
    # Nothing seen yet at the head of the stay stays NaN.
    return np.where(np.maximum.accumulate(seen, axis=1), out, np.nan)


def last_and_delta(obs_lab):
    """y_t and Delta_t for [B, T, L], using observations strictly before hour t.

    Shifting by one hour before the fill is what keeps the result of the order
    being decided at hour t out of the state.
    """
    b, t, l = obs_lab.shape
    shifted = np.full_like(obs_lab, np.nan)
    shifted[:, 1:, :] = obs_lab[:, :-1, :]
    last = forward_fill(shifted)

    seen = ~np.isnan(shifted)
    hours = np.arange(t)[None, :, None]
    last_hour = np.where(seen, hours, -1)
    np.maximum.accumulate(last_hour, axis=1, out=last_hour)
    # `shifted` is lagged by one, so a measurement taken in hour h appears at
    # index h+1. Elapsed time is therefore (t - last_hour) + 1, and a lab drawn
    # in the immediately preceding hour has Delta = 1, not 0.
    delta = np.where(last_hour >= 0, hours - last_hour + 1, np.nan)
    return last, delta.astype(np.float64)


# ------------------------------------------------------- intervention status ----
def intervention_grids(interventions, stay_ids, lengths, intimes, tmax):
    """Per-hour intervention status.

    Returns per-kind ACTIVE grids (an intervention of that kind is running in
    hour t), per-kind ONSET grids (it is running in hour t and was not in t-1),
    plus the vasopressor class and rate that SOFA's cardiovascular grade needs.

    Eq. 4 rewards a lab that is "immediately followed by an intervention", so the
    reward reads the onset grid: restarting an infusion that has been running for
    two days is not new clinical information.
    """
    b = len(stay_ids)
    row = {s: i for i, s in enumerate(stay_ids)}
    active = {k: np.zeros((b, tmax), dtype=bool) for k in ids.INTERVENTION_KINDS}
    vclass = np.zeros((b, tmax), dtype=np.int8)
    vrate = np.full((b, tmax), np.nan, dtype=np.float64)

    sub = interventions[interventions["stay_id"].isin(row)]
    for r in sub.itertuples(index=False):
        i = row[r.stay_id]
        n = int(lengths[r.stay_id])
        t0 = (r.starttime - intimes[r.stay_id]).total_seconds() / 3600.0
        t1 = t0 + 1.0 if pd.isna(r.endtime) else \
            (r.endtime - intimes[r.stay_id]).total_seconds() / 3600.0
        lo = max(int(np.floor(t0)), 0)
        hi = min(int(np.ceil(t1)), n)
        if hi <= lo:
            hi = min(lo + 1, n)
        if hi <= lo:
            continue
        if r.kind in active:
            active[r.kind][i, lo:hi] = True
        if r.kind == "vasopressor":
            c = int(sofa_mod.vasopressor_class([r.itemid])[0])
            vclass[i, lo:hi] = np.maximum(vclass[i, lo:hi], c)
            if not pd.isna(r.rate):
                cur = vrate[i, lo:hi]
                vrate[i, lo:hi] = np.where(np.isnan(cur), r.rate,
                                           np.maximum(cur, r.rate))

    onset = {}
    for k, a in active.items():
        prev = np.zeros_like(a)
        prev[:, 1:] = a[:, :-1]
        onset[k] = a & ~prev
    return active, onset, vclass, vrate


# ---------------------------------------------------------------- assembly ----
def _slice_by_stay(ev, batch_ids):
    """Rows for a batch of stays, from a frame already sorted by stay_id.

    Grouping into a dict of per-stay frames would allocate one DataFrame per ICU
    stay; at full cohort size that is tens of thousands of objects over tens of
    millions of rows. Sorting once and slicing with searchsorted is equivalent
    and stays flat in memory.
    """
    if not len(ev):
        return ev
    keys = ev["stay_id"].to_numpy()
    lo = np.searchsorted(keys, batch_ids, side="left")
    hi = np.searchsorted(keys, batch_ids, side="right")
    parts = [np.arange(a, b) for a, b in zip(lo, hi) if b > a]
    if not parts:
        return ev.iloc[:0]
    return ev.take(np.concatenate(parts))


def build_split(split, stays, ev, gcs_ev, interventions, poe_orders,
                forecaster, out_path):
    stay_ids = np.sort(stays["stay_id"].to_numpy())
    lengths = stay_lengths(stays)
    intimes = dict(zip(stays["stay_id"], stays["intime"]))
    subject_of = dict(zip(stays["stay_id"], stays["subject_id"]))

    keep = set(stay_ids.tolist())
    ev = ev[ev["stay_id"].isin(keep)].sort_values("stay_id", kind="stable")
    gcs_ev = gcs_ev[gcs_ev["stay_id"].isin(keep)].sort_values("stay_id", kind="stable")
    interventions = interventions[interventions["stay_id"].isin(keep)]

    lab_cols = [TRAIT_IDX[l] for l in ids.TARGET_LABS]
    chunks = []
    for start in range(0, len(stay_ids), BATCH_STAYS):
        batch = stay_ids[start:start + BATCH_STAYS]
        eb = _slice_by_stay(ev, batch)
        gb = _slice_by_stay(gcs_ev, batch)

        obs, tmax = dense_batch(eb, batch, lengths, TRAIT_IDX, len(TRAITS))
        gcs_obs, _ = dense_batch(gb, batch, lengths, GCS_IDX, len(GCS_TRAITS))

        # Per-stay hour counts, so the smoother's backward pass never starts in
        # the padding that batching adds to shorter stays.
        batch_lengths = np.array([int(lengths[s]) for s in batch])
        mean, std = forecaster.filter(obs, batch_lengths)
        smooth_mean, _ = forecaster.smooth(obs, batch_lengths)

        obs_lab = obs[:, :, lab_cols]
        last, delta = last_and_delta(obs_lab)
        # GCS is a sum of three components, so it is only defined once all three
        # have been seen. Summing a partial set would understate the total and
        # push SOFA's CNS grade up spuriously; leave it NaN and let sofa.cns()
        # read that as no dysfunction.
        gcs_filled = forward_fill(gcs_obs)
        complete = ~np.isnan(gcs_filled).any(axis=2)
        gcs_total = np.where(complete, np.nansum(gcs_filled, axis=2), np.nan)

        active, onset, vclass, vrate = intervention_grids(
            interventions, batch, lengths, intimes, tmax)
        poe = poe_feature_grids(poe_orders, batch, lengths, intimes, tmax)

        for i, s in enumerate(batch):
            n = int(lengths[s])
            d = {"stay_id": np.full(n, s, dtype="int32"),
                 "subject_id": np.full(n, subject_of[s], dtype="int32"),
                 "hour": np.arange(n, dtype="int32")}
            for k, t in enumerate(TRAITS):
                d[f"mean_{t}"] = mean[i, :n, k]
                d[f"std_{t}"] = std[i, :n, k]
            for j, l in enumerate(ids.TARGET_LABS):
                d[f"obs_{l}"] = obs_lab[i, :n, j].astype("float32")
                d[f"last_{l}"] = last[i, :n, j].astype("float32")
                d[f"delta_{l}"] = delta[i, :n, j].astype("float32")
                d[f"smooth_{l}"] = smooth_mean[i, :n, TRAIT_IDX[l]]
            d["gcs_total"] = gcs_total[i, :n].astype("float32")
            d["ventilated"] = active["ventilation"][i, :n]
            d["vaso_class"] = vclass[i, :n]
            d["vaso_rate"] = vrate[i, :n].astype("float32")
            for k in ids.INTERVENTION_KINDS:
                d[f"active_{k}"] = active[k][i, :n]
                d[f"onset_{k}"] = onset[k][i, :n]
            for name, values in poe.items():
                d[name] = values[i, :n]
            chunks.append(pd.DataFrame(d))
        print(f"    {split}: {min(start + BATCH_STAYS, len(stay_ids))}/{len(stay_ids)} stays",
              flush=True)

    out = pd.concat(chunks, ignore_index=True)
    out.to_parquet(out_path, index=False)
    print(f"  {split}: {len(out):,} stay-hours -> {out_path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecaster", default=None, help="override config.FORECASTER")
    args = ap.parse_args()

    cfg.ensure_dirs()
    stays = pd.read_parquet(cfg.COHORT_PARQUET)
    all_ev = load_events()
    all_ev = assign_hours(all_ev, stays)
    interventions = pd.read_parquet(cfg.INTERVENTIONS_PARQUET)
    if not cfg.POE_PARQUET.exists():
        raise SystemExit(f"{cfg.POE_PARQUET} not found; run stage 1 first")
    poe_orders = pd.read_parquet(cfg.POE_PARQUET)

    gcs_ev = all_ev[all_ev["trait"].isin(GCS_TRAITS)]
    ev = all_ev[all_ev["trait"].isin(TRAIT_IDX)]
    print(f"{len(ev):,} forecast events, {len(gcs_ev):,} GCS events, "
          f"{len(stays):,} stays")

    # Fit on TRAIN only. Selection and uncertainty calibration use VAL only;
    # TEST remains locked until the validation gate passes.
    train_stays = stays[stays["split"] == "train"]
    lengths = stay_lengths(train_stays)
    rng = np.random.default_rng(cfg.SEED)
    train_ids = train_stays["stay_id"].to_numpy()
    fit_ids = rng.choice(
        train_ids, min(cfg.FORECAST_TRAIN_STAYS, len(train_ids)), replace=False)
    fit_ev = ev[ev["stay_id"].isin(set(fit_ids))]
    fit_obs, _ = dense_batch(fit_ev, fit_ids, lengths, TRAIT_IDX, len(TRAITS))
    print(f"\nfitting {args.forecaster or cfg.FORECASTER} on {len(fit_ids)} train stays")
    forecaster = fc.build_forecaster(TRAITS, args.forecaster).fit(fit_obs)

    val_stays = stays[stays["split"] == "val"]
    val_ids_all = val_stays["stay_id"].to_numpy()
    val_ids = rng.choice(
        val_ids_all, min(cfg.FORECAST_VALIDATION_STAYS, len(val_ids_all)),
        replace=False)
    val_lengths = stay_lengths(val_stays)
    val_ev = ev[ev["stay_id"].isin(set(val_ids))]
    val_obs, _ = dense_batch(
        val_ev, val_ids, val_lengths, TRAIT_IDX, len(TRAITS))
    val_batch_lengths = np.array([int(val_lengths[s]) for s in val_ids])
    print(f"\ntuning target-lab dynamics and sigma on {len(val_ids)} validation stays")
    tuning = forecaster.tune_on_validation(
        val_obs, ids.TARGET_LABS, val_batch_lengths)
    for lab, row in tuning.items():
        print(f"  {lab:11s} nll={row['validation_nll']:.4f} "
              f"q_level*x{row['q_level_scale']:.1f} "
              f"q_slope*x{row['q_slope_scale']:.1f} "
              f"r*x{row['r_scale']:.1f} "
              f"sigma*x{row['std_calibration_scale']:.3f}")

    if hasattr(forecaster, "state_dict"):
        FORECASTER_PKL.parent.mkdir(parents=True, exist_ok=True)
        with open(FORECASTER_PKL, "wb") as fh:
            pickle.dump(forecaster.state_dict(), fh)
        for t, ql, qs, r in zip(TRAITS, forecaster.q_level_,
                                forecaster.q_slope_, forecaster.r_):
            print(f"    {t:12s} q_level={ql:8.5f}  q_slope={qs:9.6f}  r={r:7.4f}")

    print("\nbuilding train and validation hourly grids")
    for split in ("train", "val"):
        sub = stays[stays["split"] == split]
        if sub.empty:
            print(f"  {split}: no stays, skipping")
            continue
        build_split(split, sub, ev, gcs_ev, interventions, poe_orders, forecaster,
                    cfg.HOURLY_DIR / f"{split}.parquet")

    population_means = {
        lab: float(forecaster.mu_[TRAIT_IDX[lab]]) for lab in ids.TARGET_LABS
    }
    # The state-space model is intentionally only the base representation. A
    # multivariate target-lab model is fit on TRAIN labels, selected/calibrated
    # on VAL, and written back into mean/std for the four policy targets.
    train_path = cfg.HOURLY_DIR / "train.parquet"
    val_path = cfg.HOURLY_DIR / "val.parquet"
    train_hourly = pd.read_parquet(train_path)
    val_hourly = pd.read_parquet(val_path)
    print("\nfitting validated multivariate target-lab forecaster")
    target_forecaster = tlf.TargetLabForecaster().fit(train_hourly, val_hourly)
    train_hourly = target_forecaster.transform(train_hourly, training=True)
    val_hourly = target_forecaster.transform(val_hourly, training=False)
    train_hourly.to_parquet(train_path, index=False)
    val_hourly.to_parquet(val_path, index=False)
    TARGET_FORECASTER_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(TARGET_FORECASTER_PKL, "wb") as fh:
        pickle.dump(target_forecaster.state_dict(), fh)
    for lab, row in target_forecaster.selection_.items():
        candidates = ", ".join(
            f"{c['name']}={c['validation_rmse']:.4f}"
            for c in row["candidates"])
        print(f"  {lab:11s} selected={row['selected']} "
              f"interval*x{row['interval_scale']:.3f} [{candidates}]")

    population_means = {
        lab: float(train_hourly[f"obs_{lab}"].mean()) for lab in ids.TARGET_LABS
    }
    val_metrics = fv.evaluate_hourly(
        cfg.HOURLY_DIR / "val.parquet", population_means)
    val_gate = fv.validation_gate(val_metrics)
    print("\nforecaster validation gate:", "PASS" if val_gate["passed"] else "FAIL")
    print(f"  aggregate RMSE skill vs LOCF "
          f"{val_gate['aggregate_rmse_skill_vs_locf']:+.3f}")
    print(f"  targets beating LOCF {val_gate['target_labs_beating_locf']}/4")
    print(f"  median utility Spearman {val_gate['median_utility_spearman']:.3f}")
    print(f"  mean 90% coverage {val_gate['mean_coverage_90']:.3f}")
    if not val_gate["passed"]:
        fv.write_report(
            {"metrics": val_metrics, "gate": val_gate},
            {"metrics": {}, "gate": None},
            {"base_state_space": tuning,
             "target_lab_model": target_forecaster.selection_},
            cfg.REPORTS_DIR / "forecaster_validation")
        raise SystemExit(
            "forecaster validation failed; test split was not evaluated:\n  - "
            + "\n  - ".join(val_gate["reasons"]))

    print("\nvalidation passed; building locked test hourly grid")
    test_stays = stays[stays["split"] == "test"]
    build_split("test", test_stays, ev, gcs_ev, interventions, poe_orders,
                forecaster, cfg.HOURLY_DIR / "test.parquet")
    test_path = cfg.HOURLY_DIR / "test.parquet"
    test_hourly = target_forecaster.transform(
        pd.read_parquet(test_path), training=False)
    test_hourly.to_parquet(test_path, index=False)
    test_metrics = fv.evaluate_hourly(
        cfg.HOURLY_DIR / "test.parquet", population_means)
    test_gate = fv.validation_gate(test_metrics)
    md_path, json_path = fv.write_report(
        {"metrics": val_metrics, "gate": val_gate},
        {"metrics": test_metrics, "gate": test_gate},
        {"base_state_space": tuning,
         "target_lab_model": target_forecaster.selection_},
        cfg.REPORTS_DIR / "forecaster_validation")
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
