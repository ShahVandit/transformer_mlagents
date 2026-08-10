"""
Stage 3: turn the hourly grid into MDP transition tuples (paper Sec. 2.2).

State, 21 dimensions
--------------------
The paper lists s_t = [m^SOFA, m^vitals, m^labs, y^labs, Delta^labs] and states a
21-dimensional state space, but never itemizes it. Carrying a standard deviation
for all eight traits gives 25; the only clean decomposition that lands on 21 is a
predictive mean for all eight traits and a predictive standard deviation for the
four labs only:

    4  mean of HR, RR, temperature, mean BP
    4  mean of creatinine, BUN, WBC, lactate
    4  std  of creatinine, BUN, WBC, lactate
    1  predictive SOFA
    4  y_t, last observed value of each lab
    4  Delta_t, hours since each lab was last ordered
    --
    21

That is defensible on its own terms: vitals are charted roughly hourly, so their
predictive standard deviation is nearly constant and carries little signal, while
a lab's grows with time since it was last drawn and is exactly what the ordering
decision turns on. Set `config.INCLUDE_VITAL_STD = True` for the 25-dim variant.

Action
------
Binary per lab: a_t = 1 iff that lab was resulted in hour t. Four independent
policies, so L = 1 each (Sec. 2.2).

Reward, 4-vector [r_SOFA, r_treat, r_info, -r_cost]
---------------------------------------------------
Exactly Eqs. 3-6. Note that three of the four terms are gated on a_t != 0 and the
fourth only fires when a_t = 1, so NOT ordering yields the zero vector in every
dimension. That is the paper's design, not an implementation slip, and it is the
single most important thing to hold in mind when reading any V_d later: a policy
that orders more often is mechanically advantaged on the three positive
components. It is restated in the stage-5 report.

Output: data/rl/<lab>_<split>.npz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
import itemids as ids
import sofa as sofa_mod

C_L_JSON = cfg.RL_DIR / "c_l.json"


# ------------------------------------------------------------------- SOFA ----
def add_sofa(df):
    """Hourly SOFA from the forecaster's predictive means (paper Sec. 2.2)."""
    total = sofa_mod.sofa_score(
        pao2=df["mean_pao2"].to_numpy(),
        fio2=df["mean_fio2"].to_numpy(),
        ventilated=df["ventilated"].to_numpy(),
        platelets=df["mean_platelets"].to_numpy(),
        bilirubin=df["mean_bilirubin"].to_numpy(),
        mbp=df["mean_mbp"].to_numpy(),
        vaso_class=df["vaso_class"].to_numpy(),
        vaso_rate=df["vaso_rate"].to_numpy(),
        gcs_total=df["gcs_total"].to_numpy(),
        creatinine=df["mean_creatinine"].to_numpy(),
    )
    df = df.copy()
    df["sofa"] = total
    # Eq. 3 needs f(.) = m^SOFA_t - m^SOFA_{t-1}, within a stay.
    df["sofa_delta"] = df.groupby("stay_id")["sofa"].diff().fillna(0.0)
    return df


# ------------------------------------------------------------------ state ----
def state_columns():
    cols = [f"mean_{t}" for t in ids.STATE_VITALS]
    if cfg.INCLUDE_VITAL_STD:
        cols += [f"std_{t}" for t in ids.STATE_VITALS]
    cols += [f"mean_{l}" for l in ids.TARGET_LABS]
    cols += [f"std_{l}" for l in ids.TARGET_LABS]
    cols += ["sofa"]
    cols += [f"last_{l}" for l in ids.TARGET_LABS]
    cols += [f"delta_{l}" for l in ids.TARGET_LABS]
    return cols


def build_state(df):
    """[N, 21] state matrix, with the two documented imputations for the head of
    a stay before a lab has ever been drawn."""
    d = df.copy()
    for l in ids.TARGET_LABS:
        # y_t before the first ever measurement: fall back to the forecaster's
        # prior mean, which is the best estimate available at that moment.
        d[f"last_{l}"] = d[f"last_{l}"].fillna(d[f"mean_{l}"])
        # Delta_t before the first ever measurement: hours since ICU admission.
        d[f"delta_{l}"] = d[f"delta_{l}"].fillna(d["hour"] + 1.0)
    cols = state_columns()
    s = d[cols].to_numpy(dtype=np.float32)
    if not np.isfinite(s).all():
        bad = np.array(cols)[~np.isfinite(s).all(axis=0)]
        raise ValueError(f"non-finite state columns: {bad.tolist()}")
    return s, cols


# ----------------------------------------------------------------- rewards ----
def reward_sofa(action, sofa_delta):
    """Eq. 3: 1[a != 0] * 1[SOFA_t - SOFA_{t-1} >= 2]."""
    return (action != 0) * (sofa_delta >= cfg.SOFA_DELTA_THRESHOLD)


def reward_treat(action, onset_next):
    """Eq. 4: 1[a != 0] * sum_i 1[intervention i initiated at t+1]."""
    return (action != 0) * onset_next


def reward_info(action, mean, y, sigma, c_l):
    """Eq. 5: max(0, |m_t - y_t| / sigma_t - c_l) * 1[a = 1].

    Where y_t does not exist yet (no prior measurement of this lab in the stay)
    the deviation is undefined and the term is zero. That slightly understates
    the value of a stay's first draw; it affects only the leading hours.
    """
    g = np.abs(mean - y) / np.maximum(sigma, cfg.FORECAST_MIN_STD)
    g = np.where(np.isfinite(g), g, 0.0)
    return np.maximum(0.0, g - c_l) * (action == 1)


def reward_cost(action, delta):
    """Eq. 6: exp(-Delta_t / Gamma) * 1[a = 1].

    A lab drawn an hour after the last one costs ~exp(-1/6) = 0.85; one drawn a
    day later costs ~0.02. The penalty is for redundancy, not for testing itself.
    """
    d = np.where(np.isfinite(delta), delta, np.inf)
    return np.exp(-d / cfg.COST_DECAY_GAMMA) * (action == 1)


def prediction_error(df, lab):
    """|m_t - y_t| / sigma_t, the raw g(.) of Eq. 5, before the c_l threshold."""
    m = df[f"mean_{lab}"].to_numpy(dtype=np.float64)
    y = df[f"last_{lab}"].to_numpy(dtype=np.float64)
    s = np.maximum(df[f"std_{lab}"].to_numpy(dtype=np.float64), cfg.FORECAST_MIN_STD)
    g = np.abs(m - y) / s
    return np.where(np.isfinite(g), g, np.nan)


def fit_c_l(train_df, lab):
    """c_l = median prediction error over labs ORDERED in the training data.

    The paper sets it this way (Sec. 2.2), which makes the information reward
    positive only for orders more informative than a typical clinician order.
    """
    g = prediction_error(train_df, lab)
    ordered = train_df[f"obs_{lab}"].notna().to_numpy()
    vals = g[ordered & np.isfinite(g)]
    return float(np.median(vals)) if len(vals) else 0.0


# ------------------------------------------------------------- assembly ----
def build_lab_split(df, lab, c_l, state, state_cols):
    """Transition tuples for one lab on one split. Terminal at each stay's end."""
    action = df[f"obs_{lab}"].notna().to_numpy().astype(np.int8)

    onset_cols = [f"onset_{k}" for k in ids.INTERVENTION_KINDS]
    onset_now = df[onset_cols].to_numpy().sum(axis=1).astype(np.float64)
    # "initiated at s_{t+1}": shift the onset count back one hour within a stay.
    onset_next = pd.Series(onset_now).groupby(
        df["stay_id"].to_numpy()).shift(-1).fillna(0.0).to_numpy()

    r = np.stack([
        reward_sofa(action, df["sofa_delta"].to_numpy()).astype(np.float32),
        reward_treat(action, onset_next).astype(np.float32),
        reward_info(action,
                    df[f"mean_{lab}"].to_numpy(dtype=np.float64),
                    df[f"last_{lab}"].to_numpy(dtype=np.float64),
                    df[f"std_{lab}"].to_numpy(dtype=np.float64),
                    c_l).astype(np.float32),
        -reward_cost(action, df[f"delta_{lab}"].to_numpy(dtype=np.float64)).astype(np.float32),
    ], axis=1)

    stay = df["stay_id"].to_numpy()
    done = np.zeros(len(df), dtype=np.float32)
    done[np.r_[np.flatnonzero(stay[1:] != stay[:-1]), len(df) - 1]] = 1.0

    # next_state is the following hour in the same stay; terminal rows point at
    # themselves and are masked out by `done` in every Bellman backup.
    nxt = np.arange(len(df)) + 1
    nxt[done == 1.0] = np.flatnonzero(done == 1.0)
    next_state = state[nxt]

    return {
        "state": state,
        "action": action.astype(np.int64),
        "reward": r,
        "next_state": next_state,
        "done": done,
        "stay_id": stay.astype(np.int64),
        "subject_id": df["subject_id"].to_numpy(dtype=np.int64),
        "hour": df["hour"].to_numpy(dtype=np.int64),
        # Carried for stage 6 only; never part of the state.
        "obs_lab": df[f"obs_{lab}"].to_numpy(dtype=np.float32),
        "smooth_lab": df[f"smooth_{lab}"].to_numpy(dtype=np.float32),
        "mean_lab": df[f"mean_{lab}"].to_numpy(dtype=np.float32),
        "std_lab": df[f"std_{lab}"].to_numpy(dtype=np.float32),
        "onset_any": (onset_now > 0).astype(np.int8),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labs", nargs="+", default=ids.TARGET_LABS)
    args = ap.parse_args()

    cfg.ensure_dirs()
    frames = {}
    for split in ("train", "val", "test"):
        p = cfg.HOURLY_DIR / f"{split}.parquet"
        if not p.exists():
            print(f"  {split}: missing, skipping")
            continue
        df = pd.read_parquet(p).sort_values(["stay_id", "hour"]).reset_index(drop=True)
        frames[split] = add_sofa(df)
        print(f"{split}: {len(df):,} stay-hours, {df['stay_id'].nunique():,} stays, "
              f"mean SOFA {frames[split]['sofa'].mean():.2f}")

    if "train" not in frames:
        raise SystemExit("no train split; run stage 2 first")

    # c_l is fit on TRAIN only and reused for val and test.
    c_l = {lab: fit_c_l(frames["train"], lab) for lab in args.labs}
    C_L_JSON.parent.mkdir(parents=True, exist_ok=True)
    C_L_JSON.write_text(json.dumps(c_l), encoding="utf-8")
    print("\nc_l (median prediction error over ordered labs, train):")
    for lab, v in c_l.items():
        print(f"  {lab:11s} {v:.4f}")

    states = {}
    for split, df in frames.items():
        s, cols = build_state(df)
        states[split] = s

    meta = {"state_cols": state_columns(), "state_dim": len(state_columns()),
            "reward_dims": cfg.REWARD_DIMS, "gamma": cfg.GAMMA, "c_l": c_l}
    (cfg.RL_DIR / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    print("\nwriting transitions")
    for lab in args.labs:
        for split, df in frames.items():
            d = build_lab_split(df, lab, c_l[lab], states[split], meta["state_cols"])
            out = cfg.RL_DIR / f"{lab}_{split}.npz"
            np.savez_compressed(out, **d)
            rate = d["action"].mean()
            rmean = d["reward"].mean(axis=0)
            print(f"  {lab:11s} {split:5s}  n={len(d['action']):7,}  "
                  f"order_rate={rate:.4f}  "
                  f"r=[{rmean[0]:+.4f} {rmean[1]:+.4f} {rmean[2]:+.4f} {rmean[3]:+.4f}]")

    print(f"\nstate dim {meta['state_dim']}: {meta['state_cols']}")


if __name__ == "__main__":
    main()
