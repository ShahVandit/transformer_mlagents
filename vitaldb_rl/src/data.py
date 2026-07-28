"""VitalDB loading and 1-minute case cache for anesthetic infusion RL."""
from __future__ import annotations

import io
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

API = "https://api.vitaldb.net"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

TRACKS = {
    "map": ("Solar8000/ART_MBP", (20.0, 200.0)),
    "sbp": ("Solar8000/ART_SBP", (30.0, 300.0)),
    "dbp": ("Solar8000/ART_DBP", (10.0, 200.0)),
    "hr": ("Solar8000/HR", (20.0, 220.0)),
    "spo2": ("Solar8000/PLETH_SPO2", (50.0, 100.0)),
    "etco2": ("Solar8000/ETCO2", (0.0, 100.0)),
    "bis": ("BIS/BIS", (0.0, 100.0)),
    "ppf_rate": ("Orchestra/PPF20_RATE", (0.0, 1200.0)),
    "rftn_rate": ("Orchestra/RFTN20_RATE", (0.0, 1200.0)),
}

REQUIRED = ["map", "bis", "ppf_rate", "rftn_rate"]
STATIC_COLS = [
    "age",
    "sex",
    "height",
    "weight",
    "bmi",
    "asa",
    "emop",
    "optype",
    "ane_type",
    "preop_htn",
    "preop_dm",
]


def _get_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))


def get_cases(refresh: bool = False) -> pd.DataFrame:
    fp = DATA / "_cases.parquet"
    if fp.exists() and not refresh:
        return pd.read_parquet(fp)
    df = _get_csv(f"{API}/cases")
    df.to_parquet(fp)
    return df


def get_trks(refresh: bool = False) -> pd.DataFrame:
    fp = DATA / "_trks.parquet"
    if fp.exists() and not refresh:
        return pd.read_parquet(fp)
    df = _get_csv(f"{API}/trks")
    df.to_parquet(fp)
    return df


def track_case_sets(trks: pd.DataFrame) -> dict[str, set[int]]:
    return {
        key: set(trks.loc[trks.tname == name, "caseid"].astype(int))
        for key, (name, _rng) in TRACKS.items()
    }


def select_cohort(min_age: float = 18.0, min_dur_min: float = 30.0) -> pd.DataFrame:
    """Select adult cases with required state/action tracks and usable anesthesia time."""
    cases = get_cases()
    trks = get_trks()
    sets = track_case_sets(trks)
    required_ids = set.intersection(*(sets[k] for k in REQUIRED))

    c = cases[cases.caseid.astype(int).isin(required_ids)].copy()
    c["win_start"] = c["anestart"].clip(lower=0)
    c["win_end"] = c["aneend"]
    c["dur_min"] = (c["win_end"] - c["win_start"]) / 60.0
    keep = (
        c["win_end"].notna()
        & (pd.to_numeric(c["age"], errors="coerce") >= min_age)
        & (c["dur_min"] >= min_dur_min)
    )
    cols = ["caseid", "subjectid", "win_start", "win_end", "dur_min"] + [
        col for col in STATIC_COLS if col in c.columns
    ]
    return c.loc[keep, cols].reset_index(drop=True)


def subject_split(
    meta: pd.DataFrame,
    fracs=(0.6, 0.2, 0.2),
    seed: int = 0,
) -> dict[str, list[int]]:
    """Split by subject, not case, to avoid repeated-surgery leakage."""
    rng = np.random.default_rng(seed)
    subjects = meta["subjectid"].dropna().unique().copy()
    rng.shuffle(subjects)
    n = len(subjects)
    n_train = int(fracs[0] * n)
    n_val = n_train + int(fracs[1] * n)
    subj = {
        "train": subjects[:n_train],
        "val": subjects[n_train:n_val],
        "test": subjects[n_val:],
    }
    splits = {
        k: meta.loc[meta.subjectid.isin(v), "caseid"].astype(int).tolist()
        for k, v in subj.items()
    }
    with open(DATA / "splits.json", "w") as f:
        json.dump(
            {
                "subjects": {k: [int(x) for x in v] for k, v in subj.items()},
                "caseids": splits,
            },
            f,
            indent=2,
        )
    return splits


def _tids_for(trks: pd.DataFrame, caseid: int) -> dict[str, str | None]:
    sub = trks[trks.caseid.astype(int) == int(caseid)]
    tids = {}
    for key, (name, _rng) in TRACKS.items():
        rows = sub[sub.tname == name]
        tids[key] = rows.iloc[0].tid if len(rows) else None
    return tids


def _clean_signal(tid: str | None, ws: float, we: float, rng: tuple[float, float]) -> pd.Series:
    if tid is None:
        return pd.Series(dtype=float)
    df = _get_csv(f"{API}/{tid}")
    if df.shape[1] < 2 or df.empty:
        return pd.Series(dtype=float)
    t = df.iloc[:, 0].to_numpy(float)
    v = df.iloc[:, 1].to_numpy(float)
    lo, hi = rng
    keep = (t >= ws) & (t <= we) & np.isfinite(v) & (v >= lo) & (v <= hi)
    if not keep.any():
        return pd.Series(dtype=float)
    minute = np.floor((t[keep] - ws) / 60.0).astype(int)
    s = pd.Series(v[keep], index=minute)
    return s.groupby(level=0).median()


def clean_case(caseid: int, ws: float, we: float, trks: pd.DataFrame) -> pd.DataFrame:
    tids = _tids_for(trks, caseid)
    n = int(np.floor((we - ws) / 60.0)) + 1
    grid = pd.RangeIndex(0, n, name="minute")
    out = pd.DataFrame(index=grid)
    for key, (_name, rng) in TRACKS.items():
        out[key] = _clean_signal(tids[key], ws, we, rng).reindex(grid)
    if out["map"].notna().sum() < 20:
        return pd.DataFrame()
    return out.reset_index()


def build_cache(
    limit: int | None = None,
    workers: int = 8,
    refresh: bool = False,
    min_dur_min: float = 30.0,
) -> pd.DataFrame:
    """Download and cache one parquet file per selected case."""
    cohort = select_cohort(min_dur_min=min_dur_min)
    if limit is not None:
        cohort = cohort.head(limit)
    trks = get_trks()

    def work(row):
        cid = int(row.caseid)
        fp = DATA / f"{cid}.parquet"
        if fp.exists() and not refresh:
            return cid, True
        try:
            df = clean_case(cid, float(row.win_start), float(row.win_end), trks)
            if df.empty:
                return cid, False
            df.to_parquet(fp)
            return cid, True
        except Exception:
            return cid, False

    ok = set()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, row) for row in cohort.itertuples(index=False)]
        for i, fut in enumerate(as_completed(futs), 1):
            cid, good = fut.result()
            if good:
                ok.add(cid)
            if i % 100 == 0:
                print(f"  cached {i}/{len(cohort)} cases ({len(ok)} ok)", flush=True)

    meta = cohort[cohort.caseid.astype(int).isin(ok)].reset_index(drop=True)
    meta.to_parquet(DATA / "cohort.parquet")
    print(f"[data] cached {len(meta)} cases to {DATA}")
    return meta


def load_case(caseid: int) -> pd.DataFrame:
    return pd.read_parquet(DATA / f"{int(caseid)}.parquet")


def load_cohort() -> pd.DataFrame:
    return pd.read_parquet(DATA / "cohort.parquet")


def load_splits() -> dict[str, list[int]]:
    with open(DATA / "splits.json") as f:
        return json.load(f)["caseids"]

