"""
Stage 1: cohort selection and raw event extraction (paper Sec. 2.1).

Cohort
------
Adult ICU stays with a length of stay between one and twenty days, keeping only
stays with at least one recorded measure of each of the 20 traits in
`itemids.TRAITS`. The paper applied the same two filters to MIMIC-III and landed
on 6,060 admissions; MIMIC-IV is a larger and more recent database, so the count
here will be substantially higher.

Extraction
----------
`chartevents.csv.gz` (3.5 GB) and `labevents.csv.gz` (2.6 GB) are each scanned
once in chunks, filtered to the cohort and to the itemids we actually use, and
written incrementally to parquet. This is the slowest step in the pipeline and it
runs exactly once; every later stage reads the parquet caches.

`labevents` carries no `stay_id`, only `hadm_id`, so lab rows are attached to an
ICU stay by matching `hadm_id` and requiring `charttime` to fall inside
[intime, outtime].

Intervention onsets (vasopressors, ventilation, dialysis, antibiotics) are pulled
from `inputevents`, `procedureevents` and `prescriptions` for the Eq. 4 reward
term and for the stage-6 time-to-treatment metric.

Splits
------
By `subject_id`, never by stay: 65,366 subjects hold 94,458 ICU stays in MIMIC-IV,
so a stay-level split would put the same patient in train and test.

Outputs
-------
    data/cache/cohort.parquet               one row per included ICU stay
    data/cache/chartevents_filtered.parquet long-format vitals + GCS
    data/cache/labevents_filtered.parquet   long-format labs
    data/cache/interventions.parquet        one row per (stay, kind, starttime)
    data/cache/poe_lab_orders.parquet       new lab orders entered during ICU stay
    data/splits.json                        subject ids per split
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
import itemids as ids

# Populated by _iter_chunks when a source file turns out to be corrupt.
_SCAN_STATUS = {}


# ------------------------------------------------------------------ cohort ----
def build_cohort(limit_icustays=None):
    """Adult stays with 1 <= LOS <= 20 days. Completeness is applied later."""
    stays = pd.read_csv(
        cfg.ICUSTAYS_CSV,
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"],
        parse_dates=["intime", "outtime"],
    )
    pats = pd.read_csv(cfg.PATIENTS_CSV, usecols=["subject_id", "anchor_age", "gender"])
    stays = stays.merge(pats, on="subject_id", how="left")

    n0 = len(stays)
    stays = stays[stays["anchor_age"] >= cfg.MIN_AGE]
    stays = stays[(stays["los"] >= cfg.MIN_LOS_DAYS) & (stays["los"] <= cfg.MAX_LOS_DAYS)]
    stays = stays.dropna(subset=["intime", "outtime"])
    print(f"  icustays {n0} -> {len(stays)} after age>={cfg.MIN_AGE} and "
          f"{cfg.MIN_LOS_DAYS}<=los<={cfg.MAX_LOS_DAYS} days")

    stays = stays.sort_values(["subject_id", "intime"]).reset_index(drop=True)
    if limit_icustays:
        # Take the lowest subject ids so the event scans can stop early: both
        # chartevents and labevents are ordered by subject_id.
        stays = stays.head(limit_icustays).copy()
        print(f"  --limit-icustays {limit_icustays}: keeping "
              f"{stays['subject_id'].nunique()} subjects, {len(stays)} stays")
    return stays


# ------------------------------------------------------------- event scans ----
class _ShardWriter:
    """Append row groups to one parquet file without holding it all in memory."""

    def __init__(self, path, schema):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._schema = schema
        self._writer = None
        self.rows = 0

    def write(self, df):
        if df.empty:
            return
        table = pa.Table.from_pandas(df, schema=self._schema, preserve_index=False)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, self._schema, compression="snappy")
        self._writer.write_table(table)
        self.rows += len(df)

    def close(self):
        if self._writer is not None:
            self._writer.close()


EVENT_SCHEMA = pa.schema([
    ("stay_id", pa.int32()),
    ("subject_id", pa.int32()),
    ("trait", pa.string()),
    ("charttime", pa.timestamp("s")),
    ("value", pa.float32()),
])

POE_SCHEMA = pa.schema([
    ("stay_id", pa.int32()),
    ("subject_id", pa.int32()),
    ("poe_id", pa.string()),
    ("ordertime", pa.timestamp("s")),
])


def _iter_chunks(reader, path, tolerate_truncation):
    """Yield CSV chunks, optionally surviving a corrupt gzip stream.

    A partially-corrupted download raises zlib.error part-way through. With
    `--tolerate-truncation` the scan keeps whatever it decoded and stops there
    instead of losing the whole pass. Because MIMIC-IV event tables are ordered
    by subject_id, the salvaged part is a PREFIX OF THE SUBJECT ID RANGE, not a
    random sample of rows: every retained stay keeps all of its events, and the
    stays that are lost are lost entirely. subject_id is a de-identified surrogate
    key with no clinical meaning, so the retained cohort is not biased on
    severity, but it IS smaller, and the truncation is recorded in
    data/cache/scan_status.json so downstream reports can say so.
    """
    i = 0
    while True:
        try:
            chunk = next(reader)
        except StopIteration:
            return
        except (zlib.error, EOFError, OSError) as e:
            msg = f"{Path(path).name}: decode failed at chunk {i}: {type(e).__name__}: {e}"
            if not tolerate_truncation:
                raise SystemExit(
                    f"\n{msg}\n\nThe file is corrupt. Re-download it, or re-run "
                    f"with --tolerate-truncation to build a smaller cohort from "
                    f"the readable prefix.")
            print(f"    TRUNCATED  {msg}")
            _SCAN_STATUS[Path(path).name] = {"truncated": True, "chunks_read": i}
            return
        yield i, chunk
        i += 1


def _clip_to_range(df, trait_meta):
    """Drop physiologically impossible values, per-trait."""
    keep = np.ones(len(df), dtype=bool)
    for trait, (lo, hi) in trait_meta.items():
        m = df["trait"].values == trait
        keep &= ~m | ((df["value"].values >= lo) & (df["value"].values <= hi))
    return df[keep]


def scan_chartevents(stays, tolerate_truncation=False):
    """Vitals and GCS from chartevents, filtered to the cohort's stay_ids."""
    keep_stays = pd.Index(stays["stay_id"].astype("int64"))
    max_subject = int(stays["subject_id"].max())
    id2trait = ids.chart_itemid_to_trait()
    wanted = set(ids.chart_itemids())
    gcs_rev = {v: k for k, v in ids.GCS_ITEMS.items()}
    gcs_maps = {
        "gcs_eye": ids.GCS_EYE_MAP,
        "gcs_verbal": ids.GCS_VERBAL_MAP,
        "gcs_motor": ids.GCS_MOTOR_MAP,
    }
    ranges = {ids.CHART_TRAIT_ALIAS.get(n, n): r for n, (_, _, r) in ids.CHART_TRAITS.items()}

    writer = _ShardWriter(cfg.CHART_PARQUET, EVENT_SCHEMA)
    reader = pd.read_csv(
        cfg.CHARTEVENTS_CSV,
        usecols=["subject_id", "stay_id", "charttime", "itemid", "value", "valuenum"],
        chunksize=cfg.SCAN_CHUNK_ROWS,
    )
    for i, chunk in _iter_chunks(reader, cfg.CHARTEVENTS_CSV, tolerate_truncation):
        if chunk["subject_id"].min() > max_subject:
            print(f"    chunk {i}: past subject {max_subject}, stopping early")
            break
        chunk = chunk[chunk["itemid"].isin(wanted)]
        chunk = chunk[chunk["stay_id"].isin(keep_stays)]
        if chunk.empty:
            continue

        is_gcs = chunk["itemid"].isin(gcs_rev)
        parts = []

        num = chunk[~is_gcs].dropna(subset=["valuenum"])
        if not num.empty:
            trait, conv = zip(*[(id2trait[i][0], id2trait[i][1]) for i in num["itemid"]])
            vals = num["valuenum"].to_numpy(dtype="float64")
            out = np.empty_like(vals)
            conv = np.array(conv, dtype=object)
            for kind in set(conv):
                m = conv == kind
                out[m] = ids.apply_conversion(vals[m], kind)
            parts.append(pd.DataFrame({
                "stay_id": num["stay_id"].to_numpy("int32"),
                "subject_id": num["subject_id"].to_numpy("int32"),
                "trait": list(trait),
                "charttime": num["charttime"].to_numpy(),
                "value": out,
            }))

        gcs = chunk[is_gcs].dropna(subset=["value"])
        if not gcs.empty:
            names = gcs["itemid"].map(gcs_rev)
            text = gcs["value"].astype(str).str.strip().str.lower()
            scores = np.array([gcs_maps[n].get(t, np.nan) for n, t in zip(names, text)])
            ok = ~np.isnan(scores)
            parts.append(pd.DataFrame({
                "stay_id": gcs["stay_id"].to_numpy("int32")[ok],
                "subject_id": gcs["subject_id"].to_numpy("int32")[ok],
                "trait": names.to_numpy()[ok],
                "charttime": gcs["charttime"].to_numpy()[ok],
                "value": scores[ok],
            }))

        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
        df = df.dropna(subset=["charttime"])
        df = _clip_to_range(df, ranges)
        df["value"] = df["value"].astype("float32")
        writer.write(df)
        print(f"    chunk {i}: kept {len(df):,} rows (total {writer.rows:,})", flush=True)

    writer.close()
    print(f"  wrote {writer.rows:,} chartevents rows -> {cfg.CHART_PARQUET}")
    return writer.rows


def scan_labevents(stays, tolerate_truncation=False):
    """Labs from labevents, attached to a stay by hadm_id and the ICU time window."""
    max_subject = int(stays["subject_id"].max())
    id2trait = ids.lab_itemid_to_trait()
    wanted = set(ids.lab_itemids())
    ranges = {n: r for n, (_, _, r) in ids.LAB_TRAITS.items()}

    # hadm_id -> the stays inside it, for the interval join below.
    windows = (stays[["hadm_id", "stay_id", "subject_id", "intime", "outtime"]]
               .dropna(subset=["hadm_id"]).copy())
    windows["hadm_id"] = windows["hadm_id"].astype("int64")

    writer = _ShardWriter(cfg.LAB_PARQUET, EVENT_SCHEMA)
    reader = pd.read_csv(
        cfg.LABEVENTS_CSV,
        usecols=["subject_id", "hadm_id", "charttime", "itemid", "valuenum"],
        chunksize=cfg.SCAN_CHUNK_ROWS,
    )
    for i, chunk in _iter_chunks(reader, cfg.LABEVENTS_CSV, tolerate_truncation):
        if chunk["subject_id"].min() > max_subject:
            print(f"    chunk {i}: past subject {max_subject}, stopping early")
            break
        chunk = chunk[chunk["itemid"].isin(wanted)].dropna(subset=["valuenum", "hadm_id"])
        if chunk.empty:
            continue
        chunk = chunk.copy()
        chunk["hadm_id"] = chunk["hadm_id"].astype("int64")
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk = chunk.dropna(subset=["charttime"])

        # One admission can hold several ICU stays, so this merge can fan out;
        # the time-window filter immediately collapses it back to one row.
        m = chunk.merge(windows, on="hadm_id", how="inner", suffixes=("", "_stay"))
        m = m[(m["charttime"] >= m["intime"]) & (m["charttime"] <= m["outtime"])]
        if m.empty:
            continue

        trait, conv = zip(*[(id2trait[j][0], id2trait[j][1]) for j in m["itemid"]])
        vals = m["valuenum"].to_numpy(dtype="float64")
        out = np.empty_like(vals)
        conv = np.array(conv, dtype=object)
        for kind in set(conv):
            sel = conv == kind
            out[sel] = ids.apply_conversion(vals[sel], kind)

        df = pd.DataFrame({
            "stay_id": m["stay_id"].to_numpy("int32"),
            "subject_id": m["subject_id_stay"].to_numpy("int32"),
            "trait": list(trait),
            "charttime": m["charttime"].to_numpy(),
            "value": out,
        })
        df = _clip_to_range(df, ranges)
        df["value"] = df["value"].astype("float32")
        writer.write(df)
        print(f"    chunk {i}: kept {len(df):,} rows (total {writer.rows:,})", flush=True)

    writer.close()
    print(f"  wrote {writer.rows:,} labevents rows -> {cfg.LAB_PARQUET}")
    return writer.rows


def scan_poe_lab_orders(stays, tolerate_truncation=False):
    """Cache new POE lab orders entered while the patient was in the ICU.

    MIMIC-IV provides no shared key between ``poe.poe_id`` and
    ``labevents.specimen_id``. The rows are therefore retained as an independent
    workflow stream; later stages aggregate simultaneous rows into order groups
    but never claim that a particular order caused a particular specimen.
    """
    max_subject = int(stays["subject_id"].max())
    windows = (stays[["hadm_id", "stay_id", "subject_id", "intime", "outtime"]]
               .dropna(subset=["hadm_id"]).copy())
    windows["hadm_id"] = windows["hadm_id"].astype("int64")

    writer = _ShardWriter(cfg.POE_PARQUET, POE_SCHEMA)
    reader = pd.read_csv(
        cfg.POE_CSV,
        usecols=["poe_id", "subject_id", "hadm_id", "ordertime",
                 "order_type", "transaction_type"],
        chunksize=cfg.SCAN_CHUNK_ROWS,
    )
    for i, chunk in _iter_chunks(reader, cfg.POE_CSV, tolerate_truncation):
        if chunk["subject_id"].min() > max_subject:
            print(f"    chunk {i}: past subject {max_subject}, stopping early")
            break
        chunk = chunk[
            chunk["order_type"].eq("Lab")
            & chunk["transaction_type"].eq("New")
        ].dropna(subset=["hadm_id", "ordertime"])
        if chunk.empty:
            continue
        chunk = chunk.copy()
        chunk["hadm_id"] = chunk["hadm_id"].astype("int64")
        chunk["ordertime"] = pd.to_datetime(chunk["ordertime"], errors="coerce")
        chunk = chunk.dropna(subset=["ordertime"])

        m = chunk.merge(windows, on="hadm_id", how="inner", suffixes=("", "_stay"))
        m = m[(m["ordertime"] >= m["intime"]) & (m["ordertime"] <= m["outtime"])]
        if m.empty:
            continue
        out = pd.DataFrame({
            "stay_id": m["stay_id"].to_numpy("int32"),
            "subject_id": m["subject_id_stay"].to_numpy("int32"),
            "poe_id": m["poe_id"].astype(str).to_numpy(),
            "ordertime": m["ordertime"].to_numpy(),
        })
        writer.write(out)
        print(f"    chunk {i}: kept {len(out):,} rows (total {writer.rows:,})",
              flush=True)

    writer.close()
    if writer.rows == 0:
        pq.write_table(pa.Table.from_pylist([], schema=POE_SCHEMA), cfg.POE_PARQUET)
    print(f"  wrote {writer.rows:,} POE lab-order rows -> {cfg.POE_PARQUET}")
    return writer.rows


# ---------------------------------------------------------- interventions ----
def extract_interventions(stays):
    """Intervals for the four intervention classes in Eq. 4.

    `endtime` and `rate` are carried because stage 2 needs more than an onset:
    SOFA's cardiovascular component grades on which vasopressor is running at
    hour t and at what dose, not merely on whether one was ever started.
    """
    keep_stays = pd.Index(stays["stay_id"].astype("int64"))
    frames = []

    inp = pd.read_csv(cfg.INPUTEVENTS_CSV,
                      usecols=["stay_id", "starttime", "endtime", "itemid",
                               "rate", "rateuom"],
                      parse_dates=["starttime", "endtime"])
    inp = inp[inp["itemid"].isin(ids.VASOPRESSOR_ITEMS) & inp["stay_id"].isin(keep_stays)]
    # Grade only when the unit is the weight-normalized one SOFA is defined on.
    rate = inp["rate"].where(
        inp["rateuom"].astype(str).str.lower().str.contains("kg/min", na=False))
    frames.append(pd.DataFrame({"stay_id": inp["stay_id"], "kind": "vasopressor",
                                "starttime": inp["starttime"], "endtime": inp["endtime"],
                                "itemid": inp["itemid"], "rate": rate}))

    proc = pd.read_csv(cfg.PROCEDUREEVENTS_CSV,
                       usecols=["stay_id", "starttime", "endtime", "itemid"],
                       parse_dates=["starttime", "endtime"])
    proc = proc[proc["stay_id"].isin(keep_stays)]
    for kind, items in [("ventilation", ids.VENTILATION_ITEMS),
                        ("dialysis", ids.DIALYSIS_ITEMS)]:
        sub = proc[proc["itemid"].isin(items)]
        frames.append(pd.DataFrame({"stay_id": sub["stay_id"], "kind": kind,
                                    "starttime": sub["starttime"],
                                    "endtime": sub["endtime"],
                                    "itemid": sub["itemid"], "rate": np.nan}))

    # Antibiotics are ordered per admission, so map them onto stays by time window.
    windows = (stays[["hadm_id", "stay_id", "intime", "outtime"]]
               .dropna(subset=["hadm_id"]).copy())
    windows["hadm_id"] = windows["hadm_id"].astype("int64")
    pat = "|".join(ids.ANTIBIOTIC_DRUGS)
    abx_parts = []
    for chunk in pd.read_csv(cfg.PRESCRIPTIONS_CSV,
                             usecols=["hadm_id", "starttime", "stoptime", "drug"],
                             chunksize=cfg.SCAN_CHUNK_ROWS):
        chunk = chunk.dropna(subset=["hadm_id", "starttime", "drug"])
        chunk = chunk[chunk["drug"].str.contains(pat, case=False, na=False, regex=True)]
        if chunk.empty:
            continue
        chunk = chunk.copy()
        chunk["hadm_id"] = chunk["hadm_id"].astype("int64")
        chunk["starttime"] = pd.to_datetime(chunk["starttime"], errors="coerce")
        chunk["stoptime"] = pd.to_datetime(chunk["stoptime"], errors="coerce")
        m = chunk.merge(windows, on="hadm_id", how="inner")
        m = m[(m["starttime"] >= m["intime"]) & (m["starttime"] <= m["outtime"])]
        if not m.empty:
            abx_parts.append(pd.DataFrame({"stay_id": m["stay_id"], "kind": "antibiotic",
                                           "starttime": m["starttime"],
                                           "endtime": m["stoptime"],
                                           "itemid": -1, "rate": np.nan}))
    if abx_parts:
        frames.append(pd.concat(abx_parts, ignore_index=True))

    out = pd.concat(frames, ignore_index=True).dropna(subset=["starttime"])
    out["stay_id"] = out["stay_id"].astype("int32")
    out["itemid"] = out["itemid"].fillna(-1).astype("int32")
    out = out.drop_duplicates().sort_values(["stay_id", "starttime"]).reset_index(drop=True)
    out.to_parquet(cfg.INTERVENTIONS_PARQUET, index=False)
    print("  interventions per kind:\n" +
          out["kind"].value_counts().to_string().replace("\n", "\n    "))
    return out


# ------------------------------------------------------ completeness + splits ----
def apply_completeness_filter(stays):
    """Keep stays with at least one recorded value of every trait in TRAITS."""
    if not cfg.REQUIRE_ALL_TRAITS:
        return stays
    # Read row group by row group and reduce to a (stay_id, trait) set as we go.
    # Materializing tens of millions of rows with an object-dtype string column
    # just to call drop_duplicates would cost several GB for a boolean answer.
    pairs = set()
    found_any = False
    for path in (cfg.CHART_PARQUET, cfg.LAB_PARQUET):
        if not Path(path).exists():
            continue
        found_any = True
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=2_000_000, columns=["stay_id", "trait"]):
            sid = batch.column("stay_id").to_numpy(zero_copy_only=False)
            tr = batch.column("trait").to_pylist()
            pairs.update(zip(sid.tolist(), tr))
    if not found_any:
        raise SystemExit("no cached events; run the scans first")

    counts = {}
    for sid, _ in pairs:
        counts[sid] = counts.get(sid, 0) + 1
    complete = {int(s) for s, c in counts.items() if c >= len(ids.TRAITS)}

    n0 = len(stays)
    stays = stays[stays["stay_id"].isin(complete)].copy()
    print(f"  completeness filter ({len(ids.TRAITS)} traits): {n0} -> {len(stays)} stays, "
          f"{stays['subject_id'].nunique()} subjects")
    if stays.empty:
        raise SystemExit(
            "no stay has all 20 traits. Loosen cfg.REQUIRE_ALL_TRAITS or check itemids.")
    return stays


def make_splits(stays, seed=cfg.SEED):
    """Subject-level train/val/test. A subject never appears in two splits."""
    subs = np.array(sorted(stays["subject_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(subs)
    n = len(subs)
    n_tr = int(round(cfg.TRAIN_FRAC * n))
    n_va = int(round(cfg.VAL_FRAC * n))
    splits = {
        "train": subs[:n_tr].tolist(),
        "val": subs[n_tr:n_tr + n_va].tolist(),
        "test": subs[n_tr + n_va:].tolist(),
    }
    assert not (set(splits["train"]) & set(splits["val"]))
    assert not (set(splits["train"]) & set(splits["test"]))
    assert not (set(splits["val"]) & set(splits["test"]))

    cfg.SPLITS_JSON.parent.mkdir(parents=True, exist_ok=True)
    cfg.SPLITS_JSON.write_text(json.dumps(splits), encoding="utf-8")

    lookup = {s: k for k, v in splits.items() for s in v}
    stays["split"] = stays["subject_id"].map(lookup)
    for k in ("train", "val", "test"):
        sel = stays[stays["split"] == k]
        print(f"  {k:5s}: {len(splits[k]):6,} subjects  {len(sel):6,} stays")
    return stays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-icustays", type=int, default=None,
                    help="smoke test: keep only the first N stays by subject id")
    ap.add_argument("--skip-scan", action="store_true",
                    help="reuse existing parquet caches, redo cohort and splits only")
    ap.add_argument("--skip-chart", action="store_true",
                    help="reuse the chartevents cache, redo labs and interventions")
    ap.add_argument("--tolerate-truncation", action="store_true",
                    help="on a corrupt source file, keep the readable prefix "
                         "instead of aborting (records it in scan_status.json)")
    args = ap.parse_args()

    cfg.ensure_dirs()
    cfg.check_inputs()
    print(f"reading MIMIC-IV from {cfg.MIMIC4_DIR}")

    print("\n[1/6] cohort")
    stays = build_cohort(args.limit_icustays)

    if args.skip_scan:
        print("\n[2-5/6] --skip-scan: reusing cached parquet")
        if not cfg.POE_PARQUET.exists():
            print("  POE cache missing; scanning poe.csv.gz only")
            scan_poe_lab_orders(stays, args.tolerate_truncation)
    else:
        print("\n[2/5] chartevents scan")
        if args.skip_chart and cfg.CHART_PARQUET.exists():
            print("  --skip-chart: reusing cached chartevents parquet")
        else:
            scan_chartevents(stays, args.tolerate_truncation)
        print("\n[3/5] labevents scan")
        scan_labevents(stays, args.tolerate_truncation)
        print("\n[4/5] interventions")
        extract_interventions(stays)
        print("\n[5/6] POE lab orders")
        scan_poe_lab_orders(stays, args.tolerate_truncation)

    if _SCAN_STATUS:
        (cfg.RAW_CACHE_DIR / "scan_status.json").write_text(
            json.dumps(_SCAN_STATUS, indent=2), encoding="utf-8")
        print("\n  WARNING: one or more source files were truncated:")
        for name, st in _SCAN_STATUS.items():
            print(f"    {name}: stopped after {st['chunks_read']} chunks")
        print("  The cohort below covers a PREFIX of the subject id range only.")

    print("\n[6/6] completeness filter and subject splits")
    stays = apply_completeness_filter(stays)
    stays = make_splits(stays)
    stays.to_parquet(cfg.COHORT_PARQUET, index=False)
    print(f"\nwrote cohort -> {cfg.COHORT_PARQUET}")


if __name__ == "__main__":
    main()
