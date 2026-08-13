"""Validated multivariate correction for target-lab forecasts.

The state-space forecaster supplies a dense, past-only representation of all
physiology. This model predicts each target lab's change from its last observed
value using that representation. Training-hour predictions are subject-level
out-of-fold; validation selects model complexity and calibrates 90% intervals;
the locked test split is transformed only by models fit on all training data.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import config as cfg
import itemids as ids


FEATURE_COLUMNS = (
    [f"mean_{x}" for x in ids.FORECAST_TRAITS]
    + [f"std_{x}" for x in ids.FORECAST_TRAITS]
    + [f"last_{x}" for x in ids.TARGET_LABS]
    + [f"delta_{x}" for x in ids.TARGET_LABS]
    + [f"trend_{x}" for x in ids.TARGET_LABS]
    + [f"log_delta_{x}" for x in ids.TARGET_LABS]
    + ["hour", "log_hour", "gcs_total", "gcs_missing",
       "ventilated", "vaso_class", "vaso_rate"]
    + [f"active_{x}" for x in ids.INTERVENTION_KINDS]
    + ["poe_order_groups_6h", "poe_order_rows_6h",
       "poe_order_groups_24h", "poe_hours_since_lab_order"]
)


def _features(df):
    d = df.copy()
    d["log_hour"] = np.log1p(d["hour"].clip(lower=0))
    for lab in ids.TARGET_LABS:
        d[f"last_{lab}"] = d[f"last_{lab}"].fillna(d[f"mean_{lab}"])
        d[f"delta_{lab}"] = d[f"delta_{lab}"].fillna(d["hour"] + 1.0)
        d[f"trend_{lab}"] = d[f"mean_{lab}"] - d[f"last_{lab}"]
        d[f"log_delta_{lab}"] = np.log1p(d[f"delta_{lab}"].clip(lower=0))
    d["gcs_missing"] = d["gcs_total"].isna().astype(np.float32)
    d["gcs_total"] = d["gcs_total"].fillna(15.0)
    d["vaso_rate"] = d["vaso_rate"].fillna(0.0)
    X = d[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    if not np.isfinite(X).all():
        bad = np.array(FEATURE_COLUMNS)[~np.isfinite(X).all(axis=0)]
        raise ValueError(f"non-finite residual-forecast features: {bad.tolist()}")
    return X


def _eligible(df, lab):
    return (df[f"obs_{lab}"].notna() & df[f"last_{lab}"].notna()).to_numpy()


def _sample(idx, max_rows, seed):
    if len(idx) <= max_rows:
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, max_rows, replace=False))


def _select_shrinkage(actual, last, predicted_delta, candidates=None):
    candidates = cfg.FORECAST_DELTA_SHRINKAGE if candidates is None else candidates
    scores = []
    for shrinkage in candidates:
        final = last + float(shrinkage) * predicted_delta
        rmse = float(np.sqrt(np.mean((actual - final) ** 2)))
        scores.append((rmse, float(shrinkage)))
    return min(scores)


def _calibrate_interval(actual, final_mean, raw_std, target=0.90):
    raw_std = np.maximum(np.asarray(raw_std, dtype=float), cfg.FORECAST_MIN_STD)
    ratio = np.abs(np.asarray(actual) - np.asarray(final_mean)) / raw_std
    scale = float(np.quantile(ratio, target) / 1.6448536269514722)
    return float(np.clip(scale, 0.5, 5.0))


def _model(spec, loss="squared_error", quantile=None, seed=cfg.SEED):
    kw = dict(
        loss=loss, max_iter=spec.get("max_iter", 250),
        learning_rate=spec.get("learning_rate", 0.05),
        max_leaf_nodes=spec["max_leaf_nodes"],
        min_samples_leaf=spec["min_samples_leaf"],
        l2_regularization=spec["l2_regularization"], random_state=seed,
    )
    if quantile is not None:
        kw["quantile"] = quantile
    return HistGradientBoostingRegressor(**kw)


class TargetLabForecaster:
    def __init__(self, seed=cfg.SEED):
        self.seed = seed
        self.specs_ = {}
        self.models_ = {}
        self.lower_ = {}
        self.upper_ = {}
        self.interval_scale_ = {}
        self.shrinkage_ = {}
        self.bounds_ = {}
        self.oof_mean_ = {}
        self.oof_std_ = {}
        self.selection_ = {}

    def fit(self, train, val):
        Xtr, Xva = _features(train), _features(val)
        subjects = train["subject_id"].to_numpy()
        unique_subjects = np.unique(subjects)
        fold_of_subject = {
            s: i % cfg.FORECAST_RESIDUAL_FOLDS
            for i, s in enumerate(np.random.default_rng(self.seed)
                                  .permutation(unique_subjects))
        }
        folds = np.array([fold_of_subject[s] for s in subjects], dtype=np.int8)

        for lab_idx, lab in enumerate(ids.TARGET_LABS):
            tr_keep = _eligible(train, lab)
            va_keep = _eligible(val, lab)
            ytr = (train[f"obs_{lab}"] - train[f"last_{lab}"]).to_numpy(float)
            yva = (val[f"obs_{lab}"] - val[f"last_{lab}"]).to_numpy(float)
            last_va = val[f"last_{lab}"].to_numpy(float)[va_keep]
            actual_va = val[f"obs_{lab}"].to_numpy(float)[va_keep]
            train_idx = np.flatnonzero(tr_keep)
            fit_idx = _sample(
                train_idx, cfg.FORECAST_RESIDUAL_MAX_TRAIN_ROWS,
                self.seed + lab_idx)

            best = None
            candidates = []
            for spec in cfg.FORECAST_RESIDUAL_CANDIDATES:
                mdl = _model(spec, seed=self.seed + lab_idx)
                mdl.fit(Xtr[fit_idx], ytr[fit_idx])
                pred = mdl.predict(Xva[va_keep])
                rmse, shrinkage = _select_shrinkage(actual_va, last_va, pred)
                candidates.append({"name": spec["name"], "validation_rmse": rmse,
                                   "shrinkage": shrinkage})
                if best is None or rmse < best[0]:
                    best = (rmse, copy.deepcopy(spec), mdl, shrinkage)

            _, spec, mean_model, shrinkage = best
            lo_model = _model(spec, loss="quantile", quantile=0.05,
                              seed=self.seed + 100 + lab_idx)
            hi_model = _model(spec, loss="quantile", quantile=0.95,
                              seed=self.seed + 200 + lab_idx)
            lo_model.fit(Xtr[fit_idx], ytr[fit_idx])
            hi_model.fit(Xtr[fit_idx], ytr[fit_idx])

            mean_delta = shrinkage * mean_model.predict(Xva[va_keep])
            lo = lo_model.predict(Xva[va_keep])
            hi = hi_model.predict(Xva[va_keep])
            raw_std = np.maximum(
                (np.maximum(lo, hi) - np.minimum(lo, hi)) / (2 * 1.6448536269514722),
                cfg.FORECAST_MIN_STD)
            observed = train[f"obs_{lab}"].dropna().to_numpy(float)
            bounds = (float(np.quantile(observed, 0.001)),
                      float(np.quantile(observed, 0.999)))
            final_mean = np.clip(last_va + mean_delta, *bounds)
            interval_scale = _calibrate_interval(actual_va, final_mean, raw_std)

            self.specs_[lab] = spec
            self.models_[lab] = mean_model
            self.lower_[lab] = lo_model
            self.upper_[lab] = hi_model
            self.interval_scale_[lab] = interval_scale
            self.shrinkage_[lab] = shrinkage
            self.bounds_[lab] = bounds
            self.selection_[lab] = {
                "selected": spec["name"],
                "candidates": candidates,
                "interval_scale": interval_scale,
                "shrinkage": shrinkage,
                "train_labels": int(len(train_idx)),
                "validation_labels": int(va_keep.sum()),
            }

            # Subject-level cross-fitting prevents the RL training state from
            # containing predictions produced by a model fit on that patient.
            oof_mean = np.empty(len(train), dtype=np.float32)
            oof_std = np.empty(len(train), dtype=np.float32)
            for fold in range(cfg.FORECAST_RESIDUAL_FOLDS):
                fit_mask = tr_keep & (folds != fold)
                fold_idx = _sample(
                    np.flatnonzero(fit_mask), cfg.FORECAST_RESIDUAL_MAX_TRAIN_ROWS,
                    self.seed + 1000 + 10 * lab_idx + fold)
                pred_mask = folds == fold
                m = _model(spec, seed=self.seed + 1000 + fold)
                qlo = _model(spec, loss="quantile", quantile=0.05,
                             seed=self.seed + 2000 + fold)
                qhi = _model(spec, loss="quantile", quantile=0.95,
                             seed=self.seed + 3000 + fold)
                m.fit(Xtr[fold_idx], ytr[fold_idx])
                qlo.fit(Xtr[fold_idx], ytr[fold_idx])
                qhi.fit(Xtr[fold_idx], ytr[fold_idx])
                delta = shrinkage * m.predict(Xtr[pred_mask])
                low = qlo.predict(Xtr[pred_mask])
                high = qhi.predict(Xtr[pred_mask])
                last = train.loc[pred_mask, f"last_{lab}"].fillna(
                    train.loc[pred_mask, f"mean_{lab}"]).to_numpy(float)
                oof_mean[pred_mask] = np.clip(
                    last + delta, *self.bounds_[lab]).astype(np.float32)
                oof_std[pred_mask] = (
                    np.maximum((np.maximum(low, high) - np.minimum(low, high))
                               / (2 * 1.6448536269514722),
                               cfg.FORECAST_MIN_STD)
                    * interval_scale).astype(np.float32)
            self.oof_mean_[lab] = oof_mean
            self.oof_std_[lab] = oof_std
        return self

    def transform(self, df, training=False):
        out = df.copy()
        X = None if training else _features(out)
        for lab in ids.TARGET_LABS:
            if training:
                mean, std = self.oof_mean_[lab], self.oof_std_[lab]
                if len(mean) != len(out):
                    raise ValueError("OOF predictions do not match training frame")
            else:
                delta = self.shrinkage_[lab] * self.models_[lab].predict(X)
                low = self.lower_[lab].predict(X)
                high = self.upper_[lab].predict(X)
                last = out[f"last_{lab}"].fillna(out[f"mean_{lab}"]).to_numpy(float)
                mean = np.clip(last + delta, *self.bounds_[lab])
                std = (np.maximum(
                    (np.maximum(low, high) - np.minimum(low, high))
                    / (2 * 1.6448536269514722), cfg.FORECAST_MIN_STD)
                    * self.interval_scale_[lab])
            out[f"mean_{lab}"] = np.asarray(mean, dtype=np.float32)
            out[f"std_{lab}"] = np.maximum(
                np.asarray(std, dtype=np.float32), cfg.FORECAST_MIN_STD)
        return out

    def state_dict(self):
        return {
            "seed": self.seed, "specs": self.specs_, "models": self.models_,
            "lower": self.lower_, "upper": self.upper_,
            "interval_scale": self.interval_scale_, "shrinkage": self.shrinkage_,
            "bounds": self.bounds_,
            "selection": self.selection_,
        }

    @classmethod
    def from_state_dict(cls, state):
        obj = cls(state["seed"])
        obj.specs_, obj.models_ = state["specs"], state["models"]
        obj.lower_, obj.upper_ = state["lower"], state["upper"]
        obj.interval_scale_, obj.bounds_ = state["interval_scale"], state["bounds"]
        obj.shrinkage_ = state.get("shrinkage", {lab: 1.0 for lab in obj.models_})
        obj.selection_ = state["selection"]
        return obj
