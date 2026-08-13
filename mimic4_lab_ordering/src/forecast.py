"""
Hourly probabilistic forecasting of sparse, irregularly sampled clinical traits.

The paper (Sec. 2.1) uses a sparse multi-output Gaussian process to resample the
raw traces onto a one-hour grid. That MOGP feeds exactly two things downstream:

  1. (m_t, sigma_t) in the state vector                                (Sec. 2.2)
  2. the |m_t - y_t| / sigma_t term in the information reward          (Eq. 5)

So the contract a forecaster must satisfy is narrow: given a stay's observations
on an hourly grid, emit a predictive mean and standard deviation at every hour.
`Forecaster` is that contract; anything satisfying it can be dropped in.

Two methods, and the distinction between them is a correctness boundary:

    filter()  uses observations strictly BEFORE hour t. This is the only thing
              the policy is ever allowed to see. In particular the result of the
              lab being decided at hour t is not in it, which is what makes
              |m_t - y_t| a meaningful information-gain signal rather than zero.

    smooth()  uses ALL observations, including future ones. This is the paper's
              "approximated true value ... which we impute using the MOGP model
              given all the observed values" (Sec. 3.1). It is EVALUATION ONLY
              and must never reach the state vector.

Why a LOCAL LINEAR TREND model and not a simpler one
----------------------------------------------------
The obvious substitution for the MOGP is a local level (random walk) model. It
does not work here, and the reason is worth stating because it is a property of
the reward rather than of the data. A random walk's optimal forecast is flat at
the current level, so m_t collapses onto y_t, |m_t - y_t| goes to zero, and Eq. 5
returns zero at every hour for every lab. The information reward would be dead.
The paper is explicit that this term is meant to fire when "a lab is predicted to
be informative (in that the forecasted value is significantly different from the
last known measurement) due to a sudden change in disease state" - which requires
a forecaster that can extrapolate a trend.

So the shipped model carries a level and a slope:

    level_t = level_{t-1} + slope_{t-1} + w1,   w1 ~ N(0, q_level)
    slope_t =               slope_{t-1} + w2,   w2 ~ N(0, q_slope)
    y_t     = level_t                   + v,    v  ~ N(0, r)

Fit by maximum likelihood on the training split, run as a Kalman filter and an
RTS smoother. Closed form, exact, vectorized across stays, no scaling problem.
This remains a documented substitution for the MOGP, not a reimplementation: it
drops the cross-trait covariance that the "multi-output" in MOGP refers to.

`MOGPForecaster` is the slot for the closer replication, behind the same
interface, selected by `config.FORECASTER`.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

import config as cfg


class Forecaster:
    """Interface. Arrays are [n_stays, T, K] with NaN marking 'not observed'."""

    trait_names: list[str]

    def fit(self, obs: np.ndarray) -> "Forecaster":
        raise NotImplementedError

    def filter(self, obs: np.ndarray):
        """Past-only predictive (mean, std). Safe for the policy state."""
        raise NotImplementedError

    def smooth(self, obs: np.ndarray):
        """All-observation posterior (mean, std). EVALUATION ONLY."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
#  Local linear trend: x = [level, slope], F = [[1,1],[0,1]], H = [1, 0]       #
# --------------------------------------------------------------------------- #
def _predict(x, P, q_level, q_slope):
    """x <- F x,  P <- F P F' + Q, for [n,2] and [n,2,2]."""
    xp = np.stack([x[:, 0] + x[:, 1], x[:, 1]], axis=1)

    p00, p01 = P[:, 0, 0], P[:, 0, 1]
    p10, p11 = P[:, 1, 0], P[:, 1, 1]
    Pp = np.empty_like(P)
    Pp[:, 0, 0] = p00 + p01 + p10 + p11 + q_level
    Pp[:, 0, 1] = p01 + p11
    Pp[:, 1, 0] = p10 + p11
    Pp[:, 1, 1] = p11 + q_slope
    return xp, Pp


def _update(xp, Pp, y, r):
    """Fold in an observation of the level where `y` is not NaN."""
    seen = ~np.isnan(y)
    s = Pp[:, 0, 0] + r                      # innovation variance
    k = Pp[:, :, 0] / s[:, None]             # Kalman gain, [n,2]
    resid = np.where(seen, y - xp[:, 0], 0.0)
    gate = seen[:, None].astype(np.float64)

    x = xp + gate * k * resid[:, None]
    # P <- P - K H P, applied only where an observation landed
    KHP = k[:, :, None] * Pp[:, 0, :][:, None, :]
    P = Pp - gate[:, :, None] * KHP
    return x, P


def _kalman_pass(y, q_level, q_slope, r, p0_level=1.0, p0_slope=0.1):
    """Forward pass over [n, T]. Stores everything the smoother needs.

    At each hour t the ONE-STEP-AHEAD predictive distribution is recorded: the
    posterior through t-1 propagated forward one hour, before hour t's own
    observation is folded in. That is what makes filter() past-only.
    """
    n, T = y.shape
    x_pred = np.zeros((n, T, 2))
    P_pred = np.zeros((n, T, 2, 2))
    x_filt = np.zeros((n, T, 2))
    P_filt = np.zeros((n, T, 2, 2))

    x = np.zeros((n, 2))
    P = np.zeros((n, 2, 2))
    P[:, 0, 0] = p0_level
    P[:, 1, 1] = p0_slope

    for t in range(T):
        xp, Pp = _predict(x, P, q_level, q_slope)
        x_pred[:, t] = xp
        P_pred[:, t] = Pp
        x, P = _update(xp, Pp, y[:, t], r)
        x_filt[:, t] = x
        P_filt[:, t] = P
    return x_pred, P_pred, x_filt, P_filt


def _inv2(M):
    """Batched inverse of [n,2,2] with a SCALE-AWARE ridge.

    A fixed absolute floor on the determinant is not enough here. Across a long
    stretch with no observations the predictive covariance of a local trend model
    grows without bound (the level's variance goes as t^3), so `det` grows huge
    while the matrix becomes increasingly ill-conditioned. The ridge therefore
    scales with the magnitude of the matrix itself.
    """
    a, b = M[:, 0, 0], M[:, 0, 1]
    c, d = M[:, 1, 0], M[:, 1, 1]
    scale = np.maximum(np.abs(a) + np.abs(d), 1.0)
    det = a * d - b * c
    floor = 1e-10 * scale * scale
    det = np.where(np.abs(det) < floor, np.sign(det) * floor + (det == 0) * floor, det)
    out = np.empty_like(M)
    out[:, 0, 0] = d / det
    out[:, 0, 1] = -b / det
    out[:, 1, 0] = -c / det
    out[:, 1, 1] = a / det
    return out


def _rts_smooth(x_pred, P_pred, x_filt, P_filt, valid=None, want_cov=False):
    """Backward pass using every observation in the stay. G = P_filt F' P_pred^-1.

    Returns (smoothed_mean, smoothed_cov_or_None).

    `want_cov` defaults to False, and that is a numerical decision, not laziness.
    The covariance recursion

        Ps[t] = Pf + G (Ps[t+1] - P_pred[t+1]) G'

    is self-referential and compounds multiplicatively backward through every
    hour. Across the long unobserved stretches that sparse traits produce, G
    picks up directions with gain above one and the recursion diverges: measured
    on a real 1,000-stay batch it reaches inf for bilirubin (observed in 3.0% of
    hours), 2.4e14 for PaO2 (8.7%) and 3.1e11 for lactate (5.6%), while densely
    charted traits stay near 1e3. It overflows float32 on cast.

    The MEAN recursion is unaffected, because it depends only on forward-pass
    quantities (x_filt, x_pred, P_filt, P_pred) and never reads Ps. Since the
    smoothed mean is the only thing this project consumes -- it supplies the
    "approximate true value" in the Sec. 3.1 information-gain metric -- the
    covariance is simply not computed unless a caller asks for it.

    `valid[i, t]` marks the hours that are real for stay i, so a padded batch
    starts each stay's backward pass at its own final hour. That keeps a stay's
    result independent of which other stays share its batch; it is not what
    fixes the divergence above.
    """
    n, T, _ = x_filt.shape
    xs = x_filt.copy()
    Ps = P_filt.copy() if want_cov else None
    # F' = [[1,0],[1,1]], so (P F')[:,:,0] = P[:,:,0] and [:,:,1] = P[:,:,0]+P[:,:,1]
    for t in range(T - 2, -1, -1):
        Pf = P_filt[:, t]
        PFt = np.stack([Pf[:, :, 0], Pf[:, :, 0] + Pf[:, :, 1]], axis=2)
        G = PFt @ _inv2(P_pred[:, t + 1])
        dx = (xs[:, t + 1] - x_pred[:, t + 1])[:, :, None]
        x_new = x_filt[:, t] + (G @ dx)[:, :, 0]
        g = None if valid is None else valid[:, t + 1]

        xs[:, t] = x_new if g is None else np.where(g[:, None], x_new, x_filt[:, t])
        if want_cov:
            dP = Ps[:, t + 1] - P_pred[:, t + 1]
            P_new = Pf + G @ dP @ np.transpose(G, (0, 2, 1))
            Ps[:, t] = P_new if g is None else np.where(g[:, None, None], P_new, Pf)
    return xs, Ps


def _neg_loglik(theta, y):
    """Innovation-form negative log-likelihood, per observed point."""
    q_level, q_slope, r = np.exp(theta)
    x_pred, P_pred, _, _ = _kalman_pass(y, q_level, q_slope, r)
    seen = ~np.isnan(y)
    if not seen.any():
        return 0.0
    e = np.where(seen, y - x_pred[:, :, 0], 0.0)
    s = P_pred[:, :, 0, 0] + r
    ll = -0.5 * np.sum(np.where(seen, np.log(2 * np.pi * s) + e * e / s, 0.0))
    return float(-ll / seen.sum())


class LocalTrendForecaster(Forecaster):
    """Per-trait local linear trend model, fit by MLE, vectorized across stays."""

    def __init__(self, trait_names, seed=cfg.SEED,
                 fit_max_stays=cfg.FORECAST_TRAIN_STAYS,
                 fit_max_hours=cfg.FORECAST_FIT_MAX_HOURS):
        self.trait_names = list(trait_names)
        self.seed = seed
        self.fit_max_stays = fit_max_stays
        self.fit_max_hours = fit_max_hours
        self.mu_ = None    # [K] train mean per trait
        self.lo_ = None    # [K] lower bound for the smoothed mean
        self.hi_ = None    # [K] upper bound for the smoothed mean
        self.sd_ = None    # [K] train std per trait
        self.q_level_ = None
        self.q_slope_ = None
        self.r_ = None
        self.std_scale_ = None

    # -- fitting ----------------------------------------------------------- #
    def fit(self, obs):
        n, T, K = obs.shape
        assert K == len(self.trait_names), (K, len(self.trait_names))

        flat = obs.reshape(-1, K)
        self.mu_ = np.nanmean(flat, axis=0)
        self.sd_ = np.nanstd(flat, axis=0)
        self.sd_ = np.where(~np.isfinite(self.sd_) | (self.sd_ < 1e-6), 1.0, self.sd_)
        self.mu_ = np.where(np.isfinite(self.mu_), self.mu_, 0.0)

        # Range the training data actually spans, used to bound the smoothed
        # mean (see _run). Percentiles rather than min/max so one bad value
        # cannot widen the bound.
        with np.errstate(all="ignore"):
            lo = np.nanpercentile(flat, 0.1, axis=0)
            hi = np.nanpercentile(flat, 99.9, axis=0)
        span = np.where(np.isfinite(hi - lo), hi - lo, 1.0)
        self.lo_ = np.where(np.isfinite(lo), lo - 0.5 * span, -np.inf)
        self.hi_ = np.where(np.isfinite(hi), hi + 0.5 * span, np.inf)

        rng = np.random.default_rng(self.seed)
        idx = np.arange(n)
        if n > self.fit_max_stays:
            idx = rng.choice(n, self.fit_max_stays, replace=False)
        z = self._standardize(obs[idx][:, :self.fit_max_hours])

        self.q_level_ = np.empty(K)
        self.q_slope_ = np.empty(K)
        self.r_ = np.empty(K)
        self.std_scale_ = np.ones(K)
        x0 = np.log([0.05, 1e-3, 0.3])
        for k in range(K):
            yk = z[:, :, k]
            if not np.isfinite(yk).any():
                self.q_level_[k], self.q_slope_[k], self.r_[k] = 0.05, 1e-3, 0.3
                continue
            res = minimize(_neg_loglik, x0=x0, args=(yk,), method="Nelder-Mead",
                           options={"maxiter": 150, "xatol": 1e-3, "fatol": 1e-4})
            ql, qs, r = np.exp(res.x)
            # Floor the observation noise. Letting r -> 0 makes the filter
            # interpolate observations exactly, which collapses |m_t - y_t| and
            # kills Eq. 5 for that trait; it is also not credible for an assay.
            self.q_level_[k] = float(np.clip(ql, 1e-5, 1e2))
            self.q_slope_[k] = float(np.clip(qs, 1e-7, 1e1))
            self.r_[k] = float(np.clip(r, 1e-2, 1e2))
        return self

    def _standardize(self, obs):
        return (obs - self.mu_) / self.sd_

    # -- inference ---------------------------------------------------------- #
    def filter(self, obs, lengths=None):
        """One-step-ahead predictive (mean, std) of the OBSERVATION at hour t.

        std is sqrt(P_pred[0,0] + r), the spread of the value a draw would
        return, because sigma_t normalizes |m_t - y_t| where y_t is an observed
        value rather than a latent one.
        """
        return self._run(obs, smooth=False, lengths=lengths)

    def smooth(self, obs, lengths=None, return_std=False):
        """All-observation posterior mean. EVALUATION ONLY.

        Returns (mean, std), where std is None unless `return_std=True`. The
        smoothed covariance diverges on sparsely observed traits and nothing in
        this project consumes it; see _rts_smooth for the measured magnitudes.

        Pass `lengths` (hours per stay) whenever the batch is padded, so a stay's
        result does not depend on which other stays share its batch.
        """
        return self._run(obs, smooth=True, lengths=lengths, want_cov=return_std)

    def _run(self, obs, smooth, lengths=None, want_cov=True):
        if self.r_ is None:
            raise RuntimeError("call fit() before filter()/smooth()")
        n, T, K = obs.shape
        valid = None
        if lengths is not None:
            valid = np.arange(T)[None, :] < np.asarray(lengths)[:, None]

        z = self._standardize(obs)
        mean = np.empty((n, T, K), dtype=np.float32)
        std = np.empty((n, T, K), dtype=np.float32) if want_cov else None
        f32_max = float(np.finfo(np.float32).max)
        for k in range(K):
            ql, qs, r = self.q_level_[k], self.q_slope_[k], self.r_[k]
            x_pred, P_pred, x_filt, P_filt = _kalman_pass(z[:, :, k], ql, qs, r)
            if smooth:
                xs, Ps = _rts_smooth(x_pred, P_pred, x_filt, P_filt, valid, want_cov)
                m = xs[:, :, 0]
                v = None if Ps is None else Ps[:, :, 0, 0] + r
            else:
                m, v = x_pred[:, :, 0], P_pred[:, :, 0, 0] + r

            m = m * self.sd_[k] + self.mu_[k]
            if smooth:
                # A trait seen a handful of times across a 20-day stay lets the
                # smoother extrapolate far outside anything the assay can report
                # (PaO2 reached 4,460 mmHg on a real batch, against a ceiling of
                # ~700). Since this mean is the "approximate true value" the
                # information-gain metric scores orders against, an unbounded
                # excursion would show up as spurious information. Hold it to
                # the range the training data actually spans.
                m = np.clip(m, self.lo_[k], self.hi_[k])
            m = np.nan_to_num(m, nan=self.mu_[k], posinf=f32_max, neginf=-f32_max)
            mean[:, :, k] = np.clip(m, -f32_max, f32_max)

            if want_cov:
                s = np.sqrt(np.maximum(v, 0.0)) * self.sd_[k]
                s = s * self.std_scale_[k]
                s = np.nan_to_num(s, nan=cfg.FORECAST_MIN_STD,
                                  posinf=f32_max, neginf=cfg.FORECAST_MIN_STD)
                std[:, :, k] = np.clip(np.maximum(s, cfg.FORECAST_MIN_STD),
                                       cfg.FORECAST_MIN_STD, f32_max)
        return mean, std

    # -- persistence -------------------------------------------------------- #
    def state_dict(self):
        return {"trait_names": self.trait_names, "mu": self.mu_, "sd": self.sd_,
                "lo": self.lo_, "hi": self.hi_,
                "q_level": self.q_level_, "q_slope": self.q_slope_, "r": self.r_,
                "std_scale": self.std_scale_}

    @classmethod
    def from_state_dict(cls, d):
        f = cls(list(d["trait_names"]))
        f.mu_, f.sd_ = d["mu"], d["sd"]
        f.lo_, f.hi_ = d["lo"], d["hi"]
        f.q_level_, f.q_slope_, f.r_ = d["q_level"], d["q_slope"], d["r"]
        f.std_scale_ = np.asarray(d.get("std_scale", np.ones(len(f.trait_names))))
        return f

    def tune_on_validation(self, obs, trait_names, lengths=None,
                           q_level_scales=cfg.FORECAST_Q_LEVEL_SCALES,
                           q_slope_scales=cfg.FORECAST_Q_SLOPE_SCALES,
                           r_scales=cfg.FORECAST_R_SCALES):
        """Select target-trait dynamics by one-step-ahead validation NLL.

        Every candidate predicts hour t before folding in hour t's observation,
        so the validation result is never visible to the prediction it scores.
        After selecting dynamics, calibrate sigma by the validation residual RMS;
        this matters because sigma is the denominator of the policy utility.
        """
        if self.r_ is None:
            raise RuntimeError("call fit() before tune_on_validation()")
        name_to_idx = {name: i for i, name in enumerate(self.trait_names)}
        selected = {}
        for name in trait_names:
            k = name_to_idx[name]
            y = self._standardize(obs)[:, :, k]
            if lengths is not None:
                valid = np.arange(y.shape[1])[None, :] < np.asarray(lengths)[:, None]
                y = np.where(valid, y, np.nan)
            base = np.array([self.q_level_[k], self.q_slope_[k], self.r_[k]])
            best = None
            for sl in q_level_scales:
                for ss in q_slope_scales:
                    for sr in r_scales:
                        params = base * np.array([sl, ss, sr])
                        nll = _neg_loglik(np.log(params), y)
                        candidate = (float(nll), *params.tolist(), sl, ss, sr)
                        if best is None or candidate[0] < best[0]:
                            best = candidate
            nll, ql, qs, r, sl, ss, sr = best
            self.q_level_[k], self.q_slope_[k], self.r_[k] = ql, qs, r

            mean, std = self.filter(obs, lengths)
            observed = obs[:, :, k]
            keep = np.isfinite(observed) & np.isfinite(mean[:, :, k]) \
                & np.isfinite(std[:, :, k])
            z = (observed[keep] - mean[:, :, k][keep]) \
                / np.maximum(std[:, :, k][keep], cfg.FORECAST_MIN_STD)
            scale = float(np.sqrt(np.mean(z * z))) if len(z) else 1.0
            self.std_scale_[k] = float(np.clip(scale, 0.5, 3.0))
            selected[name] = {
                "validation_nll": nll,
                "q_level_scale": float(sl),
                "q_slope_scale": float(ss),
                "r_scale": float(sr),
                "std_calibration_scale": float(self.std_scale_[k]),
                "n_validation_observations": int(keep.sum()),
            }
        return selected


class MOGPForecaster(Forecaster):
    """Slot for the paper's sparse multi-output GP (its ref [10]).

    Not implemented. The interface is fixed so this can be filled in and selected
    with config.FORECASTER = "mogp" without touching any other stage.
    """

    def __init__(self, trait_names, **kw):
        self.trait_names = list(trait_names)

    def fit(self, obs):
        raise NotImplementedError(
            "MOGPForecaster is a slot for the closer replication of the paper's "
            "sparse multi-output GP. Use config.FORECASTER = 'local_trend'.")

    def filter(self, obs):
        raise NotImplementedError

    def smooth(self, obs):
        raise NotImplementedError


def build_forecaster(trait_names, kind=None):
    kind = kind or cfg.FORECASTER
    if kind == "local_trend":
        return LocalTrendForecaster(trait_names)
    if kind == "mogp":
        return MOGPForecaster(trait_names)
    raise ValueError(f"unknown forecaster {kind!r}")
