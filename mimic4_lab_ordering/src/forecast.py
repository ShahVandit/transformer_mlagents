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
    """Batched inverse of [n,2,2], with a ridge against singularity."""
    a, b = M[:, 0, 0], M[:, 0, 1]
    c, d = M[:, 1, 0], M[:, 1, 1]
    det = a * d - b * c
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    out = np.empty_like(M)
    out[:, 0, 0] = d / det
    out[:, 0, 1] = -b / det
    out[:, 1, 0] = -c / det
    out[:, 1, 1] = a / det
    return out


def _rts_smooth(x_pred, P_pred, x_filt, P_filt):
    """Backward pass using every observation in the stay. G = P_filt F' P_pred^-1."""
    n, T, _ = x_filt.shape
    xs = x_filt.copy()
    Ps = P_filt.copy()
    # F' = [[1,0],[1,1]], so (P F')[:,:,0] = P[:,:,0] and [:,:,1] = P[:,:,0]+P[:,:,1]
    for t in range(T - 2, -1, -1):
        Pf = P_filt[:, t]
        PFt = np.stack([Pf[:, :, 0], Pf[:, :, 0] + Pf[:, :, 1]], axis=2)
        G = PFt @ _inv2(P_pred[:, t + 1])
        dx = (xs[:, t + 1] - x_pred[:, t + 1])[:, :, None]
        xs[:, t] = x_filt[:, t] + (G @ dx)[:, :, 0]
        dP = Ps[:, t + 1] - P_pred[:, t + 1]
        Ps[:, t] = Pf + G @ dP @ np.transpose(G, (0, 2, 1))
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

    def __init__(self, trait_names, seed=cfg.SEED, fit_max_stays=300, fit_max_hours=240):
        self.trait_names = list(trait_names)
        self.seed = seed
        self.fit_max_stays = fit_max_stays
        self.fit_max_hours = fit_max_hours
        self.mu_ = None    # [K] train mean per trait
        self.sd_ = None    # [K] train std per trait
        self.q_level_ = None
        self.q_slope_ = None
        self.r_ = None

    # -- fitting ----------------------------------------------------------- #
    def fit(self, obs):
        n, T, K = obs.shape
        assert K == len(self.trait_names), (K, len(self.trait_names))

        flat = obs.reshape(-1, K)
        self.mu_ = np.nanmean(flat, axis=0)
        self.sd_ = np.nanstd(flat, axis=0)
        self.sd_ = np.where(~np.isfinite(self.sd_) | (self.sd_ < 1e-6), 1.0, self.sd_)
        self.mu_ = np.where(np.isfinite(self.mu_), self.mu_, 0.0)

        rng = np.random.default_rng(self.seed)
        idx = np.arange(n)
        if n > self.fit_max_stays:
            idx = rng.choice(n, self.fit_max_stays, replace=False)
        z = self._standardize(obs[idx][:, :self.fit_max_hours])

        self.q_level_ = np.empty(K)
        self.q_slope_ = np.empty(K)
        self.r_ = np.empty(K)
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
    def filter(self, obs):
        """One-step-ahead predictive (mean, std) of the OBSERVATION at hour t.

        std is sqrt(P_pred[0,0] + r), the spread of the value a draw would
        return, because sigma_t normalizes |m_t - y_t| where y_t is an observed
        value rather than a latent one.
        """
        return self._run(obs, smooth=False)

    def smooth(self, obs):
        """All-observation posterior (mean, std). EVALUATION ONLY."""
        return self._run(obs, smooth=True)

    def _run(self, obs, smooth):
        if self.r_ is None:
            raise RuntimeError("call fit() before filter()/smooth()")
        n, T, K = obs.shape
        z = self._standardize(obs)
        mean = np.empty((n, T, K), dtype=np.float32)
        std = np.empty((n, T, K), dtype=np.float32)
        for k in range(K):
            ql, qs, r = self.q_level_[k], self.q_slope_[k], self.r_[k]
            x_pred, P_pred, x_filt, P_filt = _kalman_pass(z[:, :, k], ql, qs, r)
            if smooth:
                xs, Ps = _rts_smooth(x_pred, P_pred, x_filt, P_filt)
                m, v = xs[:, :, 0], Ps[:, :, 0, 0] + r
            else:
                m, v = x_pred[:, :, 0], P_pred[:, :, 0, 0] + r
            mean[:, :, k] = m * self.sd_[k] + self.mu_[k]
            std[:, :, k] = np.maximum(np.sqrt(np.maximum(v, 0.0)) * self.sd_[k],
                                      cfg.FORECAST_MIN_STD)
        return mean, std

    # -- persistence -------------------------------------------------------- #
    def state_dict(self):
        return {"trait_names": self.trait_names, "mu": self.mu_, "sd": self.sd_,
                "q_level": self.q_level_, "q_slope": self.q_slope_, "r": self.r_}

    @classmethod
    def from_state_dict(cls, d):
        f = cls(list(d["trait_names"]))
        f.mu_, f.sd_ = d["mu"], d["sd"]
        f.q_level_, f.q_slope_, f.r_ = d["q_level"], d["q_slope"], d["r"]
        return f


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
