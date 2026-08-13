"""
Multi-objective fitted Q-iteration with strict Pareto pruning (paper Sec. 2.3).

Q maps a state-action pair to a VECTOR of expected cumulative rewards, one entry
per reward component, estimated with extremely randomized trees. Dominated
actions are pruned at every iteration before the Bellman backup.

An inference, flagged because it matters
----------------------------------------
The paper's Algorithm 1 is typeset as a figure, so the exact Bellman backup under
a vector-valued Q is not stated anywhere in its prose. The text says only that at
each iteration it eliminates dominated actions and retains

    Pi(s) = { a : not exists a' such that Q_d(s,a) < Q_d(s,a') for all d }.

The natural reading of "eliminate dominated actions, then back up" is a
per-dimension max restricted to the surviving set:

    target_d = r_d + gamma * max_{a' in Pi(s')} Q_d(s', a')

That is what is implemented here. It is an inference from the prose, not a
quoted equation.

A consequence worth knowing before reading any result
-----------------------------------------------------
With a binary action space the pruning is a NO-OP in the backup. If action 0 is
dominated then Q_d(s,1) > Q_d(s,0) for every d, so the per-dimension max over the
surviving set {1} equals the per-dimension max over {0,1} regardless. Since the
paper learns four independent policies with L = 1, its pruning could not have
changed its backups either; it only shapes the action set that Eq. 7 collapses.
The implementation below is written for general |A| so a joint action space would
actually exercise it, and `pareto_mask` reports how often pruning bites.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

import config as cfg


def pareto_mask(Q):
    """Non-dominated action mask for Q of shape [N, A, D].

    Action a is dominated iff some a' beats it STRICTLY on every objective.
    Returns a bool array [N, A]; at least one action per row always survives.
    """
    n, a, d = Q.shape
    dominated = np.zeros((n, a), dtype=bool)
    for i in range(a):
        for j in range(a):
            if i == j:
                continue
            dominated[:, i] |= np.all(Q[:, j, :] > Q[:, i, :], axis=1)
    keep = ~dominated
    # A strict-domination cycle is impossible, so this should never fire; the
    # guard keeps a numerical tie from emptying a row.
    empty = ~keep.any(axis=1)
    if empty.any():
        keep[empty, :] = True
    return keep


def pruned_max(Q, keep):
    """Per-dimension max over the surviving actions. Q [N,A,D], keep [N,A]."""
    masked = np.where(keep[:, :, None], Q, -np.inf)
    return masked.max(axis=1)


class MOFittedQ:
    """Vector-valued Q via one extra-trees regressor per reward dimension."""

    def __init__(self, n_actions=2, n_dims=4, n_trees=cfg.FQI_N_TREES,
                 min_samples_leaf=cfg.FQI_MIN_SAMPLES_LEAF,
                 max_depth=cfg.FQI_MAX_DEPTH, n_jobs=cfg.FQI_N_JOBS,
                 seed=cfg.SEED):
        self.n_actions = n_actions
        self.n_dims = n_dims
        self.n_trees = n_trees
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.seed = seed
        self.models_ = None        # list of D fitted regressors, or None at k=0
        self.history_ = []

    # -- representation ----------------------------------------------------- #
    @staticmethod
    def _design(states, actions):
        """Q takes (s, a), so the action is appended as a feature column."""
        return np.concatenate([states, np.asarray(actions,
                                                  dtype=np.float32).reshape(-1, 1)],
                              axis=1)

    def q_all_actions(self, states):
        """[N, A, D] Q-values. All zeros before the first iteration."""
        n = len(states)
        if self.models_ is None:
            return np.zeros((n, self.n_actions, self.n_dims), dtype=np.float32)
        out = np.empty((n, self.n_actions, self.n_dims), dtype=np.float32)
        for a in range(self.n_actions):
            X = self._design(states, np.full(n, a))
            for d, m in enumerate(self.models_):
                out[:, a, d] = m.predict(X)
        return out

    # -- training ----------------------------------------------------------- #
    def fit(self, data, iterations=cfg.FQI_ITERATIONS, gamma=cfg.GAMMA,
            sample_per_iter=cfg.FQI_SAMPLE_PER_ITER, verbose_every=10,
            preference=None):
        """Batch MO-FQI. `data` holds state/action/reward/next_state/done.

        When ``preference`` is supplied, one next action is selected by the
        scalarized Q-vector and that SAME action supplies every objective's
        Bellman target. This yields a policy-consistent frontier point. Applying
        weights only after independent per-objective maxima would combine
        returns from mutually incompatible future policies.
        """
        s = data["state"]
        a = data["action"]
        r = data["reward"].astype(np.float64)
        s2 = data["next_state"]
        not_done = (1.0 - data["done"]).astype(np.float64)
        n = len(a)
        pref = None if preference is None else np.asarray(
            preference, dtype=np.float64)
        if pref is not None and pref.shape != (self.n_dims,):
            raise ValueError("preference dimension does not match reward vector")

        # "sampled ... with probability inversely proportional to the frequency
        # of the action in the tuple" (Sec. 3). Orders are rare, so this is what
        # stops the fit from collapsing onto the do-nothing action.
        counts = np.bincount(a, minlength=self.n_actions).astype(np.float64)
        w = 1.0 / np.maximum(counts[a], 1.0)
        w = w / w.sum()

        rng = np.random.default_rng(self.seed)
        m = min(sample_per_iter, n)
        X_full = self._design(s, a)

        for k in range(iterations):
            idx = rng.choice(n, size=m, replace=m > n, p=w)

            Q_next = self.q_all_actions(s2[idx])                 # [m, A, D]
            keep = pareto_mask(Q_next)
            if pref is None:
                v_next = pruned_max(Q_next, keep)                # [m, D]
            else:
                scores = np.einsum("mad,d->ma", Q_next, pref)
                scores = np.where(keep, scores, -np.inf)
                best = np.argmax(scores, axis=1)
                v_next = Q_next[np.arange(m), best, :]           # [m, D]
            target = r[idx] + gamma * not_done[idx, None] * v_next

            models = []
            for d in range(self.n_dims):
                mdl = ExtraTreesRegressor(
                    n_estimators=self.n_trees,
                    min_samples_leaf=self.min_samples_leaf,
                    max_depth=self.max_depth,
                    n_jobs=self.n_jobs,
                    random_state=self.seed + k,
                )
                mdl.fit(X_full[idx], target[:, d])
                models.append(mdl)
            self.models_ = models

            pruned_frac = float(1.0 - keep.mean())
            self.history_.append({"iteration": k + 1,
                                  "pruned_action_frac": pruned_frac,
                                  "target_mean": target.mean(axis=0).tolist()})
            if verbose_every and ((k + 1) % verbose_every == 0 or k == 0):
                tm = np.array2string(target.mean(axis=0), precision=4,
                                     floatmode="fixed")
                print(f"    iter {k + 1:4d}/{iterations}  target_mean={tm}  "
                      f"pruned={pruned_frac:.3f}", flush=True)
        return self

    # -- policy extraction --------------------------------------------------- #
    def collapse(self, states, eps):
        """Eq. 7: order iff Q_d(s,1) + eps_d > Q_d(s,0) for ALL d.

        A deliberately strong condition: the paper's stated priority is to cut
        unnecessary ordering, so a lab is drawn only when it is not worse on any
        objective. `eps` relaxes that per objective; the paper tunes only the
        cost slack.
        """
        return collapse_from_q(self.q_all_actions(states), eps)


class WeightedQPolicy:
    """Deterministic binary policy from a weighted action-advantage vector.

    The joint model is trained once with objectives ``[utility, -burden]``.
    For each state, the draw advantage is ``Q(s, draw) - Q(s, no_draw)``.
    Preferences weight that same advantage vector, yielding several policies
    from the same vector-valued MO-FQI model.
    """

    def __init__(self, model, preference):
        self.model = model
        self.preference = np.asarray(preference, dtype=np.float64)
        if self.preference.shape != (model.n_dims,):
            raise ValueError("preference dimension does not match Q-vector")

    def predict(self, states):
        q = self.model.q_all_actions(np.asarray(states, dtype=np.float32))
        if q.shape[1] != 2:
            raise ValueError("weighted advantage policy requires two actions")
        advantage = q[:, 1, :] - q[:, 0, :]
        return (advantage @ self.preference > 0.0).astype(np.int64)


class LagrangianQPolicy:
    """Binary policy with a clinically calibrated burden trade-off.

    ``lambda_burden`` is selected on validation data, not used in MO-FQI
    training. The shared model predicts ``[utility, -burden]`` and the policy
    draws when ``delta_utility + lambda * delta_neg_burden > 0``.
    """

    def __init__(self, model, lambda_burden):
        self.model = model
        self.lambda_burden = float(lambda_burden)
        if not np.isfinite(self.lambda_burden) or self.lambda_burden < 0.0:
            raise ValueError("lambda_burden must be finite and non-negative")
        if model.n_dims != 2:
            raise ValueError("Lagrangian policy requires [utility, -burden]")

    def predict(self, states):
        q = self.model.q_all_actions(np.asarray(states, dtype=np.float32))
        if q.shape[1] != 2:
            raise ValueError("Lagrangian advantage policy requires two actions")
        advantage = q[:, 1, :] - q[:, 0, :]
        score = advantage[:, 0] + self.lambda_burden * advantage[:, 1]
        return (score > 0.0).astype(np.int64)


def _discount_weights(stay_ids, gamma):
    """Per-row discount weights whose dot product is mean return per stay."""
    stay = np.asarray(stay_ids)
    if len(stay) == 0:
        return np.zeros(0, dtype=np.float64)
    boundaries = np.r_[0, np.flatnonzero(stay[1:] != stay[:-1]) + 1, len(stay)]
    out = np.empty(len(stay), dtype=np.float64)
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        out[lo:hi] = float(gamma) ** np.arange(hi - lo, dtype=np.float64)
    return out / max(len(boundaries) - 1, 1)


def calibrate_burden_policies(model, split, burden_fractions, gamma=cfg.GAMMA,
                              max_breakpoints=512):
    """Choose Lagrange multipliers for clinical burden budgets on validation.

    Candidate multipliers come from the learned advantage breakpoints, where
    an action can actually change. For each fraction of clinician burden, the
    selected candidate maximizes replay utility while respecting that limit.
    The held-out test split must never be passed here.
    """
    fractions = np.asarray(burden_fractions, dtype=np.float64)
    if (fractions.ndim != 1 or len(fractions) == 0
            or not np.isfinite(fractions).all() or (fractions <= 0.0).any()):
        raise ValueError("burden fractions must be positive finite values")

    q = model.q_all_actions(np.asarray(split["state"], dtype=np.float32))
    if q.shape[1:] != (2, 2):
        raise ValueError("calibration requires binary [utility, -burden] Q-values")
    advantage = q[:, 1, :].astype(np.float64) - q[:, 0, :].astype(np.float64)
    du, dneg_burden = advantage[:, 0], advantage[:, 1]
    roots = -du[np.abs(dneg_burden) > 1e-12] / dneg_burden[np.abs(dneg_burden) > 1e-12]
    roots = np.unique(roots[np.isfinite(roots) & (roots >= 0.0)])
    if len(roots) > max_breakpoints:
        idx = np.linspace(0, len(roots) - 1, max_breakpoints).round().astype(int)
        roots = roots[idx]
    candidates = np.unique(np.r_[0.0, roots, np.nextafter(roots, -np.inf)])
    candidates = candidates[np.isfinite(candidates) & (candidates >= 0.0)]
    if len(roots):
        candidates = np.unique(np.r_[candidates, np.nextafter(roots[-1], np.inf)])

    weights = _discount_weights(split["stay_id"], gamma)
    trigger = np.asarray(split["clinical_trigger"], dtype=np.float64)
    draw_burden = np.asarray(split["draw_burden"], dtype=np.float64)
    clinician_draw = np.asarray(split["action"]) != 0
    clinician_burden = float(np.dot(weights, draw_burden * clinician_draw))
    if clinician_burden <= 0.0:
        raise ValueError("clinician validation burden must be positive")

    candidate_rows = []
    for lam in candidates:
        draw = du + float(lam) * dneg_burden > 0.0
        candidate_rows.append({
            "lambda_burden": float(lam),
            "utility": float(np.dot(weights, trigger * draw)),
            "burden": float(np.dot(weights, draw_burden * draw)),
            "draw_rate": float(draw.mean()),
        })

    selected = []
    days = max(len(trigger) / 24.0, 1e-6)
    for fraction in fractions:
        limit = float(fraction * clinician_burden)
        feasible = [row for row in candidate_rows if row["burden"] <= limit + 1e-12]
        best = (max(feasible, key=lambda row: (row["utility"], row["burden"],
                                               -row["lambda_burden"]))
                if feasible else None)
        if best is None:
            selected.append({
                "burden_fraction": float(fraction), "burden_limit": limit,
                "clinician_burden": clinician_burden, "valid": False,
            })
            continue
        row = dict(best)
        row.update({
            "burden_fraction": float(fraction),
            "burden_limit": limit,
            "clinician_burden": clinician_burden,
            "achieved_burden_fraction": row["burden"] / clinician_burden,
            "draws_per_patient_day": row["draw_rate"] * 24.0,
            "valid": bool(row["utility"] > 0.0),
        })
        selected.append(row)
    return selected


def epsilon_constraint_select(candidates, budgets, margin=0.0):
    """Maximize validation utility subject to each burden budget.

    ``candidates`` is an iterable of mappings with ``name``, ``utility``, and
    ``burden``. This is an epsilon-constraint over a finite policy set, not a
    local action threshold. Ties prefer lower burden and then a stable name.
    """
    rows = [dict(row) for row in candidates]
    required = {"name", "utility", "burden"}
    for row in rows:
        missing = required.difference(row)
        if missing:
            raise ValueError(f"epsilon candidate missing fields: {sorted(missing)}")
        row["utility"] = float(row["utility"])
        row["burden"] = float(row["burden"])
        if not np.isfinite(row["utility"]) or not np.isfinite(row["burden"]):
            raise ValueError("epsilon candidates must have finite objectives")

    selected = []
    for budget in budgets:
        budget = float(budget)
        if not np.isfinite(budget) or budget < 0.0:
            raise ValueError("epsilon burden budgets must be finite and non-negative")
        feasible = [row for row in rows if row["burden"] <= budget + margin]
        best = (max(feasible, key=lambda row: (row["utility"],
                                               -row["burden"],
                                               str(row["name"])))
                if feasible else None)
        selected.append({"budget": budget, "selected": best})
    return selected


def collapse_from_q(Q, eps):
    """Eq. 7 applied to precomputed Q-values.

    Q does not depend on eps, so tuning the slack means re-running this
    comparison, not re-running the forest. Splitting it out turns a search over
    eps from minutes of prediction into milliseconds of arithmetic.
    """
    eps = np.asarray(eps, dtype=np.float64).reshape(1, -1)
    return np.all(Q[:, 1, :] + eps > Q[:, 0, :], axis=1).astype(np.int8)


def apply_budget(actions, stay_ids, hours, budget_hours=cfg.BUDGET_HOURS):
    """Force one order at the end of every `budget_hours` window with none.

    Paper Sec. 3: "we introduce a budget that suggests taking a lab at the end of
    every 24 hour period for which our policy recommends no orders", which both
    covers stretches the policy leaves silent and reflects protocols requiring
    minimum daily monitoring.

    Note this makes the recommendation rule NON-MARKOVIAN: it depends on how long
    it has been since the last RECOMMENDED order, which is not in the state (the
    state carries time since the last ACTUAL order). Stage 5 therefore evaluates
    the Markov policy without the budget, and stage 6 reports both.
    """
    out = np.asarray(actions).copy()
    n = len(out)
    if n == 0:
        return out
    start = 0
    for i in range(1, n + 1):
        if i == n or stay_ids[i] != stay_ids[start]:
            since = 0
            for t in range(start, i):
                if out[t] == 1:
                    since = 0
                else:
                    since += 1
                    if since >= budget_hours:
                        out[t] = 1
                        since = 0
            start = i
    return out
