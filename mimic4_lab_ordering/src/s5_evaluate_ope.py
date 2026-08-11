"""
Stage 5: off-policy evaluation of the learned lab-ordering policy.

Two tiers, kept separate in the report because they answer different questions.

TIER 1 - replicate the paper (Sec. 3.1)
    Per-step weighted importance sampling at gamma_WIS = 1.0, evaluated
    separately for each of the four reward components, against the behavior
    (clinician) policy and randomized policies at p in {0.01, p_emp, 0.5}
    averaged over ten trials each. This is the paper's Figure 4.

TIER 2 - what the paper does not report
    FQE, WDR, patient-level bootstrap confidence intervals, effective sample
    size for every importance-weighted estimate, action-support diagnostics, and
    an FQE calibration check run on the behavior policy itself.

Three things to hold in mind when reading the numbers
-----------------------------------------------------
1. Three of the four reward components are gated on a != 0 and the fourth only
   fires when a = 1, so NOT ordering yields the zero vector. A policy that orders
   more often is therefore mechanically advantaged on r_SOFA, r_treat and r_info,
   and penalised only through r_cost. Comparing V_d across policies with very
   different order rates is comparing order rates as much as policy quality.
2. The paper's final policy is deterministic, which makes every importance ratio
   either 0 or 1/pi_b and collapses the cumulative product onto a handful of
   trajectories. A softened (epsilon-greedy) target is evaluated instead, and the
   ESS of both is reported so the size of that problem is visible.
3. The 24-hour budget rule is not evaluated here. It depends on time since the
   last RECOMMENDED order, which is not in the state, so the rule is not a Markov
   policy and no estimator below applies to it. Stage 6 reports its effect.

Output: reports/ope_<lab>.md
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

N_ACTIONS = 2
D = len(cfg.REWARD_DIMS)


# ------------------------------------------------------------------- data ----
def load_split(lab, split):
    d = np.load(cfg.RL_DIR / f"{lab}_{split}.npz")
    return {k: d[k] for k in d.files}


def to_trajectories(split):
    """Index arrays for each ICU stay, in row order."""
    ids = split["stay_id"]
    trajs, start = [], 0
    for i in range(1, len(ids) + 1):
        if i == len(ids) or ids[i] != ids[start]:
            trajs.append(np.arange(start, i))
            start = i
    return trajs


# --------------------------------------------------------------- policies ----
def fit_behavior_policy(train):
    """pi_b(a|s), the clinician's ordering probability, from a fitted classifier.

    The paper does the same ("the behaviour policy was found by training a
    regressor on real state-action pairs observed in the dataset").
    """
    clf = HistGradientBoostingClassifier(random_state=cfg.SEED, max_iter=300,
                                         learning_rate=0.1)
    clf.fit(train["state"], train["action"])
    return clf


def behavior_probs(clf, states):
    p = clf.predict_proba(states)
    full = np.full((len(states), N_ACTIONS), cfg.OPE_PROB_FLOOR)
    for j, c in enumerate(clf.classes_):
        full[:, c] = np.maximum(p[:, j], cfg.OPE_PROB_FLOOR)
    return full / full.sum(axis=1, keepdims=True)


def deterministic_actions(bundle, states):
    """The stage-4 policy's action, before any softening or budget rule."""
    if bundle["policy"] is not None:
        return bundle["policy"].predict(states).astype(int)
    return bundle["model"].collapse(states, bundle["eps"]).astype(int)


def epsilon_greedy_probs(actions, epsilon):
    """Soften a deterministic policy so importance ratios are finite."""
    p = np.full((len(actions), N_ACTIONS), epsilon / N_ACTIONS)
    p[np.arange(len(actions)), actions] += 1.0 - epsilon
    return p


def constant_probs(n, p_order):
    """A state-independent randomized policy, the paper's Fig. 4 baseline."""
    out = np.empty((n, N_ACTIONS))
    out[:, 1] = p_order
    out[:, 0] = 1.0 - p_order
    return np.clip(out, cfg.OPE_PROB_FLOOR, 1.0)


# ------------------------------------------------- importance-sampling core ----
def per_step_log_weights(split, trajs, pi_e, pi_b):
    """Cumulative clipped log importance ratios rho_t, per trajectory."""
    a = split["action"]
    rows = np.arange(len(a))
    lr = np.log(pi_e[rows, a]) - np.log(pi_b[rows, a])
    lr = np.clip(lr, -cfg.OPE_RATIO_CLIP, cfg.OPE_RATIO_CLIP)
    return [np.cumsum(lr[tr]) for tr in trajs]


def _bucket_by_step(trajs, traj_logw, values):
    """Regroup per-trajectory arrays into per-timestep lists."""
    max_len = max(len(tr) for tr in trajs)
    logw = [[] for _ in range(max_len)]
    vals = [[] for _ in range(max_len)]
    for tr, lw in zip(trajs, traj_logw):
        v = values[tr]
        for t in range(len(tr)):
            logw[t].append(lw[t])
            vals[t].append(v[t])
    return logw, vals


def ps_wis(split, trajs, traj_logw, dim, gamma=cfg.WIS_GAMMA):
    """Per-step weighted importance sampling for one reward dimension.

    V = sum_t gamma^t * [ sum_i rho_t(i) r_t(i) / sum_i rho_t(i) ]

    Self-normalized per step, in log space so a large ratio cannot overflow.
    """
    r = split["reward"][:, dim]
    logw, vals = _bucket_by_step(trajs, traj_logw, r)
    total = 0.0
    for t, (lw, rr) in enumerate(zip(logw, vals)):
        lw = np.asarray(lw)
        rr = np.asarray(rr)
        w = np.exp(lw - lw.max())
        total += (gamma ** t) * float((w * rr).sum() / w.sum())
    return total


def effective_sample_size(trajs, traj_logw):
    """ESS = (sum w)^2 / sum w^2, per step, plus the whole-horizon figure.

    The single cheapest diagnostic the paper omits. An ESS of 3 out of 2,400
    trajectories means the estimate is three patients wearing a trenchcoat.
    """
    max_len = max(len(tr) for tr in trajs)
    per_step = []
    for t in range(max_len):
        lw = np.array([w[t] for w in traj_logw if len(w) > t])
        w = np.exp(lw - lw.max())
        per_step.append(float(w.sum() ** 2 / np.maximum((w ** 2).sum(), 1e-300)))
    final = np.array([w[-1] for w in traj_logw])
    wf = np.exp(final - final.max())
    return {
        "ess_final": float(wf.sum() ** 2 / np.maximum((wf ** 2).sum(), 1e-300)),
        "ess_min_over_steps": float(np.min(per_step)),
        "ess_mean_over_steps": float(np.mean(per_step)),
        "n_trajectories": len(trajs),
    }


# ---------------------------------------------------------------- FQE / WDR ----
class QNet(nn.Module):
    """One trunk, A*D outputs: a Q-vector per action.

    The output layer is zero-initialized so the network starts at Q == 0
    everywhere. With a randomly initialized head the first Bellman targets are
    `r + gamma * V_random`, which is dominated by noise and can drive the fit to
    values of the wrong sign entirely: three of the four reward components here
    are non-negative by construction, so a negative Q is a diagnosable error
    rather than merely a noisy one. Starting at zero removes that failure mode.
    """

    def __init__(self, state_dim, hidden=cfg.FQE_HIDDEN):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, N_ACTIONS * D)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, s):
        return self.head(self.trunk(s)).view(-1, N_ACTIONS, D)


def train_fqe(train, pi_e_next, epochs=cfg.FQE_EPOCHS, gamma=cfg.GAMMA):
    """Fitted Q evaluation of the target policy, all reward dimensions at once."""
    torch.manual_seed(cfg.SEED)
    sd = train["state"].shape[1]
    q = QNet(sd)
    q_tgt = QNet(sd)
    q_tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=cfg.FQE_LR)

    s = torch.tensor(train["state"], dtype=torch.float32)
    a = torch.tensor(train["action"], dtype=torch.long)
    r = torch.tensor(train["reward"], dtype=torch.float32)
    s2 = torch.tensor(train["next_state"], dtype=torch.float32)
    done = torch.tensor(train["done"], dtype=torch.float32)
    pin = torch.tensor(pi_e_next, dtype=torch.float32)

    n = len(a)
    # On a small split, `epochs` passes can be too few gradient steps for the
    # Polyak-averaged target to move at all. Hold the step count, not the pass
    # count, so a smoke run is undertrained by a knowable amount.
    steps_per_epoch = max(1, int(np.ceil(n / cfg.FQE_BATCH)))
    epochs = max(epochs, int(np.ceil(cfg.FQE_MIN_STEPS / steps_per_epoch)))
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, cfg.FQE_BATCH):
            idx = perm[i:i + cfg.FQE_BATCH]
            with torch.no_grad():
                qn = q_tgt(s2[idx])                                # [B,A,D]
                vn = (pin[idx].unsqueeze(-1) * qn).sum(dim=1)      # [B,D]
                target = r[idx] + gamma * (1 - done[idx]).unsqueeze(-1) * vn
            qsa = q(s[idx])[torch.arange(len(idx)), a[idx]]        # [B,D]
            loss = F.mse_loss(qsa, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                for p, tp in zip(q.parameters(), q_tgt.parameters()):
                    tp.data.mul_(0.995).add_(0.005 * p.data)
    return q


def q_values(q, states):
    with torch.no_grad():
        return q(torch.tensor(states, dtype=torch.float32)).numpy()


def fqe_initial_values(qv, pi_e, trajs):
    """V(s0) = sum_a pi_e(a|s0) Q(s0,a), one row per trajectory, [N_traj, D]."""
    v = (pi_e[:, :, None] * qv).sum(axis=1)
    return np.stack([v[tr[0]] for tr in trajs])


def wdr(split, trajs, traj_logw, qv, pi_e, dim, gamma=cfg.GAMMA):
    """Weighted doubly robust: the FQE baseline plus a self-normalized correction."""
    a = split["action"]
    rows = np.arange(len(a))
    q_sa = qv[rows, a, dim]
    v_s = (pi_e[:, :, None] * qv).sum(axis=1)[:, dim]
    r = split["reward"][:, dim]

    max_len = max(len(tr) for tr in trajs)
    buckets = [[] for _ in range(max_len)]
    base = []
    for tr, lw in zip(trajs, traj_logw):
        base.append(v_s[tr[0]])
        for t in range(len(tr)):
            v_next = v_s[tr[t + 1]] if t + 1 < len(tr) else 0.0
            buckets[t].append((lw[t], r[tr[t]], q_sa[tr[t]], v_next))

    corr = 0.0
    for t, arr in enumerate(buckets):
        lw = np.array([x[0] for x in arr])
        rr = np.array([x[1] for x in arr])
        qq = np.array([x[2] for x in arr])
        vn = np.array([x[3] for x in arr])
        w = np.exp(lw - lw.max())
        corr += (gamma ** t) * float((w * (rr + gamma * vn - qq)).sum() / w.sum())
    return float(np.mean(base)) + corr


# --------------------------------------------------------------- bootstrap ----
def subject_bootstrap(estimator, trajs, subjects, n_boot=cfg.OPE_N_BOOTSTRAP,
                      seed=cfg.SEED):
    """Resample PATIENTS, not trajectories: one patient can hold several stays."""
    point = estimator(trajs)
    by_subject = {}
    for k, tr in enumerate(trajs):
        by_subject.setdefault(subjects[k], []).append(k)
    keys = list(by_subject)

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        idx = [j for p in pick for j in by_subject[keys[p]]]
        if not idx:
            continue
        boots.append(estimator([trajs[j] for j in idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


# ------------------------------------------------------------------- main ----
def factual_return(split, trajs, dim, gamma):
    """Observed discounted return of the logged clinician behaviour."""
    vals = []
    for tr in trajs:
        r = split["reward"][tr, dim]
        vals.append(float((gamma ** np.arange(len(r)) * r).sum()))
    return np.array(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", required=True)
    ap.add_argument("--learner", default="mofqi", choices=["mofqi", "cql"],
                    help="which stage-4 bundle to evaluate")
    ap.add_argument("--epsilon", type=float, default=cfg.OPE_EPSILON)
    args = ap.parse_args()

    cfg.ensure_dirs()
    lab = args.lab
    with open(cfg.MODELS_DIR / f"{lab}_{args.learner}.pkl", "rb") as fh:
        bundle = pickle.load(fh)

    train = load_split(lab, "train")
    test = load_split(lab, "test")
    trajs = to_trajectories(test)
    subj_of_traj = [int(test["subject_id"][tr[0]]) for tr in trajs]
    print(f"{lab}: {len(trajs):,} test stays, "
          f"{len(set(subj_of_traj)):,} test patients, "
          f"{len(test['action']):,} transitions")

    # ---- policies ----
    beh = fit_behavior_policy(train)
    pi_b_test = behavior_probs(beh, test["state"])
    p_emp = float(train["action"].mean())

    det_test = deterministic_actions(bundle, test["state"])
    pi_e_test = epsilon_greedy_probs(det_test, args.epsilon)
    pi_e_next = epsilon_greedy_probs(
        deterministic_actions(bundle, train["next_state"]), args.epsilon)
    print(f"  learned policy order rate (test, deterministic): {det_test.mean():.4f}"
          f"   clinician: {test['action'].mean():.4f}")

    logw_learned = per_step_log_weights(test, trajs, pi_e_test, pi_b_test)
    logw_det = per_step_log_weights(
        test, trajs, epsilon_greedy_probs(det_test, 1e-6), pi_b_test)

    # ---- tier 1: the paper's comparison ----
    policies = {"MO-FQI": pi_e_test}
    for p in cfg.RANDOM_BASELINE_PS:
        pv = p_emp if p is None else p
        policies[f"random p={pv:.3f}"] = constant_probs(len(test["action"]), pv)

    tier1 = {}
    for name, pi in policies.items():
        if name.startswith("random"):
            trials = []
            for t in range(cfg.RANDOM_BASELINE_TRIALS):
                lw = per_step_log_weights(test, trajs, pi, pi_b_test)
                trials.append([ps_wis(test, trajs, lw, d) for d in range(D)])
            arr = np.array(trials)
            tier1[name] = {"mean": arr.mean(axis=0), "std": arr.std(axis=0)}
        else:
            lw = logw_learned
            tier1[name] = {"mean": np.array([ps_wis(test, trajs, lw, d)
                                             for d in range(D)]),
                           "std": np.zeros(D)}
    beh_factual = np.array([factual_return(test, trajs, d, cfg.WIS_GAMMA).mean()
                            for d in range(D)])
    tier1["clinician (factual)"] = {"mean": beh_factual, "std": np.zeros(D)}

    print("\ntier 1: PS-WIS per reward component (gamma_WIS=1.0)")
    print(f"  {'policy':22s} " + " ".join(f"{d:>12s}" for d in cfg.REWARD_DIMS))
    for name, v in tier1.items():
        print(f"  {name:22s} " + " ".join(f"{x:12.4f}" for x in v["mean"]))

    # ---- tier 2: FQE, WDR, CIs, ESS, support ----
    print("\ntier 2: training FQE")
    qnet = train_fqe(train, pi_e_next)
    qv_test = q_values(qnet, test["state"])
    fqe_v0 = fqe_initial_values(qv_test, pi_e_test, trajs)

    idx_of = {id(tr): k for k, tr in enumerate(trajs)}
    tier2 = {}
    for d, dim_name in enumerate(cfg.REWARD_DIMS):
        clin = factual_return(test, trajs, d, cfg.GAMMA)

        def clin_est(sample, clin=clin):
            return float(np.mean([clin[idx_of[id(tr)]] for tr in sample]))

        def fqe_est(sample, d=d):
            return float(np.mean([fqe_v0[idx_of[id(tr)], d] for tr in sample]))

        def wis_est(sample, d=d):
            lw = [logw_learned[idx_of[id(tr)]] for tr in sample]
            return ps_wis(test, sample, lw, d)

        def wdr_est(sample, d=d):
            lw = [logw_learned[idx_of[id(tr)]] for tr in sample]
            return wdr(test, sample, lw, qv_test, pi_e_test, d)

        tier2[dim_name] = {
            "clinician": subject_bootstrap(clin_est, trajs, subj_of_traj),
            "FQE": subject_bootstrap(fqe_est, trajs, subj_of_traj),
            "PS-WIS": subject_bootstrap(wis_est, trajs, subj_of_traj),
            "WDR": subject_bootstrap(wdr_est, trajs, subj_of_traj),
        }

    print(f"\n  {'component':12s} {'clinician':>26s} {'FQE':>26s} "
          f"{'PS-WIS':>26s} {'WDR':>26s}")
    for name, row in tier2.items():
        cells = []
        for est in ("clinician", "FQE", "PS-WIS", "WDR"):
            pt, lo, hi = row[est]
            cells.append(f"{pt:+8.4f} [{lo:+7.3f},{hi:+7.3f}]")
        print(f"  {name:12s} " + " ".join(f"{c:>26s}" for c in cells))

    ess_soft = effective_sample_size(trajs, logw_learned)
    ess_det = effective_sample_size(trajs, logw_det)
    print(f"\n  ESS softened (eps={args.epsilon}): final "
          f"{ess_soft['ess_final']:.1f} / {ess_soft['n_trajectories']} trajectories")
    print(f"  ESS deterministic:            final "
          f"{ess_det['ess_final']:.1f} / {ess_det['n_trajectories']} trajectories")

    # ---- support diagnostics ----
    rows = np.arange(len(test["action"]))
    pi_b_rec = pi_b_test[rows, det_test]
    support = {
        "floor_hit_rate": float((pi_b_test <= cfg.OPE_PROB_FLOOR * 1.001).mean()),
        "recommended_below_0.01": float((pi_b_rec < 0.01).mean()),
        "recommended_below_0.05": float((pi_b_rec < 0.05).mean()),
        "median_pi_b_of_recommended": float(np.median(pi_b_rec)),
        "action_agreement": float((det_test == test["action"]).mean()),
    }
    print("\n  support diagnostics:")
    for k, v in support.items():
        print(f"    {k:28s} {v:.4f}")

    # ---- FQE calibration check on the behavior policy ----
    print("\n  FQE calibration check (pi_e := pi_b)")
    qnet_b = train_fqe(train, behavior_probs(beh, train["next_state"]))
    qv_b = q_values(qnet_b, test["state"])
    fqe_b_v0 = fqe_initial_values(qv_b, pi_b_test, trajs)
    calib = {}
    for d, dim_name in enumerate(cfg.REWARD_DIMS):
        clin = factual_return(test, trajs, d, cfg.GAMMA)

        def f_est(sample, d=d):
            return float(np.mean([fqe_b_v0[idx_of[id(tr)], d] for tr in sample]))

        def c_est(sample, clin=clin):
            return float(np.mean([clin[idx_of[id(tr)]] for tr in sample]))

        fq = subject_bootstrap(f_est, trajs, subj_of_traj)
        cl = subject_bootstrap(c_est, trajs, subj_of_traj)
        inside = cl[1] <= fq[0] <= cl[2]
        calib[dim_name] = {"fqe_on_pi_b": fq, "factual": cl, "inside_ci": bool(inside)}
        print(f"    {dim_name:12s} FQE {fq[0]:+.4f}  factual {cl[0]:+.4f} "
              f"[{cl[1]:+.4f},{cl[2]:+.4f}]  {'OK' if inside else 'MISMATCH'}")

    results = {
        "lab": lab, "epsilon": args.epsilon,
        "reward_dims": cfg.REWARD_DIMS,
        "tier1": {k: {"mean": v["mean"].tolist(), "std": v["std"].tolist()}
                  for k, v in tier1.items()},
        "tier2": {k: {e: list(v) for e, v in row.items()}
                  for k, row in tier2.items()},
        "ess_softened": ess_soft, "ess_deterministic": ess_det,
        "support": support,
        "calibration": {k: {"fqe_on_pi_b": list(v["fqe_on_pi_b"]),
                            "factual": list(v["factual"]),
                            "inside_ci": v["inside_ci"]} for k, v in calib.items()},
        "order_rate_policy": float(det_test.mean()),
        "order_rate_clinician": float(test["action"].mean()),
        "n_trajectories": len(trajs), "n_patients": len(set(subj_of_traj)),
    }
    sfx = "" if args.learner == "mofqi" else f"_{args.learner}"
    (cfg.REPORTS_DIR / f"ope_{lab}{sfx}.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    write_report(lab, args, tier1, tier2, ess_soft, ess_det, support, calib,
                 det_test, test, len(trajs), len(set(subj_of_traj)), sfx)


def write_report(lab, args, tier1, tier2, ess_soft, ess_det, support, calib,
                 det_test, test, n_traj, n_subj, sfx=""):
    L = [
        f"# Off-policy evaluation: {lab}\n\n",
        f"Test split, patient-disjoint. {n_traj:,} ICU stays from {n_subj:,} patients, "
        f"{len(test['action']):,} hourly decisions.\n\n",
        f"Learned policy order rate {det_test.mean():.4f} vs clinician "
        f"{test['action'].mean():.4f}.\n\n",
        "## Tier 1 - the paper's comparison (Fig. 4)\n\n",
        f"Per-step weighted importance sampling, `gamma_WIS = {cfg.WIS_GAMMA}`, one "
        f"column per reward component. Randomized baselines are averaged over "
        f"{cfg.RANDOM_BASELINE_TRIALS} trials.\n\n",
        "| policy | " + " | ".join(cfg.REWARD_DIMS) + " |\n",
        "|---" * (D + 1) + "|\n",
    ]
    for name, v in tier1.items():
        cells = [f"{m:+.4f}" + (f" ± {s:.4f}" if s > 0 else "")
                 for m, s in zip(v["mean"], v["std"])]
        L.append(f"| {name} | " + " | ".join(cells) + " |\n")

    L += [
        "\n## Tier 2 - estimators and intervals the paper does not report\n\n",
        f"`gamma = {cfg.GAMMA}` (the MDP discount, not the tier-1 undiscounted "
        f"horizon). 95% intervals from {cfg.OPE_N_BOOTSTRAP} PATIENT-level "
        f"bootstrap resamples.\n\n",
        "| component | clinician (factual) | FQE | PS-WIS | WDR |\n",
        "|---|---|---|---|---|\n",
    ]
    for name, row in tier2.items():
        cells = []
        for est in ("clinician", "FQE", "PS-WIS", "WDR"):
            pt, lo, hi = row[est]
            cells.append(f"{pt:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        L.append(f"| {name} | " + " | ".join(cells) + " |\n")

    L += [
        "\n## Effective sample size\n\n",
        "| target policy | ESS at final step | mean ESS over steps | trajectories |\n",
        "|---|---|---|---|\n",
        f"| softened, eps={args.epsilon} | {ess_soft['ess_final']:.1f} | "
        f"{ess_soft['ess_mean_over_steps']:.1f} | {ess_soft['n_trajectories']:,} |\n",
        f"| deterministic | {ess_det['ess_final']:.1f} | "
        f"{ess_det['ess_mean_over_steps']:.1f} | {ess_det['n_trajectories']:,} |\n",
        "\nESS is the number of trajectories the importance-weighted estimate is "
        "effectively averaging over. Read the absolute values, not the gap "
        "between the two rows: both are computed under a per-step log-ratio clip "
        "of +/-"
        f"{cfg.OPE_RATIO_CLIP}, which bounds how much a single action mismatch "
        "can cost and therefore compresses the difference between a deterministic "
        "and a softened target. An ESS in the single digits means the PS-WIS "
        "column rests on a handful of patients regardless of how many are in the "
        "split. The paper reports neither ESS nor intervals, so there is no way "
        "to tell from it how many patients its Fig. 4 values actually rest on; "
        "that is the reason this table exists.\n",
        "\n## Action support\n\n| diagnostic | value |\n|---|---|\n",
    ]
    for k, v in support.items():
        L.append(f"| {k} | {v:.4f} |\n")

    L += [
        "\n## FQE calibration check\n\n",
        "FQE is re-run with the target policy set to the behavior policy. Its "
        "estimate should then land inside the bootstrap interval of the observed "
        "factual return. Where it does not, the Q model is misspecified and the "
        "FQE column above should not be trusted for that component.\n\n",
        "| component | FQE on pi_b | factual return | inside CI |\n|---|---|---|---|\n",
    ]
    for name, c in calib.items():
        f0 = c["fqe_on_pi_b"][0]
        pt, lo, hi = c["factual"]
        L.append(f"| {name} | {f0:+.4f} | {pt:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
                 f"{'yes' if c['inside_ci'] else '**NO**'} |\n")

    L.append(
        "\n## Reading these numbers\n\n"
        "Three of the four reward components are gated on `a != 0` (Eqs. 3-5) and "
        "the fourth only fires when `a = 1` (Eq. 6), so declining to order yields "
        "the zero vector in every dimension. A policy that orders more often is "
        "therefore mechanically advantaged on `r_sofa`, `r_treat` and `r_info`, and "
        "penalised only through `neg_r_cost`. Any comparison of `V_d` between "
        "policies with different order rates is partly a comparison of order rates. "
        "This is a property of the paper's reward design, reproduced faithfully "
        "here, not an artefact of this implementation.\n\n"
        "The 24-hour budget rule is absent from every estimate above. It triggers "
        "on time since the last RECOMMENDED order, which is not a state variable, "
        "so the rule is not a Markov policy and none of these estimators apply to "
        "it. Its effect on order counts is in the stage-6 report.\n")

    out = cfg.REPORTS_DIR / f"ope_{lab}{sfx}.md"
    out.write_text("".join(L), encoding="utf-8")
    print(f"\nwrote report -> {out}")


if __name__ == "__main__":
    main()
