"""
Stage 4: train the multi-objective policy for one lab (paper Sec. 2.3 and 3).

Four steps, in the paper's order:

  1. MO-FQI on the TRAIN split: a vector-valued Q via extra-trees, Pareto pruning
     each iteration, 200 iterations at gamma = 0.9, resampling 100k transitions
     per iteration inversely to action frequency.
  2. Collapse to a deterministic rule with Eq. 7, tuning the cost slack eps_cost
     so the recommended order count approximates the observed count. The paper
     tunes this on its training set; it is tuned on VAL here, which is strictly
     cleaner and does not change the method.
  3. Fit the final policy function pi: S -> A with an extra-trees classifier on
     the collapsed actions, and report its Gini feature importances (Fig. 2).
  4. Apply the 24-hour budget rule to the recommendations.

The budget rule is deliberately kept OUT of the saved policy function. It depends
on time since the last RECOMMENDED order, which is not in the state, so a policy
including it is not Markov and cannot be evaluated by any of the stage-5
estimators. Stage 6 reports order counts both with and without it.

Output: models/<lab>_mofqi.pkl, reports/train_<lab>.md
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
import mofqi


def load_split(lab, split):
    p = cfg.RL_DIR / f"{lab}_{split}.npz"
    if not p.exists():
        raise SystemExit(f"{p} not found; run stage 3 first")
    d = np.load(p)
    return {k: d[k] for k in d.files}


def tune_eps(model, val, meta):
    """Pick eps_cost so the recommended order count best matches the observed one.

    Only the cost slack moves, per the paper: "if cost is a softer constraint,
    setting eps_cost > 0 is an intuitive way to specify this preference".
    """
    target = int(val["action"].sum())
    rows = []
    best = (None, None)
    for e in cfg.EPS_GRID:
        eps = np.zeros(len(cfg.REWARD_DIMS))
        eps[cfg.REWARD_DIMS.index("neg_r_cost")] = e
        rec = model.collapse(val["state"], eps)
        n_rec = int(rec.sum())
        gap = abs(n_rec - target)
        rows.append({"eps_cost": e, "n_recommended": n_rec, "n_observed": target,
                     "abs_gap": gap, "order_rate": float(rec.mean())})
        if best[1] is None or gap < best[1]:
            best = (e, gap)
    return best[0], rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", required=True)
    ap.add_argument("--iterations", type=int, default=cfg.FQI_ITERATIONS)
    ap.add_argument("--sample-per-iter", type=int, default=cfg.FQI_SAMPLE_PER_ITER)
    args = ap.parse_args()

    cfg.ensure_dirs()
    lab = args.lab
    meta = json.loads((cfg.RL_DIR / "meta.json").read_text())
    train = load_split(lab, "train")
    val = load_split(lab, "val")

    print(f"{lab}: train n={len(train['action']):,}  "
          f"observed order rate={train['action'].mean():.4f}  "
          f"state dim={train['state'].shape[1]}")

    print(f"\n[1/4] MO-FQI, {args.iterations} iterations, gamma={cfg.GAMMA}")
    t0 = time.time()
    model = mofqi.MOFittedQ(n_actions=2, n_dims=len(cfg.REWARD_DIMS))
    model.fit(train, iterations=args.iterations, gamma=cfg.GAMMA,
              sample_per_iter=args.sample_per_iter)
    print(f"  done in {time.time() - t0:.1f}s")

    print("\n[2/4] tuning eps_cost on val (Eq. 7)")
    eps_cost, eps_rows = tune_eps(model, val, meta)
    for r in eps_rows:
        mark = " <-" if r["eps_cost"] == eps_cost else ""
        print(f"  eps_cost={r['eps_cost']:<5} recommended={r['n_recommended']:6,} "
              f"observed={r['n_observed']:6,} gap={r['abs_gap']:6,}{mark}")
    eps = np.zeros(len(cfg.REWARD_DIMS))
    eps[cfg.REWARD_DIMS.index("neg_r_cost")] = eps_cost

    print("\n[3/4] fitting the policy function")
    y_train = model.collapse(train["state"], eps)
    if y_train.sum() == 0 or y_train.sum() == len(y_train):
        print(f"  WARNING: collapsed action is constant ({int(y_train.sum())} of "
              f"{len(y_train)} are orders). The classifier below is degenerate.")
        policy = None
        importances = None
    else:
        policy = ExtraTreesClassifier(
            n_estimators=cfg.FQI_N_TREES, min_samples_leaf=cfg.FQI_MIN_SAMPLES_LEAF,
            n_jobs=cfg.FQI_N_JOBS, random_state=cfg.SEED)
        policy.fit(train["state"], y_train)
        importances = dict(zip(meta["state_cols"],
                               policy.feature_importances_.tolist()))
        top = sorted(importances.items(), key=lambda kv: -kv[1])[:8]
        for name, v in top:
            print(f"  {name:20s} {v:.4f}")

    print("\n[4/4] budget rule on val")
    rec_val = policy.predict(val["state"]) if policy is not None \
        else model.collapse(val["state"], eps)
    rec_budget = mofqi.apply_budget(rec_val, val["stay_id"], val["hour"])
    print(f"  val: policy {int(rec_val.sum()):,} orders -> "
          f"{int(rec_budget.sum()):,} after budget "
          f"(observed {int(val['action'].sum()):,})")

    out = cfg.MODELS_DIR / f"{lab}_mofqi.pkl"
    with open(out, "wb") as fh:
        pickle.dump({"lab": lab, "model": model, "policy": policy, "eps": eps,
                     "eps_cost": eps_cost, "state_cols": meta["state_cols"],
                     "iterations": args.iterations, "history": model.history_},
                    fh)
    print(f"\nsaved -> {out}")

    lines = [
        f"# MO-FQI training: {lab}\n\n",
        f"gamma={cfg.GAMMA} iterations={args.iterations} "
        f"sample_per_iter={args.sample_per_iter} n_trees={cfg.FQI_N_TREES}\n\n",
        f"Train transitions: {len(train['action']):,} "
        f"({train['stay_id'].max() and len(set(train['stay_id'].tolist())):,} stays), "
        f"observed order rate {train['action'].mean():.4f}\n\n",
        "## eps_cost tuning (val)\n\n",
        "| eps_cost | recommended | observed | abs gap | order rate |\n",
        "|---|---|---|---|---|\n",
    ]
    for r in eps_rows:
        mark = " **<-**" if r["eps_cost"] == eps_cost else ""
        lines.append(f"| {r['eps_cost']} | {r['n_recommended']:,} | "
                     f"{r['n_observed']:,} | {r['abs_gap']:,} | "
                     f"{r['order_rate']:.4f}{mark} |\n")

    if importances:
        lines.append("\n## Policy feature importances (Gini, Fig. 2 analogue)\n\n")
        lines.append("| feature | importance |\n|---|---|\n")
        for name, v in sorted(importances.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {name} | {v:.4f} |\n")

    pruned = np.mean([h["pruned_action_frac"] for h in model.history_])
    lines.append(
        f"\n## Pareto pruning\n\nMean fraction of actions pruned per iteration: "
        f"{pruned:.4f}.\n\nWith a binary action space this figure is descriptive "
        f"only: pruning cannot change the Bellman backup, because the "
        f"per-dimension max over the surviving action set equals the max over "
        f"both actions whenever one dominates the other. The same is true of the "
        f"paper's four L=1 policies. Pruning shapes only the action set that "
        f"Eq. 7 collapses.\n")
    (cfg.REPORTS_DIR / f"train_{lab}.md").write_text("".join(lines), encoding="utf-8")
    print(f"wrote report -> {cfg.REPORTS_DIR / f'train_{lab}.md'}")


if __name__ == "__main__":
    main()
