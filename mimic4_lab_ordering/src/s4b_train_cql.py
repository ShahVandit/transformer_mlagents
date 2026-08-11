"""
Stage 4b: Conservative Q-Learning, an OPTIONAL alternative to MO-FQI.

    python run_pipeline.py --learner cql
    python src/s4b_train_cql.py --lab wbc

This is NOT part of Cheng et al. (2019). It trains a different learner on the
same MDP so the two can be compared, and it is opt-in precisely so the
replication in s4_train_mofqi.py stays untouched.

What changes, and what it costs
-------------------------------
CQL is scalar-Q. The paper's 4-vector reward has to be collapsed into a single
number by `config.REWARD_WEIGHTS`. That collapse is the exact thing the paper's
vector-valued Q and Pareto pruning exist to avoid: with a scalarization you have
already chosen a preference among the objectives by hand, and you can no longer
say the policy is multi-objective. Running this arm buys a modern, familiar
learner at the price of the paper's actual contribution. Say so when reporting it.

What CQL adds is conservatism: a penalty pushing down Q-values for actions the
data does not support, so the policy cannot chase value it has no evidence for.
That matters far less here than in a many-action problem. The action space is
binary and both actions are well represented at every hour (order rate 5-8%, so
"do not order" is 92-95%), which leaves little out-of-distribution room for the
Q-function to hallucinate into. Expect the conservative term to be close to
inert; that is a finding worth reporting, not a bug.

What is kept identical to the MO-FQI arm so the comparison is fair: the state,
the action, the reward components, the train/val/test split, the discount, the
Eq. 7-style order-count calibration on val, and the 24-hour budget rule.

Output: models/<lab>_cql.pkl, reports/train_<lab>_cql.md
The saved bundle exposes the same interface as the MO-FQI bundle (a `policy`
object with `.predict(states)`), so stages 5, 6 and 7 read it unchanged.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
import mofqi
from cql_policy import QNet, GreedyQPolicy

N_ACTIONS = 2


def load_split(lab, split):
    p = cfg.RL_DIR / f"{lab}_{split}.npz"
    if not p.exists():
        raise SystemExit(f"{p} not found; run stage 3 first")
    d = np.load(p)
    return {k: d[k] for k in d.files}


def scalarize(reward, weights=None):
    """Collapse the 4-vector reward to the single number CQL needs.

    This is the step that forfeits the multi-objective claim. It is isolated in
    one function so it is obvious where the preference is being imposed.
    """
    w = weights or cfg.REWARD_WEIGHTS
    vec = np.array([w[d] for d in cfg.REWARD_DIMS], dtype=np.float64)
    return (reward.astype(np.float64) * vec).sum(axis=1).astype(np.float32)


def train_cql(train, steps=cfg.CQL_STEPS, alpha=cfg.CQL_ALPHA, gamma=cfg.GAMMA,
              verbose_every=cfg.CQL_EVAL_EVERY):
    """Discrete CQL: TD loss with a double-DQN target, plus the conservative term.

        conservative = logsumexp_a Q(s,a) - Q(s, a_behavior)

    which pushes down every action's value and pulls back up only the one the
    clinician actually took, so unsupported actions cannot look attractive.
    """
    torch.manual_seed(cfg.SEED)
    mu = train["state"].mean(axis=0)
    sd = train["state"].std(axis=0)
    sd[sd < 1e-6] = 1.0

    s = torch.tensor((train["state"] - mu) / sd, dtype=torch.float32)
    s2 = torch.tensor((train["next_state"] - mu) / sd, dtype=torch.float32)
    a = torch.tensor(train["action"], dtype=torch.long)
    r = torch.tensor(scalarize(train["reward"]), dtype=torch.float32)
    done = torch.tensor(train["done"], dtype=torch.float32)

    net = QNet(s.shape[1])
    tgt = QNet(s.shape[1])
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=cfg.CQL_LR)

    n = len(a)
    rng = np.random.default_rng(cfg.SEED)
    history = []
    for step in range(1, steps + 1):
        idx = torch.tensor(rng.integers(0, n, cfg.CQL_BATCH), dtype=torch.long)
        with torch.no_grad():
            a_next = net(s2[idx]).argmax(dim=1)                  # double DQN
            q_next = tgt(s2[idx]).gather(1, a_next[:, None]).squeeze(1)
            target = r[idx] + gamma * (1 - done[idx]) * q_next

        q_all = net(s[idx])
        q_sa = q_all.gather(1, a[idx][:, None]).squeeze(1)
        td = F.mse_loss(q_sa, target)
        conservative = (torch.logsumexp(q_all, dim=1) - q_sa).mean()
        loss = td + alpha * conservative

        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for p, tp in zip(net.parameters(), tgt.parameters()):
                tp.data.mul_(1 - cfg.CQL_TARGET_TAU).add_(cfg.CQL_TARGET_TAU * p.data)

        if step % verbose_every == 0 or step == 1:
            rec = {"step": step, "loss": float(loss), "td_loss": float(td),
                   "conservative_loss": float(conservative),
                   "q_mean": float(q_all.mean())}
            history.append(rec)
            print(f"    step {step:6d}/{steps}  loss={rec['loss']:8.4f}  "
                  f"td={rec['td_loss']:8.4f}  cons={rec['conservative_loss']:7.4f}  "
                  f"q_mean={rec['q_mean']:7.4f}", flush=True)

    return net, mu, sd, history


def tune_bias(policy, val):
    """Calibrate the order count on val, mirroring the paper's eps_cost step.

    CQL has no eps_cost, but the same problem exists: the greedy argmax gives
    whatever order rate it gives. A scalar bias is added to Q(order) and bisected
    until the recommended count matches the clinician's, so the two arms are
    compared at a comparable ordering volume rather than at whatever rate each
    happens to produce.
    """
    target = int(val["action"].sum())
    q = policy.q_values(val["state"])
    gap = q[:, 0] - q[:, 1]          # order iff bias > gap

    def n_orders(b):
        return int((gap < b).sum())

    lo, hi = float(gap.min()) - 1e-6, float(gap.max()) + 1e-6
    for _ in range(cfg.EPS_BISECT_STEPS):
        mid = 0.5 * (lo + hi)
        if n_orders(mid) < target:
            lo = mid
        else:
            hi = mid
    best = min([lo, hi], key=lambda b: abs(n_orders(b) - target))
    return float(best), n_orders(best), target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", required=True)
    ap.add_argument("--steps", type=int, default=cfg.CQL_STEPS)
    ap.add_argument("--alpha", type=float, default=cfg.CQL_ALPHA)
    ap.add_argument("--no-calibrate", action="store_true",
                    help="skip the order-count calibration and use the raw argmax")
    args = ap.parse_args()

    cfg.ensure_dirs()
    lab = args.lab
    meta = json.loads((cfg.RL_DIR / "meta.json").read_text())
    train = load_split(lab, "train")
    val = load_split(lab, "val")

    w = ", ".join(f"{k}={v}" for k, v in cfg.REWARD_WEIGHTS.items())
    print(f"{lab}: train n={len(train['action']):,}  "
          f"observed order rate={train['action'].mean():.4f}  "
          f"state dim={train['state'].shape[1]}")
    print(f"  scalarized reward weights: {w}")
    print(f"  NOTE: scalarizing forfeits the paper's multi-objective claim.")

    print(f"\n[1/3] CQL, {args.steps:,} steps, alpha={args.alpha}, gamma={cfg.GAMMA}")
    t0 = time.time()
    net, mu, sd, history = train_cql(train, steps=args.steps, alpha=args.alpha)
    print(f"  done in {time.time() - t0:.1f}s")

    policy = GreedyQPolicy(net, mu, sd)
    raw_rate = float(policy.predict(val["state"]).mean())
    print(f"\n[2/3] order-count calibration on val")
    print(f"  raw argmax order rate: {raw_rate:.4f}  "
          f"(clinician {val['action'].mean():.4f})")
    if args.no_calibrate:
        bias, n_rec, target = 0.0, int(policy.predict(val["state"]).sum()), \
            int(val["action"].sum())
    else:
        bias, n_rec, target = tune_bias(policy, val)
        policy.bias = bias
    print(f"  bias={bias:+.5f}  recommended={n_rec:,}  observed={target:,}  "
          f"gap={abs(n_rec - target):,}")

    print("\n[3/3] budget rule on val")
    rec_val = policy.predict(val["state"])
    rec_budget = mofqi.apply_budget(rec_val, val["stay_id"], val["hour"])
    print(f"  val: policy {int(rec_val.sum()):,} orders -> "
          f"{int(rec_budget.sum()):,} after budget "
          f"(observed {int(val['action'].sum()):,})")

    out = cfg.MODELS_DIR / f"{lab}_cql.pkl"
    with open(out, "wb") as fh:
        pickle.dump({"lab": lab, "learner": "cql", "model": None, "policy": policy,
                     "eps": None, "bias": bias, "alpha": args.alpha,
                     "steps": args.steps, "state_cols": meta["state_cols"],
                     "reward_weights": dict(cfg.REWARD_WEIGHTS),
                     "history": history}, fh)
    print(f"\nsaved -> {out}")

    L = [
        f"# CQL training: {lab}\n\n",
        "**Not part of Cheng et al. (2019).** An alternative learner on the same "
        "MDP, run with `--learner cql`.\n\n",
        f"`alpha={args.alpha}` `steps={args.steps:,}` `gamma={cfg.GAMMA}` "
        f"`lr={cfg.CQL_LR}` `batch={cfg.CQL_BATCH}`\n\n",
        "## Scalarized reward\n\n",
        "CQL is scalar-Q, so the 4-vector reward is collapsed by a fixed weighting:\n\n",
        "| component | weight |\n|---|---|\n",
    ]
    for k, v in cfg.REWARD_WEIGHTS.items():
        L.append(f"| {k} | {v} |\n")
    L.append(
        "\nThis is the step the paper's vector-valued Q and Pareto pruning exist "
        "to avoid. With a scalarization the preference among objectives is fixed "
        "by hand up front, so results from this arm are not a multi-objective "
        "claim and should not be presented as one.\n")

    L += ["\n## Training\n\n",
          "| step | loss | td_loss | conservative_loss | q_mean |\n",
          "|---|---|---|---|---|\n"]
    for h in history:
        L.append(f"| {h['step']:,} | {h['loss']:.4f} | {h['td_loss']:.4f} | "
                 f"{h['conservative_loss']:.4f} | {h['q_mean']:.4f} |\n")
    L.append(
        "\n`conservative_loss` is `logsumexp_a Q(s,a) - Q(s,a_behavior)`. With a "
        "binary action space and both actions well represented at every hour, "
        "there is little out-of-distribution room for the Q-function to "
        "hallucinate into, so this term is expected to stay small. A near-inert "
        "conservative penalty here is a property of the action space, not a "
        "failure of the implementation.\n")

    L += [f"\n## Order-count calibration (val)\n\n",
          f"Raw argmax order rate {raw_rate:.4f} against a clinician rate of "
          f"{val['action'].mean():.4f}. A bias of {bias:+.5f} is added to "
          f"`Q(order)` and bisected until the counts match ({n_rec:,} vs "
          f"{target:,}), which is the CQL analogue of the paper's `eps_cost` "
          f"tuning and puts both arms at a comparable ordering volume.\n"]

    (cfg.REPORTS_DIR / f"train_{lab}_cql.md").write_text("".join(L), encoding="utf-8")
    print(f"wrote report -> {cfg.REPORTS_DIR / f'train_{lab}_cql.md'}")


if __name__ == "__main__":
    main()
