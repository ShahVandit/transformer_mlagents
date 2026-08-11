"""
Stage 4c: train a family of joint-panel CQL policies.

Each policy uses the same two raw objectives and the same normalized reward:

    r_lambda = z_detection - lambda * z_burden

Output: models/joint_cql_lam{lambda}.pkl
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
from cql_policy import GreedyQPolicy, QNet


N_ACTIONS = len(cfg.JOINT_PANEL_BITS)


def lam_slug(lam):
    return str(lam).replace(".", "p")


def load_split(split):
    p = cfg.RL_DIR / f"joint_{split}.npz"
    if not p.exists():
        raise SystemExit(f"{p} not found; run stage 3b first")
    d = np.load(p)
    return {k: d[k] for k in d.files}


def scalar_reward(split, lam):
    r = split["reward_norm"].astype(np.float32)
    return (r[:, 0] - float(lam) * r[:, 1]).astype(np.float32)


def train_cql(train, lam, steps=cfg.CQL_STEPS, alpha=cfg.CQL_ALPHA,
              gamma=cfg.GAMMA, verbose_every=cfg.CQL_EVAL_EVERY):
    torch.manual_seed(cfg.SEED)
    mu = train["state"].mean(axis=0)
    sd = train["state"].std(axis=0)
    sd[sd < 1e-6] = 1.0

    s = torch.tensor((train["state"] - mu) / sd, dtype=torch.float32)
    s2 = torch.tensor((train["next_state"] - mu) / sd, dtype=torch.float32)
    a = torch.tensor(train["action"], dtype=torch.long)
    r = torch.tensor(scalar_reward(train, lam), dtype=torch.float32)
    done = torch.tensor(train["done"], dtype=torch.float32)

    net = QNet(s.shape[1], n_actions=N_ACTIONS)
    tgt = QNet(s.shape[1], n_actions=N_ACTIONS)
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=cfg.CQL_LR)

    n = len(a)
    rng = np.random.default_rng(cfg.SEED)
    history = []
    for step in range(1, steps + 1):
        idx = torch.tensor(rng.integers(0, n, cfg.CQL_BATCH), dtype=torch.long)
        with torch.no_grad():
            a_next = net(s2[idx]).argmax(dim=1)
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
            rec = {
                "step": step,
                "loss": float(loss),
                "td_loss": float(td),
                "conservative_loss": float(conservative),
                "q_mean": float(q_all.mean()),
            }
            history.append(rec)
            print(f"    step {step:6d}/{steps}  loss={rec['loss']:8.4f}  "
                  f"td={rec['td_loss']:8.4f}  cons={rec['conservative_loss']:7.4f}  "
                  f"q_mean={rec['q_mean']:7.4f}", flush=True)
    return net, mu, sd, history


def summarize_policy(policy, split):
    a = policy.predict(split["state"])
    any_draw = a != 0
    days = max(1e-6, len(a) / 24.0)
    return {
        "draw_rate": float(any_draw.mean()),
        "draws_per_patient_day": float(any_draw.sum() / days),
        "action_counts": {str(i): int((a == i).sum()) for i in range(N_ACTIONS)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", nargs="+", type=float, default=cfg.JOINT_LAMBDAS)
    ap.add_argument("--steps", type=int, default=cfg.CQL_STEPS)
    ap.add_argument("--alpha", type=float, default=cfg.CQL_ALPHA)
    args = ap.parse_args()

    cfg.ensure_dirs()
    meta = json.loads((cfg.RL_DIR / "joint_meta.json").read_text())
    train = load_split("train")
    val = load_split("val")

    print(f"joint CQL family: train n={len(train['action']):,}, "
          f"state dim={train['state'].shape[1]}, actions={N_ACTIONS}")
    print("reward: z_detection - lambda * z_burden")

    rows = []
    for lam in args.lambdas:
        print(f"\n[lambda={lam}] CQL, {args.steps:,} steps, alpha={args.alpha}")
        t0 = time.time()
        net, mu, sd, history = train_cql(train, lam, steps=args.steps, alpha=args.alpha)
        policy = GreedyQPolicy(net, mu, sd)
        val_summary = summarize_policy(policy, val)
        print(f"  done in {time.time() - t0:.1f}s")
        print(f"  val draws/patient-day={val_summary['draws_per_patient_day']:.3f}  "
              f"draw rate={val_summary['draw_rate']:.4f}")

        bundle = {
            "track": "joint",
            "learner": "cql",
            "lambda": float(lam),
            "policy": policy,
            "n_actions": N_ACTIONS,
            "panel_bits": cfg.JOINT_PANEL_BITS,
            "panel_names": cfg.JOINT_PANEL_NAMES,
            "reward_dims": cfg.JOINT_REWARD_DIMS,
            "state_cols": meta["state_cols"],
            "alpha": args.alpha,
            "steps": args.steps,
            "history": history,
            "val_summary": val_summary,
        }
        out = cfg.MODELS_DIR / f"joint_cql_lam{lam_slug(lam)}.pkl"
        with open(out, "wb") as fh:
            pickle.dump(bundle, fh)
        print(f"  saved -> {out}")
        rows.append({"lambda": float(lam), **val_summary})

    report = ["# Joint CQL policy family\n\n",
              "| lambda | val draws/patient-day | val draw rate |\n",
              "|---|---:|---:|\n"]
    for row in rows:
        report.append(f"| {row['lambda']} | {row['draws_per_patient_day']:.3f} | "
                      f"{row['draw_rate']:.4f} |\n")
    out_report = cfg.REPORTS_DIR / "train_joint_cql_family.md"
    out_report.write_text("".join(report), encoding="utf-8")
    print(f"\nwrote report -> {out_report}")


if __name__ == "__main__":
    main()
