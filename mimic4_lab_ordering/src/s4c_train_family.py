"""
Stage 4c: train a family of joint-panel CQL policies with d3rlpy.

Each policy uses the same two raw objectives and the same normalized reward:

    r_w = w_detection * z_detection - w_burden * z_burden

Output:
  models/joint_cql_w{w_det}_{w_bur}.d3
  models/joint_cql_w{w_det}_{w_bur}.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "d3rlpy"))

import config as cfg
import objectives


N_ACTIONS = len(cfg.JOINT_PANEL_BITS)


def lam_slug(lam):
    """Retained for the older lambda-named artefacts; new runs use pref_slug."""
    return str(lam).replace(".", "p")


def clean_tag(tag):
    tag = str(tag or "").strip()
    if not tag:
        return ""
    keep = [c if c.isalnum() or c in ("-", "_") else "_" for c in tag]
    return "_" + "".join(keep)


def load_split(split):
    p = cfg.RL_DIR / f"joint_{split}.npz"
    if not p.exists():
        raise SystemExit(f"{p} not found; run stage 3b first")
    d = np.load(p)
    return {k: d[k] for k in d.files}


def pref_slug(pref):
    """Filename-safe tag for a preference, e.g. (0.7, 0.3) -> w0p7_0p3."""
    return "w" + "_".join(str(round(float(x), 3)).replace(".", "p") for x in pref)


def scalar_reward(split, pref):
    """w_detection * z_detection - w_burden * z_burden.

    Weights lie on the simplex so |r| stays roughly constant across the sweep;
    see config.JOINT_PREFERENCES for why that matters to CQL_ALPHA.
    """
    w_det, w_bur = float(pref[0]), float(pref[1])
    r = split["reward_norm"].astype(np.float32)
    return (w_det * r[:, 0] - w_bur * r[:, 1]).astype(np.float32)


def scalar_policy_reward(split, actions, pref, norm_meta):
    """Recompute both objectives UNDER `actions`, then apply the preference.

    This is a REPLAY quantity: the policy's draws are scored against logged
    states, so a recommended draw never actually updates the patient. It is not
    an OPE estimate and must not be read as one.
    """
    det = objectives.detection_objective(split, actions)[0]
    bur = objectives.burden_objective(split, actions)
    raw = np.stack([det, bur], axis=1).astype(np.float32)
    mu = np.asarray(norm_meta["reward_mean"], dtype=np.float32)
    sd = np.asarray(norm_meta["reward_sd"], dtype=np.float32)
    sd[sd < 1e-6] = 1.0
    r = (raw - mu) / sd
    w_det, w_bur = float(pref[0]), float(pref[1])
    return (w_det * r[:, 0] - w_bur * r[:, 1]).astype(np.float32)


def per_stay_sum(values, stay):
    out, start = [], 0
    for i in range(1, len(stay) + 1):
        if i == len(stay) or stay[i] != stay[start]:
            out.append(float(np.sum(values[start:i])))
            start = i
    return np.asarray(out, dtype=np.float64)


def constant_policy_returns(split, norm_meta):
    """Per-stay mean of normalized (detection, burden) for every CONSTANT policy.

    One entry for never-draw and one for each of the seven non-empty panels.
    "Always draw" is not a single baseline: `np.full(n, 1)` means "always order
    creatinine+bun+wbc", which is one arbitrary panel out of seven and is not
    the strongest constant policy at every preference. The gate compares against
    the BEST constant panel, so a learned policy has to beat the best fixed rule
    rather than a convenient one.

    Both objectives are computed once per constant action and cached. The
    preference only enters as a linear combination afterwards, and expectation is
    linear, so weighting the cached means is exact and avoids recomputing the
    objectives for every preference.
    """
    n = len(split["action"])
    stay = split["stay_id"]
    mu = np.asarray(norm_meta["reward_mean"], dtype=np.float32)
    sd = np.asarray(norm_meta["reward_sd"], dtype=np.float32)
    sd = np.where(sd < 1e-6, 1.0, sd)

    out = {}
    actions = [("never_draw", np.zeros(n, dtype=np.int64))]
    for a in range(1, len(cfg.JOINT_PANEL_BITS)):
        actions.append((f"always_{cfg.JOINT_PANEL_BITS[a]}",
                        np.full(n, a, dtype=np.int64)))
    for name, acts in actions:
        det = objectives.detection_objective(split, acts)[0]
        bur = objectives.burden_objective(split, acts)
        zd = (det - mu[0]) / sd[0]
        zb = (bur - mu[1]) / sd[1]
        out[name] = (float(per_stay_sum(zd, stay).mean()),
                     float(per_stay_sum(zb, stay).mean()))
    return out


def trivial_baselines(split, pref, norm_meta, cache=None):
    """Constant-policy baselines scored under one preference.

    A policy that cannot beat the best of these on its OWN objective has not
    learned a trade-off, it has diverged. Without this check a failed run is
    indistinguishable from a preference that genuinely wants more testing, which
    is exactly what happened at lambda >= 0.3.
    """
    cache = constant_policy_returns(split, norm_meta) if cache is None else cache
    w_det, w_bur = float(pref[0]), float(pref[1])
    return {name: w_det * d - w_bur * b for name, (d, b) in cache.items()}


def make_dataset(split, reward):
    from d3rlpy.constants import ActionSpace
    from d3rlpy.dataset import MDPDataset

    return MDPDataset(
        observations=split["state"].astype(np.float32),
        actions=split["action"].astype(np.int64),
        rewards=np.asarray(reward, dtype=np.float32).reshape(-1, 1),
        terminals=split["done"].astype(np.float32),
        action_space=ActionSpace.DISCRETE,
        action_size=N_ACTIONS,
    )


def make_cql(device, alpha):
    import d3rlpy
    from d3rlpy.preprocessing import StandardObservationScaler

    return d3rlpy.algos.DiscreteCQLConfig(
        learning_rate=cfg.CQL_LR,
        batch_size=cfg.CQL_BATCH,
        gamma=cfg.GAMMA,
        alpha=alpha,
        target_update_interval=cfg.CQL_EVAL_EVERY,
        observation_scaler=StandardObservationScaler(),
    ).create(device=device)


def summarize_policy(algo, split):
    actions = algo.predict(split["state"].astype(np.float32)).astype(np.int64)
    any_draw = actions != 0
    days = max(1e-6, len(actions) / 24.0)
    return {
        "draw_rate": float(any_draw.mean()),
        "draws_per_patient_day": float(any_draw.sum() / days),
        "replay_detection_mean": float(split["reward"][:, 0][any_draw].mean()
                                       if any_draw.any() else 0.0),
        "replay_burden_mean": float(split["reward"][:, 1][any_draw].mean()
                                    if any_draw.any() else 0.0),
        "action_counts": {str(i): int((actions == i).sum()) for i in range(N_ACTIONS)},
    }


def summarize_policy_with_pref(algo, split, pref, norm_meta, baselines=None):
    actions = algo.predict(split["state"].astype(np.float32)).astype(np.int64)
    any_draw = actions != 0
    stay = split["stay_id"]
    scalar = scalar_policy_reward(split, actions, pref, norm_meta)
    clinician_scalar = scalar_policy_reward(
        split, split["action"].astype(np.int64), pref, norm_meta)
    scalar_stay = per_stay_sum(scalar, stay)
    clinician_scalar_stay = per_stay_sum(clinician_scalar, stay)

    event = split["event"].astype(bool)
    covered = np.zeros(len(event), dtype=bool)
    start = 0
    lookback = cfg.JOINT_DETECTION_LOOKAHEAD_HOURS
    for i in range(1, len(event) + 1):
        if i == len(event) or stay[i] != stay[start]:
            d = any_draw[start:i].astype(np.int8)
            c = np.r_[0, np.cumsum(d)]
            local = np.zeros(i - start, dtype=bool)
            for j in np.flatnonzero(event[start:i]):
                lo = max(0, j - lookback)
                local[j] = (c[j] - c[lo]) > 0
            covered[start:i] = local
            start = i

    base = baselines or {}
    best_trivial = max(base.values()) if base else -np.inf
    policy_rew = float(np.mean(scalar_stay))
    return {
        "ep_rew_mean": policy_rew,
        "ep_rew_std": float(np.std(scalar_stay)),
        "clin_ep_rew_mean": float(np.mean(clinician_scalar_stay)),
        "rew_gap": policy_rew - float(np.mean(clinician_scalar_stay)),
        "never_draw_rew": base.get("never_draw", np.nan),
        "best_constant_rew": best_trivial if base else np.nan,
        "best_constant_name": (max(base, key=base.get) if base else None),
        "beats_trivial": bool(policy_rew > best_trivial + cfg.VALIDITY_MARGIN)
                         if base else None,
        "draws_per_patient_day": float(any_draw.sum() / max(1e-6, len(actions) / 24.0)),
        "clin_draws_per_patient_day": float((split["action"] != 0).sum() / max(1e-6, len(actions) / 24.0)),
        "event_coverage": float(covered[event].mean() if event.any() else 0.0),
        "action_counts": {str(i): int((actions == i).sum()) for i in range(N_ACTIONS)},
    }


def make_epoch_callback(pref, val, norm_meta, rows, baselines):
    def _callback(algo, epoch, total_step):
        s = summarize_policy_with_pref(algo, val, pref, norm_meta, baselines)
        rec = {"epoch": int(epoch), "step": int(total_step), **s}
        rows.append(rec)
        flag = "" if s["beats_trivial"] else "   <== BELOW BEST CONSTANT POLICY"
        print(
            f"  val epoch={epoch} step={total_step} "
            f"draws/day={s['draws_per_patient_day']:.2f} "
            f"(clin {s['clin_draws_per_patient_day']:.2f})  "
            f"coverage={s['event_coverage']:.3f}  "
            f"rew={s['ep_rew_mean']:+.2f} "
            f"(clin {s['clin_ep_rew_mean']:+.2f}, "
            f"never {s['never_draw_rew']:+.2f}, "
            f"best-const {s['best_constant_rew']:+.2f} "
            f"[{s['best_constant_name']}])"
            f"{flag}",
            flush=True,
        )
    return _callback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefs", nargs="+", type=float, default=None,
                    help="flat list of w_detection w_burden pairs, e.g. 0.9 0.1 0.5 0.5")
    ap.add_argument("--steps", type=int, default=cfg.CQL_STEPS)
    ap.add_argument("--alpha", type=float, default=cfg.CQL_ALPHA)
    ap.add_argument("--tag", default="",
                    help="optional filename suffix, e.g. a0p1, to avoid overwriting runs")
    ap.add_argument("--device", default=False,
                    help="d3rlpy device argument: False, cpu, cuda:0, or GPU id")
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()

    cfg.ensure_dirs()
    import d3rlpy
    d3rlpy.seed(cfg.SEED)

    meta = json.loads((cfg.RL_DIR / "joint_meta.json").read_text())
    norm_meta = meta["reward_normalization"]
    train = load_split("train")
    val = load_split("val")

    if args.prefs:
        flat = list(args.prefs)
        if len(flat) % 2:
            raise SystemExit("--prefs needs an even count: w_detection w_burden pairs")
        prefs = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
    else:
        prefs = [tuple(x) for x in cfg.JOINT_PREFERENCES]

    print(f"joint d3rlpy CQL family: train n={len(train['action']):,}, "
          f"state dim={train['state'].shape[1]}, actions={N_ACTIONS}")
    print("reward: w_detection * z_detection - w_burden * z_burden")
    print(f"preferences: {prefs}")

    print(f"\ncaching constant-policy baselines on val "
          f"({len(cfg.JOINT_PANEL_BITS)} constant actions)")
    base_cache = constant_policy_returns(val, norm_meta)

    rows = []
    for pref in prefs:
        print(f"\n[w_det={pref[0]}, w_bur={pref[1]}] d3rlpy DiscreteCQL, "
              f"{args.steps:,} steps, alpha={args.alpha}")
        base = trivial_baselines(val, pref, norm_meta, base_cache)
        best_name = max(base, key=base.get)
        print(f"  constant baselines on val: never_draw={base['never_draw']:+.3f}  "
              f"best={best_name}={base[best_name]:+.3f}")
        reward = scalar_reward(train, pref)
        dataset = make_dataset(train, reward)
        algo = make_cql(args.device, args.alpha)
        epoch_rows = []
        cb = make_epoch_callback(pref, val, norm_meta, epoch_rows, base)

        t0 = time.time()
        history = algo.fit(
            dataset,
            n_steps=args.steps,
            n_steps_per_epoch=cfg.CQL_EVAL_EVERY,
            experiment_name=f"joint_cql_{pref_slug(pref)}",
            with_timestamp=False,
            show_progress=args.progress,
            save_interval=max(1, args.steps // max(1, cfg.CQL_EVAL_EVERY)),
            epoch_callback=cb,
        )
        elapsed = time.time() - t0

        val_summary = summarize_policy_with_pref(algo, val, pref, norm_meta, base)
        verdict = "VALID" if val_summary["beats_trivial"] else "REJECTED (below trivial)"
        print(f"  done in {elapsed:.1f}s")
        print(f"  val draws/day={val_summary['draws_per_patient_day']:.3f} "
              f"(clin {val_summary['clin_draws_per_patient_day']:.3f})  "
              f"coverage={val_summary['event_coverage']:.3f}")
        print(f"  val rew={val_summary['ep_rew_mean']:+.3f}  "
              f"clin={val_summary['clin_ep_rew_mean']:+.3f}  "
              f"never={base['never_draw']:+.3f}  "
              f"best-const={val_summary['best_constant_rew']:+.3f} "
              f"[{val_summary['best_constant_name']}]  -> {verdict}")

        stem = cfg.MODELS_DIR / f"joint_cql_{pref_slug(pref)}{clean_tag(args.tag)}"
        model_path = stem.with_suffix(".d3")
        meta_path = stem.with_suffix(".json")
        algo.save(str(model_path))
        payload = {
            "track": "joint",
            "library": "d3rlpy",
            "learner": "DiscreteCQL",
            "model_path": str(model_path),
            "preference": [float(pref[0]), float(pref[1])],
            "lambda_equivalent": float(pref[1] / pref[0]) if pref[0] else float("inf"),
            "constant_baselines": base,
            "best_constant_name": val_summary["best_constant_name"],
            "beats_trivial": val_summary["beats_trivial"],
            "n_actions": N_ACTIONS,
            "panel_bits": cfg.JOINT_PANEL_BITS,
            "panel_names": cfg.JOINT_PANEL_NAMES,
            "reward_dims": cfg.JOINT_REWARD_DIMS,
            "state_cols": meta["state_cols"],
            "alpha": args.alpha,
            "steps": args.steps,
            "history": [(int(e), {k: float(v) for k, v in m.items()})
                        for e, m in history],
            "epoch_validation": epoch_rows,
            "val_summary": val_summary,
        }
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  saved -> {model_path}")
        rows.append({"w_detection": float(pref[0]), "w_burden": float(pref[1]),
                     **val_summary})

    n_valid = sum(1 for r in rows if r.get("beats_trivial"))
    report = [
        "# Joint d3rlpy CQL policy family\n\n",
        "Reward is `w_detection * z_detection - w_burden * z_burden`, weights on "
        "the simplex.\n\n",
        "Every number below is a **replay** quantity on val: the policy's actions "
        "are scored against logged states, so a recommended draw never updates "
        "the patient. These are exact behavioural metrics, not off-policy "
        "estimates. Counterfactual values come from stage 8.\n\n",
        f"`valid` means the policy beat the BEST CONSTANT policy, never-draw or "
        f"any single fixed panel, on its own weighted objective. "
        f"{n_valid}/{len(rows)} "
        f"passed. A policy below the best constant rule has not found a trade-off, "
        f"it has diverged, and must not be plotted as a frontier point.\n\n",
        "| w_det | w_bur | draws/day | clin draws/day | coverage | rew | clin rew | never | always | valid |\n",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n",
    ]
    for row in rows:
        report.append(f"| {row['w_detection']:.1f} | {row['w_burden']:.1f} | "
                      f"{row['draws_per_patient_day']:.3f} | "
                      f"{row['clin_draws_per_patient_day']:.3f} | "
                      f"{row['event_coverage']:.3f} | "
                      f"{row['ep_rew_mean']:+.3f} | "
                      f"{row['clin_ep_rew_mean']:+.3f} | "
                      f"{row['never_draw_rew']:+.3f} | "
                      f"{row['best_constant_rew']:+.3f} ({row['best_constant_name']}) | "
                      f"{'yes' if row.get('beats_trivial') else '**NO**'} |\n")
    out_report = cfg.REPORTS_DIR / "train_joint_cql_family.md"
    out_report.write_text("".join(report), encoding="utf-8")
    print(f"\nwrote report -> {out_report}")


if __name__ == "__main__":
    main()
