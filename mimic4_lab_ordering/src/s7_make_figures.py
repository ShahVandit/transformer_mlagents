"""
Stage 7: figures, as analogues of the paper's Figures 2-6.

  fig2_feature_importance.png  Gini importances of each lab's policy function
  fig3_trajectory_<lab>.png    one test stay: lab trace, SOFA, orders vs recs
  fig4_ope_values.png          V_d per policy per reward component, with spread
  fig5_information_gain.png    information gain per order, clinician vs MO-FQI
  fig6_time_to_treatment.png   lead time from order to intervention onset

Every panel reads artefacts the earlier stages wrote, so this stage is cheap to
re-run while iterating on presentation.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
import itemids as ids
import mofqi

plt.rcParams.update({"figure.dpi": 130, "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False})


def _have(path):
    return Path(path).exists()


def fig2_feature_importance(labs):
    bundles = {}
    for lab in labs:
        p = cfg.MODELS_DIR / f"{lab}_mofqi.pkl"
        if not _have(p):
            continue
        with open(p, "rb") as fh:
            b = pickle.load(fh)
        if b["policy"] is not None:
            bundles[lab] = (b["state_cols"], b["policy"].feature_importances_)
    if not bundles:
        return
    n = len(bundles)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (lab, (cols, imp)) in zip(axes, bundles.items()):
        order = np.argsort(imp)[::-1][:10][::-1]
        ax.barh(np.arange(len(order)), imp[order], color="#4878A6")
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels([cols[i] for i in order], fontsize=6)
        ax.set_title(lab)
        ax.set_xlabel("Gini importance")
    fig.suptitle("Policy feature importances (paper Fig. 2)", fontsize=9)
    fig.tight_layout()
    fig.savefig(cfg.FIGURES_DIR / "fig2_feature_importance.png")
    plt.close(fig)


def fig3_trajectory(lab):
    p = cfg.RL_DIR / f"{lab}_test.npz"
    mp = cfg.MODELS_DIR / f"{lab}_mofqi.pkl"
    if not (_have(p) and _have(mp)):
        return
    d = {k: v for k, v in np.load(p).items()}
    with open(mp, "rb") as fh:
        b = pickle.load(fh)
    rec = (b["policy"].predict(d["state"]) if b["policy"] is not None
           else b["model"].collapse(d["state"], b["eps"])).astype(int)
    budget = mofqi.apply_budget(rec, d["stay_id"], d["hour"])

    # pick the stay with the most clinician orders, so the panel has content
    stays, counts = np.unique(d["stay_id"][d["action"] == 1], return_counts=True)
    if not len(stays):
        return
    s = stays[np.argmax(counts)]
    sel = d["stay_id"] == s
    h = d["hour"][sel]
    sofa_col = b["state_cols"].index("sofa")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 4.2), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(h, d["mean_lab"][sel], color="#4878A6", lw=1.0, label="forecast m_t")
    ax1.fill_between(h, d["mean_lab"][sel] - d["std_lab"][sel],
                     d["mean_lab"][sel] + d["std_lab"][sel],
                     color="#4878A6", alpha=0.18, lw=0, label="+/- sigma_t")
    obs = d["obs_lab"][sel]
    ax1.scatter(h[~np.isnan(obs)], obs[~np.isnan(obs)], s=14, color="#222",
                zorder=3, label="measured")
    ax1.set_ylabel(lab)
    ax1.legend(fontsize=6, loc="upper right", ncol=2)
    ax1.set_title(f"{lab}: one test stay (paper Fig. 3)", fontsize=9)

    ax2.plot(h, d["state"][sel, sofa_col], color="#B5651D", lw=1.0, label="SOFA")
    for y, mask, c, lbl in [(-0.8, d["action"][sel] == 1, "#222", "clinician order"),
                            (-1.6, rec[sel] == 1, "#4878A6", "MO-FQI"),
                            (-2.4, budget[sel] == 1, "#8FBF6A", "+ budget")]:
        xs = h[mask]
        ax2.scatter(xs, np.full(len(xs), y), s=8, marker="|", color=c, label=lbl)
    ax2.set_ylabel("SOFA / orders")
    ax2.set_xlabel("hours since ICU admission")
    ax2.legend(fontsize=6, loc="upper right", ncol=4)
    fig.tight_layout()
    fig.savefig(cfg.FIGURES_DIR / f"fig3_trajectory_{lab}.png")
    plt.close(fig)


def fig4_ope_values(labs):
    data = {}
    for lab in labs:
        p = cfg.REPORTS_DIR / f"ope_{lab}.json"
        if _have(p):
            data[lab] = json.loads(Path(p).read_text())
    if not data:
        return
    dims = cfg.REWARD_DIMS
    fig, axes = plt.subplots(len(data), len(dims),
                             figsize=(2.6 * len(dims), 2.2 * len(data)),
                             squeeze=False)
    for i, (lab, res) in enumerate(data.items()):
        names = list(res["tier1"])
        for j, dim in enumerate(dims):
            ax = axes[i][j]
            vals = [res["tier1"][n]["mean"][j] for n in names]
            errs = [res["tier1"][n]["std"][j] for n in names]
            colors = ["#B5651D" if n == "MO-FQI" else
                      "#222" if n.startswith("clinician") else "#9AA5B1"
                      for n in names]
            ax.barh(np.arange(len(names)), vals, xerr=errs, color=colors,
                    error_kw={"lw": 0.7})
            ax.set_yticks(np.arange(len(names)))
            ax.set_yticklabels(names if j == 0 else [], fontsize=6)
            if i == 0:
                ax.set_title(dim, fontsize=8)
            if j == 0:
                ax.set_ylabel(lab, fontsize=8)
    fig.suptitle("PS-WIS value per reward component (paper Fig. 4)", fontsize=9)
    fig.tight_layout()
    fig.savefig(cfg.FIGURES_DIR / "fig4_ope_values.png")
    plt.close(fig)


def _clinical(labs):
    out = {}
    for lab in labs:
        p = cfg.RL_DIR / f"{lab}_clinical.npz"
        if _have(p):
            out[lab] = {k: v for k, v in np.load(p).items()}
    return out


def fig5_information_gain(labs):
    data = _clinical(labs)
    if not data:
        return
    fig, axes = plt.subplots(1, len(data), figsize=(3.0 * len(data), 2.8),
                             squeeze=False)
    for ax, (lab, d) in zip(axes[0], data.items()):
        for v, c, lbl in [(d["ig_clin"], "#222", "clinician"),
                          (d["ig_pol"], "#B5651D", "MO-FQI")]:
            v = v[np.isfinite(v)]
            if len(v) == 0:
                continue
            ax.hist(v, bins=30, histtype="step", density=True, color=c, label=lbl)
            ax.axvline(np.mean(v), color=c, ls="--", lw=0.8)
        ax.set_title(f"{lab}", fontsize=8)
        ax.set_xlabel("|smoothed - forecast|")
        ax.legend(fontsize=6)
    fig.suptitle("Information gain per order (paper Fig. 5)", fontsize=9)
    fig.tight_layout()
    fig.savefig(cfg.FIGURES_DIR / "fig5_information_gain.png")
    plt.close(fig)


def fig6_time_to_treatment(labs):
    data = _clinical(labs)
    if not data:
        return
    fig, axes = plt.subplots(1, len(data), figsize=(3.0 * len(data), 2.8),
                             squeeze=False)
    for ax, (lab, d) in zip(axes[0], data.items()):
        for v, c, lbl in [(d["tt_clin"], "#222", "clinician"),
                          (d["tt_pol"], "#B5651D", "MO-FQI")]:
            if len(v) == 0:
                continue
            ax.hist(v, bins=24, histtype="step", density=True, color=c, label=lbl)
            ax.axvline(np.mean(v), color=c, ls="--", lw=0.8)
        ax.set_title(f"{lab}", fontsize=8)
        ax.set_xlabel("hours from order to onset")
        ax.legend(fontsize=6)
    fig.suptitle("Time to treatment onset (paper Fig. 6)", fontsize=9)
    fig.tight_layout()
    fig.savefig(cfg.FIGURES_DIR / "fig6_time_to_treatment.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labs", nargs="+", default=ids.TARGET_LABS)
    args = ap.parse_args()

    cfg.ensure_dirs()
    fig2_feature_importance(args.labs)
    for lab in args.labs:
        fig3_trajectory(lab)
    fig4_ope_values(args.labs)
    fig5_information_gain(args.labs)
    fig6_time_to_treatment(args.labs)

    made = sorted(p.name for p in cfg.FIGURES_DIR.glob("*.png"))
    print(f"wrote {len(made)} figures -> {cfg.FIGURES_DIR}")
    for m in made:
        print(f"  {m}")


if __name__ == "__main__":
    main()
