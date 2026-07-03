"""
Concept -> unit ownership map: contrast labeling + set-level mediation.

Which units of the policy are causally owned by the navigation inputs
(PT dims [86,87,94]: target dir x/z + distance) vs the energy inputs
(dims [88,89,90,95,96]: battery, BS dir x/z, BS distance, urgency gate)?

Stage 1a — SCREEN + LABEL (total effect, per unit):
  do(concept): resample the concept's dims jointly from a random donor window
  (whole 8-step segment -> physically consistent, in-distribution), re-run,
  score S_G(n) = mean |a_G(n) - a_clean(n)| / sigma_n.
  Null: the 95th percentile of the same statistic (same repeats) over
  N_PLACEBO random same-size non-concept dim groups.
  Labels from the CONTRAST:
    PT-owned      : screened for PT only, or both and S_PT > rho * S_E
    energy-owned  : symmetric
    shared        : both screened, no rho dominance
                    (soft w = S_PT / (S_PT + S_E), continuous)
    inert         : neither above its placebo null

Stage 1b — SET-LEVEL mediation validation (not a per-unit gate):
  Per-unit restoration shares are ~1/n when a concept routes through n units
  (measured: PT flows through 100-200 units per boundary in task1_v8, max
  per-unit share 0.005-0.015) — no absolute per-unit threshold is meaningful.
  Instead: patch the concept at the input, restore the ENTIRE owned set's
  clean activations at once (all 8 positions), and measure the fraction of
  the action change (post-tanh mu) undone:
      set_share = E[(D_pat - D_restored) / D_pat]
  against a null of random same-size unit sets. A high set share validates
  that the labeled set collectively carries the concept -> the causal claim
  lives at the set level, matching how the surgery uses the labels.

Surgery units: ffn_hidden.{i} per-neuron; attention heads at attn_in.{i}
(32-dim slices). Other boundaries get Stage-1a maps as diagnostics only.
Run per model; surgery_merge.py combines the two models' maps.

Usage
-----
  python ownership_map.py --model task1_v8
  python ownership_map.py --model task2_v3
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from patchable_backend import PatchableBackend            # noqa: E402
from resid_alignment_check import load_shared_buffer      # noqa: E402

PROJECT = r"C:\Users\shahvandit\Downloads\Drone23_curent\drone_battery1"
RESULTS = os.path.join(PROJECT, "results")
OUT_DIR = os.path.join(RESULTS, "mech_interp", "ownership")

CONCEPTS = {
    "PT":     [86, 87, 94],
    "energy": [88, 89, 90, 95, 96],
}
ALL_CONCEPT_DIMS = sorted({d for v in CONCEPTS.values() for d in v})
N_PLACEBO = 20        # placebo dim-groups for the Stage-1a null
N_SET_PLACEBO = 10    # random unit-sets for the Stage-1b set-share null


def resample_dims(window: torch.Tensor, dims, gen) -> torch.Tensor:
    """do(dims): swap in those dims from a random donor window (whole 8-step
    segment, jointly -> coherent counterfactual)."""
    perm = torch.randperm(window.shape[0], generator=gen)
    w = window.clone()
    w[:, :, dims] = window[perm][:, :, dims]
    return w


def placebo_groups(obs_dim: int, size: int, n: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    pool = [i for i in range(obs_dim) if i not in ALL_CONCEPT_DIMS]
    return [sorted(torch.tensor(pool)[torch.randperm(len(pool), generator=g)[:size]].tolist())
            for _ in range(n)]


def screen_effect(backend, buf, caps0, sigma, dims, repeats, gen):
    """Stage-1a statistic per boundary: mean |a_pat - a_clean| / sigma."""
    eff = {b: torch.zeros_like(sigma[b]) for b in caps0 if b != "mu"}
    for _ in range(repeats):
        caps = backend.forward(resample_dims(buf, dims, gen))
        for b in eff:
            eff[b] += (caps[b] - caps0[b]).abs().mean(0) / repeats
    return {b: eff[b] / sigma[b] for b in eff}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--buffer-models", nargs=2, default=["task1_v8", "task2_v3"],
                    help="captures whose union forms the shared buffer")
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=60_000)
    ap.add_argument("--med-tokens", type=int, default=8_192,
                    help="subsample size for Stage-1b set-mediation runs")
    ap.add_argument("--repeats", type=int, default=4, help="Stage-1a resamples")
    ap.add_argument("--med-repeats", type=int, default=2, help="Stage-1b donor draws")
    ap.add_argument("--rho", type=float, default=2.0,
                    help="ownership needs S_top > rho * S_other; else shared")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpt = os.path.join(RESULTS, args.model, "Drone", "checkpoint.pt")
    backend = PatchableBackend(ckpt, n_head=args.n_head)
    print(backend)
    gen = torch.Generator().manual_seed(args.seed)

    buf = load_shared_buffer(args.buffer_models, args.max_tokens, seed=args.seed)
    print(f"[buffer] {tuple(buf.shape)}   concepts={list(CONCEPTS)}")

    # ── Stage 1a: screen ─────────────────────────────────────────────────────
    caps0 = backend.forward(buf)
    caps0.pop("mu")
    sigma = {b: caps0[b].std(0).clamp(min=1e-6) for b in caps0}

    S = {c: screen_effect(backend, buf, caps0, sigma, CONCEPTS[c],
                          args.repeats, gen) for c in CONCEPTS}

    # Placebo stats use the SAME number of repeats as the concept stats, so the
    # null is a like-for-like order statistic.
    null = {}
    for c in CONCEPTS:
        print(f"[placebo] {N_PLACEBO} groups of size {len(CONCEPTS[c])} "
              f"x {args.repeats} repeats for the {c} null ...")
        stats = []
        for grp in placebo_groups(backend.obs_dim, len(CONCEPTS[c]),
                                  N_PLACEBO, seed=args.seed):
            stats.append(screen_effect(backend, buf, caps0, sigma, grp,
                                       args.repeats, gen))
        null[c] = {b: torch.stack([s[b] for s in stats]).quantile(0.95, dim=0)
                   for b in caps0}

    concepts = list(CONCEPTS)
    print(f"\n[Stage 1a] screened counts (S > 95th-pct placebo null)")
    hdr = f"{'boundary':<16}{'d':>6}" + "".join(f"{c:>10}" for c in concepts)
    print(hdr + "\n" + "-" * len(hdr))
    screened = {c: {} for c in concepts}
    for b in caps0:
        row = f"{b:<16}{caps0[b].shape[1]:>6}"
        for c in concepts:
            screened[c][b] = S[c][b] > null[c][b]
            row += f"{int(screened[c][b].sum()):>10}"
        print(row)

    # ── labels from the Stage-1a contrast (per surgery unit) ────────────────
    ffn_bnds = [f"ffn_hidden.{i}" for i in range(backend.n_layer)]
    head_bnds = [f"attn_in.{i}" for i in range(backend.n_layer)]
    hd = backend.head_dim

    def unit_stats(b):
        """Per-unit S and null: neurons as-is; heads = slice means."""
        if b in ffn_bnds:
            return ({c: S[c][b] for c in concepts},
                    {c: null[c][b] for c in concepts})
        return ({c: S[c][b].reshape(backend.n_head, hd).mean(1)
                 for c in concepts},
                {c: null[c][b].reshape(backend.n_head, hd).mean(1)
                 for c in concepts})

    labels, weights, signal = {}, {}, {}
    print(f"\n[labels] Stage-1a contrast (rho={args.rho})")
    hdr = (f"{'unit set':<16}{'n':>6}"
           + "".join(f"{c + '-own':>11}" for c in concepts)
           + f"{'shared':>9}{'inert':>8}")
    print(hdr + "\n" + "-" * len(hdr))
    for b in ffn_bnds + head_bnds:
        Su, Nu = unit_stats(b)
        scr = {c: Su[c] > Nu[c] for c in concepts}
        s0, s1 = Su[concepts[0]], Su[concepts[1]]
        both = scr[concepts[0]] & scr[concepts[1]]
        only0 = scr[concepts[0]] & ~scr[concepts[1]]
        only1 = scr[concepts[1]] & ~scr[concepts[0]]
        lab = torch.full((len(s0),), -1)
        lab[only0 | (both & (s0 > args.rho * s1))] = 0
        lab[only1 | (both & (s1 > args.rho * s0))] = 1
        lab[both & ~(s0 > args.rho * s1) & ~(s1 > args.rho * s0)] = 2
        labels[b] = lab
        weights[b] = s0 / (s0 + s1).clamp(min=1e-9)          # soft PT share
        signal[b] = scr[concepts[0]] | scr[concepts[1]]
        print(f"{b:<16}{len(s0):>6}"
              + "".join(f"{int((lab == i).sum()):>11}" for i in range(2))
              + f"{int((lab == 2).sum()):>9}{int((lab == -1).sum()):>8}")

    # ── Stage 1b: SET-level mediation validation ─────────────────────────────
    g2 = torch.Generator().manual_seed(args.seed + 1)
    med_buf = buf[torch.randperm(len(buf), generator=g2)[:args.med_tokens]]
    clean = backend.forward(med_buf, capture=ffn_bnds + head_bnds,
                            full_seq=True)
    # tanh_squash=True: distances in POST-squash action space, otherwise
    # saturated states dominate D with deltas that change no behavior.
    mu_c = torch.tanh(clean["mu"])

    def unit_channels(b, unit_mask):
        """Channel indices for a set of units (heads expand to 32-dim slices)."""
        units = torch.nonzero(unit_mask).flatten()
        if b in ffn_bnds:
            return units
        return torch.cat([torch.arange(h * hd, (h + 1) * hd)
                          for h in units.tolist()]) if len(units) else units

    def set_share(pat_buf, mu_p, b, ch):
        """Fraction of the action change undone by restoring channels `ch`."""
        D_pat = (mu_p - mu_c).norm(dim=1)
        valid = D_pat > 1e-4
        if len(ch) == 0 or valid.sum() == 0:
            return float("nan")
        mu_r = torch.tanh(backend.forward(
            pat_buf, patches={b: (ch, clean[b][:, :, ch])}, capture=[])["mu"])
        D_r = (mu_r - mu_c).norm(dim=1)
        return float(((D_pat - D_r) / D_pat)[valid].mean())

    set_med = {}
    print(f"\n[Stage 1b] set-level mediation (restore whole owned set; "
          f"null = {N_SET_PLACEBO} random same-size sets)")
    hdr = (f"{'unit set':<16}{'concept':>8}{'n_owned':>9}{'set_share':>11}"
           f"{'null_q95':>10}{'passes':>8}")
    print(hdr + "\n" + "-" * len(hdr))
    for ci, c in enumerate(concepts):
        for _ in range(args.med_repeats):
            pat_buf = resample_dims(med_buf, CONCEPTS[c], g2)
            mu_p = torch.tanh(backend.forward(pat_buf, capture=[])["mu"])
            for b in ffn_bnds + head_bnds:
                owned = (labels[b] == ci) | (labels[b] == 2)
                n_owned = int((labels[b] == ci).sum())
                key = (b, c)
                rec = set_med.setdefault(
                    key, {"shares": [], "null_shares": [], "n_owned": n_owned})
                ch = unit_channels(b, owned)
                rec["shares"].append(set_share(pat_buf, mu_p, b, ch))
                # null: random unit sets of the same size, drawn from the rest
                pool = torch.nonzero(~owned).flatten()
                k = int(owned.sum())
                if len(pool) >= k > 0:
                    for _p in range(N_SET_PLACEBO // args.med_repeats):
                        pick = pool[torch.randperm(len(pool), generator=g2)[:k]]
                        m = torch.zeros_like(owned)
                        m[pick] = True
                        rec["null_shares"].append(
                            set_share(pat_buf, mu_p, b, unit_channels(b, m)))
    for (b, c), rec in set_med.items():
        sh = torch.tensor([s for s in rec["shares"] if s == s])
        nl = torch.tensor([s for s in rec["null_shares"] if s == s])
        share = float(sh.mean()) if len(sh) else float("nan")
        nq = float(nl.quantile(0.95)) if len(nl) else float("nan")
        rec["share"], rec["null_q95"] = share, nq
        rec["passes"] = bool(share == share and nq == nq and share > nq)
        print(f"{b:<16}{c:>8}{rec['n_owned']:>9}{share:>11.4f}{nq:>10.4f}"
              f"{str(rec['passes']):>8}")

    out = {"model": args.model, "concepts": CONCEPTS, "rho": args.rho,
           "screen": S, "null": null, "screened": screened,
           "labels": labels, "soft_w": weights, "signal": signal,
           "set_mediation": set_med,
           "head_dim": hd, "n_head": backend.n_head}
    os.makedirs(OUT_DIR, exist_ok=True)
    save = os.path.join(OUT_DIR, f"ownership_{args.model}.pt")
    torch.save(out, save)
    print(f"\n[saved] {save}")
    print("  labels: 0=PT-owned, 1=energy-owned, 2=shared, -1=inert; "
          "soft_w = S_PT/(S_PT+S_E).")
    print("  set_share validates the LABELED SET carries the concept to the "
          "action (owned+shared restored together, vs random-set null).")


if __name__ == "__main__":
    main()
