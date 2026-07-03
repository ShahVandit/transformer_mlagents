"""
Mechanistic validation of the surgery merge.

The merge's CLAIM is that it preserves each specialist's task computation at
the transplanted units. Verify the mechanism, not just the behavior: re-run
the input-resample effect scoring on the MERGED checkpoint and check that

  PT-owned units still respond to dims [86,87,94]   with effect ~ nav's
  energy-owned units still respond to [88,89,90,95,96] with effect ~ bat's

Retention(u) = S_model(u) / S_owner(u) per owned unit. If behavior is good
but retention is scrambled, the merge worked for a different reason than the
ownership hypothesis — that must be known before writing it up.

Comparison columns:
  ret_merged : the surgery merge (the claim under test)
  ret_soup   : the alpha-soup INIT, built IN MEMORY from the two specialist
               checkpoints (never a results/soup_* run dir — those hold
               FINE-TUNED soups, which would confound the comparison)
  ret_nav / ret_bat : both specialists. The owner's own column reads 1.0 by
               construction (sanity check); the OTHER specialist's column
               shows how much of the concept's effect survives 30M steps of
               fine-tuning alone — the natural floor for cross-task retention.

Usage
-----
  python validate_merge.py --merged surgery_v1
  python validate_merge.py --merged surgery_v1 --alpha 0.5
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from patchable_backend import PatchableBackend                       # noqa: E402
from resid_alignment_check import load_shared_buffer                 # noqa: E402
from ownership_map import CONCEPTS, screen_effect                    # noqa: E402
from surgery_merge import combine_labels, PT, EN                     # noqa: E402

PROJECT = r"C:\Users\shahvandit\Downloads\Drone23_curent\drone_battery1"
RESULTS = os.path.join(PROJECT, "results")
OWN_DIR = os.path.join(RESULTS, "mech_interp", "ownership")


def screen_model(ckpt, buf, n_head, repeats, seed):
    """ckpt: checkpoint path OR an in-memory checkpoint dict."""
    backend = PatchableBackend(ckpt, n_head=n_head)
    gen = torch.Generator().manual_seed(seed)
    caps0 = backend.forward(buf)
    caps0.pop("mu")
    sigma = {b: caps0[b].std(0).clamp(min=1e-6) for b in caps0}
    return {c: screen_effect(backend, buf, caps0, sigma, CONCEPTS[c],
                             repeats, gen) for c in CONCEPTS}


def build_soup(ck_nav, ck_bat, alpha):
    """alpha*nav + (1-alpha)*bat over Policy floats — the soup INIT, in memory
    (alpha = NAV share, run_soup_sweep.py convention)."""
    d1 = torch.load(ck_nav, map_location="cpu")["Policy"]
    d2 = torch.load(ck_bat, map_location="cpu")["Policy"]
    return {"Policy": {
        k: (alpha * d1[k] + (1 - alpha) * d2[k]
            if torch.is_tensor(d1[k]) and d1[k].dtype.is_floating_point
            and d1[k].shape == d2[k].shape else d2[k])
        for k in d2}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--nav", default="task1_v8")
    ap.add_argument("--bat", default="task2_v3")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="alpha for the in-memory soup column (NAV share)")
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=30_000)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    map_nav = torch.load(os.path.join(OWN_DIR, f"ownership_{args.nav}.pt"),
                         map_location="cpu")
    map_bat = torch.load(os.path.join(OWN_DIR, f"ownership_{args.bat}.pt"),
                         map_location="cpu")
    buf = load_shared_buffer([args.nav, args.bat], args.max_tokens,
                             seed=args.seed)
    print(f"[buffer] {tuple(buf.shape)}")

    ck = lambda m: os.path.join(RESULTS, m, "Drone", "checkpoint.pt")
    models = {"nav": ck(args.nav), "bat": ck(args.bat),
              "merged": ck(args.merged),
              "soup": build_soup(ck(args.nav), ck(args.bat), args.alpha)}
    S = {}
    for tag, ckpt in models.items():
        print(f"[screen] {tag} ...")
        S[tag] = screen_model(ckpt, buf, args.n_head, args.repeats, args.seed)
    hd, n_head = map_nav["head_dim"], map_nav["n_head"]

    def unit_S(Sm, bnd, concept, is_head):
        s = Sm[concept][bnd]
        return s.reshape(n_head, hd).mean(1) if is_head else s

    cols = ["merged", "soup", "nav", "bat"]
    hdr = (f"{'boundary':<16}{'owner':>8}{'n_units':>9}{'S_owner':>10}"
           + "".join(f"{'ret_' + c:>10}" for c in cols))
    print("\nretention = S_model / S_owner, mean over owned units")
    print(hdr + "\n" + "-" * len(hdr))
    report = {}
    for bnd in map_nav["labels"]:
        is_head = bnd.startswith("attn_in")
        lab = combine_labels(map_nav["labels"][bnd], map_bat["labels"][bnd])
        for l, cname, owner in ((PT, "PT", "nav"), (EN, "energy", "bat")):
            sel = lab == l
            if sel.sum() == 0:
                continue
            s_own = unit_S(S[owner], bnd, cname, is_head)[sel]
            row = f"{bnd:<16}{cname:>8}{int(sel.sum()):>9}{s_own.mean():>10.4f}"
            rets = {}
            for c in cols:
                ret = (unit_S(S[c], bnd, cname, is_head)[sel]
                       / s_own.clamp(min=1e-9)).mean().item()
                rets[c] = ret
                row += f"{ret:>10.3f}"
            print(row)
            report[(bnd, cname)] = {"n": int(sel.sum()),
                                    "S_owner": s_own, "retention": rets}

    save = os.path.join(OWN_DIR, f"validation_{args.merged}.pt")
    torch.save({"merged": args.merged, "report": report}, save)
    print(f"\n[saved] {save}")
    print("  success: ret_merged ~ 1.0 at owned units, above ret_soup, and "
          "above the OTHER specialist's column (the fine-tuning-only floor). "
          "The owner's own column must read 1.0 (sanity). Good behavior + "
          "scrambled retention = merge worked for a different reason than "
          "the ownership hypothesis.")


if __name__ == "__main__":
    main()
