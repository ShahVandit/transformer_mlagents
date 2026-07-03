"""
Analytic actor param count for the TemporalTransformer policy, straight from a
cbm*.yaml. Validated to reproduce the real task1_v7 checkpoint (279,309).

  python scripts\param_count.py config\cbm.yaml          # -> 279,309
  python scripts\param_count.py config\cbm_bigger.yaml    # -> ~28.4M
  python scripts\param_count.py --d-model 768 --n-layer 4 --n-head 12 --ff-mult 4
"""
import argparse
import os
import re

OBS = 99          # input dim (fixed by the env)


def count(d_model, n_layer, ff_mult, obs=OBS, seq_len=8, n_actions=2):
    ff = d_model * ff_mult
    per_layer = (
        (3 * d_model * d_model + 3 * d_model)      # qkv
        + (d_model * d_model + d_model)            # attn_out (O)
        + (d_model * ff + ff)                      # ffn linear1
        + (ff * d_model + d_model)                 # ffn linear2
        + 4 * d_model                              # norm1 + norm2 (w+b)
        + 2                                        # attn_scale + ffn_scale
    )
    body = (
        n_layer * per_layer
        + (d_model * obs + d_model)                # input_proj
        + seq_len * d_model                        # temporal_pos_encoding
        + 2 * d_model                              # final_norm
    )
    head = n_actions * d_model + n_actions         # mu
    breakdown = {
        "per_transformer_layer": per_layer,
        f"transformer_body ({n_layer}L)": n_layer * per_layer,
        "input_proj": d_model * obs + d_model,
        "pos_encoding": seq_len * d_model,
        "final_norm": 2 * d_model,
        "action_head(mu)": head,
    }
    return body + head, breakdown


def from_yaml(path):
    txt = open(path).read()
    def g(key, default):
        m = re.search(rf"^\s*{key}:\s*([0-9]+)", txt, re.M)
        return int(m.group(1)) if m else default
    return dict(d_model=g("d_model", 128), n_layer=g("n_layer", 2),
               ff_mult=g("ff_mult", 2), n_head=g("n_head", 4),
               seq_len=g("sequence_length", 8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml", nargs="?", help="cbm*.yaml to read")
    ap.add_argument("--d-model", type=int)
    ap.add_argument("--n-layer", type=int)
    ap.add_argument("--n-head", type=int)
    ap.add_argument("--ff-mult", type=int)
    args = ap.parse_args()

    if args.yaml:
        cfg = from_yaml(args.yaml)
        src = os.path.basename(args.yaml)
    else:
        cfg = dict(d_model=128, n_layer=2, ff_mult=2, n_head=4, seq_len=8)
        src = "defaults"
    for k in ("d_model", "n_layer", "n_head", "ff_mult"):
        if getattr(args, k.replace("-", "_")) is not None:
            cfg[k] = getattr(args, k.replace("-", "_"))

    assert cfg["d_model"] % cfg["n_head"] == 0, \
        f"d_model {cfg['d_model']} not divisible by n_head {cfg['n_head']}"
    total, bd = count(cfg["d_model"], cfg["n_layer"], cfg["ff_mult"],
                      seq_len=cfg["seq_len"])

    print(f"[{src}] d_model={cfg['d_model']} n_layer={cfg['n_layer']} "
          f"n_head={cfg['n_head']} (head_dim={cfg['d_model']//cfg['n_head']}) "
          f"ff_mult={cfg['ff_mult']} (ffn={cfg['d_model']*cfg['ff_mult']}) "
          f"seq_len={cfg['seq_len']} obs={OBS}")
    print("-" * 56)
    for k, v in bd.items():
        print(f"  {k:28s} {v:>14,}")
    print("-" * 56)
    print(f"  {'ACTOR TOTAL':28s} {total:>14,}   (~{total/1e6:.1f}M)")


if __name__ == "__main__":
    main()
