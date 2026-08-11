"""Joint lab-panel action encoding for the Pareto track."""
import numpy as np

import config as cfg
import itemids as ids


LABS = list(ids.TARGET_LABS)
PANEL_BITS = list(cfg.JOINT_PANEL_BITS)
PANEL_NAMES = list(cfg.JOINT_PANEL_NAMES)
PANEL_ARRAY = np.array([[int(c) for c in bits] for bits in PANEL_BITS], dtype=np.int8)
BITS_TO_ACTION = {bits: i for i, bits in enumerate(PANEL_BITS)}


def bits_from_frame(df):
    cols = [f"obs_{lab}" for lab in LABS]
    return df[cols].notna().to_numpy(dtype=np.int8)


def encode_bits(bits):
    """Map every four-lab order combination to one of the retained panels.

    A combination with any lab set can never map to the empty panel. Ties on
    Hamming distance are otherwise broken by index order, and `none` is index 0,
    so creatinine-alone and bun-alone (both distance 1 from `0000` and from
    `1100`) would silently become no-draw. That relabels a real blood draw as no
    draw, zeroes its burden, and teaches the behaviour policy the wrong action.
    329 rows in train, small but wrong in the one direction that matters.
    """
    bits = np.asarray(bits, dtype=np.int8)
    out = np.empty(len(bits), dtype=np.int64)
    drew = PANEL_ARRAY.sum(axis=1) > 0
    for i, row in enumerate(bits):
        key = "".join(str(int(x)) for x in row)
        if key in BITS_TO_ACTION:
            out[i] = BITS_TO_ACTION[key]
            continue
        dist = np.abs(PANEL_ARRAY - row[None, :]).sum(axis=1).astype(float)
        if row.sum() > 0:
            dist[~drew] = np.inf          # never collapse a draw into `none`
        out[i] = int(np.argmin(dist))
    return out


def encode_frame(df):
    return encode_bits(bits_from_frame(df))


def action_bits(actions):
    return PANEL_ARRAY[np.asarray(actions, dtype=np.int64)]


def panel_n_labs(actions):
    return action_bits(actions).sum(axis=1).astype(np.int16)


def panel_names_for(actions):
    return [PANEL_NAMES[int(a)] for a in actions]
