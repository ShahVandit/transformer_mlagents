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
    """Encode any non-empty lab combination as one blood draw."""
    bits = np.asarray(bits, dtype=np.int8)
    return (bits.sum(axis=1) > 0).astype(np.int64)


def encode_frame(df):
    return encode_bits(bits_from_frame(df))


def action_bits(actions):
    return PANEL_ARRAY[np.asarray(actions, dtype=np.int64)]


def panel_n_labs(actions):
    # Retained for file compatibility; in the binary track this is the number
    # of physical draws (zero or one), not the number of ordered analytes.
    return (np.asarray(actions) != 0).astype(np.int16)


def panel_names_for(actions):
    return [PANEL_NAMES[int(a)] for a in actions]
