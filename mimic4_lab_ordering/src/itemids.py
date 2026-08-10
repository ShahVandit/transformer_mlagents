"""
MIMIC-IV v3.1 itemid maps.

The paper ran on MIMIC-III, whose itemids do not carry over: MIMIC-IV charts the
MetaVision item space only, so every chartevents id here is a 22xxxx code. Lab
itemids in `labevents` are mostly stable between the two databases.

The paper filters its cohort on "20 vital signs and lab tests commonly ordered
and reviewed by clinicians" but never enumerates them. TRAITS below is our
instantiation of that list. It is deliberately explicit so the cohort filter is
reproducible.

Three groups matter downstream:
  TRAITS      -> forecast hourly, drive the cohort completeness filter
  TARGET_LABS -> the four labs we learn ordering policies for (Sec. 2.1)
  SOFA_*      -> inputs to the hourly SOFA score (sofa.py)

Several concepts map to more than one itemid (arterial vs non-invasive BP,
Fahrenheit vs Celsius, serum vs blood-gas haemoglobin). Each entry is a list and
the extractor coalesces them, applying `convert` first where units differ.
"""

# --------------------------------------------------------------- chartevents ----
# label -> (itemids, unit conversion applied to valuenum, plausible range)
CHART_TRAITS = {
    "hr":        ([220045], None, (20.0, 250.0)),
    "rr":        ([220210], None, (4.0, 60.0)),
    "mbp":       ([220052, 220181], None, (20.0, 200.0)),   # arterial, then NIBP
    "temp_c":    ([223762], None, (30.0, 43.0)),            # Celsius, native
    "temp_f":    ([223761], "f_to_c", (30.0, 43.0)),        # Fahrenheit -> Celsius
    "fio2":      ([223835], "fio2_frac", (0.21, 1.0)),      # charted as % or frac
}

# The two temperature items collapse into one trait after conversion.
CHART_TRAIT_ALIAS = {"temp_f": "temp", "temp_c": "temp"}

# GCS is last-value-imputed, never forecast (paper Sec. 2.1). Text-valued in
# MIMIC-IV, so the extractor reads `value` and maps it to the numeric score.
GCS_ITEMS = {
    "gcs_eye":    220739,
    "gcs_verbal": 223900,
    "gcs_motor":  223901,
}

GCS_VERBAL_MAP = {
    "no response": 1, "no response-ett": 1, "none": 1,
    "incomprehensible sounds": 2,
    "inappropriate words": 3,
    "confused": 4,
    "oriented": 5,
}
GCS_EYE_MAP = {
    "none": 1, "no response": 1,
    "to pain": 2, "to painful stimuli": 2,
    "to speech": 3, "to voice": 3,
    "spontaneously": 4,
}
GCS_MOTOR_MAP = {
    "no response": 1, "none": 1,
    "abnormal extension": 2, "abnorm extensn": 2,
    "abnormal flexion": 3, "abnorm flexion": 3,
    "flex-withdraws": 4, "withdraws": 4,
    "localizes pain": 5,
    "obeys commands": 6,
}

# --------------------------------------------------------------- labevents ----
# label -> (itemids, unit conversion, plausible range)
LAB_TRAITS = {
    "creatinine": ([50912], None, (0.1, 20.0)),        # mg/dL
    "bun":        ([51006], None, (1.0, 200.0)),       # mg/dL
    "wbc":        ([51301], None, (0.1, 100.0)),       # K/uL
    "lactate":    ([50813], None, (0.1, 30.0)),        # mmol/L
    "bilirubin":  ([50885], None, (0.1, 60.0)),        # total, mg/dL
    "platelets":  ([51265], None, (5.0, 2000.0)),      # K/uL
    "pao2":       ([50821], None, (20.0, 700.0)),      # mmHg
    "sodium":     ([50983], None, (100.0, 180.0)),
    "potassium":  ([50971], None, (1.0, 10.0)),
    "chloride":   ([50902], None, (60.0, 150.0)),
    "bicarbonate": ([50882], None, (5.0, 60.0)),
    "hemoglobin": ([51222, 50811], None, (2.0, 25.0)),
}

# The four labs we learn independent binary ordering policies for (Sec. 2.1).
TARGET_LABS = ["creatinine", "bun", "wbc", "lactate"]

# The four vitals in the state vector (Sec. 2.2).
STATE_VITALS = ["hr", "rr", "temp", "mbp"]

# Everything forecast on the hourly grid. Order is fixed and is the column order
# of every (mean, std) array the forecaster emits.
FORECAST_TRAITS = STATE_VITALS + TARGET_LABS + [
    "bilirubin", "platelets", "fio2", "pao2",
    "sodium", "potassium", "chloride", "bicarbonate",
    "hemoglobin",
]

# Cohort completeness filter. The paper filters on 20 traits and forecasts 17 of
# them, imputing the three GCS components by last value: 17 + 3 = 20. This list
# matches those counts exactly.
TRAITS = FORECAST_TRAITS + list(GCS_ITEMS)
assert len(FORECAST_TRAITS) == 17, len(FORECAST_TRAITS)
assert len(TRAITS) == 20, len(TRAITS)

# ------------------------------------------------------------ interventions ----
# Paper Sec. 2.2 Eq. 4: "antibiotics, vasopressors, dialysis or ventilation".
VASOPRESSOR_ITEMS = [
    221906,  # Norepinephrine
    221289,  # Epinephrine
    222315,  # Vasopressin
    221749,  # Phenylephrine
    221662,  # Dopamine
    221653,  # Dobutamine
    229617,  # Epinephrine.
    229630, 229631, 229632,  # Phenylephrine premixes
]

VENTILATION_ITEMS = [
    224385,  # Intubation (procedureevents)
    225792,  # Invasive Ventilation
    225794,  # Non-invasive Ventilation
]

DIALYSIS_ITEMS = [
    225441,  # Hemodialysis
    225802,  # Dialysis - CRRT
    225803,  # Dialysis - CVVHD
    225805,  # Peritoneal Dialysis
    224270,  # Dialysis Catheter
]

# Matched case-insensitively against prescriptions.drug. Systemic antibacterials
# in routine ICU use; a starttime for any of these counts as an antibiotic onset.
ANTIBIOTIC_DRUGS = [
    "vancomycin", "piperacillin", "cefepime", "ceftriaxone", "ceftazidime",
    "cefazolin", "meropenem", "imipenem", "ertapenem", "aztreonam",
    "levofloxacin", "ciprofloxacin", "moxifloxacin", "azithromycin",
    "clindamycin", "metronidazole", "gentamicin", "tobramycin", "amikacin",
    "linezolid", "daptomycin", "doxycycline", "ampicillin", "nafcillin",
    "oxacillin", "penicillin", "amoxicillin", "trimethoprim",
    "sulfamethoxazole", "colistin", "tigecycline", "ceftaroline",
]

INTERVENTION_KINDS = ["vasopressor", "ventilation", "dialysis", "antibiotic"]


# ------------------------------------------------------------- conversions ----
def apply_conversion(values, kind):
    """Unit conversions named in CHART_TRAITS / LAB_TRAITS."""
    if kind is None:
        return values
    if kind == "f_to_c":
        return (values - 32.0) * 5.0 / 9.0
    if kind == "fio2_frac":
        # MIMIC-IV charts FiO2 as a percentage (21-100) but a minority of rows
        # are already fractions. Scale only the ones above 1.
        import numpy as np
        return np.where(values > 1.0, values / 100.0, values)
    raise ValueError(f"unknown conversion {kind!r}")


def chart_itemids():
    """Every chartevents itemid the pipeline reads, GCS included."""
    ids = set()
    for items, _, _ in CHART_TRAITS.values():
        ids.update(items)
    ids.update(GCS_ITEMS.values())
    return sorted(ids)


def lab_itemids():
    """Every labevents itemid the pipeline reads."""
    ids = set()
    for items, _, _ in LAB_TRAITS.values():
        ids.update(items)
    return sorted(ids)


def chart_itemid_to_trait():
    """itemid -> (trait name after aliasing, conversion kind, valid range)."""
    out = {}
    for name, (items, conv, rng) in CHART_TRAITS.items():
        trait = CHART_TRAIT_ALIAS.get(name, name)
        for i in items:
            out[i] = (trait, conv, rng)
    return out


def lab_itemid_to_trait():
    """itemid -> (trait name, conversion kind, valid range)."""
    out = {}
    for name, (items, conv, rng) in LAB_TRAITS.items():
        for i in items:
            out[i] = (name, conv, rng)
    return out
