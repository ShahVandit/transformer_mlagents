"""
Hourly SOFA (Sequential Organ Failure Assessment) score.

The paper (Sec. 2.2) computes a predictive SOFA "using its clinical definition,
from the predictive means on five traits - mean BP, bilirubin, platelet,
creatinine, FiO2 - along with GCS and related medication history (e.g.,
dopamine)". That is six organ systems, each graded 0-4, summed to 0-24.

MIMIC-IV's CSV distribution ships no derived tables (the `mimic-code` SOFA
concept requires BigQuery or Postgres), so this is computed from raw items.

Two things to keep straight:

  - Inputs are the forecaster's PREDICTIVE MEANS, not raw observations. The
    score therefore exists at every hour of the stay even though bilirubin is
    measured once a day. That is the paper's construction, and it is what makes
    the Eq. 3 reward term (a SOFA rise of >= 2 between consecutive hours) a
    usable per-hour signal.
  - GCS is last-value-imputed rather than forecast, per Sec. 2.1.

Respiratory grading needs PaO2/FiO2 and, strictly, whether the patient is
ventilated. Ventilation status is available from the intervention intervals, so
the 3 and 4 grades are gated on it as the definition requires.
"""
import numpy as np

# Vasopressor itemids, split by how SOFA grades them.
NOREPI_EPI = {221906, 221289, 229617}
DOPAMINE = {221662}
DOBUTAMINE = {221653}
OTHER_PRESSORS = {222315, 221749, 229630, 229631, 229632}  # vasopressin, phenylephrine


def _grade(x, thresholds, scores):
    """Piecewise grading. `thresholds` ascending; returns scores[i] for the first
    threshold x falls under, else scores[-1]."""
    out = np.full(x.shape, scores[-1], dtype=np.float32)
    for thr, sc in zip(reversed(thresholds), reversed(scores[:-1])):
        out = np.where(x < thr, np.float32(sc), out)
    return out


def respiration(pao2, fio2, ventilated):
    """PaO2/FiO2 ratio. Grades 3 and 4 require respiratory support."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(fio2 > 0, pao2 / fio2, np.nan)
    ratio = np.nan_to_num(ratio, nan=500.0)          # missing -> no dysfunction
    score = _grade(ratio, [100.0, 200.0, 300.0, 400.0], [4, 3, 2, 1, 0])
    # Without ventilation, cap at 2: the 3 and 4 grades are defined on support.
    return np.where(ventilated, score, np.minimum(score, 2.0)).astype(np.float32)


def coagulation(platelets):
    """Platelet count, K/uL."""
    return _grade(platelets, [20.0, 50.0, 100.0, 150.0], [4, 3, 2, 1, 0])


def liver(bilirubin):
    """Total bilirubin, mg/dL."""
    return _grade(bilirubin, [1.2, 2.0, 6.0, 12.0], [0, 1, 2, 3, 4])


def cardiovascular(mbp, vaso_class, vaso_rate):
    """Mean arterial pressure and vasopressor support.

    vaso_class: 0 none, 1 dobutamine, 2 dopamine, 3 norepi/epi, 4 other pressor
    vaso_rate:  mcg/kg/min where the charted unit allowed grading, else NaN
    """
    score = np.where(mbp < 70.0, 1.0, 0.0).astype(np.float32)

    dobu = vaso_class == 1
    score = np.where(dobu, np.maximum(score, 2.0), score)

    dopa = vaso_class == 2
    r = np.nan_to_num(vaso_rate, nan=5.0)   # unknown dose -> the lowest bracket
    dopa_score = np.where(r > 15.0, 4.0, np.where(r > 5.0, 3.0, 2.0))
    score = np.where(dopa, np.maximum(score, dopa_score), score)

    ne = vaso_class == 3
    r2 = np.nan_to_num(vaso_rate, nan=0.05)
    ne_score = np.where(r2 > 0.1, 4.0, 3.0)
    score = np.where(ne, np.maximum(score, ne_score), score)

    # Vasopressin and phenylephrine are not in the original SOFA table. Scoring
    # them at 3 follows common practice; flagged here because it is a choice.
    other = vaso_class == 4
    score = np.where(other, np.maximum(score, 3.0), score)
    return score.astype(np.float32)


def cns(gcs_total):
    """Glasgow Coma Scale total, 3-15. Missing is read as 15 (no dysfunction)."""
    g = np.nan_to_num(gcs_total, nan=15.0)
    return _grade(g, [6.0, 10.0, 13.0, 15.0], [4, 3, 2, 1, 0])


def renal(creatinine):
    """Serum creatinine, mg/dL. Urine output is not used."""
    return _grade(creatinine, [1.2, 2.0, 3.5, 5.0], [0, 1, 2, 3, 4])


def sofa_score(*, pao2, fio2, ventilated, platelets, bilirubin, mbp,
               vaso_class, vaso_rate, gcs_total, creatinine, return_parts=False):
    """Total SOFA, 0-24. Every argument is an array of the same shape."""
    parts = {
        "respiration": respiration(pao2, fio2, ventilated),
        "coagulation": coagulation(platelets),
        "liver": liver(bilirubin),
        "cardiovascular": cardiovascular(mbp, vaso_class, vaso_rate),
        "cns": cns(gcs_total),
        "renal": renal(creatinine),
    }
    total = sum(parts.values()).astype(np.float32)
    return (total, parts) if return_parts else total


def vasopressor_class(itemids):
    """Map vasopressor itemids to the grading class used by cardiovascular()."""
    out = np.zeros(len(itemids), dtype=np.int8)
    arr = np.asarray(itemids)
    out[np.isin(arr, list(DOBUTAMINE))] = 1
    out[np.isin(arr, list(DOPAMINE))] = 2
    out[np.isin(arr, list(NOREPI_EPI))] = 3
    out[np.isin(arr, list(OTHER_PRESSORS))] = 4
    return out
