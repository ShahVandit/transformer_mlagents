"""
Central configuration: every path and hyperparameter the pipeline uses.

Generated artifacts (parquet caches, RL tensors, models, reports, figures) live
inside the project folder. The one input that may not be redistributed is read
from a location you can set with an environment variable:

    MIMIC4_DIR   folder holding the MIMIC-IV v3.1 *.csv.gz files

PowerShell example:
    $env:MIMIC4_DIR = "C:\\data\\mimiciv\\3.1"

Bash example:
    export MIMIC4_DIR=/data/mimiciv/3.1

If unset, it defaults to ../mimiciv/3.1 relative to this repository.

Every constant traceable to the paper carries its section or equation number:
  Cheng L-F, Prasad N, Engelhardt BE. "An Optimal Policy for Patient Laboratory
  Tests in Intensive Care Units." Pac Symp Biocomput. 2019;24:320-331.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- inputs ----
# Credentialed data. Never stored in this repository (see README, Data access).
#
# $MIMIC4_DIR wins outright. Otherwise try the usual places, because the data
# and the checkout do not necessarily sit next to each other: a clone at
# ~/transformer_mlagents with data at ~/mimiciv/3.1 is the common server layout,
# and a project-relative default alone would silently miss it.
_CANDIDATES = [
    PROJECT_ROOT.parent / "mimiciv" / "3.1",   # data beside the checkout
    Path.home() / "mimiciv" / "3.1",           # data in the home directory
    PROJECT_ROOT / "data" / "mimiciv" / "3.1",
]


def _resolve_mimic_dir():
    env = os.environ.get("MIMIC4_DIR")
    if env:
        return Path(env).expanduser()
    for c in _CANDIDATES:
        if (c / "icustays.csv.gz").exists():
            return c
    return _CANDIDATES[0]          # report this one in the error message


MIMIC4_DIR = _resolve_mimic_dir()

ICUSTAYS_CSV = MIMIC4_DIR / "icustays.csv.gz"
PATIENTS_CSV = MIMIC4_DIR / "patients.csv.gz"
ADMISSIONS_CSV = MIMIC4_DIR / "admissions.csv.gz"
CHARTEVENTS_CSV = MIMIC4_DIR / "chartevents.csv.gz"
LABEVENTS_CSV = MIMIC4_DIR / "labevents.csv.gz"
INPUTEVENTS_CSV = MIMIC4_DIR / "inputevents.csv.gz"
PROCEDUREEVENTS_CSV = MIMIC4_DIR / "procedureevents.csv.gz"
PRESCRIPTIONS_CSV = MIMIC4_DIR / "prescriptions.csv.gz"

# ---------------------------------------------------- generated artifacts ----
DATA_DIR = PROJECT_ROOT / "data"
RAW_CACHE_DIR = DATA_DIR / "cache"        # stage 1 parquet caches
PROCESSED_DIR = DATA_DIR / "processed"    # stage 2 hourly grids
RL_DIR = DATA_DIR / "rl"                  # stage 3 npz tensors, per lab per split
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"

COHORT_PARQUET = RAW_CACHE_DIR / "cohort.parquet"
CHART_PARQUET = RAW_CACHE_DIR / "chartevents_filtered.parquet"
LAB_PARQUET = RAW_CACHE_DIR / "labevents_filtered.parquet"
INTERVENTIONS_PARQUET = RAW_CACHE_DIR / "interventions.parquet"
SPLITS_JSON = DATA_DIR / "splits.json"
HOURLY_DIR = PROCESSED_DIR / "hourly"     # one parquet shard per split

SEED = 0

# ------------------------------------------------- cohort selection (Sec. 2.1) ----
MIN_AGE = 18
MIN_LOS_DAYS = 1.0        # paper: ICU stay between one and twenty days
MAX_LOS_DAYS = 20.0
REQUIRE_ALL_TRAITS = True  # paper: at least one recorded measure of each trait

# Chunk size for the single full scan of chartevents/labevents. These are the
# 3.5 GB and 2.6 GB files; the scan runs once and caches to parquet.
SCAN_CHUNK_ROWS = 5_000_000

# ------------------------------------------------------ splits (Sec. 3) ----
# BY SUBJECT, not by stay: 65,366 subjects hold 94,458 ICU stays, so splitting on
# stay_id would leak a patient across train and test. The paper used a large
# held-out set; this project uses more training support for CQL/FQE while keeping
# validation and test patient-disjoint.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# ------------------------------------------------- decision process (Sec. 2.2) ----
BIN_HOURS = 1              # paper resamples to a one-hour grid
GAMMA = 0.9                # discount factor used by FQI (Sec. 3)
INCLUDE_VITAL_STD = False  # False -> 21-dim state (paper); True -> 25-dim variant

# ------------------------------------------------------- reward (Sec. 2.2) ----
SOFA_DELTA_THRESHOLD = 2.0   # Eq. 3: a SOFA rise >= 2 is the critical sepsis index
COST_DECAY_GAMMA = 6.0       # Eq. 6: Gamma_l, hours; paper sets 6 for all labs
TREAT_LOOKAHEAD_BINS = 1     # Eq. 4: intervention started at s_{t+1}
# Eq. 5: c_l, the minimum prediction error that triggers an information reward.
# Set at run time to the median prediction error over labs ordered in TRAIN.
REWARD_DIMS = ["r_sofa", "r_treat", "r_info", "neg_r_cost"]

# ----------------------------------------------- joint draw-timing Pareto track ----
# These objectives can identify WHEN a blood draw is useful, but they do not
# identify WHICH panel is appropriate for a particular deterioration. Giving
# all non-empty panels identical utility makes the smallest panel
# dominate by construction. Keep the action aligned with the supported causal
# question: no draw versus one blood draw. Utility is the strongest expected
# information signal among the four target labs. Summing all four would assume
# every physical draw necessarily obtains all four assays, which is not true in
# the logged data; exact panel selection is outside this policy.
JOINT_PANEL_BITS = ["0000", "1111"]
JOINT_PANEL_NAMES = ["none", "blood_draw"]
JOINT_REWARD_DIMS = ["utility", "burden"]
JOINT_DETECTION_LOOKAHEAD_HOURS = 12
# Preference weights on the simplex: (w_utility, w_burden), summing to 1.
#
# Replaces a bare lambda multiplying burden, for two reasons measured on this
# data. First, lambda = w_burden / w_utility, so a sweep of lambda in
# [0.1, 0.9] only reaches w_burden in [0.09, 0.47]: it never crosses the
# balanced point and never asks for a burden-dominant policy, leaving half the
# frontier unexplored. Second, with w_utility pinned at 1 the reward
# magnitude grows with lambda, while CQL_ALPHA is a fixed weight against the TD
# loss, so conservatism silently weakens as lambda rises. On the simplex |r|
# stays roughly constant and alpha means the same thing at every point.
JOINT_PREFERENCES = [(0.9, 0.1), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.1, 0.9)]

# A policy only joins the frontier if it beats both trivial baselines on its own
# weighted objective. Without this a diverged run is indistinguishable from a
# preference that genuinely wants more testing.
VALIDITY_BASELINES = ("never_draw", "always_draw")
VALIDITY_MARGIN = 0.0     # required improvement over the best trivial baseline

# --------------------------------------------------------- forecaster (Sec. 2.1) ----
# The paper uses a multi-output Gaussian process. It feeds exactly two things:
# (m_t, sigma_t) in the state, and the normalizer in r_info. Any forecaster
# emitting an hourly predictive mean and std satisfies that contract.
FORECASTER = "local_trend"   # "local_trend" | "mogp"
FORECAST_MIN_STD = 1e-3      # floor on sigma_t so r_info cannot divide by ~0

# ------------------------------------------------------------ MO-FQI (Sec. 2.3) ----
FQI_ITERATIONS = 200
FQI_SAMPLE_PER_ITER = 100_000   # transitions drawn each iteration
FQI_N_TREES = 50
FQI_MIN_SAMPLES_LEAF = 5
FQI_MAX_DEPTH = None
FQI_N_JOBS = -1

# Eq. 7: order iff Q_d(s,1) + eps_d > Q_d(s,0) for ALL d. Only the cost slack is
# tuned; the paper tunes it so the recommended order count approximates the
# observed count. Search grid is over eps_cost only, on the VAL split.
EPS_GRID = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
EPS_BISECT_STEPS = 40   # refine inside the bracketing interval; the grid alone
                        # is too coarse (lactate jumps 0 -> 104k across one step)
BUDGET_HOURS = 24            # force one order per 24h window with no recommendation

# ------------------------------------------- CQL arm (optional, non-paper) ----
# Conservative Q-Learning is NOT part of Cheng et al. It is an alternative
# learner behind --learner cql, kept separate so the replication stays intact.
#
# CQL is scalar-Q, so the paper's 4-vector reward has to be collapsed to one
# number by these weights. That collapse is exactly what the paper's Pareto
# pruning exists to avoid, so running this arm means giving up the
# multi-objective claim and fixing a preference by hand instead.
REWARD_WEIGHTS = {"r_sofa": 1.0, "r_treat": 1.0, "r_info": 1.0, "neg_r_cost": 1.0}

CQL_ALPHA = 1.0            # weight on the conservative penalty
CQL_STEPS = 40_000         # gradient steps
CQL_LR = 1e-4
CQL_BATCH = 1024
CQL_HIDDEN = 128
CQL_TARGET_TAU = 0.005     # Polyak rate for the target network
CQL_EVAL_EVERY = 2_000

# ------------------------------------------------- off-policy evaluation ----
# Tier 1 replicates the paper: per-step WIS with an undiscounted horizon.
# Paper-faithful setting for the PER-LAB replication in s5_evaluate_ope.py only:
# "The discount factor was set to gamma_WIS = 1.0, so all time steps contribute
# equally to the value of a trajectory" (Sec. 3.1). Do not use it anywhere the
# clinician's discounted factual return appears in the same table.
WIS_GAMMA = 1.0

# The joint Pareto track discounts everything at GAMMA. Mixing a discounted
# clinician value with an undiscounted policy estimate introduces a fixed scale
# factor of (stay length)/(1/(1-GAMMA)) -- about 15x on this cohort -- which
# reads as the policies beating the clinician 15-fold on both objectives when
# their per-step values are in fact comparable.
JOINT_WIS_GAMMA = GAMMA
RANDOM_BASELINE_PS = [0.01, None, 0.5]   # None -> empirical order rate p_emp
RANDOM_BASELINE_TRIALS = 10

# Tier 2 additions the paper does not have.
OPE_EPSILON = 0.05        # epsilon-greedy softening of the deterministic policy
OPE_RATIO_CLIP = 5.0      # clip on the per-step log importance ratio
OPE_PROB_FLOOR = 1e-3     # floor on estimated behavior-policy probability
OPE_N_BOOTSTRAP = 200     # patient-level bootstrap resamples
FQE_EPOCHS = 40
FQE_MIN_STEPS = 4000    # floor on gradient steps, so small splits still converge
FQE_STEPS = 20_000      # hard cap on FQE training budget
FQE_STEPS_PER_EPOCH = 2_000
FQE_LR = 1e-3
FQE_BATCH = 1024
FQE_HIDDEN = 128

# ------------------------------------------------- clinical metrics (Sec. 3.1) ----
TREATMENT_LOOKBACK_HOURS = 48   # trace back from an intervention onset to an order

LABS = ["creatinine", "bun", "wbc", "lactate"]


REQUIRED_INPUTS = {
    "icustays": ICUSTAYS_CSV, "patients": PATIENTS_CSV,
    "admissions": ADMISSIONS_CSV, "chartevents": CHARTEVENTS_CSV,
    "labevents": LABEVENTS_CSV, "inputevents": INPUTEVENTS_CSV,
    "procedureevents": PROCEDUREEVENTS_CSV, "prescriptions": PRESCRIPTIONS_CSV,
}


def check_inputs():
    """Fail immediately, and loudly, if any source table is missing.

    Stage 1 spends the better part of an hour scanning chartevents, so a wrong
    MIMIC4_DIR should surface in the first second rather than after the scan.
    """
    missing = [n for n, p in REQUIRED_INPUTS.items() if not p.exists()]
    if not missing:
        return
    tried = "\n".join(f"    {c}" for c in _CANDIDATES)
    raise SystemExit(
        f"\nMIMIC-IV files not found under:\n    {MIMIC4_DIR}\n\n"
        f"Missing: {', '.join(missing)}\n\n"
        f"Set MIMIC4_DIR to the folder holding the .csv.gz files, e.g.\n"
        f"    export MIMIC4_DIR=~/mimiciv/3.1\n\n"
        f"Searched by default:\n{tried}\n\n"
        f"The pipeline expects a flat layout. If yours has hosp/ and icu/ "
        f"subfolders, either point MIMIC4_DIR at a directory of symlinks or "
        f"flatten it:\n"
        f"    mkdir -p flat && find hosp icu -name '*.csv.gz' "
        f"-exec ln -s ../{{}} flat/ \\;\n")


def ensure_dirs():
    """Create the output folders if they do not exist yet."""
    for d in (RAW_CACHE_DIR, PROCESSED_DIR, HOURLY_DIR, RL_DIR,
              MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
