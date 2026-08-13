# Multi-Objective Offline RL for ICU Lab-Test Ordering (MIMIC-IV)

A replication of

> Cheng L-F, Prasad N, Engelhardt BE. **An Optimal Policy for Patient Laboratory
> Tests in Intensive Care Units.** *Pacific Symposium on Biocomputing* 2019;24:320-331.

on **MIMIC-IV v3.1** instead of the paper's MIMIC-III, with an off-policy
evaluation section the original does not have.

The clinical problem is over-ordering. Lab tests drive up to 70% of diagnostic
and treatment decisions, but repeat panels are frequently ordered at intervals
too short to contain a clinically relevant change, and phlebotomy accounts for
roughly half the variation in how much blood ICU patients are transfused. The
question is not whether to test, it is *when*: an ordering policy has to trade
the expected information in a test against its cost and its cumulative harm to
the patient.

The paper frames that as a multi-objective MDP over an ICU admission, learns a
binary order/no-order policy per lab with a **vector-valued** reward and Pareto
action pruning, and evaluates against clinicians with importance sampling.

---

## What this repository does differently

**Replicated faithfully:** the 21-dimensional state, the binary per-lab action
space, all four reward components (Eqs. 3-6) with the paper's constants, MO-FQI
with strict Pareto pruning, the Eq. 7 policy collapse with a tuned cost slack,
the 24-hour budget rule, and every metric in Sec. 3.1.

**Changed deliberately, and documented at each site:**

| | paper | here | why |
|---|---|---|---|
| Database | MIMIC-III | MIMIC-IV v3.1 | newer cohort; itemids renumbered, so none of the extraction carries over |
| Forecaster | sparse multi-output GP | local linear trend state-space model | same `(mean, std)` contract, behind a swappable interface (see below) |
| Splits | 3,636/2,424 admissions | subject-level train/val/test | 65,366 patients hold 94,458 stays, so a stay-level split leaks a patient across train and test |
| `eps_cost`, `c_l` tuned on | train | val | strictly cleaner; does not change the method |
| OPE | PS-WIS only, no intervals | PS-WIS **plus** FQE, WDR, patient bootstrap CIs, ESS, support diagnostics, an FQE calibration check | see *What the paper does not report* |
| Workflow context | not used | four strictly lagged POE lab-order features, with a physiology-only ablation and validation report | tests whether prior ordering workflow adds signal without asserting an order-to-specimen link |

---

## Method

**Cohort** (Sec. 2.1). Adult ICU stays with 1-20 days length of stay, keeping
only stays with at least one recorded value of each of 20 traits. The paper does
not enumerate its 20; ours are listed explicitly in [`src/itemids.py`](src/itemids.py)
as 17 forecast traits plus the three GCS components, matching the paper's split
of "hourly predictions on 17 of the 20" with GCS imputed by last value.

**State**, 21 dimensions (Sec. 2.2). The paper states 21 without itemizing, and
its listed blocks sum to 25 if a standard deviation is carried for all eight
traits. The only clean decomposition reaching 21:

| block | dims |
|---|---|
| predictive mean of HR, RR, temperature, mean BP | 4 |
| predictive mean of creatinine, BUN, WBC, lactate | 4 |
| predictive **std** of those four labs | 4 |
| predictive SOFA | 1 |
| `y_t`, last observed value per lab | 4 |
| `Delta_t`, hours since each lab was last ordered | 4 |

Vitals are charted roughly hourly, so their predictive std is near-constant and
carries little signal, while a lab's grows with time since it was last drawn and
is exactly what the ordering decision turns on. Set
`config.INCLUDE_VITAL_STD = True` for the 25-dim variant.

The default MIMIC-IV state adds four strictly past-only POE workflow features:
lab-order groups in the previous 6 and 24 hours, component order rows in the
previous 6 hours, and time since the latest lab order. MIMIC-IV does not provide
a shared key from `poe_id` to `specimen_id`, so these are independent workflow
features, not claimed order-to-draw matches. Stage 2 writes
`reports/poe_feature_validation.md`; pass `--exclude-poe-state` to stage 3 for
the original 21-dimensional physiology-only ablation.

**Action.** Binary per lab; four independent policies (creatinine, BUN, WBC,
lactate), so `L = 1` each.

**Reward**, a 4-vector (Eqs. 3-6): `[r_SOFA, r_treat, r_info, -r_cost]` —
ordering when the SOFA score jumped by ≥2, ordering immediately before an
intervention is initiated, ordering when the forecast has drifted from the last
known value by more than `c_l` standard deviations, and a redundancy penalty
decaying as `exp(-Delta/6)`.

**Learner** (Sec. 2.3). MO-FQI: one extra-trees regressor per reward dimension,
Pareto-dominated actions pruned each iteration, 200 iterations at `gamma = 0.9`,
100k transitions resampled per iteration inversely to action frequency. Collapsed
by Eq. 7 to a deterministic rule, then fit as an extra-trees classifier.

---

## What the paper does not report, and this does

**Effective sample size.** PS-WIS at `gamma_WIS = 1.0` over 24+ step admissions
multiplies importance ratios across the whole stay. The paper reports no ESS and
no confidence intervals, so there is no way to tell from it how many patients its
value estimates actually rest on. ESS is one line of arithmetic and it is
reported for every importance-weighted number here.

**A softened target policy.** The paper's final policy is deterministic, which
makes every ratio either 0 or `1/pi_b`. An epsilon-greedy version is evaluated
instead and the deterministic policy's ESS is reported alongside, so the size of
the problem is visible rather than hidden.

**An FQE calibration check.** FQE is re-run with the target policy set to the
*behavior* policy, where the right answer is known: it must land inside the
bootstrap interval of the observed factual return. If it does not, the Q model is
misspecified and its numbers on the learned policy mean nothing.

**Patient-level bootstrap intervals** on every estimate, resampling patients
rather than stays.

**The reward's gating structure, stated plainly.** Three of the four components
are gated on `a != 0` and the fourth only fires when `a = 1`, so *not* ordering
yields the zero vector in every dimension. A policy that orders more often is
mechanically advantaged on three of four objectives and penalised only through
cost. Any comparison of `V_d` across policies with different order rates is
partly a comparison of order rates. This is a property of the paper's reward
design, reproduced faithfully, and it is restated in every OPE report.

**Where Pareto pruning actually bites.** With a binary action space it is a
**no-op in the Bellman backup**: if one action dominates the other on every
objective, the per-dimension max over the surviving set equals the max over both.
Since the paper trains four `L = 1` policies, the same was true of its backups;
pruning shapes only the action set Eq. 7 collapses. The implementation is written
for general `|A|` so a joint action space would exercise it.

---

## The forecaster, and why it is not a random walk

The MOGP feeds exactly two things: `(m_t, sigma_t)` in the state, and the
`|m_t - y_t| / sigma_t` term in Eq. 5. So the contract is narrow, and
[`src/forecast.py`](src/forecast.py) defines it as an interface with two methods:

- `filter()` — uses observations strictly **before** hour `t`. The only thing the
  policy ever sees.
- `smooth()` — uses **all** observations. This is the paper's "approximated true
  value ... imputed given all the observed values" (Sec. 3.1), used **only** in
  the information-gain metric, never in the state.

The obvious substitution for the MOGP is a local level (random walk) model. It
does not work, for a reason worth stating: a random walk's optimal forecast is
flat at the last observation, so `|m_t - y_t|` collapses to zero and Eq. 5 returns
zero at every hour for every lab — the information reward dies. The paper is
explicit that the term should fire when the forecast "is significantly different
from the last known measurement due to a sudden change in disease state", which
requires extrapolation. The shipped `LocalTrendForecaster` therefore carries a
level *and* a slope. `MOGPForecaster` is a slot behind the same interface,
selected by `config.FORECASTER`, for a closer replication.

This remains a documented substitution: it drops the cross-trait covariance that
the "multi-output" in MOGP refers to.

---

## Data access (not included in this repository)

MIMIC-IV v3.1 is credentialed data and cannot be redistributed. Nothing derived
from individual patients is stored here.

1. Create a [PhysioNet](https://physionet.org/) account, complete the CITI "Data
   or Specimens Only Research" training, and sign the data use agreement.
2. Download **MIMIC-IV v3.1** (https://doi.org/10.13026/kpb9-mt58). The pipeline
   reads `icustays`, `patients`, `admissions`, `chartevents`, `labevents`,
   `inputevents`, `procedureevents` and `prescriptions`, as `.csv.gz`.
3. Point `MIMIC4_DIR` at the folder holding them:

```bash
export MIMIC4_DIR=/path/to/mimiciv/3.1          # Linux / macOS
```
```powershell
$env:MIMIC4_DIR = "C:\path\to\mimiciv\3.1"      # Windows PowerShell
```

If unset it defaults to `../mimiciv/3.1`. Every other path is derived in
[`src/config.py`](src/config.py).

---

## Layout

```
run_pipeline.py            runs the whole study
src/
  config.py                every path and hyperparameter, in one place
  itemids.py               MIMIC-IV itemid maps
  forecast.py              Forecaster interface, local trend model, MOGP slot
  sofa.py                  hourly SOFA from predictive means
  mofqi.py                 MO-FQI, Pareto pruning, budget rule
  s1_extract_cohort.py     cohort + chart/lab/intervention scan     (Sec. 2.1)
  s2_hourly_grid.py        1h resample, forecaster -> (mean, std)   (Sec. 2.1)
  s3_build_mdp.py          SOFA, 21-dim state, action, reward       (Sec. 2.2)
  s4_train_mofqi.py        MO-FQI, eps collapse, budget, policy fn  (Sec. 2.3, 3)
  s5_evaluate_ope.py       PS-WIS + FQE/WDR/bootstrap/ESS           (Sec. 3.1)
  s6_clinical_metrics.py   order reduction, info gain, lead time    (Sec. 3.1)
  s7_make_figures.py       analogues of Figures 2-6
tests/test_core.py         50 property tests, including leakage guards
reports/                   generated markdown + json
figures/                   generated png
```

No stage imports another; each reads what the previous one wrote.

---

## Running it

```bash
pip install -r requirements.txt

python run_pipeline.py                     # full study, all four labs
python run_pipeline.py --quick             # 200 stays, WBC only, smoke test
python run_pipeline.py --from 4            # resume from a stage
python run_pipeline.py --only 5            # one stage
python run_pipeline.py --labs wbc lactate  # restrict the per-lab stages
python tests/test_core.py                  # 50 property tests
```

For the joint MO-FQI experiment, stage 4 trains one shared vector-Q model and
extracts the default five preference policies from its action advantages:

```bash
python run_pipeline.py --track joint --family mofqi --from 4 --to 8
```

Use `--prefs` to provide another set of utility/burden pairs. The older mode
that fits a separate Bellman backup for every preference remains available with
`python src/s4e_train_mofqi_joint.py --per-preference-backup`.

To add POE to an existing cached cohort without rescanning `chartevents` or
`labevents`, then rebuild the hourly data and joint MDP:

```bash
python run_pipeline.py --only 1 --reuse-cache
python run_pipeline.py --track joint --from 2 --to 3
```

Inspect `reports/poe_feature_validation.md` before training. It reports POE
coverage, temporal association with later specimens, and the held-out change in
ROC-AUC, PR-AUC, and Brier score after adding past POE features.

Stage 1 scans `chartevents` (3.5 GB) and `labevents` (2.6 GB) once and caches the
filtered result to parquet; it is by far the slowest step and does not need to be
repeated. `--skip-scan` reuses the caches. The pipeline runs on CPU.

---

## Verification

`tests/test_core.py` checks properties rather than golden numbers, so it stays
meaningful when the cohort or hyperparameters change. Four of the checks are
leakage guards:

- `filter()` at hour `t` is unchanged when every observation after `t` is deleted,
  and `smooth()` is *not* (it is evaluation-only by design)
- `y_t` excludes the measurement taken in hour `t` itself, so the result of the
  order being decided is never in the state that decides it
- no patient appears in two splits
- the forecast extrapolates a trend, so Eq. 5 is not identically zero

Plus: each reward term against a hand-computed value, Pareto pruning on
dominated / mutually non-dominated / tied / three-action cases, the budget rule's
window and stay boundaries, SOFA at 0 and at 24, and ESS equal to `n` under
uniform weights and 1 under a point mass.

At runtime, the FQE calibration check in stage 5 is the end-to-end sanity test: if
FQE cannot recover the behavior policy's known return, its estimate for the
learned policy is not usable.

---

## Interpretation limits

- **MIMIC-IV is not MIMIC-III.** Different years (2008-2022), a different charting
  system, renumbered itemids, and far more eligible stays. The paper's exact
  figures will not reproduce; the target is method replication and the same
  directional findings.
- **The 20-trait completeness filter selects sicker patients.** Requiring a
  lactate, a bilirubin and a blood gas on every stay is close to requiring a
  sepsis workup. The paper's filter has the same property. Cohort SOFA runs
  higher than an unfiltered ICU population as a result.
- **The budget rule is outside the MDP.** It triggers on time since the last
  *recommended* order, which is not a state variable, so it is not a Markov policy
  and none of the stage-5 estimators apply to it. Stage 6 reports order counts
  with and without it.
- **Time-to-treatment is a lead-time comparison, not a counterfactual.** Nothing
  here establishes that an intervention would have started sooner, and a policy
  that orders more often has more chances to land early in the lookback window.
- **None of this is a clinical claim.** The rewards are proxies for clinical
  value, not outcomes.

---

## Citing

- Cheng L-F, Prasad N, Engelhardt BE. An Optimal Policy for Patient Laboratory
  Tests in Intensive Care Units. Pac Symp Biocomput. 2019;24:320-331.
- Johnson AEW, Bulgarelli L, Shen L, et al. MIMIC-IV, a freely accessible
  electronic health record dataset. Scientific Data. 2023;10:1.
- Goldberger AL, Amaral LAN, Glass L, et al. PhysioBank, PhysioToolkit, and
  PhysioNet. Circulation. 2000;101(23):e215-e220.

MIMIC-IV is governed by the PhysioNet Credentialed Health Data Use Agreement and
is **not** included in this repository.
