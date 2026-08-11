# Clinical metrics: wbc

Test split, 5,374 ICU stays, 806,295 hourly decisions.

## 1. Order counts (paper Sec. 3.1)

| policy | orders | vs clinician |
|---|---|---|
| clinician | 54,974 | - |
| policy (raw) | 56,965 | +3.6% |
| policy (run-filtered) | 21,397 | -61.1% |
| policy (+ 24h budget) | 46,309 | -15.8% |

`run-filtered` applies the paper's rule of counting only the first recommendation in a run with no clinician order in between: later recommendations in a run are made from a state that was never updated with the first draw's result, so they are not independent decisions. `+ 24h budget` then forces one order per 24-hour window the policy leaves silent. The headline reduction the paper quotes corresponds to the run-filtered row.

## 2. Information gain per order (paper Fig. 5)

| | n orders | mean | median |
|---|---|---|---|
| clinician (raw) | 54,974 | 3.4449 | 1.8715 |
| MO-FQI (raw) | 46,309 | 3.0357 | 1.3202 |
| clinician (sigma-normalized) | 54,974 | 0.4782 | 0.3229 |
| MO-FQI (sigma-normalized) | 46,309 | 0.3922 | 0.2330 |

The approximate true value comes from the forecaster's SMOOTHER, which uses every observation in the stay including future ones. That is legitimate here and only here: this metric scores a decision after the fact rather than informing one. The same quantity never enters the state vector.

## 3. Time to treatment onset (paper Fig. 6)

Lookback window 48h, over initiations of vasopressors, antibiotics, ventilation or dialysis.

| | n onsets matched | mean hours | median hours |
|---|---|---|---|
| clinician | 13,829 | 21.51 | 22.00 |
| MO-FQI | 16,852 | 18.31 | 13.00 |

A larger number means the order preceded the intervention by longer, which the paper reads as the lab having been available earlier to inform it. Note this is a lead-time comparison between two order schedules, not evidence that the intervention would actually have started sooner: nothing here is counterfactual, and a policy that simply orders more often has more chances to land early in the window.
