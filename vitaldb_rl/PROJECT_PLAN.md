# Offline RL for Anesthetic Infusion Control

## Goal

Build a small offline reinforcement learning project using VitalDB to recommend anesthetic infusion adjustments while balancing hemodynamic stability, anesthetic depth, and intervention burden.

This is framed as decision support and policy evaluation, not as a clinically validated controller.

## Data

Use VitalDB timestamped tracks:

- Physiology: MAP, SBP, DBP, HR, SpO2, ETCO2
- Anesthetic depth: BIS
- Treatments/actions: propofol rate, remifentanil rate
- Static features: age, sex, ASA, surgery/anesthesia metadata where available

Each case becomes a time-indexed trajectory:

```text
state_t -> clinician_action_t -> state_t+1 -> reward_t
```

## State, Action, Reward

Formulation:

The underlying clinical process is partially observed because the dataset does
not contain every factor affecting physiology, such as surgical stimulation,
fluid responsiveness, clinician intent, and all concurrent interventions.

For implementation, use a finite-horizon offline MDP approximation by defining
the state as a history window of recent observations:

```text
S_t -> A_t -> R_t -> S_t+1
```

State:

```text
last 15-30 minutes of vitals + BIS + drug rates
recent trends
missingness masks
static patient/surgery features
```

Action:

```text
propofol: decrease / hold / increase
remifentanil: decrease / hold / increase
combined action space: 9 actions
```

Reward objectives:

```text
minimize hypotension severity/duration
keep BIS near 40-60
minimize drug changes and intervention burden
```

## Algorithms

1. Behavior cloning baseline:
   Train a supervised policy to imitate clinician actions.

2. Offline RL:
   Train a conservative policy using IQL or CQL so the learned policy stays close to observed clinical behavior.

3. Pareto analysis:
   Sweep reward weights for MAP stability, BIS target control, and intervention burden to construct a policy tradeoff frontier.

4. Optional sequence encoder:
   Use GRU/TCN/Transformer encoder for patient history representation before policy learning.

## Evaluation

Compare:

- Clinician behavior policy
- Behavior cloning policy
- Offline RL policies under different reward weights
- Simple rule baselines, such as hold-only or MAP/BIS threshold rules

Metrics:

- Hypotension burden
- BIS out-of-range burden
- Number of drug changes
- Estimated policy value using FQE or conservative off-policy evaluation
- Action support: whether recommended actions were observed in similar states

## Interview Framing

This project directly demonstrates:

- Sequential decision modeling
- Offline RL with observational healthcare data
- Multi-objective optimization
- Pareto frontier analysis
- Practical safety limits through conservative policy learning and action-support checks

Main limitation:

The data contains observed clinician actions and outcomes, but not randomized counterfactual outcomes. Results should be presented as conservative offline policy evaluation, not proof of clinical superiority.
