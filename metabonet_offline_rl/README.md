# MetaboNet Offline Multi-Objective RL

Minimal interview-ready offline RL project for insulin decision support using the open-access MetaboNet parquet release.

## What It Does

- Builds an offline MDP from CGM, insulin, carbs, and metadata.
- Uses a 2-hour state window, a 30-minute bolus macro-action, and delayed 30-120 minute glucose response.
- Optionally trains an MLP behavior-cloning policy over engineered 2-hour state features.
- Trains discrete Conservative Q-Learning policies under different clinical tradeoffs.
- Evaluates learned policies with FQE-style off-policy evaluation.
- Reports a policy-level Pareto frontier over glycemic effectiveness and low treatment burden.

Hypoglycemia is reported as the CMDP safety constraint, not as a Pareto axis.

- maximize glycemic effectiveness, reducing future high-glucose exposure
- maximize low treatment burden, using fewer/lower bolus interventions
- constrain hypoglycemia risk, `% CGM <70` and `% CGM <54`

## Run

```bash
python metabonet_offline_rl/run_pipeline.py \
  --stage all \
  --parquet /scratch/metabonet_public.parquet \
  --download-url "INSERT_METABONET_PARQUET_URL" \
  --max-transitions 300000 \
  --history-steps 24 \
  --action-steps 6 \
  --reward-delay-steps 6 \
  --reward-steps 18 \
  --stride-steps 6 \
  --action-mode bolus4 \
  --encoder mlp \
  --skip-bc
```

If the parquet already exists, `--download-url` is ignored.

For a quick smoke test:

```bash
python metabonet_offline_rl/run_pipeline.py \
  --stage all \
  --parquet /scratch/metabonet_public.parquet \
  --max-transitions 50000 \
  --bc-epochs 2 \
  --cql-epochs 2 \
  --fqe-epochs 2 \
  --history-steps 24 \
  --action-steps 6 \
  --reward-delay-steps 6 \
  --reward-steps 18 \
  --stride-steps 6 \
  --action-mode bolus4 \
  --encoder mlp \
  --skip-bc
```

## d3rlpy Run

Use this path for the d3rlpy port. It uses ordered episodes and four train-quantile total-insulin actions, not bolus-only actions.

```bash
python metabonet_offline_rl/run_d3rlpy_pipeline.py \
  --stage all \
  --parquet /scratch/metabonet_public.parquet \
  --max-transitions 300000 \
  --n-steps 50000 \
  --train-batch-size 1024 \
  --device cuda:0
```

FQE/Pareto is intentionally separate because it is slower:

```bash
python metabonet_offline_rl/run_d3rlpy_pipeline.py \
  --stage fqe \
  --parquet /scratch/metabonet_public.parquet \
  --fqe-steps 30000 \
  --device cuda:0
```

## Outputs

Outputs are written to `metabonet_offline_rl/results/`.

- `action_distribution.csv`
- `bc_metrics.csv`
- `cql_metrics.csv`
- `policy_values.csv`
- `pareto_policies.csv`
- `pareto_frontier.png`
- `observed_group_values.csv`
- `final_report.md`

## Ground Truth

Observed glucose outcomes are factual ground truth only for logged insulin decisions and deployed controller/treatment groups. Learned-policy outcomes are not directly observed. They are estimated with FQE and interpreted as off-policy estimates, not clinical proof.
