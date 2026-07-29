# MetaboNet Offline Multi-Objective RL

Minimal interview-ready offline RL project for insulin decision support using the open-access MetaboNet parquet release.

## What It Does

- Builds an offline MDP from CGM, insulin, carbs, and metadata.
- Trains a Transformer behavior-cloning policy.
- Trains discrete Conservative Q-Learning policies under different clinical tradeoffs.
- Evaluates learned policies with FQE-style off-policy evaluation.
- Reports a policy-level Pareto frontier over hypoglycemia, hyperglycemia, and treatment burden.

Time-in-range is reported as a primary clinical metric, but the Pareto axes are the actual competing objectives:

- minimize severe hypoglycemia, `% CGM <54`
- minimize severe hyperglycemia, `% CGM >250`
- minimize treatment burden, bolus events + basal changes

## Run

```bash
python metabonet_offline_rl/run_pipeline.py \
  --stage all \
  --parquet /scratch/metabonet_public.parquet \
  --download-url "INSERT_METABONET_PARQUET_URL" \
  --max-transitions 300000
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
  --fqe-epochs 2
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
