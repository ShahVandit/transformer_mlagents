"""
Run the whole study end to end.

    python run_pipeline.py                    # full pipeline, all four labs
    python run_pipeline.py --quick            # 200 ICU stays, one lab, smoke test
    python run_pipeline.py --from 4           # resume from a given stage
    python run_pipeline.py --only 5           # run a single stage
    python run_pipeline.py --labs wbc lactate # restrict the per-lab stages
    python run_pipeline.py --learner cql      # CQL arm instead of MO-FQI
    python run_pipeline.py --track joint      # joint-panel CQL frontier track
    python run_pipeline.py --reuse-cache      # reuse cached stage-1 scans

Stages map onto the paper (Cheng, Prasad & Engelhardt, PSB 2019):

    1  extract cohort     icustays + chart/lab/intervention scan     (Sec. 2.1)
    2  hourly grid        1h resample, forecaster -> (mean, std)     (Sec. 2.1)
    3  build MDP          SOFA, 21-dim state, action, 4-vec reward   (Sec. 2.2)
    4  train MO-FQI       vector Q, Pareto pruning, eps + budget     (Sec. 2.3, 3)
    5  evaluate OPE       PS-WIS (paper) + FQE/WDR/bootstrap/ESS     (Sec. 3.1)
    6  clinical metrics   order reduction, info gain, lead time      (Sec. 3.1)
    7  figures            analogues of Figures 2-6

Stages 4, 5 and 6 run once per lab. Every other stage runs once.

--learner selects which policy is trained at stage 4 and read by stages 5-7.
"mofqi" (default) is the paper's method. "cql" swaps in Conservative Q-Learning
on a scalarized reward; it is NOT part of the paper, and choosing it means
giving up the multi-objective claim. See src/s4b_train_cql.py.

--track joint keeps stages 1 and 2, then runs:
    3  s3b_build_joint_mdp.py
    4  s4c_train_family.py
    8  s8_frontier.py
"""
import argparse
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
PY = sys.executable

DEFAULT_LABS = ["creatinine", "bun", "wbc", "lactate"]

# (number, name, script, per_lab)
STAGES = [
    (1, "extract cohort", "s1_extract_cohort.py", False),
    (2, "hourly grid", "s2_hourly_grid.py", False),
    (3, "build MDP", "s3_build_mdp.py", False),
    (4, "train policy", "s4_train_mofqi.py", True),
    (5, "evaluate OPE", "s5_evaluate_ope.py", True),
    (6, "clinical metrics", "s6_clinical_metrics.py", True),
    (7, "make figures", "s7_make_figures.py", False),
]

JOINT_STAGES = [
    (1, "extract cohort", "s1_extract_cohort.py", False),
    (2, "hourly grid", "s2_hourly_grid.py", False),
    (3, "build joint MDP", "s3b_build_joint_mdp.py", False),
    (4, "train joint CQL family", "s4c_train_family.py", False),
    (8, "joint frontier", "s8_frontier.py", False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=1, help="first stage to run")
    ap.add_argument("--to", dest="end", type=int, default=7, help="last stage to run")
    ap.add_argument("--only", type=int, default=None, help="run just this stage")
    ap.add_argument("--labs", nargs="+", default=None, choices=DEFAULT_LABS,
                    help="labs to run the per-lab stages for (default: all four)")
    ap.add_argument("--learner", default="mofqi", choices=["mofqi", "cql"],
                    help="stage-4 learner; mofqi is the paper's method")
    ap.add_argument("--track", default="perlab", choices=["perlab", "joint"],
                    help="pipeline track to run")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: 200 ICU stays, WBC only, short FQI")
    ap.add_argument("--reuse-cache", action="store_true",
                    help="stage 1: reuse cached parquet scans and rebuild splits/cohort")
    args = ap.parse_args()

    start, end = (args.only, args.only) if args.only else (args.start, args.end)
    labs = args.labs or (["wbc"] if args.quick else DEFAULT_LABS)

    stages = JOINT_STAGES if args.track == "joint" else STAGES
    for num, name, script, per_lab in stages:
        if not (start <= num <= end):
            continue
        if args.track == "perlab" and num == 4 and args.learner == "cql":
            script, name = "s4b_train_cql.py", "train CQL"
        base = [PY, str(SRC / script)]
        if args.reuse_cache and num == 1:
            base += ["--reuse-cache"]
        if args.track == "perlab" and num in (5, 6, 7) and args.learner != "mofqi":
            base += ["--learner", args.learner]
        if args.quick and num == 1:
            base += ["--limit-icustays", "200"]
        if args.quick and num == 4:
            if args.track == "joint":
                base += ["--steps", "2000", "--lambdas", "0.1", "0.9"]
            else:
                base += (["--steps", "2000"] if args.learner == "cql"
                         else ["--iterations", "10"])
        if args.quick and args.track == "joint" and num == 8:
            base += ["--lambdas", "0.1", "0.9"]
        if args.track == "perlab" and num == 7:
            base += ["--labs"] + labs

        targets = labs if (args.track == "perlab" and per_lab) else [None]
        for lab in targets:
            cmd = base + (["--lab", lab] if lab else [])
            header = f"stage {num}  |  {name}" + (f"  |  {lab}" if lab else "")
            print(f"\n{'=' * 62}\n  {header}\n{'=' * 62}", flush=True)
            subprocess.run(cmd, check=True)

    print("\npipeline finished")


if __name__ == "__main__":
    main()
