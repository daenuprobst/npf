"""Run every (task, regime, model, seed) combination that has no result file yet, a few at a time.

    uv run python -m benchmarks.synthetic.sweep --task transitions --workers 10
"""
import argparse
import itertools
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .experiment import REGIMES

MODELS = {
    "transitions": ["se-only", "npf", "npf-prior", "npf-mlp", "pgnn", "pgnn-eq10", "pgnn+", "pgnn+se", "gnn"],
    "next": ["npf", "npf@16", "pgnn", "pgnn+", "pgnn+se", "gnn"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(REGIMES), required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--root", default="results")
    ap.add_argument("--regimes", nargs="*", help="default: all regimes of the task")
    ap.add_argument("--models", nargs="*", help="default: all models of the task")
    args = ap.parse_args()
    regimes, names = args.regimes or list(REGIMES[args.task]), args.models or MODELS[args.task]

    base = [sys.executable, "-m", "benchmarks.synthetic.experiment", "--task", args.task, "--root", args.root, "--iters", str(args.iters)]

    # build the datasets once, before the workers race for them
    for regime in regimes:
        subprocess.run(base + ["--regime", regime, "--model", "npf", "--prepare"], check=True)

    jobs = [
        base + ["--regime", regime, "--model", model, "--seed", str(seed)]
        for regime, model, seed in itertools.product(regimes, names, range(args.seeds))
        if not (Path(args.root) / args.task / regime / f"{model}-{seed}.json").exists() and (model != "se-only" or seed == 0)
    ]

    def run(cmd):
        done = subprocess.run(cmd, capture_output=True, text=True)
        print(done.stdout.strip().splitlines()[-1] if done.returncode == 0 and done.stdout.strip() else f"FAILED {' '.join(cmd[3:])}\n{done.stderr[-2000:]}", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(run, jobs))


if __name__ == "__main__":
    main()
