#!/bin/sh
# the learned tie-breaker: enumerate the net's own answers, train on the unique ones, score the Golden set
cd /home/daenu/Code/pgnn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES="" uv run python -m benchmarks.chemistry.tie_breaker build --limit 14000 --seconds 8 --processes 20 > results/chem/tie-build.log 2>&1
uv run python -m benchmarks.chemistry.tie_breaker train --epochs 8 > results/chem/tie-train.log 2>&1
uv run python -m benchmarks.chemistry.tie_breaker evaluate --seconds 20 --processes 12 > results/chem/tie-eval.log 2>&1
echo done > results/chem/tie-breaker.done
