#!/bin/sh
# CARE task 2, easy split: three seeds of the state-equation readout, then the generic counterpart
cd /home/daenu/Code/pgnn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for s in 0 1 2; do
    uv run python -m benchmarks.chemistry.care train --model npf --seed $s > results/care/easy/npf-$s.log 2>&1
done
for s in 0 1 2; do
    uv run python -m benchmarks.chemistry.care train --model pgnn --seed $s > results/care/easy/pgnn-$s.log 2>&1
done
echo done > results/care/easy/seeds.done
