#!/bin/sh
# the generic counterpart of the firing vector classifier, same maps, same width, one term per seated atom
cd /home/daenu/Code/pgnn
export NPF_WORKERS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for s in 0 1 2 3 4; do
    uv run python -m benchmarks.chemistry.experiment --task classify --model pgnn-sigma --seed $s > results/chem/classify-pgnn-sigma-$s.log 2>&1
done
echo done > results/chem/classify-pgnn-sigma.done
