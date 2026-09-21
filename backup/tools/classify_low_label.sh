#!/bin/sh
# is the firing vector head load bearing when labels are scarce. Same maps and width for both heads
cd /home/daenu/Code/pgnn
export NPF_WORKERS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for n in 250 1000; do
    for s in 0 1 2; do
        for m in npf-sigma pgnn-sigma; do
            uv run python -m benchmarks.chemistry.experiment --task classify --model $m --labels $n --seed $s > results/chem/classify-$m-$s-labels$n.log 2>&1
        done
    done
done
echo done > results/chem/classify-low-label.done
