#!/bin/sh
# the I-projection variant of NPF on the transition task, three seeds with the final code
cd /home/daenu/Code/pgnn
for s in 0 1 2; do
    uv run python -m benchmarks.synthetic.experiment --task transitions --regime petri --model npf-kl --seed $s > results/transitions-petri-npf-kl-$s.log 2>&1
done
echo done > results/transitions-petri-npf-kl.done
