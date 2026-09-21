#!/bin/sh
# classification with firing vectors from the exact net mapper, no recorded map anywhere. Waits for the maps.
cd /home/daenu/Code/pgnn
export NPF_WORKERS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=data/exact_maps_schneider50k.pkl
until [ -f results/chem/exact_map/schneider50k-1-1-1-nolabile-ch.json ]; do sleep 60; done
for s in 0 1 2 3 4; do
    uv run python -m benchmarks.chemistry.experiment --task classify --model npf-sigma --firing --net-maps $M --maps $M --tag=-exact --seed $s > results/chem/classify-npf-sigma-$s-exact.log 2>&1
done
# the gated readout with its gate supervised by the same maps, so that this row is free of recorded maps as well
for s in 0 1 2 3 4; do
    uv run python -m benchmarks.chemistry.experiment --task classify --model npf --net-maps $M --tag=-exact --seed $s > results/chem/classify-npf-$s-exact.log 2>&1
done
echo done > results/chem/classify-exact.done
