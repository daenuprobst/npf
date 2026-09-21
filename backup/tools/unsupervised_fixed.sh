#!/bin/sh
# the mapper trained without atom maps, with the corrected M step and the last epoch kept, then scored on the Golden set
cd /home/daenu/Code/pgnn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv run python -m benchmarks.chemistry.unsupervised_map --candidates data/candidates.pkl --epochs 12 --last-epoch > results/chem/map-npf-unsupervised-last-0.log 2>&1
uv run python -m benchmarks.chemistry.golden evaluate results/chem/rxnmapper_golden.json results/chem/map/npf-unsupervised-last-0.pt results/chem/golden-unsupervised.json > results/chem/golden-unsupervised.log 2>&1
echo done > results/chem/unsupervised-fixed.done
