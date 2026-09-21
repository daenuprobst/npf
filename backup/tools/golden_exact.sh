#!/bin/sh
# the exact net mapper on the whole Golden set, started once the CPU is free of the Schneider maps
cd /home/daenu/Code/pgnn
until [ -f results/chem/exact_map/schneider50k-1-1-1.json ]; do sleep 60; done
CUDA_VISIBLE_DEVICES="" uv run python -m benchmarks.chemistry.exact_map --data golden --secondary 1,1,1 --seconds 20 --processes 6 --workers 4 > results/chem/exact_map/golden-1-1-1.log 2>&1
CUDA_VISIBLE_DEVICES="" uv run python -m benchmarks.chemistry.exact_map --data golden --secondary 0,0,0 --seconds 20 --processes 6 --workers 4 > results/chem/exact_map/golden-0-0-0.log 2>&1
