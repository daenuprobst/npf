#!/bin/sh
# enumerate the ties of the exact mapper on the Golden set once the CPU queue is done
cd /home/daenu/Code/pgnn
until [ -f results/chem/exact_map/queue.done ]; do sleep 60; done
CUDA_VISIBLE_DEVICES="" uv run python -m benchmarks.chemistry.exact_ties --data golden > results/chem/exact_map/ties-golden.log 2>&1
CUDA_VISIBLE_DEVICES="" uv run python -m benchmarks.chemistry.balance --data schneider50k --processes 10 > results/chem/open_net_schneider.log 2>&1
