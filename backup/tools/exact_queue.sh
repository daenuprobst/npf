#!/bin/sh
# exact net mapper, CPU queue. Golden set with the chosen cost, then the maps of Schneider 50k for the classifier,
# then the other variants of the cost on the Golden set as ablations
cd /home/daenu/Code/pgnn
export CUDA_VISIBLE_DEVICES=""
E=results/chem/exact_map
uv run python -m benchmarks.chemistry.exact_map --data golden --no-labile-h --ch-places --seconds 20 --processes 6 --workers 4 > $E/golden-nolabile-ch.log 2>&1
uv run python -m benchmarks.chemistry.exact_map --data schneider50k --no-labile-h --ch-places --seconds 3 --processes 22 --workers 1 --hints data/net_maps_schneider50k.pkl --write data/exact_maps_schneider50k.pkl > $E/schneider50k.log 2>&1
uv run python -m benchmarks.chemistry.exact_map --data golden --seconds 20 --processes 6 --workers 4 > $E/golden-plain.log 2>&1
uv run python -m benchmarks.chemistry.exact_map --data golden --no-labile-h --seconds 20 --processes 6 --workers 4 > $E/golden-nolabile.log 2>&1
uv run python -m benchmarks.chemistry.exact_map --data golden --secondary 0,0,0 --seconds 20 --processes 6 --workers 4 > $E/golden-level1.log 2>&1
echo done > $E/queue.done
