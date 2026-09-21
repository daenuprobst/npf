#!/bin/sh
# GPU queue for the refactored tree, to be started after swap.sh once the two training runs have finished.
cd /home/daenu/Code/pgnn
export NPF_WORKERS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=results/uspto_mit

# where the deep token game fails, on the official test set
uv run python -m benchmarks.chemistry.errors $R/forward/npf-deep-0.pt --dataset uspto_mit --width 256 --rounds 8 --attention 8 > $R/errors-npf-deep-0.log 2>&1

# what the decoding rules are worth for the same network
uv run python -m benchmarks.chemistry.decoding_rules $R/forward/npf-deep-0.pt --dataset uspto_mit --width 256 --rounds 8 --attention 8 > $R/decoding-rules-npf-deep-0.log 2>&1

# Golden atom mapping set with the mapper trained on the official training split
uv run python -m benchmarks.chemistry.golden evaluate results/chem/rxnmapper_golden.json $R/map/npf-mit-0.pt $R/golden.json > $R/golden-npf-mit-0.log 2>&1

# the same for the mapper trained on Schneider 50k, in the current result format
uv run python -m benchmarks.chemistry.golden evaluate results/chem/rxnmapper_golden.json results/chem/map/npf-0.pt results/chem/golden.json > results/chem/golden-npf-0.log 2>&1

# classification from the explicit firing vector with the auxiliary firing task, which crashed before the fix
for s in 0 1 2 3 4; do
    uv run python -m benchmarks.chemistry.experiment --task classify --model npf-sigma --firing --seed $s > results/chem/classify-npf-sigma-$s-firing.log 2>&1
done

# top-k by beam search on the official test set, keeps the training record of the run
uv run python -m benchmarks.chemistry.experiment --task forward --model npf --seed 0 --dataset uspto_mit --width 256 --rounds 8 --attention 8 --amp --tag=-deep --evaluate-only > $R/forward-npf-deep-0-beam.log 2>&1
# (a verifier that re-ranks the beam, benchmarks.chemistry.verify, brought no gain on Schneider 50k and is not run here)

# data efficiency, the Molecular Transformer on the same nested subsets of the training file as the token game
# (the token game runs on the subsets were started earlier from the staged tree, see results/uspto_mit/forward-npf-sub*.log)
for n in 4090 40900; do
    uv run python -m benchmarks.chemistry.baselines.molecular_transformer train $n --steps 30000 > $R/mt-sub$n-train.log 2>&1
    uv run python -m benchmarks.chemistry.baselines.molecular_transformer score $n > $R/mt-sub$n-score.log 2>&1
done
echo done > $R/queue.done
