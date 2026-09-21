# Neural Petri Flow

A neural network that is a Petri net rather than one that runs on a Petri net. The only learned object is the rate
law. Conservation, enabling and the state equation are parameter-free layers, so for every value of the weights the
outputs are markings of the given net and firing counts that satisfy its state equation.

A chemical reaction is written as a firing sequence of a valence net. Bond places hold bond orders, slack places hold
free valence, transitions make and break bonds. Forward prediction, atom mapping and reaction classification become
three questions about one firing vector.

## Install

    uv sync

## Layout

Models and data live in the package, one model per file. Everything that produces a number of the paper lives under
`benchmarks/` and is run as a module.

### Package

| File | What it does |
|---|---|
| `src/npf/nets.py` | attributed nets, incidence matrix, invariants, random and chain nets |
| `src/npf/simulate.py` | stochastic token game and continuous flow |
| `src/npf/datasets.py` | pairs of markings with firing counts, next-state samples |
| `src/npf/batching.py` | many nets as one disjoint net |
| `src/npf/layers.py` | enabling, conflict resolution, token game flow, state-equation projection |
| `src/npf/models/npf.py` | rate law, token game, bridge, projection, occupancy refinement |
| `src/npf/models/pgnn.py` | PGNN as published, and its variants |
| `src/npf/models/linear_pgnn.py` | the linear special case of the PGNN paper |
| `src/npf/models/gnn.py` | message passing on the place graph |
| `src/npf/models/state_equation_only.py` | projection without learning |
| `src/npf/models/equilibrium.py` | equilibrium layer, learned place energies, dual Newton |
| `src/npf/models/learned_incidence.py` | learned arc weights |
| `src/npf/models/coloured.py` | coloured tokens |
| `src/npf/chem/featurisation.py` | reactions as markings of the valence net |
| `src/npf/chem/encoder.py` | message passing on the valence net |
| `src/npf/chem/decode.py` | products, by-products and atom maps read from a marking |
| `src/npf/chem/token_game.py` | forward prediction as a firing sequence with enabling |
| `src/npf/chem/one_shot.py` | one-shot counterpart of the token game |
| `src/npf/chem/classifier.py` | state-equation readout, and the explicit firing vector |
| `src/npf/chem/mapper.py` | atom mapping as an equilibrium, Sinkhorn with a slack row |
| `src/npf/chem/exact.py` | exact minimum firing vector as an integer program |
| `src/npf/chem/open_net.py` | source transitions for reactions whose reactants the record omits |
| `src/npf/chem/minimise.py` | older branch and bound, superseded by `exact.py` |
| `src/npf/chem/orders.py` | enabled linearisations of a firing vector |
| `src/npf/chem/verifier.py` | re-ranker for token game candidates, no gain, kept for the record |

### Benchmarks

| File | What it does |
|---|---|
| `benchmarks/synthetic/experiment.py` | one run on one regime |
| `benchmarks/synthetic/sweep.py` | all models on one task |
| `benchmarks/synthetic/paper_protocol.py` | the protocol of the PGNN paper |
| `benchmarks/synthetic/locality.py` | chain nets of growing length |
| `benchmarks/synthetic/equilibrium.py` | equilibria of unseen reversible nets |
| `benchmarks/synthetic/learned_incidence.py`, `coloured.py` | extensions |
| `benchmarks/chemistry/build_data.py` | Schneider 50k and USPTO-MIT to `data/` |
| `benchmarks/chemistry/experiment.py` | forward, map and classify, all datasets |
| `benchmarks/chemistry/care.py` | EC number classification on CARE task 2 |
| `benchmarks/chemistry/exact_map.py` | the learning-free mapper on a data set |
| `benchmarks/chemistry/exact_map_report.py` | comparison with RXNMapper, intervals, sign test |
| `benchmarks/chemistry/exact_ties.py` | the mappings the net cannot tell apart |
| `benchmarks/chemistry/tie_breaker.py` | a learned rate law that orders those ties |
| `benchmarks/chemistry/balance.py` | balanced equations from the open net |
| `benchmarks/chemistry/golden.py` | the Golden atom mapping set |
| `benchmarks/chemistry/errors.py` | where the token game fails |
| `benchmarks/chemistry/decoding_rules.py`, `calibrate.py`, `ensemble.py`, `verify.py` | decoding and ranking |
| `benchmarks/chemistry/unsupervised_map.py`, `net_maps.py` | mapping without recorded maps |
| `benchmarks/chemistry/insights.py`, `invariance.py`, `figures.py`, `review.py` | analyses and figures |
| `benchmarks/chemistry/baselines/molecular_transformer.py` | the baseline, trained here |
| `benchmarks/checks/paper_checks.py` | numerical check of every proposition, exits non-zero on failure |
| `benchmarks/checks/data_facts.py` | the facts about the data that the paper quotes |
| `benchmarks/report.py` | `results/*.json` to `results/REPORT.md` |
| `benchmarks/paper_tables.py` | `results/REPORT.md` to `paper/tables/appendix_tables.tex` |
| `tests/` | unit tests, and equivalence tests against the code that produced earlier results |

## Reproduce

Tests and the proofs checked numerically.

    uv run pytest -q
    uv run python -m benchmarks.checks.paper_checks

Synthetic nets.

    uv run python -m benchmarks.synthetic.sweep --task transitions
    uv run python -m benchmarks.synthetic.sweep --task next --iters 3000
    uv run python -m benchmarks.synthetic.paper_protocol

Chemistry. The first two commands write `data/schneider50k.pkl` and `data/uspto_mit.pkl`.

    uv run python -m benchmarks.chemistry.build_data
    uv run python -m benchmarks.chemistry.build_data uspto-mit
    uv run python -m benchmarks.chemistry.experiment --task classify --model npf --seed 0
    uv run python -m benchmarks.chemistry.experiment --task forward --model npf --seed 0 \
        --dataset uspto_mit --width 256 --rounds 8 --attention 8 --amp

Atom mapping with no learning and no recorded map, then the comparison with RXNMapper.

    uv run python -m benchmarks.chemistry.exact_map --data golden --no-labile-h --ch-places
    uv run python -m benchmarks.chemistry.balance "CC(=O)Cl.NCCN>>CC(=O)NCCNC(C)=O"

EC numbers on CARE task 2. Download `CARE_datasets.zip` from the Zenodo record of the CARE benchmark and unpack it.

    uv run python -m benchmarks.chemistry.care prepare <CARE_datasets>
    uv run python -m benchmarks.chemistry.care train --model npf --seed 0


## Conventions

Results are JSON under `results/`, one file per run, and every table of the paper is generated from them. No number
in the paper is typed by hand. Side outputs go in their own folder, because `benchmarks/report.py` globs the result
folders.
