# Neural Petri Flow

A neural network that is a Petri net rather than one that runs on a Petri net. The learned parts are the rate law and,
for classification, a readout of the firing. Conservation, enabling and the state equation are parameter-free layers,
so for every value of the weights the outputs are markings of the given net and firing counts that satisfy its state
equation.

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
| `src/npf/chem/targets.py` | training targets from the net, and the scoring of a marking against the recorded product |
| `src/npf/chem/mapper.py` | the atom mapper, the minimum firing vector without a solver, all its ties, and the third level that chooses among them |
| `src/npf/chem/search.py` | the branch and bound behind the mapper, bounded by linear assignments on the places of the net |
| `src/npf/chem/third_level.py` | the three counts that rank maps of equal cost, oxidation state of carbon, aromatic bonds, electron sinks |
| `src/npf/chem/cost.py` | the two levels of the cost of a mapping |
| `src/npf/chem/cgr.py` | the condensed graph of reaction, which merges maps into classes and scores a map against a reference |
| `src/npf/chem/mapping.py` | the maps of the mapper as SMILES with map numbers |
| `src/npf/chem/api.py` | train, validate, test and use the models on your own reactions, with PyTorch Lightning |
| `src/npf/chem/exact.py` | the same minimum as an integer program with CP-SAT, the first version of the mapper, kept as the reference the search is checked against |
| `src/npf/chem/open_net.py` | source transitions for reactions whose reactants the record omits |
| `src/npf/chem/minimise.py` | a heuristic seating, the start of the integer program |
| `src/npf/chem/orders.py` | enabled linearisations of a firing vector |
| `src/npf/chem/verifier.py` | re-ranker for token game candidates, no gain, kept for the record |
| `src/npf/chem/arrows.py` | a mechanistic step as a net of electron pairs, arrows as transitions, the octet rule as enabling |
| `src/npf/chem/arrow_game.py` | elementary steps as firing sequences of that net |

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
| `benchmarks/chemistry/experiment.py` | forward and classify, all datasets |
| `benchmarks/chemistry/care.py` | EC number classification on CARE task 2 |
| `benchmarks/chemistry/exact_map.py` | the mapper on a data set, and the maps the classifier reads |
| `benchmarks/chemistry/exact_map_report.py` | comparison with RXNMapper, intervals, sign test |
| `benchmarks/chemistry/synrxn_map.py` | the learning-free mapper on the five SynRXN sets, scored with SynKit like the published mappers |
| `benchmarks/chemistry/mechanism.py` | elementary steps of the FlowER mechanism benchmark, `--net arrow` or `--net electron` |
| `benchmarks/chemistry/mechanism_validity.py` | untrained arrow and electron nets, with and without the octet rule, end only in valid molecules |
| `benchmarks/chemistry/exact_ties.py` | the mappings the net cannot tell apart |
| `benchmarks/chemistry/balance.py` | balanced equations from the open net |
| `benchmarks/chemistry/golden.py` | the Golden atom mapping set |
| `benchmarks/chemistry/errors.py` | where the token game fails |
| `benchmarks/chemistry/decoding_rules.py`, `calibrate.py`, `ensemble.py`, `verify.py` | decoding and ranking |
| `benchmarks/chemistry/net_targets.py` | forward training targets from the net, every minimum firing vector that reaches the product |
| `benchmarks/chemistry/insights.py`, `invariance.py`, `figures.py` | analyses and figures |
| `benchmarks/chemistry/baselines/molecular_transformer.py` | the baseline, trained here |
| `benchmarks/checks/paper_checks.py` | numerical check of every proposition, exits non-zero on failure |
| `benchmarks/checks/data_facts.py` | the facts about the data that the paper quotes |
| `benchmarks/report.py` | `results/*.json` to `results/REPORT.md` |
| `benchmarks/paper_tables.py` | `results/REPORT.md` to `paper_v2/tables/appendix_tables.tex` (chemistry) and `paper_v2/tables/synthetic_tables.tex` |
| `tests/` | unit tests and end-to-end runs of the experiment scripts |

## Reproduce

Tests, the propositions checked numerically, and the facts about the data that the text quotes.

    uv run pytest -q
    uv run python -m benchmarks.checks.paper_checks
    uv run python -m benchmarks.checks.data_facts

Synthetic nets (Appendix E). The `npf-kl` rows come from `benchmarks.synthetic.experiment` with `--model npf-kl` on every
regime of the sweep.

    uv run python -m benchmarks.synthetic.sweep --task transitions --workers 10
    uv run python -m benchmarks.synthetic.sweep --task next --iters 3000 --workers 10
    uv run python -m benchmarks.synthetic.paper_protocol
    uv run python -m benchmarks.synthetic.locality
    uv run python -m benchmarks.synthetic.equilibrium
    uv run python -m benchmarks.synthetic.learned_incidence
    uv run python -m benchmarks.synthetic.coloured

Data. The Golden set is the RDF of Lin et al. (2022).

    uv run python -m benchmarks.chemistry.build_data                    # data/schneider50k.pkl
    uv run python -m benchmarks.chemistry.build_data uspto-mit          # data/uspto_mit.pkl
    uv run python -m benchmarks.chemistry.golden prepare <golden.rdf>   # data/golden.pkl, data/golden_unmapped.txt

Atom mapping. The mapper of the paper on the Golden set and on EnzymeMap, with its third level, then the chosen cost and
its alternatives, the 200 dev reactions on which the cost was chosen, the ties, the five SynRXN sets, the open net, and
RXNMapper on the same reactions, which runs in an environment of its own. With the chosen cost `exact_map` runs the
mapper of the paper, with its third level, and `--no-third-level` keeps the first optimum. Add `--solver cp-sat` to
`exact_map` or `synrxn_map` for the integer program of the first version, the only path that loads OR-tools.

    uv run --no-project --python 3.11 --with rxnmapper --with rdkit --with "setuptools<81" --with "numpy<2" python benchmarks/chemistry/baselines/rxnmapper_golden.py
    uv run python -m benchmarks.chemistry.exact_map --data golden --third-level
    uv run python -m benchmarks.chemistry.exact_map --data enzymemap_3k --third-level
    uv run python -m benchmarks.chemistry.exact_map_report results/chem/exact_map/enzymemap_3k-1-1-1-nolabile-ch-third.pkl results/chem/rxnmapper_enzymemap_3k.json
    uv run python -m benchmarks.chemistry.exact_map --data golden-dev --labile-h --no-ch-places
    uv run python -m benchmarks.chemistry.exact_map --data golden-dev --no-ch-places
    uv run python -m benchmarks.chemistry.exact_map --data golden-dev --no-third-level
    uv run python -m benchmarks.chemistry.exact_map --data golden --secondary 0,0,0 --labile-h --no-ch-places
    uv run python -m benchmarks.chemistry.exact_map --data golden --labile-h --no-ch-places
    uv run python -m benchmarks.chemistry.exact_map --data golden --no-ch-places
    uv run python -m benchmarks.chemistry.exact_map --data golden --no-third-level
    uv run python -m benchmarks.chemistry.exact_ties
    uv run python -m benchmarks.chemistry.synrxn_map map --seconds 60
    uv run --with "synkit>=1.5,<1.6" python -m benchmarks.chemistry.synrxn_map score --seconds 60
    uv run python -m benchmarks.chemistry.balance --data golden
    uv run python -m benchmarks.chemistry.balance --data schneider50k
    uv run python -m benchmarks.chemistry.balance "CC(=O)Cl.NCCN>>CC(=O)NCCNC(C)=O"

Targets of the net. The maps that classification reads, and the firing vectors that forward prediction trains on.
`experiment --task forward` builds a missing target file itself, so these commands only build ahead of time. The
forward runs of the paper trained on the files of the integer program, kept as `data/net_targets_<name>-cpsat.pkl`,
pass them with `--net-targets` to reproduce those numbers.

    uv run python -m benchmarks.chemistry.exact_map --data schneider50k --third-level --processes 12 --write data/exact_maps_schneider50k.pkl
    uv run python -m benchmarks.chemistry.net_targets --dataset schneider50k
    uv run python -m benchmarks.chemistry.net_targets --dataset uspto_mit --subset 40900
    uv run python -m benchmarks.chemistry.net_targets --dataset uspto_mit

Classification on Schneider 50k. Seeds 0 to 4 for the three models of the main table, 0 to 2 elsewhere.

    C="uv run python -m benchmarks.chemistry.experiment --task classify"
    $C --model npf --seed 0                                          # also npf-sigma, pgnn-sigma, pgnn, npf-nogate, drfp
    $C --model npf --size-split --seed 0                             # also npf-nogate, pgnn
    $C --model npf --labels 1000 --select last --seed 0              # also 250 labels; pgnn, npf-sigma, pgnn-sigma, drfp
    $C --model npf --firing --labels 1000 --select last --seed 0     # also 250 labels; pgnn
    uv run python -m benchmarks.chemistry.ensemble npf-sigma         # also npf
    uv run python -m benchmarks.chemistry.invariance
    uv run python -m benchmarks.chemistry.insights

Forward prediction on Schneider 50k, seeds 0 to 2.

    F="uv run python -m benchmarks.chemistry.experiment --task forward"
    $F --model npf --seed 0                       # also npf-noenabling, npf-oneshot --matched, pgnn --matched
    $F --model npf --limit 2000 --seed 0          # also 8000; npf-noenabling, pgnn --matched
    uv run python -m benchmarks.chemistry.beam_width
    uv run python -m benchmarks.chemistry.figures tokengame 14440
    uv run python -m benchmarks.chemistry.figures attribution
    uv run python -m benchmarks.chemistry.figures loadbearing

Forward prediction on USPTO-MIT, seeds 0 to 2, and the Molecular Transformer baseline on the same subsets.

    M="uv run python -m benchmarks.chemistry.experiment --task forward --model npf --dataset uspto_mit --amp"
    $M --width 256 --rounds 8 --attention 8 --lr 4e-4 --tag=-deep --seed 0
    $M --width 256 --rounds 8 --attention 8 --lr 4e-4 --tag=-deep --recorded-maps --seed 0
    $M --subset 40900 --seed 0                                            # also --single-target, --subset 4090, --recorded-maps
    uv run python -m benchmarks.chemistry.decoding_rules results/uspto_mit/forward/npf-deep-nettargets-0.pt --dataset uspto_mit --width 256 --rounds 8 --attention 8
    uv run python -m benchmarks.chemistry.baselines.molecular_transformer prepare 40900    # then train 40900 --steps 30000 and score 40900; also 4090

Elementary steps on the FlowER mechanism benchmark, seeds 0 to 2.

Both models are token games. Given the reactants of one elementary step they fire arrows until STOP, and the marking
reached is the predicted products. The arrow net moves an electron pair per firing, so its transitions are the curly
arrows of arrow pushing and the octet rule is what enables them. The electron net moves one electron, so a fishhook is
a transition of weight one, radical steps become expressible, and the arrow net is its sub-net of weight two.
Molecules are Kekulé structures: tokens are electrons, and an aromatic bond of order 3/2 would hold three, half a pair.
Products are compared after aromaticity is perceived again, so the Kekulé structure chosen does not change the score.

The download stage fetches the published split from figshare and checks it, so a fresh machine needs nothing else.
Each net keeps its own prepared data, weights and results. The results in `results/mechanism` are seeds 0 to 2 of both
nets, seed 0 evaluated with a beam of 10 and seeds 1 and 2 with a beam of 5.

    uv run python -m benchmarks.chemistry.mechanism download
    uv run python -m benchmarks.chemistry.mechanism prepare --net electron --processes 20
    for seed in 0 1 2; do
      uv run python -m benchmarks.chemistry.mechanism train --net electron --seed $seed --epochs 12 --budget 2000000
      uv run python -m benchmarks.chemistry.mechanism evaluate --net electron --seed $seed --beam 10
    done
    uv run python -m benchmarks.chemistry.mechanism prepare --processes 20          # the arrow net, the same stages

Validity for all weights, untrained games of both nets with and without the octet rule as enabling, on 1,000 test
steps and 3 seeds.

    uv run python -m benchmarks.chemistry.mechanism_validity                      # results/mechanism/validity.json

The report and the tables of the paper.

    uv run python -m benchmarks.report > results/REPORT.md && uv run python -m benchmarks.paper_tables

## Your own reactions

The models train on reaction SMILES without atom maps. A file holds one reaction per line, `precursors>>product`, with
an optional tab separated class label. The mapper computes the targets once per file, milliseconds per reaction for
most reactions, and caches them next to it.

    from npf.chem import api

    data = api.ReactionData("train.txt", "val.txt", "test.txt")
    model = api.ForwardModel()                                 # width=256, rounds=8, attention=8 is the USPTO-MIT model
    trainer = api.trainer(model, epochs=60)
    trainer.fit(model, data)
    trainer.test(model, data, ckpt_path="best")
    api.predict(model, ["CC(=O)Cl.NCC"])               # ranked products with probabilities
    api.map_reaction("CC(=O)Cl.NCC>>CC(=O)NCC")        # the reaction with atom map numbers

    data = api.ReactionData("train.txt", "val.txt", "test.txt", task="classify")
    model = api.ClassifierModel(data.n_classes, sigma=True)   # the classifier of the paper, reads the mapper's firing vector
    api.classify(model, ["CC(=O)Cl.NCC>>CC(=O)NCC"], data.classes)

`api.trainer` is a Lightning trainer with the settings of the paper, AdamW, a one-cycle schedule, gradient clipping and
the best epoch kept. Any Lightning trainer works, and `ForwardModel.load_from_checkpoint` reloads a run. Without
`sigma` the classifier is the state-equation readout, which needs no mapper at test time. `map_reaction` and the
training targets use the mapper of the paper, `npf.chem.mapper`, the minimum firing vector of the chosen cost, found and
proved by a branch and bound without a solver, with a third level that chooses among its ties. The integer program of
`npf.chem.exact` gives the same minimum and stays as a reference, and the benchmark scripts run it with `--solver cp-sat`.

## Conventions

Results are JSON under `results/`, one file per run, and every table of the paper is generated from them. No number
in the paper is typed by hand. Side outputs go in their own folder, because `benchmarks/report.py` globs the result
folders.

## Changelog

Since commit 9d3986f.

- The electron net ends a step only with whole pairs on every bond place, so an untrained game can no longer stop on a
  one-electron bond. No recorded FlowER step ends on one. `mechanism_validity.py` checks validity for all weights.
- The octet capacities of the arrow net are in pairs. The tables halved the shell, which the arrow net already counts in
  pairs, so C, N and O had a capacity of 2 and 6.8 % of the test steps had no enabled order; now none has.
- The mapper without a solver, `mapper.py`, `search.py`, `third_level.py`. A branch and bound, bounded by linear
  assignments on the places of the net, finds and proves the same minimum as the integer program on every Golden and
  SynRXN reaction, 20 to 60 times faster, and lists all ties completely. The listing of the integer program could stop
  at its budget and still report a proof. A third level, the change of oxidation state of carbon, bonds switched
  between aromatic carbon and heteroatoms, and exchanges at electron sinks, chooses among the ties, and the class with
  the smallest graph hash breaks what remains, so the map does not depend on atom order. On the Golden set it maps
  88.8 % against 85.6 % for RXNMapper, before 83.8 %. The cost moved to `cost.py` and the condensed graph to `cgr.py`,
  `exact.py` re-exports them. `exact_map.py`, `exact_ties.py` and `synrxn_map.py` use the search by default and write
  the integer program's results with `-cpsat` or into their old folders. The training targets of `targets.py`, the
  API and the maps the classifier reads use the new mapper, and the classification runs were repeated on them. The
  forward targets of the paper's runs were built with the integer program.
- Mechanism prediction on the FlowER benchmark with the arrow net, in `arrows.py`, `arrow_game.py` and `mechanism.py`.
- The electron net, `electron.py` and `electron_game.py`, one electron as the token instead of one pair, so a fishhook
  arrow is a transition of weight one and radical steps are expressible. The arrow net is its sub-net of weight two.
  Chosen with `--net electron`, which keeps its own prepared data, weights and results.
- The tail of an arrow keeps its electron count, so `arrows.Orders.delta` no longer moves the shell of the tail. Every
  test step of the FlowER split now has an enabled order, 44 of them had none.
- The learning-free mapper on the five SynRXN sets, in `synrxn_map.py`, with `--retries` to double the solver budget
  until optimality is proved.
- `paper_tables.py` writes chemistry and synthetic tables to separate files. `report.py` uses the names of the paper and
  lists the run without the enabling mask.
- `paper_checks.py` numbers the propositions as the paper does. The Sinkhorn check and a valence check that could not
  fail are gone.
- One atom mapper. The learned assignment net, its unsupervised variant, the tie-breaker, the older heuristic search as a
  mapper and the review package are removed. The heuristic survives as the feasible start of the integer program.
- Forward prediction trains on targets of the net (`net_targets.py`, `--net-targets`), with a loss over the whole set of
  minimum firing vectors. The loss leaves out states in which no correct firing is enabled, where it was unbounded.
- Classification reads the maps of the exact mapper everywhere (`--maps`), and `--select last` keeps the last epoch
  without looking at validation labels.
- Forward results carry `product_major`, the largest molecule made against the recorded main product.
- The Golden scorer matches RXNMapper's molecules as graphs when double-bond stereo reorders the atoms.
- The one-shot repair uses the valence rule of the token game. The substitution events of the token game are removed.
- Old-code equivalence tests are removed with the old code, the end-to-end tests run the current scripts alone.
- On USPTO-MIT both kinds of targets share the size caps and the validation reactions. The mapper, the ties and the open
  net run on a budget in deterministic time with interleaved solver workers, so their maps do not depend on the load of the
  machine, and `exact_map` defaults to the chosen cost.
- An API for your own reactions, `npf.chem.api`, on PyTorch Lightning. The targets of the net, the scoring and the
  mapped SMILES moved into the package (`targets.py`, `mapping.py`), the benchmark scripts read them from there.
- The Molecular Transformer is scored by exact match, by the largest molecule and under the rule of the token game.
  `report.py` follows the new run names and reports the mapper with the chosen cost. `data_facts` reads the targets of
  the net and the USPTO-MIT test facts. RXNMapper's run on the Golden set has a script, `baselines/rxnmapper_golden.py`.
