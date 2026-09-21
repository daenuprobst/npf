"""Replace the module docstrings of the benchmark scripts by hand-written ones."""
import ast
from pathlib import Path

ROOT = Path("/home/daenu/Code/pgnn/refactor/benchmarks")

DOCS = {
"checks/paper_checks.py": '''Numerical checks of the propositions as they are stated in the paper (paper/sections/method.tex and chem.tex).
Wherever possible the check runs against the implementation that produced the results, with random untrained weights,
because a guarantee for all weights must also hold at initialisation.

    CUDA_VISIBLE_DEVICES="" uv run python -m benchmarks.checks.paper_checks
''',
"checks/theory_checks.py": '''Numerical checks of the Petri net mathematics that the architecture relies on.

    uv run python -m benchmarks.checks.theory_checks

A  Thermodynamic flux law with an arbitrary positive neural conductance. P-invariants are conserved, the free energy
   is a Lyapunov function, equilibrium potentials lie in ker C^T.
B  Softmax is the equilibrium of the simplest conservative Petri net, a state machine.
C  Sinkhorn, that is entropic optimal transport, is the equilibrium of the partner-swap Petri net.
D  The discrete token game layer, allocation by softmax, then synchronisation by min, then the state equation,
   keeps markings non-negative and conserves P-invariants for arbitrary scores and gates.
E  A rate-independent net with min semantics computes ReLU.
F  A weakly reversible mass-action net of deficiency zero that is NOT detailed balanced has a unique equilibrium per
   compatibility class for arbitrary rate constants, and the pseudo-Helmholtz function decreases.
''',
"chemistry/build_data.py": '''Build the reaction data sets.

    uv run python -m benchmarks.chemistry.build_data              # writes data/schneider50k.pkl
    uv run python -m benchmarks.chemistry.build_data uspto-mit    # writes data/uspto_mit.pkl
''',
"chemistry/experiment.py": '''Atom mapping, forward prediction and reaction classification on the valence Petri net.

    uv run python -m benchmarks.chemistry.experiment --task classify --model npf --seed 0
    uv run python -m benchmarks.chemistry.experiment --task map --model pgnn
    uv run python -m benchmarks.chemistry.experiment --task forward --model npf --dataset uspto_mit

Models. npf uses the Petri semantics, pgnn is the same message passing with a generic readout, drfp is DRFP with an
MLP for classification, and the npf-no... variants remove one Petri component each.

Splits. classify uses the published split of Schneider 50k with 200 training and 800 test reactions per class.
On Schneider 50k, map and forward use a fixed random 80/10/10 split of the reactions that come with a clean atom
mapping, which is an internal protocol for ablations. With the dataset uspto_mit the official split of Jin et al. is
used, and forward prediction is scored on the whole official test set. Training records whose recorded mapping moves
more than MAX_TOKENS tokens are dropped as label noise. Validation and test sets are never filtered.
''',
"chemistry/figures.py": '''Figures for the paper that only the Petri formulation can draw.

tokengame     A reaction as a token game on the valence net. The marking after every firing with free valence tokens
              as dots on the atoms, the rates of the enabled transitions, the transitions that enabling forbids, and
              the lattice of markings in which count-equivalent firing sequences merge and their probabilities add.
attribution   Exact per-atom contributions to the state-equation readout of the classifier, which are zero beyond
              the receptive field of the reaction centre.
loadbearing   The ablations that show where the net is load-bearing.

    CUDA_VISIBLE_DEVICES="" uv run python -m benchmarks.chemistry.figures tokengame 14440
''',
"chemistry/golden.py": '''Atom mapping on the Golden dataset of Lin et al. (Mol. Inform. 2022) with 1,851 manually curated reactions, the
accepted benchmark for atom-to-atom mapping. RXNMapper, GraphormerMapper, LocalMapper and SAMMNet report on it.

The criterion is the one of Lin et al. A mapping is correct if its condensed graph of reaction (CGR) is identical to
the CGR of the curated mapping. The CGR is built as a labelled graph over all precursor atoms, with the element on
the atoms and the bond order before and after on the bonds, and compared by isomorphism, so that symmetry-equivalent
mappings count as correct. RXNMapper is scored with exactly the same code.

    uv run python -m benchmarks.chemistry.golden prepare <golden_dataset.rdf>       # writes data/golden.pkl and data/golden_unmapped.txt
    uv run python -m benchmarks.chemistry.golden evaluate [rxnmapper_output.json]   # writes results/chem/golden.json
''',
"chemistry/insights.py": '''What the Petri formulation gives beyond accuracy, on Schneider 50k with the trained models of
benchmarks.chemistry.experiment.

    uv run python -m benchmarks.chemistry.insights    # writes results/chem/insights.json and prints a summary

1  validity      Products are valence-valid by construction because markings are non-negative. The one-shot
                 counterpart has no such guarantee.
2  by-products   The state equation conserves tokens, so the fragments that leave come with the prediction.
3  calibration   The probability of a product is the probability of its trace with all firing orders merged.
                 Is it calibrated?
4  firing order  Enabling constrains the order in which transitions fire. What has to break before what forms?
5  attribution   The classifier reads sigma through the state equation, so untouched places contribute exactly
                 zero. Attribution against the distance from the reaction centre, without any saliency method.
6  data audit    Recorded atom mappings that move more tokens than ours, or that violate the state equation.
''',
"chemistry/invariance.py": '''Is the Petri readout of the classifier load-bearing? A counterfactual test of a provable property.

Proposition (THEORY.md, P5). If the readout of a place depends on its R-ball only, the state-equation readout
    r = sum_B phi(place) - sum_A phi(place)
does not depend on anything further than R bonds from the places that a firing changes, because the contributions of
an atom and of its partner cancel exactly when their R-balls are untouched. So if the same substituent is attached on
both sides of the reaction at a site more than 2R bonds from everything that changes, r and with it every logit
stays exactly the same (6e-14 in double precision). A readout that pools the two sides separately has no such
invariance.

    uv run python -m benchmarks.chemistry.invariance    # writes results/chem/insights_invariance.json

The test adds a methyl group to a remote C-H site of test reactions, consistently in precursor and product, and
compares the logits and the predicted class before and after, for every trained classifier in results/chem/classify.
''',
"chemistry/review.py": '''Blinded review package for the atom mapping disagreements, to be judged by a chemist.

    uv run python -m benchmarks.chemistry.review build    # writes results/chem/mapping_review/{index.html, answers.csv, key.json}
    uv run python -m benchmarks.chemistry.review score    # after answers.csv has been filled in

The sample is stratified over test reactions on which the NPF mapper, RXNMapper and the recorded mapping do not all
imply the same firing vector. Every distinct candidate mapping is drawn with atom map numbers on the mapped atoms and
with the bonds that change under that mapping highlighted, and is shown under a random letter. The reviewer marks each
candidate as correct (1), wrong (0) or unclear (?). key.json holds which sources produced which letter.
''',
"chemistry/rxnmapper.py": '''Score the mappings of RXNMapper on the Schneider test reactions with the metrics used for our mapper.

RXNMapper (Schwaller et al., Sci. Adv. 2021) is pretrained without labels. Its output is produced in a separate
environment and stored in results/chem/rxnmapper_test.json.

    uv run python -m benchmarks.chemistry.rxnmapper
''',
"synthetic/coloured.py": '''Coloured tokens. Tokens carry features and transitions compute on them (THEORY.md, P9).

The ground truth, hidden from the models, is a continuous coloured Petri net. Place p holds mass m_p of colour c_p in
R^2. Transition t fires at rate k_t prod_p (m_p / (K_p + m_p))^Pre(p,t) * guard_t(c_in) with the guard
(1 + <c_in, u_t>) / 2, where c_in is the Pre-weighted mean colour of its input places, and it emits tokens of colour
o_t = Rot(theta_t) c_in. Places mix what arrives with what stays,
    d(m_p c_p)/dt = sum_in Pos v_t o_t - sum_out Pre v_t c_p.

npf-coloured  Token game for the masses with a learned colour guard in the rate law, a neural arc expression o_theta
              for the colour of emitted tokens, and the mixing rule that token conservation dictates,
                  c_p <- ((m_p - out_p) c_p + sum_in Pos v_t o_t) / m_p'
              The aggregation is the flow-weighted mean and the update gate is the share of tokens that stayed.
              Neither is learned.
npf-free      Ablation with the same masses, but the colour update is a free function of the same messages.
pgnn+         Generic message passing between places and transitions on the features [mass, colour].

Provable for npf-coloured and any weights. Masses obey P1 and P2, a place without inflow keeps its colour exactly,
and new colours are convex combinations of the old colour and the emitted ones.

    uv run python -m benchmarks.synthetic.coloured    # writes results/coloured.json
''',
"synthetic/equilibrium.py": '''Thermodynamic equilibrium layer (THEORY.md, P8) and its evaluation.

A reversible net with place energies E_p = f(e_p), f hidden, relaxes from a marking m_0. The task is to predict where
it ends up. The answer is the unique minimiser of the free energy on the compatibility class of m_0,

    m* = argmin sum_p m_p (log(m_p / m_ref,p) - 1)   subject to   X^T m = X^T m_0   with   m_ref = exp(-E),

where X is a basis of the P-invariants, that is of ker C^T. ThermoNPF learns only the energies E_theta(e_p). The layer
solves the strictly convex dual with Newton's method and is differentiated by one Newton step at the solution, which
is the implicit function theorem. Conservation is exact by construction, positivity follows from the exponential form
and uniqueness from convexity. A message-passing readout has none of the three, and the conserved totals are global,
so a fixed number of rounds cannot compute m* on nets larger than the receptive field.

    uv run python -m benchmarks.synthetic.equilibrium    # writes results/equilibrium.json
''',
"synthetic/learned_incidence.py": '''Learned incidence. The arc weights, here conversion ratios, are hidden and have to be learned.

The motivating example of the PGNN paper is flow with conversion between semantic layers (1 GBP gives 1.32 USD). Here
every place has a type, its currency. Every transition moves one unit out of its input place and R[type_in, type_out]
units into its output place, and R is not given to any model. NPF learns R as a table of restriction maps indexed by
the types of the two ends of an arc. This is the one-dimensional case of a cellular sheaf on the net with stalks R and
restriction maps Pre and Pos. The dense incidence matrix C_theta is rebuilt from the table in every forward pass, and
the token game, the projection and the Laplacian C W C^T all run on the learned net. Whatever R_theta is, the model is
a Petri net, its conservation laws are ker C_theta^T, and they can be read off. Here they are the price vector up to
scale.

    uv run python -m benchmarks.synthetic.learned_incidence    # writes results/sheaf.json

npf-learned   NPF with the learned conversion table
npf-unit      NPF that assumes all ratios are 1
npf-oracle    NPF with the true ratios
pgnn, pgnn+   no incidence matrix at all
''',
"synthetic/locality.py": '''Locality lower bound (THEORY.md, P3). Message passing cannot replace the state-equation projection.

The chain net p_0 -> p_1 -> ... -> p_L has no T-invariants, so the firing counts between two states are fixed by the
state equation, sigma_k = -sum_{j<k} dM_j, but only through sums along the whole chain. Models are trained on the
usual random nets with 6 to 12 places and tested on chains of growing length.

    uv run python -m benchmarks.synthetic.locality    # writes results/locality.json
''',
"synthetic/paper_protocol.py": '''The protocol of the PGNN paper, applied to the transition prediction task. One fixed net, 100 samples, a 70/30
split, 300 epochs of Adam, parameters per net (transductive), a clean and a noisy scenario. This is the setting in
which the paper trains its linear special case (Eq. 13), so that model is included here.

    uv run python -m benchmarks.synthetic.paper_protocol
''',
}

for rel, doc in DOCS.items():
    path = ROOT / rel
    source = path.read_text()
    tree = ast.parse(source)
    node = tree.body[0]
    assert isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str), rel
    lines = source.split("\n")
    new = '"""' + doc.rstrip("\n") + '\n"""'
    lines[node.lineno - 1:node.end_lineno] = new.split("\n")
    path.write_text("\n".join(lines))
    ast.parse(path.read_text())
    assert all(ord(ch) < 128 for ch in doc), rel
print("replaced", len(DOCS), "module docstrings")
