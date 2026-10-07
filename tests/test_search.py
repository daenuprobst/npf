"""The branch and bound against enumeration of every seating of small reactions.

    uv run python -m pytest -q tests/test_search.py
"""

import itertools

import numpy as np
import pytest

from npflow import chem
from npflow.chem import exact, open_net
from npflow.chem.cgr import same_cgr
from npflow.chem.search import Problem, Search
from npflow.chem.targets import firing_vector


def solve(r, options=None, use_start=False, enumerate_ties=False, tie_limit=64, sources=()):
    """The search with or without the heuristic start, its minimum, its levels and what it did."""
    pr = Problem(r, sources=sources, **(options or exact.CHOSEN))
    start = None

    if use_start and not len(sources):
        hint = chem.feasible_start(r)
        start = None if hint is None else np.asarray(hint, np.int64)

    search = Search(pr, start, node_limit=10**6, enumerate_ties=enumerate_ties, tie_limit=tie_limit).run()

    return search.best, search.best_levels, {"proved": search.complete, "ties": search.ties}


SMALL = ["CC(=O)O.OC>>CC(=O)OC", "CC(=O)Cl.NC>>CC(=O)NC", "C=C.BrBr>>BrCCBr", "CC(=O)OC.O>>CC(=O)O",
         "OCC=C.CC(=O)O>>CC(=O)OCC=C", "CC(C)=O.NO>>CC(C)=NO", "C=CC=C.C=CC#N>>N#CC1CC=CCC1",
         "CC(=O)C.[BH4-]>>CC(O)C", "C=CC.BrBr.O>>OC(C)CBr", "FC(F)(F)C(=O)Cl.OCC>>FC(F)(F)C(=O)OCC",
         "CC(=O)OC(C)=O.OCC>>CC(=O)OCC", "O=C(O)CCC(=O)O.OC>>COC(=O)CCC(=O)O", "c1ccccc1.ClC(=O)C>>CC(=O)c1ccccc1"]

VARIANTS = [dict(secondary=(1, 1, 1), labile_h=False, ch_places=True), dict(secondary=(1, 1, 1)),
            dict(secondary=(1, 1, 1), labile_h=False), dict(secondary=(3, 2, 5), ch_places=True)]


def reaction(smiles):
    return chem.featurise({"original_rxn": smiles, "rxn": smiles, "label": 0, "split": "test", "id": 0})


def every_seating(r):
    """Every injective seating by element, the permutations of each element combined."""
    a, b = r["a"]["element"], r["b"]["element"]
    elements = sorted(set(b.tolist()))
    rows = [np.nonzero(b == e)[0] for e in elements]
    options = [list(itertools.permutations(np.nonzero(a == e)[0].tolist(), len(i))) for e, i in zip(elements, rows)]
    assert np.prod([float(len(o)) for o in options]) < 2e6
    out = []
    for choice in itertools.product(*options):
        m = np.empty(len(b), np.int64)
        for i, seats in zip(rows, choice):
            m[i] = seats

        out.append(m)

    return out


@pytest.mark.parametrize("smiles", SMALL)
@pytest.mark.parametrize("options", VARIANTS, ids=[str(i) for i in range(len(VARIANTS))])
def test_search_finds_and_proves_the_lexicographic_minimum(smiles, options):
    r = reaction(smiles)
    seatings = every_seating(r)
    best = min(exact.cost_levels(r, m, **options) for m in seatings)
    for use_start in (False, True):
        mapping, levels, info = solve(r, options=options, use_start=use_start)
        assert info["proved"] and tuple(levels) == best
        assert exact.cost_levels(r, mapping, **options) == best


@pytest.mark.parametrize("smiles", SMALL)
def test_levels_agree_with_exact(smiles):
    r = reaction(smiles)
    for options in VARIANTS:
        pr = Problem(r, **options)
        for m in every_seating(r)[:200]:
            assert pr.levels(m) == exact.cost_levels(r, m, **options)


@pytest.mark.parametrize("smiles", SMALL)
def test_listing_holds_every_class_of_optimal_seatings(smiles):
    """Every optimal seating has the condensed graph of one listed seating, and every listed seating is optimal."""
    r = reaction(smiles)
    seatings = every_seating(r)
    costs = [exact.cost_levels(r, m, **exact.CHOSEN) for m in seatings]
    best = min(costs)
    optimal = [m for m, c in zip(seatings, costs) if c == best]
    _, _, info = solve(r, enumerate_ties=True, tie_limit=10000)
    listed = info["ties"]
    assert info["proved"]
    assert all(exact.cost_levels(r, m, **exact.CHOSEN) == best for m in listed)
    assert all(any(same_cgr(r, m, x) for x in listed) for m in optimal)


@pytest.mark.parametrize("smiles", ["CC(=O)Cl.OC>>CC(=O)OC.CC(=O)OC", "CC(=O)O>>CC(=O)OC"])
def test_open_net_matches_the_integer_program(smiles):
    r = reaction(smiles)
    short = {int(e) for e in np.nonzero(open_net.deficit(r))[0]}
    ref, _, proved, opened = open_net.solve_open(r, seconds=30, **exact.CHOSEN)
    mapping, levels, info = solve(opened, sources=short)
    assert proved and info["proved"]
    assert tuple(levels) == exact.cost_levels(opened, ref, **exact.CHOSEN)
    assert exact.cost_levels(opened, mapping, **exact.CHOSEN) == tuple(levels)


def test_a_bad_incumbent_is_improved():

    r = reaction(SMALL[4])
    pr = Problem(r, **exact.CHOSEN)
    seatings = every_seating(r)
    worst = max(seatings, key=pr.levels)
    best = min(pr.levels(m) for m in seatings)
    search = Search(pr, worst).run()
    assert search.complete and search.best_levels == best


@pytest.mark.parametrize("smiles", SMALL[:10])
def test_api_matches_the_integer_program(smiles):
    from npflow.chem import mapper as api

    r = reaction(smiles)
    ref, cost, proved = exact.solve(r, seconds=30, **exact.CHOSEN)
    mapping, mine, ok = api.solve(r, **exact.CHOSEN)
    assert proved and ok and cost == mine
    assert exact.cost_levels(r, mapping, **exact.CHOSEN) == exact.cost_levels(r, ref, **exact.CHOSEN)
    maps, done = exact.cheapest_mappings(r, limit=1000, seconds=60, deterministic=30, **exact.CHOSEN)
    listed, finished = api.cheapest_mappings(r, limit=1000, **exact.CHOSEN)
    vectors = lambda ms: {firing_vector(r, np.asarray(m, np.int64)).tobytes() for m in ms}
    assert finished and vectors(listed) == vectors(maps)


def test_the_third_level_picks_acyl_cleavage_in_fischer_esterification():
    """Acid and alcohol tie on both levels between acyl C-O and alkyl C-O cleavage, the question isotope labelling
    settles for acyl cleavage. The sink term of the third level takes the carbonyl carbon, which can take the incoming
    pair on its oxygen."""
    from npflow.chem import mapper
    from npflow.chem.third_level import terms

    r = reaction("CC(=O)O.OCC>>CC(=O)OCC")
    maps, proved = mapper.cheapest_mappings(r, limit=1000, **exact.CHOSEN)
    classes = mapper.tie_classes(r, [np.asarray(m, np.int64) for m in maps])
    mapping, levels, finished = mapper.best_mapping(r)
    assert proved and finished and len(classes) == 2
    assert terms(r, mapping) == min(terms(r, m) for _, m in classes)
    assert terms(r, mapping)[2] < 0


def test_best_mapping_does_not_depend_on_the_atom_order():
    """The class with the smallest graph hash is taken among those the third level cannot tell apart."""
    from npflow.chem import mapper
    from npflow.chem.cgr import same_cgr as same

    r = reaction("O=C(O)CCC(=O)O.OC>>COC(=O)CCC(=O)O")
    first, _, _ = mapper.best_mapping(r)
    again, _, _ = mapper.best_mapping(r)
    assert same(r, first, again)


@pytest.mark.parametrize("smiles", ["CC(=O)Cl.OC>>CC(=O)OC.CC(=O)OC", "CC(=O)O>>CC(=O)OC"])
def test_best_mapping_seats_only_written_atoms(smiles):
    """On the open net, a product atom that no written precursor supplies has the seat -1, never a copy."""
    from npflow.chem import mapper

    r = reaction(smiles)
    mapping, _, proved = mapper.best_mapping(r)
    n = len(r["a"]["element"])
    seated = mapping[mapping >= 0]
    assert proved and (mapping < n).all() and (mapping < 0).any()
    assert len(set(seated.tolist())) == len(seated)
