"""The integer program for the minimum firing vector against enumeration of every seating of small reactions."""
import itertools

import numpy as np
import pytest

from npf import chem
from npf.chem import exact
from npf.chem.featurisation import BOND_ORDER, dense_bonds

SMALL = ["CC(=O)O.OC>>CC(=O)OC", "CC(=O)Cl.NC>>CC(=O)NC", "C=C.BrBr>>BrCCBr", "CC(=O)OC.O>>CC(=O)O", "OCC=C.CC(=O)O>>CC(=O)OCC=C",
         "CC(C)=O.NO>>CC(C)=NO", "C=CC=C.C=CC#N>>N#CC1CC=CCC1"]


def reaction(smiles):
    return chem.featurise({"original_rxn": smiles, "rxn": smiles, "label": 0, "split": "test", "id": 0})


def key(r, mapping, all_orders=False):
    """The two levels of the cost, computed without the solver."""
    a, b = r["a"], r["b"]
    before = BOND_ORDER[dense_bonds(a)]
    after = np.zeros_like(before)
    after[np.ix_(mapping, mapping)] = BOND_ORDER[dense_bonds(b)]
    seated = np.zeros(len(before), bool)
    seated[mapping] = True
    mask = np.triu(seated[:, None] | seated[None, :], 1)
    places = int((((after > 0) != (before > 0)) & mask).sum())
    counted = mask if all_orders else mask & (after > 0) & (before > 0)
    tokens = 2 * np.abs(after - before)[counted].sum() + np.abs(b["h"] - a["h"][mapping]).sum() + np.abs(b["q"] - a["q"][mapping]).sum()

    return places, float(tokens)


def enumerate_minimum(r, all_orders=False):
    seats = [np.nonzero(r["a"]["element"] == e)[0] for e in r["b"]["element"]]
    keys = [key(r, np.array(m), all_orders) for m in itertools.product(*seats) if len(set(m)) == len(m)]

    return min(keys), min(k[0] for k in keys)


@pytest.mark.parametrize("smiles", SMALL)
@pytest.mark.parametrize("all_orders", [False, True])
def test_solver_finds_the_lexicographic_minimum(smiles, all_orders):
    r = reaction(smiles)
    best, fewest_places = enumerate_minimum(r, all_orders)
    mapping, cost, proved = exact.solve(r, seconds=30, secondary=(1, 1, 1), all_orders=all_orders)
    assert proved and cost == fewest_places
    assert key(r, mapping, all_orders) == best
    assert len(set(mapping.tolist())) == len(mapping) and (r["a"]["element"][mapping] == r["b"]["element"]).all()


@pytest.mark.parametrize("smiles", SMALL)
def test_first_level_alone_counts_bond_places(smiles):
    r = reaction(smiles)
    mapping, cost, proved = exact.solve(r, seconds=30)
    assert proved and cost == enumerate_minimum(r)[1] == chem.weighted_cost(r, mapping, (0, 1, 0, 0))


def test_acyl_substitution_is_a_tie_on_the_net():
    """Which oxygen of an ester comes from the alcohol is not decided by the two levels of the cost. Both seatings
    change two bond places and move one hydrogen, so only a learned rate law can order them."""
    r = reaction(SMALL[0])
    mapping, _, _ = exact.solve(r, seconds=30, secondary=(1, 1, 1))
    ester_oxygen = [i for i in np.nonzero(r["b"]["element"] == 8)[0] if (dense_bonds(r["b"])[i] > 0).sum() == 2][0]
    hydroxyls = [j for j in np.nonzero(r["a"]["element"] == 8)[0] if r["a"]["h"][j] == 1]
    keys = []
    for j in hydroxyls:
        other = mapping.copy()
        other[ester_oxygen] = j

        # the carbon of methanol and the methyl of the acid are both seated already, only the oxygen moves
        keys.append(key(r, other))

    assert len(hydroxyls) == 2 and keys[0] == keys[1] == key(r, mapping)


@pytest.mark.parametrize("smiles", SMALL[:4])
def test_star_bound_never_exceeds_the_minimum(smiles):
    r = reaction(smiles)
    assert exact.star_bound(r) <= enumerate_minimum(r)[1]


def test_enumeration_lists_exactly_the_optimal_seatings():
    r = reaction(SMALL[0])
    best, _ = enumerate_minimum(r)
    seats = [np.nonzero(r["a"]["element"] == e)[0] for e in r["b"]["element"]]
    expected = {m for m in itertools.product(*seats) if len(set(m)) == len(m) and key(r, np.array(m)) == best}
    maps, proved = exact.cheapest_mappings(r, secondary=(1, 1, 1), limit=1000, seconds=120, deterministic=30)
    assert proved and {tuple(m.tolist()) for m in maps} == expected


VARIANTS = [dict(secondary=(1, 1, 1)), dict(secondary=(1, 1, 1), all_orders=True), dict(secondary=(1, 1, 1), labile_h=False),
            dict(secondary=(1, 1, 1), labile_h=False, ch_places=True), dict(secondary=(3, 2, 5), ch_places=True, all_orders=True)]


@pytest.mark.parametrize("smiles", SMALL + ["CC(=O)C.[BH4-]>>CC(O)C", "C=CC.BrBr.O>>OC(C)CBr"])
@pytest.mark.parametrize("options", VARIANTS, ids=[str(i) for i in range(len(VARIANTS))])
def test_cost_levels_agree_with_the_objective_of_the_model(smiles, options):
    """The value that the solver minimises is the value that cost_levels reports, and it is the enumerated minimum."""
    from ortools.sat.python import cp_model
    r = reaction(smiles)
    model, seat, n_b, scale, _, _, _ = exact._model(r, **options)
    solver = cp_model.CpSolver()
    assert solver.Solve(model) == cp_model.OPTIMAL
    mapping = exact._read(solver, seat, n_b)
    places, tokens = exact.cost_levels(r, mapping, **options)
    assert int(round(solver.ObjectiveValue())) == scale * places + tokens
    seats = [np.nonzero(r["a"]["element"] == e)[0] for e in r["b"]["element"]]
    best = min(exact.cost_levels(r, np.array(m), **options) for m in itertools.product(*seats) if len(set(m)) == len(m))
    assert (places, tokens) == best


def test_a_bad_hint_is_improved_and_a_good_hint_is_never_lost():
    r = reaction(SMALL[4])
    best, _, _ = exact.solve(r, seconds=30, secondary=(1, 1, 1))
    seats = [np.nonzero(r["a"]["element"] == e)[0] for e in r["b"]["element"]]
    worst = max((np.array(m) for m in itertools.product(*seats) if len(set(m)) == len(m)), key=lambda m: exact.cost_levels(r, m, (1, 1, 1)))
    from_worst, cost, proved = exact.solve(r, seconds=30, secondary=(1, 1, 1), hint=worst)
    assert proved and exact.cost_levels(r, from_worst, (1, 1, 1)) == exact.cost_levels(r, best, (1, 1, 1))

    # with no time at all the solver has nothing better, the hint comes back
    kept, _, _ = exact.solve(r, seconds=1e-9, secondary=(1, 1, 1), hint=best)
    assert exact.cost_levels(r, kept, (1, 1, 1)) == exact.cost_levels(r, best, (1, 1, 1))
