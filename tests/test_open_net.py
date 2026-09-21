"""The valence net as an open net. Further equivalents, atoms from outside and the balanced equation."""
import numpy as np
import pytest

from npf import chem
from npf.chem import exact, open_net

OPTIONS = dict(secondary=(1, 1, 1), labile_h=False, ch_places=True, seconds=30)


def reaction(smiles):
    return chem.featurise({"original_rxn": smiles, "rxn": smiles, "label": 0, "split": "test", "id": 0})


def test_a_balanced_reaction_is_solved_as_before():
    r = reaction("CC(=O)O.OC>>CC(=O)OC")
    mapping, cost, proved, solved = open_net.solve_open(r, **OPTIONS)
    plain = exact.solve(r, **OPTIONS)
    assert solved is r and proved and cost == plain[1] and (mapping >= 0).all()
    out = open_net.balance(solved, mapping)
    assert out["by_products"] == ["O"] and out["entered"] == [] and out["spectators"] == []
    assert out["balanced"] == "CC(=O)O.CO>>CC(=O)OC.O"


def test_a_reagent_written_once_and_used_twice_enters_again():
    r = reaction("CC(=O)Cl.NCCN>>CC(=O)NCCNC(C)=O")
    assert open_net.deficit(r).sum() == 3
    mapping, cost, proved, solved = open_net.solve_open(r, **OPTIONS)
    assert proved and (mapping >= 0).all() and exact.entered(solved, mapping) == 1
    out = open_net.balance(solved, mapping)
    assert out["equivalents"] == {"CC(=O)Cl": 2, "NCCN": 1} and out["by_products"] == ["Cl", "Cl"]

    # two C-Cl bonds break, two C-N bonds form, and one molecule enters
    assert cost == 5


def test_an_atom_that_nothing_supplies_enters_alone():
    r = reaction("CSC>>CS(C)=O")
    mapping, cost, proved, solved = open_net.solve_open(r, **OPTIONS)
    oxygen = int(np.nonzero(r["b"]["element"] == 8)[0][0])
    assert proved and mapping[oxygen] == -1 and (np.delete(mapping, oxygen) >= 0).all()
    out = open_net.balance(solved, mapping)
    assert out["entered"] == [(oxygen, "O")] and out["by_products"] == [] and out["balanced"] == "CSC.[O]>>CS(C)=O"

    # the S=O bond forms and one atom enters
    assert cost == 2


def test_a_second_equivalent_is_cheaper_than_atoms_from_outside():
    r = reaction("OC(=O)c1ccccc1>>O=C(OC(=O)c1ccccc1)c1ccccc1")
    mapping, cost, proved, solved = open_net.solve_open(r, **OPTIONS)
    assert proved and (mapping >= 0).all()
    out = open_net.balance(solved, mapping)
    assert out["equivalents"] == {"O=C(O)c1ccccc1": 2} and out["by_products"] == ["O"]
    assert cost == 3


@pytest.mark.parametrize("smiles", ["CC(=O)Cl.NCCN>>CC(=O)NCCNC(C)=O", "CSC>>CS(C)=O", "OC(=O)C>>CC(=O)OC(C)=O"])
def test_cost_levels_agree_with_the_objective_of_the_open_model(smiles):
    from ortools.sat.python import cp_model
    options = dict(secondary=(1, 1, 1), labile_h=False, ch_places=True)
    opened = open_net.with_equivalents(reaction(smiles))
    model, seat, n_b, scale, _, _, _ = exact._model(opened, sources=set(np.nonzero(open_net.deficit(reaction(smiles)))[0].tolist()), **options)
    solver = cp_model.CpSolver()
    assert solver.Solve(model) == cp_model.OPTIMAL
    places, tokens = exact.cost_levels(opened, exact._read(solver, seat, n_b), **options)
    assert int(round(solver.ObjectiveValue())) == scale * places + tokens


def test_copies_repeat_the_molecule_and_point_back_to_it():
    r = reaction("CC(=O)Cl.NCCN>>CC(=O)NCCNC(C)=O")
    opened = open_net.with_equivalents(r)["a"]
    n = len(r["a"]["element"])
    assert (opened["copy"][:n] == 0).all() and (opened["origin"][:n] == np.arange(n)).all()
    extra = np.nonzero(opened["copy"] > 0)[0]
    assert (opened["element"][extra] == r["a"]["element"][opened["origin"][extra]]).all()
    full, first = chem.dense_bonds(opened), chem.dense_bonds(r["a"])
    one = extra[opened["copy"][extra] == 1]
    one = one[opened["fragment"][one] == opened["fragment"][one][0]]
    assert (full[np.ix_(one, one)] == first[np.ix_(opened["origin"][one], opened["origin"][one])]).all()
    assert not full[np.ix_(one, np.arange(n))].any()


@pytest.mark.parametrize("smiles", ["CC(=O)Cl.NCN>>CC(=O)NCNC(C)=O", "CSC>>CS(C)=O", "CO.CBr>>COC.CO"])
def test_open_net_finds_the_enumerated_minimum(smiles):
    """Every seating of the product atoms on the written atoms, on the copies, or outside, against the solver."""
    import itertools
    options = dict(secondary=(1, 1, 1), labile_h=False, ch_places=True)
    r = reaction(smiles)
    short = set(np.nonzero(open_net.deficit(r))[0].tolist())
    opened = open_net.with_equivalents(r)
    a, b = opened["a"], opened["b"]
    seats = [list(np.nonzero(a["element"] == e)[0]) + ([-1] if int(e) in short else []) for e in b["element"]]
    best = None
    for m in itertools.product(*seats):
        on = [j for j in m if j >= 0]
        if len(set(on)) < len(on):
            continue

        value = exact.cost_levels(opened, np.array(m), **options)
        best = value if best is None or value < best else best

    mapping, cost, proved, solved = open_net.solve_open(r, seconds=60, **options)
    assert proved and exact.cost_levels(solved, mapping, **options) == best
