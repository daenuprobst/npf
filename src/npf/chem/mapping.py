"""Atom maps of the exact mapper, written as reaction SMILES with map numbers."""

import numpy as np
from rdkit import Chem

from . import exact
from .featurisation import reaction
from .minimise import feasible_start


def ranked(smiles):
    """The molecule and its atoms in the canonical order of featurisation.graph, so that index i is the same atom."""
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    return mol, np.argsort(list(Chem.CanonicalRankAtoms(mol, breakTies=True)))


def mapped_smiles(smiles, mapping):
    """The reaction with map number i+1 on product atom i and on the precursor atom it sits on. An atom that entered
    from outside, or sits on a copy the open net added, stays unmapped."""
    reactants, reagents, products = smiles.split(">")
    (a, order_a), (b, order_b) = ranked(
        ".".join(s for s in (reactants, reagents) if s)
    ), ranked(products)
    for i, j in enumerate(mapping):
        if 0 <= j < len(order_a):
            a.GetAtomWithIdx(int(order_a[j])).SetAtomMapNum(i + 1)
            b.GetAtomWithIdx(int(order_b[i])).SetAtomMapNum(i + 1)

    return f"{Chem.MolToSmiles(a)}>>{Chem.MolToSmiles(b)}"


def map_reaction(smiles, seconds=20.0):
    """The reaction with the atom map numbers of the minimum firing vector, precursors>>product. None when RDKit
    cannot read it or the product holds atoms that no precursor supplies. seconds is the deterministic solver budget.
    """
    r = reaction(smiles, max_atoms=None)
    if r is None or len(r["b"]["x"]) > len(r["a"]["x"]):
        return None

    # the setting of the benchmark runs, four interleaved workers under a deterministic budget
    maps, _ = exact.cheapest_mappings(
        r,
        limit=1,
        seconds=4 * seconds,
        hint=feasible_start(r),
        deterministic=seconds,
        workers=4,
        **exact.CHOSEN,
    )

    return mapped_smiles(r["smiles"], np.asarray(maps[0], np.int64)) if maps else None
