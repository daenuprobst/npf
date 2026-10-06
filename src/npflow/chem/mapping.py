"""Atom maps of the mapper, written as reaction SMILES with map numbers."""

import numpy as np
from rdkit import Chem

from .featurisation import reaction


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


def map_reaction(smiles, seconds=60.0):
    """The reaction with the atom map numbers of the mapper, precursors>>product. None when RDKit cannot read it or the
    product holds atoms that no precursor supplies. seconds bounds the wall time of the search.
    """
    from .mapper import best_mapping

    r = reaction(smiles, max_atoms=None)
    if r is None or len(r["b"]["x"]) > len(r["a"]["x"]):
        return None

    mapping, _, _ = best_mapping(r, seconds=seconds)

    return (
        mapped_smiles(r["smiles"], np.asarray(mapping, np.int64))
        if mapping is not None
        else None
    )
