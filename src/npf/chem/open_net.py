"""The valence net as an open net. Atoms leave, and atoms and molecules enter from the environment.

Recorded reactions are not balanced. By-products are left out, so precursor atoms leave, which the injection of the
product atoms into the precursor atoms allows already. A reagent can also be used twice and written once, or not be
written at all, and then no injection exists. The open net adds two source transitions. One lets a further copy of a
precursor molecule enter, the other lets a single atom of an element in deficit enter. By convention a firing of a
source is charged one unit on each level, the price of one bond place. Sources are offered only when the product holds
more atoms of an element than the precursors do, so a reaction that can be mapped without them is mapped as before.

The mapping then yields a balanced equation. The atoms of a used molecule that take no seat leave as by-products,
with every bond to a seated atom closed by hydrogen, which balances the heavy atoms and leaves hydrogen to the medium.
"""
from collections import Counter

import numpy as np
from rdkit import Chem
from scipy.sparse.csgraph import connected_components

from . import exact
from .decode import molecule
from .featurisation import BOND_ORDER, dense_bonds

MAX_COPIES = 3


def deficit(reaction):
    """Per element, the number of product atoms that the written precursors cannot supply."""
    count = lambda g: np.bincount(g["element"].astype(np.int64), minlength=120)

    return np.maximum(count(reaction["b"]) - count(reaction["a"]), 0)


def with_equivalents(reaction, cap=MAX_COPIES):
    """The reaction with further copies of every precursor molecule that holds an element in deficit.

    copy numbers the copies of a molecule from 1, the written molecule is 0, and origin is the written atom of every
    atom. The solver decides which copies enter, each one at the cost of one place.
    """
    a, short = reaction["a"], deficit(reaction)
    n = len(a["element"])
    per_atom = [key for key, value in a.items() if key != "bonds" and isinstance(value, np.ndarray) and len(value) == n]
    parts, bonds = {key: [a[key]] for key in per_atom}, [a["bonds"]]
    copy, origin, offset = [np.zeros(n, np.int64)], [np.arange(n)], n

    for fragment in np.unique(a["fragment"]):
        atoms = np.nonzero(a["fragment"] == fragment)[0]
        counts = np.bincount(a["element"][atoms].astype(np.int64), minlength=120)
        helps = (counts > 0) & (short > 0)
        if not helps.any():
            continue

        index = np.full(n, -1)
        index[atoms] = np.arange(len(atoms))
        inner = a["bonds"][np.isin(a["bonds"][:, 0], atoms)]

        # as many copies as would cover the deficit from this molecule alone
        for number in range(1, min(cap, int(np.ceil(short[helps] / counts[helps]).max())) + 1):
            shifted = inner.copy()
            shifted[:, 0], shifted[:, 1] = index[inner[:, 0]] + offset, index[inner[:, 1]] + offset
            bonds.append(shifted)
            copy.append(np.full(len(atoms), number))
            origin.append(atoms)
            offset += len(atoms)

            for key in per_atom:
                parts[key].append(a[key][atoms])

    opened = {key: np.concatenate(parts[key]) for key in per_atom} | {"bonds": np.concatenate(bonds), "copy": np.concatenate(copy),
                                                                      "origin": np.concatenate(origin)}

    return reaction | {"a": opened}


def solve_open(reaction, **options):
    """Cheapest mapping of a reaction that may lack reactants. Returns the mapping, its cost, whether it is proved,
    and the reaction as it was solved, whose precursors hold the copies that were offered. A seat of -1 marks a
    product atom that no written molecule supplies."""
    if not deficit(reaction).any():
        return (*exact.solve(reaction, **options), reaction)

    opened = with_equivalents(reaction)
    options.pop("hint", None)

    # only an atom of an element in deficit may enter alone
    short = {int(e) for e in np.nonzero(deficit(reaction))[0]}

    return (*exact.solve(opened, sources=short, **options), opened)


def _smiles(a, types, atoms, h):
    mol = molecule(a["element"][atoms], types[np.ix_(atoms, atoms)], h, a["q"][atoms])

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        # a piece of an aromatic ring is no longer aromatic, it is written as it stands
        pass

    return Chem.MolToSmiles(mol)


def balance(reaction, mapping):
    """The balanced equation that a mapping implies.

    equivalents counts every precursor molecule that takes part, spectators are the written molecules that do not,
    by_products are the pieces that leave, and entered lists the product atoms that came from outside.
    """
    a = reaction["a"]
    types = dense_bonds(a)
    n = len(a["element"])
    copies = a.get("copy", np.zeros(n, np.int64))
    seated = np.zeros(n, bool)
    seated[mapping[mapping >= 0]] = True
    equivalents, spectators, by_products = Counter(), [], []

    for fragment, number in sorted({(int(f), int(c)) for f, c in zip(a["fragment"], copies)}):
        atoms = np.nonzero((a["fragment"] == fragment) & (copies == number))[0]
        name = _smiles(a, types, atoms, a["h"][atoms])

        if not seated[atoms].any():
            # a copy that did not enter is not part of the reaction at all
            if number == 0:
                spectators.append(name)

            continue

        equivalents[name] += 1
        leaving, staying = atoms[~seated[atoms]], atoms[seated[atoms]]
        if not len(leaving):
            continue

        _, piece = connected_components(types[np.ix_(leaving, leaving)] > 0, directed=False)
        for k in range(piece.max() + 1):
            part = leaving[piece == k]

            # every bond to a seated atom is closed with hydrogen
            closed = np.floor(BOND_ORDER[types[np.ix_(part, staying)]].sum(1) + 0.5).astype(np.int64)
            by_products.append(_smiles(a, types, part, a["h"][part] + closed))

    product = reaction["smiles"].split(">>")[1]
    entered = [(int(i), Chem.GetPeriodicTable().GetElementSymbol(int(reaction["b"]["element"][i]))) for i in np.nonzero(mapping < 0)[0]]

    # the atoms that entered are written on the left as bare atoms, their true source is not known
    left = ".".join([name for name, count in equivalents.items() for _ in range(count)] + [f"[{symbol}]" for _, symbol in entered])

    return {"equivalents": dict(equivalents), "spectators": spectators, "by_products": sorted(by_products), "entered": entered,
            "balanced": left + ">>" + ".".join([product] + sorted(by_products))}
