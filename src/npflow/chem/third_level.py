"""The third level of the cost, which only orders mappings that tie on both levels of cost.CHOSEN.

Among the classes of optimal mappings the mapper keeps those that minimise, in this order,

    redox       the change of the formal oxidation state summed over product carbons
    aromatic    bond places switched between an aromatic carbon and a heteroatom
    sink        minus the bond places switched between an electron sink and an atom other than carbon

The three terms are counts with no fitted parameter, and none names a reaction type, a reagent or a data set. They
were chosen on the 200 Golden reactions that also fixed the second level, benchmarks.chemistry.exact_map.golden_dev.
"""

import numpy as np

from .featurisation import BOND_ORDER, dense_bonds

# Pauling electronegativity, unlisted elements, mostly metals, sit below carbon
PAULING = {
    1: 2.20,
    3: 0.98,
    5: 2.04,
    6: 2.55,
    7: 3.04,
    8: 3.44,
    9: 3.98,
    11: 0.93,
    12: 1.31,
    13: 1.61,
    14: 1.90,
    15: 2.19,
    16: 2.58,
    17: 3.16,
    19: 0.82,
    29: 1.90,
    30: 1.65,
    34: 2.55,
    35: 2.96,
    46: 2.20,
    50: 1.96,
    53: 2.66,
    55: 0.79,
}


def electronegativity(elements):
    return np.array([PAULING.get(int(e), 1.5) for e in elements])


def oxidation(elements, orders, h, q):
    """Formal oxidation state, plus the order of every bond to a more electronegative atom, minus it to a less
    electronegative one, minus the hydrogens, plus the charge. Used for carbon, which outranks hydrogen.
    """
    x = electronegativity(elements)
    sign = np.sign(x[None, :] - x[:, None])

    return (sign * orders).sum(1) - h + q


def terms(reaction, mapping):
    """The three terms of one mapping, lower is better."""
    a, b = reaction["a"], reaction["b"]
    mapping = np.asarray(mapping, np.int64)
    element = a["element"]
    before, orders_b = BOND_ORDER[dense_bonds(a)], BOND_ORDER[dense_bonds(b)]
    after = np.zeros_like(before)
    after[np.ix_(mapping, mapping)] = orders_b
    seated = np.zeros(len(before), bool)
    seated[mapping] = True
    touched = np.triu(seated[:, None] | seated[None, :], 1)
    switched = list(zip(*np.nonzero(touched & ((after > 0) != (before > 0)))))
    aromatic = a["x"][:, -2].astype(bool)

    # a firing that is not a redox step leaves oxidation states alone, so a mapping that moves electrons between
    # carbons is the less plausible reading of the same bond changes
    carbon = b["element"] == 6
    ox_a = oxidation(element, before, a["h"], a["q"])
    ox_b = oxidation(b["element"], orders_b, b["h"], b["q"])
    redox = float((np.abs(ox_b - ox_a[mapping]) * carbon).sum())

    # substitution at an aromatic carbon needs activation, so of two equally cheap exchanges the one away from the ring
    # is preferred
    arom = sum(
        bool(
            (aromatic[i] and element[i] == 6 and element[j] != 6)
            or (aromatic[j] and element[j] == 6 and element[i] != 6)
        )
        for i, j in switched
    )

    # a sink, a non-aromatic atom with a multiple bond to a more electronegative partner, can take an incoming pair on
    # that partner and give it back, addition then elimination, so every intermediate marking keeps the valence rule
    x = electronegativity(element)
    sink = ((before >= 2) & (x[None, :] > x[:, None])).any(1) & ~aromatic
    at_sink = sum(
        bool((sink[i] and element[j] != 6) or (sink[j] and element[i] != 6))
        for i, j in switched
    )

    return redox, arom, -at_sink
