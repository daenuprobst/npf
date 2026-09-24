"""The cost of a mapping between two markings of the valence net, without a solver.

A mapping seats every product atom on a precursor atom of the same element. The first level counts the bond places a
firing empties or fills, the bonds made and broken, the second the order tokens on kept bonds, hydrogen and charge. Both
the solver-free search of npf.chem.search and the integer program of npf.chem.exact minimise these two levels.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .featurisation import BOND_ORDER, dense_bonds

# the cost chosen on 200 Golden reactions in benchmarks.chemistry.exact_map, hydrogen on heteroatoms is not counted,
# hydrogen on carbon is a bond place of its own and the second level weighs order tokens, hydrogen and charge alike
CHOSEN = dict(secondary=(1, 1, 1), labile_h=False, ch_places=True)


def _hydrogen_weights(reaction, secondary, labile_h, ch_places):
    """Per product atom, the weight of a hydrogen move on the first and on the second level. A hydrogen on a heteroatom
    exchanges with the medium, so with labile_h False it says nothing about the seat, and with ch_places a hydrogen
    on carbon is a bond place of its own."""
    carbon = reaction["b"]["element"] == 6
    first = np.where(carbon & ch_places, 1, 0)
    second = np.where(
        carbon, 0 if ch_places else secondary[1], secondary[1] if labile_h else 0
    )

    return first, second


def entered(reaction, mapping):
    """Number of source firings of a mapping, atoms that enter from outside plus further copies of a molecule."""
    a = reaction["a"]
    copies = a.get("copy", np.zeros(len(a["element"]), np.int64))
    on = mapping[mapping >= 0]
    used = {(int(a["fragment"][j]), int(copies[j])) for j in on if copies[j] > 0}

    return int((mapping < 0).sum()) + len(used)


def cost_levels(
    reaction,
    mapping,
    secondary=(0, 0, 0),
    all_orders=False,
    labile_h=True,
    ch_places=False,
):
    """The two levels of the cost of a mapping, computed without the solver. A product atom with seat -1 entered
    from outside, all of its bonds were formed."""
    a, b = reaction["a"], reaction["b"]
    before, orders_b = BOND_ORDER[dense_bonds(a)], BOND_ORDER[dense_bonds(b)]
    on = mapping >= 0
    after = np.zeros_like(before)
    after[np.ix_(mapping[on], mapping[on])] = orders_b[np.ix_(on, on)]
    seated = np.zeros(len(before), bool)
    seated[mapping[on]] = True
    mask = np.triu(seated[:, None] | seated[None, :], 1)
    outside = np.triu(orders_b * ~(on[:, None] & on[None, :]), 1)
    first, second = _hydrogen_weights(reaction, secondary, labile_h, ch_places)
    moved = np.abs(b["h"] - a["h"][mapping]) * on
    sources = entered(reaction, mapping)
    places = (
        int(
            (((after > 0) != (before > 0)) & mask).sum()
            + (outside > 0).sum()
            + (first * moved).sum()
        )
        + sources
    )
    counted = mask if all_orders else mask & (after > 0) & (before > 0)
    tokens = (
        secondary[0]
        * 2
        * (np.abs(after - before)[counted].sum() + all_orders * outside.sum())
        + (second * moved).sum()
        + secondary[2] * (np.abs(b["q"] - a["q"][mapping]) * on).sum()
        + sources
    )

    return places, int(round(tokens))


def star_bound(reaction):
    """A lower bound on the number of bond places that change, from one linear assignment. An atom seated on p changes
    at least the l1 distance of the element counts of the two neighbourhoods, and every changed place is seen from at
    most two seated atoms."""
    a, b = reaction["a"], reaction["b"]
    count = lambda g: np.stack(
        [
            np.bincount(
                g["element"][np.nonzero(row)[0]].astype(np.int64), minlength=120
            )
            for row in dense_bonds(g) > 0
        ]
    )
    distance = (
        np.abs(count(b)[:, None, :] - count(a)[None, :, :]).sum(-1).astype(np.float64)
    )
    distance[b["element"][:, None] != a["element"][None, :]] = 1e6
    rows, cols = linear_sum_assignment(distance)
    total = distance[rows, cols].sum()

    return int(np.ceil(total / 2)) if total < 1e6 else 0
