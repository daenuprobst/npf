"""Enabled linearisations of a firing vector on the valence net.

A firing vector says which bond places change, not in which order, and a transition that adds tokens to a bond place
needs free valence tokens on both atoms. The enabled orders are counted by a dynamic program over subsets, each the
marking reached after those firings. A firing vector with no enabled order cannot occur in the net, whatever the rate
law is.
"""

import numpy as np

from .featurisation import BOND_ORDER, EXTRA_CAPACITY, dense_bonds

MAX_FIRINGS = 12

# aromatic bonds count one and a half, so every atom carries half a token of tolerance
TOLERANCE = 0.5


def slack(reaction, bonds):
    """Free valence tokens of every atom at a marking, counted from the precursors."""
    a = reaction["a"]
    before = BOND_ORDER[dense_bonds(a)]
    capacity = np.array(
        [EXTRA_CAPACITY.get(int(e), 0) for e in a["element"]]
    ) + np.maximum(-a["q"], 0)

    return a["h"] + capacity - (BOND_ORDER[bonds] - before).sum(1)


def count_orders(reaction, edits, cap=MAX_FIRINGS):
    """Number of enabled orders of a firing vector, and whether any order is enabled."""
    k = len(edits)
    if k == 0:
        return 1, True

    if k > cap:
        return None, True

    base = dense_bonds(reaction["a"])
    reach = np.zeros(1 << k, np.float64)
    reach[0] = 1.0

    for s in range(1 << k):
        if reach[s] == 0.0:
            continue

        bonds = base.copy()
        for t in range(k):
            if s >> t & 1:
                i, j, typ = edits[t]
                bonds[i, j] = bonds[j, i] = typ

        free = slack(reaction, bonds)
        for t in range(k):
            if s >> t & 1:
                continue

            i, j, typ = edits[t]
            gain = BOND_ORDER[typ] - BOND_ORDER[bonds[i, j]]

            # a firing that adds tokens to the bond place needs them on both atoms
            if gain <= free[i] + TOLERANCE and gain <= free[j] + TOLERANCE:
                reach[s | 1 << t] += reach[s]

    return float(reach[-1]), reach[-1] > 0
