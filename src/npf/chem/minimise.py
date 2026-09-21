"""Minimum firing vector between two markings of the valence net, used as an atom mapping.

A mapping of the product atoms onto the precursor atoms determines the firing vector sigma with m_B = m_A + C sigma,
and its size is the number of tokens that move. Choosing the mapping is therefore maximum a posteriori inference in a
pairwise model, unary terms for hydrogens and charges, pairwise terms for bond orders, under an injectivity
constraint. A firing only changes the R-ball of the places it touches (the cancellation proposition), so atoms whose
environment is the same on both sides can only be seated on matching atoms, which cuts the search down to the
reaction centre.
"""
import numpy as np

from .featurisation import BOND_ORDER, dense_bonds

DEPTH = 4


def colours(graph, depth=DEPTH):
    """Weisfeiler Lehman colours at every depth, from element, charge, hydrogen count and bond orders only."""
    bonds = dense_bonds(graph)
    neighbours = [np.nonzero(bonds[i])[0] for i in range(len(bonds))]
    colour = np.array([hash((int(e), int(q), int(h))) for e, q, h in zip(graph["element"], graph["q"], graph["h"])], np.int64)
    out = [colour]
    for _ in range(depth):
        colour = np.array([hash((int(colour[i]), tuple(sorted((int(bonds[i, j]), int(colour[j])) for j in neighbours[i]))))
                           for i in range(len(colour))], np.int64)
        out.append(colour)

    return out


def domains(reaction, depth=DEPTH):
    """Candidate precursor atoms for every product atom, the deepest environment that still has a match."""
    a, b = reaction["a"], reaction["b"]
    ca, cb = colours(a, depth), colours(b, depth)
    out = []
    for i in range(len(b["x"])):
        same_element = a["element"] == b["element"][i]
        best = np.nonzero(same_element)[0]

        for d in range(depth, 0, -1):
            match = np.nonzero(same_element & (ca[d] == cb[d][i]))[0]
            if len(match):
                best = match
                break

        out.append(best)

    return out


def cost_of(reaction, mapping):
    """Tokens that move under a mapping, the size of the firing vector."""
    a, b = reaction["a"], reaction["b"]
    before = BOND_ORDER[dense_bonds(a)]
    after = np.zeros_like(before)
    after[np.ix_(mapping, mapping)] = BOND_ORDER[dense_bonds(b)]
    kept = np.zeros(len(before), bool)
    kept[mapping] = True
    bonds = np.abs(np.triu((after - before) * (kept[:, None] | kept[None, :]), 1)).sum()

    return float(bonds + np.abs(b["h"] - a["h"][mapping]).sum() + np.abs(b["q"] - a["q"][mapping]).sum())


def map_inference(reaction, unary, pair, incumbent, budget=200000, depth=DEPTH):
    """Exact maximum a posteriori seating under an arbitrary pairwise energy.

    unary[i, j] is the energy of seating product atom i on precursor atom j, pair(i, j, i2, j2) the energy of a pair
    of seats. Both are non-negative, so the energy of a partial seating bounds every completion from below and a
    branch that reaches the incumbent can be cut. With unary = token moves and pair = bond changes this is the
    minimum firing vector, with the energies of a trained mapper it is the mode of its own distribution.
    """
    b = reaction["b"]
    doms = domains(reaction, depth)
    free = sorted(range(len(b["x"])), key=lambda i: (len(doms[i]), -float(unary[i].min())))
    floor = np.array([min(unary[i][j] for j in doms[i]) for i in range(len(b["x"]))])
    tail = np.zeros(len(free) + 1)
    for k in range(len(free) - 1, -1, -1):
        tail[k] = tail[k + 1] + floor[free[k]]

    energy = lambda m: sum(unary[i, m[i]] for i in range(len(m))) + sum(pair(i, m[i], i2, m[i2]) for i in range(len(m)) for i2 in range(i))
    best, best_map, steps, exhaustive = energy(incumbent), incumbent.copy(), 0, True

    def recurse(k, mapping, seated, used, partial):
        nonlocal best, best_map, steps, exhaustive

        if steps > budget:
            exhaustive = False
            return

        if partial + tail[k] >= best - 1e-9:
            return

        if k == len(free):
            if partial < best - 1e-9:
                best, best_map = partial, mapping.copy()

            return

        i = free[k]
        for j in doms[i]:
            if j in used:
                continue

            steps += 1
            add = unary[i, j] + sum(pair(i, j, i2, mapping[i2]) for i2 in seated)
            mapping[i] = j
            used.add(j)
            seated.append(i)
            recurse(k + 1, mapping, seated, used, partial + add)
            seated.pop()
            used.discard(j)
            mapping[i] = -1

    recurse(0, np.full(len(b["x"]), -1), [], set(), 0.0)

    return best_map, best, exhaustive


def branch_and_bound(reaction, incumbent, budget=200000, depth=DEPTH):
    """Exact minimiser over the domains, unless the budget runs out.

    Atoms are seated in order of how constrained they are. The bound is the cost among the seated atoms plus the
    unavoidable unary cost of the rest, both of which only grow, so a partial seating that reaches the incumbent is cut.
    """
    a, b = reaction["a"], reaction["b"]
    order_a, order_b = BOND_ORDER[dense_bonds(a)], BOND_ORDER[dense_bonds(b)]
    doms = domains(reaction, depth)
    free = sorted(range(len(b["x"])), key=lambda i: (len(doms[i]), -int((order_b[i] > 0).sum())))

    # cheapest hydrogen and charge move still to come, a valid bound for the unseated atoms
    unary = np.array([min(abs(b["h"][i] - a["h"][j]) + abs(b["q"][i] - a["q"][j]) for j in doms[i]) for i in range(len(b["x"]))])
    tail = np.zeros(len(free) + 1)
    for k in range(len(free) - 1, -1, -1):
        tail[k] = tail[k + 1] + unary[free[k]]

    best, best_map, steps, exhaustive = cost_of(reaction, incumbent), incumbent.copy(), 0, True

    def recurse(k, mapping, seated, used, partial):
        nonlocal best, best_map, steps, exhaustive

        if steps > budget:
            exhaustive = False
            return

        if partial + tail[k] >= best - 1e-9:
            return

        if k == len(free):
            # bonds to the precursor atoms that leave are not in the incremental sum, so score the whole mapping
            total = cost_of(reaction, mapping)
            if total < best - 1e-9:
                best, best_map = total, mapping.copy()

            return

        i = free[k]
        for j in doms[i]:
            if j in used:
                continue

            steps += 1

            # cost this seat adds, bonds to the atoms already seated plus its own hydrogens and charge
            idx = np.array(seated, dtype=int) if seated else np.zeros(0, int)
            add = abs(b["h"][i] - a["h"][j]) + abs(b["q"][i] - a["q"][j])

            if len(idx):
                add += float(np.abs(order_b[i, idx] - order_a[j, mapping[idx]]).sum())

            mapping[i] = j
            used.add(j)
            seated.append(i)
            recurse(k + 1, mapping, seated, used, partial + add)
            seated.pop()
            used.discard(j)
            mapping[i] = -1

        return

    recurse(0, np.full(len(b["x"]), -1), [], set(), 0.0)

    return best_map, best, exhaustive


def weighted_cost(reaction, mapping, weights=(0.0, 1.0, 0.0, 0.0)):
    """Size of the firing vector with separate weights for its four kinds of token moves.

    weights are (bond order tokens, bond places that change between empty and marked, hydrogens, charges). The
    default counts only the bond places that a firing empties or fills, which is the number of bonds made and
    broken, and the cost under which curated maps are least often beaten by wrong ones.
    """
    a, b = reaction["a"], reaction["b"]
    before = BOND_ORDER[dense_bonds(a)]
    after = np.zeros_like(before)
    after[np.ix_(mapping, mapping)] = BOND_ORDER[dense_bonds(b)]
    kept = np.zeros(len(before), bool)
    kept[mapping] = True
    mask = np.triu(kept[:, None] | kept[None, :], 1)
    w_order, w_link, w_h, w_q = weights

    return float(w_order * np.abs((after - before) * mask).sum() + w_link * (((after > 0) != (before > 0)) & mask).sum()
                 + w_h * np.abs(b["h"] - a["h"][mapping]).sum() + w_q * np.abs(b["q"] - a["q"][mapping]).sum())


def ordered_domains(reaction, depth=DEPTH):
    """Every precursor atom of the right element for every product atom, the most similar environment first.

    Nothing is excluded, because an atom next to the reaction centre changes its environment and its true seat
    then shares no deep colour with it. The order only decides what the search tries first.
    """
    a, b = reaction["a"], reaction["b"]
    ca, cb = colours(a, depth), colours(b, depth)
    out = []
    for i in range(len(b["x"])):
        seats = np.nonzero(a["element"] == b["element"][i])[0]
        agreement = np.zeros(len(seats))

        for d in range(1, depth + 1):
            agreement += ca[d][seats] == cb[d][i]

        out.append(seats[np.argsort(-agreement, kind="stable")])

    return out


def minimise_weighted(reaction, incumbent, weights=(0.0, 1.0, 0.0, 0.0), prior=None, budget=60000, depth=DEPTH):
    """Branch and bound for the weighted cost over the full domains, seeded with a feasible mapping.

    prior[i, j] is an optional small non-negative energy of seating i on j, for instance the negative log probability
    of a trained mapper scaled down so that it can never outweigh one unit of the cost. It then only orders the
    mappings that the net cannot tell apart. Returns the mapping, its cost and whether the search was exhaustive.
    """
    a, b = reaction["a"], reaction["b"]
    order_a, order_b = BOND_ORDER[dense_bonds(a)], BOND_ORDER[dense_bonds(b)]
    link_a, link_b = order_a > 0, order_b > 0
    w_order, w_link, w_h, w_q = weights
    n_b = len(b["x"])
    unary = w_h * np.abs(b["h"][:, None] - a["h"][None, :]) + w_q * np.abs(b["q"][:, None] - a["q"][None, :])

    if prior is not None:
        unary = unary + prior

    doms = ordered_domains(reaction, depth)

    # atoms with many bonds first, their seats constrain the most
    free = sorted(range(n_b), key=lambda i: (-int(link_b[i].sum()), len(doms[i])))
    floor = np.array([unary[i, doms[i]].min() if len(doms[i]) else 0.0 for i in range(n_b)])
    tail = np.zeros(n_b + 1)
    for k in range(n_b - 1, -1, -1):
        tail[k] = tail[k + 1] + floor[free[k]]

    def total(mapping):
        return weighted_cost(reaction, mapping, weights) + (float(prior[np.arange(n_b), mapping].sum()) if prior is not None else 0.0)

    best, best_map, steps, exhaustive = total(incumbent), incumbent.copy(), 0, True

    def recurse(k, mapping, seated, used, partial):
        nonlocal best, best_map, steps, exhaustive

        if steps > budget:
            exhaustive = False
            return

        if partial + tail[k] >= best - 1e-9:
            return

        if k == n_b:
            value = total(mapping)
            if value < best - 1e-9:
                best, best_map = value, mapping.copy()

            return

        i = free[k]
        idx = np.array(seated, dtype=int)
        for j in doms[i]:
            if j in used:
                continue

            steps += 1
            add = unary[i, j]

            if len(idx):
                cols = mapping[idx]
                add += w_order * np.abs(order_b[i, idx] - order_a[j, cols]).sum() + w_link * (link_b[i, idx] != link_a[j, cols]).sum()

            if partial + add + tail[k + 1] >= best - 1e-9:
                continue

            mapping[i] = j
            used.add(j)
            seated.append(i)
            recurse(k + 1, mapping, seated, used, partial + add)
            seated.pop()
            used.discard(j)
            mapping[i] = -1

    recurse(0, np.full(n_b, -1), [], set(), 0.0)

    return best_map, best, exhaustive

