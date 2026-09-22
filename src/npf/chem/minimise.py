"""The feasible start of the exact mapper in chem.exact, a seating found without a solver.

A mapping of the product atoms onto the precursor atoms determines the firing vector sigma with m_B = m_A + C sigma,
and its size is the number of tokens that move. A firing only changes the R-ball of the places it touches, so atoms
with the same environment on both sides are seated on matching atoms, which cuts the search to the reaction centre.
Token descent and a bounded branch and bound give a cheap seating that the integer program never returns worse than.
"""

import numpy as np

from .featurisation import BOND_ORDER, dense_bonds

DEPTH = 4


def colours(graph, depth=DEPTH):
    """Weisfeiler Lehman colours at every depth, from element, charge, hydrogen count and bond orders only."""
    bonds = dense_bonds(graph)
    neighbours = [np.nonzero(bonds[i])[0] for i in range(len(bonds))]
    colour = np.array(
        [
            hash((int(e), int(q), int(h)))
            for e, q, h in zip(graph["element"], graph["q"], graph["h"])
        ],
        np.int64,
    )
    out = [colour]
    for _ in range(depth):
        colour = np.array(
            [
                hash(
                    (
                        int(colour[i]),
                        tuple(
                            sorted(
                                (int(bonds[i, j]), int(colour[j]))
                                for j in neighbours[i]
                            )
                        ),
                    )
                )
                for i in range(len(colour))
            ],
            np.int64,
        )
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

    return float(
        bonds
        + np.abs(b["h"] - a["h"][mapping]).sum()
        + np.abs(b["q"] - a["q"][mapping]).sum()
    )


def branch_and_bound(reaction, incumbent, budget=200000, depth=DEPTH):
    """Exact minimiser over the domains unless the budget runs out. Atoms are seated most constrained first, and the
    bound is the cost among the seated atoms plus the unavoidable unary cost of the rest, both of which only grow, so
    a partial seating that reaches the incumbent is cut."""
    a, b = reaction["a"], reaction["b"]
    order_a, order_b = BOND_ORDER[dense_bonds(a)], BOND_ORDER[dense_bonds(b)]
    doms = domains(reaction, depth)
    free = sorted(
        range(len(b["x"])), key=lambda i: (len(doms[i]), -int((order_b[i] > 0).sum()))
    )

    # cheapest hydrogen and charge move still to come, a valid bound for the unseated atoms
    unary = np.array(
        [
            min(
                abs(b["h"][i] - a["h"][j]) + abs(b["q"][i] - a["q"][j]) for j in doms[i]
            )
            for i in range(len(b["x"]))
        ]
    )
    tail = np.zeros(len(free) + 1)
    for k in range(len(free) - 1, -1, -1):
        tail[k] = tail[k + 1] + unary[free[k]]

    best, best_map, steps, exhaustive = (
        cost_of(reaction, incumbent),
        incumbent.copy(),
        0,
        True,
    )

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
    """Size of the firing vector with weights for (bond order tokens, bond places that change between empty and marked,
    hydrogens, charges). The default counts only the bonds made and broken, the cost under which curated maps are
    least often beaten by wrong ones."""
    a, b = reaction["a"], reaction["b"]
    before = BOND_ORDER[dense_bonds(a)]
    after = np.zeros_like(before)
    after[np.ix_(mapping, mapping)] = BOND_ORDER[dense_bonds(b)]
    kept = np.zeros(len(before), bool)
    kept[mapping] = True
    mask = np.triu(kept[:, None] | kept[None, :], 1)
    w_order, w_link, w_h, w_q = weights

    return float(
        w_order * np.abs((after - before) * mask).sum()
        + w_link * (((after > 0) != (before > 0)) & mask).sum()
        + w_h * np.abs(b["h"] - a["h"][mapping]).sum()
        + w_q * np.abs(b["q"] - a["q"][mapping]).sum()
    )


def token_descent(reaction, mapping, passes=3):
    """Re-seat one product atom at a time, swapping if the seat is taken, whenever that strictly shrinks the firing
    vector."""
    a, b = reaction["a"], reaction["b"]
    mapping = mapping.copy()
    best = cost_of(reaction, mapping)
    order_a, order_b = BOND_ORDER[dense_bonds(a)], BOND_ORDER[dense_bonds(b)]
    for _ in range(passes):
        improved = False

        # only atoms that take part in a firing under the current mapping can be badly seated
        mism = (
            np.abs(order_b - order_a[np.ix_(mapping, mapping)]).sum(1)
            + np.abs(b["h"] - a["h"][mapping])
            + np.abs(b["q"] - a["q"][mapping])
        )
        for i in np.nonzero(mism > 0)[0]:
            for j in np.nonzero(a["element"] == b["element"][i])[0]:
                if j == mapping[i]:
                    continue

                trial = mapping.copy()
                holder = np.nonzero(mapping == j)[0]
                if len(holder):
                    trial[holder[0]] = mapping[i]

                trial[i] = j
                cost = cost_of(reaction, trial)
                if cost < best - 1e-9:
                    mapping, best, improved = trial, cost, True

        if not improved:
            break

    return mapping


def feasible_start(reaction, budget=20000):
    """A seating by element and environment, improved by token descent and branch and bound on the tokens moved. None
    when some product atom has no precursor atom of its element left."""
    out, used = np.zeros(len(reaction["b"]["x"]), np.int64), set()
    for i, d in enumerate(domains(reaction)):
        free = [int(j) for j in d if int(j) not in used]
        if not free:
            free = [
                int(j)
                for j in np.nonzero(
                    reaction["a"]["element"] == reaction["b"]["element"][i]
                )[0]
                if int(j) not in used
            ]

        if not free:
            return None

        out[i] = free[0]
        used.add(free[0])

    return branch_and_bound(reaction, token_descent(reaction, out), budget=budget)[0]
