"""Exact minimum firing vector between two markings of the valence net, as an integer program.

A mapping seats every product atom on a precursor atom of the same element. The cost is the number of bond places a
firing empties or fills, the bonds made and broken, not the distance of Jochum, Gasteiger and Ugi, which counts
valence electrons. A product bond is kept when its atoms sit on the two ends of a precursor bond, every other product
bond was formed and every precursor bond with a seated end that is not kept was broken. Below that level the tokens
that move on kept bonds, on hydrogen and on charge are counted. The solver proves optimality and can list every
optimal mapping, the ties the net cannot break.
"""

import numpy as np
from ortools.sat.python import cp_model
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


def _model(
    reaction,
    secondary=(0, 0, 0),
    all_orders=False,
    labile_h=True,
    ch_places=False,
    sources=(),
):
    """secondary = integer weights of (bond order tokens on kept bonds, hydrogen moves, charge moves), each far below
    the scale of one bond place, so they only order the mappings that the connectivity cost cannot tell apart.
    sources = the elements whose product atoms may enter from outside instead of taking a seat, see open_net.
    """
    a, b = reaction["a"], reaction["b"]
    types_a, types_b = dense_bonds(a), dense_bonds(b)
    bonds_a, bonds_b = types_a > 0, types_b > 0
    n_a, n_b = len(a["x"]), len(b["x"])
    w_order, w_h, w_q = secondary
    model = cp_model.CpModel()
    seat, free, meaning = {}, {}, []
    for i in range(n_b):
        for j in np.nonzero(a["element"] == b["element"][i])[0]:
            seat[i, int(j)] = model.NewBoolVar(f"x{i}_{j}")

        if int(b["element"][i]) in sources:
            free[i] = model.NewBoolVar(f"s{i}")
            meaning.append((free[i], "free", i))

        model.AddExactlyOne(
            [seat[i, j] for (k, j) in seat if k == i] + ([free[i]] if i in free else [])
        )

    seats_of = {i: [j for (k, j) in seat if k == i] for i in range(n_b)}
    kept_atom = {}
    for j in range(n_a):
        users = [seat[i, j] for i in range(n_b) if (i, j) in seat]
        kept_atom[j] = model.NewBoolVar(f"u{j}")
        meaning.append((kept_atom[j], "seated", j))
        model.Add(sum(users) == kept_atom[j]) if users else model.Add(kept_atom[j] == 0)

    kinds = sorted(set(types_a[bonds_a].tolist())) if w_order else [None]
    neighbours = {
        kind: [
            set(
                np.nonzero(bonds_a[j] if kind is None else types_a[j] == kind)[
                    0
                ].tolist()
            )
            for j in range(n_a)
        ]
        for kind in kinds
    }
    kept, order_cost, kept_at = (
        [],
        [],
        {(i, kind): [] for i in range(n_b) for kind in kinds},
    )
    for i, k in zip(*np.nonzero(np.triu(bonds_b, 1))):
        i, k = int(i), int(k)
        for kind in kinds:
            y = model.NewBoolVar("")
            meaning.append((y, "kept", (i, k, kind)))
            kept_at[i, kind].append((y, k))
            kept_at[k, kind].append((y, i))

            # a kept product bond lies on a precursor bond, so if one end sits on p the other sits on a neighbour of p
            for p in seats_of[i]:
                model.Add(
                    y
                    + seat[i, p]
                    - sum(seat[k, q] for q in seats_of[k] if q in neighbours[kind][p])
                    <= 1
                )

            for q in seats_of[k]:
                model.Add(
                    y
                    + seat[k, q]
                    - sum(seat[i, p] for p in seats_of[i] if p in neighbours[kind][q])
                    <= 1
                )

            # an atom that entered from outside keeps no bond
            for end in (i, k):
                if end in free:
                    model.Add(y + free[end] <= 1)

            kept.append(y)

            if kind is not None and all_orders:
                # a kept bond moves the difference of the orders and a dropped one all of them, so keeping saves twice
                # the smaller order, doubled to keep aromatic halves whole
                order_cost.append(
                    -int(round(4 * min(BOND_ORDER[types_b[i, k]], BOND_ORDER[kind])))
                    * y
                )
            elif kind is not None:
                order_cost.append(
                    int(round(2 * abs(BOND_ORDER[types_b[i, k]] - BOND_ORDER[kind])))
                    * y
                )

        if len(kinds) > 1:
            model.AddAtMostOne(kept[-len(kinds) :])

    # a redundant cut, an atom seated on p keeps at most as many bonds as p has neighbours of the right elements
    for (i, kind), ys in kept_at.items():
        if not ys:
            continue

        wanted = np.bincount([int(b["element"][k]) for _, k in ys], minlength=120)
        room = {
            p: int(
                np.minimum(
                    wanted,
                    np.bincount(
                        a["element"][sorted(neighbours[kind][p])].astype(np.int64),
                        minlength=120,
                    ),
                ).sum()
            )
            for p in seats_of[i]
        }
        model.Add(
            sum(y for y, _ in ys) <= sum(room[p] * seat[i, p] for p in seats_of[i])
        )

    touched = []
    for j, l in zip(*np.nonzero(np.triu(bonds_a, 1))):
        t = model.NewBoolVar("")
        model.AddMaxEquality(t, [kept_atom[int(j)], kept_atom[int(l)]])
        meaning.append((t, "touched", (int(j), int(l))))
        touched.append(t)

        if w_order and all_orders:
            order_cost.append(int(round(2 * BOND_ORDER[types_a[j, l]])) * t)

    # a further copy of a precursor molecule enters when one of its atoms is seated, and only after the copy before it
    copies, arrived = a.get("copy", np.zeros(n_a, np.int64)), {}
    for fragment, copy in sorted(
        {(int(f), int(c)) for f, c in zip(a["fragment"], copies) if c > 0}
    ):
        atoms = [
            j for j in range(n_a) if a["fragment"][j] == fragment and copies[j] == copy
        ]
        arrived[fragment, copy] = model.NewBoolVar("")
        model.AddMaxEquality(arrived[fragment, copy], [kept_atom[j] for j in atoms])
        meaning.append((arrived[fragment, copy], "arrived", atoms))

        if (fragment, copy - 1) in arrived:
            model.Add(arrived[fragment, copy] <= arrived[fragment, copy - 1])

    # every firing of a source fills one place, and at equal cost the mapping with fewer of them is preferred
    n_sources = sum(free.values()) + sum(arrived.values())
    n_bonds_b = int(np.triu(bonds_b, 1).sum())
    first_h, second_h = _hydrogen_weights(reaction, secondary, labile_h, ch_places)
    seat_cost = {
        (i, j): int(
            round(
                second_h[i] * abs(b["h"][i] - a["h"][j])
                + w_q * abs(b["q"][i] - a["q"][j])
            )
        )
        for (i, j) in seat
    }
    seat_places = {
        (i, j): int(first_h[i] * abs(b["h"][i] - a["h"][j])) for (i, j) in seat
    }
    lower = (
        w_order * sum(order_cost)
        + sum(c * seat[k] for k, c in seat_cost.items() if c)
        + n_sources
    )

    if w_order and all_orders:
        # the tokens of all product bonds, which a kept bond then saves, so the level stays non-negative
        lower = lower + w_order * int(
            round(2 * BOND_ORDER[types_b[np.triu(bonds_b, 1)]].sum())
        )

    # the two levels are strictly lexicographic, the second can never outweigh one bond place
    bound = (
        w_order * 6 * (n_bonds_b + len(touched))
        + sum(max([seat_cost[i, j] for j in seats_of[i]] or [0]) for i in range(n_b))
        + len(free)
        + len(arrived)
    )
    scale = bound + 1

    # formed bonds are product bonds not kept, broken ones precursor bonds with a seated end and no kept bond on them
    bonds = n_bonds_b + sum(touched) - 2 * sum(kept)

    if not sources:
        model.Add(bonds >= star_bound(reaction))

    levels = (
        scale
        * (bonds + sum(c * seat[k] for k, c in seat_places.items() if c) + n_sources)
        + lower
    )
    model.Minimize(levels)

    return model, seat, n_b, scale, levels, meaning, (types_a, kinds)


def _write(model, seat, meaning, types, mapping):
    """A mapping written into every variable of the model, so that the solver starts from it as a solution."""
    types_a, kinds = types
    seated = set(mapping[mapping >= 0].tolist())

    for (i, j), v in seat.items():
        model.AddHint(v, int(mapping[i] == j))

    for v, what, key in meaning:
        if what == "seated":
            model.AddHint(v, int(key in seated))
        elif what == "touched":
            model.AddHint(v, int(key[0] in seated or key[1] in seated))
        elif what == "free":
            model.AddHint(v, int(mapping[key] < 0))
        elif what == "arrived":
            model.AddHint(v, int(any(j in seated for j in key)))
        else:
            i, k, kind = key
            on = (
                types_a[mapping[i], mapping[k]]
                if mapping[i] >= 0 and mapping[k] >= 0
                else 0
            )
            model.AddHint(v, int(on > 0 if kind is None else on == kind))


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


def _read(solver, seat, n_b):
    mapping = np.full(n_b, -1, np.int64)
    for (i, j), v in seat.items():
        if solver.Value(v):
            mapping[i] = j

    return mapping


def _solver(seconds, workers, deterministic):
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = workers
    solver.parameters.max_time_in_seconds = seconds

    if deterministic:
        # a budget in deterministic time and a fixed seed, so the result does not depend on the load of the machine
        solver.parameters.max_deterministic_time = deterministic
        solver.parameters.random_seed = 0
        solver.parameters.interleave_search = workers > 1

    return solver


def solve(
    reaction,
    seconds=10.0,
    hint=None,
    secondary=(0, 0, 0),
    workers=1,
    all_orders=False,
    deterministic=None,
    labile_h=True,
    ch_places=False,
    sources=(),
):
    """Cheapest mapping, its cost in bond places, and whether optimality was proved.

    hint = a feasible mapping. The solver starts from it and the result is never dearer, also when time runs out.
    """
    options = dict(
        secondary=secondary,
        all_orders=all_orders,
        labile_h=labile_h,
        ch_places=ch_places,
    )
    model, seat, n_b, scale, levels, meaning, types = _model(
        reaction, sources=sources, **options
    )
    value_of = (
        lambda m: scale * cost_levels(reaction, m, **options)[0]
        + cost_levels(reaction, m, **options)[1]
    )

    # only a complete seating is taken as a start, a hint with atoms from outside is ignored
    feasible = (
        hint is not None
        and len(set(np.asarray(hint).tolist())) == n_b
        and all((i, int(j)) in seat for i, j in enumerate(hint))
    )
    if feasible:
        hint = np.asarray(hint, np.int64)
        _write(model, seat, meaning, types, hint)

    solver = _solver(seconds, workers, deterministic)
    status = solver.Solve(model)
    found = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    if not found and not feasible:
        return None, None, False

    mapping = _read(solver, seat, n_b) if found else hint
    if feasible and value_of(hint) < value_of(mapping):
        mapping = hint

    value, proved = value_of(mapping), status == cp_model.OPTIMAL

    return mapping, value // scale, proved


class _Collect(cp_model.CpSolverSolutionCallback):
    def __init__(self, seat, n_b, limit):
        super().__init__()
        self.seat, self.n_b, self.limit, self.maps = seat, n_b, limit, []

    def on_solution_callback(self):
        self.maps.append(_read(self, self.seat, self.n_b))

        if len(self.maps) >= self.limit:
            self.StopSearch()


def cheapest_mappings(
    reaction,
    limit=64,
    seconds=10.0,
    hint=None,
    deterministic=None,
    workers=1,
    secondary=(0, 0, 0),
    all_orders=False,
    labile_h=True,
    ch_places=False,
):
    """Every mapping of proved minimum cost up to limit and True, else the cheapest mapping found in time and False.
    The list is empty without a feasible seating and partial when the listing runs out of budget. deterministic is a
    budget in deterministic time per solve, seconds then only guards the wall time, and workers share the proof while
    the listing of the ties runs on one."""
    options = dict(
        secondary=secondary,
        all_orders=all_orders,
        labile_h=labile_h,
        ch_places=ch_places,
    )
    model, seat, n_b, scale, levels, meaning, types = _model(reaction, **options)
    feasible = (
        hint is not None
        and len(set(np.asarray(hint).tolist())) == n_b
        and all((i, int(j)) in seat for i, j in enumerate(hint))
    )
    if feasible:
        hint = np.asarray(hint, np.int64)
        _write(model, seat, meaning, types, hint)

    solver = _solver(seconds, workers, deterministic)
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return ([hint] if feasible else []), False

    best = _read(solver, seat, n_b)

    if status == cp_model.FEASIBLE:
        value_of = (
            lambda m: scale * cost_levels(reaction, m, **options)[0]
            + cost_levels(reaction, m, **options)[1]
        )

        return [hint if feasible and value_of(hint) < value_of(best) else best], False

    # the objective becomes a constraint, then every solution is an optimal mapping
    model.ClearHints()
    model.Add(levels == int(round(solver.ObjectiveValue())))
    model.ClearObjective()
    solver = _solver(seconds, 1, deterministic)
    solver.parameters.enumerate_all_solutions = True
    collect = _Collect(seat, n_b, limit)
    solver.Solve(model, collect)

    return collect.maps or [best], True
