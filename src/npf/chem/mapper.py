"""The atom mapper, the minimum firing vector of the valence net found and proved without a solver.

The search of npf.chem.search returns the lexicographic minimum of cost.CHOSEN with every optimal mapping, the classes
of these mappings up to isomorphism of the condensed graph are ranked by the third level of npf.chem.third_level, and
of the classes that remain the one with the smallest graph hash is taken, so the map does not depend on the order of
the atoms. solve, cheapest_mappings and solve_open keep the signatures of their counterparts in npf.chem.exact and
npf.chem.open_net, which remain as the integer program they were checked against.

    mapping, levels, proved = best_mapping(reaction)
"""

import numpy as np

from . import cost
from .cgr import key, same_cgr
from .open_net import deficit, with_equivalents
from .search import Problem, Search
from .third_level import terms

# one unit of the deterministic budget of npf.chem.exact is read as this many nodes of the search
NODES_PER_UNIT = 10000


def _options(secondary, labile_h, ch_places, all_orders):
    if all_orders:
        raise NotImplementedError("all_orders is not bounded by the search")

    return dict(secondary=secondary, labile_h=labile_h, ch_places=ch_places)


def _budget(seconds, deterministic):
    nodes = int(deterministic * NODES_PER_UNIT) if deterministic else 10**9

    return nodes, (None if deterministic else seconds)


def _hint(pr, hint):
    """A complete seating by element, the first incumbent, or None."""
    if hint is None:
        return None

    hint = np.asarray(hint, np.int64)
    if len(hint) != pr.nb or len(set(hint.tolist())) != pr.nb or (hint < 0).any():
        return None

    return hint if (pr.ea[hint] == pr.eb).all() else None


def expand_orbits(pr, mappings, rounds=4):
    """The listing closed under the automorphisms of the precursors that the search skipped, one mapping per firing
    vector as exact.cheapest_mappings lists them."""
    out = {m.tobytes(): m for m in (np.asarray(x, np.int64) for x in mappings)}
    for _ in range(rounds):
        new = {}
        for m in out.values():
            for s in pr.automorphisms:
                x = np.where(m >= 0, s[np.maximum(m, 0)], m)
                new.setdefault(x.tobytes(), x)

        size = len(out)
        out.update(new)

        if len(out) == size:
            break

    return list(out.values())


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
    """Cheapest mapping, its cost in bond places, and whether optimality was proved, as exact.solve. workers is
    accepted and ignored."""
    pr = Problem(
        reaction,
        sources=sources,
        **_options(secondary, labile_h, ch_places, all_orders),
    )
    nodes, wall = _budget(seconds, deterministic)
    search = Search(pr, _hint(pr, hint), node_limit=nodes, seconds=wall).run()
    if search.best is None:
        return None, None, False

    return search.best, int(search.best_levels[0]), bool(search.complete)


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
    expand=True,
):
    """Every mapping of proved minimum cost up to limit and True, as exact.cheapest_mappings, else the cheapest mapping
    found and False. True also means that the listing finished. Mappings that differ by a symmetry of the product
    are listed once, those that differ by a symmetry of the precursors are added back with expand.
    """
    pr = Problem(reaction, **_options(secondary, labile_h, ch_places, all_orders))
    nodes, wall = _budget(seconds, deterministic)
    search = Search(
        pr,
        _hint(pr, hint),
        node_limit=nodes,
        seconds=wall,
        enumerate_ties=True,
        tie_limit=limit,
    ).run()
    if search.best is None:
        return [], False

    if not search.complete:
        return [search.best], False

    maps = expand_orbits(pr, search.ties) if expand else list(search.ties)

    return maps[:limit], True


def solve_open(reaction, **options):
    """Cheapest mapping of a reaction that may lack reactants, as open_net.solve_open. A seat of -1 marks a product
    atom that no written molecule supplies."""
    if not deficit(reaction).any():
        return (*solve(reaction, **options), reaction)

    opened = with_equivalents(reaction)
    options.pop("hint", None)
    short = {int(e) for e in np.nonzero(deficit(reaction))[0]}

    return (*solve(opened, sources=short, **options), opened)


def tie_classes(reaction, mappings):
    """Mappings merged by isomorphism of the condensed graph of reaction, one representative per class, in order."""
    out = []
    for m in mappings:
        k = key(reaction, m)
        if not any(k == other and same_cgr(reaction, m, first) for other, first in out):
            out.append((k, m))

    return out


def choose(reaction, mappings):
    """The mapping of the paper among optimal mappings, ranked by the third level, and of the classes it cannot tell
    apart the one with the smallest graph hash."""
    classes = tie_classes(reaction, [np.asarray(m, np.int64) for m in mappings])
    scores = [terms(reaction, m) for _, m in classes]
    best = min(scores)
    winners = [c for c, s in enumerate(scores) if s == best]

    return classes[min(winners, key=lambda c: classes[c][0])][1]


def best_mapping(reaction, seconds=60.0, tie_limit=512, nodes=10**6):
    """The map of the paper, the mapping, its two levels and whether the search finished. A reaction whose products
    hold atoms the precursors cannot supply is solved on the open net, where the third level does not apply, and a
    product atom that no written precursor atom supplies has the seat -1."""
    if deficit(reaction).any():
        mapping, _, proved, _ = solve_open(reaction, seconds=seconds, **cost.CHOSEN)
        if mapping is None:
            return None, None, proved

        # seats on the copies and sources that the open net adds do not exist in the written reaction
        n = len(reaction["a"]["element"])

        return np.where((mapping >= 0) & (mapping < n), mapping, -1), None, proved

    pr = Problem(reaction, **cost.CHOSEN)
    search = Search(
        pr,
        None,
        node_limit=nodes,
        seconds=seconds,
        enumerate_ties=True,
        tie_limit=tie_limit,
    ).run()
    if search.best is None:
        return None, None, False

    mapping = choose(reaction, search.ties or [search.best])

    return mapping, tuple(int(x) for x in search.best_levels), bool(search.complete)
