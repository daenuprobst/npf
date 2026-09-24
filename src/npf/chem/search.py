"""The lexicographic minimum firing vector of cost.CHOSEN without a solver, by branch and bound.

A mapping seats every product atom i on a precursor atom pi(i) of the same element. With S the seated product atoms
of a search node, U the unseated ones and F the precursor atoms still free, the first level splits exactly into

    g1                          changed bond places among S and hydrogen moves on carbon of S
    + sum_i cross(i, pi(i))     places between a seat of U and a seat of S, known once i is seated
    + sum_q unused del(q)       precursor bonds from a seat of S to an atom q of F that leaves
    + inner                     places among the seats of U and from them to leaving atoms of F

and inner >= 1/2 sum_i d(i, pi(i)), d the l1 distance of the element counts of the neighbours of i in U and of pi(i)
in F, since a seated atom changes at least that many incident places and a place has two ends. So one linear
assignment of the rows U, padded by rows that delete a column at cost del(q), bounds the first level of every
completion. The second level (order tokens on kept bonds and charge moves) is bounded the same way with no inner term.
The dual of the assignment gives reduced costs, and forcing i onto q raises the bound by at least the reduced cost, so
seats that cannot beat the incumbent are removed from the whole subtree. Atoms left with one seat are seated at once,
the rest are branched most constrained first. Two free precursor atoms with the same bonds (twins) are exchanged by an
automorphism that fixes the node, so only one of them is branched on. The search is exhaustive unless the node budget
runs out, so a finished search proves the minimum, and a root bound equal to the incumbent is a certificate on its own.

"""

import itertools
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import cost
from .featurisation import BOND_ORDER, dense_bonds

BIG = 10**7


DEG = 4


def _neighbours(bonds, width):
    """Neighbours of every atom padded with the index n, and which atoms have more than width of them."""
    n = len(bonds)
    degree = bonds.sum(1)
    out = np.full((n, width), n, np.int64)
    for i in range(n):
        nb = np.nonzero(bonds[i])[0][:width]
        out[i, : len(nb)] = nb

    return out, degree > width


CAP = 256

# nodes after which the automorphisms of the precursors are computed for pruning
SYMMETRY_AFTER = 20


def _twins(kind, ta):
    """Pairs of atoms of the same kind whose bonds to every third atom agree."""
    n = len(kind)
    twin = np.zeros((n, n), bool)
    for p in range(n):
        others = np.nonzero((kind == kind[p]).all(1))[0]
        for r in others[others > p]:
            row_p, row_r = ta[p].copy(), ta[r].copy()
            row_p[[p, r]] = 0
            row_r[[p, r]] = 0

            if np.array_equal(row_p, row_r):
                twin[p, r] = twin[r, p] = True

    return twin


def _automorphisms(pr, kind, ta):
    import networkx as nx
    from networkx.algorithms.isomorphism import GraphMatcher

    a = pr.reaction["a"]
    copies = a.get("copy", np.zeros(pr.na, np.int64))
    graph = nx.Graph()
    for i in range(pr.na):
        graph.add_node(i, label=tuple(kind[i].tolist()))

    for i, j in zip(*np.nonzero(np.triu(ta, 1))):
        graph.add_edge(int(i), int(j), label=int(ta[i, j]))

    same = lambda x, y: x["label"] == y["label"]
    molecules = {}
    for i in range(pr.na):
        molecules.setdefault((int(a["fragment"][i]), int(copies[i])), []).append(i)

    # a molecule of the precursors is a connected part of one written fragment
    parts = []

    for atoms in molecules.values():
        parts += [sorted(c) for c in nx.connected_components(graph.subgraph(atoms))]

    out = []
    for atoms in parts:
        if len(atoms) < 2:
            continue

        matcher = GraphMatcher(
            graph.subgraph(atoms),
            graph.subgraph(atoms),
            node_match=same,
            edge_match=same,
        )
        for k, iso in enumerate(matcher.isomorphisms_iter()):
            if k >= CAP:
                break

            image = np.arange(pr.na)
            for u, v in iso.items():
                image[u] = v

            if (image != np.arange(pr.na)).any():
                out.append(image)

    # identical molecules swap as a whole, a further copy only when it is its whole copy, so the copies that enter
    # stay as many
    buckets = {}
    size = (
        np.bincount(pr.group[pr.group >= 0], minlength=pr.n_groups)
        if pr.n_groups
        else np.zeros(0, np.int64)
    )
    for atoms in parts:
        group = int(pr.group[atoms[0]])
        if group >= 0 and len(atoms) != size[group]:
            continue

        labels = tuple(sorted(tuple(kind[i].tolist()) for i in atoms))
        edges = graph.subgraph(atoms).number_of_edges()
        buckets.setdefault((labels, edges), []).append(atoms)

    for group in buckets.values():
        for x in range(len(group)):
            for y in range(x + 1, len(group)):
                if len(out) > 8 * CAP:
                    break

                matcher = GraphMatcher(
                    graph.subgraph(group[x]),
                    graph.subgraph(group[y]),
                    node_match=same,
                    edge_match=same,
                )
                if not matcher.is_isomorphic():
                    continue

                image = np.arange(pr.na)
                for u, v in matcher.mapping.items():
                    image[u], image[v] = v, u

                out.append(image)

    return np.array(out, np.int64).reshape(-1, pr.na)


class Problem:
    """The arrays of one reaction under one cost, product rows by precursor columns."""

    def __init__(
        self, reaction, secondary=(0, 0, 0), labile_h=True, ch_places=False, sources=()
    ):
        a, b = reaction["a"], reaction["b"]
        self.reaction = reaction
        self.options = dict(secondary=secondary, labile_h=labile_h, ch_places=ch_places)
        self.na, self.nb = len(a["element"]), len(b["element"])
        ta, tb = dense_bonds(a).astype(np.int64), dense_bonds(b).astype(np.int64)
        self.pa, self.pb = ta > 0, tb > 0
        self.pa_i, self.pb_i = self.pa.astype(np.int64), self.pb.astype(np.int64)

        # orders doubled, so an aromatic bond is a whole number of tokens
        self.oa2 = np.rint(2 * BOND_ORDER[ta]).astype(np.int64)
        self.ob2 = np.rint(2 * BOND_ORDER[tb]).astype(np.int64)
        self.ea, self.eb = a["element"].astype(np.int64), b["element"].astype(np.int64)
        self.same = self.eb[:, None] == self.ea[None, :]
        first, second = cost._hydrogen_weights(reaction, secondary, labile_h, ch_places)
        dh = np.abs(b["h"].astype(np.int64)[:, None] - a["h"].astype(np.int64)[None, :])
        dq = np.abs(b["q"].astype(np.int64)[:, None] - a["q"].astype(np.int64)[None, :])
        self.unary1 = (first[:, None] * dh).astype(np.int64)
        self.unary2 = (second[:, None] * dh + secondary[2] * dq).astype(np.int64)
        self.w_order = int(secondary[0])

        elements = np.unique(np.concatenate([self.ea, self.eb]))
        self.ka, self.kb = np.searchsorted(elements, self.ea), np.searchsorted(
            elements, self.eb
        )
        self.E = len(elements)
        self.count_a = np.zeros((self.na, self.E), np.int64)
        self.count_b = np.zeros((self.nb, self.E), np.int64)
        np.add.at(
            self.count_a, (np.nonzero(self.pa)[0], self.ka[np.nonzero(self.pa)[1]]), 1
        )
        np.add.at(
            self.count_b, (np.nonzero(self.pb)[0], self.kb[np.nonzero(self.pb)[1]]), 1
        )

        # padded neighbour lists for the matching bound, atoms with more than DEG neighbours fall back to counts
        width = int(
            min(DEG, max(self.pa.sum(1).max(initial=0), self.pb.sum(1).max(initial=0)))
        )
        self.nbr_a, self.wide_a = _neighbours(self.pa, width)
        self.nbr_b, self.wide_b = _neighbours(self.pb, width)
        self.perms = (
            np.array(list(itertools.permutations(range(self.nbr_a.shape[1]))), np.int64)
            if self.nbr_a.shape[1]
            else None
        )

        # further copies of a molecule enter at one place and one token each, see open_net
        copies = a.get("copy", np.zeros(self.na, np.int64))
        keys = sorted(
            {(int(f), int(c)) for f, c in zip(a["fragment"], copies) if c > 0}
        )
        self.group = np.full(self.na, -1, np.int64)

        for g, (f, c) in enumerate(keys):
            self.group[(a["fragment"] == f) & (copies == c)] = g

        self.n_groups = len(keys)
        self._auts = None
        self.free_ok = np.isin(self.eb, np.array(sorted(sources), np.int64))

        # twins, two precursor atoms of the same kind whose bonds to every third atom agree, the transposition of the
        # two is an automorphism of the precursors that keeps the cost, a further copy only swaps with a copy
        self.kind = np.stack([self.ea, a["h"], a["q"], self.group >= 0], 1).astype(
            np.int64
        )
        self.twin = _twins(self.kind, ta)

        # the first level alone sees elements, hydrogen on carbon and which bonds exist, not orders or charges, so
        # while only the first level is searched the oxygens of a carboxylate or a phosphate are interchangeable
        ch = (self.ea == 6) & bool(ch_places)
        self.kind1 = np.stack(
            [self.ea, np.where(ch, a["h"], 0), self.group >= 0], 1
        ).astype(np.int64)
        self.twin1 = _twins(self.kind1, self.pa_i)
        self._auts1 = None

    @property
    def automorphisms(self):
        """Automorphisms of the precursors that keep the cost, as rows of images, computed on first use. Those of
        every molecule (VF2, at most CAP each) and the swaps of two identical molecules. The list is not a group, every
        member is an automorphism, which is all the pruning needs."""
        if self._auts is None:
            self._auts = _automorphisms(
                self, self.kind, dense_bonds(self.reaction["a"])
            )

        return self._auts

    @property
    def automorphisms1(self):
        """Automorphisms of the precursors that keep the first level."""
        if self._auts1 is None:
            self._auts1 = _automorphisms(self, self.kind1, self.pa_i)

        return self._auts1

    def levels(self, mapping):
        """Both levels of a complete mapping, the same numbers as cost.cost_levels."""
        on = mapping >= 0
        seats = mapping[on]
        rows = np.nonzero(on)[0]
        seated = np.zeros(self.na, bool)
        seated[seats] = True
        pa_s = self.pa[np.ix_(seats, seats)]
        pb_s = self.pb[np.ix_(rows, rows)]
        changed = int(np.triu(pa_s != pb_s, 1).sum())

        # precursor bonds from a seated atom to one that leaves are broken
        leaving = int(self.pa[np.ix_(seats, np.nonzero(~seated)[0])].sum())
        outside = int(np.triu(self.pb & ~(on[:, None] & on[None, :]), 1).sum())
        kept = np.triu(pa_s & pb_s, 1)
        tokens = int(
            np.abs(self.oa2[np.ix_(seats, seats)] - self.ob2[np.ix_(rows, rows)])[
                kept
            ].sum()
        )
        sources = int((~on).sum()) + len(
            {int(self.group[p]) for p in seats if self.group[p] >= 0}
        )
        l1 = changed + leaving + outside + int(self.unary1[rows, seats].sum()) + sources
        l2 = self.w_order * tokens + int(self.unary2[rows, seats].sum()) + sources

        return l1, l2


class Node:
    """A partial seating with the incremental terms of the bound. seat -2 = unseated, -1 = entered from outside."""

    __slots__ = (
        "seat",
        "used",
        "g1",
        "g2",
        "cross1",
        "cross2",
        "del1",
        "src1",
        "nb_u",
        "na_f",
        "allowed",
        "src_ok",
        "arrived",
    )

    @classmethod
    def root(cls, pr):
        n = cls()
        n.seat = np.full(pr.nb, -2, np.int64)
        n.used = np.zeros(pr.na, bool)
        n.g1 = n.g2 = 0
        n.cross1 = np.zeros((pr.nb, pr.na), np.int64)
        n.cross2 = np.zeros((pr.nb, pr.na), np.int64)
        n.del1 = np.zeros(pr.na, np.int64)
        n.src1 = np.zeros(pr.nb, np.int64)
        n.nb_u = pr.count_b.copy()
        n.na_f = pr.count_a.copy()
        n.allowed = pr.same.copy()
        n.src_ok = pr.free_ok.copy()
        n.arrived = np.zeros(pr.n_groups, bool)

        return n

    def copy(self):
        n = Node()
        n.seat, n.used = self.seat.copy(), self.used.copy()
        n.g1, n.g2 = self.g1, self.g2
        n.cross1, n.cross2 = self.cross1.copy(), self.cross2.copy()
        n.del1, n.src1 = self.del1.copy(), self.src1.copy()
        n.nb_u, n.na_f = self.nb_u.copy(), self.na_f.copy()
        n.allowed, n.src_ok = self.allowed.copy(), self.src_ok.copy()
        n.arrived = self.arrived.copy()

        return n

    def place(self, pr, k, p):
        """Seat product atom k on precursor atom p, or let it enter from outside with p = -1, in place."""
        bk = pr.pb_i[:, k]

        if p >= 0:
            self.g1 += int(pr.unary1[k, p] + self.cross1[k, p])
            self.g2 += int(pr.unary2[k, p] + self.cross2[k, p])
            ap = pr.pa_i[:, p]

            # every pair (i, q) now knows whether the place between the seats of i and k changes
            self.cross1 += bk[:, None] ^ ap[None, :]

            if pr.w_order:
                both = np.outer(bk, ap).astype(bool)
                if both.any():
                    diff = np.abs(pr.ob2[:, k][:, None] - pr.oa2[:, p][None, :])
                    self.cross2 += pr.w_order * np.where(both, diff, 0)

            self.del1 += ap
            self.na_f[:, pr.ka[p]] -= ap
            self.used[p] = True
            self.allowed[:, p] = False

            if pr.group[p] >= 0 and not self.arrived[pr.group[p]]:
                self.arrived[pr.group[p]] = True
                self.g1 += 1
                self.g2 += 1
        else:
            # an atom from outside forms every bond it has, those to seated atoms now and the rest later
            self.g1 += 1 + int(self.src1[k])
            self.g2 += 1
            self.cross1 += bk[:, None]

        self.src1 += bk
        self.nb_u[:, pr.kb[k]] -= bk
        self.seat[k] = p
        self.allowed[k] = False
        self.src_ok[k] = False


def duals(cost, cols):
    """Reduced costs of an optimal assignment, from shortest paths in its residual graph. cols[r] is the column of row
    r, the result is cost - u - v >= 0 with zeros on the assignment."""
    n = len(cost)
    rows_of = np.empty(n, np.int64)
    rows_of[cols] = np.arange(n)

    # moving the row of column a to column b costs cost[row, b] - cost[row, a], no cycle is negative at an optimum
    w = cost[rows_of] - cost[rows_of, np.arange(n)][:, None]
    dist = np.zeros(n)
    for _ in range(n):
        new = np.minimum(dist, (dist[:, None] + w).min(0))
        if np.array_equal(new, dist):
            break

        dist = new

    u = cost[np.arange(n), cols] - dist[cols]

    return cost - u[:, None] - dist[None, :]


def matching_distance(pr, node, U, F):
    """A kept bond at a seat needs its two other ends to be an allowed pair, so the incident places that change are
    at least |N(i)| + |N(q)| - 2 m(i, q), m the largest matching of allowed pairs between the free neighbours of i in U
    and of q in F. Seats removed by reduced costs cannot occur in a mapping below the target, so the bound holds for
    every mapping the search still looks for. Neighbour lists are short, the matching is a maximum over permutations.
    """
    width = pr.nbr_a.shape[1]
    free_b = np.append(node.seat == -2, False)
    free_a = np.append(~node.used, False)
    nb = np.where(free_b[pr.nbr_b[U]], pr.nbr_b[U], pr.nb)
    na = np.where(free_a[pr.nbr_a[F]], pr.nbr_a[F], pr.na)
    allowed = np.zeros((pr.nb + 1, pr.na + 1), bool)
    allowed[: pr.nb, : pr.na] = node.allowed
    pair = allowed[nb[:, None, :, None], na[None, :, None, :]]
    best = np.zeros(pair.shape[:2], np.int64)
    for perm in pr.perms:
        best = np.maximum(best, pair[:, :, np.arange(width), perm].sum(-1))

    count_b = (nb < pr.nb).sum(1)
    count_a = (na < pr.na).sum(1)
    d = count_b[:, None] + count_a[None, :] - 2 * best

    # an atom with more neighbours than the list holds keeps the bound of the element counts
    wide = pr.wide_b[U][:, None] | pr.wide_a[F][None, :]

    return np.where(wide, 0, d)


def bound(pr, node, matching=True):
    """Lower bounds on both levels over every completion of the node, the assignment behind the first and the reduced
    costs of both, or None when no completion exists."""
    U = np.nonzero(node.seat == -2)[0]
    F = np.nonzero(~node.used)[0]
    src = U[node.src_ok[U]]
    n_cols = len(F) + len(src)
    if len(U) > n_cols:
        return None

    ok = node.allowed[np.ix_(U, F)]
    d = np.abs(node.nb_u[U][:, None, :] - node.na_f[F][None, :, :]).sum(-1)

    if matching and pr.perms is not None and len(U) and len(F):
        d = np.maximum(d, matching_distance(pr, node, U, F))

    c1 = np.full((n_cols, n_cols), 0, np.int64)
    c2 = np.zeros((n_cols, n_cols), np.int64)
    c1[: len(U), : len(F)] = np.where(
        ok, 2 * (pr.unary1[np.ix_(U, F)] + node.cross1[np.ix_(U, F)]) + d, BIG
    )
    c2[: len(U), : len(F)] = np.where(
        ok, pr.unary2[np.ix_(U, F)] + node.cross2[np.ix_(U, F)], BIG
    )

    # a leaving atom breaks its bonds to the seated atoms, counted on the rows that delete a column
    c1[len(U) :, : len(F)] = 2 * node.del1[F]

    if len(src):
        # one column per product atom that may enter from outside, one place, every bond formed, one token
        own = np.full((len(U), len(src)), BIG, np.int64)
        own2 = np.full((len(U), len(src)), BIG, np.int64)
        at = np.searchsorted(U, src)
        own[at, np.arange(len(src))] = 2 * (1 + node.src1[src]) + node.nb_u[src].sum(1)
        own2[at, np.arange(len(src))] = 1
        c1[: len(U), len(F) :] = own
        c2[: len(U), len(F) :] = own2

    r1, k1 = linear_sum_assignment(c1)
    v1 = int(c1[r1, k1].sum())
    if v1 >= BIG:
        return None

    r2, k2 = linear_sum_assignment(c2)
    v2 = int(c2[r2, k2].sum())
    rc1, rc2 = duals(c1.astype(np.float64), k1), duals(c2.astype(np.float64), k2)

    return {
        "U": U,
        "F": F,
        "src": src,
        "v1": v1,
        "v2": v2,
        "lb1": node.g1 + (v1 + 1) // 2,
        "lb2": node.g2 + v2,
        "cols": k1[: len(U)],
        "rc1": rc1[: len(U)],
        "rc2": rc2[: len(U)],
    }


class Search:
    """Depth first branch and bound for a mapping lexicographically below the incumbent."""

    def __init__(
        self,
        pr,
        incumbent=None,
        node_limit=20000,
        seconds=None,
        enumerate_ties=False,
        tie_limit=64,
        connected=True,
        deepen=True,
        matching=True,
        rounds=1,
        symmetry=True,
    ):
        self.pr = pr
        self.symmetry = symmetry
        self.matching, self.rounds = matching, rounds
        self.deepen, self.T, self.passes = deepen, None, 1

        if incumbent is None:
            self.best, self.best_levels = None, (np.inf, np.inf)
        else:
            self.best = np.asarray(incumbent, np.int64).copy()
            self.best_levels = pr.levels(self.best)

        self.nodes, self.node_limit = 0, node_limit
        self.deadline = None if seconds is None else time.time() + seconds
        self.complete = True
        self.root_bound = None
        self.ties, self.tie_limit, self.enumerate_ties = [], tie_limit, enumerate_ties
        self.connected = connected
        self.last_guess = None
        self.loose = False

    def target(self):
        """What a mapping has to beat, the incumbent or, while the first level is deepened, (T, infinity)."""
        if self.T is not None and self.best_levels[0] > self.T:
            return self.T, np.inf

        return self.best_levels

    def prunes(self, lb1, lb2):
        """True when no mapping below these bounds can beat the target (or tie it, when ties are listed)."""
        b1, b2 = self.target()

        if self.enumerate_ties:
            return lb1 > b1 or (lb1 == b1 and lb2 > b2)

        return lb1 > b1 or (lb1 == b1 and lb2 >= b2)

    def signature(self, mapping):
        """The seated precursor atoms and the bonds after the firing, equal for mappings that differ by a symmetry of
        the product, whose condensed graphs of reaction are then identical."""
        pr, on = self.pr, mapping >= 0
        after = np.zeros((pr.na, pr.na), np.int8)
        after[np.ix_(mapping[on], mapping[on])] = pr.ob2[np.ix_(on, on)]
        seated = np.zeros(pr.na, bool)
        seated[mapping[on]] = True

        return (
            seated.tobytes() + after.tobytes() + np.sort(np.nonzero(~on)[0]).tobytes()
        )

    def offer(self, mapping):
        cost = self.pr.levels(mapping)
        if cost < self.best_levels:
            self.best, self.best_levels = mapping.copy(), cost
            self.ties = [mapping.copy()]
            self.seen = {self.signature(mapping)} if self.enumerate_ties else set()
            self.leaves = 1
        elif self.enumerate_ties and cost == self.best_levels:
            self.leaves += 1
            key = self.signature(mapping)
            if key not in self.seen:
                self.seen.add(key)
                self.ties.append(mapping.copy())

    def run(self, root=None):
        """Iterative deepening on the first level. With a target (T, infinity) every node whose first bound exceeds T is
        cut and every seat whose forced bound exceeds T is removed. A finished pass that finds no mapping with T places
        proves the minimum above T, so T grows by one, and the first T that is reached is the minimum.
        """
        root = root if root is not None else Node.root(self.pr)
        self.seen, self.leaves = set(), 0

        if self.enumerate_ties and self.best is not None:
            self.ties = [self.best.copy()]
            self.seen = {self.signature(self.best)}

        b = bound(self.pr, root.copy(), self.matching)
        if b is None:
            self.best_levels = (np.inf, np.inf)
            return self

        self.T = b["lb1"] if self.deepen else None

        while True:
            self.loose = False
            self.visit(root.copy(), depth=0)

            if not self.complete or self.T is None or self.best_levels[0] <= self.T:
                break

            self.T += 1
            self.passes += 1

        # a subtree cut by a symmetry of the first level alone may hold a better second level or another tie, so the
        # minimum of the first level is searched once more with the incumbent as target and the full symmetry
        if self.complete and self.loose and self.best is not None:
            self.T = None
            self.passes += 1
            self.visit(root.copy(), depth=0)

        return self

    def out_of_budget(self):
        if self.nodes >= self.node_limit or (
            self.deadline is not None and time.time() > self.deadline
        ):
            self.complete = False
            return True

        return False

    def visit(self, node, depth):
        pr = self.pr
        rounds = 0

        while True:
            if self.out_of_budget():
                return

            self.nodes += 1
            b = bound(pr, node, self.matching)
            if b is None:
                return

            if self.root_bound is None:
                self.root_bound = (b["lb1"], b["lb2"])

            # the assignment itself completes the seating, a free primal heuristic
            U, F, src = b["U"], b["F"], b["src"]
            if len(U):
                guess = node.seat.copy()
                cols = b["cols"]
                inside = cols < len(F)
                guess[U[inside]] = F[cols[inside]]
                guess[U[~inside]] = -1
                key = guess.tobytes()
                if key != self.last_guess:
                    self.last_guess = key
                    self.offer(guess)
            else:
                self.offer(node.seat)
                return

            if self.prunes(b["lb1"], b["lb2"]):
                return

            # the limit counts optima only, not the placeholders of a pass that has not reached its target yet
            reached = self.T is None or self.best_levels[0] <= self.T
            if self.enumerate_ties and reached and len(self.ties) >= self.tie_limit:
                self.complete = False
                return

            # reduced cost fixing, a seat whose forced bound cannot beat the target leaves the subtree
            b1, b2 = self.target()
            lb1 = (
                node.g1 + (b["v1"] + np.ceil(b["rc1"] - 1e-9).astype(np.int64) + 1) // 2
            )
            lb2 = node.g2 + b["v2"] + np.ceil(b["rc2"] - 1e-9).astype(np.int64)

            if self.enumerate_ties:
                drop = (lb1 > b1) | ((lb1 == b1) & (lb2 > b2))
            else:
                drop = (lb1 > b1) | ((lb1 == b1) & (lb2 >= b2))

            before = int(node.allowed.sum()) + int(node.src_ok.sum())
            node.allowed[np.ix_(U, F)] &= ~drop[:, : len(F)]

            if len(src):
                at = np.searchsorted(U, src)
                node.src_ok[src] &= ~drop[at, len(F) + np.arange(len(src))]

            size = node.allowed[U].sum(1) + node.src_ok[U]
            if (size == 0).any():
                return

            singles = U[size == 1]
            if not len(singles):
                # fewer allowed pairs tighten the matching bound, so the node is bounded again until nothing drops
                removed = before - int(node.allowed.sum()) - int(node.src_ok.sum())
                if self.matching and removed and rounds < self.rounds:
                    rounds += 1
                    continue

                break

            # seat every atom that has one seat left, two of them on the same seat end the node
            for k in singles:
                if node.src_ok[k]:
                    node.place(pr, k, -1)
                    continue

                p = np.nonzero(node.allowed[k])[0]
                if not len(p):
                    return

                node.place(pr, int(k), int(p[0]))

        # branch next to the seated atoms, where the places to them tell the seats apart, fewest seats first
        seated_nb = (
            (pr.pb[np.ix_(U, np.nonzero(node.seat >= 0)[0])]).sum(1)
            if (node.seat >= 0).any()
            else np.zeros(len(U))
        )
        order = (
            np.lexsort((-seated_nb, size, seated_nb == 0))
            if self.connected
            else np.lexsort((-seated_nb, size))
        )
        i = int(U[order[0]])
        row = int(np.searchsorted(U, i))
        choices = [
            (float(b["rc1"][row, c]), float(b["rc2"][row, c]), int(F[c]))
            for c in range(len(F))
            if node.allowed[i, F[c]]
        ]

        if node.src_ok[i]:
            c = len(F) + int(np.searchsorted(src, i))
            choices.append((float(b["rc1"][row, c]), float(b["rc2"][row, c]), -1))

        # automorphisms of the precursors that fix every used seat map the subtree of a tried seat onto others, worth
        # computing only when the search is not tiny
        # while a pass only asks for T places the second level is free and the symmetries of the first level serve
        loose = self.symmetry and self.target()[1] == np.inf
        twin = pr.twin1 if loose else pr.twin
        wanted = self.symmetry and len(choices) > 1 and self.nodes >= SYMMETRY_AFTER
        auts = (
            (pr.automorphisms1 if loose else pr.automorphisms)
            if wanted
            else np.zeros((0, pr.na), np.int64)
        )
        if len(auts):
            used = np.nonzero(node.used)[0]
            auts = auts[(auts[:, used] == used).all(1)]

        tried = []
        for _, _, p in sorted(choices):
            if p >= 0 and any(twin[p, t] for t in tried):
                self.loose |= loose
                continue

            if p >= 0 and tried and len(auts) and (auts[:, tried] == p).any():
                self.loose |= loose
                continue

            # the incumbent may have improved in an earlier sibling, the forced bound of this seat is checked again
            c = (
                int(np.searchsorted(F, p))
                if p >= 0
                else len(F) + int(np.searchsorted(src, i))
            )
            if self.prunes(int(lb1[row, c]), int(lb2[row, c])) or self.prunes(
                b["lb1"], b["lb2"]
            ):
                continue

            child = node.copy()
            child.place(pr, i, p)
            self.visit(child, depth + 1)

            if not self.complete:
                return

            if p >= 0:
                tried.append(p)
