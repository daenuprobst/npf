"""Petri nets with attributes and random nets for the synthetic benchmarks."""
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import null_space


@dataclass(eq=False)
class Net:
    n_places: int
    n_trans: int

    # arcs from places to transitions as index arrays with weights Pre(p, t)
    pre_p: np.ndarray
    pre_t: np.ndarray
    pre_w: np.ndarray

    # arcs from transitions to places with weights Pos(p, t)
    pos_p: np.ndarray
    pos_t: np.ndarray
    pos_w: np.ndarray

    # static attributes of places [P, 1] and of transitions [T, 1]
    e: np.ndarray
    a: np.ndarray
    Pre: np.ndarray = field(init=False)
    Pos: np.ndarray = field(init=False)
    C: np.ndarray = field(init=False)
    C_pinv: np.ndarray = field(init=False)
    X: np.ndarray = field(init=False)
    n_tinv: int = field(init=False)
    edges: tuple = field(init=False)

    def __post_init__(self):
        self.Pre = np.zeros((self.n_places, self.n_trans))
        self.Pos = np.zeros((self.n_places, self.n_trans))
        self.Pre[self.pre_p, self.pre_t] = self.pre_w
        self.Pos[self.pos_p, self.pos_t] = self.pos_w

        # incidence matrix of the state equation m' = m + C sigma
        self.C = self.Pos - self.Pre
        self.C_pinv = np.linalg.pinv(self.C)

        # P-invariants are ker C^T and T-invariants are ker C
        self.X = null_space(self.C.T)
        self.n_tinv = self.n_trans - np.linalg.matrix_rank(self.C)

        # place graph for the GNN baseline with one edge per input and output place of a transition
        i, j = np.nonzero(self.pre_t[:, None] == self.pos_t[None, :])
        self.edges = (self.pre_p[i], self.pos_p[j], self.pre_t[i], self.pre_w[i], self.pos_w[j])


def random_net(rng, n_places, n_base, max_arity, p_reverse=0.5):
    """Random net in which every place lies on a transition. max_arity 1 gives a directed graph."""
    pre, pos = [], []

    # unused places are handed out first so that every place takes part in the dynamics
    unused = list(rng.permutation(n_places))
    t = 0
    for _ in range(n_base):
        n_in, n_out = rng.integers(1, max_arity + 1, size=2)
        nodes, unused = unused[:n_in + n_out], unused[n_in + n_out:]
        others = [p for p in rng.permutation(n_places) if p not in nodes]
        nodes = list(rng.permutation(nodes + others[:n_in + n_out - len(nodes)]))
        w = rng.integers(1, 3, size=n_in + n_out) if max_arity > 1 else np.ones(n_in + n_out, int)
        ins, outs = list(zip(nodes[:n_in], w[:n_in])), list(zip(nodes[n_in:], w[n_in:]))
        directions = [(ins, outs)] + ([(outs, ins)] if rng.random() < p_reverse else [])
        for src, dst in directions:
            pre += [(p, t, wt) for p, wt in src]
            pos += [(p, t, wt) for p, wt in dst]
            t += 1

    pre, pos = np.array(pre), np.array(pos)
    assert len(set(pre[:, 0]) | set(pos[:, 0])) == n_places

    return Net(
        n_places, t,
        pre[:, 0], pre[:, 1], pre[:, 2].astype(float),
        pos[:, 0], pos[:, 1], pos[:, 2].astype(float),
        e=rng.uniform(-0.7, 0.7, size=(n_places, 1)),
        a=rng.uniform(-1.0, 1.0, size=(t, 1)),
    )


def chain_net(rng, n_places):
    """Path net p_0 -> t_1 -> p_1 ... without T-invariants."""
    k = np.arange(n_places - 1)

    return Net(n_places, n_places - 1, k, k, np.ones(n_places - 1), k + 1, k, np.ones(n_places - 1),
               e=rng.uniform(-0.7, 0.7, size=(n_places, 1)), a=rng.uniform(-1.0, 1.0, size=(n_places - 1, 1)))
