"""Synthetic dynamic attributed graphs / Petri nets.

The dynamic graph is a random Petri net whose node attributes evolve by the stochastic token game
(Gillespie simulation of a stochastic Petri net). Places carry a dynamic attribute (integer marking
m_p) and a static one (e_p); transitions carry a static attribute (a_t). `max_arity=1` gives an
ordinary directed graph (every transition moves a token along one edge u -> v); `max_arity>1` gives
genuine multi-body Petri transitions with arc weights.

Main task ("transitions"): sample two states A, B of the same run and predict how often every
transition fired in between (the firing-count / Parikh vector sigma, with M_B = M_A + C sigma).
Secondary task ("next"): continuous token flow, predict the state one step ahead.
"""
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.linalg import null_space

VOLUME = 4.0  # system size used to scale multi-token propensities


@dataclass(eq=False)
class Net:
    n_places: int
    n_trans: int
    pre_p: np.ndarray  # arcs place -> transition: place index, transition index, weight Pre(p, t)
    pre_t: np.ndarray
    pre_w: np.ndarray
    pos_p: np.ndarray  # arcs transition -> place: place index, transition index, weight Pos(p, t)
    pos_t: np.ndarray
    pos_w: np.ndarray
    e: np.ndarray  # [P, 1] static place attribute (log half-saturation / log activity scale)
    a: np.ndarray  # [T, 1] static transition attribute (log rate constant)
    Pre: np.ndarray = field(init=False)
    Pos: np.ndarray = field(init=False)
    C: np.ndarray = field(init=False)  # incidence matrix Pos - Pre, [P, T]
    C_pinv: np.ndarray = field(init=False)
    X: np.ndarray = field(init=False)  # [P, r] orthonormal basis of ker C^T (P-invariants)
    n_tinv: int = field(init=False)  # dim ker C (independent T-invariants)
    edges: tuple = field(init=False)  # place graph: one edge per (input place, output place) of a transition

    def __post_init__(self):
        self.Pre = np.zeros((self.n_places, self.n_trans))
        self.Pos = np.zeros((self.n_places, self.n_trans))
        self.Pre[self.pre_p, self.pre_t] = self.pre_w
        self.Pos[self.pos_p, self.pos_t] = self.pos_w
        self.C = self.Pos - self.Pre
        self.C_pinv = np.linalg.pinv(self.C)
        self.X = null_space(self.C.T)
        self.n_tinv = self.n_trans - np.linalg.matrix_rank(self.C)
        i, j = np.nonzero(self.pre_t[:, None] == self.pos_t[None, :])
        self.edges = (self.pre_p[i], self.pos_p[j], self.pre_t[i], self.pre_w[i], self.pos_w[j])


def random_net(rng, n_places, n_base, max_arity, p_reverse=0.5):
    pre, pos = [], []
    unused = list(rng.permutation(n_places))  # handed out first, so every place takes part in the dynamics
    t = 0
    for _ in range(n_base):
        n_in, n_out = rng.integers(1, max_arity + 1, size=2)
        nodes, unused = unused[:n_in + n_out], unused[n_in + n_out:]
        others = [p for p in rng.permutation(n_places) if p not in nodes]
        nodes = list(rng.permutation(nodes + others[:n_in + n_out - len(nodes)]))
        w = rng.integers(1, 3, size=n_in + n_out) if max_arity > 1 else np.ones(n_in + n_out, int)
        ins, outs = list(zip(nodes[:n_in], w[:n_in])), list(zip(nodes[n_in:], w[n_in:]))
        for src, dst in [(ins, outs)] + ([(outs, ins)] if rng.random() < p_reverse else []):
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


# ------------------------------------------------------------------ ground-truth dynamics (hidden from all models)

def propensity(net, M):
    """Stochastic mass action: lambda_t = k_t V prod_p falling(m_p, w) / (V K_p)^w; zero unless t is enabled."""
    K, k = np.exp(net.e[:, 0]), np.exp(net.a[:, 0])
    w = net.Pre[None]  # [1, P, T]
    Mx = M[:, :, None]
    falling = np.where(w == 2, Mx * (Mx - 1), np.where(w == 1, Mx, 1.0))
    enabled = np.where(w > 0, Mx >= w - 1e-9, True).all(1)  # only matters for non-integer markings (real arc weights)
    return k * VOLUME * np.prod(falling / (VOLUME * K[None, :, None]) ** w, axis=1) * enabled


def gillespie_pairs(net, M0, t_a, t_b, rng):
    """Run S independent token games; return the states at times t_a < t_b and the firing counts in between."""
    S = M0.shape[0]
    M, t = M0.astype(float).copy(), np.zeros(S)
    MA, got_a = M.copy(), np.zeros(S, bool)
    sigma, done = np.zeros((S, net.n_trans)), np.zeros(S, bool)
    while not done.all():
        lam = propensity(net, M)
        total = lam.sum(1)
        t_next = t + rng.exponential(1.0, S) / np.maximum(total, 1e-300)
        t_next[total <= 0] = np.inf  # dead marking: nothing can fire any more
        snap = ~got_a & (t_next > t_a)
        MA[snap], got_a[snap] = M[snap], True
        fire = ~done & (t_next <= t_b)
        done |= ~fire
        if fire.any():
            idx = np.nonzero(fire)[0]
            cum = np.cumsum(lam[idx], 1)
            j = (cum < (rng.random(len(idx)) * total[idx])[:, None]).sum(1).clip(max=net.n_trans - 1)
            M[idx] += net.C[:, j].T
            counted = t_next[idx] > t_a[idx]
            np.add.at(sigma, (idx[counted], j[counted]), 1.0)
            t[idx] = t_next[idx]
    return MA, M, sigma


def flux(net, M, kind):
    """Continuous firing rates for the secondary task, [S, P] -> [S, T].
    sat: v_t = k_t prod_p (m_p / (K_p + m_p))^Pre(p,t)      min: v_t = k_t min_p m_p / (K_p Pre(p,t))"""
    K, k = np.exp(net.e[:, 0]), np.exp(net.a[:, 0])
    mask = net.Pre > 0
    if kind == "sat":
        logu = np.log(np.maximum(M / (K + M), 1e-300))
        return k * np.exp(np.where(mask, logu[:, :, None] * net.Pre, 0.0).sum(1))
    ratio = np.where(mask, (M / K)[:, :, None] / np.where(mask, net.Pre, 1.0), np.inf)
    return k * ratio.min(1)


def simulate(net, M0, kind, n_steps, dt=0.5, substeps=50):
    """RK4 integration of dm/dt = C v(m). Returns states [S, n_steps + 1, P] and the firing amount of
    every transition within each step [S, n_steps, T]."""
    v = lambda M: flux(net, np.maximum(M, 0.0), kind)
    h = dt / substeps
    states, fired, M = [M0], [], M0
    for _ in range(n_steps):
        F = np.zeros((M0.shape[0], net.n_trans))
        for _ in range(substeps):
            v1 = v(M); v2 = v(M + h / 2 * v1 @ net.C.T); v3 = v(M + h / 2 * v2 @ net.C.T); v4 = v(M + h * v3 @ net.C.T)
            dF = h / 6 * (v1 + 2 * v2 + 2 * v3 + v4)
            M, F = np.maximum(M + dF @ net.C.T, 0.0), F + dF
        states.append(M); fired.append(F)
    return np.stack(states, 1), np.stack(fired, 1)


# ------------------------------------------------------------------ datasets

@dataclass
class Group:
    """All samples that live on one net."""
    net: Net
    m: np.ndarray  # [S, P] state A (or current state)
    m_b: np.ndarray = None  # [S, P] state B
    dt: np.ndarray = None  # [S] time between the two states
    sigma: np.ndarray = None  # [S, T] firing counts between A and B (target of the main task)
    y: np.ndarray = None  # [S, P] next state (target of the secondary task)
    states: np.ndarray = None  # [R, L + 1, P] full trajectories (secondary task, rollouts)
    fired: np.ndarray = None  # [R, L, T] true firing amounts per step (never shown to a model)


def _nets(rng, n_nets, places, max_arity):
    for _ in range(n_nets):
        n_places = int(rng.integers(places[0], places[1] + 1))
        yield random_net(rng, n_places, n_base=max(2, int(round(0.6 * n_places))), max_arity=max_arity)


def make_pairs(seed, n_nets, places, max_arity, n_samples, tokens=8, gap=(0.2, 1.0)):
    rng = np.random.default_rng(seed)
    groups = []
    for net in _nets(rng, n_nets, places, max_arity):
        M0 = rng.integers(0, tokens + 1, size=(n_samples, net.n_places)) * (rng.random((n_samples, net.n_places)) > 0.25)
        t_a = rng.uniform(0.0, 0.5, n_samples)
        dt = rng.uniform(*gap, n_samples)
        MA, MB, sigma = gillespie_pairs(net, M0, t_a, t_a + dt, rng)
        groups.append(Group(net, MA, MB, dt, sigma))
    return groups


def make_flow_pairs(seed, n_nets, places, max_arity, n_samples, kind="sat", scale=4.0, gap=(1, 4), dt=0.25, n_steps=8):
    """Deterministic version of `make_pairs`: two states of a continuous token flow, `gap` observation steps apart."""
    rng = np.random.default_rng(seed)
    groups, rows = [], np.arange(n_samples)
    for net in _nets(rng, n_nets, places, max_arity):
        M0 = rng.uniform(0, scale, size=(n_samples, net.n_places)) * (rng.random((n_samples, net.n_places)) > 0.25)
        states, fired = simulate(net, M0, kind, n_steps, dt=dt, substeps=25)
        total = np.concatenate([np.zeros_like(fired[:, :1]), fired.cumsum(1)], 1)
        n = rng.integers(gap[0], gap[1] + 1, n_samples)
        i = rng.integers(0, n_steps - n + 1)
        groups.append(Group(net, states[rows, i], states[rows, i + n], n * dt, total[rows, i + n] - total[rows, i]))
    return groups


def make_flows(seed, n_nets, places, max_arity, n_samples, kind, n_steps, scale=4.0):
    rng = np.random.default_rng(seed)
    groups = []
    for net in _nets(rng, n_nets, places, max_arity):
        M0 = rng.uniform(0, scale, size=(n_samples, net.n_places)) * (rng.random((n_samples, net.n_places)) > 0.25)
        states, fired = simulate(net, M0, kind, n_steps)
        P = net.n_places  # every (trajectory, step) is one training pair
        groups.append(Group(net, states[:, :-1].reshape(-1, P), y=states[:, 1:].reshape(-1, P), states=states, fired=fired))
    return groups


# ------------------------------------------------------------------ batching

@dataclass
class Batch:
    """Disjoint union of nets, one copy of the net per state sample (the usual GNN batching)."""
    m: torch.Tensor  # [NP] marking of state A
    m_b: torch.Tensor  # [NP] marking of state B (zeros for the secondary task)
    e: torch.Tensor  # [NP, 1]
    a: torch.Tensor  # [NT, 1]
    dt_p: torch.Tensor  # [NP] time between the states, broadcast to places / transitions
    dt_t: torch.Tensor  # [NT]
    pre_p: torch.Tensor; pre_t: torch.Tensor; pre_w: torch.Tensor
    pos_p: torch.Tensor; pos_t: torch.Tensor; pos_w: torch.Tensor
    edge_u: torch.Tensor; edge_v: torch.Tensor; edge_t: torch.Tensor; edge_wu: torch.Tensor; edge_wv: torch.Tensor
    n_places: int
    n_trans: int
    # dense per-net incidence matrices, zero padded, for the state-equation projection
    C: torch.Tensor  # [G, Pmax, Tmax]
    C_pinv: torch.Tensor  # [G, Tmax, Pmax]
    pad_p: torch.Tensor  # [NP] position of every place in the padded [G, S, Pmax] layout
    pad_t: torch.Tensor  # [NT]
    pad_shape: tuple  # (G, S, Pmax, Tmax)
    X: torch.Tensor = None  # [G, Pmax, rmax] orthonormal bases of the P-invariants (ker C^T), zero padded


def collate(groups, rows, device):
    """groups: list of Group; rows: list of index arrays selecting S samples per group (same S everywhere)."""
    S = len(rows[0])
    G, Pmax, Tmax = len(groups), max(g.net.n_places for g in groups), max(g.net.n_trans for g in groups)
    cols = {k: [] for k in ("m", "m_b", "e", "a", "dt_p", "dt_t", "pre_p", "pre_t", "pre_w", "pos_p", "pos_t", "pos_w",
                            "edge_u", "edge_v", "edge_t", "edge_wu", "edge_wv", "pad_p", "pad_t")}
    C, C_pinv = np.zeros((G, Pmax, Tmax)), np.zeros((G, Tmax, Pmax))
    X = np.zeros((G, Pmax, max(1, max(g.net.X.shape[1] for g in groups))))
    off_p = off_t = 0
    for gi, (g, r) in enumerate(zip(groups, rows)):
        net = g.net
        P, T = net.n_places, net.n_trans
        sp, st = off_p + P * np.arange(S)[:, None], off_t + T * np.arange(S)[:, None]
        dt = g.dt[r] if g.dt is not None else np.zeros(S)
        cols["m"].append(g.m[r].ravel())
        cols["m_b"].append(g.m_b[r].ravel() if g.m_b is not None else np.zeros(S * P))
        cols["e"].append(np.tile(net.e, (S, 1))); cols["a"].append(np.tile(net.a, (S, 1)))
        cols["dt_p"].append(np.repeat(dt, P)); cols["dt_t"].append(np.repeat(dt, T))
        cols["pre_p"].append((net.pre_p[None] + sp).ravel()); cols["pre_t"].append((net.pre_t[None] + st).ravel())
        cols["pos_p"].append((net.pos_p[None] + sp).ravel()); cols["pos_t"].append((net.pos_t[None] + st).ravel())
        cols["pre_w"].append(np.tile(net.pre_w, S)); cols["pos_w"].append(np.tile(net.pos_w, S))
        eu, ev, et, wu, wv = net.edges
        cols["edge_u"].append((eu[None] + sp).ravel()); cols["edge_v"].append((ev[None] + sp).ravel())
        cols["edge_t"].append((et[None] + st).ravel())
        cols["edge_wu"].append(np.tile(wu, S)); cols["edge_wv"].append(np.tile(wv, S))
        cols["pad_p"].append(((gi * S + np.arange(S))[:, None] * Pmax + np.arange(P)[None]).ravel())
        cols["pad_t"].append(((gi * S + np.arange(S))[:, None] * Tmax + np.arange(T)[None]).ravel())
        C[gi, :P, :T], C_pinv[gi, :T, :P] = net.C, net.C_pinv
        X[gi, :P, :net.X.shape[1]] = net.X
        off_p += S * P; off_t += S * T
    is_index = lambda k: k.startswith("pad_") or k in ("pre_p", "pre_t", "pos_p", "pos_t", "edge_u", "edge_v", "edge_t")
    tensors = {
        k: torch.as_tensor(np.concatenate(v), dtype=torch.long if is_index(k) else torch.float32, device=device)
        for k, v in cols.items()
    }
    return Batch(
        **tensors, n_places=off_p, n_trans=off_t,
        C=torch.as_tensor(C, dtype=torch.float32, device=device),
        C_pinv=torch.as_tensor(C_pinv, dtype=torch.float32, device=device),
        pad_shape=(G, S, Pmax, Tmax), X=torch.as_tensor(X, dtype=torch.float64, device=device),
    )


def flat_targets(groups, rows, key, device):
    return torch.as_tensor(np.concatenate([getattr(g, key)[r].ravel() for g, r in zip(groups, rows)]),
                           dtype=torch.float32, device=device)
