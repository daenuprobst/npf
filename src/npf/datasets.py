"""Synthetic datasets. A sample is a pair of markings of one run or one step of a fluid net."""
from dataclasses import dataclass

import numpy as np

from .nets import Net, random_net
from .simulate import gillespie_pairs, simulate


@dataclass
class Group:
    """All samples that live on one net."""
    net: Net

    # marking A or the current marking, [S, P]
    m: np.ndarray

    # marking B, [S, P]
    m_b: np.ndarray = None

    # time between the two markings, [S]
    dt: np.ndarray = None

    # firing counts between A and B, the target of the inverse task, [S, T]
    sigma: np.ndarray = None

    # next marking, the target of the forward task, [S, P]
    y: np.ndarray = None

    # full trajectories [R, L + 1, P] and true firing amounts per step [R, L, T] for rollouts
    states: np.ndarray = None
    fired: np.ndarray = None


def _nets(rng, n_nets, places, max_arity):
    for _ in range(n_nets):
        n_places = int(rng.integers(places[0], places[1] + 1))
        yield random_net(rng, n_places, n_base=max(2, int(round(0.6 * n_places))), max_arity=max_arity)


def _initial(rng, n_samples, n_places, high, integer):
    draw = rng.integers(0, high + 1, size=(n_samples, n_places)) if integer else rng.uniform(0, high, size=(n_samples, n_places))

    return draw * (rng.random((n_samples, n_places)) > 0.25)


def make_pairs(seed, n_nets, places, max_arity, n_samples, tokens=8, gap=(0.2, 1.0)):
    """Two markings of a stochastic run at two random times and the firing counts in between."""
    rng = np.random.default_rng(seed)
    groups = []
    for net in _nets(rng, n_nets, places, max_arity):
        M0 = _initial(rng, n_samples, net.n_places, tokens, integer=True)
        t_a = rng.uniform(0.0, 0.5, n_samples)
        dt = rng.uniform(*gap, n_samples)
        MA, MB, sigma = gillespie_pairs(net, M0, t_a, t_a + dt, rng)
        groups.append(Group(net, MA, MB, dt, sigma))

    return groups


def make_flow_pairs(seed, n_nets, places, max_arity, n_samples, kind="sat", scale=4.0, gap=(1, 4), dt=0.25, n_steps=8):
    """Deterministic version of make_pairs on a fluid net, the two markings are gap observation steps apart."""
    rng = np.random.default_rng(seed)
    groups, rows = [], np.arange(n_samples)
    for net in _nets(rng, n_nets, places, max_arity):
        M0 = _initial(rng, n_samples, net.n_places, scale, integer=False)
        states, fired = simulate(net, M0, kind, n_steps, dt=dt, substeps=25)
        total = np.concatenate([np.zeros_like(fired[:, :1]), fired.cumsum(1)], 1)
        n = rng.integers(gap[0], gap[1] + 1, n_samples)
        i = rng.integers(0, n_steps - n + 1)
        groups.append(Group(net, states[rows, i], states[rows, i + n], n * dt, total[rows, i + n] - total[rows, i]))

    return groups


def make_flows(seed, n_nets, places, max_arity, n_samples, kind, n_steps, scale=4.0):
    """Trajectories of a fluid net. Every (trajectory, step) is one training pair."""
    rng = np.random.default_rng(seed)
    groups = []
    for net in _nets(rng, n_nets, places, max_arity):
        M0 = _initial(rng, n_samples, net.n_places, scale, integer=False)
        states, fired = simulate(net, M0, kind, n_steps)
        P = net.n_places
        groups.append(Group(net, states[:, :-1].reshape(-1, P), y=states[:, 1:].reshape(-1, P), states=states, fired=fired))

    return groups
