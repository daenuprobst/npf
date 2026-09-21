"""Seeded synthetic data and batching are bit identical in the old and the new package."""
import dataclasses

import numpy as np
import pytest
import torch
from conftest import assert_same_arrays, assert_same_tensors

from npf import batching, datasets, nets, simulate

NET_FIELDS = ("n_places", "n_trans", "pre_p", "pre_t", "pre_w", "pos_p", "pos_t", "pos_w", "e", "a", "Pre", "Pos", "C", "C_pinv", "X", "n_tinv")
GROUP_FIELDS = ("m", "m_b", "dt", "sigma", "y", "states", "fired")


def assert_same_net(a, b):
    assert [f.name for f in dataclasses.fields(a)] == [f.name for f in dataclasses.fields(b)]

    for name in NET_FIELDS:
        x, y = getattr(a, name), getattr(b, name)
        if isinstance(x, (int, np.integer)):
            assert x == y, name
        else:
            assert_same_arrays(x, y, f"net.{name}")

    assert len(a.edges) == len(b.edges) == 5

    for k, (x, y) in enumerate(zip(a.edges, b.edges)):
        assert_same_arrays(x, y, f"net.edges[{k}]")


def assert_same_groups(old_groups, new_groups):
    assert len(old_groups) == len(new_groups)

    for g, h in zip(old_groups, new_groups):
        assert [f.name for f in dataclasses.fields(g)] == [f.name for f in dataclasses.fields(h)]
        assert_same_net(g.net, h.net)

        for name in GROUP_FIELDS:
            x, y = getattr(g, name), getattr(h, name)
            assert (x is None) == (y is None), name

            if x is not None:
                assert_same_arrays(x, y, f"group.{name}")


@pytest.mark.parametrize("seed", [0, 1, 7, 123])
@pytest.mark.parametrize("max_arity", [1, 2, 3])
@pytest.mark.parametrize("p_reverse", [0.5, 1.0])
def test_random_net(old, seed, max_arity, p_reverse):
    data = old("data")
    for n_places in (6, 9, 25):
        r_old, r_new = np.random.default_rng(seed), np.random.default_rng(seed)
        n_base = max(2, int(round(0.6 * n_places)))
        a = data.random_net(r_old, n_places, n_base, max_arity, p_reverse)
        b = nets.random_net(r_new, n_places, n_base, max_arity, p_reverse)
        assert_same_net(a, b)

        # the generators must be left in the same state
        assert r_old.bit_generator.state == r_new.bit_generator.state


@pytest.mark.parametrize("seed", [0, 3])
@pytest.mark.parametrize("n_places", [2, 4, 13, 49])
def test_chain_net(old, seed, n_places):
    r_old, r_new = np.random.default_rng(seed), np.random.default_rng(seed)
    a, b = old("locality").chain(r_old, n_places), nets.chain_net(r_new, n_places)
    assert_same_net(a, b)
    assert r_old.bit_generator.state == r_new.bit_generator.state
    assert b.n_tinv == 0


def test_chains_of_the_locality_benchmark(old):
    from benchmarks.synthetic import locality
    assert locality.LENGTHS == old("locality").LENGTHS

    for length in (3, 12):
        assert_same_groups(old("locality").chains(length, n_nets=3, n_samples=16), locality.chains(length, n_nets=3, n_samples=16))


@pytest.mark.parametrize("seed", [1, 2, 5])
@pytest.mark.parametrize("max_arity", [1, 2])
@pytest.mark.parametrize("kw", [{}, {"tokens": 24}, {"gap": (1.0, 2.0)}])
def test_make_pairs(old, seed, max_arity, kw):
    assert_same_groups(old("data").make_pairs(seed, 6, (6, 12), max_arity, 32, **kw), datasets.make_pairs(seed, 6, (6, 12), max_arity, 32, **kw))


@pytest.mark.parametrize("seed", [1, 4])
@pytest.mark.parametrize("kw", [{}, {"scale": 12.0}, {"gap": (5, 8)}, {"kind": "min"}])
def test_make_flow_pairs(old, seed, kw):
    assert_same_groups(old("data").make_flow_pairs(seed, 4, (6, 12), 2, 16, **kw), datasets.make_flow_pairs(seed, 4, (6, 12), 2, 16, **kw))


@pytest.mark.parametrize("seed", [1, 3])
@pytest.mark.parametrize("kind", ["sat", "min"])
@pytest.mark.parametrize("max_arity", [1, 2])
def test_make_flows(old, seed, kind, max_arity):
    assert_same_groups(old("data").make_flows(seed, 4, (6, 12), max_arity, 4, kind, 8), datasets.make_flows(seed, 4, (6, 12), max_arity, 4, kind, 8))
    assert_same_groups(old("data").make_flows(seed, 2, (20, 30), max_arity, 2, kind, 5, scale=12.0),
                       datasets.make_flows(seed, 2, (20, 30), max_arity, 2, kind, 5, scale=12.0))


def test_simulation_functions(old):
    data = old("data")
    assert simulate.VOLUME == data.VOLUME
    rng = np.random.default_rng(11)
    net_old, net_new = data.random_net(np.random.default_rng(5), 9, 5, 2), nets.random_net(np.random.default_rng(5), 9, 5, 2)
    M = rng.integers(0, 9, size=(16, 9)).astype(float)
    assert_same_arrays(data.propensity(net_old, M), simulate.propensity(net_new, M), "propensity")

    for kind in ("sat", "min"):
        assert_same_arrays(data.flux(net_old, M, kind), simulate.flux(net_new, M, kind), f"flux {kind}")

        for x, y in zip(data.simulate(net_old, M, kind, 3), simulate.simulate(net_new, M, kind, 3)):
            assert_same_arrays(x, y, f"simulate {kind}")

    t_a = rng.uniform(0.0, 0.5, 16)
    out_old = data.gillespie_pairs(net_old, M, t_a, t_a + 0.7, np.random.default_rng(2))
    out_new = simulate.gillespie_pairs(net_new, M, t_a, t_a + 0.7, np.random.default_rng(2))
    for x, y in zip(out_old, out_new):
        assert_same_arrays(x, y, "gillespie_pairs")


def test_splits_of_the_experiment(old, monkeypatch):
    """make_splits with the sizes of the real runs scaled down. Same seeds, same keyword arguments, same order."""
    from benchmarks.synthetic import experiment as new_exp
    old_exp = old("experiment")
    assert new_exp.REGIMES == old_exp.REGIMES and (new_exp.SMALL, new_exp.LARGE) == (old_exp.SMALL, old_exp.LARGE)
    calls = {"old": [], "new": []}

    def record(name, fn):
        def wrapped(seed, n_nets, *args, **kw):
            calls[name].append((fn.__name__, seed, n_nets, args, tuple(sorted(kw.items()))))

            return fn(seed, 2, *args, **kw)

        return wrapped

    for fn in ("make_pairs", "make_flow_pairs", "make_flows"):
        monkeypatch.setattr(old("data"), fn, record("old", getattr(old("data"), fn)))
        monkeypatch.setattr(datasets, fn, record("new", getattr(datasets, fn)))

    for task, regimes in new_exp.REGIMES.items():
        for regime in regimes:
            a, b = old_exp.make_splits(task, regime), new_exp.make_splits(task, regime)
            assert list(a) == list(b)

            for split in a:
                assert_same_groups(a[split], b[split])

    assert calls["old"] == calls["new"] and len(calls["old"]) == 3 * 6 + 3 * 5


def _rows(groups, rng, n):
    return [rng.choice(len(g.m), n, replace=False) for g in groups]


@pytest.mark.parametrize("maker", ["pairs", "flow_pairs", "flows"])
def test_collate(old, maker):
    data = old("data")
    args = {"pairs": (3, 5, (6, 12), 2, 24), "flow_pairs": (3, 5, (6, 12), 2, 24), "flows": (3, 5, (6, 12), 2, 4, "sat", 8)}[maker]
    g_old, g_new = getattr(data, f"make_{maker}")(*args), getattr(datasets, f"make_{maker}")(*args)
    rows = _rows(g_old, np.random.default_rng(0), 8)
    b_old, b_new = data.collate(g_old, rows, "cpu"), batching.collate(g_new, rows, "cpu")
    names = [f.name for f in dataclasses.fields(b_old)]
    assert names == [f.name for f in dataclasses.fields(b_new)]

    for name in names:
        x, y = getattr(b_old, name), getattr(b_new, name)
        if torch.is_tensor(x):
            assert_same_tensors(x, y, f"batch.{name}")
        else:
            assert x == y, name

    key = "y" if maker == "flows" else "sigma"
    assert_same_tensors(data.flat_targets(g_old, rows, key, "cpu"), batching.flat_targets(g_new, rows, key, "cpu"), "flat_targets")


def test_collate_accepts_groups_of_the_other_package(old):
    """Group and Net are plain records, so batching must not depend on the class that carries them."""
    g_old = old("data").make_pairs(2, 3, (6, 12), 2, 8)
    rows = [np.arange(8)] * 3
    a, b = old("data").collate(g_old, rows, "cpu"), batching.collate(g_old, rows, "cpu")
    for f in dataclasses.fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        assert torch.equal(x, y) if torch.is_tensor(x) else x == y
