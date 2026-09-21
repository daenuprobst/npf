"""Data generation that lives in the benchmark scripts. Same seeds, same arrays as the old scripts."""
import numpy as np
import pytest
from conftest import assert_same_arrays

from test_synthetic_data import assert_same_groups


def test_learned_incidence_split(old):
    from benchmarks.synthetic import learned_incidence as new_bench
    old_bench = old("sheaf")
    assert new_bench.N_TYPES == old_bench.N_TYPES and (new_bench.SMALL, new_bench.LARGE) == (old_bench.SMALL, old_bench.LARGE)
    assert_same_arrays(new_bench.PRICE, old_bench.PRICE)

    for seed, places in ((1, (6, 12)), (4, (20, 30))):
        seen_old, true_old = old_bench.make_split(seed, 3, places, n_samples=16)
        seen_new, true_new = new_bench.make_split(seed, 3, places, n_samples=16)
        assert_same_groups(seen_old, seen_new)
        assert_same_groups(true_old, true_new)


def test_equilibrium_split(old):
    from benchmarks.synthetic import equilibrium as new_bench
    old_bench = old("thermo")
    e = np.linspace(-1, 1, 7)
    assert_same_arrays(old_bench.energy(e), new_bench.energy(e))

    for seed, places, scale in ((1, (6, 12), 4.0), (5, (6, 12), 12.0), (4, (20, 30), 4.0)):
        assert_same_groups(old_bench.make_split(seed, 3, places, n_samples=8, scale=scale), new_bench.make_split(seed, 3, places, n_samples=8, scale=scale))


def test_coloured_split(old):
    from benchmarks.synthetic import coloured as new_bench
    old_bench = old("coloured")
    assert (new_bench.DT, new_bench.SUBSTEPS) == (old_bench.DT, old_bench.SUBSTEPS)

    for g, h in zip(old_bench.make_split(2, 3, (6, 12), n_traj=3, n_steps=2), new_bench.make_split(2, 3, (6, 12), n_traj=3, n_steps=2)):
        assert list(g) == list(h)

        for key in ("angles", "m", "c"):
            assert_same_arrays(g[key], h[key], key)

        assert_same_arrays(g["net"].C, h["net"].C)


@pytest.mark.parametrize("noise", [0.0, 1.0])
def test_paper_protocol_data_and_models(old, noise):
    import torch
    from benchmarks.synthetic import paper_protocol as new_bench
    old_bench = old("paper_protocol")
    assert (new_bench.N_NETS, new_bench.N_SEEDS, new_bench.EPOCHS, new_bench.BATCH, new_bench.DT) == \
        (old_bench.N_NETS, old_bench.N_SEEDS, old_bench.EPOCHS, old_bench.BATCH, old_bench.DT)
    net_old, seen_old, true_old = old_bench.make_net_data(101, noise)
    net_new, seen_new, true_new = new_bench.make_net_data(101, noise)
    assert_same_groups([seen_old, true_old], [seen_new, true_new])

    for name in ("pgnn-linear", "npf", "npf-hard", "pgnn", "gnn", "se-only"):
        torch.manual_seed(0)
        a = old_bench.build(name, net_old, noise)
        torch.manual_seed(0)
        b = new_bench.build(name, net_new, noise)
        assert list(a.state_dict()) == list(b.state_dict())
        assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def test_paper_protocol_run(old, monkeypatch):
    """One short training run of the transductive protocol gives the same metrics."""
    from benchmarks.synthetic import paper_protocol as new_bench
    old_bench = old("paper_protocol")
    monkeypatch.setattr(old_bench, "EPOCHS", 2)
    monkeypatch.setattr(new_bench, "EPOCHS", 2)
    out_old = old_bench.run("npf", *old_bench.make_net_data(100, 1.0), 1.0, 0, "cpu")
    out_new = new_bench.run("npf", *new_bench.make_net_data(100, 1.0), 1.0, 0, "cpu")
    assert out_old == out_new
