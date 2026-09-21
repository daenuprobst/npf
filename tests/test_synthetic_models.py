"""Same seed, same weights, same outputs. Old models against the refactored ones on the same batch."""
import dataclasses
import os

import numpy as np
import pytest
import torch
from conftest import assert_same_state, assert_same_tensors

from npf import batching, datasets, layers, models, nets
from npf.models.coloured import ColouredNPF, ColouredPGNN, transition_features
from npf.models.equilibrium import ThermoNPF
from npf.models.learned_incidence import SheafNPF

NAMES = ("npf", "npf-mlp", "npf-prior", "npf-1pass", "pgnn", "pgnn-eq10", "pgnn+", "pgnn+se", "gnn", "se-only")

# identical operations in identical order, so 0.0 passes as well. NPF_TEST_TOL=0 checks bit equality
TOL = float(os.environ.get("NPF_TEST_TOL", 1e-6))


def pair_batch(seed=3, n_nets=4, n_samples=8):
    groups = datasets.make_pairs(seed, n_nets, (6, 12), 2, 16)
    rows = [np.random.default_rng(seed).choice(16, n_samples, replace=False) for _ in groups]

    return groups, rows, batching.collate(groups, rows, "cpu")


def flow_batch(seed=3, n_nets=4, n_samples=8):
    groups = datasets.make_flows(seed, n_nets, (6, 12), 2, 4, "sat", 8)
    rows = [np.random.default_rng(seed).choice(len(g.m), n_samples, replace=False) for g in groups]

    return groups, rows, batching.collate(groups, rows, "cpu")


def build_both(old_build, new_build, seed):
    torch.manual_seed(seed)
    a = old_build()
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    b = new_build()

    # both constructors must consume the same number of random draws
    assert torch.equal(state, torch.random.get_rng_state()), "constructors consume different amounts of randomness"

    return a, b


def compare_outputs(a, b, tol=TOL):
    if isinstance(a, (tuple, list)):
        assert len(a) == len(b)

        for x, y in zip(a, b):
            compare_outputs(x, y, tol)
    else:
        assert_same_tensors(a, b, "output", tol)


@pytest.mark.parametrize("task", ["transitions", "next"])
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("seed", [0, 1])
def test_build(old, name, task, seed):
    old_models = old("models")
    a, b = build_both(lambda: old_models.build(name, task), lambda: models.build(name, task), seed)
    assert type(a).__name__ == type(b).__name__
    assert_same_state(a, b)
    assert [n for n, _ in a.named_parameters()] == [n for n, _ in b.named_parameters()]
    _, _, batch = pair_batch() if task == "transitions" else flow_batch()
    for mode in ("train", "eval"):
        getattr(a, mode)()
        getattr(b, mode)()
        out_a, out_b = a(batch), b(batch)
        compare_outputs(out_a, out_b)

    if sum(p.numel() for p in a.parameters()):
        # gradients, the training signal has to agree as well
        target = torch.linspace(0, 1, out_a.numel())
        for model, out in ((a, out_a), (b, out_b)):
            model.zero_grad()
            torch.nn.functional.mse_loss(out, target).backward()

        for (n, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
            assert (p.grad is None) == (q.grad is None), n

            if p.grad is not None:
                assert_same_tensors(p.grad, q.grad, f"grad {n}", 1e-6 * max(1.0, p.grad.abs().max().item()))


@pytest.mark.parametrize("name", ["npf@16", "pgnn@2", "gnn@1", "npf-mlp@3"])
def test_build_with_rounds(old, name):
    a, b = build_both(lambda: old("models").build(name, "next", hidden=16), lambda: models.build(name, "next", hidden=16), 0)
    assert_same_state(a, b)
    _, _, batch = flow_batch()
    compare_outputs(a(batch), b(batch))


def test_build_rejects_unknown_names(old):
    for build in (old("models").build, models.build):
        with pytest.raises(KeyError):
            build("no-such-model", "next")


def test_names_cover_build():
    for name in models.NAMES:
        models.build(name, "transitions", hidden=8)

    # npf-kl is new, the flat package that produced the earlier numbers does not have it
    assert set(models.NAMES) - {"npf-kl"} == set(NAMES)


def test_npf_details(old):
    """return_firing, a marking passed explicitly, noisy states (Kalman branch) and the step function."""
    old_models = old("models")
    _, _, batch = flow_batch()
    a, b = build_both(lambda: old_models.NPF("next", hidden=16), lambda: models.NPF("next", hidden=16), 2)
    compare_outputs(a(batch, return_firing=True), b(batch, return_firing=True))
    m = batch.m * 0.5 + 0.1
    compare_outputs(a(batch, m), b(batch, m))
    compare_outputs(a.step(m, batch, 0.3), b.step(m, batch, 0.3))
    compare_outputs(a.rate(m, batch), b.rate(m, batch))

    # empty places, the enabling factor and the series branch of the served fraction
    m0 = torch.where(torch.arange(len(m)) % 3 == 0, torch.zeros_like(m), m)
    compare_outputs(a.step(m0, batch), b.step(m0, batch))
    compare_outputs(a.step(m0 * 1e6, batch), b.step(m0 * 1e6, batch))
    _, _, pairs = pair_batch()
    a, b = build_both(lambda: old_models.NPF("transitions", hidden=16, noisy_states=True),
                      lambda: models.NPF("transitions", hidden=16, noisy_states=True), 2)
    assert_same_state(a, b)
    compare_outputs(a(pairs), b(pairs))


def test_layers_match_old_helpers(old):
    old_models = old("models")
    assert (layers.N_MARK, layers.N_PLACE_FEAT, layers.N_TRANS_FEAT) == (old_models.N_MARK, old_models.N_PLACE_FEAT, old_models.N_TRANS_FEAT)
    _, _, batch = pair_batch()
    g = torch.Generator().manual_seed(0)
    v = torch.rand(batch.n_trans, generator=g)
    compare_outputs(old_models.apply_incidence(v, batch), layers.apply_incidence(v, batch), 0.0)
    compare_outputs(old_models.place_features(batch, batch.m), layers.place_features(batch, batch.m), 0.0)
    compare_outputs(old_models.trans_features(batch), layers.trans_features(batch), 0.0)
    compare_outputs(old_models.marking_features(batch.m - 1), layers.marking_features(batch.m - 1), 0.0)

    for kw in ({}, {"weighted": False}, {"n_iter": 2}, {"obs_var": torch.tensor(0.3)}):
        compare_outputs(old_models.project_state_equation(v, batch, **kw), layers.project_state_equation(v, batch, **kw), 0.0)

    # the factored helpers against the inlined code of the old NPF.rate and NPF.step
    m = batch.m
    tokens = (m[batch.pre_p] / batch.pre_w).clamp(0.0, 1.0)
    enabled = tokens.new_ones(batch.n_trans).scatter_reduce(0, batch.pre_t, tokens, reduce="amin")
    compare_outputs(enabled, layers.enabling_factor(m, batch), 0.0)
    x = layers.scatter_sum(batch.pre_w * v[batch.pre_t], batch.pre_p, batch.n_places) / m.clamp(min=1e-30)
    served = torch.where(x < 1e-4, 1.0 - x / 2, -torch.expm1(-x) / x.clamp(min=1e-4))
    flow = v * served.new_ones(batch.n_trans).scatter_reduce(0, batch.pre_t, served[batch.pre_p], reduce="amin")
    compare_outputs(flow, layers.token_game_flow(v, m, batch), 0.0)


def test_linear_pgnn(old):
    group = datasets.make_pairs(1, 1, (7, 7), 2, 8)[0]
    a, b = build_both(lambda: old("models").LinearPGNN(group.net), lambda: models.LinearPGNN(group.net), 0)
    assert_same_state(a, b)
    batch = batching.collate([group], [np.arange(8)], "cpu")
    compare_outputs(a(batch), b(batch), 0.0)
    assert group.net.n_places == 7 and a.mask.shape == (group.net.n_trans, 7)


def test_thermo_npf(old):
    from benchmarks.synthetic import equilibrium as new_bench
    old_bench = old("thermo")
    g_old, g_new = old_bench.make_split(3, 4, (6, 12), n_samples=8), new_bench.make_split(3, 4, (6, 12), n_samples=8)
    for g, h in zip(g_old, g_new):
        assert np.array_equal(g.m, h.m) and np.array_equal(g.y, h.y) and np.array_equal(g.net.C, h.net.C)

    rows = [np.arange(8)] * 4
    batch = batching.collate(g_new, rows, "cpu")
    assert batch.X is not None and batch.X.dtype == torch.float64
    a, b = build_both(lambda: old_bench.ThermoNPF(), lambda: ThermoNPF(), 0)
    assert_same_state(a, b)
    out_a, out_b = a(batch), b(batch)
    compare_outputs(out_a, out_b)

    for model, out in ((a, out_a), (b, out_b)):
        out.square().sum().backward()

    for (n, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        assert_same_tensors(p.grad, q.grad, f"grad {n}", 1e-6 * max(1.0, p.grad.abs().max().item()))

    assert ThermoNPF(hidden=8, newton_steps=3).newton_steps == 3


def typed_batch(seed=2):
    """Place attribute e holds the integer type of a place, as in the learned incidence benchmark."""
    rng = np.random.default_rng(seed)
    groups = []
    for g in datasets.make_pairs(seed, 4, (6, 12), 1, 16):
        n = g.net
        e = rng.integers(0, 4, n.n_places)[:, None].astype(float)
        groups.append(datasets.Group(nets.Net(n.n_places, n.n_trans, n.pre_p, n.pre_t, n.pre_w, n.pos_p, n.pos_t, n.pos_w, e=e, a=n.a),
                                     g.m, g.m_b, g.dt, g.sigma))

    rows = [np.arange(8)] * len(groups)

    return batching.collate(groups, rows, "cpu"), batching.flat_targets(groups, rows, "sigma", "cpu")


@pytest.mark.parametrize("consistency", [True, False])
def test_sheaf_npf(old, consistency):
    old_sheaf = old("sheaf")
    import npf.models.learned_incidence as new_sheaf
    assert old_sheaf.N_TYPES == 4 and old_sheaf.RANK_TOLERANCE == new_sheaf.RANK_TOLERANCE
    batch, sigma = typed_batch()
    a, b = build_both(lambda: old_sheaf.SheafNPF(consistency=consistency), lambda: SheafNPF(consistency=consistency), 1)
    assert_same_state(a, b)

    with torch.no_grad():
        ratios = 0.05 * torch.randn(4, 4, generator=torch.Generator().manual_seed(0))
        a.raw_ratio.copy_(ratios)
        b.raw_ratio.copy_(ratios)

    out_a, out_b = a(batch), b(batch)
    compare_outputs(out_a, out_b)
    aux_a, aux_b = a.auxiliary_loss(batch, sigma), b.auxiliary_loss(batch, sigma)

    if consistency:
        compare_outputs(aux_a, aux_b)
    else:
        assert aux_a == aux_b == 0.0

    net_a, net_b = a.learned_net(batch), b.learned_net(batch)
    for f in dataclasses.fields(net_a):
        x, y = getattr(net_a, f.name), getattr(net_b, f.name)
        if torch.is_tensor(x):
            assert_same_tensors(x, y, f"learned_net.{f.name}", TOL)

    for model, loss in ((a, out_a.sum() + aux_a), (b, out_b.sum() + aux_b)):
        loss.backward()

    assert_same_tensors(a.raw_ratio.grad, b.raw_ratio.grad, "grad raw_ratio", 1e-6 * max(1.0, a.raw_ratio.grad.abs().max().item()))


def test_sheaf_npf_signature_change():
    """The new first positional argument is n_types. A positional call with the old meaning would be silent."""
    assert SheafNPF(n_types=3).raw_ratio.shape == (3, 3)
    assert SheafNPF().raw_ratio.shape == (4, 4) and SheafNPF().consistency is True


def coloured_batch(seed=1):
    from benchmarks.synthetic import coloured as new_bench
    groups = new_bench.make_split(seed, 3, (6, 12), n_traj=4, n_steps=3)

    return new_bench, groups


@pytest.mark.parametrize("mixing", [True, False])
def test_coloured_npf(old, mixing):
    old_bench = old("coloured")
    new_bench, groups = coloured_batch()
    g_old = old_bench.make_split(1, 3, (6, 12), n_traj=4, n_steps=3)
    for g, h in zip(g_old, groups):
        assert np.array_equal(g["m"], h["m"]) and np.array_equal(g["c"], h["c"]) and np.array_equal(g["angles"], h["angles"])

    b_old = old_bench.collate(g_old, "cpu", rng=np.random.default_rng(0), n=8)
    b_new = new_bench.collate(groups, "cpu", rng=np.random.default_rng(0), n=8)
    for name in ("m", "colour", "next_colour", "next_mass", "angles", "pre_p", "pos_w"):
        assert_same_tensors(getattr(b_old, name), getattr(b_new, name), name)

    a, b = build_both(lambda: old_bench.ColouredNPF(mixing=mixing), lambda: ColouredNPF(mixing=mixing), 0)
    assert_same_state(a, b)
    out_a, out_b = a(b_old), b(b_new)
    compare_outputs(out_a, out_b)
    compare_outputs(old_bench.loss_fn(out_a, b_old), new_bench.loss_fn(out_b, b_new))
    m, c = b_new.m * 0.7, b_new.colour.flip(0)
    compare_outputs(a(b_old, m, c), b(b_new, m, c))
    compare_outputs(old_bench.transition_features(b_old), transition_features(b_new), 0.0)


def test_coloured_pgnn(old):
    old_bench = old("coloured")
    new_bench, groups = coloured_batch()
    batch = new_bench.collate(groups, "cpu", step=0)
    a, b = build_both(lambda: old_bench.ColouredPGNN(), lambda: ColouredPGNN(), 0)
    assert_same_state(a, b)
    compare_outputs(a(batch), b(batch))
    a.eval()
    b.eval()
    assert old_bench.evaluate(a, groups, "cpu", horizon=3) == new_bench.evaluate(b, groups, "cpu", horizon=3)
