"""The model registry of the general net, and one forward pass of every model on a small net."""
import numpy as np
import pytest
import torch

from npflow import batching, layers, models
from npflow.datasets import Group
from npflow.nets import Net

NAMES = ("npf", "pgnn", "pgnn+", "pgnn+se", "se-only", "npf-k", "npf-gma", "npf-gma-row", "pgnn-gma", "pgnn+se-gma")

# the models whose output is projected onto the state equation
PROJECTED = ("npf", "pgnn+se", "se-only", "npf-k", "npf-gma", "npf-gma-row", "pgnn+se-gma")


def small_batch():
    """The cycle p0 -> p1 -> p2 -> p0 and 2 p0 -> p2, two pairs of markings of one run each."""
    net = Net(
        3, 4, np.array([0, 1, 2, 0]), np.arange(4), np.array([1.0, 1, 1, 2]), np.array([1, 2, 0, 2]), np.arange(4),
        np.ones(4), e=np.zeros((3, 1)), a=np.zeros((4, 1)),
    )
    sigma = np.array([[1.0, 2, 0, 1], [0, 1, 1, 0]])

    # m_A holds every token that sigma consumes, so the counts can fire in any order
    m_a = np.array([[4.0, 2, 1], [1, 3, 0]]) + sigma @ net.Pre.T
    group = Group(net, m_a, m_a + sigma @ net.C.T, np.ones(2), t_id=np.arange(4), env=np.ones((2, 2)))

    return batching.collate([group], [np.arange(2)], "cpu")


def test_names_cover_build():
    for name in models.NAMES:
        models.build(name, "transitions", hidden=8)

    assert set(models.NAMES) == set(NAMES)


@pytest.mark.parametrize("kl_options", [None, {"n_iter": 30}], ids=["gauss", "kl"])
@pytest.mark.parametrize("name", NAMES)
def test_forward_pass(name, kl_options):
    torch.manual_seed(0)
    b = small_batch()
    model = models.build(name, "transitions", hidden=8, n_ids=4, n_env=2, kl_options=kl_options)

    with torch.no_grad():
        sigma = model(b)

    assert sigma.shape == (b.n_trans,) and torch.isfinite(sigma).all()

    if name in PROJECTED:
        assert (b.m_b - b.m - layers.apply_incidence(sigma, b)).abs().max() < 1e-3

    # the I projection returns positive firing counts
    if name in PROJECTED and kl_options is not None:
        assert (sigma > 0).all()


def test_next_is_npf_only():
    b = small_batch()

    with torch.no_grad():
        m = models.build("npf", "next", hidden=8)(b)

    assert m.shape == b.m.shape and (m >= 0).all()

    with pytest.raises(ValueError):
        models.build("pgnn", "next")
