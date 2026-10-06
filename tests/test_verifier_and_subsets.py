"""The verifier of beam candidates, the nested training subsets and the grouped softmax of the verifier loss."""
import numpy as np
import torch

from npflow import chem
from benchmarks.chemistry import experiment, verify


def test_subsets_are_nested_and_deterministic():
    small, large = experiment.subset_lines(500), experiment.subset_lines(5000)
    assert np.array_equal(small, large[:500])
    assert np.array_equal(small, experiment.subset_lines(500))
    assert len(set(large.tolist())) == 5000 and large.max() < experiment.USPTO_MIT_TRAIN_LINES


def test_group_logsumexp_matches_torch():
    x = torch.randn(11)
    owner = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3])
    expected = torch.stack([torch.logsumexp(x[owner == k], 0) for k in range(4)])
    assert torch.allclose(verify.group_logsumexp(x, owner, 4), expected, atol=1e-6)


def test_listwise_loss_is_zero_when_all_candidates_are_correct():
    score, owner = torch.randn(6), torch.tensor([0, 0, 0, 1, 1, 1])
    assert verify.listwise_loss(score, torch.ones(6, dtype=torch.bool), owner, 2).abs() < 1e-6


def test_untrained_verifier_keeps_the_beam_order(schneider):
    reactions = [r for r in schneider["reactions"] if r["edits"] is not None and len(r["edits"])][:4]
    batch = [(r, [(r["edits"], -0.1, True), (r["edits"][:0], -2.0, False)]) for r in reactions]
    rows, edits, log_p, correct, owner = verify.collate(batch, "cpu")
    torch.manual_seed(0)
    score = chem.Verifier(d=32, rounds=2)(rows, edits, log_p)
    assert torch.equal(score, log_p)
    assert correct.tolist() == [True, False] * len(reactions) and owner.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


def test_verifier_readout_vanishes_without_firing(schneider):
    reactions = schneider["reactions"][:3]
    batch = [(r, [(np.zeros((0, 3), np.int64), -1.0, False)]) for r in reactions]
    rows, edits, log_p, _, _ = verify.collate(batch, "cpu")
    model = chem.Verifier(d=32, rounds=2)
    fired = edits > 0
    before = model.marking(rows, rows["ba"], torch.zeros_like(fired))
    assert torch.equal(model.marking(rows, torch.where(fired, edits - 1, rows["ba"]), fired), before)


def test_exact_minimisation_never_costs_more_than_its_seed(schneider):
    """Branch and bound starts from a feasible mapping and may only improve it."""
    reactions = [r for r in schneider["reactions"] if r["target"] is not None][:8]
    for r in reactions:
        seed = r["target"].astype(np.int64)
        best, cost, _ = chem.branch_and_bound(r, seed, budget=4000)
        assert cost <= chem.cost_of(r, seed) + 1e-9
        assert len(set(best.tolist())) == len(best)
        assert (r["a"]["element"][best] == r["b"]["element"]).all()


def test_domains_contain_the_recorded_seat_for_unchanged_atoms(schneider):
    """An atom whose environment is the same on both sides keeps its seat in the domain."""
    reactions = [r for r in schneider["reactions"] if r["target"] is not None][:8]
    inside = total = 0

    for r in reactions:
        doms = chem.domains(r)
        for i, j in enumerate(r["target"].astype(np.int64)):
            total += 1
            inside += j in doms[i]

    assert inside / total > 0.8


def test_count_orders_matches_the_factorial_when_nothing_blocks(schneider):
    """Firings on disjoint atom pairs that only break bonds are enabled in every order."""
    import itertools
    import math

    for r in schneider["reactions"]:
        if r["edits"] is None or not 2 <= len(r["edits"]) <= 4:
            continue

        breaks = np.array([e for e in r["edits"] if e[2] == 0], dtype=np.int64)
        atoms = [set(e[:2]) for e in breaks]
        if len(breaks) < 2 or any(a & b for a, b in itertools.combinations(atoms, 2)):
            continue

        n, feasible = chem.count_orders(r, breaks)
        assert feasible and n == float(math.factorial(len(breaks)))
        return
