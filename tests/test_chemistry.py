"""Batched beam search and the training loss of the token game on a few hundred Schneider 50k reactions."""
import numpy as np
import torch
from conftest import assert_same_arrays

from npflow import chem


def test_beam_search_batch_matches_one_at_a_time(schneider):
    # padding changes the summation order, so log probabilities agree to round-off and candidates of equal
    # probability may swap places, every candidate with a probability of its own must be the same
    torch.manual_seed(0)
    model = chem.TokenGame(d=32, rounds=2, attention=1).eval()
    rs = [r for r in schneider["reactions"] if r["edits"] is not None][:8]
    for r, batched in zip(rs, model.beam_search_batch(chem.collate(rs, "cpu"), 3)):
        single = model.beam_search(chem.collate([r], "cpu"), 3)
        assert len(single) == len(batched)
        assert np.allclose(sorted(lp for _, lp in single), sorted(lp for _, lp in batched), atol=1e-4)

        for k, ((edits_a, lp_a), (edits_b, _)) in enumerate(zip(single, batched)):
            if all(abs(lp_a - lp) > 1e-4 for j, (_, lp) in enumerate(single) if j != k):
                assert_same_arrays(edits_a, edits_b)


def test_token_game_loss_is_a_negative_log_probability(schneider, monkeypatch):
    # a noisy record can leave firings to come of which none is enabled. with every firing disabled and large rates
    # on the disabled ones, a loss that scored them outside log_z would be about -50
    torch.manual_seed(0)
    model = chem.TokenGame(d=32, rounds=2, attention=1)
    batch = chem.collate([r for r in schneider["reactions"] if r["edits"] is not None][:8], "cpu")
    rates = model.rates

    def disabled(*args, **kw):
        logits, enabled, *rest = rates(*args, **kw)

        return (logits + 50.0, torch.zeros_like(enabled), *rest)

    monkeypatch.setattr(model, "rates", disabled)

    for seed in range(5):
        torch.manual_seed(seed)
        loss = model.loss(batch)
        assert torch.isfinite(loss) and loss.item() >= 0


def reference_loss(model, b):
    """The loss for one firing vector per reaction, as written before target sets, the regression reference."""
    true = b["edits"] > 0
    keep = torch.rand(len(true), 1, 1)
    draw = torch.triu(torch.rand(true.shape), 1)
    draw = draw + draw.transpose(1, 2)
    fired = true & (draw < keep)
    cur = torch.where(fired, b["edits"] - 1, b["ba"])
    flat, logits, enabled = model.events(b, cur, fired)
    remaining = torch.nn.functional.one_hot((b["edits"] - 1).clamp(min=0), chem.N_BOND).bool() & (true & ~fired)[..., None] & (logits > -1e3)
    target = remaining & enabled
    hit, found = logits.masked_fill(~target, -1e4).flatten(1), target.flatten(1).any(1)

    done = ~remaining.flatten(1).any(1)
    per_state = torch.logsumexp(flat, 1) - torch.where(done, flat[:, -1], torch.logsumexp(hit, 1))
    usable = done | found

    return (per_state * usable).sum() / usable.sum().clamp(min=1)


def test_token_game_loss_with_one_vector_is_unchanged(schneider):
    # a set holding only the recorded vector draws one more random number, after the fired part, so it matches too
    torch.manual_seed(0)
    model = chem.TokenGame(d=32, rounds=2, attention=1)
    rs = [r for r in schneider["reactions"] if r["edits"] is not None][:8]
    plain = chem.collate(rs, "cpu")
    single = chem.collate([dict(r, edits_set=[r["edits"]]) for r in rs], "cpu")

    for seed in range(3):
        torch.manual_seed(seed)
        expected = reference_loss(model, plain)
        torch.manual_seed(seed)
        assert torch.allclose(model.loss(plain), expected, atol=1e-6)
        torch.manual_seed(seed)
        assert torch.allclose(model.loss(single), expected, atol=1e-6)


def test_continuations_of_a_set_of_firing_vectors():
    # three atoms, vector 0 makes bond 0-1 single, vector 1 makes bond 0-2 single, vector 2 is padding
    options = torch.zeros(1, 3, 3, 3, dtype=torch.int8)
    options[0, 0, 0, 1] = options[0, 0, 1, 0] = 2
    options[0, 1, 0, 2] = options[0, 1, 2, 0] = 2
    real = torch.tensor([[True, True, False]])
    allowed = torch.triu(torch.ones(3, 3, dtype=torch.bool), 1)[None, ..., None].expand(1, 3, 3, chem.N_BOND)
    edits = options[:, 0].long()

    # nothing fired, both vectors continue and their first firings are the correct next events
    live, agree = chem.TokenGame.continuations(options, real, edits, torch.zeros(1, 3, 3, dtype=torch.bool), allowed)
    assert agree.tolist() == [[True, True, False]]
    assert sorted(map(tuple, live.any(1).nonzero().tolist())) == [(0, 0, 1, 1), (0, 0, 2, 1)]

    # after 0-1 fired only vector 0 contains the fired part, and it has nothing left, so STOP is correct
    fired = torch.zeros(1, 3, 3, dtype=torch.bool)
    fired[0, 0, 1] = fired[0, 1, 0] = True
    live, agree = chem.TokenGame.continuations(options, real, edits, fired, allowed)
    assert agree.tolist() == [[True, False, False]] and not live.any()


def test_token_game_loss_with_target_sets_is_bounded(schneider):
    torch.manual_seed(0)
    model = chem.TokenGame(d=32, rounds=2, attention=1)
    rs = [r for r in schneider["reactions"] if r["edits"] is not None][:8]

    # a second vector per reaction that fires the first place of the record to another type
    def other(e):
        e = np.array(e, copy=True)

        if len(e):
            e[0, 2] = (e[0, 2] + 1) % chem.N_BOND

        return e

    batch = chem.collate([dict(r, edits_set=[r["edits"], other(r["edits"])]) for r in rs], "cpu")
    assert batch["edits_set"].shape[1] == 2 and batch["n_set"].tolist() == [2] * 8

    for seed in range(3):
        torch.manual_seed(seed)
        loss = model.loss(batch)
        assert torch.isfinite(loss) and loss.item() >= 0
        loss.backward()




