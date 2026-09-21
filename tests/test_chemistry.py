"""Chemistry code. Old chem_data and chem_models against npf.chem on a few hundred Schneider 50k reactions."""
import csv
import importlib
import itertools
import os

import numpy as np
import pytest
import torch
from conftest import DATA, RESULTS, assert_same_arrays, assert_same_state, assert_same_tensors

from npf import chem

featurise, decode, encoder = (importlib.import_module(f"npf.chem.{name}") for name in ("featurisation", "decode", "encoder"))

# NPF_TEST_TOL=0 asks for bit equality, which holds on the CPU
TOL = float(os.environ.get("NPF_TEST_TOL", 1e-6))

# checkpoints written by the old code and the constructor that the old experiment script used for them
CHECKPOINTS = {
    "chem/classify/npf-0.pt": ("Classifier", (50,), {}),
    "chem/classify/npf-firing-0.pt": ("Classifier", (50,), {"n_firing_types": 300}),
    "chem/map/npf-0.pt": ("Mapper", (), {}),
    "chem/map/pgnn-0.pt": ("Mapper", (), {"petri": False}),
    "chem/map/npf-nomorphism-0.pt": ("Mapper", (), {"morphism": False}),
    "chem/forward/npf-0.pt": ("TokenGame", (), {}),
    "chem/forward/npf-hops-0.pt": ("TokenGame", (), {"hops": 7}),
    "chem/forward/npf-noenabling-0.pt": ("TokenGame", (), {"enabling": False}),
    "chem/forward/pgnn-0.pt": ("Forward", (), {"petri": False}),
    "chem/forward/npf-oneshot-0.pt": ("Forward", (), {"petri": True}),
}


def mapped(reactions):
    return [r for r in reactions if r["target"] is not None]


def small_batches(reactions, size=6, n=3):
    rs = sorted(mapped(reactions), key=lambda r: len(r["a"]["x"]))[:size * n]

    return [rs[k:k + size] for k in range(0, len(rs), size)]


def compare(a, b, tol=TOL):
    if isinstance(a, (tuple, list)):
        assert len(a) == len(b)

        for x, y in zip(a, b):
            compare(x, y, tol)
    elif torch.is_tensor(a):
        assert_same_tensors(a, b, "output", tol)
    else:
        assert_same_arrays(a, b, "output")


def test_constants(old):
    data, models = old("chem_data"), old("chem_models")
    for name in ("ELEMENTS", "BOND_TYPE", "EXTRA_CAPACITY", "HYPERVALENT", "N_ATOM_FEAT", "MAX_PRECURSOR_ATOMS", "MAX_PRODUCT_ATOMS", "RD_BOND"):
        assert getattr(data, name) == getattr(featurise, name), name

    for name in ("RECONSTRUCTION", "NORMAL_VALENCE", "ANION_PRIORITY", "NEUTRALISE_MIN_ATOMS"):
        assert getattr(data, name) == getattr(decode, name), name

    assert_same_arrays(data.BOND_ORDER, featurise.BOND_ORDER)
    assert models.N_BOND == encoder.N_BOND == chem.N_BOND
    assert chem.BOND_ORDER is featurise.BOND_ORDER


def test_public_names_of_the_old_modules_are_still_reachable(old):
    """Every public function, class and constant of chem_data and chem_models lives somewhere in npf.chem."""
    import types
    new_names = set()

    for name in ("", ".featurisation", ".decode", ".encoder", ".classifier", ".mapper", ".token_game", ".one_shot"):
        new_names |= set(vars(importlib.import_module("npf.chem" + name)))

    missing = []
    for module in (old("chem_data"), old("chem_models")):
        for name, value in vars(module).items():
            if name.startswith("__") or isinstance(value, types.ModuleType):
                continue

            if getattr(value, "__module__", module.__name__) != module.__name__:
                continue

            # the molecule builder became public, it is used by the review script
            if {"_mol": "molecule"}.get(name, name) not in new_names:
                missing.append(f"{module.__name__}.{name}")

    assert not missing, missing


def test_collate(old, schneider):
    data = old("chem_data")
    reactions = schneider["reactions"]
    for k in range(0, len(reactions), 40):
        rs = reactions[k:k + 40]
        a, b = data.collate(rs, "cpu"), chem.collate(rs, "cpu")
        assert list(a) == list(b)

        for key in a:
            assert_same_tensors(a[key], b[key], key)

    assert any(r["target"] is None for r in reactions) and any(r["edits"] is not None and len(r["edits"]) for r in reactions)


def test_collate_with_histograms_and_label_mask(old, schneider):
    rs = [dict(r) for r in schneider["reactions"][:16]]
    for k, r in enumerate(rs):
        r["hist"], r["labelled"] = np.full(7, k, np.float32), k % 2 == 0

    a, b = old("chem_data").collate(rs, "cpu"), chem.collate(rs, "cpu")
    assert list(a) == list(b) and "hist" in b

    for key in a:
        assert_same_tensors(a[key], b[key], key)


def test_dense_bonds_and_products(old, schneider):
    data = old("chem_data")
    for r in schneider["reactions"][:120]:
        assert_same_arrays(data.dense_bonds(r["a"]), chem.dense_bonds(r["a"]))
        product = r["smiles"].split(">>")[1]
        assert data.canonical_product(product) == chem.canonical_product(product)


def test_marking_fragments(old, schneider):
    data = old("chem_data")
    rs = mapped(schneider["reactions"])
    assert len(rs) > 250
    reached = 0

    for r in rs:
        a, b = data.marking_fragments(r["a"], r["edits"]), chem.marking_fragments(r["a"], r["edits"])
        assert a == b, r["smiles"]
        assert data.marking_to_products(r["a"], r["edits"]) == chem.marking_to_products(r["a"], r["edits"])
        reached += chem.canonical_product(r["smiles"].split(">>")[1]) in b[0]

    # the recorded firing vector reaches the recorded product for most reactions, so the comparison is not vacuous
    assert reached > 0.8 * len(rs)


@pytest.mark.parametrize("version,min_atoms", [(1, 2), (2, 2), (3, 2), (3, 1)])
def test_reconstruction_switches(old, schneider, monkeypatch, version, min_atoms):
    """The switches moved from chem_data to npf.chem.decode and have to be set there."""
    data = old("chem_data")
    monkeypatch.setattr(data, "RECONSTRUCTION", version)
    monkeypatch.setattr(data, "NEUTRALISE_MIN_ATOMS", min_atoms)
    monkeypatch.setattr(decode, "RECONSTRUCTION", version)
    monkeypatch.setattr(decode, "NEUTRALISE_MIN_ATOMS", min_atoms)

    for r in mapped(schneider["reactions"])[:150]:
        assert data.marking_fragments(r["a"], r["edits"]) == chem.marking_fragments(r["a"], r["edits"])


def test_mapping_from_marking(old, schneider):
    data = old("chem_data")
    found = 0

    for r in mapped(schneider["reactions"])[:80]:
        a, b = data.mapping_from_marking(r, r["edits"]), chem.mapping_from_marking(r, r["edits"])
        assert (a is None) == (b is None)

        if a is not None:
            assert_same_arrays(a, b)
            found += 1

    assert found > 40


def test_graph_and_featurise(old):
    tsv = DATA / "schneider50k.tsv"
    if not tsv.exists():
        pytest.skip(f"{tsv} not found")

    data = old("chem_data")

    with open(tsv) as f:
        rows = list(itertools.islice(csv.DictReader(f, delimiter="\t"), 0, 3000, 50))

    for k, row in enumerate(rows):
        row["label"], row["id"] = 0, k
        a, b = data.featurise(dict(row)), chem.featurise(dict(row))
        assert (a is None) == (b is None)

        if a is None:
            continue

        assert list(a) == list(b)

        for key in a:
            if isinstance(a[key], dict):
                assert list(a[key]) == list(b[key])

                for name in a[key]:
                    assert_same_arrays(a[key][name], b[key][name], f"{key}.{name}")
            elif isinstance(a[key], np.ndarray):
                assert_same_arrays(a[key], b[key], key)
            else:
                assert a[key] == b[key], key


def test_mlp_and_layers(old):
    models = old("chem_models")
    for cls in ("PetriLayer", "Encoder"):
        torch.manual_seed(0)
        a = getattr(models, cls)(32)
        torch.manual_seed(0)
        b = getattr(encoder, cls)(32)
        assert_same_state(a, b)

    torch.manual_seed(0)
    a = models.Encoder(32, 3, 2, n_extra=8)
    torch.manual_seed(0)
    b = chem.Encoder(32, 3, 2, n_extra=8)
    assert_same_state(a, b)


@pytest.mark.parametrize("cls,args,kw", [
    ("Classifier", (50,), {}), ("Classifier", (50,), {"petri": False}), ("Classifier", (50,), {"gate": False}),
    ("Classifier", (50,), {"n_firing_types": 300}), ("Classifier", (50,), {"explicit_firing": True, "gate": False}),
    ("Mapper", (), {}), ("Mapper", (), {"petri": False}), ("Mapper", (), {"equilibrium": False}), ("Mapper", (), {"kept_bonds": False}),
    ("Mapper", (), {"morphism": False}), ("Mapper", (), {"token_cost": False}),
    ("TokenGame", (), {}), ("TokenGame", (), {"enabling": False}), ("TokenGame", (), {"hops": 3}), ("TokenGame", (), {"d": 192, "attention": 6}),
    ("Forward", (), {}), ("Forward", (), {"petri": False}),
])
def test_same_initialisation_and_outputs(old, schneider, cls, args, kw):
    torch.manual_seed(0)
    a = getattr(old("chem_models"), cls)(*args, **kw)
    state = torch.random.get_rng_state()
    torch.manual_seed(0)
    b = getattr(chem, cls)(*args, **kw)
    assert torch.equal(state, torch.random.get_rng_state()), "constructors consume different amounts of randomness"
    assert_same_state(a, b)
    a.eval()
    b.eval()
    rs = small_batches(schneider["reactions"], 4, 1)[0]
    batch = chem.collate(rs, "cpu")

    with torch.no_grad():
        compare(a(batch), b(batch))


@pytest.mark.parametrize("path", list(CHECKPOINTS))
def test_checkpoints_load_into_both(old, schneider, path):
    file = RESULTS / path
    if not file.exists():
        pytest.skip(f"{file} not found")

    cls, args, kw = CHECKPOINTS[path]
    state = torch.load(file, map_location="cpu")
    a, b = getattr(old("chem_models"), cls)(*args, **kw), getattr(chem, cls)(*args, **kw)
    assert list(b.state_dict()) == list(state), "keys of the checkpoint differ from the new model"
    a.load_state_dict(state)
    b.load_state_dict(state)
    a.eval()
    b.eval()

    for rs in small_batches(schneider["reactions"]):
        batch = chem.collate(rs, "cpu")

        with torch.no_grad():
            out_a, out_b = a(batch), b(batch)
            compare(out_a, out_b)

            if cls == "Mapper":
                compare(a.decode(out_a, batch), b.decode(out_b, batch))
                compare(a.loss(out_a, batch), b.loss(out_b, batch))
                compare(a(batch, all_rounds=True), b(batch, all_rounds=True))
            elif cls == "Classifier":
                compare(a(batch, return_gate=True), b(batch, return_gate=True))
                compare(a.auxiliary_loss(batch), b.auxiliary_loss(batch))
            elif cls == "Forward":
                compare(a.loss(out_a, batch), b.loss(out_b, batch))
                compare(a.decode(out_a, batch, rs), b.decode(out_b, batch, rs))
            else:
                compare(a.decode(out_a, batch, rs), b.decode(out_b, batch, rs))
                torch.manual_seed(5)
                loss_a = a.loss(batch)
                torch.manual_seed(5)
                compare(loss_a, b.loss(batch))


def test_beam_search(old, schneider):
    file = RESULTS / "chem/forward/npf-0.pt"
    if not file.exists():
        pytest.skip(f"{file} not found")

    state = torch.load(file, map_location="cpu")
    a, b = old("chem_models").TokenGame(), chem.TokenGame()
    a.load_state_dict(state)
    b.load_state_dict(state)
    a.eval()
    b.eval()

    for r in small_batches(schneider["reactions"], 3, 1)[0]:
        batch = chem.collate([r], "cpu")
        ranked_a, ranked_b = a.beam_search(batch, 3), b.beam_search(batch, 3)
        assert len(ranked_a) == len(ranked_b)

        for (edits_a, lp_a), (edits_b, lp_b) in zip(ranked_a, ranked_b):
            assert_same_arrays(edits_a, edits_b)
            assert abs(lp_a - lp_b) <= TOL


@pytest.mark.parametrize("kw", [{"n_firing_types": 20}, {"explicit_firing": True, "gate": False}])
def test_firing_heads_of_the_classifier(old, schneider, kw):
    """n_firing_types together with explicit_firing fails in the old code as well (Linear of 2048 inputs fed with 2304),
    so the two options are compared one at a time."""
    torch.manual_seed(1)
    a = old("chem_models").Classifier(50, **kw)
    torch.manual_seed(1)
    b = chem.Classifier(50, **kw)

    # dropout is random in training mode
    a.eval()
    b.eval()
    rs = [dict(r) for r in small_batches(schneider["reactions"], 6, 1)[0]]
    for k, r in enumerate(rs):
        r["hist"] = np.arange(20, dtype=np.float32) * (k % 3)

    batch = chem.collate(rs, "cpu")
    compare(a(batch), b(batch))
    compare(a.firing_loss(batch), b.firing_loss(batch))

    if "explicit_firing" not in kw:
        return

    compare(a.fired_transitions(batch, *[torch.ones(len(rs), n, 128) for n in (batch["xa"].shape[1], batch["xb"].shape[1])]),
            b.fired_transitions(batch, *[torch.ones(len(rs), n, 128) for n in (batch["xa"].shape[1], batch["xb"].shape[1])]))


def test_splits_and_metrics_of_the_experiment(old, schneider):
    """The helper functions of the experiment script, which moved without changes."""
    from benchmarks.chemistry import experiment as new_exp
    old_exp = old("chem_experiment")
    assert (new_exp.EPOCHS, new_exp.BATCH, new_exp.MAX_TOKENS, new_exp.REFINE, new_exp.MAIN_METRIC) == \
        (old_exp.EPOCHS, old_exp.BATCH, old_exp.MAX_TOKENS, old_exp.REFINE, old_exp.MAIN_METRIC)

    for task in ("classify", "map", "forward", "map-clean"):
        for x, y in zip(old_exp.splits(schneider, task), new_exp.splits(schneider, task)):
            assert [r["id"] for r in x] == [r["id"] for r in y]

    rs = mapped(schneider["reactions"])[:60]
    for r in rs:
        assert old_exp.tokens_moved(r, r["target"]) == new_exp.tokens_moved(r, r["target"])
        assert old_exp.firing(r, r["target"]) == new_exp.firing(r, r["target"])
        assert old_exp.firing(r, r["target"], False) == new_exp.firing(r, r["target"], False)
        assert old_exp.product_found(r, r["edits"]) == new_exp.product_found(r, r["edits"])
        assert old_exp.recorded_products(r) == new_exp.recorded_products(r)
        assert_same_arrays(old_exp.token_descent(r, r["target"].astype(np.int64)), new_exp.token_descent(r, r["target"].astype(np.int64)))

    for seed in (None, 0):
        rng = (lambda: None) if seed is None else (lambda: np.random.default_rng(seed))
        x, y = old_exp.batch_indices(rs, 8, rng()), new_exp.batch_indices(rs, 8, rng())
        assert [list(map(int, b)) for b in x] == [list(map(int, b)) for b in y]
