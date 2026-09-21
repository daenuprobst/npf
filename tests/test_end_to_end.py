"""The experiment scripts run end to end, old and new, on a small budget. Result files must agree in every number.

The data sets are cut down by patching the loaders, everything else is the unmodified main function with its
command line. All output goes to the temporary directory of the test.
"""
import json
import sys

import pytest
import torch


def run_main(module, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", [module.__name__] + argv)
    module.main()


def load_result(path):
    result = json.loads(path.read_text())
    assert result.pop("train_seconds") >= 0

    return result


@pytest.mark.parametrize("task,regime,model", [("transitions", "petri", "npf"), ("transitions", "graph", "pgnn+se"), ("transitions", "petri-ode", "gnn"),
                                               ("next", "petri-min", "npf@8"), ("next", "graph-sat", "pgnn")])
def test_synthetic_experiment(old, monkeypatch, tmp_path, task, regime, model):
    from benchmarks.synthetic import experiment as new_exp
    from npf import datasets
    old_exp, old_data = old("experiment"), old("data")

    def fewer_nets(fn):
        return lambda seed, n_nets, *args, **kw: fn(seed, 16 if n_nets == 300 else 3, *args, **kw)

    for name in ("make_pairs", "make_flow_pairs", "make_flows"):
        monkeypatch.setattr(old_data, name, fewer_nets(getattr(old_data, name)))
        monkeypatch.setattr(datasets, name, fewer_nets(getattr(datasets, name)))

    argv = ["--task", task, "--regime", regime, "--model", model, "--seed", "1", "--iters", "6"]
    run_main(old_exp, argv + ["--root", str(tmp_path / "old")], monkeypatch)
    run_main(new_exp, argv + ["--root", str(tmp_path / "new")], monkeypatch)
    file = f"{task}/{regime}/{model}-1.json"
    a, b = load_result(tmp_path / "old" / file), load_result(tmp_path / "new" / file)
    assert a == b
    assert len(a["curve"]) == 1 and a["metrics"]["test"]

    # a second run reads the cache that the first one wrote
    run_main(new_exp, argv + ["--root", str(tmp_path / "new")], monkeypatch)
    assert load_result(tmp_path / "new" / file) == b


CHEM_RUNS = [
    ("classify", "npf", []), ("classify", "pgnn", []), ("classify", "npf-nogate", ["--firing", "--labels", "40"]),
    ("map", "npf", []), ("map", "pgnn", []), ("map", "npf-nokept", []),
    ("forward", "npf", ["--width", "32", "--rounds", "2", "--attention", "1"]),
    ("forward", "npf-noenabling", ["--width", "32", "--rounds", "2", "--attention", "1", "--hops", "3", "--no-beam"]),
    ("forward", "pgnn", []), ("forward", "npf-oneshot", []),
]


@pytest.mark.parametrize("task,model,extra", CHEM_RUNS, ids=[f"{t}-{m}" for t, m, _ in CHEM_RUNS])
def test_chemistry_experiment(old, schneider_all, monkeypatch, tmp_path, task, model, extra):
    from benchmarks.chemistry import experiment as new_exp
    from npf import chem
    old_exp = old("chem_experiment")
    import copy
    reactions = schneider_all["reactions"]

    if task == "classify":
        # the script holds out the first 500 shuffled training reactions for validation
        picked = [r for r in reactions if r["split"] == "train"][:620] + [r for r in reactions if r["split"] == "test"][:40]
    else:
        picked = [r for r in reactions if r["target"] is not None][:150]

    subset = {"reactions": picked, "classes": schneider_all["classes"]}

    # the scripts attach histograms and label masks to the reactions, so each run gets its own copy
    monkeypatch.setattr(old("chem_data"), "load", lambda path="": copy.deepcopy(subset))
    monkeypatch.setattr(chem, "load", lambda path="": copy.deepcopy(subset))
    argv = ["--task", task, "--model", model, "--seed", "0", "--epochs", "1", "--batch", "8"] + extra
    run_main(old_exp, argv + ["--root", str(tmp_path / "old")], monkeypatch)
    run_main(new_exp, argv + ["--root", str(tmp_path / "new")], monkeypatch)
    files = sorted(p.relative_to(tmp_path / "old") for p in (tmp_path / "old").rglob("*") if p.is_file())

    # the new tree also writes the beam candidates of the token game, which the flat one did not
    produced = sorted(p.relative_to(tmp_path / "new") for p in (tmp_path / "new").rglob("*") if p.is_file())
    assert files == [p for p in produced if p.parts[-2] != "beams"]
    assert {p.suffix for p in files} == {".json", ".pt"}

    for file in files:
        if file.suffix == ".json":
            a, b = load_result(tmp_path / "old" / file), load_result(tmp_path / "new" / file)

            # merged top-k and the official denominator are new, every metric the flat script reported has to agree
            if isinstance(b.get("metrics"), dict):
                b = b | {"metrics": {k: v for k, v in b["metrics"].items() if "merged" not in k and not k.endswith("_official")}}

            assert a == b
        else:
            a, b = torch.load(tmp_path / "old" / file), torch.load(tmp_path / "new" / file)

            # since the token game runs over events, one weight of one ablation (the stop bias without enabling)
            # differs by 2e-10, which is round-off. every other weight is equal bit for bit
            assert list(a) == list(b) and all(torch.allclose(a[k], b[k], rtol=0, atol=1e-9) for k in a)
            assert sum(not torch.equal(a[k], b[k]) for k in a) <= 1

    # the saved weights are scored again without training
    run_main(new_exp, argv + ["--root", str(tmp_path / "new"), "--evaluate-only"], monkeypatch)
