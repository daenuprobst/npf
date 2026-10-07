"""The experiment scripts run end to end on a small budget and write complete result files.

The data sets are cut down by patching the loaders, everything else is the unmodified main function with its
command line. All output goes to the temporary directory of the test.
"""
import copy
import json
import pickle
import sys

import pytest


def run_main(module, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", [module.__name__] + argv)
    module.main()


def load_result(path):
    result = json.loads(path.read_text())
    assert result.pop("train_seconds") >= 0

    return result


SMALL = ["--width", "32", "--rounds", "2", "--attention", "1"]
CHEM_RUNS = [
    ("classify", "npf", []), ("classify", "pgnn", []), ("classify", "npf-sigma", []), ("classify", "pgnn-sigma", []),
    ("classify", "npf", ["--firing", "--labels", "40", "--select", "last"]),
    ("forward", "npf", SMALL), ("forward", "npf", SMALL + ["--net-targets", "TARGETS"]),
    ("forward", "npf-noenabling", SMALL + ["--no-beam"]), ("forward", "pgnn", []), ("forward", "npf-oneshot", []),
]


@pytest.mark.parametrize("task,model,extra", CHEM_RUNS, ids=[f"{t}-{m}-{k}" for k, (t, m, _) in enumerate(CHEM_RUNS)])
def test_chemistry_experiment(schneider_all, monkeypatch, tmp_path, task, model, extra):
    from benchmarks.chemistry import experiment, net_targets
    from npflow import chem
    reactions = schneider_all["reactions"]

    if task == "classify":
        # the script holds out the first 500 shuffled training reactions for validation
        picked = [r for r in reactions if r["split"] == "train"][:620] + [r for r in reactions if r["split"] == "test"][:40]
    else:
        picked = [r for r in reactions if r["target"] is not None][:150]

    subset = {"reactions": picked, "classes": schneider_all["classes"]}

    # the scripts attach histograms and label masks to the reactions, so each run gets its own copy
    monkeypatch.setattr(chem, "load", lambda path="": copy.deepcopy(subset))

    # targets of the net for the training reactions, computed the way benchmarks.chemistry.net_targets does
    if "TARGETS" in extra:
        train = experiment.splits(copy.deepcopy(subset), "forward", recorded=False)[0]
        rows = [net_targets.targets((r, 0.5, 16)) for r in train]
        (tmp_path / "targets.pkl").write_bytes(pickle.dumps({"targets": {x["id"]: x["vectors"] for x in rows if x["vectors"]}}))
        extra = [str(tmp_path / "targets.pkl") if a == "TARGETS" else a for a in extra]

    argv = ["--task", task, "--model", model, "--seed", "0", "--epochs", "1", "--batch", "8", "--root", str(tmp_path)] + extra
    run_main(experiment, argv, monkeypatch)
    results = [p for p in tmp_path.rglob("*.json")]
    assert len(results) == 1 and results[0].with_suffix(".pt").exists()
    result = load_result(results[0])
    assert result["args"]["task"] == task and len(result["curve"]) == 1
    assert ("accuracy" if task == "classify" else "product_top1") in result["metrics"]

    # the saved weights are scored again without training
    run_main(experiment, argv + ["--evaluate-only"], monkeypatch)
