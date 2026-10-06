"""The API on a few Schneider reactions written the way a user writes them, SMILES and a label per line, no maps."""
import numpy as np

from npflow.chem import api


def files(schneider, tmp_path):
    rs = [r for r in schneider["reactions"] if r["edits"] is not None][:24]
    lines = [f"{r['smiles']}\t{schneider['classes'][r['label']]}" for r in rs]
    for name, part in (("train", lines[:16]), ("val", lines[16:20]), ("test", lines[20:])):
        (tmp_path / f"{name}.txt").write_text("\n".join(part) + "\n")

    return rs, [str(tmp_path / f"{name}.txt") for name in ("train", "val", "test")]


def test_forward_model_trains_tests_and_predicts(schneider, tmp_path):
    rs, paths = files(schneider, tmp_path)
    data = api.ReactionData(*paths, batch=8, seconds=0.5, processes=2, workers=0)
    model = api.ForwardModel(width=32, rounds=2, attention=1, compile_events=False)
    trainer = api.trainer(model, epochs=1, root=str(tmp_path / "runs"), logger=False, enable_progress_bar=False)
    trainer.fit(model, data)
    metrics = trainer.test(model, data, verbose=False)[0]
    assert 0.0 <= metrics["test_top1"] <= 1.0 and 0.0 <= metrics["test_major"] <= 1.0

    # the cache makes a second data module instant, and it holds every reaction of the file
    assert (tmp_path / "train.txt.targets.pkl").exists()
    assert api.ReactionData(*paths, task="classify", workers=0).n_classes > 0

    out = api.predict(model, [r["smiles"].split(">>")[0] for r in rs[:3]] + ["not a smiles"], width=2)
    assert len(out) == 4 and out[3] is None
    assert all(isinstance(smiles, str) and 0.0 <= p <= 1.0 for candidates in out[:3] for smiles, p in candidates)


def test_classifier_model_trains_and_classifies(schneider, tmp_path):
    rs, paths = files(schneider, tmp_path)
    data = api.ReactionData(*paths, task="classify", batch=8, seconds=0.5, processes=2, workers=0)
    for sigma in (False, True):
        model = api.ClassifierModel(data.n_classes, sigma=sigma)
        trainer = api.trainer(model, epochs=1, root=str(tmp_path / "runs"), logger=False, enable_progress_bar=False)
        trainer.fit(model, data)
        assert 0.0 <= trainer.test(model, data, verbose=False)[0]["test_acc"] <= 1.0

    labels = api.classify(model, [r["smiles"] for r in rs[:2]] + ["not a smiles"], data.classes, seconds=0.5)
    assert labels[0] in data.classes and labels[2] is None


def test_map_reaction_writes_map_numbers(schneider):
    r = next(r for r in schneider["reactions"] if r["edits"] is not None and len(r["b"]["x"]) < 20)
    mapped = api.map_reaction(r["smiles"], seconds=2.0)
    assert mapped is not None and mapped.count(":") >= len(r["b"]["x"]) and ">>" in mapped

    # map numbers only, the molecules stay the same
    from rdkit import Chem
    molecules = lambda side: (Chem.MolFromSmiles(s) for s in side.split("."))
    plain = lambda side: ".".join(sorted(Chem.MolToSmiles(m) for m in molecules(side) if [a.SetAtomMapNum(0) for a in m.GetAtoms()] is not None))
    assert [plain(s) for s in mapped.split(">>")] == [plain(s) for s in r["smiles"].split(">>")]
    assert np.array_equal(np.sort([a.GetAtomMapNum() for a in Chem.MolFromSmiles(mapped.split(">>")[1]).GetAtoms()]), np.arange(1, len(r["b"]["x"]) + 1))
