"""Facts about the data that the paper quotes in its text, written to results/data_facts.json so each has a source.

uv run python -m benchmarks.checks.data_facts
"""

import json
import pickle
from pathlib import Path

import numpy as np

from benchmarks.chemistry.experiment import product_found, splits
from npf import chem
from npf.chem import open_net
from npf.chem.orders import slack

SAMPLE = 5000


def balance_facts(reactions):
    count = lambda g: np.bincount(g["element"].astype(np.int64), minlength=120)
    surplus = [bool((count(r["a"]) > count(r["b"])).any()) for r in reactions]
    lacking = [bool(open_net.deficit(r).any()) for r in reactions]

    return {
        "reactions": len(reactions),
        "precursor_atoms_leave": float(np.mean(surplus)),
        "product_atoms_lack_a_precursor": float(np.mean(lacking)),
        "lacking": int(np.sum(lacking)),
    }


def training_states(reactions, vectors, rng):
    """Share of the random parts of a training firing vector, drawn as in the loss of the token game, that leave an
    atom with fewer free valence tokens than none, or than the half token of tolerance.
    """
    below_zero, below_tolerance = [], []
    for r in reactions:
        options = vectors(r)
        edits = np.asarray(options[rng.integers(len(options))])
        fired = edits[rng.random(len(edits)) < rng.random()]
        bonds = chem.dense_bonds(r["a"]).copy()

        for i, j, t in fired:
            bonds[i, j] = bonds[j, i] = t

        free = slack(r, bonds)
        below_zero.append(bool((free < 0).any()))
        below_tolerance.append(bool((free < -0.5).any()))

    return {
        "states": len(reactions),
        "negative_slack": float(np.mean(below_zero)),
        "below_half_token": float(np.mean(below_tolerance)),
    }


def main(out="results/data_facts.json"):
    rng = np.random.default_rng(0)
    schneider = chem.load("data/schneider50k.pkl")
    usable = [
        r
        for r in schneider["reactions"]
        if r["target"] is not None and r["edits"] is not None
    ]
    train = splits(schneider, "forward", recorded=False)[0]
    golden = pickle.loads(Path("data/golden.pkl").read_bytes())["reactions"]
    facts = {
        "schneider50k": balance_facts(schneider["reactions"]),
        "golden": balance_facts(golden)
        | {
            "with_a_complete_curated_map": sum(
                r["target"] is not None and len(r["b"]["x"]) <= len(r["a"]["x"])
                for r in golden
            )
        },
        "schneider50k_usable": {
            "reactions": len(usable),
            "mean_recorded_firings": float(np.mean([len(r["edits"]) for r in usable])),
            "product_atoms_with_conserved_valence": float(
                np.mean(np.concatenate([r["conserved"] for r in usable]))
            ),
        },
    }

    # training states of the token game, from the targets of the net where they exist, else from the recorded vectors
    targets = Path("data/net_targets_schneider50k.pkl")
    if targets.exists():
        vectors = pickle.loads(targets.read_bytes())["targets"]
        pool = [r for r in train if r["id"] in vectors]
        source = lambda r: vectors[r["id"]]
    else:
        pool = [r for r in train if r["edits"] is not None and len(r["edits"])]
        source = lambda r: [r["edits"]]

    sample = [
        pool[i] for i in rng.choice(len(pool), min(SAMPLE, len(pool)), replace=False)
    ]
    recorded = [r for r in sample if r["edits"] is not None]
    facts["schneider50k_training_sample"] = {
        "reactions": len(sample),
        "targets": "net" if targets.exists() else "recorded",
        "recorded_firings_decode_to_the_product": float(
            np.mean([product_found(r, r["edits"]) for r in recorded])
        ),
        "mean_firings": float(np.mean([len(source(r)[0]) for r in sample])),
    } | training_states(sample, source, rng)

    # USPTO-MIT, the official test lines and what the size caps and the recorded maps leave
    mit = Path("data/uspto_mit.pkl")
    if mit.exists():
        reactions = chem.load(str(mit))["reactions"]
        small = (
            lambda r: len(r["a"]["x"]) <= chem.MAX_PRECURSOR_ATOMS
            and len(r["b"]["x"]) <= chem.MAX_PRODUCT_ATOMS
        )
        test = [r for r in reactions if r["split"] == "test"]
        mit_train = [r for r in reactions if r["split"] == "train"]
        facts["uspto_mit"] = {
            "test_lines": 40000,
            "test_parsed": len(test),
            "test_without_usable_recorded_map": sum(r["target"] is None for r in test),
            "train_parsed": len(mit_train),
            "train_within_size_caps": sum(small(r) for r in mit_train),
            "train_with_usable_recorded_map": sum(
                r["target"] is not None for r in mit_train
            ),
        }

    Path(out).write_text(json.dumps(facts, indent=1))
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
