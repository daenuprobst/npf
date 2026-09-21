"""Facts about the data that the paper quotes in its text, written to results/data_facts.json so each has a source.

    uv run python -m benchmarks.checks.data_facts
"""
import json
import pickle
from pathlib import Path

import numpy as np

from npf import chem
from npf.chem import open_net
from npf.chem.orders import slack

from benchmarks.chemistry.experiment import product_found, splits

SAMPLE = 5000


def balance_facts(reactions):
    count = lambda g: np.bincount(g["element"].astype(np.int64), minlength=120)
    surplus = [bool((count(r["a"]) > count(r["b"])).any()) for r in reactions]
    lacking = [bool(open_net.deficit(r).any()) for r in reactions]

    return {"reactions": len(reactions), "precursor_atoms_leave": float(np.mean(surplus)), "product_atoms_lack_a_precursor": float(np.mean(lacking)),
            "lacking": int(np.sum(lacking))}


def unreachable_training_states(reactions, rng):
    """Share of the random parts of a recorded firing vector, drawn as in the loss of the token game, that leave an
    atom with fewer free valence tokens than none, or than the half token of tolerance."""
    below_zero, below_tolerance = [], []
    for r in reactions:
        edits = r["edits"]
        fired = edits[rng.random(len(edits)) < rng.random()]
        bonds = chem.dense_bonds(r["a"]).copy()
        for i, j, t in fired:
            bonds[i, j] = bonds[j, i] = t

        free = slack(r, bonds)
        below_zero.append(bool((free < 0).any()))
        below_tolerance.append(bool((free < -0.5).any()))

    return {"states": len(reactions), "negative_slack": float(np.mean(below_zero)), "below_half_token": float(np.mean(below_tolerance))}


def main(out="results/data_facts.json"):
    rng = np.random.default_rng(0)
    schneider = chem.load("data/schneider50k.pkl")
    usable = [r for r in schneider["reactions"] if r["target"] is not None and r["edits"] is not None]
    train, _, _ = splits(schneider, "forward")
    sample = [train[i] for i in rng.choice(len(train), min(SAMPLE, len(train)), replace=False)]
    golden = pickle.loads(Path("data/golden.pkl").read_bytes())["reactions"]
    candidates = pickle.loads(Path("data/candidates.pkl").read_bytes())
    facts = {
        "schneider50k": balance_facts(schneider["reactions"]),
        "golden": balance_facts(golden) | {"with_a_complete_curated_map": sum(r["target"] is not None and len(r["b"]["x"]) <= len(r["a"]["x"]) for r in golden)},
        "schneider50k_usable": {"reactions": len(usable), "mean_recorded_firings": float(np.mean([len(r["edits"]) for r in usable])),
                                "product_atoms_with_conserved_valence": float(np.mean(np.concatenate([r["conserved"] for r in usable])))},
        "schneider50k_training_sample": {"reactions": len(sample), "recorded_firings_decode_to_the_product": float(np.mean([product_found(r, r["edits"]) for r in sample]))}
        | unreachable_training_states([r for r in sample if len(r["edits"])], rng),
        "candidates": {"reactions": len(candidates), "cheapest_firing_vector_unique": float(np.mean([len(c["maps"]) == 1 for c in candidates])),
                       "search_exhaustive": float(np.mean([bool(c["exact"]) for c in candidates]))},
    }
    Path(out).write_text(json.dumps(facts, indent=1))
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
