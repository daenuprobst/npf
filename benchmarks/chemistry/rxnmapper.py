"""Score the mappings of RXNMapper on the Schneider test reactions with the metrics used for our mapper.

RXNMapper (Schwaller et al., Sci. Adv. 2021) is pretrained without labels. Its output is produced in a separate
environment and stored in results/chem/rxnmapper_test.json.

    uv run python -m benchmarks.chemistry.rxnmapper
"""
import json
from collections import Counter
from pathlib import Path


from npf import chem
from .experiment import firing, splits, tokens_moved


def as_reaction(mapped_rxn):
    """Our reaction record (graphs, target) built from a mapped reaction SMILES."""
    row = {"original_rxn": mapped_rxn.replace(">>", ">>"), "label": 0, "split": "test", "id": -1, "rxn": ""}
    precursors, product = mapped_rxn.split(">>")
    row["original_rxn"] = f"{precursors}>>{product}"

    return chem.featurise(row)


def main(root="results/chem"):
    data = chem.load()
    _, _, test = splits(data, "map")
    theirs = json.loads((Path(root) / "rxnmapper_test.json").read_text())
    stats = Counter()
    for r in test:
        stats["n"] += 1
        entry = theirs.get(str(r["id"]))
        other = as_reaction(entry["mapped_rxn"]) if entry and entry["mapped_rxn"] else None
        if other is None or other["target"] is None:
            stats["no_complete_mapping"] += 1
            continue

        same = firing(other, other["target"]) == firing(r, r["target"])
        stats["same_firing"] += same

        if not same:
            a, b = tokens_moved(other, other["target"]), tokens_moved(r, r["target"])
            stats["fewer" if a < b else "equal" if a == b else "more"] += 1

    n = stats["n"]
    out = {"reactions_same_firing_vector": stats["same_firing"] / n, "no_complete_mapping": stats["no_complete_mapping"] / n,
           "disagree_ours_moves_fewer_tokens": stats["fewer"] / n, "disagree_equal": stats["equal"] / n,
           "disagree_ours_moves_more_tokens": stats["more"] / n, "n": n}
    (Path(root) / "map" / "rxnmapper.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    main()
