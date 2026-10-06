"""Where the greedy token game fails. Every wrong prediction is classed by comparing its fired bond places with the
recorded ones, and accuracy is broken down by the number of recorded firings and of precursor atoms.

    uv run python -m benchmarks.chemistry.errors results/uspto_mit/forward/npf-deep-0.pt --dataset uspto_mit --width 256 --rounds 8 --attention 8
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from npflow import chem

from .experiment import batches, product_found, splits, symmetric


def error_class(reaction, pred, true):
    """One label for a wrong prediction. Bond places are compared up to the symmetry classes of the precursors."""
    a, before = reaction["a"], chem.dense_bonds(reaction["a"])
    classes = a["symmetry"]
    place = lambda e: Counter(
        (min(classes[i], classes[j]), max(classes[i], classes[j])) for i, j in e[:, :2]
    )
    fired_pred, fired_true = place(pred), place(true)

    if symmetric(
        a, pred[:, 0], pred[:, 1], before[pred[:, 0], pred[:, 1]], pred[:, 2]
    ) == symmetric(
        a, true[:, 0], true[:, 1], before[true[:, 0], true[:, 1]], true[:, 2]
    ):
        return "recorded firings, other product after decoding"

    if len(pred) == 0:
        return "nothing fired"

    if fired_pred == fired_true:
        return "recorded places, other bond types"

    if not (fired_pred - fired_true):
        return "stopped early"

    if not (fired_true - fired_pred):
        return "stopped late"

    if fired_pred & fired_true:
        return "overlaps the recorded places"

    shared_atoms = {classes[i] for i in pred[:, :2].ravel()} & {
        classes[i] for i in true[:, :2].ravel()
    }

    return "other places, shares an atom" if shared_atoms else "other site"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--dataset", default="schneider50k")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--attention", type=int, default=4)
    ap.add_argument(
        "--limit", type=int, help="only the first reactions of the test set"
    )
    ap.add_argument("--split", default="test", choices=["val", "test"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = chem.load(f"data/{args.dataset}.pkl")
    _, val, test = splits(data, "forward")
    reactions = (test if args.split == "test" else val)[: args.limit]
    model = chem.TokenGame(
        d=args.width, rounds=args.rounds, attention=args.attention
    ).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    kinds, by_firings, by_size = Counter(), {}, {}
    for rs in batches(reactions, 64):
        b = chem.collate(rs, device)
        for r, pred in zip(rs, model.decode(model(b), b, rs)):
            true = r["edits"] if r["edits"] is not None else np.zeros((0, 3), np.int64)
            ok = product_found(r, pred)
            n_firings, n_atoms = min(len(true), 6), min(len(r["a"]["x"]) // 20, 5)
            by_firings.setdefault(n_firings, []).append(ok)
            by_size.setdefault(n_atoms, []).append(ok)

            if ok:
                kinds["correct"] += 1
            elif r["edits"] is None:
                kinds["no recorded firings"] += 1
            elif not product_found(r, true):
                kinds["recorded firings do not decode to the recorded product"] += 1
            else:
                kinds[error_class(r, pred, true)] += 1

    n = len(reactions)
    print(f"{n} reactions, {kinds['correct'] / n:.2%} correct")

    for kind, count in kinds.most_common():
        print(f"  {kind:60s} {count:6d}  {count / n:7.2%}")

    print("accuracy by number of recorded firings (6 = six or more)")

    for k in sorted(by_firings):
        print(f"  {k}: {np.mean(by_firings[k]):.2%}  ({len(by_firings[k])} reactions)")

    print("accuracy by number of precursor atoms (bins of 20, last bin open)")

    for k in sorted(by_size):
        print(
            f"  {20 * k:3d}+: {np.mean(by_size[k]):.2%}  ({len(by_size[k])} reactions)"
        )

    # a folder of its own, the report reads every json file next to the weights as a result of a run
    out = (
        Path(args.weights).parent.parent
        / "errors"
        / (Path(args.weights).stem + ".json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "n": n,
                "kinds": dict(kinds),
                "by_firings": {k: float(np.mean(v)) for k, v in by_firings.items()},
                "by_size": {k: float(np.mean(v)) for k, v in by_size.items()},
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
