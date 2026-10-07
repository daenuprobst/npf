"""Average the class probabilities of all trained NPF classifiers (results/chem/classify/npf-<seed>.pt).

uv run python -m benchmarks.chemistry.ensemble
"""

import json
from pathlib import Path

import numpy as np
import torch

from npflow import chem

from .experiment import batches, splits, use_predicted_firing


@torch.no_grad()
def main(kind="npf", root="results/chem/classify"):
    """kind = npf, state-equation readout, kind = npf-sigma, explicit firing vector from the maps of the exact mapper."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = chem.load()
    train, val, test = splits(data, "classify")
    sigma = kind == "npf-sigma"

    # the classifiers read the maps of the exact mapper, as in benchmarks.chemistry.experiment
    use_predicted_firing(train, val, test, blank_missing=True)

    labels = np.array([r["label"] for rs in batches(test, 128) for r in rs])
    probs = []
    for path in sorted(Path(root).glob(f"{kind}-[0-9].pt")):
        model = (
            chem.Classifier(len(data["classes"]), gate=False, explicit_firing=True)
            if sigma
            else chem.Classifier(len(data["classes"]))
        ).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        probs.append(
            np.concatenate(
                [
                    torch.softmax(model(chem.collate(rs, device)), -1).cpu().numpy()
                    for rs in batches(test, 128)
                ]
            )
        )
        print(
            f"{path.name}: accuracy {(probs[-1].argmax(1) == labels).mean():.4f}",
            flush=True,
        )

    for k in range(2, len(probs) + 1):
        print(
            f"ensemble of the first {k}: accuracy {(np.mean(probs[:k], 0).argmax(1) == labels).mean():.4f}"
        )

    accuracy = float((np.mean(probs, 0).argmax(1) == labels).mean())
    (Path(root) / f"{kind}-ensemble.json").write_text(
        json.dumps({"members": len(probs), "accuracy": accuracy})
    )


if __name__ == "__main__":
    import sys

    main(*sys.argv[1:])
