"""Average the class probabilities of all trained NPF classifiers (results/chem/classify/npf-<seed>.pt).

    uv run python -m npf.chem_ensemble
"""
import json
from pathlib import Path

import numpy as np
import torch

from . import chem_data, chem_models
from .chem_experiment import batches, splits


@torch.no_grad()
def main(kind="npf", root="results/chem/classify"):
    """kind = npf: state-equation readout; kind = npf-sigma: explicit firing vector from our own (clean) mapper."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = chem_data.load()
    train, val, test = splits(data, "classify")
    sigma = kind == "npf-sigma"
    if sigma:
        from .chem_experiment import use_predicted_firing
        use_predicted_firing(train, val, test)
    labels = np.array([r["label"] for rs in batches(test, 128) for r in rs])
    probs = []
    for path in sorted(Path(root).glob(f"{kind}-[0-9].pt")):
        model = (chem_models.Classifier(len(data["classes"]), gate=False, explicit_firing=True) if sigma else chem_models.Classifier(len(data["classes"]))).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        probs.append(np.concatenate([torch.softmax(model(chem_data.collate(rs, device)), -1).cpu().numpy() for rs in batches(test, 128)]))
        print(f"{path.name}: accuracy {(probs[-1].argmax(1) == labels).mean():.4f}", flush=True)
    for k in range(2, len(probs) + 1):
        print(f"ensemble of the first {k}: accuracy {(np.mean(probs[:k], 0).argmax(1) == labels).mean():.4f}")
    accuracy = float((np.mean(probs, 0).argmax(1) == labels).mean())
    (Path(root) / f"{kind}-ensemble.json").write_text(json.dumps({"members": len(probs), "accuracy": accuracy}))


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
