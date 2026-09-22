"""How much is left in the search on Schneider 50k? Greedy against beams of growing width, and whether the recorded firing vector is
in the beam at all. If widening stops helping, the mode of the model is reached and only the model can improve.

    uv run python -m benchmarks.chemistry.beam_width    # results/chem/search/beam_width.log
"""

import time

import numpy as np
import torch

from npf import chem

from .experiment import batches, product_found, splits


def main():
    torch.set_num_threads(8)
    data = chem.load("data/schneider50k.pkl")
    _, _, test = splits(data, "forward")
    sample = [
        test[i] for i in np.random.default_rng(0).choice(len(test), 600, replace=False)
    ]
    model = chem.TokenGame()
    model.load_state_dict(
        torch.load("results/chem/forward/npf-nettargets-0.pt", map_location="cpu")
    )
    model.eval()

    with torch.no_grad():
        hits = 0

        for rs in batches(sample, 32):
            b = chem.collate(rs, "cpu")
            for r, pred in zip(rs, model.decode(model(b), b, rs)):
                hits += product_found(r, pred)

        print(f"greedy            top-1 {hits / len(sample):.4f}", flush=True)

        for width in (5, 20, 64):
            start, top1, oracle, in_beam = time.time(), 0, 0, 0

            for r in sample:
                ranked = model.beam_search(chem.collate([r], "cpu"), width)
                found = [product_found(r, edits) for edits, _ in ranked]
                top1 += found[0] if found else 0
                oracle += any(found)

                # is the recorded firing vector among the hypotheses at all?
                if r["edits"] is not None:
                    true = {tuple(e) for e in r["edits"].tolist()}
                    in_beam += any(
                        {tuple(e) for e in edits.tolist()} == true
                        for edits, _ in ranked
                    )

            n = len(sample)
            print(
                f"beam width {width:3d}  top-1 {top1 / n:.4f}   product in beam {oracle / n:.4f}   recorded firing vector in beam {in_beam / n:.4f}"
                f"   ({time.time() - start:.0f}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
