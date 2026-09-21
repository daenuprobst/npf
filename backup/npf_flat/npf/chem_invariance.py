"""Is the Petri readout of the classifier load-bearing? A counterfactual test of a provable property.

Proposition (THEORY.md, P5). If the readout of a place depends on its R-ball only, the state-equation readout
    r = sum_B phi(place) - sum_A phi(place)
does not depend on anything further than R bonds from the places a firing changes: the contributions of an atom
and of its partner cancel exactly when their R-balls are untouched. So if the *same* substituent is attached on both
sides of the reaction at a site more than 2R bonds from everything that changes, r - and with it every logit - stays
exactly the same (6e-14 in double precision). A readout that pools the two sides separately has no such invariance.

    uv run python -m npf.chem_invariance        # writes results/chem/insights_invariance.json

Test: add a methyl group to a remote C-H site of test reactions (precursor and product consistently) and compare the
logits and the predicted class before and after, for every trained classifier found in results/chem/classify.
"""
import copy
import json
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.csgraph import shortest_path

from . import chem_data, chem_models
from .chem_experiment import batches, splits

RADIUS = 3 + 1  # locality radius of the readout: 3 message-passing rounds, and the bond places read one hop further
N_ELEM, N_DEG, N_CHARGE, N_H = len(chem_data.ELEMENTS) + 1, 6, 4, 5


def one_hot_slice(x, start, size, index):
    x[start:start + size] = 0
    x[start + min(index, size - 1)] = 1


def add_methyl(g, atom):
    """Graph with a CH3 attached to `atom` (which must carry a hydrogen)."""
    g = copy.deepcopy(g)
    n = len(g["element"])
    x = g["x"][atom].copy()
    degree = int((chem_data.dense_bonds(g)[atom] > 0).sum())
    one_hot_slice(x, N_ELEM, N_DEG, degree + 1)
    one_hot_slice(x, N_ELEM + N_DEG + N_CHARGE, N_H, int(g["h"][atom]) - 1)
    g["x"][atom] = x
    methyl = np.zeros_like(x)
    one_hot_slice(methyl, 0, N_ELEM, chem_data.ELEMENTS.index("C"))
    one_hot_slice(methyl, N_ELEM, N_DEG, 1)
    one_hot_slice(methyl, N_ELEM + N_DEG, N_CHARGE, 1)  # neutral
    one_hot_slice(methyl, N_ELEM + N_DEG + N_CHARGE, N_H, 3)
    g["x"] = np.vstack([g["x"], methyl])
    g["element"] = np.append(g["element"], 6).astype(g["element"].dtype)
    g["h"] = np.append(g["h"], 3).astype(g["h"].dtype); g["h"][atom] -= 1
    g["q"] = np.append(g["q"], 0).astype(g["q"].dtype)
    g["bonds"] = np.vstack([g["bonds"].reshape(-1, 3), [atom, n, 1]]).astype(np.int16)
    g["fragment"] = np.append(g["fragment"], g["fragment"][atom]).astype(g["fragment"].dtype)
    for key in ("symmetry", "skeleton"):
        g[key] = np.append(g[key], g[key].max() + 1).astype(g[key].dtype)
    return g


def remote_site(r, min_distance):
    """A product carbon with a hydrogen whose partner is at least `min_distance` bonds from every atom of the reaction
    centre (measured on both sides), or None."""
    # everything a firing changes: atoms of re-typed bonds, atoms whose own attributes differ from their partner's
    # (hydrogens, charge, ring membership, aromaticity) and precursor atoms that leave
    differs = np.nonzero((r["b"]["x"] != r["a"]["x"][r["target"]]).any(1))[0]
    kept = np.zeros(len(r["a"]["x"]), bool); kept[r["target"]] = True
    leaving_neighbours = np.nonzero((chem_data.dense_bonds(r["a"])[~kept] > 0).any(0) & kept)[0]
    centre = np.unique(np.concatenate([r["edits"][:, :2].ravel(), r["target"][differs], leaving_neighbours])).astype(int)
    if not len(centre):
        return None
    dist_a = shortest_path(chem_data.dense_bonds(r["a"]) > 0, unweighted=True)[:, centre].min(1)
    inverse = {int(j): i for i, j in enumerate(r["target"])}
    centre_b = [inverse[int(c)] for c in centre if int(c) in inverse]
    if not centre_b:
        return None
    dist_b = shortest_path(chem_data.dense_bonds(r["b"]) > 0, unweighted=True)[:, centre_b].min(1)
    for i, j in enumerate(r["target"]):
        if r["b"]["element"][i] == 6 and r["b"]["h"][i] >= 1 and r["a"]["h"][j] >= 1 and dist_a[j] >= min_distance and dist_b[i] >= min_distance \
                and np.isfinite(dist_a[j]) and np.isfinite(dist_b[i]):
            return i, int(j)
    return None


@torch.no_grad()
def main(root="results/chem", n=4000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = chem_data.load()
    _, _, test = splits(data, "classify")
    pairs = []
    for r in test:
        if r["target"] is None or len(r["a"]["x"]) > 150:
            continue
        site = remote_site(r, 2 * RADIUS + 1)
        if site is not None:
            twin = dict(r, a=add_methyl(r["a"], site[1]), b=add_methyl(r["b"], site[0]),
                        target=np.append(r["target"], len(r["a"]["x"])).astype(r["target"].dtype))  # the new atoms are partners
            pairs.append((r, twin))
        if len(pairs) >= n:
            break
    out = {"reactions_with_a_remote_site": len(pairs), "min_distance_bonds": 2 * RADIUS + 1}
    for path in sorted(Path(root, "classify").glob("*-[0-9].pt")):
        name = path.stem.rsplit("-", 1)[0]
        if name not in ("npf", "npf-nogate", "pgnn"):
            continue
        model = chem_models.Classifier(len(data["classes"]), petri=name != "pgnn", gate=name == "npf").to(device)
        model.load_state_dict(torch.load(path, map_location=device)); model.eval()
        logits = lambda rs: torch.cat([model(chem_data.collate(chunk, device)) for chunk in batches(rs, 64)])
        # `batches` sorts by size within chunks, so keep an explicit order
        order = lambda rs: [r for chunk in batches(rs, 64) for r in chunk]
        before, after = logits([p[0] for p in pairs]), logits([p[1] for p in pairs])
        ids_before, ids_after = [r["id"] for r in order([p[0] for p in pairs])], [r["id"] for r in order([p[1] for p in pairs])]
        position = {k: i for i, k in enumerate(ids_after)}
        after = after[[position[k] for k in ids_before]]
        entry = out.setdefault(name, {"max_logit_change": [], "mean_logit_change": [], "predictions_changed": []})
        entry["max_logit_change"].append(float((before - after).abs().max()))
        entry["mean_logit_change"].append(float((before - after).abs().mean()))
        entry["predictions_changed"].append(float((before.argmax(1) != after.argmax(1)).float().mean()))
    Path(root, "insights_invariance.json").write_text(json.dumps(out, indent=1))
    print(f"{len(pairs)} test reactions with a C-H site at least {2 * RADIUS + 1} bonds from the reaction centre")
    for name, entry in out.items():
        if isinstance(entry, dict):
            print(f"{name:>11}: max |logit change| {np.mean(entry['max_logit_change']):.2e}   mean {np.mean(entry['mean_logit_change']):.2e}   "
                  f"predictions changed {np.mean(entry['predictions_changed']):.4%}   ({len(entry['predictions_changed'])} models)")


if __name__ == "__main__":
    main()
