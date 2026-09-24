"""What the Petri formulation gives beyond accuracy, on Schneider 50k with the trained models of
benchmarks.chemistry.experiment.

    uv run python -m benchmarks.chemistry.insights    # writes results/chem/insights_<section>.json and prints a summary

1  validity      Products are valence-valid by construction, the one-shot counterpart has no such guarantee.
2  by-products   The state equation conserves tokens, so the fragments that leave come with the prediction.
3  calibration   The probability of a product is that of its trace with all firing orders merged.
4  firing order  Enabling constrains the order of firings, what breaks before what forms.
5  attribution   Untouched places contribute exactly zero to the readout, attribution against the distance from
                 the reaction centre without a saliency method.
6  data audit    Recorded atom mappings that make and break more bonds than the minimum firing vector.
"""

import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.csgraph import shortest_path

from npf import chem
from npf.chem import cost

from .experiment import batches, splits

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path("results/chem")


def load(model, path):
    model.load_state_dict(torch.load(path, map_location=DEVICE))

    return model.to(DEVICE).eval()


@torch.no_grad()
def forward_insights(data, n=2000):
    _, _, test = splits(data, "forward")
    test = test[:n]
    game = load(chem.TokenGame(), ROOT / "forward/npf-nettargets-0.pt")
    out = {}
    one_shot_path = ROOT / "forward/pgnn-nettargets-matched-0.pt"

    # 1 validity of the one-shot counterpart
    if one_shot_path.exists():
        one_shot = load(
            chem.Forward(petri=False, d=128, rounds=6, attention=4), one_shot_path
        )
        invalid = total = 0

        for rs in batches(test, 64):
            b = chem.collate(rs, DEVICE)
            for r, edits in zip(rs, one_shot.decode(one_shot(b), b, rs)):
                before = chem.dense_bonds(r["a"])
                after = before.copy()
                after[edits[:, 0], edits[:, 1]] = edits[:, 2]
                after[edits[:, 1], edits[:, 0]] = edits[:, 2]
                left = r["a"]["h"] - (
                    chem.BOND_ORDER[after] - chem.BOND_ORDER[before]
                ).sum(1)
                cap = np.array(
                    [chem.EXTRA_CAPACITY.get(int(e), 0) for e in r["a"]["element"]]
                ) + np.maximum(-r["a"]["q"], 0)
                invalid += bool((left < -cap - 0.5).any())
                total += 1

        out["one_shot_valence_violations"] = invalid / total

    byproducts, order, confidence = defaultdict(Counter), Counter(), []
    beam_total = beam_valid = 0

    for r in test:
        b = chem.collate([r], DEVICE)
        want = chem.canonical_product(r["smiles"].split(">>")[1])
        ranked = game.beam_search(b, 5)
        if not ranked:
            continue

        edits, logp = ranked[0]
        products = chem.marking_to_products(r["a"], edits)

        # 3 calibration, traces that differ only by symmetry-equivalent atoms give the same product, their probabilities add
        mass = Counter()
        for e, lp in ranked:
            outcome = chem.marking_to_products(r["a"], e)
            beam_total += 1
            beam_valid += bool(outcome)
            mass[max(outcome, key=len) if outcome else None] += float(np.exp(lp))

        top, prob = mass.most_common(1)[0]
        confidence.append((min(prob, 1.0), top == want))

        # 2 byproducts, every other fragment that a firing touched
        if want in products:
            for frag in products - {want}:
                byproducts[data["classes"][r["label"]]][frag] += 1

        # 4 firing order of the greedy token game, is a bond at a saturated carbon broken before the new one forms?
        cur, fired = b["ba"].clone(), torch.zeros_like(b["ba"], dtype=torch.bool)
        sequence = []
        for _ in range(game.max_steps):
            logits, enabled, stop = game.rates(b, cur, fired)
            logits = logits.masked_fill(~enabled, -1e4)
            best, where = logits.flatten(1).max(1)
            if best.item() <= stop.item():
                break

            nn_ = cur.shape[1]
            i, j, k = (
                int(where) // (nn_ * chem.N_BOND),
                (int(where) // chem.N_BOND) % nn_,
                int(where) % chem.N_BOND,
            )
            gain = chem.BOND_ORDER[k] - chem.BOND_ORDER[int(cur[0, i, j])]
            sequence.append("form" if gain > 0 else "break")
            cur[0, i, j] = cur[0, j, i] = k
            fired[0, i, j] = fired[0, j, i] = True

        if "form" in sequence and "break" in sequence:
            order[
                (
                    "break first"
                    if sequence.index("break") < sequence.index("form")
                    else "form first"
                )
            ] += 1

    conf = np.array(confidence)
    bins = np.clip((conf[:, 0] * 10).astype(int), 0, 9)
    out["calibration"] = [
        {
            "confidence": float(conf[bins == k, 0].mean()),
            "accuracy": float(conf[bins == k, 1].mean()),
            "n": int((bins == k).sum()),
        }
        for k in range(10)
        if (bins == k).any()
    ]
    out["expected_calibration_error"] = float(
        sum(abs(c["confidence"] - c["accuracy"]) * c["n"] for c in out["calibration"])
        / len(conf)
    )
    out["beam_candidates_that_are_valid_molecules"] = beam_valid / max(beam_total, 1)
    out["byproducts_by_class"] = {
        c: v.most_common(3) for c, v in sorted(byproducts.items())
    }
    out["firing_order"] = dict(order)

    return out


@torch.no_grad()
def attribution_insights(data, n=3000):
    """Per-atom contribution to the state-equation readout, phi(atom in B) - phi(its partner in A), against the
    graph distance of the atom from the nearest re-typed bond. Beyond rounds bonds it must be exactly zero.
    """
    _, _, test = splits(data, "classify")
    test = [r for r in test if r["target"] is not None and len(r["edits"])][:n]
    clf = load(chem.Classifier(len(data["classes"])), ROOT / "classify/npf-0.pt")
    by_distance = defaultdict(list)
    for rs in batches(test, 64):
        b = chem.collate(rs, DEVICE)
        ha = clf.encoder(b["xa"], b["ba"], b["mask_a"], all_depths=True)
        hb = clf.encoder(b["xb"], b["bb"], b["mask_b"], all_depths=True)
        pa = torch.cat([f(h) for f, h in zip(clf.atom, ha)], -1)
        pb = torch.cat([f(h) for f, h in zip(clf.atom, hb)], -1)
        for k, r in enumerate(rs):
            m = len(r["b"]["x"])
            delta = (
                (
                    pb[k, :m]
                    - pa[
                        k, torch.as_tensor(r["target"].astype(np.int64), device=DEVICE)
                    ]
                )
                .norm(dim=-1)
                .cpu()
                .numpy()
            )
            centre = np.unique(r["edits"][:, :2])
            dist = shortest_path(chem.dense_bonds(r["a"]) > 0, unweighted=True)[
                :, centre
            ].min(1)[r["target"]]
            for d, x in zip(dist, delta):
                by_distance[int(min(d, 8))].append(float(x))

    return {
        "attribution_norm_by_distance_to_reaction_centre": {
            d: {
                "mean": float(np.mean(v)),
                "max": float(np.max(v)),
                "share_exactly_zero": float(np.mean(np.array(v) < 1e-5)),
                "n": len(v),
            }
            for d, v in sorted(by_distance.items())
        }
    }


def mapping_audit(data, maps="data/exact_maps_schneider50k.pkl"):
    """Test reactions whose recorded mapping makes and breaks more bonds than the minimum firing vector of the mapper,
    the first level of its cost (chem.cost)."""
    _, _, test = splits(data, "forward")
    exact_maps, more, examples = pickle.loads(Path(maps).read_bytes()), Counter(), []
    for r in test:
        # a map that seats a product atom on no written precursor atom, from the open net, has no cost on this net
        if (
            r["id"] not in exact_maps
            or len(exact_maps[r["id"]]) != len(r["b"]["x"])
            or (exact_maps[r["id"]] < 0).any()
        ):
            continue

        ours, recorded = (
            cost.cost_levels(r, m.astype(np.int64), **cost.CHOSEN)[0]
            for m in (exact_maps[r["id"]], r["target"])
        )
        if recorded > ours:
            more[data["classes"][r["label"]]] += 1
            examples.append(
                {
                    "reaction": r["smiles"],
                    "class": data["classes"][r["label"]],
                    "places_exact": ours,
                    "places_recorded": recorded,
                }
            )

    examples.sort(key=lambda e: e["places_exact"] - e["places_recorded"])

    return {
        "recorded_mapping_changes_more_places": sum(more.values()) / len(test),
        "by_class": more.most_common(8),
        "examples": examples[:10],
    }


def main():
    data = chem.load()
    out = {}
    for name, fn in (
        ("forward", forward_insights),
        ("attribution", attribution_insights),
        ("mapping_audit", mapping_audit),
    ):
        try:
            out[name] = fn(data)
        except FileNotFoundError as missing:
            print(f"skipping {name}: {missing}")
            continue

        # one file per section, the report and the figures read them
        (ROOT / f"insights_{name}.json").write_text(json.dumps(out[name], indent=1))

    print(json.dumps(out, indent=1)[:6000])


if __name__ == "__main__":
    main()
