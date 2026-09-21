"""The exact net mapper against RXNMapper on the same reactions of the Golden set.

Reads the rows that benchmarks.chemistry.exact_map wrote and the output of RXNMapper, and reports both on all usable
reactions and on the part that played no role in the choice of the cost, with a bootstrap interval and a paired test.

    uv run python -m benchmarks.chemistry.exact_map_report results/chem/exact_map/golden-1-1-1-nolabile-ch.pkl
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from npf.chem import exact

from .exact_map import golden_dev, load
from .golden import mapping_from_smiles, same_cgr


def interval(hits, rng, draws=10000):
    means = hits[rng.integers(0, len(hits), (draws, len(hits)))].mean(1)

    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


BINS = ((1, 2), (3, 4), (5, 6), (7, None))


def main(rows_file, rxnmapper_json="results/chem/rxnmapper_golden.json"):
    reactions = load("golden")
    rows = pickle.loads(Path(rows_file).read_bytes())
    summary = json.loads(Path(rows_file).with_suffix(".json").read_text())
    options = dict(secondary=tuple(summary["secondary"]), all_orders=summary["all_orders"], labile_h=summary["labile_h"], ch_places=summary["ch_places"])
    assert [r["id"] for r in reactions] == [x["id"] for x in rows]
    mapped = json.loads(Path(rxnmapper_json).read_text())
    ours = np.array([x["correct"] for x in rows])
    theirs = []
    for r in reactions:
        m = mapping_from_smiles(r, mapped[str(r["id"])]["mapped_rxn"]) if str(r["id"]) in mapped else None
        theirs.append(bool(m is not None and len(m) == len(r["target"]) and same_cgr(r, m, r["target"].astype(np.int64))))

    theirs = np.array(theirs)
    held_out = np.ones(len(rows), bool)
    held_out[golden_dev(len(rows))] = False
    rng, report = np.random.default_rng(0), {"rows": str(rows_file)}
    for name, part in (("all", np.ones(len(rows), bool)), ("held_out", held_out), ("dev", ~held_out)):
        a, b = ours[part], theirs[part]

        # reactions that only one of the two maps correctly decide the paired test
        only_ours, only_theirs = int((a & ~b).sum()), int((~a & b).sum())
        report[name] = {"reactions": int(part.sum()), "exact_net_mapper": float(a.mean()), "exact_net_mapper_ci95": interval(a, rng),
                        "rxnmapper": float(b.mean()), "rxnmapper_ci95": interval(b, rng), "either": float((a | b).mean()),
                        "only_ours": only_ours, "only_rxnmapper": only_theirs,
                        "sign_test_p": float(binomtest(only_ours, only_ours + only_theirs, 0.5).pvalue) if only_ours + only_theirs else 1.0,
                        "proved": float(np.mean([x["proved"] for x, p in zip(rows, part) if p])),
                        "exact_net_mapper_when_proved": float(np.mean([x["correct"] for x, p in zip(rows, part) if p and x["proved"]]))}

    # by the number of places that the curated map changes, a proxy for one-pot and multi-step reactions
    curated = np.array([exact.cost_levels(r, r["target"].astype(np.int64), **options)[0] for r in reactions])
    found = np.array([exact.cost_levels(r, x["mapping"], **options)[0] for r, x in zip(reactions, rows)])
    proved = np.array([x["proved"] for x in rows])
    report["by_places"] = []

    for low, high in BINS:
        part = (curated >= low) & (curated <= (high or curated.max()))
        report["by_places"].append({"places": f"{low} to {high}" if high else f"{low} or more", "reactions": int(part.sum()),
                                    "exact_net_mapper": float(ours[part].mean()), "rxnmapper": float(theirs[part].mean()),
                                    "curated_not_minimal": float((curated > found)[part].mean()), "proved": float(proved[part].mean())})

    Path(rows_file).with_suffix(".report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
