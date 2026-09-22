"""The mappings that the net cannot tell apart. Every optimal mapping of a Golden reaction is enumerated, mappings with
the same condensed graph of reaction are merged, and the classes that remain are the true ties.

Reported are the share of reactions whose curated map is among the optima, which no tie-breaker can exceed, and the
accuracy of a uniform choice among the classes, which is what the net alone is worth when the arbitrary choice of the
solver is taken out.

    uv run python -m benchmarks.chemistry.exact_ties               # results/chem/exact_map/ties-golden.json
"""

import argparse
import json
import pickle
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import numpy as np

from npf import chem
from npf.chem import exact

from .exact_map import golden_dev, load
from .golden import cgr, same_cgr


def key(reaction, mapping):
    """A cheap invariant of the condensed graph, equal for isomorphic graphs."""
    g = cgr(reaction, mapping)
    for _, data in g.nodes(data=True):
        data["text"] = str(data["label"])

    for _, _, data in g.edges(data=True):
        data["text"] = str(data["label"])

    return nx.weisfeiler_lehman_graph_hash(
        g, node_attr="text", edge_attr="text", iterations=3
    )


def work(job):
    reaction, seconds, limit = job

    try:
        hint = chem.feasible_start(reaction)
    except Exception:
        hint = None

    # a budget in deterministic time and one worker, so the ties do not depend on the load of the machine
    try:
        maps, proved = exact.cheapest_mappings(
            reaction,
            limit=limit,
            seconds=4 * seconds,
            hint=hint,
            deterministic=seconds,
            workers=4,
            **exact.CHOSEN,
        )
    except Exception:
        maps, proved = [], False

    if not proved or not maps:
        return {"id": reaction["id"], "enumerated": False}

    classes = []
    for m in maps:
        k = key(reaction, m)
        if not any(
            k == other and same_cgr(reaction, m, first) for other, first in classes
        ):
            classes.append((k, m))

    target = reaction["target"].astype(np.int64)
    right = [bool(same_cgr(reaction, m, target)) for _, m in classes]

    return {
        "id": reaction["id"],
        "enumerated": True,
        "complete": len(maps) < limit,
        "mappings": len(maps),
        "classes": [m.astype(np.int16) for _, m in classes],
        "right": right,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="golden")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--limit", type=int, default=512)
    ap.add_argument("--processes", type=int, default=6)
    args = ap.parse_args()
    reactions = load(args.data)

    with Pool(args.processes) as pool:
        rows = pool.map(
            work, [(r, args.seconds, args.limit) for r in reactions], chunksize=1
        )

    done = [x for x in rows if x["enumerated"]]
    tied = [x for x in done if len(x["classes"]) > 1]
    held_out = np.ones(len(rows), bool)

    if args.data == "golden":
        held_out[golden_dev(len(rows))] = False

    share = lambda xs: float(np.mean(xs)) if len(xs) else None
    summary = {
        "data": args.data,
        "reactions": len(rows),
        "enumerated": len(done) / len(rows),
        "complete": share([x["complete"] for x in done]),
        "mappings_per_reaction": share([x["mappings"] for x in done]),
        "classes_per_reaction": share([len(x["classes"]) for x in done]),
        "true_tie": len(tied) / len(done),
        "curated_among_optima": share([any(x["right"]) for x in done]),
        "uniform_choice": share([np.mean(x["right"]) for x in done]),
        "first_optimum": share([x["right"][0] for x in done]),
        "curated_among_optima_when_tied": share([any(x["right"]) for x in tied]),
        "uniform_choice_when_tied": share([np.mean(x["right"]) for x in tied]),
        "uniform_choice_held_out": share(
            [
                np.mean(x["right"])
                for x, h in zip(rows, held_out)
                if h and x["enumerated"]
            ]
        ),
    }
    out = Path("results/chem/exact_map")
    (out / f"ties-{args.data}.json").write_text(json.dumps(summary, indent=1))
    (out / f"ties-{args.data}.pkl").write_bytes(pickle.dumps(rows))
    print(json.dumps(summary, indent=1))
