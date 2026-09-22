"""Atom maps as exact minimum firing vectors of the valence net, with no learning and no recorded map.

The cost is lexicographic. First the number of bond places that a firing empties or fills, then the number of tokens
that move on the places that stay marked, on hydrogen and on charge. An integer program finds the minimum and proves
it. The second level was chosen on 200 reactions of the Golden set, the other 1,560 are held out and reported apart.
Recorded maps of Schneider 50k cannot serve for that choice, they often swap the two oxygens of an acid.

The defaults are the chosen cost (exact.CHOSEN) and a budget in deterministic time with one solver worker, so the
maps do not depend on the load of the machine.

    uv run python -m benchmarks.chemistry.exact_map --data golden-dev --secondary 0,0,0 --labile-h --no-ch-places
    uv run python -m benchmarks.chemistry.exact_map --data golden-dev --labile-h --no-ch-places
    uv run python -m benchmarks.chemistry.exact_map --data golden-dev --no-ch-places     # the three alternatives
    uv run python -m benchmarks.chemistry.exact_map --data golden-dev                   # the chosen cost
    uv run python -m benchmarks.chemistry.exact_map --data golden                       # results/chem/exact_map/
    uv run python -m benchmarks.chemistry.exact_map --data schneider50k --deterministic 3 --processes 18 \\
        --write data/exact_maps_schneider50k.pkl                                        # the maps classification reads
"""

import argparse
import json
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from npf import chem
from npf.chem import exact

from .golden import same_cgr

DEV_REACTIONS = 400
GOLDEN_DEV_REACTIONS = 200


def golden_dev(n):
    """The reactions of the Golden set on which the second level of the cost was chosen, the rest is held out."""
    return np.random.default_rng(0).choice(n, GOLDEN_DEV_REACTIONS, replace=False)


def load(name):
    if name.startswith("golden"):
        reactions = pickle.loads(Path("data/golden.pkl").read_bytes())["reactions"]
    elif name == "dev":
        reactions = [
            r
            for r in pickle.loads(Path("data/schneider50k.pkl").read_bytes())[
                "reactions"
            ]
            if r["split"] == "train"
        ]
    else:
        # every reaction of a benchmark, a product larger than its precursors cannot be seated
        return [
            r
            for r in pickle.loads(Path(f"data/{name}.pkl").read_bytes())["reactions"]
            if len(r["b"]["x"]) <= len(r["a"]["x"])
        ]

    usable = [
        r
        for r in reactions
        if r["target"] is not None and len(r["b"]["x"]) <= len(r["a"]["x"])
    ]

    if name == "dev":
        usable = [
            usable[i]
            for i in np.random.default_rng(0).choice(
                len(usable), DEV_REACTIONS, replace=False
            )
        ]

    if name == "golden-dev":
        usable = [usable[i] for i in golden_dev(len(usable))]

    return usable


def work(job):
    reaction, options, hint = job
    start = time.time()

    # a feasible seating by element and environment, the solver starts from it and never returns worse
    if hint is None:
        try:
            hint = chem.feasible_start(reaction)
        except Exception:
            hint = None

    try:
        mapping, cost, proved = exact.solve(
            reaction, hint=None if hint is None else hint.astype(np.int64), **options
        )
    except Exception:
        mapping, cost, proved = None, None, False

    known = (
        reaction["target"] is not None
        and mapping is not None
        and (np.asarray(reaction["target"]) >= 0).all()
    )
    correct = known and same_cgr(reaction, mapping, reaction["target"].astype(np.int64))

    return {
        "id": reaction["id"],
        "mapping": mapping,
        "cost": cost,
        "proved": bool(proved),
        "correct": bool(correct),
        "scored": bool(known),
        "seconds": time.time() - start,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default="golden",
        help="golden, golden-dev, dev, or the name of a benchmark file in data/",
    )
    ap.add_argument(
        "--secondary",
        default=",".join(map(str, exact.CHOSEN["secondary"])),
        help="weights of bond order tokens, hydrogen moves and charge moves below the connectivity cost",
    )
    ap.add_argument(
        "--all-orders",
        action="store_true",
        help="count the order tokens of formed and broken bonds too, not only of kept bonds",
    )
    ap.add_argument(
        "--labile-h",
        action="store_true",
        help="count hydrogen on heteroatoms as well, which the chosen cost does not",
    )
    ap.add_argument(
        "--no-ch-places",
        action="store_true",
        help="hydrogen on carbon on the second level, not as a bond place of the first",
    )
    ap.add_argument(
        "--deterministic",
        type=float,
        default=20.0,
        help="budget in deterministic time per solve",
    )
    ap.add_argument(
        "--seconds",
        type=float,
        help="limit on wall time per solve, four times the budget by default",
    )
    ap.add_argument("--processes", type=int, default=6)
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="solver workers per reaction, interleaved, so the result stays deterministic",
    )
    ap.add_argument(
        "--hints",
        help="a file of mappings that the solver starts from, in the format of --write",
    )
    ap.add_argument(
        "--write",
        help="write {reaction id, product -> precursor} to this file, the maps classification reads",
    )
    ap.add_argument("--out", default="results/chem/exact_map")
    args = ap.parse_args()
    secondary = tuple(int(x) for x in args.secondary.split(","))
    reactions = load(args.data)
    hints = pickle.loads(Path(args.hints).read_bytes()) if args.hints else {}
    fits = lambda r: r["id"] in hints and len(hints[r["id"]]) == len(r["b"]["x"])
    options = dict(
        secondary=secondary,
        seconds=args.seconds or 4 * args.deterministic,
        workers=args.workers,
        all_orders=args.all_orders,
        deterministic=args.deterministic,
        labile_h=args.labile_h,
        ch_places=not args.no_ch_places,
    )
    jobs = [(r, options, hints[r["id"]] if fits(r) else None) for r in reactions]
    start = time.time()

    with Pool(args.processes) as pool:
        rows = []
        for row in pool.imap(work, jobs, chunksize=1):
            rows.append(row)

            if len(rows) % 200 == 0:
                scored = [x["correct"] for x in rows if x["scored"]]
                print(
                    f"{len(rows)}/{len(reactions)} proved {np.mean([x['proved'] for x in rows]):.4f} agrees with the record {np.mean(scored or [0.0]):.4f} "
                    f"{time.time() - start:.0f}s",
                    flush=True,
                )

            if args.write and len(rows) % 5000 == 0:
                Path(args.write).write_bytes(
                    pickle.dumps(
                        {
                            x["id"]: x["mapping"].astype(np.int16)
                            for x in rows
                            if x["mapping"] is not None
                        }
                    )
                )

    if args.write:
        Path(args.write).write_bytes(
            pickle.dumps(
                {
                    x["id"]: x["mapping"].astype(np.int16)
                    for x in rows
                    if x["mapping"] is not None
                }
            )
        )

    scored = [x for x in rows if x["scored"]]
    mean = lambda xs: float(np.mean(xs)) if len(xs) else None
    summary = {
        "data": args.data,
        "secondary": secondary,
        "all_orders": args.all_orders,
        "labile_h": args.labile_h,
        "ch_places": not args.no_ch_places,
        "seconds": options["seconds"],
        "deterministic": args.deterministic,
        "reactions": len(rows),
        "seated": mean([x["mapping"] is not None for x in rows]),
        "scored": len(scored),
        "correct": mean([x["correct"] for x in scored]),
        "proved": mean([x["proved"] for x in rows]),
        "correct_when_proved": mean([x["correct"] for x in scored if x["proved"]]),
        "correct_when_not_proved": mean(
            [x["correct"] for x in scored if not x["proved"]]
        ),
        "median_seconds": float(np.median([x["seconds"] for x in rows])),
        "mean_seconds": mean([x["seconds"] for x in rows]),
    }

    if args.data == "golden":
        dev = set(golden_dev(len(rows)).tolist())
        summary["correct_held_out"] = mean(
            [x["correct"] for i, x in enumerate(rows) if i not in dev]
        )
        summary["held_out"] = len(rows) - len(dev)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{args.data}-{'-'.join(map(str, secondary))}"
        + ("-all" if args.all_orders else "")
        + ("-nolabile" if not args.labile_h else "")
        + ("-ch" if not args.no_ch_places else "")
    )
    (out / f"{stem}.json").write_text(json.dumps(summary, indent=1))
    (out / f"{stem}.pkl").write_bytes(pickle.dumps(rows))
    print(json.dumps(summary, indent=1))

    if args.data == "golden":
        # the comparison with RXNMapper on the same reactions, written next to the rows
        from .exact_map_report import main as report

        report(out / f"{stem}.pkl")
