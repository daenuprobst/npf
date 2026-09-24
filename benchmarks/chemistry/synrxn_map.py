"""Atom mapping on the five held-out sets of SynRXN, Phan et al., Sci. Data 2026, with the learning-free mapper.

SynRXN scores RXNMapper, GraphormerMapper, LocalMapper and RDTool with the validator of SynKit on three sets of organic
reactions, the Golden set, the set of Jaworski et al. in Nat. Commun. and 3,000 reactions of USPTO-50k, and two of
metabolic reactions, Recon3D and E. coli. We map the same unmapped reactions with the minimum firing vector, write
atom-mapped SMILES and score them with the same validator next to the outputs of the four mappers the files carry, so
every number comes from one scorer.

    git clone https://github.com/TieuLongPhan/synrxn <root>
    uv run python -m benchmarks.chemistry.synrxn_map prepare <root>                     # data/synrxn_aam.pkl
    uv run python -m benchmarks.chemistry.synrxn_map map --seconds 60                   # results/chem/synrxn_map/60s/<set>.csv
    uv run --with "synkit>=1.5,<1.6" python -m benchmarks.chemistry.synrxn_map score --seconds 60   # .../60s/scores.json
    uv run --with "synkit>=1.5,<1.6" python -m benchmarks.chemistry.synrxn_map score --single-product

The mapper is npf.chem.mapper, the search without a solver and the third level of npf.chem.third_level, and its maps
land in search-<seconds>s. --solver cp-sat maps with the integer program of npf.chem.exact instead, as the first
version of the paper did, and with --retries its budget doubles until optimality is proved, in <seconds>s-proved.

    uv run python -m benchmarks.chemistry.synrxn_map map --seconds 60
    uv run --with "synkit>=1.5,<1.6" python -m benchmarks.chemistry.synrxn_map score --seconds 60
    uv run python -m benchmarks.chemistry.synrxn_map map --solver cp-sat --seconds 20 --retries 4
"""

import argparse
import csv
import gzip
import json
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

from npf import chem
from npf.chem import cost, mapper, open_net
from npf.chem.mapping import mapped_smiles

from .golden import mapping_from_smiles, same_cgr

RDLogger.DisableLog("rdApp.*")

SETS = ("golden", "natcomm", "uspto_3k", "recon3d", "ecoli")
BASELINES = ("rxn_mapper", "graphormer", "local_mapper", "rdt")
DATA = Path("data/synrxn_aam.pkl")
OUT = Path("results/chem/synrxn_map")


def prepare(root, out=DATA):
    """Featurise every reaction of the five sets and keep the curated map and the four mappers' outputs beside it."""
    rows = []
    for name in SETS:
        with gzip.open(Path(root) / "Data" / "aam" / f"{name}.csv.gz", "rt") as f:
            for r in csv.DictReader(f):
                row = {
                    "original_rxn": r["reactions"],
                    "rxn": r["reactions"],
                    "label": 0,
                    "split": "test",
                    "id": len(rows),
                }

                try:
                    reaction = chem.featurise(row, max_atoms=None)
                except Exception:
                    reaction = None

                # our own reading of the curated map, where it is a clean injection over the same molecules
                if reaction is not None:
                    try:
                        reaction["target"] = mapping_from_smiles(
                            reaction, r["ground_truth"]
                        )
                    except Exception:
                        reaction["target"] = None

                rows.append(
                    {
                        "set": name,
                        "r_id": r["r_id"],
                        "input": r["reactions"],
                        "ground_truth": r["ground_truth"],
                        "baselines": {k: r[k] for k in BASELINES},
                        "reaction": reaction,
                    }
                )

    out.write_bytes(pickle.dumps(rows))

    for name in SETS:
        rs = [x for x in rows if x["set"] == name]
        read = [x["reaction"] for x in rs if x["reaction"] is not None]
        short = sum(open_net.deficit(r).any() for r in read)
        print(
            f"{name}: {len(rs)} reactions, {len(read)} featurised, {short} with product atoms the precursors cannot supply, "
            f"{sum(r['target'] is not None for r in read)} with a curated map we can read"
        )


def work(job):
    k, reaction, options = job
    start = time.time()
    options = dict(options)
    retries = options.pop("retries", 0)

    if options.pop("solver", "cp-sat") == "search":
        try:
            mapping, levels, proved = mapper.best_mapping(
                reaction, seconds=options["seconds"]
            )
            places = None if levels is None else levels[0]
        except Exception:
            mapping, places, proved = None, None, False

        return scored(k, reaction, mapping, places, proved, start)

    try:
        if open_net.deficit(reaction).any():
            # seats on the copies the open net adds do not exist in the written reaction and stay unmapped
            mapping, places, proved, _ = open_net.solve_open(reaction, **options)
        else:
            # the integer program, and with it OR-tools, is loaded only when asked for
            from npf.chem import exact

            hint = chem.feasible_start(reaction)
            mapping, places, proved = exact.solve(
                reaction,
                hint=None if hint is None else hint.astype(np.int64),
                **options,
            )
    except Exception:
        mapping, places, proved = None, None, False

    # the stopping rule is a proof of optimality, so a solve that runs out of budget is repeated with twice as much
    if retries and not proved and mapping is not None:
        doubled = options | dict(
            seconds=2 * options["seconds"],
            deterministic=2 * options["deterministic"],
            retries=retries - 1,
        )

        return work((k, reaction, doubled)) | {"seconds": time.time() - start}

    return scored(k, reaction, mapping, places, proved, start)


def scored(k, reaction, mapping, places, proved, start):
    known = (
        mapping is not None and reaction["target"] is not None and (mapping >= 0).all()
    )
    correct = bool(
        known and same_cgr(reaction, mapping, reaction["target"].astype(np.int64))
    )

    return {
        "k": k,
        "mapping": mapping,
        "cost": places,
        "proved": bool(proved),
        "scored": bool(known),
        "correct": correct,
        "seconds": time.time() - start,
    }


def folder(seconds, retries=0, solver="search"):
    """Every mapper and time limit keeps its own maps and scores."""
    if solver == "search":
        return OUT / f"search-{seconds:g}s"

    return OUT / (f"{seconds:g}s-proved" if retries else f"{seconds:g}s")


def run(sets, seconds, processes, workers, limit=None, retries=0, solver="search"):
    rows = pickle.loads(DATA.read_bytes())

    # the chosen cost and a budget in deterministic time, so the maps do not depend on the load of the machine
    options = cost.CHOSEN | dict(
        seconds=4 * seconds, deterministic=seconds, workers=workers, retries=retries
    )

    # the search runs under a limit on wall time, which only the slowest proofs reach
    if solver == "search":
        options = dict(seconds=seconds, solver=solver)

    out = folder(seconds, retries, solver)
    out.mkdir(parents=True, exist_ok=True)

    for name in sets:
        chosen = [k for k, x in enumerate(rows) if x["set"] == name][:limit]
        jobs = [
            (k, rows[k]["reaction"], options)
            for k in chosen
            if rows[k]["reaction"] is not None
        ]
        start = time.time()

        with Pool(processes) as pool:
            done = {x["k"]: x for x in pool.imap_unordered(work, jobs)}

        # an unread or unsolved reaction is written unmapped, it counts as wrong
        with open(out / f"{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["r_id", "npf", "proved", "cost", "seconds"])

            for k in chosen:
                x = done.get(k)
                mapped = (
                    mapped_smiles(rows[k]["input"], x["mapping"])
                    if x and x["mapping"] is not None
                    else rows[k]["input"]
                )
                w.writerow(
                    [
                        rows[k]["r_id"],
                        mapped,
                        int(bool(x and x["proved"])),
                        x and x["cost"],
                        f"{x['seconds']:.2f}" if x else "",
                    ]
                )

        solved = [x for x in done.values() if x["mapping"] is not None]
        scored = [x["correct"] for x in done.values() if x["scored"]]
        print(
            f"{name}: {len(chosen)} reactions, solved {len(solved)}, proved {np.mean([x['proved'] for x in done.values()]):.4f}, "
            f"our CGR check {np.mean(scored or [0.0]):.4f} on {len(scored)}, {time.time() - start:.0f}s",
            flush=True,
        )


def score(
    sets,
    seconds,
    columns=("npf",) + BASELINES,
    single=False,
    retries=0,
    solver="search",
):
    """Accuracy of ours and of the four mappers under the SynRXN metric, SynKit's AAMValidator with its defaults. The
    mappers' outputs do not depend on our time limit, so a second limit needs only our column. single scores the
    reactions with one product molecule alone."""
    import pandas as pd
    from synkit.Chem.Reaction.Mapper import AAMValidator

    validator = AAMValidator()
    rows = pickle.loads(DATA.read_bytes())
    result, verdicts = {}, {}

    for name in sets:
        rs = [
            x
            for x in rows
            if x["set"] == name
            and (not single or len(x["input"].split(">>")[1].split(".")) == 1)
        ]
        ours = {
            r["r_id"]: r["npf"]
            for r in csv.DictReader(
                open(folder(seconds, retries, solver) / f"{name}.csv")
            )
        }
        if len(ours) < len(rs):
            print(
                f"{name}: only {len(ours)} of {len(rs)} reactions mapped, scored on those"
            )
            rs = [x for x in rs if x["r_id"] in ours]

        predictions = {"npf": [ours[x["r_id"]] for x in rs]} | {
            k: [x["baselines"][k] for x in rs] for k in BASELINES
        }
        result[name] = {"reactions": len(rs)}
        verdicts[name] = {"r_id": [x["r_id"] for x in rs]}

        for col in columns:
            predicted = predictions[col]
            df = pd.DataFrame(
                {"ground_truth": [x["ground_truth"] for x in rs], col: predicted}
            )
            first = validator.validate_smiles(
                data=df,
                ground_truth_col="ground_truth",
                mapped_cols=[col],
                ignore_tautomers=False,
            )[0]
            first = first[0] if isinstance(first, (tuple, list)) else first
            result[name][col] = float(first["accuracy"])

            # one verdict per reaction, for paired tests between the mappers
            verdicts[name][col] = [bool(b) for b in first["results"]]

        print(name, json.dumps(result[name]), flush=True)

    (
        folder(seconds, retries, solver)
        / ("scores-single-product.json" if single else "scores.json")
    ).write_text(json.dumps(result, indent=1))
    (
        folder(seconds, retries, solver)
        / ("verdicts-single-product.json" if single else "verdicts.json")
    ).write_text(json.dumps(verdicts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prepare", "map", "score"])
    ap.add_argument(
        "root", nargs="?", help="the checkout of the SynRXN repository, for prepare"
    )
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument(
        "--solver",
        choices=("search", "cp-sat"),
        default="search",
        help="the mapper of npf.chem.mapper, or the integer program of npf.chem.exact",
    )
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--processes", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--retries",
        type=int,
        default=0,
        help="map, repeat a solve that did not prove optimality with twice the budget, this many times",
    )
    ap.add_argument(
        "--limit",
        type=int,
        help="only the first reactions of every set, for a quick test",
    )
    ap.add_argument(
        "--columns",
        default=",".join(("npf",) + BASELINES),
        help="score, which outputs to score",
    )
    ap.add_argument(
        "--single-product",
        action="store_true",
        help="score, only the reactions with one product molecule",
    )
    args = ap.parse_args()
    sets = args.sets.split(",")

    if args.phase == "prepare":
        prepare(args.root)
    elif args.phase == "map":
        run(
            sets,
            args.seconds,
            args.processes,
            args.workers,
            args.limit,
            args.retries,
            args.solver,
        )
    else:
        score(
            sets,
            args.seconds,
            args.columns.split(","),
            args.single_product,
            args.retries,
            args.solver,
        )


if __name__ == "__main__":
    main()
