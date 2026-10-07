"""The enzymatic data of EC number classification, and the atom maps the classifiers read.

EnzymeMap at EC level 3 (Heid et al., Chem. Sci. 2023). The source table lists 349,458 rows but far fewer distinct
reactions, the same chemistry once per organism and EC entry, so it is deduplicated on the unmapped SMILES, keeping the
best-quality, most natural record with the fewest steps. The reactions of an EC class split across the holdout of
whole EC classes that the maps were first computed on are left out, which keeps the reaction set of the paper. A
classifier cannot score on held-out classes, so reactions are held out instead: 10 % of every class with at least three
reactions go to test, one draw with a fixed seed. benchmarks.chemistry.experiment trains on it with --dataset
enzymemap_ec.

ECREACT in the split of Enzyformer (Liu et al. 2026, zenodo 18083829): train and valid pooled for training as in its
EC module, every test reaction kept, and the EC numbers of all splits as classes, so a test EC never seen in training
can only be wrong. benchmarks.chemistry.care trains on it with --dataset ecreact_enzyformer.

Maps come from three sources: the mapper of the paper (exact_map --third-level --write), RXNMapper
(baselines/rxnmapper_golden.py on the unmapped reactions) and, on EnzymeMap only, the curated maps of the record.
convert writes the last two as {reaction id: product atom -> precursor atom}, the format of exact_map --write.

    uv run python -m benchmarks.chemistry.enzymes prepare enzymemap      # data/enzymemap_ec.pkl
    uv run python -m benchmarks.chemistry.enzymes prepare enzyformer     # data/ecreact_enzyformer.pkl, downloads the CSV
    uv run python -m benchmarks.chemistry.enzymes unmapped enzymemap_ec  # data/enzyme_maps/enzymemap_ec_unmapped.txt
    uv run python -m benchmarks.chemistry.enzymes convert enzymemap_ec   # data/enzyme_maps/{rxnmapper,recorded}_maps_*.pkl
"""

import argparse
import functools
import hashlib
import json
import pickle
import urllib.request
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from npflow import chem
from npflow.chem import featurisation

MAPS = Path("data/enzyme_maps")
ENZYMEMAP = Path("data/enzymemap/enzymemap_v2_brenda2023.csv.gz")
ENZYFORMER = Path("data/external/ecreact_enzyformer.csv")
ENZYFORMER_URL = "https://zenodo.org/api/records/18083829/files/ECREACT.csv/content"
ENZYFORMER_MD5 = "d1c519d2cb6f54b4a23485e4ffe2a305"

# EnzymeMap is balanced and has two or three products, the size caps set on single-product USPTO reactions would cost
# a fifth of it. featurise keeps its own hard ceiling above these
ENZYMEMAP_ATOMS = 120

# the order of the records a distinct reaction keeps, after quality and naturalness
STEP_RANK = {"single": 0, "single from multi": 1, "multi": 2}


def _featurise(job):
    row, meta = job
    try:
        out = featurisation.featurise(row, max_atoms=None)
    except Exception:
        return None

    return None if out is None else (out, meta)


def ec_holdout(records):
    """Whole EC3 classes to train, val or test, largest first to the split furthest below 80/10/10. A reaction also
    recorded under an EC3 of another split is left out."""
    by_class = defaultdict(list)
    for k, r in enumerate(records):
        by_class[r["ec3"]].append(k)

    target = {"train": 0.8 * len(records), "val": 0.1 * len(records), "test": 0.1 * len(records)}
    size = {"train": 0, "val": 0, "test": 0}
    assigned = {}
    for c in sorted(by_class, key=lambda c: (-len(by_class[c]), c)):
        name = max(target, key=lambda s: target[s] - size[s])
        assigned[c] = name
        size[name] += len(by_class[c])

    return [r for r in records if all(assigned.get(e, assigned[r["ec3"]]) == assigned[r["ec3"]] for e in r["all_ec3"])]


def prepare_enzymemap(processes=12, seed=0):
    # pandas comes with ortools
    import pandas as pd

    df = pd.read_csv(ENZYMEMAP)
    df["ec_num"] = df["ec_num"].astype(str)
    df["ec3"] = df["ec_num"].str.rsplit(".", n=1).str[0]
    ec3_of = df.groupby("unmapped")["ec3"].agg(lambda s: sorted(set(s))).to_dict()
    ec4_of = df.groupby("unmapped")["ec_num"].agg(lambda s: sorted(set(s))).to_dict()
    n_dup = df["unmapped"].value_counts().to_dict()
    df["_step"] = df["steps"].map(STEP_RANK).fillna(len(STEP_RANK))
    reps = df.sort_values(["quality", "natural", "_step"], ascending=[False, False, True], kind="stable")
    reps = reps.drop_duplicates("unmapped", keep="first")

    jobs = []
    for k, r in enumerate(reps.itertuples()):
        if ">>" not in r.mapped or ">>" not in r.unmapped:
            continue

        row = {"original_rxn": r.mapped, "rxn": r.unmapped, "label": 0, "split": "train", "id": k}
        meta = {"ec_num": r.ec_num, "ec3": r.ec3, "all_ec3": ec3_of[r.unmapped], "all_ec4": ec4_of[r.unmapped],
                "quality": float(r.quality), "natural": bool(r.natural), "steps": r.steps, "rule_id": int(r.rule_id),
                "n_duplicates": int(n_dup[r.unmapped])}
        jobs.append((row, meta))

    # set before the pool forks, so every worker has them
    featurisation.MAX_PRECURSOR_ATOMS = featurisation.MAX_PRODUCT_ATOMS = ENZYMEMAP_ATOMS
    with Pool(processes) as pool:
        done = [d for d in pool.map(_featurise, jobs, chunksize=64) if d is not None]

    records = [out | meta for out, meta in done]
    classes = sorted({r["ec3"] for r in records})
    for r in records:
        r["label"] = classes.index(r["ec3"])

    reactions = ec_holdout(records)

    # 10 % of every class with at least three reactions to test, the same draw for every model and seed. No split is
    # named val, experiment.splits draws validation from the training reactions
    rng = np.random.default_rng(seed)
    by_label = defaultdict(list)
    for r in reactions:
        by_label[r["label"]].append(r)

    for label in sorted(by_label):
        members = by_label[label]
        n_test = max(1, round(0.1 * len(members))) if len(members) >= 3 else 0
        for rank, k in enumerate(rng.permutation(len(members))):
            members[k]["split"] = "test" if rank < n_test else "train"

    out = Path("data/enzymemap_ec.pkl")
    out.write_bytes(pickle.dumps({"reactions": reactions, "classes": classes}, protocol=4))
    print(f"{len(df)} rows, {len(jobs)} distinct reactions, {len(records)} featurised, {len(reactions)} kept, "
          f"{sum(r['split'] == 'train' for r in reactions)} train and {sum(r['split'] == 'test' for r in reactions)} "
          f"test over {len(classes)} EC3 classes, wrote {out}")


def prepare_enzyformer(processes=8):
    import csv

    if not ENZYFORMER.exists():
        ENZYFORMER.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ENZYFORMER_URL, ENZYFORMER)

    md5 = hashlib.md5(ENZYFORMER.read_bytes()).hexdigest()
    if md5 != ENZYFORMER_MD5:
        raise SystemExit(f"{ENZYFORMER} has md5 {md5}, not the {ENZYFORMER_MD5} of zenodo 18083829")

    rows = list(csv.DictReader(open(ENZYFORMER)))
    classes = sorted({r["EC_Number"] for r in rows})
    index = {ec: k for k, ec in enumerate(classes)}
    jobs = [{"original_rxn": r["EnzymaticReaction"], "rxn": r["EnzymaticReaction"], "label": index[r["EC_Number"]],
             "split": "test" if r["split"] == "test" else "train", "id": k} for k, r in enumerate(rows)]
    with Pool(processes) as pool:
        featurise = functools.partial(featurisation.featurise, max_atoms=None)
        reactions = [x for x in pool.map(featurise, jobs, chunksize=64) if x is not None]

    out = Path("data/ecreact_enzyformer.pkl")
    out.write_bytes(pickle.dumps({"reactions": reactions, "classes": classes}, protocol=4))
    print(f"{len(reactions)} of {len(rows)} reactions, {sum(r['split'] == 'train' for r in reactions)} train and "
          f"{sum(r['split'] == 'test' for r in reactions)} test, {len(classes)} EC numbers, wrote {out}")


def unmapped(name):
    """The reactions of data/<name>.pkl as id and SMILES per line, the input of baselines/rxnmapper_golden.py."""
    reactions = pickle.loads(Path(f"data/{name}.pkl").read_bytes())["reactions"]
    MAPS.mkdir(parents=True, exist_ok=True)
    (MAPS / f"{name}_unmapped.txt").write_text("\n".join(f"{r['id']}\t{r['smiles']}" for r in reactions))


def convert(name):
    """RXNMapper's mapped SMILES and the recorded maps of data/<name>.pkl as {reaction id: map}. A map that does not
    seat every product atom is left out, and the classifier reads that reaction without a firing vector."""
    from .golden import mapping_from_smiles

    # enzymatic reactions carry cofactors, so the size bounds of the USPTO featurisation would drop most maps
    featurisation.MAX_PRECURSOR_ATOMS = featurisation.MAX_PRODUCT_ATOMS = 10**6
    chem.featurise = functools.partial(featurisation.featurise, max_atoms=None)

    reactions = pickle.loads(Path(f"data/{name}.pkl").read_bytes())["reactions"]
    mapped = json.loads((MAPS / f"rxnmapper_{name}.json").read_text())
    out = {}
    for r in reactions:
        if str(r["id"]) in mapped:
            try:
                m = mapping_from_smiles(r, mapped[str(r["id"])]["mapped_rxn"])
            except Exception:
                m = None

            if m is not None and len(m) == len(r["b"]["element"]):
                out[r["id"]] = m.astype(np.int16)

    (MAPS / f"rxnmapper_maps_{name}.pkl").write_bytes(pickle.dumps(out))
    print(f"{name}: RXNMapper mapped {len(mapped)}, usable maps {len(out)} of {len(reactions)} reactions")

    recorded = {r["id"]: r["target"] for r in reactions if r["target"] is not None}
    if recorded:
        (MAPS / f"recorded_maps_{name}.pkl").write_bytes(pickle.dumps(recorded))
        print(f"{name}: recorded maps {len(recorded)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prepare", "unmapped", "convert"])
    ap.add_argument("name", help="enzymemap or enzyformer for prepare, else the name of a file in data/")
    ap.add_argument("--processes", type=int, default=12)
    args = ap.parse_args()

    if args.phase == "prepare":
        {"enzymemap": prepare_enzymemap, "enzyformer": prepare_enzyformer}[args.name](args.processes)
    else:
        {"unmapped": unmapped, "convert": convert}[args.phase](args.name)
