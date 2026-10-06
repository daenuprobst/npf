"""Small forward prediction benchmarks of the literature, each on its published split.

bv is the Baeyer-Villiger set of Wu et al., Chem. Commun. 57, 4114 (2021), 1670 / 204 / 197 reactions from Reaxys
with peracetic acid added as the oxidant of every reaction and tetrahedral stereo removed. A WLDN trained from scratch
reaches 90.4 % top-1 there, scored by the rexgen_direct script, the recorded product among the molecules of the
prediction. suzuki_<n> is the learning curve of Wu et al., Sci. Rep. 12 (2022) on Suzuki couplings from Reaxys, a
random 8 / 1 / 1 split of n reactions for each n, where a Transformer trained from scratch reaches the top-1 of their
Table S6, scored as equal canonical SMILES with stereo. uspto_50k is the split of Schneider et al. 2016 in the
canonical form of Somnath et al. 2021 (graphretro), 40,008 / 5,001 / 5,007 reactions, the forward numbers published
for it are those of Table 2 of SynBridge (arXiv 2507.08475).

Products are compared without stereo as in every other forward run, and summary rescores the beams of every run with
stereo, which the decoded molecules take from the precursors where no firing touched them (npflow.chem.decode.copy_stereo).

Every set is written like data/uspto_mit.pkl, precursors and reagents merged by npflow.chem.reaction, with its number of
test lines, so a test reaction that cannot be read counts as wrong and the denominator is the published one.

    uv run python -m benchmarks.chemistry.small_forward download    # data/small_forward/
    uv run python -m benchmarks.chemistry.small_forward build       # data/bv.pkl, data/suzuki_1k.pkl, ..., data/uspto_50k.pkl
    uv run python -m benchmarks.chemistry.net_targets --dataset bv --processes 8
    uv run python -m benchmarks.chemistry.experiment --task forward --model npf --dataset bv --amp --seed 0 --root results/small_forward
    uv run python -m benchmarks.chemistry.small_forward summary     # results/small_forward/summary.json

results/small_forward_2026-09-25.sh and results/uspto50k_2026-09-26.sh run all of it with three seeds.
"""

import argparse
import csv
import hashlib
import io
import json
import pickle
import urllib.request
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from npflow import chem
from npflow.chem.decode import stereo_source

SOURCE = Path("data/small_forward")

# the published files, pinned to a commit, with their digests
BV_REPO = (
    "https://raw.githubusercontent.com/hongliangduan/A-graph-convolutional-neural-network-for-addressing-small-scale-"
    "reaction-prediction/a4a84d89407a3ee7848ff84e1a9ea60fe8bba423/rexgen_direct/data"
)
BV_FILES = {
    "train": ("train.txt.proc", "5a9c107a0635e26c99dfce2af63da2cc7b0eeb717e45ed9df617a84757b5dd79"),
    "val": ("valid.txt.proc", "065b2e6442fee82585006974948258b8421ef16b5541cb07ffa555fea92f634c"),
    "test": ("test.txt.proc", "1080c317352601c7805902e9435f1adc8e533c999ec0905a8b04bf73d35e89b8"),
}
SUZUKI_ZIP = (
    "https://raw.githubusercontent.com/hongliangduan/Virtual-data-augmentation-methood-for-reaction-prediction-in-"
    "small-dataset-scenario/aba2979f3f73e4cc48dc7a4af0dfa8e7a03be9e1/reaction_data.zip"
)
SUZUKI_SHA = "eb5734b28593aadcb84b191f23e9e16a3c425a4ef3765e7c02d186a5ce701ea5"
SUZUKI_FILES = {"train": "train", "val": "dev", "test": "test"}
USPTO_50K_REPO = (
    "https://raw.githubusercontent.com/vsomnath/graphretro/971712b3874c178fe511daacabf3310f9340e061/datasets/uspto-50k"
)
USPTO_50K_FILES = {
    "train": ("canonicalized_train.csv", "071aca066b1d20a8bfb2d7d04729cece2eff900b1e24831a47c7de8d825dd5a2"),
    "val": ("canonicalized_eval.csv", "f74d44a8e86fa3a3b37d862189b0d58fce445a3919c395fa98937fbd8f593d7f"),
    "test": ("canonicalized_test.csv", "89cda6601ba85bafe262acaa99c1a9849eaa2d0936c3bcb6851eb372e7c1d72f"),
}

BV_SOURCE = "Wu et al., Chem. Commun. 2021, 57, 4114, doi 10.1039/d1cc00586c, abstract"
SUZUKI_SOURCE = "Wu et al., Sci. Rep. 2022, 12, doi 10.1038/s41598-022-21524-6, SI Table S6"
SYNBRIDGE_SOURCE = "SynBridge, arXiv 2507.08475, Table 2, forward, trained on USPTO-50K alone"

# how each published number treats stereo, read in the code of the paper where there is code
BV_STEREO = (
    "yes, rexgen_direct eval_by_smiles edits the reactant molecule and compares isomeric SMILES, the data hold no "
    "tetrahedral stereo, E/Z in 1 of 197 test products"
)
SUZUKI_STEREO = "yes, 'Compare target and predicted accuracy.py' compares RDKit canonical SMILES with stereo"
SYNBRIDGE_STEREO = (
    "no, test.py of github.com/EDAPINENUT/synbridge (a23f23b) rebuilds prediction and target from atom, bond, "
    "aromaticity and charge tensors (synflow.chem.utils.result2mol) without chirality or bond stereo, and leaves test "
    "reactions whose target it cannot rebuild and top-k lists of invalid predictions out of the denominator"
)
RERUN_STEREO = "not stated, retrained by the SynBridge authors, whose code for it is not released"
G2G_STEREO = "probably no, the backbone of SynBridge predicts the same graph without stereo, its code is not released"


def published(model, top1, source, stereo, top3=None, top5=None):
    return {"model": model, "top1": top1, "top3": top3, "top5": top5, "source": source, "scores_stereo": stereo}


# lines per split and the published numbers in percent
SETS = {
    "bv": ({"train": 1670, "val": 204, "test": 197}, [published("WLDN from scratch", 90.4, BV_SOURCE, BV_STEREO)]),
    **{
        f"suzuki_{k}k": (
            {"train": 800 * k, "val": 100 * k, "test": 100 * k},
            [published("Transformer from scratch", top1, SUZUKI_SOURCE, SUZUKI_STEREO)],
        )
        for k, top1 in ((1, 1.20), (3, 45.04), (5, 69.50), (7, 78.34), (10, 83.57), (30, 91.17), (60, 93.13))
    },
    "uspto_50k": (
        {"train": 40008, "val": 5001, "test": 5007},
        [
            published("MEGAN", 88.9, SYNBRIDGE_SOURCE, RERUN_STEREO, 90.8, 93.0),
            published("NeRF", 94.6, SYNBRIDGE_SOURCE, RERUN_STEREO, 96.1, 97.9),
            published("G2G-Former", 93.0, SYNBRIDGE_SOURCE, G2G_STEREO, 93.9, 94.0),
            published("S2S-Former", 95.0, SYNBRIDGE_SOURCE, RERUN_STEREO, 95.2, 95.8),
            published("SynBridge", 95.9, SYNBRIDGE_SOURCE, SYNBRIDGE_STEREO, 96.2, 96.5),
        ],
    ),
}


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def download(_):
    """Fetch the published files and check them against their digests."""
    for folder, repo, files in (("bv", BV_REPO, BV_FILES), ("uspto_50k", USPTO_50K_REPO, USPTO_50K_FILES)):
        (SOURCE / folder).mkdir(parents=True, exist_ok=True)

        for name, sha in files.values():
            path = SOURCE / folder / name
            if not path.exists():
                print(f"downloading {repo}/{name}", flush=True)
                path.write_bytes(urllib.request.urlopen(f"{repo}/{name}", timeout=60).read())

            assert _digest(path.read_bytes()) == sha, f"checksum of {path}"

    if not (SOURCE / "suzuki_ratio" / "60k" / "test.target").exists():
        print(f"downloading {SUZUKI_ZIP}, 23 MB", flush=True)
        archive = urllib.request.urlopen(SUZUKI_ZIP, timeout=120).read()
        assert _digest(archive) == SUZUKI_SHA, "checksum of reaction_data.zip"

        # only the learning curve, the other folders hold augmented and cross-validation data
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            for member in z.namelist():
                if member.startswith("data/Suzuki_ratio/") and not member.endswith("/"):
                    path = SOURCE / "suzuki_ratio" / member[len("data/Suzuki_ratio/") :]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(z.read(member))

    print(f"published files in {SOURCE}")


def rows(name):
    """(split, reaction SMILES) of every line of a set, in the order of its files."""
    out = []

    if name == "bv":
        for split, (file, _) in BV_FILES.items():
            # a line is the mapped reaction and its recorded edits, the edits are not read
            out += [(split, line.split()[0]) for line in (SOURCE / "bv" / file).read_text().splitlines()]
    elif name == "uspto_50k":
        for split, (file, _) in USPTO_50K_FILES.items():
            # reactants>reagents>product with atom maps, which npflow.chem.reaction removes
            with open(SOURCE / "uspto_50k" / file) as f:
                out += [(split, row["reactants>reagents>production"]) for row in csv.DictReader(f)]
    else:
        folder = SOURCE / "suzuki_ratio" / name.split("_")[1]

        for split, file in SUZUKI_FILES.items():
            source = (folder / f"{file}.source").read_text().splitlines()
            target = (folder / f"{file}.target").read_text().splitlines()
            assert len(source) == len(target), f"{folder}/{file} has unequal sides"
            out += [(split, f"{s.strip()}>>{t.strip()}") for s, t in zip(source, target)]

    lines = {split: sum(s == split for s, _ in out) for split in SUZUKI_FILES}
    assert lines == SETS[name][0], f"{name} has {lines} lines, expected {SETS[name][0]}"

    return out


def _featurise(job):
    k, (split, smiles) = job

    try:
        return chem.reaction(smiles, split=split, id=k)
    except Exception:
        return None


def stereo(smiles):
    return any(c in smiles for c in "@/\\")


def build_set(name, processes=8):
    """Write data/<name>.pkl and print what could not be read and what the size caps of training leave out."""
    lines = rows(name)

    with Pool(processes) as pool:
        featurised = pool.map(_featurise, list(enumerate(lines)), chunksize=64)

    data = [r for r in featurised if r is not None]
    failed = [(split, k, smiles) for k, ((split, smiles), r) in enumerate(zip(lines, featurised)) if r is None]
    stats = {}

    for split in SUZUKI_FILES:
        rs = [r for r in data if r["split"] == split]
        stats[split] = {
            "lines": SETS[name][0][split],
            "parsed": len(rs),
            "over_size_caps": sum(
                len(r["a"]["x"]) > chem.MAX_PRECURSOR_ATOMS or len(r["b"]["x"]) > chem.MAX_PRODUCT_ATOMS for r in rs
            ),
            "product_with_stereo": sum(stereo(r["smiles"].split(">>")[1]) for r in rs),
            "several_products": sum("." in r["smiles"].split(">>")[1] for r in rs),
            "precursor_atoms_mean": float(np.mean([len(r["a"]["x"]) for r in rs])),
        }

    # the same reaction in the training and the test file
    seen = {r["smiles"] for r in data if r["split"] == "train"}
    stats["test"]["also_in_train"] = sum(r["smiles"] in seen for r in data if r["split"] == "test")
    out = {
        "reactions": data,
        "classes": ["-"],
        "name": name,
        "test_lines": SETS[name][0]["test"],
        "failed": failed,
        "stats": stats,
    }
    Path(f"data/{name}.pkl").write_bytes(pickle.dumps(out, protocol=4))
    print(f"data/{name}.pkl, {len(failed)} lines could not be read, {json.dumps(stats)}", flush=True)

    return stats


def build(args):
    for name in args.sets or SETS:
        build_set(name, args.processes)


def _rank(job):
    """First place of the beam whose product is right with stereo, the width of the beam when none is."""
    reaction, cands, width = job
    found = [chem.product_found(reaction, np.asarray(edits, np.int64), stereo=True) for edits, _, _ in cands]

    return found.index(True) if True in found else width


def rescore(name, run, test, processes, width=5):
    """Top-k with and without stereo from the beams a run wrote, over every test line."""
    beams = pickle.loads(run.read_bytes())
    lines = SETS[name][0]["test"]

    with Pool(processes) as pool:
        ranks = pool.map(_rank, [(test[i], cands, width) for i, cands in beams], chunksize=32)

    flat = [next((k for k, (_, _, ok) in enumerate(cands) if ok), width) for _, cands in beams]

    return {
        **{f"top{k}": sum(r < k for r in flat) / lines for k in (1, 5)},
        **{f"top{k}_stereo": sum(r < k for r in ranks) / lines for k in (1, 5)},
    }


def summary(args):
    """Mean and sample standard deviation over the seeds of every set, with and without stereo, next to the published
    numbers."""
    out, root = {}, Path(args.root)

    for name, (lines, published) in SETS.items():
        runs = sorted((root / name / "forward").glob("npf-nettargets-[0-9].json"))
        if not runs:
            continue

        data = chem.load(f"data/{name}.pkl")
        test = {r["id"]: r for r in data["reactions"] if r["split"] == "test"}
        results = [json.loads(p.read_text()) for p in runs]
        scores = [rescore(name, root / name / "beams" / f"{p.stem}-test.pkl", test, args.processes) for p in runs]

        # the stored flags of the beams give the numbers of the run itself
        for result, score in zip(results, scores):
            assert abs(score["top1"] - result["metrics"]["product_top1_beam_official"]) < 1e-9, name

        # precursors whose SMILES does not give their graph lend no stereo
        unread = sum(stereo_source(r["smiles"].split(">>")[0], r["a"]) is None for r in test.values())
        stat = lambda values: {
            "mean": 100 * float(np.mean(values)),
            "sd": 100 * float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "runs": [100 * v for v in values],
        }
        metric = lambda key: stat([r["metrics"][key] for r in results])
        out[name] = {
            "seeds": [r["seed"] for r in results],
            "top1": metric("product_top1_beam_official"),
            "top1_greedy": metric("product_top1_official"),
            "top5": metric("product_top5_beam_official"),
            "top1_stereo": stat([score["top1_stereo"] for score in scores]),
            "top5_stereo": stat([score["top5_stereo"] for score in scores]),
            "published": published,
            "test_lines": lines["test"],
            "test_scored": results[0]["n_test"],
            "test_products_with_stereo": data["stats"]["test"]["product_with_stereo"],
            "test_precursors_without_stereo_source": unread,
            "test_also_in_train": data["stats"]["test"]["also_in_train"],
            "train_lines": lines["train"],
            "train_used": results[0]["n_train"],
            "train_minutes_mean": float(np.mean([r["train_seconds"] for r in results])) / 60,
            "params": results[0]["params"],
        }
        mean = lambda key: f"{out[name][key]['mean']:.2f} +- {out[name][key]['sd'] or 0:.2f}"
        print(
            f"{name:11s} top-1 {mean('top1')}, with stereo {mean('top1_stereo')}, top-5 {mean('top5')}, with stereo "
            f"{mean('top5_stereo')}, published " + ", ".join(f"{p['model']} {p['top1']:.2f}" for p in published)
            + f", {len(results)} seeds",
            flush=True,
        )

    out["scoring"] = (
        "ours, top-k of the beam of five over every test line, the recorded product among the molecules of the "
        "marking (npflow.chem.product_found), without stereo, and with stereo taken from the precursors where no firing "
        "touched them. Each published row says how its paper treats stereo."
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["download", "build", "summary"])
    ap.add_argument("--sets", nargs="*", choices=list(SETS), help="build, only these sets")
    ap.add_argument("--processes", type=int, default=4)
    ap.add_argument("--root", default="results/small_forward", help="summary, the --root of the runs")
    args = ap.parse_args()
    {"download": download, "build": build, "summary": summary}[args.stage](args)


if __name__ == "__main__":
    main()
