"""A D-MPNN on the condensed graph of reaction (Chemprop 2.3.1, Heid and Green) as an established baseline of the
Schneider 50k classification of the paper, on the same maps, splits and label draws as the classifiers of the net.

    uv run python -m benchmarks.chemistry.baselines.chemprop_cgr --seed 0                  # all training reactions
    uv run python -m benchmarks.chemistry.baselines.chemprop_cgr --seed 0 --labels 250     # 250 labelled reactions
    uv run python -m benchmarks.chemistry.baselines.chemprop_cgr --seed 0 --size-split     # larger test molecules
    uv run python -m benchmarks.chemistry.baselines.chemprop_cgr --seed 0 --mode REAC_DIFF_BALANCE
    uv run python -m benchmarks.chemistry.baselines.chemprop_cgr --seed 0 --hidden 850      # 2.4M parameters
    uv run python -m benchmarks.chemistry.baselines.chemprop_cgr --seed 0 --labels 250 --matched-steps

Data. The reactions, the 500 validation reactions, the 39,994 test reactions, the stratified label draws of a seed and
the size split come from benchmarks.chemistry.experiment, so every run sees exactly the reactions of the classifier
run of the same seed. Every reaction carries the atom maps of the mapper (data/exact_maps_schneider50k.pkl) that the
firing-vector classifier reads, written into its SMILES by npflow.chem.mapping.mapped_smiles, and a reaction the mapper
leaves without a complete map enters unmapped, as it does for the classifier.

Model. Chemprop's own command line with its defaults (hidden size 300, depth 3, norm aggregation, 50 epochs, batch
64, learning rate 1e-4 to 1e-3 to 1e-4), multiclass over the 50 classes, and nothing tuned. The reaction mode is
Chemprop's default REAC_DIFF, and REAC_DIFF_BALANCE, which keeps the atoms that leave from counting as broken bonds,
is the second arm. The third arm is the default model with message and output widths of 850 instead of 300, 2.4M
parameters instead of 357K, close to the firing-vector classifier (2.47M). Chemprop runs in its own environment
through uv, so the project keeps its dependencies.

Steps. With a label budget, Chemprop's 50 epochs are 200 optimiser steps at 250 labels, and its training loss is
still at 2.66 against 3.91 at the start, so --matched-steps scales the epochs to the steps of the default run on all
9,500 training reactions, 50 x 149, which gives the baseline far more steps than the classifiers of the net get
(60 epochs). This arm was added after the default runs at 250 labels had been seen.

Selection. With all training reactions and on the size split the checkpoint of the lowest validation loss is kept,
Chemprop's default, and with --labels no validation reaction enters at all, so Chemprop keeps the checkpoint of the
lowest training loss, which reads no validation label, as --select last does for the classifiers of the net. The last
epoch itself is not available, since Chemprop rewrites last.ckpt only together with the checkpoint it keeps. The test reactions are read once, by the kept checkpoint, and a reaction without a prediction
counts as wrong. The result goes to results/chem/classify/cgr-dmpnn[-balance][-labels<n>|-sizesplit]-<seed>.json in
the format of the other classifier runs.
"""

import argparse
import csv
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from npflow import chem
from npflow.chem.mapping import mapped_smiles

from ..experiment import splits, stratified

MAPS = Path("data/exact_maps_schneider50k.pkl")
CHEMPROP = ["uv", "run", "--no-project", "--python", "3.12", "--with", "chemprop==2.3.1", "chemprop"]


def with_maps(r, maps):
    """Reaction SMILES with the maps of the mapper, or without maps when the mapper has no complete map, the rule of
    use_predicted_firing with blank_missing."""
    mapping = maps.get(r["id"])

    if mapping is None:
        return unmapped(r["smiles"])

    mapping = np.asarray(mapping, np.int64)

    if len(mapping) != len(r["b"]["x"]) or (mapping < 0).any() or (mapping >= len(r["a"]["x"])).any():
        return unmapped(r["smiles"])

    return mapped_smiles(r["smiles"], mapping)


def unmapped(smiles):
    """precursors>>products without atom maps, reagents joined to the precursors as mapped_smiles does."""
    reactants, reagents, products = smiles.split(">")

    return ".".join(s for s in (reactants, reagents) if s) + ">>" + products


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rxn", "label", "split", "id"])
        w.writerows(rows)


def chemprop(args, log):
    with open(log, "a") as f:
        f.write("$ " + " ".join(args) + "\n")
        f.flush()
        subprocess.run(CHEMPROP + args, stdout=f, stderr=subprocess.STDOUT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labels", type=int, help="labelled training reactions, the same stratified draw as the classifiers")
    ap.add_argument("--size-split", action="store_true", help="train on the smaller half, test on the largest quarter")
    ap.add_argument("--mode", default="REAC_DIFF", choices=["REAC_DIFF", "REAC_DIFF_BALANCE"])
    ap.add_argument("--hidden", type=int, help="message and output width, Chemprop's default is 300")
    ap.add_argument("--epochs", type=int, help="only for smoke tests, Chemprop's default is 50")
    ap.add_argument("--matched-steps", action="store_true", help="with --labels, as many optimiser steps as the default run on all training reactions")
    ap.add_argument("--smoke", action="store_true", help="score the validation reactions, never the test reactions")
    ap.add_argument("--root", default="results/chem")
    args = ap.parse_args()

    tag = ("-balance" if args.mode == "REAC_DIFF_BALANCE" else "") + (f"-h{args.hidden}" if args.hidden else "") + (
        f"-labels{args.labels}" if args.labels else ""
    ) + ("-sizesplit" if args.size_split else "") + ("-steps" if args.matched_steps else "")
    name = f"cgr-dmpnn{tag}-{args.seed}" + ("-smoke" if args.smoke else "")
    out = Path(args.root) / "chemprop" / name
    result = Path(args.root) / "classify" / f"{name}.json"

    if result.exists() and not args.smoke:
        print(f"{result} exists")
        return

    out.mkdir(parents=True, exist_ok=True)
    log = out / "chemprop.log"
    data = chem.load("data/schneider50k.pkl")
    train, val, test = splits(data, "classify")

    # the label draw indexes the training list before any other filter, as in experiment.main
    if args.labels:
        chosen = stratified(train, args.labels, args.seed)
        train = [r for k, r in enumerate(train) if k in chosen]

    if args.size_split:
        size = lambda r: len(r["a"]["x"])
        small, large = np.median([size(r) for r in train]), np.quantile([size(r) for r in test], 0.75)
        train, val, test = (
            [r for r in train if size(r) <= small],
            [r for r in val if size(r) <= small],
            [r for r in test if size(r) >= large],
        )

    if args.smoke:
        test = val

    # Chemprop's default run on all 9,500 training reactions, 50 epochs of ceil(9500 / 64) batches
    if args.matched_steps:
        batches = lambda n: -(-n // 64)
        args.epochs = round(50 * batches(9500) / batches(len(train)))

    import pickle

    maps = pickle.loads(MAPS.read_bytes())
    rows = lambda rs, part: [[with_maps(r, maps), r["label"], part, r["id"]] for r in rs]
    # with a label budget no validation reaction enters, Chemprop then keeps the checkpoint of the lowest training
    # loss, and its last.ckpt is only rewritten when that checkpoint is, so it cannot give the last epoch otherwise
    write_csv(out / "train.csv", rows(train, "train") + ([] if args.labels else rows(val, "val")))
    write_csv(out / "test.csv", rows(test, "test"))
    print(f"{name}: {len(train)} training, {len(val)} validation, {len(test)} test reactions", flush=True)

    start = time.time()
    save = out / "model"
    common = ["--reaction-columns", "rxn", "--rxn-mode", args.mode, "--accelerator", "gpu", "--devices", "1"]
    chemprop(
        ["train", "--data-path", str(out / "train.csv"), "--splits-column", "split", "--target-columns", "label",
         "--task-type", "multiclass", "--multiclass-num-classes", str(len(data["classes"])), "--save-dir", str(save),
         "--pytorch-seed", str(args.seed), "--data-seed", str(args.seed)]
        + (["--epochs", str(args.epochs)] if args.epochs else [])
        + (["--message-hidden-dim", str(args.hidden), "--ffn-hidden-dim", str(args.hidden)] if args.hidden else [])
        + common,
        log,
    )
    train_seconds = time.time() - start

    # the checkpoint of the lowest validation loss, or of the lowest training loss when no validation reaction entered
    checkpoints = sorted(save.glob("**/checkpoints/*.ckpt"))
    best = [c for c in checkpoints if c.name.startswith("best")]
    kept = best[-1]
    preds = out / "test_preds.csv"
    chemprop(
        ["predict", "--test-path", str(out / "test.csv"), "--model-paths", str(kept), "--preds-path", str(preds)]
        + common,
        log,
    )

    # predictions by row in pred_0, the input columns and so the true label are copied, a reaction Chemprop could not
    # read counts as wrong
    with open(preds) as f:
        predicted = {int(row["id"]): int(float(row["pred_0"])) for row in csv.DictReader(f) if row.get("pred_0") not in ("", None)}

    classes = len(data["classes"])
    tp, pred_n, true_n = np.zeros(classes), np.zeros(classes), np.zeros(classes)
    correct = 0
    for r in test:
        p = predicted.get(r["id"], -1)
        true_n[r["label"]] += 1
        if p >= 0:
            pred_n[p] += 1
            tp[r["label"]] += p == r["label"]
        correct += p == r["label"]

    f1 = 2 * tp / np.maximum(pred_n + true_n, 1)
    metrics = {"accuracy": correct / len(test), "macro_f1": float(f1.mean()), "missing": len(test) - sum(r["id"] in predicted for r in test)}

    # parameters from Lightning's summary in the log
    params = re.findall(r"Total params: ([\d.]+ ?[KM]?)", log.read_text())
    record = {
        "task": "classify",
        "model": f"cgr-dmpnn{'-balance' if args.mode == 'REAC_DIFF_BALANCE' else ''}{f'-h{args.hidden}' if args.hidden else ''}",
        "seed": args.seed,
        "args": vars(args) | {"maps": str(MAPS), "maps_sha256": hashlib.sha256(MAPS.read_bytes()).hexdigest(),
                              "chemprop": "2.3.1", "checkpoint": kept.name, "select": "lowest train_loss, no validation reactions" if args.labels else "best val_loss"},
        "params": params[-1].strip() if params else None,
        "n_train": len(train),
        "n_test": len(test),
        "train_seconds": train_seconds,
        "metrics": metrics,
    }

    if args.smoke:
        print(json.dumps(record, indent=1))
        return

    result.write_text(json.dumps(record, indent=1))
    print(f"chem/classify {name}: accuracy={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
