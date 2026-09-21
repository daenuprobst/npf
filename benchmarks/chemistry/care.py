"""EC number classification on the CARE benchmark, Task 2, from the reaction alone.

CARE \\citep{yang2024care} asks for the enzyme commission number of a reaction. The easy split holds out reactions,
not EC numbers, so every test label is seen in training and the task is classification into 4\\,960 classes with a
median of four training reactions each. We report k=1 accuracy at EC level 4, 3, 2 and 1 as the benchmark does, by
truncating the predicted EC number.

    uv run python -m benchmarks.chemistry.care prepare <path to CARE_datasets>   # data/care_easy.pkl
    uv run python -m benchmarks.chemistry.care train --seed 0                    # results/care/easy/<model>-<seed>.json
"""
import argparse
import json
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from npf import chem

from .experiment import batch_indices, batches

DATA = Path("data/care_easy.pkl")
EPOCHS = 40

# the published numbers are over every test reaction of the split, so the ones RDKit cannot read count as wrong
CARE_EASY_TEST = 393


def prepare(root, out=DATA):
    """Featurise both splits and index the EC numbers. Reactions RDKit cannot read are dropped and counted."""
    import csv

    base = Path(root) / "splits" / "task2"
    rows = {name: list(csv.DictReader(open(base / f"easy_reaction_{name}.csv"))) for name in ("train", "test")}
    classes = sorted({r["EC number"] for r in rows["train"]})
    index = {ec: k for k, ec in enumerate(classes)}
    reactions, dropped, unseen = [], Counter(), 0

    for split, items in rows.items():
        for k, r in enumerate(items):
            if r["EC number"] not in index:
                unseen += 1
                continue

            smiles = r["Reaction"]
            g = chem.featurise({"original_rxn": smiles, "rxn": smiles, "label": index[r["EC number"]], "split": split,
                                "id": len(reactions)})
            if g is None:
                dropped[split] += 1
                continue

            reactions.append(g)

    out.write_bytes(pickle.dumps({"reactions": reactions, "classes": classes}))
    kept = Counter(r["split"] for r in reactions)
    print(f"{len(classes)} EC numbers; kept {kept['train']} train and {kept['test']} test reactions, "
          f"dropped {dropped['train']} and {dropped['test']} that RDKit could not read, {unseen} with an unseen EC; wrote {out}")


def levels(predicted, true, classes, denominator=None):
    """k=1 accuracy at EC level 4, 3, 2 and 1, by truncating both EC numbers to that many fields."""
    out = {}
    for depth in (4, 3, 2, 1):
        cut = lambda k: ".".join(classes[k].split(".")[:depth])
        hits = sum(cut(p) == cut(t) for p, t in zip(predicted, true))
        out[f"level{depth}"] = hits / len(predicted)

        if denominator:
            out[f"level{depth}_official"] = hits / denominator

    return out


@torch.no_grad()
def predict(model, reactions, device, batch=16):
    model.eval()
    out, truth = [], []

    for rs in batches(reactions, batch):
        b = chem.collate(rs, device)
        out += model(b).argmax(1).cpu().tolist()
        truth += [r["label"] for r in rs]

    return out, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prepare", "train"])
    ap.add_argument("root", nargs="?", help="the unpacked CARE_datasets directory, for prepare")
    ap.add_argument("--model", default="npf", help="npf (state-equation readout) or pgnn (generic readout)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    if args.phase == "prepare":
        prepare(args.root)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = pickle.loads(DATA.read_bytes())
    classes = data["classes"]
    train = [r for r in data["reactions"] if r["split"] == "train"]
    test = [r for r in data["reactions"] if r["split"] == "test"]
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # the gate is left unsupervised, no atom map of any kind is read
    model = chem.Classifier(len(classes), petri=args.model.startswith("npf"), gate=args.model == "npf").to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = args.epochs * (len(train) // args.batch + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.1)
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        losses = []

        for idx in batch_indices(train, args.batch, rng):
            b = chem.collate([train[i] for i in idx], device)
            loss = F.cross_entropy(model(b), b["label"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())

        if epoch % 5 == 4 or epoch == args.epochs - 1:
            accuracy = levels(*predict(model, test, device), classes)
            print(f"epoch {epoch}: loss {np.mean(losses):.4f}  level4 {accuracy['level4']:.4f}  "
                  f"level1 {accuracy['level1']:.4f}  ({time.time() - start:.0f}s)", flush=True)

    metrics = levels(*predict(model, test, device), classes, CARE_EASY_TEST)
    result = {"benchmark": "CARE task 2, easy split", "model": args.model, "seed": args.seed, "params": params,
              "n_train": len(train), "n_test": len(test), "n_test_official": CARE_EASY_TEST, "n_classes": len(classes),
              "train_seconds": time.time() - start, "metrics": metrics}
    out = Path("results/care/easy")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.model}-{args.seed}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
