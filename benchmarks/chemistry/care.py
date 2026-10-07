"""EC number classification on the CARE benchmark, Task 2, from the reaction alone.

CARE, Yang et al. 2024, asks for the enzyme commission number of a reaction. The easy split holds out reactions, not
EC numbers, so every test label is seen in training and the task is classification into 4960 classes with a median
of four training reactions each. We report k=1 accuracy at EC level 4, 3, 2 and 1 as the benchmark does, by
truncating the predicted EC number.

The same classifier runs on ECREACT in the split of Enzyformer (Liu et al. 2026), which benchmarks.chemistry.enzymes
prepares. With --maps it also reads the explicit firing vector of every reaction under the atom maps of a file in the
format of exact_map --write, and a reaction the file does not cover is read without one.

    uv run python -m benchmarks.chemistry.care prepare <path to CARE_datasets>   # data/care_easy.pkl
    uv run python -m benchmarks.chemistry.care train --seed 0                    # results/care/easy/<model>-<seed>.json
    uv run python -m benchmarks.chemistry.care train --maps data/enzyme_maps/mapper_maps_care_easy.pkl --tag=-mapper
    uv run python -m benchmarks.chemistry.care train --dataset ecreact_enzyformer # results/ecreact/enzyformer/
"""

import argparse
import json
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from npflow import chem

from .experiment import batches, epoch_loader, use_predicted_firing

DATA = Path("data/care_easy.pkl")
EPOCHS = 40

# prefix owners of the classes at levels 1 to 3, for the marginal rule on a model without coarse heads
OWNERS = None

# the published numbers are over every test reaction of the split, so the ones RDKit cannot read count as wrong
CARE_EASY_TEST = 393

# data file, name of the benchmark, size of its official test set and where the results go
DATASETS = {
    "care_easy": (DATA, "CARE task 2, easy split", CARE_EASY_TEST, Path("results/care/easy")),
    "ecreact_enzyformer": (
        Path("data/ecreact_enzyformer.pkl"),
        "ECREACT, Enzyformer split",
        4907,
        Path("results/ecreact/enzyformer"),
    ),
}


def prepare(root, out=DATA):
    """Featurise both splits and index the EC numbers. Every reaction is kept, the size bound of USPTO would drop the
    peptides and oligosaccharides of up to 323 atoms, and 1.4 GB of GPU memory holds a batch of the sixteen largest.
    """
    import csv

    base = Path(root) / "splits" / "task2"
    rows = {
        name: list(csv.DictReader(open(base / f"easy_reaction_{name}.csv")))
        for name in ("train", "test")
    }
    classes = sorted({r["EC number"] for r in rows["train"]})
    index = {ec: k for k, ec in enumerate(classes)}
    reactions, dropped, unseen = [], Counter(), 0

    for split, items in rows.items():
        for k, r in enumerate(items):
            if r["EC number"] not in index:
                unseen += 1
                continue

            smiles = r["Reaction"]
            g = chem.featurise(
                {
                    "original_rxn": smiles,
                    "rxn": smiles,
                    "label": index[r["EC number"]],
                    "split": split,
                    "id": len(reactions),
                },
                max_atoms=None,
            )
            if g is None:
                dropped[split] += 1
                continue

            reactions.append(g)

    out.write_bytes(pickle.dumps({"reactions": reactions, "classes": classes}))
    kept = Counter(r["split"] for r in reactions)
    print(
        f"{len(classes)} EC numbers, kept {kept['train']} train and {kept['test']} test reactions, "
        f"dropped {dropped['train']} and {dropped['test']} that could not be featurised, {unseen} with an unseen EC, wrote {out}"
    )


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


def prefixes(classes, depth):
    """Index of the EC prefix of every class at this depth, and the number of distinct prefixes."""
    names = sorted({".".join(c.split(".")[:depth]) for c in classes})
    index = {name: k for k, name in enumerate(names)}

    return np.array([index[".".join(c.split(".")[:depth])] for c in classes]), len(
        names
    )


class Hierarchy(nn.Module):
    """The classifier with one auxiliary head per coarser EC level on the same readout. A flat loss over 4960 classes
    treats a sibling EC like an unrelated one, so the coarse levels get no gradient of their own, the extra heads give
    them one and vote on the prefix at test time."""

    def __init__(self, classes, **kwargs):
        super().__init__()
        self.body = chem.Classifier(len(classes), **kwargs)
        width = self.body.out[0].in_features
        self.levels = nn.ModuleList()

        for depth in (1, 2, 3):
            owner, n = prefixes(classes, depth)
            self.levels.append(
                nn.Sequential(nn.Linear(width, 256), nn.SiLU(), nn.Linear(256, n))
            )
            self.register_buffer(
                f"owner{depth}", torch.as_tensor(owner, dtype=torch.long)
            )

        # the readout is the input of the body's last block, a hook keeps it for the coarse heads
        self._readout = None
        self.body.out.register_forward_pre_hook(
            lambda _, inputs: setattr(self, "_readout", inputs[0])
        )

    def forward(self, b):
        fine = self.body(b)

        return fine, [level(self._readout) for level in self.levels]

    def owners(self):
        return [self.owner1, self.owner2, self.owner3]


def decode(fine, coarse, owners, rule):
    """One predicted EC per reaction, so the benchmark's metric applies unchanged. flat takes the most probable class,
    marginal adds for every class the log of the total probability of its prefix at each level, so a class with likely
    siblings wins over an isolated one, and hierarchy adds the log probability the coarse heads give its prefixes.
    """
    log_p = torch.log_softmax(fine, 1)

    if rule == "flat":
        return log_p.argmax(1)

    score = log_p.clone()
    for depth, owner in enumerate(owners):
        if rule == "marginal":
            n = int(owner.max()) + 1
            mass = torch.zeros(len(fine), n, device=fine.device).index_add_(
                1, owner, log_p.exp()
            )
            score = score + torch.log(mass.clamp(min=1e-30))[:, owner]
        else:
            score = score + torch.log_softmax(coarse[depth], 1)[:, owner]

    return score.argmax(1)


@torch.no_grad()
def predict(model, reactions, device, batch=16, rule="flat"):
    model.eval()
    out, truth = [], []

    for rs in batches(reactions, batch):
        b = chem.collate(rs, device)

        if isinstance(model, Hierarchy):
            fine, coarse = model(b)
            owners = model.owners()
        else:
            fine, coarse, owners = model(b), None, OWNERS

        out += decode(fine, coarse, owners, rule).cpu().tolist()
        truth += [r["label"] for r in rs]

    return out, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prepare", "train"])
    ap.add_argument(
        "root", nargs="?", help="the unpacked CARE_datasets directory, for prepare"
    )
    ap.add_argument(
        "--model",
        default="npf",
        help="npf, the state-equation readout, or pgnn, the generic readout",
    )
    ap.add_argument("--dataset", choices=list(DATASETS), default="care_easy")
    ap.add_argument(
        "--maps",
        help="read the explicit firing vector under the atom maps of this file, written by exact_map --write",
    )
    ap.add_argument("--tag", default="", help="appended to the name of the result files")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument(
        "--hierarchy",
        type=float,
        default=0.0,
        help="weight of the auxiliary losses at EC levels 1 to 3",
    )
    args = ap.parse_args()

    if args.phase == "prepare":
        prepare(args.root)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    path, benchmark, n_official, out = DATASETS[args.dataset]
    data = pickle.loads(path.read_bytes())
    if args.maps:
        use_predicted_firing(data["reactions"], path=args.maps, blank_missing=True)

    classes = data["classes"]
    train = [r for r in data["reactions"] if r["split"] == "train"]
    test = [r for r in data["reactions"] if r["split"] == "test"]
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    global OWNERS
    OWNERS = [
        torch.as_tensor(prefixes(classes, depth)[0], dtype=torch.long, device=device)
        for depth in (1, 2, 3)
    ]

    # the gate is left unsupervised, an atom map is read only with --maps, as the explicit firing vector
    kwargs = dict(
        petri=args.model.startswith("npf"),
        gate=args.model == "npf",
        explicit_firing=bool(args.maps),
    )
    model = (
        Hierarchy(classes, **kwargs)
        if args.hierarchy
        else chem.Classifier(len(classes), **kwargs)
    ).to(device)
    tag = args.tag + (f"-hier{args.hierarchy:g}" if args.hierarchy else "")
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = args.epochs * (len(train) // args.batch + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, 1e-3, total_steps=steps, pct_start=0.1
    )
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        losses = []

        # collation runs in worker processes, the batches and their order are the same as before
        for b in epoch_loader(train, args.batch, rng, device):
            if args.hierarchy:
                fine, coarse = model(b)
                loss = F.cross_entropy(fine, b["label"])
                for owner, logits in zip(model.owners(), coarse):
                    loss = loss + args.hierarchy * F.cross_entropy(
                        logits, owner[b["label"]]
                    )
            else:
                loss = F.cross_entropy(model(b), b["label"])

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())

        if epoch % 5 == 4 or epoch == args.epochs - 1:
            accuracy = levels(*predict(model, test, device), classes)
            print(
                f"epoch {epoch}: loss {np.mean(losses):.4f}  level4 {accuracy['level4']:.4f}  "
                f"level1 {accuracy['level1']:.4f}  ({time.time() - start:.0f}s)",
                flush=True,
            )

    metrics = levels(*predict(model, test, device), classes, n_official)
    rules = ["marginal"] + (["hierarchy"] if args.hierarchy else [])

    # the same model read with a decoding rule that respects the EC hierarchy
    for rule in rules:
        metrics.update(
            {
                f"{k}_{rule}": v
                for k, v in levels(
                    *predict(model, test, device, rule=rule), classes, n_official
                ).items()
            }
        )

    result = {
        "benchmark": benchmark,
        "model": args.model,
        "seed": args.seed,
        "params": params,
        "n_train": len(train),
        "n_test": len(test),
        "n_test_official": n_official,
        "n_classes": len(classes),
        "train_seconds": time.time() - start,
        "metrics": metrics,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.model}{tag}-{args.seed}.json").write_text(
        json.dumps(result, indent=1)
    )
    torch.save(model.state_dict(), out / f"{args.model}{tag}-{args.seed}.pt")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
