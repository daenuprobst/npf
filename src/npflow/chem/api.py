"""Train, validate, test and use the models on reactions of your own, with PyTorch Lightning.

A reaction file holds one reaction SMILES per line, precursors>>product, with an optional tab separated class label.
No atom map is needed. The mapper of npflow.chem.mapper, the minimum firing vector of the valence net found without a
solver, computes the targets of every reaction once, milliseconds each for most reactions, and caches them next to the
file. map_reaction writes its map as map numbers.

    from npflow.chem import api
    data = api.ReactionData("train.txt", "val.txt", "test.txt")
    model = api.ForwardModel()
    trainer = api.trainer(model, epochs=60)
    trainer.fit(model, data)
    trainer.test(model, data, ckpt_path="best")
    api.predict(model, ["CC(=O)Cl.NCC"])
    api.map_reaction("CC(=O)Cl.NCC>>CC(=O)NCC")

    data = api.ReactionData("train.txt", "val.txt", "test.txt", task="classify")
    model = api.ClassifierModel(data.n_classes)
"""

import math
import os
import pickle
from multiprocessing import Pool
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from .classifier import Classifier
from .decode import marking_fragments
from .featurisation import batch_indices, collate, reaction
from .mapping import map_reaction
from .targets import attach, product_found, product_major, targets
from .token_game import TokenGame

__all__ = [
    "ClassifierModel",
    "ForwardModel",
    "ReactionData",
    "classify",
    "labels",
    "map_reaction",
    "predict",
    "read",
    "trainer",
]


def rows(path):
    return [
        line.split("\t") for line in Path(path).read_text().splitlines() if line.strip()
    ]


def labels(*paths):
    """The sorted class names of every file that has a label column."""
    return sorted(
        {row[1] for path in paths if path for row in rows(path) if len(row) > 1}
    )


def read(path, classes=()):
    """The reactions of a file, one SMILES per line with an optional tab separated class label, the label as its
    index in classes. Reactions RDKit cannot read are left out."""
    lines = rows(path)
    out = [
        reaction(
            row[0],
            label=classes.index(row[1]) if len(row) > 1 else 0,
            split=Path(path).stem,
            id=k,
        )
        for k, row in enumerate(lines)
    ]
    out = [r for r in out if r is not None]
    print(f"{path}: {len(out)} of {len(lines)} reactions read", flush=True)

    return out


def _targets(job):
    return targets(*job)


def with_targets(reactions, path, processes, seconds):
    """The reactions with the targets of the net attached, computed once per reaction and cached next to the file."""
    cache = Path(f"{path}.targets.pkl")
    rows = pickle.loads(cache.read_bytes()) if cache.exists() else {}
    todo = [r for r in reactions if r["smiles"] not in rows]

    if todo:
        print(
            f"{path}: targets of the net for {len(todo)} reactions on {processes} processes",
            flush=True,
        )

        with Pool(processes) as pool:
            for r, row in zip(
                todo, pool.imap(_targets, [(r, seconds) for r in todo], chunksize=4)
            ):
                rows[r["smiles"]] = row

        cache.write_bytes(pickle.dumps(rows))

    return [attach(r, rows[r["smiles"]]) for r in reactions]


def as_is(item):
    """Keeps the numpy arrays of the reactions as they are, the scoring reads them."""
    return item


class Batches(torch.utils.data.Dataset):
    """One epoch of padded batches, collated in worker processes. An item is the tensors and the reactions."""

    def __init__(self, reactions, index_batches):
        self.reactions, self.index_batches = reactions, index_batches

    def __len__(self):
        return len(self.index_batches)

    def __getitem__(self, k):
        rs = [self.reactions[i] for i in self.index_batches[k]]

        return collate(rs, "cpu"), rs


class ReactionData(L.LightningDataModule):
    """Reaction files as batches of similar size. task forward trains the token game on the reactions that have a
    target of the net, task classify trains a classifier on every reaction."""

    def __init__(
        self,
        train,
        val=None,
        test=None,
        task="forward",
        batch=32,
        seconds=3.0,
        processes=None,
        workers=4,
        classes=None,
        seed=0,
    ):
        super().__init__()
        self.paths = {"train": train, "val": val, "test": test}
        self.task, self.batch, self.seconds, self.workers, self.classes, self.seed = (
            task,
            batch,
            seconds,
            workers,
            classes,
            seed,
        )
        self.processes = processes or max(1, (os.cpu_count() or 2) - 2)
        self.reactions = {}

    def setup(self, stage=None):
        self.classes = self.classes or labels(*self.paths.values())

        for split, path in self.paths.items():
            if path and split not in self.reactions:
                self.reactions[split] = with_targets(
                    read(path, self.classes), path, self.processes, self.seconds
                )

    @property
    def n_classes(self):
        self.setup()

        return len(self.classes)

    def loader(self, split, shuffle=False):
        rs = self.reactions[split]

        # a reaction without a target of the net has nothing to teach the token game
        if split == "train" and self.task == "forward":
            rs = [r for r in rs if r.get("edits_set")]

        epoch = self.trainer.current_epoch if self.trainer is not None else 0
        rng = np.random.default_rng(self.seed + epoch) if shuffle else None

        return torch.utils.data.DataLoader(
            Batches(rs, batch_indices(rs, self.batch, rng)),
            batch_size=None,
            num_workers=self.workers,
            collate_fn=as_is,
        )

    def train_dataloader(self):
        return self.loader("train", shuffle=True)

    def val_dataloader(self):
        return self.loader("val")

    def test_dataloader(self):
        return self.loader("test")


class Schedule:
    """The optimiser of the paper, AdamW with a one-cycle learning rate over the whole run."""

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-5
        )
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            self.hparams.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }


class ForwardModel(Schedule, L.LightningModule):
    """The token game. Trains on the target sets of the net, scores greedy predictions by product top-1."""

    monitor = "val_top1"

    def __init__(
        self, width=128, rounds=6, attention=4, lr=1e-3, amp=True, compile_events=True
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = TokenGame(d=width, rounds=rounds, attention=attention)
        self.compiled = None

    def training_step(self, batch, _):
        b, rs = batch

        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=self.hparams.amp and self.device.type == "cuda",
        ):
            loss = self.net.loss(b)

        self.log("loss", loss, prog_bar=True, batch_size=len(rs))

        return loss

    def on_train_epoch_start(self):
        # the compiled rate law launches far fewer kernels, evaluation keeps the eager one so scores do not depend on it
        if self.hparams.compile_events and self.device.type == "cuda":
            self.compiled = self.compiled or torch.compile(
                self.net.events, dynamic=True
            )
            self.net.events = self.compiled

    def on_validation_epoch_start(self):
        self.net.__dict__.pop("events", None)

    on_test_epoch_start = on_predict_epoch_start = on_validation_epoch_start

    def validation_step(self, batch, _):
        self.score(batch, "val")

    def test_step(self, batch, _):
        self.score(batch, "test")

    @torch.no_grad()
    def score(self, batch, split):
        b, rs = batch
        preds = self.net.decode(self.net(b), b, rs)
        self.log(
            f"{split}_top1",
            float(np.mean([product_found(r, p) for r, p in zip(rs, preds)])),
            prog_bar=True,
            batch_size=len(rs),
        )
        self.log(
            f"{split}_major",
            float(np.mean([product_major(r, p) for r, p in zip(rs, preds)])),
            batch_size=len(rs),
        )


class ClassifierModel(Schedule, L.LightningModule):
    """The state-equation readout with the participation gate, or with sigma the explicit firing vector of the mapper."""

    monitor = "val_acc"

    def __init__(self, n_classes, sigma=False, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.net = Classifier(n_classes, gate=not sigma, explicit_firing=sigma)

    def training_step(self, batch, _):
        b, rs = batch
        loss = F.cross_entropy(self.net(b), b["label"]) + 0.5 * self.net.auxiliary_loss(
            b
        )
        self.log("loss", loss, prog_bar=True, batch_size=len(rs))

        return loss

    def validation_step(self, batch, _):
        self.score(batch, "val")

    def test_step(self, batch, _):
        self.score(batch, "test")

    @torch.no_grad()
    def score(self, batch, split):
        b, rs = batch
        self.log(
            f"{split}_acc",
            float((self.net(b).argmax(-1) == b["label"]).float().mean()),
            prog_bar=True,
            batch_size=len(rs),
        )


def trainer(model, epochs=60, root="runs", **kw):
    """A Lightning trainer with the settings of the paper, one device, gradient clipping, the best epoch kept."""
    return L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        gradient_clip_val=1.0,
        reload_dataloaders_every_n_epochs=1,
        default_root_dir=root,
        logger=kw.pop("logger", CSVLogger(root)),
        callbacks=[ModelCheckpoint(monitor=model.monitor, mode="max")],
        **kw,
    )


@torch.no_grad()
def predict(model, precursors, width=5, batch=16):
    """Ranked products of new precursors, one list per input of (molecules the firings touch, probability), None
    for precursors RDKit cannot read."""
    net, out = model.net.eval(), []
    for k in range(0, len(precursors), batch):
        # the product side is a placeholder, the token game reads the precursors only
        rs = [
            reaction(f"{smi}>>C", split="predict", id=k + i)
            for i, smi in enumerate(precursors[k : k + batch])
        ]
        ranked = iter(
            net.beam_search_batch(
                collate([r for r in rs if r is not None], model.device), width
            )
        )
        out += [
            (
                None
                if r is None
                else [
                    (".".join(sorted(marking_fragments(r["a"], e)[0])), math.exp(lp))
                    for e, lp in next(ranked)
                ]
            )
            for r in rs
        ]

    return out


@torch.no_grad()
def classify(model, reactions, classes, seconds=3.0, batch=32):
    """The class of every reaction SMILES, None where RDKit cannot read it. A model with sigma seats the product
    atoms with the mapper first."""
    net, out = model.net.eval(), []
    for k in range(0, len(reactions), batch):
        rs = [
            reaction(smi, split="predict", id=k + i)
            for i, smi in enumerate(reactions[k : k + batch])
        ]
        found = [attach(r, targets(r, seconds)) for r in rs if r is not None]
        labels = (
            iter(net(collate(found, model.device)).argmax(-1).tolist())
            if found
            else iter(())
        )
        out += [None if r is None else classes[next(labels)] for r in rs]

    return out
