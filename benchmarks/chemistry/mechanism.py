"""Elementary step prediction on the FlowER mechanism benchmark (Joung et al., Nature 2025) with the arrow net.

The benchmark imputes arrow-pushing pathways for USPTO-Full reactions with expert templates and adds the curated steps
of PMechDB and RMechDB. Every line is one elementary step, fully atom mapped with explicit hydrogens. A prediction is
correct if the SMILES of its whole product side, without maps and stereochemistry and with every hydrogen written,
equals the recorded one, the criterion of the FlowER code. The results use the first release of the data,
flower_dataset with 1,445,189 / 15,744 / 162,002 steps. Every test step counts, also the single electron steps the
arrow net cannot express, which are wrong by construction.

The electron net of electron.py moves single electrons as well, so it expresses the radical steps too. It is chosen
with --net electron and keeps its own prepared data, weights and results.

    uv run python -m benchmarks.chemistry.mechanism download              # data/flower/2025/data/flower_dataset/
    uv run python -m benchmarks.chemistry.mechanism prepare               # data/flower/npf/{train,val,test}/
    uv run python -m benchmarks.chemistry.mechanism train --seed 0        # results/mechanism/npf-0.pt
    uv run python -m benchmarks.chemistry.mechanism evaluate --seed 0     # results/mechanism/npf-0.json
    uv run python -m benchmarks.chemistry.mechanism prepare --net electron
    uv run python -m benchmarks.chemistry.mechanism train --net electron --seed 0
    uv run python -m benchmarks.chemistry.mechanism evaluate --net electron --seed 0
"""

import argparse
import hashlib
import json
import os
import pickle
import random
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from npf.chem import arrows, electron
from npf.chem.arrow_game import MAX_ARROWS, ArrowGame
from npf.chem.electron_game import KINDS, ElectronGame

SOURCE = Path("data/flower/2025/data/flower_dataset")
RESULTS = Path("results/mechanism")

# the published data of Joung et al., figshare 10.6084/m9.figshare.28359407, MIT licence
FIGSHARE = "https://api.figshare.com/v2/articles/28359407/files"
ARCHIVE = "https://ndownloader.figshare.com/files/55904909"
ARCHIVE_SHA = "8e2333563bdf94051181898b4a77adf3e34bf696a6c33e03823f5ea77aa23ee5"
SPLIT_LINES = {"train": 1445189, "val": 15744, "test": 162002}
OK, RADICAL, UNCOVERED, FAILED = 0, 1, 2, 3
TABLES = {}

# the arrow net moves electron pairs, the electron net moves single electrons too and expresses radical steps
NETS = {
    "arrow": {
        "module": arrows,
        "game": ArrowGame,
        "atom": ("z", "q", "lone", "odd"),
        "state": ("z", "odd", "cap", "lo", "hi", "q", "lone"),
        "bonds": "bonds",
        "kinds": 2,
        "data": Path("data/flower/npf"),
        "name": "npf",
    },
    "electron": {
        "module": electron,
        "game": ElectronGame,
        "atom": ("z", "q2", "nb"),
        "state": ("z", "cap", "lo", "hi", "q2", "nb"),
        "bonds": "be",
        "kinds": 4,
        "data": Path("data/flower/npf-electron"),
        "name": "npf-electron",
    },
}
NET = NETS["arrow"]


def net():
    return NET["module"]


def data_folder():
    return NET["data"]


def download(args):
    """Fetch and unpack the published split, so a fresh machine needs nothing but this repository."""
    if all((SOURCE / f"{name}.txt").exists() for name in SPLIT_LINES):
        print(f"{SOURCE} is already there")
    else:
        root = SOURCE.parent.parent
        root.mkdir(parents=True, exist_ok=True)
        archive = root / "data.zip"

        if not archive.exists() or _digest(archive) != ARCHIVE_SHA:
            url = ARCHIVE

            # the file id may change, the article does not
            try:
                with urllib.request.urlopen(FIGSHARE, timeout=60) as f:
                    listed = json.load(f)

                url = next(x["download_url"] for x in listed if x["name"] == "data.zip")
            except Exception as error:
                print(f"figshare listing failed ({error}), using the recorded link")

            print(f"downloading {url} to {archive}, 238 MB", flush=True)
            urllib.request.urlretrieve(url, archive)

        got = _digest(archive)
        assert got == ARCHIVE_SHA, f"checksum of {archive} is {got}, expected {ARCHIVE_SHA}"

        # the archive also carries the 2026 re-release, which the published numbers are not on
        with zipfile.ZipFile(archive) as z:
            wanted = [n for n in z.namelist() if "flower_dataset/" in n and "new" not in n]
            print(f"unpacking {len(wanted)} files", flush=True)
            z.extractall(root, members=wanted)

    for name, expected in SPLIT_LINES.items():
        path = SOURCE / f"{name}.txt"
        n = sum(1 for _ in path.open())
        assert n == expected, f"{path} has {n} lines, expected {expected}"
        print(f"{path} {n} steps")


def _digest(path, chunk=1 << 20):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)

    return h.hexdigest()


def _stats(lines):
    shells, charges = defaultdict(Counter), defaultdict(Counter)
    for line in lines:
        try:
            m_a, m_b, _ = net().step(line)
        except Exception:
            continue

        # the tables are per element octet capacities in pairs and whole charge windows, the same for both nets
        for m in (m_a, m_b):
            pairs = np.ceil(net().shell(m) / 2).astype(np.int64)
            whole = m["q"] if "q" in m else m["q2"] // 2
            for z, s, q in zip(m["z"].tolist(), pairs.tolist(), whole.tolist()):
                shells[z][s] += 1
                charges[z][q] += 1

    return shells, charges


def _init(octet, window, targets):
    TABLES["octet"], TABLES["window"], TABLES["targets"] = octet, window, targets


def _record(line):
    """Status, start marking, one enabled order of the recorded arrows and the scoring form of the recorded product."""
    reaction, pathway = line.rsplit("|", 1)
    target = net().target_form(reaction.split(">>")[1]) if TABLES["targets"] else None

    try:
        m_a, m_b, _ = net().step(line)
    except AssertionError as e:
        if str(e) != "a single electron moves":
            return FAILED, None, [], pathway, target

        return RADICAL, net().molecule(reaction.split(">>")[0]), [], pathway, target
    except Exception:
        return FAILED, None, [], pathway, target

    cap, lo, hi = net().limits(m_a, TABLES["octet"], TABLES["window"])
    for k, solution in enumerate(net().decompose(m_a, m_b)):
        seq = net().Orders(m_a, solution, cap, lo, hi).first()
        if seq is not None:
            return OK, m_a, seq, pathway, target

        if k > 50:
            break

    return UNCOVERED, m_a, [], pathway, target


def _chunks(lines, size=2000):
    return [lines[k : k + size] for k in range(0, len(lines), size)]


def _records(lines):
    """One chunk of steps as compact arrays, so that the main process never holds a Python object per step."""
    names = (
        "status",
        "sizes",
        "n_bonds",
        "n_arrows",
        "bonds",
        "arrows",
        "pathway",
        "target",
    )
    out = {name: [] for name in names + NET["atom"]}
    bonds, width = NET["bonds"], 3 + (NET["kinds"] > 2)
    for line in lines:
        status, m, seq, pathway, target = _record(line)
        out["status"].append(status)
        out["pathway"].append(pathway)
        out["target"].append(target)
        rows = (
            np.array([(i, j, o) for (i, j), o in m[bonds].items()], np.int16).reshape(
                -1, 3
            )
            if m is not None
            else np.zeros((0, 3), np.int16)
        )
        out["sizes"].append(len(m["z"]) if m is not None else 0)
        out["n_bonds"].append(len(rows))
        out["n_arrows"].append(len(seq))
        out["bonds"].append(rows)
        out["arrows"].append(np.array(seq, np.int16).reshape(-1, width))

        if m is not None:
            for name in NET["atom"]:
                out[name].append(m[name].astype(np.int16))

    for name in NET["atom"]:
        out[name] = np.concatenate(out[name]) if out[name] else np.zeros(0, np.int16)

    for name in ("bonds", "arrows"):
        out[name] = np.concatenate(out[name])

    for name in ("status", "sizes", "n_bonds", "n_arrows"):
        out[name] = np.array(out[name], np.int64)

    return out


def save(split, chunks):
    """Ragged arrays, one row per step, so that the training set loads with memory mapping."""
    folder = data_folder() / split
    folder.mkdir(parents=True, exist_ok=True)
    cat = lambda name: np.concatenate([c[name] for c in chunks])
    offsets = lambda name: np.concatenate([[0], np.cumsum(cat(name))]).astype(np.int64)
    arrays = {name: cat(name).astype(np.int16) for name in NET["atom"]}
    arrays |= {
        "atoms": offsets("sizes"),
        "edges": offsets("n_bonds"),
        "fired": offsets("n_arrows"),
        "bonds": cat("bonds").astype(np.int16),
        "arrows": cat("arrows").astype(np.int16),
        "status": cat("status").astype(np.int8),
    }
    for name, array in arrays.items():
        np.save(folder / f"{name}.npy", array)

    meta = {
        "pathway": [p for c in chunks for p in c["pathway"]],
        "target": [t for c in chunks for t in c["target"]],
    }
    (folder / "meta.pkl").write_bytes(pickle.dumps(meta))

    return arrays["status"]


def prepare(args):
    # the per element tables come from the training markings, and a rerun reuses them
    if (data_folder() / "tables.pkl").exists():
        tables = pickle.loads((data_folder() / "tables.pkl").read_bytes())
        octet, window = tables["octet"], tables["window"]
    else:
        with Pool(args.processes) as pool:
            lines = (SOURCE / "train.txt").read_text().splitlines()
            shells, charges = defaultdict(Counter), defaultdict(Counter)
            for s, c in pool.imap_unordered(_stats, _chunks(lines)):
                for z in s:
                    shells[z].update(s[z])
                    charges[z].update(c[z])

        # the two nets share one table format, octet capacities in pairs and whole charges
        octet, window = arrows.tables(shells, charges)
        data_folder().mkdir(parents=True, exist_ok=True)
        (data_folder() / "tables.pkl").write_bytes(
            pickle.dumps({"octet": octet, "window": window})
        )

    for split in ("val", "test", "train"):
        lines = (SOURCE / f"{split}.txt").read_text().splitlines()
        start = time.time()

        with Pool(
            args.processes,
            initializer=_init,
            initargs=(octet, window, split != "train"),
        ) as pool:
            chunks = list(pool.imap(_records, _chunks(lines)))

        counts = Counter(save(split, chunks).tolist())
        print(
            f"{split}: {len(lines)} steps, ok {counts[OK]}, single electron {counts[RADICAL]}, no enabled order {counts[UNCOVERED]}, "
            f"unreadable {counts[FAILED]} ({time.time() - start:.0f}s)",
            flush=True,
        )


class Steps(torch.utils.data.Dataset):
    """Stored steps of one split, with the markings as dictionaries and the per atom limits of the arrow net."""

    def __init__(self, split):
        folder = data_folder() / split
        self.a = {p.stem: np.load(p, mmap_mode="r") for p in folder.glob("*.npy")}
        self.meta = pickle.loads((folder / "meta.pkl").read_bytes())
        tables = pickle.loads((data_folder() / "tables.pkl").read_bytes())
        self.octet, self.window = tables["octet"], tables["window"]
        self.seed, self.epoch = 0, 0

    def __len__(self):
        return len(self.a["status"])

    def size(self, k):
        return int(self.a["atoms"][k + 1] - self.a["atoms"][k])

    def marking(self, k):
        a, b = self.a["atoms"][k], self.a["atoms"][k + 1]
        rows = self.a["bonds"][self.a["edges"][k] : self.a["edges"][k + 1]]
        m = {name: np.asarray(self.a[name][a:b], np.int64) for name in NET["atom"]}
        m[NET["bonds"]] = {(int(i), int(j)): int(o) for i, j, o in rows}

        return m

    def recorded(self, k):
        return [
            tuple(int(x) for x in row)
            for row in self.a["arrows"][self.a["fired"][k] : self.a["fired"][k + 1]]
        ]

    def __getitem__(self, k):
        # a random enabled part of the recorded arrows, the state it reaches and the arrows that may come next
        rng = random.Random(hash((self.seed, self.epoch, k)))
        m = self.marking(k)
        cap, lo, hi = net().limits(m, self.octet, self.window)
        orders = net().Orders(m, self.recorded(k), cap, lo, hi)
        rest, prefix = orders.arrows, []
        for _ in range(rng.randint(0, len(rest))):
            arrow = rng.choice(orders.moves(rest))
            prefix.append(arrow)
            at = rest.index(arrow)
            rest = rest[:at] + rest[at + 1 :]

        state = net().apply(m, prefix)
        fixed = {"cap": cap, "lo": lo, "hi": hi}

        return {
            name: (fixed.get(name) if name in fixed else state.get(name, m.get(name)))
            for name in NET["state"]
        } | {
            NET["bonds"]: state[NET["bonds"]],
            "prefix": prefix,
            "targets": orders.moves(rest) if rest else [],
            "stop": not rest,
        }


def slot(arrow):
    """Index of a transition among the kinds of the net, the pair net has two and the electron net four."""
    if NET["kinds"] == 2:
        return arrow[0]

    return KINDS.index((arrow[0], arrow[3]))


def collate(items):
    """Pads a list of states to the largest step of the batch."""
    batch, n = len(items), max(len(it["z"]) for it in items)
    t = {name: torch.zeros(batch, n, dtype=torch.long) for name in NET["state"]}
    t["mask"] = torch.zeros(batch, n, dtype=torch.bool)
    t["touched"] = torch.zeros(batch, n, dtype=torch.bool)

    for name in ("cur", "fired_a", "fired_b"):
        t[name] = torch.zeros(batch, n, n, dtype=torch.long)

    t["target"] = torch.zeros(batch, n, n, NET["kinds"], dtype=torch.bool)
    t["target_stop"] = torch.tensor([it.get("stop", False) for it in items])
    t["n_fired"] = torch.tensor([len(it.get("prefix", [])) for it in items])
    for r, it in enumerate(items):
        k = len(it["z"])
        for name in NET["state"]:
            t[name][r, :k] = torch.from_numpy(np.asarray(it[name], np.int64))

        t["mask"][r, :k] = True

        for (i, j), o in it[NET["bonds"]].items():
            t["cur"][r, i, j] = t["cur"][r, j, i] = o

        for arrow in it.get("prefix", []):
            kind, i, j = arrow[0], arrow[1], arrow[2]
            t["fired_a" if kind == arrows.A else "fired_b"][r, i, j] += 1
            t["touched"][r, [i, j]] = True

        for arrow in it.get("targets", []):
            t["target"][r, arrow[1], arrow[2], slot(arrow)] = True

    return t


def size_batches(sizes, budget, rng, most=256):
    """Batches of similar size whose padded pair count stays under budget, in random order."""
    index = np.argsort(sizes, kind="stable")
    out, start = [], 0
    while start < len(index):
        n = sizes[index[min(start + most, len(index)) - 1]]
        take = max(1, min(most, budget // max(1, n * n)))
        chunk = index[start : start + take]
        n = sizes[chunk[-1]]
        take = max(1, min(len(chunk), budget // max(1, n * n)))
        out.append(chunk[:take].tolist())
        start += take

    rng.shuffle(out)

    return out


def evaluate(model, data, index, device, width=10, budget=600_000, processes=16):
    """Top-k accuracy of the most probable end markings, merged by the scoring form."""
    sizes = np.array([data.size(k) for k in index])
    batches = size_batches(sizes, budget // width, random.Random(0), most=64)
    ranks, valid = {}, {}

    with Pool(processes) as pool:
        for b in batches:
            ks = [index[x] for x in b]
            items = []
            for k in ks:
                m = data.marking(k)
                cap, lo, hi = net().limits(m, data.octet, data.window)
                fixed = {"cap": cap, "lo": lo, "hi": hi}
                items.append(
                    {name: fixed.get(name, m.get(name)) for name in NET["state"]}
                    | {NET["bonds"]: m[NET["bonds"]]}
                )

            t = {name: v.to(device) for name, v in collate(items).items()}

            with torch.autocast("cuda", dtype=torch.bfloat16):
                found = model.beam(t, width=width, top=width)

            jobs = [
                (data.marking(k), keys, data.meta["target"][k])
                for k, keys in zip(ks, found)
            ]
            for k, (rank, ok) in zip(ks, pool.map(_rank, jobs)):
                ranks[k], valid[k] = rank, ok

    return ranks, valid


def _rank(job):
    """Rank of the recorded product among the predicted end markings merged by scoring form, and whether the best one
    is a valid SMILES."""
    m, keys, target = job
    merged = {}
    for key, lp in keys:
        form = net().scoring_form(net().to_mol(net().apply(m, key)))
        merged[form] = float(np.logaddexp(merged.get(form, -np.inf), lp))

    ranked = [f for f, _ in sorted(merged.items(), key=lambda kv: -kv[1])]
    rank = ranked.index(target) + 1 if target is not None and target in ranked else None

    return rank, bool(ranked) and ranked[0] is not None


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    data, val = Steps("train"), Steps("val")
    status = np.asarray(data.a["status"])
    counts = np.diff(np.asarray(data.a["fired"]))
    index = np.nonzero((status == OK) & (counts <= MAX_ARROWS))[0]

    if args.limit:
        index = index[: args.limit]

    sizes = np.array([data.size(k) for k in index])
    model = NET["game"](
        d=args.width,
        rounds=args.rounds,
        attention=args.attention,
        enabling=not args.no_enabling,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01, fused=True
    )

    # training is limited by kernel launches, the compiled rate law fuses them. evaluation keeps the eager one
    compiled = None if args.no_compile else torch.compile(model.events, dynamic=True)
    per_epoch = len(size_batches(sizes, args.budget, random.Random(0)))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, args.lr, total_steps=args.epochs * per_epoch + 10, pct_start=0.05
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{args.name}-{args.seed}.pt"
    val_index = [k for k in range(len(val)) if val.a["status"][k] == OK][
        : args.val_steps
    ]
    best, start, log = -1.0, time.time(), []
    print(
        f"{len(index)} training steps, {per_epoch} batches per epoch, {n_params} parameters",
        flush=True,
    )

    for epoch in range(args.epochs):
        data.seed, data.epoch = args.seed, epoch
        order = size_batches(
            sizes, args.budget, random.Random(args.seed * 1000 + epoch)
        )
        loader = torch.utils.data.DataLoader(
            data,
            batch_sampler=[[int(index[x]) for x in b] for b in order],
            collate_fn=collate,
            num_workers=args.workers,
            persistent_workers=False,
            prefetch_factor=4,
            pin_memory=True,
        )
        model.train()

        # the loss stays on the device, reading it every step would wait for the device
        total, seen = torch.zeros((), device=device), 0

        if compiled is not None:
            model.events = compiled

        for step_, t in enumerate(loader):
            t = {name: v.to(device, non_blocking=True) for name, v in t.items()}

            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model.loss(t)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.detach() * len(t["z"])
            seen += len(t["z"])

            if step_ % 500 == 0:
                print(
                    f"epoch {epoch} batch {step_}/{per_epoch} loss {float(total) / max(seen, 1):.4f} {time.time() - start:.0f}s",
                    flush=True,
                )

        if compiled is not None:
            del model.events

        total = float(total)

        # greedy top-1 on validation steps selects the checkpoint
        model.eval()
        ranks, _ = evaluate(model, val, val_index, device, width=1)
        acc = float(np.mean([ranks[k] == 1 for k in val_index]))
        log.append(
            {
                "epoch": epoch,
                "loss": total / max(seen, 1),
                "val_top1_greedy": acc,
                "seconds": time.time() - start,
            }
        )
        print(
            f"epoch {epoch}: loss {total / max(seen, 1):.4f} val top-1 (greedy, {len(val_index)} steps) {acc:.4f}",
            flush=True,
        )

        if acc > best:
            best = acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "params": n_params,
                    "train_seconds": time.time() - start,
                    "curve": log,
                    "n_train": len(index),
                },
                path,
            )


def run_evaluation(args):
    device = torch.device("cuda")
    saved = torch.load(
        RESULTS / f"{args.name}-{args.seed}.pt", map_location=device, weights_only=False
    )
    a = saved["args"]
    model = NET["game"](
        d=a["width"],
        rounds=a["rounds"],
        attention=a["attention"],
        enabling=not a["no_enabling"],
    ).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    test = Steps("test")
    index = list(range(len(test)))

    if args.limit:
        index = index[: args.limit]

    readable = [k for k in index if test.a["status"][k] != FAILED]
    start = time.time()
    ranks, valid = evaluate(model, test, readable, device, width=args.beam)
    status = np.asarray(test.a["status"])
    metrics = {
        f"step_top{k}": float(
            np.mean([ranks.get(x) is not None and ranks[x] <= k for x in index])
        )
        for k in (1, 2, 3, 5, 10)
        if k <= args.beam
    }
    covered = [x for x in index if status[x] == OK]
    metrics |= {
        f"step_top{k}_covered": float(
            np.mean([ranks.get(x) is not None and ranks[x] <= k for x in covered])
        )
        for k in (1, 5)
        if k <= args.beam
    }

    # over every test step, so a step the model cannot read counts as invalid
    metrics["valid_smiles_top1"] = float(np.mean([valid.get(x, False) for x in index]))

    # one row per test step, for confidence intervals and for the split by the number of arrows that FlowER reports
    np.savez_compressed(
        RESULTS / f"{args.name}-{args.seed}-ranks.npz",
        step=np.asarray(index, np.int64),
        rank=np.asarray([ranks.get(x) or 0 for x in index], np.int16),
        status=status[index],
        arrows=np.diff(np.asarray(test.a["fired"]))[index],
    )
    metrics["share_single_electron"] = float(np.mean(status[index] == RADICAL))
    metrics["share_no_enabled_order"] = float(np.mean(status[index] == UNCOVERED))
    result = {
        "benchmark": "FlowER flower_dataset, elementary steps",
        "model": args.name,
        "seed": args.seed,
        "params": saved["params"],
        "n_train": saved["n_train"],
        "n_test": len(index),
        "epoch": saved["epoch"],
        "train_seconds": saved["train_seconds"],
        "eval_seconds": time.time() - start,
        "beam": args.beam,
        "curve": saved["curve"],
        "metrics": metrics,
    }
    out = RESULTS / (
        f"{args.name}-{args.seed}.json"
        if not args.limit
        else f"{args.name}-{args.seed}.limit{args.limit}.json"
    )
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(metrics, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["download", "prepare", "train", "evaluate"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--net",
        choices=sorted(NETS),
        default="arrow",
        help="arrow moves electron pairs, electron moves single electrons too",
    )
    ap.add_argument("--name", help="default, the name of the net")
    ap.add_argument("--processes", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--attention", type=int, default=6)
    ap.add_argument(
        "--budget",
        type=int,
        default=1_200_000,
        help="padded atom pairs per training batch",
    )
    ap.add_argument(
        "--workers", type=int, default=int(os.environ.get("NPF_WORKERS", 10))
    )
    ap.add_argument("--val-steps", type=int, default=3000)
    ap.add_argument("--beam", type=int, default=10)
    ap.add_argument(
        "--limit", type=int, default=0, help="first steps only, for smoke tests"
    )
    ap.add_argument(
        "--no-enabling",
        action="store_true",
        help="ablation without octet capacities and charge windows",
    )
    ap.add_argument(
        "--no-compile",
        action="store_true",
        help="train with the eager rate law instead of the compiled one",
    )
    args = ap.parse_args()

    # every net keeps its own prepared data, weights and results
    global NET
    NET = NETS[args.net]
    args.name = args.name or NET["name"]
    stages = {
        "download": download,
        "prepare": prepare,
        "train": train,
        "evaluate": run_evaluation,
    }
    stages[args.stage](args)


if __name__ == "__main__":
    main()
