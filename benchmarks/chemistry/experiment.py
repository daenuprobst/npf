"""Forward prediction and reaction classification on the valence Petri net.

    uv run python -m benchmarks.chemistry.experiment --task classify --model npf --seed 0
    uv run python -m benchmarks.chemistry.experiment --task forward --model npf --dataset uspto_mit

Models. npf uses the Petri semantics, pgnn is the same message passing with a generic readout, drfp is DRFP with an
MLP for classification, and the npf-no... variants remove one Petri component each.

Maps. Every map and target comes from the mapper of npf.chem.mapper, no recorded map is read where avoidable.
Classification reads its maps with --maps. Forward prediction trains on the firing vectors of the mapper in
data/net_targets_<dataset>[-sub<n>].pkl, which benchmarks.chemistry.net_targets builds when the file is missing, or in
the file of --net-targets, and --recorded-maps trains on the recorded atom maps instead, the ablation of the paper.

Splits. classify uses the published split of Schneider 50k with 200 training and 800 test reactions per class. On
Schneider 50k, forward uses a fixed random 80/10/10 split of the reactions with a clean atom mapping, an internal
protocol for ablations. With uspto_mit the official split of Jin et al. is used and forward prediction is scored on
the whole official test set. Training records whose firing vector moves more than MAX_TOKENS tokens are dropped as
label noise, validation and test sets are never filtered.
"""

import argparse
import copy
import hashlib
import json
import os
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from npf import chem
from npf.chem import decode
from npf.chem import (
    MAX_TOKENS,
    batch_indices,
    batches,
    firing_vector,
    product_found,
    product_major,
    recorded_products,
)

EPOCHS = {"classify": 60, "forward": 60}
BATCH = {"classify": 64, "forward": 32}

# lines of the official USPTO-MIT training and test files
USPTO_MIT_TRAIN_LINES = 409035
USPTO_MIT_TEST_LINES = 40000

# validation reactions whose beam candidates train the verifier, the first 1500 are used for model selection
VERIFIER_REACTIONS = 12000

# USPTO-MIT validation reactions on which runs are compared, disjoint from the 1500 that select the epoch
SCREEN = slice(3000, 10500)


def subset_lines(n):
    """The first n lines of one fixed random order of the official USPTO-MIT training file, so subsets are nested and
    shared with the baselines. A reaction id of the training split is its line number.
    """
    return np.random.default_rng(0).permutation(USPTO_MIT_TRAIN_LINES)[:n]


def splits(data, task, clean_train=True, recorded=True):
    """Training, validation and test reactions. With recorded=False no training reaction is kept or dropped because of
    a recorded map, the targets then come from the net. On USPTO-MIT both kinds of runs share the validation
    reactions, on Schneider 50k the benchmark itself is defined on reactions with a clean recorded map.
    """
    reactions = data["reactions"]

    # USPTO-MIT
    if any(r["split"] == "val" for r in reactions):
        # the size caps of training and model selection, the same for both kinds of targets and read off the molecules
        part = lambda name: [
            r
            for r in reactions
            if r["split"] == name
            and len(r["a"]["x"]) <= chem.MAX_PRECURSOR_ATOMS
            and len(r["b"]["x"]) <= chem.MAX_PRODUCT_ATOMS
        ]
        train = part("train")

        if recorded:
            train = [
                r
                for r in train
                if r["target"] is not None
                and tokens_moved(r, r["target"]) <= MAX_TOKENS
            ]

        # models trained here are scored on the Golden set too, so no training reaction may share a main product with it
        golden = Path("data/golden.pkl")
        if golden.exists():
            main = lambda r: chem.canonical_product(r["smiles"].split(">>")[1])
            banned = {main(r) for r in pickle.loads(golden.read_bytes())["reactions"]}
            n = len(train)
            train = [r for r in train if main(r) not in banned]
            print(
                f"removed {n - len(train)} training reactions whose main product occurs in the Golden dataset",
                flush=True,
            )

        # every test reaction is scored, whatever its size
        return train, part("val"), [r for r in reactions if r["split"] == "test"]

    if task == "classify":
        train = [r for r in reactions if r["split"] == "train"]
        rng = np.random.default_rng(0)
        rng.shuffle(train)
        return train[500:], train[:500], [r for r in reactions if r["split"] == "test"]

    # the Schneider 50k protocol is defined on reactions with a clean recorded map, targets can still come from the net
    mapped = [r for r in reactions if r["target"] is not None]
    order = np.random.default_rng(0).permutation(len(mapped))
    n = len(order) // 10
    pick = lambda idx: [mapped[i] for i in idx]
    train = pick(order[2 * n :])

    # the minimum firing prior against label noise applies to training only, without recorded maps to the net targets
    if clean_train and recorded:
        train = [r for r in train if tokens_moved(r, r["target"]) <= MAX_TOKENS]

    return train, pick(order[:n]), pick(order[n : 2 * n])


def attach_firing_histograms(train, *others, n_types=300):
    """Auxiliary target, counts of the fired transition types (atom type, atom type, old bond, new bond) with atom
    type (element, aromatic, degree), over the n_types most frequent types of the training set.
    """

    def types(r):
        a, bonds = r["a"], chem.dense_bonds(r["a"])
        kind = lambda i: (
            int(a["element"][i]),
            int(a["x"][i][-2]),
            int(min((bonds[i] > 0).sum(), 4)),
        )

        return [
            (*sorted([kind(i), kind(j)]), int(bonds[i, j]), int(new))
            for i, j, new in r["edits"]
        ]

    counts = Counter(t for r in train if r["edits"] is not None for t in types(r))
    index = {t: k + 1 for k, (t, _) in enumerate(counts.most_common(n_types - 1))}
    for r in [*train, *(r for rs in others for r in rs)]:
        hist = np.zeros(n_types, np.float32)

        if r["edits"] is not None:
            for t in types(r):
                hist[index.get(t, 0)] += 1

        r["hist"] = hist

    return n_types


def use_predicted_firing(
    *splits_, path="data/exact_maps_schneider50k.pkl", blank_missing=False
):
    """Replace the recorded mapping and edits of every reaction by those of the mapper in a file written by
    benchmarks.chemistry.exact_map --write."""
    maps = pickle.loads(Path(path).read_bytes())
    for r in (r for rs in splits_ for r in rs):
        # a reaction the file does not cover keeps its record, or loses it when no recorded map may be used at all
        if r["id"] not in maps:
            if blank_missing:
                r["target"], r["edits"] = np.full(
                    len(r["b"]["x"]), -1, np.int16
                ), np.zeros((0, 3), np.int16)

            continue

        mapping = maps[r["id"]].astype(np.int64)

        # product atoms without a precursor, an incomplete record with no firing vector
        if (
            len(mapping) != len(r["b"]["x"])
            or (mapping < 0).any()
            or (mapping >= len(r["a"]["x"])).any()
        ):
            r["target"], r["edits"] = np.full(len(r["b"]["x"]), -1, np.int16), np.zeros(
                (0, 3), np.int16
            )
            continue

        r["target"], r["edits"] = mapping.astype(np.int16), firing_vector(r, mapping)


def use_net_targets(train, path, single=False):
    """Forward targets from the net alone, the firing vectors benchmarks.chemistry.net_targets found. The recorded
    mapping is dropped, a reaction without a vector is not trained on, and single keeps only the first vector.
    """
    vectors = pickle.loads(Path(path).read_bytes())["targets"]
    kept = []
    for r in train:
        found = vectors.get(r["id"])
        if not found:
            continue

        r["target"], r["edits"], r["edits_set"] = (
            None,
            found[0],
            found[:1] if single else found,
        )
        kept.append(r)

    print(
        f"net targets for {len(kept)} of {len(train)} training reactions, {np.mean([len(r['edits_set']) for r in kept]):.2f} "
        f"firing vectors per reaction",
        flush=True,
    )

    return kept


def within(rs, pairs=640_000):
    """A batch sorted by size, split where its padded atom pairs would exceed pairs, since the forward models hold
    every pair densely."""
    out, group = [], []
    for r in rs:
        n = len(r["a"]["x"])
        if group and (len(group) + 1) * max(n, len(group[-1]["a"]["x"])) ** 2 > pairs:
            out.append(group)
            group = []

        group.append(r)

    return out + [group] if group else out


class Collated(torch.utils.data.Dataset):
    """One epoch of padded batches, collated in worker processes so the GPU is not starved."""

    def __init__(self, reactions, index_batches):
        self.reactions, self.index_batches = reactions, index_batches

    def __len__(self):
        return len(self.index_batches)

    def __getitem__(self, k):
        return chem.collate([self.reactions[i] for i in self.index_batches[k]], "cpu")


def epoch_loader(
    reactions, size, rng, device, workers=int(os.environ.get("NPF_WORKERS", 4))
):
    loader = torch.utils.data.DataLoader(
        Collated(reactions, batch_indices(reactions, size, rng)),
        batch_size=None,
        shuffle=False,
        num_workers=workers,
        prefetch_factor=4,
        pin_memory=device == "cuda",
    )
    for batch in loader:
        yield {k: v.to(device, non_blocking=True) for k, v in batch.items()}


# metrics


def tokens_moved(reaction, mapping):
    """Size of the firing vector under a mapping, bond tokens plus hydrogens and charges that have to move."""
    return chem.cost_of(reaction, mapping)


def symmetric(a, i, j, old, new, classes="symmetry"):
    s = a[classes]

    return Counter(
        (min(s[x], s[y]), max(s[x], s[y]), int(o), int(n))
        for x, y, o, n in zip(i, j, old, new)
    )


@torch.no_grad()
def evaluate(model, task, reactions, device):
    model.eval()
    stats = Counter()
    groups = batches(reactions, 64)

    if task == "forward":
        groups = [part for rs in groups for part in within(rs)]

    for rs in groups:
        b = chem.collate(rs, device)
        out = model(b)

        if task == "classify":
            stats["n"] += len(rs)
            stats["classes"] = out.shape[-1]
            stats["correct"] += int((out.argmax(-1) == b["label"]).sum())

            for p, t in zip(out.argmax(-1).tolist(), b["label"].tolist()):
                stats["tp", t] += p == t
                stats["pred", p] += 1
                stats["true", t] += 1
        else:
            for r, pred in zip(rs, model.decode(out, b, rs)):
                a, before = r["a"], chem.dense_bonds(r["a"])

                # no clean recorded mapping
                true = (
                    r["edits"] if r["edits"] is not None else np.zeros((0, 3), np.int64)
                )
                same = lambda e: symmetric(
                    a, e[:, 0], e[:, 1], before[e[:, 0], e[:, 1]], e[:, 2]
                )
                after = before.copy()
                after[pred[:, 0], pred[:, 1]] = pred[:, 2]
                after[pred[:, 1], pred[:, 0]] = pred[:, 2]
                hydrogens = a["h"] - (
                    chem.BOND_ORDER[after] - chem.BOND_ORDER[before]
                ).sum(1)
                stats["n"] += 1
                stats["exact"] += {tuple(e) for e in pred.tolist()} == {
                    tuple(e) for e in true.tolist()
                }
                stats["product"] += product_found(r, pred)

                # a touched fragment RDKit cannot sanitise is invalid chemistry, whatever the enabling rule says
                stats["rdkit_valid"] += decode.DROPPED["touched"] == 0
                stats["dropped"] += decode.DROPPED["touched"]
                stats["major"] += product_major(r, pred)
                stats["exact_sym"] += same(pred) == same(true)
                capacity = np.array(
                    [chem.EXTRA_CAPACITY.get(int(e), 0) for e in a["element"]]
                ) + np.maximum(-a["q"], 0)

                # no hydrogen place below its capacity, the rule enabling enforces, so it cannot fail for the net
                stats["enabled"] += bool((hydrogens >= -capacity - 0.5).all())

    n = stats["n"]

    if task == "classify":
        f1 = [
            2 * stats["tp", c] / max(stats["pred", c] + stats["true", c], 1)
            for c in range(stats["classes"])
        ]
        return {"accuracy": stats["correct"] / n, "macro_f1": float(np.mean(f1))}

    return {
        "product_top1": stats["product"] / n,
        "product_major_top1": stats["major"] / n,
        "edits_exact_up_to_symmetry": stats["exact_sym"] / n,
        "edits_exact": stats["exact"] / n,
        "valence_valid": stats["rdkit_valid"] / n,
        "enabling_rule_valid": stats["enabled"] / n,
        "dropped_touched_fragments": stats["dropped"] / n,
    }


def ranked_candidates(model, reactions, device, width, batch, pairs=400_000):
    """(reaction, ranked candidates) for every reaction. batch 1 searches one reaction at a time, larger batches share
    the rate law over up to batch reactions of similar size, fewer when their hypotheses would exceed pairs atom
    pairs."""
    if batch <= 1:
        for r in reactions:
            yield r, model.beam_search(chem.collate([r], device), width)

        return

    group = []
    for i in sorted(
        range(len(reactions)), key=lambda i: len(reactions[i]["a"]["x"])
    ) + [None]:
        n = len(reactions[i]["a"]["x"]) if i is not None else 0
        if group and (
            i is None or len(group) >= batch or (len(group) + 1) * width * n * n > pairs
        ):
            rs = [reactions[j] for j in group]
            yield from zip(rs, model.beam_search_batch(chem.collate(rs, device), width))
            group = []

        if i is not None:
            group.append(i)


@torch.no_grad()
def evaluate_beam(model, reactions, device, width=5, dump=None, batch=16):
    """Top-k product accuracy of the token game, the recorded product among the k most probable markings. With dump
    the candidates of every reaction are written there as (id, [(edits, log probability, correct)]).
    """
    model.eval()
    hits, major_hits, merged_hits, candidates = (
        np.zeros(width),
        np.zeros(width),
        np.zeros(width),
        [],
    )
    for r, ranked in ranked_candidates(model, reactions, device, width, batch):
        found = [product_found(r, edits) for edits, _ in ranked]
        first = found.index(True) if True in found else width
        hits[first:] += 1
        major = [product_major(r, edits) for edits, _ in ranked]
        major_hits[major.index(True) if True in major else width :] += 1

        # firing vectors that decode to the same molecules are one prediction, their probabilities add
        groups = {}
        for (edits, lp), ok in zip(ranked, found):
            key = frozenset(chem.marking_fragments(r["a"], edits)[0])
            total, right = groups.get(key, (-np.inf, False))
            groups[key] = (float(np.logaddexp(total, lp)), right or ok)

        order = [right for _, right in sorted(groups.values(), key=lambda g: -g[0])]
        first = order.index(True) if True in order else width
        merged_hits[first:] += 1
        candidates.append(
            (r["id"], [(edits, lp, ok) for (edits, lp), ok in zip(ranked, found)])
        )

    if dump:
        Path(dump).parent.mkdir(parents=True, exist_ok=True)
        Path(dump).write_bytes(pickle.dumps(candidates))

    n = len(reactions)

    return (
        {f"product_top{k + 1}": float(hits[k] / n) for k in range(width)}
        | {
            f"product_merged_top{k + 1}": float(merged_hits[k] / n)
            for k in range(width)
        }
        | {f"product_major_top{k + 1}": float(major_hits[k] / n) for k in range(width)}
    )


MAIN_METRIC = {"classify": "accuracy", "forward": "product_top1"}


# DRFP reference, classification only


def drfp_baseline(train, val, test, device, seed, everything=None, select="best"):
    from multiprocessing import Pool

    cache = Path("data/drfp.npy")

    # the cache is indexed by the rank of the reaction id among ALL reactions
    everything = everything or (train + val + test)

    if not cache.exists():
        with Pool(12) as pool:
            fps = pool.map(
                _drfp,
                [r["smiles"] for r in sorted(everything, key=lambda r: r["id"])],
                chunksize=128,
            )

        np.save(cache, np.array(fps, np.uint8))

    rank = {r["id"]: k for k, r in enumerate(sorted(everything, key=lambda r: r["id"]))}
    fps = np.load(cache)
    tensor = lambda rs: (
        torch.as_tensor(
            fps[[rank[r["id"]] for r in rs]], dtype=torch.float32, device=device
        ),
        torch.as_tensor([r["label"] for r in rs], device=device),
    )
    (xt, yt), (xv, yv), (xs, ys) = tensor(train), tensor(val), tensor(test)
    n_classes = max(r["label"] for r in everything) + 1
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(2048, 1024),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(1024, n_classes),
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, state = 0.0, None
    for _ in range(40):
        net.train()

        for idx in torch.randperm(len(xt), device=device).split(128):
            loss = F.cross_entropy(net(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        net.eval()

        # the last epoch is kept without looking at a validation label
        acc = (
            float((net(xv).argmax(-1) == yv).float().mean())
            if select == "best"
            else 1.0
        )
        if acc >= best:
            best, state = acc, copy.deepcopy(net.state_dict())

    net.load_state_dict(state)
    pred = net(xs).argmax(-1)
    f1 = [
        2
        * float(((pred == c) & (ys == c)).sum())
        / max(float((pred == c).sum() + (ys == c).sum()), 1)
        for c in range(n_classes)
    ]

    return {
        "accuracy": float((pred == ys).float().mean()),
        "macro_f1": float(np.mean(f1)),
    }, sum(p.numel() for p in net.parameters())


def _drfp(smiles):
    from drfp import DrfpEncoder

    return DrfpEncoder.encode(smiles, n_folded_length=2048)[0]


# main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(EPOCHS), required=True)
    ap.add_argument(
        "--model",
        default="npf",
        help="npf, pgnn, drfp, or an ablation, npf-nogate, npf-sigma and pgnn-sigma for classify, npf-oneshot and "
        "npf-noenabling for forward",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int)
    ap.add_argument(
        "--limit",
        type=int,
        help="only this many training reactions, for smoke tests and data efficiency",
    )
    ap.add_argument(
        "--subset",
        type=int,
        help="uspto_mit, train on the reactions among this many random lines of the official "
        "training file, the baselines get the same lines",
    )
    ap.add_argument("--root", default="results")
    ap.add_argument(
        "--dataset", choices=["schneider50k", "uspto_mit"], default="schneider50k"
    )
    ap.add_argument(
        "--evaluate-only",
        action="store_true",
        help="re-score the saved weights on the test set",
    )
    ap.add_argument(
        "--no-beam", action="store_true", help="skip the beam search evaluation"
    )
    ap.add_argument(
        "--dump-validation-beams",
        action="store_true",
        help="forward, also write the beam candidates of the validation "
        "reactions not used for model selection",
    )
    ap.add_argument(
        "--width", type=int, default=128, help="hidden width of the token game"
    )
    ap.add_argument(
        "--rounds", type=int, default=6, help="message-passing rounds of the token game"
    )
    ap.add_argument(
        "--attention",
        type=int,
        default=0,
        help="attention layers of the token game, default 4, or 6 if width > 128",
    )
    ap.add_argument(
        "--matched",
        action="store_true",
        help="forward, pgnn or npf-oneshot, give the one-shot counterpart the encoder "
        "of the token game for equal capacity, the default keeps the smaller published one",
    )
    ap.add_argument("--batch", type=int, default=0, help="batch size, default per task")
    ap.add_argument("--amp", action="store_true", help="bfloat16 autocast")
    ap.add_argument(
        "--no-compile",
        action="store_true",
        help="token game on CUDA, train with the eager rate law instead of the "
        "compiled one, which launches far fewer kernels",
    )
    ap.add_argument(
        "--beam-batch",
        type=int,
        default=16,
        help="reactions that share one beam search, 1 searches them one at a time",
    )
    ap.add_argument("--tag", default="", help="suffix of the result files, e.g. -large")
    ap.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="peak learning rate of the one-cycle schedule",
    )
    ap.add_argument(
        "--maps",
        default="data/exact_maps_schneider50k.pkl",
        help="classify, the atom maps of the mapper from exact_map --write that the gate, "
        "the firing histograms and the explicit firing vector read",
    )
    ap.add_argument(
        "--net-targets",
        help="forward, the sets of firing vectors of the mapper to train on, by default "
        "data/net_targets_<dataset>[-sub<n>].pkl, built by benchmarks.chemistry.net_targets when missing",
    )
    ap.add_argument(
        "--recorded-maps",
        action="store_true",
        help="forward, train on the recorded atom maps of the data set instead of the firing vectors of the mapper",
    )
    ap.add_argument(
        "--single-target",
        action="store_true",
        help="with --net-targets, only the first vector of every set",
    )
    ap.add_argument(
        "--firing",
        action="store_true",
        help="classify, auxiliary task on which transition types fired, from the maps",
    )
    ap.add_argument(
        "--select",
        choices=["best", "last"],
        default="best",
        help="the epoch kept, the best on the validation reactions or the "
        "last, which reads no validation label",
    )
    ap.add_argument(
        "--labels",
        type=int,
        help="classify, number of training reactions whose class label is used, the rest only "
        "contribute their firing histograms with --firing",
    )
    ap.add_argument(
        "--size-split",
        action="store_true",
        help="classify, train on the smaller half of the training reactions and "
        "test on the largest quarter of the test reactions",
    )
    args = ap.parse_args()

    # forward models train on the firing vectors of the mapper unless the recorded maps are asked for, and a missing
    # target file is built before anything else starts
    if args.task == "forward" and args.recorded_maps:
        args.net_targets = None
    elif args.task == "forward" and not args.net_targets:
        from .net_targets import build, built_by_search, path

        default = path(args.dataset, args.subset)
        if not default.exists():
            print(f"building {default} with the mapper", flush=True)
            build(args.dataset, args.subset)
        elif not built_by_search(default):
            raise SystemExit(
                f"{default} was not built by the mapper, pass --net-targets or rebuild it"
            )

        args.net_targets = str(default)

    # maps and targets are data files outside version control, their digests say which ones a run read
    for name, used in (
        ("maps", args.task == "classify"),
        ("net_targets", bool(args.net_targets)),
    ):
        if used and Path(getattr(args, name)).exists():
            setattr(
                args,
                f"{name}_sha256",
                hashlib.sha256(Path(getattr(args, name)).read_bytes()).hexdigest(),
            )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = chem.load(f"data/{args.dataset}.pkl")
    train, val, test = splits(data, args.task, recorded=not args.net_targets)
    train = train[: args.limit] if args.limit else train

    if args.subset:
        chosen = set(subset_lines(args.subset).tolist())
        train = [r for r in train if r["id"] in chosen]
        args.tag += f"-sub{args.subset}"
        print(
            f"subset of {args.subset} lines of the training file: {len(train)} usable training reactions",
            flush=True,
        )

    if args.net_targets:
        train = use_net_targets(train, args.net_targets, single=args.single_target)
        args.tag += "-nettargets" + ("-single" if args.single_target else "")

    # every head that reads atom maps reads those of the mapper, on test reactions too, so no recorded map enters
    if args.task == "classify":
        use_predicted_firing(train, val, test, path=args.maps, blank_missing=True)

    n_types = 0

    if args.task == "classify" and (args.firing or args.labels):
        if args.firing:
            n_types = attach_firing_histograms(train, val, test)

        if args.labels:
            for k, r in enumerate(train):
                r["labelled"] = k < args.labels

            if not args.firing:
                train = train[: args.labels]

        args.tag += ("-firing" if args.firing else "") + (
            f"-labels{args.labels}" if args.labels else ""
        )

    if args.matched:
        args.tag += "-matched"

    if args.size_split:
        size = lambda r: len(r["a"]["x"])
        small, large = np.median([size(r) for r in train]), np.quantile(
            [size(r) for r in test], 0.75
        )
        train, val, test = (
            [r for r in train if size(r) <= small],
            [r for r in val if size(r) <= small],
            [r for r in test if size(r) >= large],
        )
        args.tag += "-sizesplit"
        print(
            f"size split: train on <= {small:.0f} precursor atoms ({len(train)} reactions), test on >= {large:.0f} ({len(test)})",
            flush=True,
        )

    start = time.time()

    if args.model == "drfp":
        metrics, n_params = drfp_baseline(
            train, val, test, device, args.seed, data["reactions"], args.select
        )
        curve = []
    else:
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        petri = args.model.startswith("npf")
        size = dict(
            d=args.width,
            rounds=args.rounds,
            attention=args.attention or (6 if args.width > 128 else 4),
        )
        model = {
            "classify": lambda: chem.Classifier(
                len(data["classes"]),
                petri=petri,
                gate=args.model not in ("npf-nogate", "npf-sigma"),
                n_firing_types=n_types,
                explicit_firing=args.model.startswith("npf-sigma"),
                mapped_atoms=args.model == "pgnn-sigma",
            ),
            "forward": lambda: (
                chem.Forward(petri=petri, **(size if args.matched else {}))
                if args.model in ("pgnn", "npf-oneshot")
                else chem.TokenGame(**size, enabling=args.model != "npf-noenabling")
            ),
        }[args.task]().to(device)
        n_params = sum(p.numel() for p in model.parameters())
        epochs = args.epochs or EPOCHS[args.task]

        # the token game is limited by kernel launches, the compiled rate law fuses them and serves training only, so
        # scores never depend on compilation
        compiled = None

        if (
            isinstance(model, chem.TokenGame)
            and device == "cuda"
            and not args.no_compile
        ):
            compiled = torch.compile(model.events, dynamic=True)

        opt = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=1e-5, fused=device == "cuda"
        )
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            args.lr,
            total_steps=epochs
            * len(batch_indices(train, args.batch or BATCH[args.task])),
            pct_start=0.1,
        )
        best, state, curve = -1.0, None, []
        weights = (
            Path(args.root)
            / ("chem" if args.dataset == "schneider50k" else args.dataset)
            / args.task
            / f"{args.model}{args.tag}-{args.seed}{'-n' + str(args.limit) if args.limit else ''}.pt"
        )

        if args.evaluate_only:
            state, epochs = torch.load(weights, map_location=device), 0

        for epoch in range(epochs):
            model.train()
            losses = []

            if compiled is not None:
                model.events = compiled

            for b in epoch_loader(train, args.batch or BATCH[args.task], rng, device):
                if isinstance(model, chem.TokenGame):
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                        loss = model.loss(b)

                else:
                    out = model(b)

                    if args.task == "classify":
                        known = b["labelled"]
                        loss = (
                            (
                                F.cross_entropy(out[known], b["label"][known])
                                if known.any()
                                else 0.0
                            )
                            + 0.5 * model.auxiliary_loss(b)
                            + model.firing_loss(b)
                        )
                    else:
                        loss = model.loss(out, b)

                # a batch without labels and without recorded firings
                if not torch.is_tensor(loss):
                    sched.step()
                    continue

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()

                # reading the loss every step would wait for the device, the mean is read once per epoch
                losses.append(loss.detach())

            # evaluation runs the eager rate law, the instance attribute shadowed the method during training
            if compiled is not None:
                del model.events

            mean_loss = (
                float(torch.stack(losses).float().mean()) if losses else float("nan")
            )

            # the last epoch is kept without looking at a validation label
            score = (
                evaluate(model, args.task, val[:1500], device)[MAIN_METRIC[args.task]]
                if args.select == "best"
                else float(epoch)
            )
            curve.append((epoch, mean_loss, score))
            print(
                f"epoch {epoch}: loss {mean_loss:.4f}  val {MAIN_METRIC[args.task]} {score:.4f}  ({time.time() - start:.0f}s)",
                flush=True,
            )

            if score >= best:
                best, state = score, copy.deepcopy(model.state_dict())
                weights.parent.mkdir(parents=True, exist_ok=True)

                # long runs should survive a crash
                torch.save(state, weights)

        model.load_state_dict(state)
        metrics = evaluate(model, args.task, test, device)

        if isinstance(model, chem.TokenGame) and not args.no_beam:
            beams = weights.parent.parent / "beams" / f"{weights.stem}-test.pkl"
            metrics |= {
                k + "_beam": v
                for k, v in evaluate_beam(
                    model, test, device, dump=beams, batch=args.beam_batch
                ).items()
            }

            # candidates on validation reactions that were not used for model selection, for a verifier
            if args.dump_validation_beams:
                evaluate_beam(
                    model,
                    val[1500 : 1500 + VERIFIER_REACTIONS],
                    device,
                    batch=args.beam_batch,
                    dump=weights.parent.parent / "beams" / f"{weights.stem}-val.pkl",
                )

        torch.save(state, weights)

        # test reactions that RDKit cannot read count as wrong, so that the denominator is the official one
        if args.dataset == "uspto_mit" and args.task == "forward":
            metrics |= {
                f"{k}_official": v * len(test) / USPTO_MIT_TEST_LINES
                for k, v in metrics.items()
                if k.startswith("product_")
            }

            # validation reactions that no run selects on and that no recorded map chooses, where runs are compared
            screen = [r for r in data["reactions"] if r["split"] == "val"][SCREEN]
            metrics |= {
                f"screen_{k}": v
                for k, v in evaluate(model, "forward", screen, device).items()
            }

            if isinstance(model, chem.TokenGame) and not args.no_beam:
                metrics |= {
                    f"screen_{k}_beam": v
                    for k, v in evaluate_beam(
                        model, screen, device, batch=args.beam_batch
                    ).items()
                }

    result = {
        "task": args.task,
        "model": args.model,
        "seed": args.seed,
        "args": vars(args),
        "params": n_params,
        "n_train": len(train),
        "n_test": len(test),
        "train_seconds": time.time() - start,
        "curve": curve,
        "metrics": metrics,
    }
    folder = "chem" if args.dataset == "schneider50k" else args.dataset
    out = (
        Path(args.root)
        / folder
        / args.task
        / f"{args.model}{args.tag}-{args.seed}{'-n' + str(args.limit) if args.limit else ''}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    # keep the record of the training run
    if args.evaluate_only and out.exists():
        previous = json.loads(out.read_text())
        result |= {
            "curve": previous["curve"],
            "train_seconds": previous["train_seconds"],
        }

    out.write_text(json.dumps(result, indent=1))
    print(
        f"chem/{args.task} {args.model} seed {args.seed}: "
        + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        + f"  ({n_params} params, {result['train_seconds']:.0f}s)"
    )


if __name__ == "__main__":
    main()
