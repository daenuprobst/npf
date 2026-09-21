"""Atom mapping, forward prediction and reaction classification on the valence Petri net.

    uv run python -m benchmarks.chemistry.experiment --task classify --model npf --seed 0
    uv run python -m benchmarks.chemistry.experiment --task map --model pgnn
    uv run python -m benchmarks.chemistry.experiment --task forward --model npf --dataset uspto_mit

Models. npf uses the Petri semantics, pgnn is the same message passing with a generic readout, drfp is DRFP with an
MLP for classification, and the npf-no... variants remove one Petri component each.

Splits. classify uses the published split of Schneider 50k with 200 training and 800 test reactions per class.
On Schneider 50k, map and forward use a fixed random 80/10/10 split of the reactions that come with a clean atom
mapping, which is an internal protocol for ablations. With the dataset uspto_mit the official split of Jin et al. is
used, and forward prediction is scored on the whole official test set. Training records whose recorded mapping moves
more than MAX_TOKENS tokens are dropped as label noise. Validation and test sets are never filtered.
"""
import argparse
import copy
import json
import pickle
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from npf import chem

EPOCHS = {"classify": 60, "map": 25, "forward": 60}
BATCH = {"classify": 64, "map": 48, "forward": 32}


# token descent after the Hungarian step, with uniform costs it lowers the agreement with recorded maps by 4 points
REFINE = False

# training records whose recorded mapping moves more tokens are mostly incomplete (1.3 %), not trained on
MAX_TOKENS = 10

# lines of the official USPTO-MIT training and test files
USPTO_MIT_TRAIN_LINES = 409035
USPTO_MIT_TEST_LINES = 40000

# validation reactions whose beam candidates train the verifier, the first 1500 are used for model selection
VERIFIER_REACTIONS = 12000


def subset_lines(n):
    """The first n entries of one fixed random order of the lines of the official USPTO-MIT training file. Subsets
    are nested and shared with the baselines, a reaction id of the training split is its line number."""
    return np.random.default_rng(0).permutation(USPTO_MIT_TRAIN_LINES)[:n]


def splits(data, task, clean_train=True):
    reactions = data["reactions"]

    # USPTO-MIT
    if any(r["split"] == "val" for r in reactions):
        part = lambda name: [r for r in reactions if r["split"] == name]
        train = [r for r in part("train") if r["target"] is not None and tokens_moved(r, r["target"]) <= MAX_TOKENS]

        # models trained here are also scored on the Golden atom mapping set, so no training reaction may share its
        # main product with that set
        golden = Path("data/golden.pkl")
        if golden.exists():
            main = lambda r: chem.canonical_product(r["smiles"].split(">>")[1])
            banned = {main(r) for r in pickle.loads(golden.read_bytes())["reactions"]}
            n = len(train)
            train = [r for r in train if main(r) not in banned]
            print(f"removed {n - len(train)} training reactions whose main product occurs in the Golden dataset", flush=True)

        if task == "map":
            return train, [r for r in part("val") if r["target"] is not None], [r for r in part("test") if r["target"] is not None]

        return train, [r for r in part("val") if r["target"] is not None], part("test")

    if task == "classify":
        train = [r for r in reactions if r["split"] == "train"]
        rng = np.random.default_rng(0)
        rng.shuffle(train)
        return train[500:], train[:500], [r for r in reactions if r["split"] == "test"]

    mapped = [r for r in reactions if r["target"] is not None]

    # the mapper used inside the classification pipeline, trained on Schneider's training split only
    if task == "map-clean":
        train = [r for r in mapped if r["split"] == "train" and tokens_moved(r, r["target"]) <= MAX_TOKENS]
        return train[500:], train[:500], [r for r in mapped if r["split"] == "test"]

    order = np.random.default_rng(0).permutation(len(mapped))
    n = len(order) // 10
    pick = lambda idx: [mapped[i] for i in idx]
    train = pick(order[2 * n:])

    # minimum-firing prior against label noise, validation and test sets are left as they are
    if clean_train:
        train = [r for r in train if tokens_moved(r, r["target"]) <= MAX_TOKENS]

    return train, pick(order[:n]), pick(order[n:2 * n])


def attach_firing_histograms(train, *others, n_types=300):
    """Auxiliary target, counts of the fired transition types (atom type, atom type, old bond, new bond) with
    atom type = (element, aromatic, degree), the vocabulary is the n_types most frequent types of the training set."""
    def types(r):
        a, bonds = r["a"], chem.dense_bonds(r["a"])
        kind = lambda i: (int(a["element"][i]), int(a["x"][i][-2]), int(min((bonds[i] > 0).sum(), 4)))

        return [(*sorted([kind(i), kind(j)]), int(bonds[i, j]), int(new)) for i, j, new in r["edits"]]

    counts = Counter(t for r in train if r["edits"] is not None for t in types(r))
    index = {t: k + 1 for k, (t, _) in enumerate(counts.most_common(n_types - 1))}
    for r in [*train, *(r for rs in others for r in rs)]:
        hist = np.zeros(n_types, np.float32)

        if r["edits"] is not None:
            for t in types(r):
                hist[index.get(t, 0)] += 1

        r["hist"] = hist

    return n_types


@torch.no_grad()
def write_predicted_maps(model, reactions, device, path="data/predicted_maps.pkl"):
    """Mapping of every reaction (with or without a recorded one) by the given mapper, {reaction id, product -> precursor}."""
    import pickle
    model.eval()
    out = {}
    for rs in batches(reactions, 32):
        b = chem.collate(rs, device)
        for r, mapping in zip(rs, model.decode(model(b), b)):
            out[r["id"]] = mapping.astype(np.int16)

    Path(path).write_bytes(pickle.dumps(out))
    print(f"wrote {len(out)} predicted mappings to {path}", flush=True)


def use_predicted_firing(*splits_, path="data/predicted_maps.pkl", blank_missing=False):
    """Replace the recorded mapping and edits of every reaction by the ones our own mapper predicts."""
    import pickle
    maps = pickle.loads(Path(path).read_bytes())
    for r in (r for rs in splits_ for r in rs):
        # a reaction the file does not cover keeps its record, or loses it when no recorded map may be used at all
        if r["id"] not in maps:
            if blank_missing:
                r["target"], r["edits"] = np.full(len(r["b"]["x"]), -1, np.int16), np.zeros((0, 3), np.int16)

            continue

        mapping = maps[r["id"]].astype(np.int64)

        # product atoms without a precursor (incomplete record), no firing vector
        if len(mapping) != len(r["b"]["x"]):
            r["target"], r["edits"] = np.full(len(r["b"]["x"]), -1, np.int16), np.zeros((0, 3), np.int16)
            continue

        before, after = chem.dense_bonds(r["a"]), np.zeros((len(r["a"]["x"]),) * 2, np.int8)
        after[np.ix_(mapping, mapping)] = chem.dense_bonds(r["b"])
        kept = np.zeros(len(before), bool)
        kept[mapping] = True
        i, j = np.nonzero(np.triu((after != before) & (kept[:, None] | kept[None, :]), 1))
        r["target"], r["edits"] = mapping.astype(np.int16), np.stack([i, j, after[i, j]], 1).astype(np.int16)


def batch_indices(reactions, size, rng=None):
    """Index batches of similar size (less padding), shuffled if an rng is given."""
    idx = np.arange(len(reactions)) if rng is None else rng.permutation(len(reactions))
    out = []

    for s in range(0, len(idx), size * 50):
        chunk = sorted(idx[s:s + size * 50], key=lambda i: len(reactions[i]["a"]["x"]))
        out += [chunk[k:k + size] for k in range(0, len(chunk), size)]

    if rng is not None:
        rng.shuffle(out)

    return out


def batches(reactions, size, rng=None):
    return [[reactions[i] for i in b] for b in batch_indices(reactions, size, rng)]


class Collated(torch.utils.data.Dataset):
    """One epoch of padded batches, collation (pure Python / numpy) runs in worker processes so the GPU is not starved."""

    def __init__(self, reactions, index_batches):
        self.reactions, self.index_batches = reactions, index_batches

    def __len__(self):
        return len(self.index_batches)

    def __getitem__(self, k):
        return chem.collate([self.reactions[i] for i in self.index_batches[k]], "cpu")


def epoch_loader(reactions, size, rng, device, workers=int(os.environ.get("NPF_WORKERS", 4))):
    loader = torch.utils.data.DataLoader(Collated(reactions, batch_indices(reactions, size, rng)), batch_size=None, shuffle=False,
                                         num_workers=workers, prefetch_factor=4, pin_memory=device == "cuda")
    for batch in loader:
        yield {k: v.to(device, non_blocking=True) for k, v in batch.items()}


# ------------------------------------------------------------------ metrics

def firing(reaction, mapping, connectivity=True):
    """The transitions that must fire to turn A into B under a product -> precursor atom mapping, as a multiset
    over equivalence classes of precursor atoms. connectivity=True, bonds formed and broken, atoms up to
    skeleton symmetry - invariant under resonance and tautomerism on either side, the usual notion of a
    correct atom mapping. connectivity=False, every change of bond type, atoms up to exact symmetry."""
    a, bb = reaction["a"], chem.dense_bonds(reaction["b"])
    before = chem.dense_bonds(a)
    after = np.zeros_like(before)
    after[np.ix_(mapping, mapping)] = bb
    kept = np.zeros(len(before), bool)
    kept[mapping] = True

    if connectivity:
        before, after = (before > 0).astype(np.int8), (after > 0).astype(np.int8)

    i, j = np.nonzero(np.triu((after != before) & (kept[:, None] | kept[None, :]), 1))

    return symmetric(a, i, j, before[i, j], after[i, j], "skeleton" if connectivity else "symmetry")


def tokens_moved(reaction, mapping):
    """Size of the firing vector under a mapping, bond tokens plus hydrogens and charges that have to move."""
    a, b = reaction["a"], reaction["b"]
    before = chem.BOND_ORDER[chem.dense_bonds(a)]
    after = np.zeros_like(before)
    after[np.ix_(mapping, mapping)] = chem.BOND_ORDER[chem.dense_bonds(b)]
    kept = np.zeros(len(before), bool)
    kept[mapping] = True
    bonds = np.abs(np.triu((after - before) * (kept[:, None] | kept[None, :]), 1)).sum()

    return float(bonds + np.abs(b["h"] - a["h"][mapping]).sum() + np.abs(b["q"] - a["q"][mapping]).sum())


def token_descent(reaction, mapping, passes=3):
    """Discrete counterpart of the consensus rounds, starting from the learned correspondence, re-seat one product
    atom at a time (swapping if the seat is taken) whenever that strictly shrinks the firing vector."""
    a, b = reaction["a"], reaction["b"]
    mapping = mapping.copy()
    best = tokens_moved(reaction, mapping)
    order_a, order_b = chem.BOND_ORDER[chem.dense_bonds(a)], chem.BOND_ORDER[chem.dense_bonds(b)]
    for _ in range(passes):
        improved = False

        # only atoms that take part in a firing under the current mapping can be badly seated
        mism = np.abs(order_b - order_a[np.ix_(mapping, mapping)]).sum(1) + np.abs(b["h"] - a["h"][mapping]) + np.abs(b["q"] - a["q"][mapping])
        for i in np.nonzero(mism > 0)[0]:
            for j in np.nonzero(a["element"] == b["element"][i])[0]:
                if j == mapping[i]:
                    continue

                trial = mapping.copy()
                holder = np.nonzero(mapping == j)[0]
                if len(holder):
                    trial[holder[0]] = mapping[i]

                trial[i] = j
                cost = tokens_moved(reaction, trial)
                if cost < best - 1e-9:
                    mapping, best, improved = trial, cost, True

        if not improved:
            break

    return mapping


def symmetric(a, i, j, old, new, classes="symmetry"):
    s = a[classes]

    return Counter((min(s[x], s[y]), max(s[x], s[y]), int(o), int(n)) for x, y, o, n in zip(i, j, old, new))


@torch.no_grad()
def evaluate(model, task, reactions, device):
    model.eval()
    stats = Counter()
    for rs in batches(reactions, 64):
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
        elif task == "map":
            for r, pred in zip(rs, model.decode(out, b)):
                if getattr(model, "petri", False) and REFINE:
                    pred = token_descent(r, pred)

                ok = r["a"]["skeleton"][pred] == r["a"]["skeleton"][r["target"]]
                ours, recorded = firing(r, pred), firing(r, r["target"])
                stats["atoms"] += len(ok)
                stats["atoms_ok"] += int(ok.sum())
                stats["n"] += 1
                stats["same_firing"] += ours == recorded
                stats["same_firing_strict"] += firing(r, pred, False) == firing(r, r["target"], False)

                # minimum-firing audit of the disagreements, whose firing vector moves fewer tokens?
                if ours != recorded:
                    n_ours, n_rec = tokens_moved(r, pred), tokens_moved(r, r["target"])
                    stats["disagree_fewer" if n_ours < n_rec else "disagree_equal" if n_ours == n_rec else "disagree_more"] += 1
        else:
            for r, pred in zip(rs, model.decode(out, b, rs)):
                a, before = r["a"], chem.dense_bonds(r["a"])

                # no clean recorded mapping
                true = r["edits"] if r["edits"] is not None else np.zeros((0, 3), np.int64)
                same = lambda e: symmetric(a, e[:, 0], e[:, 1], before[e[:, 0], e[:, 1]], e[:, 2])
                after = before.copy()
                after[pred[:, 0], pred[:, 1]] = pred[:, 2]
                after[pred[:, 1], pred[:, 0]] = pred[:, 2]
                hydrogens = a["h"] - (chem.BOND_ORDER[after] - chem.BOND_ORDER[before]).sum(1)
                stats["n"] += 1
                stats["exact"] += {tuple(e) for e in pred.tolist()} == {tuple(e) for e in true.tolist()}
                stats["product"] += product_found(r, pred)
                stats["exact_sym"] += same(pred) == same(true)
                capacity = np.array([chem.EXTRA_CAPACITY.get(int(e), 0) for e in a["element"]]) + np.maximum(-a["q"], 0)

                # no hydrogen place below its capacity
                stats["enabled"] += bool((hydrogens >= -capacity - 0.5).all())

    n = stats["n"]

    if task == "classify":
        f1 = [2 * stats["tp", c] / max(stats["pred", c] + stats["true", c], 1) for c in range(stats["classes"])]
        return {"accuracy": stats["correct"] / n, "macro_f1": float(np.mean(f1))}

    if task == "map":
        return {"reactions_same_firing_vector": stats["same_firing"] / n, "atoms_correct": stats["atoms_ok"] / stats["atoms"],
                "reactions_same_firing_vector_strict": stats["same_firing_strict"] / n,
                "disagree_ours_moves_fewer_tokens": stats["disagree_fewer"] / n, "disagree_equal": stats["disagree_equal"] / n,
                "disagree_ours_moves_more_tokens": stats["disagree_more"] / n}

    return {"product_top1": stats["product"] / n, "edits_exact_up_to_symmetry": stats["exact_sym"] / n, "edits_exact": stats["exact"] / n,
            "valence_valid": stats["enabled"] / n}


def recorded_products(reaction):
    """Every recorded product molecule (canonical, no stereochemistry) has to be predicted."""
    out = set()
    for smi in reaction["smiles"].split(">>")[1].split("."):
        out.add(chem.canonical_product(smi))

    return out


def product_found(reaction, edits):
    """The main recorded product is made by the firings, and every recorded product molecule (counter-ions of
    salts are spectators) is part of the final marking."""
    touched, everything = chem.marking_fragments(reaction["a"], edits)
    recorded = reaction["smiles"].split(">>")[1]

    return chem.canonical_product(recorded) in touched and recorded_products(reaction) <= everything


@torch.no_grad()
def evaluate_beam(model, reactions, device, width=5, dump=None):
    """Top-k product accuracy of the token game, the recorded product is among the k most probable markings.
    With dump, the candidates of every reaction are written there, as (id, [(edits, log probability, correct)])."""
    model.eval()
    hits, merged_hits, candidates = np.zeros(width), np.zeros(width), []
    for r in reactions:
        ranked = model.beam_search(chem.collate([r], device), width)
        found = [product_found(r, edits) for edits, _ in ranked]
        first = found.index(True) if True in found else width
        hits[first:] += 1

        # firing vectors that decode to the same molecules are one prediction, their probabilities add
        groups = {}
        for (edits, lp), ok in zip(ranked, found):
            key = frozenset(chem.marking_fragments(r["a"], edits)[0])
            total, right = groups.get(key, (-np.inf, False))
            groups[key] = (float(np.logaddexp(total, lp)), right or ok)

        order = [right for _, right in sorted(groups.values(), key=lambda g: -g[0])]
        first = order.index(True) if True in order else width
        merged_hits[first:] += 1
        candidates.append((r["id"], [(edits, lp, ok) for (edits, lp), ok in zip(ranked, found)]))

    if dump:
        Path(dump).parent.mkdir(parents=True, exist_ok=True)
        Path(dump).write_bytes(pickle.dumps(candidates))

    n = len(reactions)

    return {f"product_top{k + 1}": float(hits[k] / n) for k in range(width)} | {f"product_merged_top{k + 1}": float(merged_hits[k] / n) for k in range(width)}


MAIN_METRIC = {"classify": "accuracy", "map": "reactions_same_firing_vector", "forward": "product_top1"}


# ------------------------------------------------------------------ DRFP reference (classification only)

def drfp_baseline(train, val, test, device, seed, everything=None):
    from multiprocessing import Pool
    cache = Path("data/drfp.npy")

    # the cache is indexed by the rank of the reaction id among ALL reactions
    everything = everything or (train + val + test)

    if not cache.exists():
        with Pool(12) as pool:
            fps = pool.map(_drfp, [r["smiles"] for r in sorted(everything, key=lambda r: r["id"])], chunksize=128)

        np.save(cache, np.array(fps, np.uint8))

    rank = {r["id"]: k for k, r in enumerate(sorted(everything, key=lambda r: r["id"]))}
    fps = np.load(cache)
    tensor = lambda rs: (torch.as_tensor(fps[[rank[r["id"]] for r in rs]], dtype=torch.float32, device=device),
                         torch.as_tensor([r["label"] for r in rs], device=device))
    (xt, yt), (xv, yv), (xs, ys) = tensor(train), tensor(val), tensor(test)
    n_classes = max(r["label"] for r in everything) + 1
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(2048, 1024), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(1024, n_classes)).to(device)
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
        acc = float((net(xv).argmax(-1) == yv).float().mean())
        if acc >= best:
            best, state = acc, copy.deepcopy(net.state_dict())

    net.load_state_dict(state)
    pred = net(xs).argmax(-1)
    f1 = [2 * float(((pred == c) & (ys == c)).sum()) / max(float((pred == c).sum() + (ys == c).sum()), 1) for c in range(n_classes)]

    return {"accuracy": float((pred == ys).float().mean()), "macro_f1": float(np.mean(f1))}, sum(p.numel() for p in net.parameters())


def _drfp(smiles):
    from drfp import DrfpEncoder

    return DrfpEncoder.encode(smiles, n_folded_length=2048)[0]


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(EPOCHS), required=True)
    ap.add_argument("--model", default="npf", help="npf | pgnn | drfp | ablations: npf-nogate, npf-sigma, pgnn-sigma (classify), npf-oneshot, npf-noenabling "
                    "(forward), npf-noequilibrium, npf-nokept, npf-nomorphism, npf-nocost (map)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--limit", type=int, help="use only this many training reactions (smoke tests, data-efficiency)")
    ap.add_argument("--subset", type=int, help="uspto_mit: train on the reactions among this many random lines of the official "
                    "training file (data efficiency, the baselines get the same lines)")
    ap.add_argument("--root", default="results")
    ap.add_argument("--dataset", choices=["schneider50k", "uspto_mit"], default="schneider50k")
    ap.add_argument("--evaluate-only", action="store_true", help="re-score the saved weights on the test set")
    ap.add_argument("--no-beam", action="store_true", help="skip the beam search (top-k) evaluation")
    ap.add_argument("--dump-validation-beams", action="store_true", help="forward: also write the beam candidates of the validation "
                    "reactions that model selection did not use")
    ap.add_argument("--width", type=int, default=128, help="hidden width of the token game")
    ap.add_argument("--rounds", type=int, default=6, help="message-passing rounds of the token game")
    ap.add_argument("--attention", type=int, default=0, help="attention layers of the token game (default: 4, or 6 if width > 128)")
    ap.add_argument("--composites", type=int, default=0, help="token game: also offer substitutions, one firing for a break and a "
                    "formation at the shared atom, built from this many of the most probable breaks")
    ap.add_argument("--hops", type=int, default=0, help="token game: pair feature 'distance in the current marking', up to this many bonds")
    ap.add_argument("--batch", type=int, default=0, help="batch size (default: per task)")
    ap.add_argument("--amp", action="store_true", help="bfloat16 autocast")
    ap.add_argument("--tag", default="", help="suffix of the result files, e.g. -large")
    ap.add_argument("--lr", type=float, default=1e-3, help="peak learning rate of the one-cycle schedule")
    ap.add_argument("--clean-mapper", action="store_true", help="map: train on Schneider's training split only and write the predicted "
                    "mapping of every reaction to data/predicted_maps.pkl (input of `--model npf-sigma` in classify)")
    ap.add_argument("--maps", default="data/predicted_maps.pkl", help="classify with npf-sigma, the file of predicted maps to read the "
                    "firing vector from, for instance the one of the mapper that was trained without any recorded map")
    ap.add_argument("--net-maps", help="replace every recorded mapping by the one this file holds, before any target is built, "
                    "so that no head of any task sees a recorded map (benchmarks.chemistry.net_maps writes such a file)")
    ap.add_argument("--firing", action="store_true", help="classify: auxiliary task 'which transition types fired' (from recorded mappings)")
    ap.add_argument("--labels", type=int, help="classify: number of training reactions whose class label is used; the rest only "
                    "contribute their firing histograms (needs --firing to be of any use)")
    ap.add_argument("--size-split", action="store_true", help="classify: train on the smaller half of the training reactions, "
                    "test on the largest quarter of the test reactions (extrapolation in molecule size)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = chem.load(f"data/{args.dataset}.pkl")
    train, val, test = splits(data, "map-clean" if args.clean_mapper else args.task)
    train = train[:args.limit] if args.limit else train

    if args.subset:
        chosen = set(subset_lines(args.subset).tolist())
        train = [r for r in train if r["id"] in chosen]
        args.tag += f"-sub{args.subset}"
        print(f"subset of {args.subset} lines of the training file: {len(train)} usable training reactions", flush=True)

    if args.clean_mapper:
        args.tag += "-clean"

    # every target below is derived from the mapping, so replacing it here makes the whole run free of recorded maps
    if args.net_maps:
        # training never sees a recorded map. classification reads the firing vector of the test reactions too, so
        # those are replaced as well, forward prediction and mapping score the test reactions against the record
        use_predicted_firing(train, path=args.net_maps, blank_missing=True)

        if args.task == "classify":
            use_predicted_firing(val, test, path=args.net_maps, blank_missing=True)

        args.tag += "-netmaps"

    # classification from the explicit firing vector that our own mapper predicts
    if args.model in ("npf-sigma", "pgnn-sigma"):
        use_predicted_firing(train, val, test, path=args.maps, blank_missing=True)

    n_types = 0

    if args.task == "classify" and (args.firing or args.labels):
        if args.firing:
            n_types = attach_firing_histograms(train, val, test)

        if args.labels:
            for k, r in enumerate(train):
                r["labelled"] = k < args.labels

            if not args.firing:
                train = train[:args.labels]

        args.tag += ("-firing" if args.firing else "") + (f"-labels{args.labels}" if args.labels else "")

    if args.composites:
        args.tag += f"-comp{args.composites}"

    if args.size_split:
        size = lambda r: len(r["a"]["x"])
        small, large = np.median([size(r) for r in train]), np.quantile([size(r) for r in test], 0.75)
        train, val, test = [r for r in train if size(r) <= small], [r for r in val if size(r) <= small], [r for r in test if size(r) >= large]
        args.tag += "-sizesplit"
        print(f"size split: train on <= {small:.0f} precursor atoms ({len(train)} reactions), test on >= {large:.0f} ({len(test)})", flush=True)

    start = time.time()

    if args.model == "drfp":
        metrics, n_params = drfp_baseline(train, val, test, device, args.seed, data["reactions"])
        curve = []
    else:
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        petri = args.model.startswith("npf")
        model = {"classify": lambda: chem.Classifier(len(data["classes"]), petri=petri, gate=args.model not in ("npf-nogate", "npf-sigma", "npf-sigma-recorded"),
                                                            n_firing_types=n_types, explicit_firing=args.model.startswith("npf-sigma"),
                                                            mapped_atoms=args.model == "pgnn-sigma"),
                 "map": lambda: chem.Mapper(petri=petri, equilibrium=args.model != "npf-noequilibrium", kept_bonds=args.model != "npf-nokept",
                                                   morphism=args.model != "npf-nomorphism", token_cost=args.model != "npf-nocost"),
                 "forward": lambda: chem.Forward(petri=petri) if args.model in ("pgnn", "npf-oneshot") else chem.TokenGame(
                     d=args.width, rounds=args.rounds, attention=args.attention or (6 if args.width > 128 else 4),
                     enabling=args.model != "npf-noenabling", hops=args.hops, composites=args.composites)}[args.task]().to(device)
        n_params = sum(p.numel() for p in model.parameters())
        epochs = args.epochs or EPOCHS[args.task]
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=epochs * len(batch_indices(train, args.batch or BATCH[args.task])), pct_start=0.1)
        best, state, curve = -1.0, None, []
        weights = Path(args.root) / ("chem" if args.dataset == "schneider50k" else args.dataset) / args.task / f"{args.model}{args.tag}-{args.seed}{'-n' + str(args.limit) if args.limit else ''}.pt"

        if args.evaluate_only:
            state, epochs = torch.load(weights, map_location=device), 0

        for epoch in range(epochs):
            model.train()
            losses = []
            for b in epoch_loader(train, args.batch or BATCH[args.task], rng, device):
                if isinstance(model, chem.TokenGame):
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                        loss = model.loss(b)

                # every round of the consensus is supervised
                elif args.task == "map":
                    loss = torch.stack([model.loss(lp, b) for lp in model(b, all_rounds=True)]).mean()
                else:
                    out = model(b)

                    if args.task == "classify":
                        known = b["labelled"]
                        loss = (F.cross_entropy(out[known], b["label"][known]) if known.any() else 0.0) \
                            + 0.5 * model.auxiliary_loss(b) + model.firing_loss(b)
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
                losses.append(loss.item())

            score = evaluate(model, args.task, val[:1500], device)[MAIN_METRIC[args.task]]
            curve.append((epoch, float(np.mean(losses)), score))
            print(f"epoch {epoch}: loss {np.mean(losses):.4f}  val {MAIN_METRIC[args.task]} {score:.4f}  ({time.time() - start:.0f}s)", flush=True)

            if score >= best:
                best, state = score, copy.deepcopy(model.state_dict())
                weights.parent.mkdir(parents=True, exist_ok=True)

                # long runs should survive a crash
                torch.save(state, weights)

        model.load_state_dict(state)
        metrics = evaluate(model, args.task, test, device)

        if isinstance(model, chem.TokenGame) and not args.no_beam:
            beams = weights.parent.parent / "beams" / f"{weights.stem}-test.pkl"
            metrics |= {k + "_beam": v for k, v in evaluate_beam(model, test, device, dump=beams).items()}

            # candidates on validation reactions that were not used for model selection, for a verifier
            if args.dump_validation_beams:
                evaluate_beam(model, val[1500:1500 + VERIFIER_REACTIONS], device, dump=weights.parent.parent / "beams" / f"{weights.stem}-val.pkl")

        torch.save(state, weights)

        # test reactions that RDKit cannot read count as wrong, so that the denominator is the official one
        if args.dataset == "uspto_mit" and args.task == "forward":
            metrics |= {f"{k}_official": v * len(test) / USPTO_MIT_TEST_LINES for k, v in metrics.items() if k.startswith("product_")}

    if args.clean_mapper:
        write_predicted_maps(model, data["reactions"], device)

    result = {"task": args.task, "model": args.model, "seed": args.seed, "params": n_params, "n_train": len(train), "n_test": len(test),
              "train_seconds": time.time() - start, "curve": curve, "metrics": metrics}
    folder = "chem" if args.dataset == "schneider50k" else args.dataset
    out = Path(args.root) / folder / args.task / f"{args.model}{args.tag}-{args.seed}{'-n' + str(args.limit) if args.limit else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    # keep the record of the training run
    if args.evaluate_only and out.exists():
        previous = json.loads(out.read_text())
        result |= {"curve": previous["curve"], "train_seconds": previous["train_seconds"]}

    out.write_text(json.dumps(result, indent=1))
    print(f"chem/{args.task} {args.model} seed {args.seed}: " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()) + f"  ({n_params} params, {result['train_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
