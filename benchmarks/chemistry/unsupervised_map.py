"""Atom mapping learned without a single recorded map.

The mapping is a latent firing vector of the valence net. Only the two markings are observed, and every candidate
mapping is a sigma with m_B = m_A + C sigma, so the likelihood of the reaction is a sum over the candidates,

    log P(m_B | m_A) = log sum_sigma P(sigma),

with P from the mapper's own energy. The candidates come from the net itself, the cheapest firing vectors found by
exact minimisation, which needs no labels. Training raises the probability of the cheapest ones in proportion to
their posterior, which is expectation maximisation with a hard E step. Reactions whose cheapest firing vector is
unique pin the shared energy, and reactions with ties inherit the preference through it.

    uv run python -m benchmarks.chemistry.unsupervised_map --epochs 12
    uv run python -m benchmarks.chemistry.unsupervised_map --candidates data/candidates.pkl --epochs 12
"""
import argparse
import copy
import json
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from npf import chem

from .experiment import splits, subset_lines, token_descent, tokens_moved, write_predicted_maps
from .golden import same_cgr

CANDIDATES = 4


def candidates_of(reaction, n=CANDIDATES, budget=20000, tries=8, seed=0):
    """The cheapest distinct mappings of one reaction, found without any label.

    The seed is a matching by atom environment, then exact minimisation. Further candidates come from restarting the
    minimisation with one seat forbidden, which is the usual way to enumerate the next best assignments.
    """
    rng = np.random.default_rng(seed)
    start = chem.domains(reaction)
    seed_map = np.array([int(d[0]) for d in start], dtype=np.int64)

    # a feasible start, every product atom on a free seat of its own domain
    used = set()
    for i, d in enumerate(start):
        free = [int(j) for j in d if int(j) not in used]
        if not free:
            free = [int(j) for j in np.nonzero(reaction["a"]["element"] == reaction["b"]["element"][i])[0] if int(j) not in used]

        seed_map[i] = free[0]
        used.add(free[0])

    best, cost, exact = chem.branch_and_bound(reaction, token_descent(reaction, seed_map), budget=budget)
    found = {tuple(best.tolist()): (best, cost)}
    element = reaction["b"]["element"]
    for _ in range(tries):
        m = best.copy()
        for _ in range(rng.integers(1, 4)):
            i, j = rng.integers(0, len(m), 2)
            if element[i] == element[j]:
                m[i], m[j] = m[j], m[i]

        m = token_descent(reaction, m)
        found.setdefault(tuple(m.tolist()), (m, tokens_moved(reaction, m)))

    ranked = sorted(found.values(), key=lambda x: x[1])[:n]

    return [m for m, _ in ranked], [c for _, c in ranked], exact


def _work(args):
    reaction, seed = args

    try:
        maps, costs, exact = candidates_of(reaction, seed=seed)
    except Exception:
        return None

    return {"id": reaction["id"], "maps": [m.astype(np.int16) for m in maps], "costs": costs, "exact": exact}


def build_candidates(reactions, out, workers=10):
    with Pool(workers) as pool:
        rows = [row for row in pool.map(_work, [(r, k) for k, r in enumerate(reactions)], chunksize=32) if row]

    Path(out).write_bytes(pickle.dumps(rows))
    unique = np.mean([len(row["maps"]) == 1 for row in rows])
    print(f"{len(rows)} reactions, cheapest firing vector unique for {unique:.1%}, search exhaustive for "
          f"{np.mean([row['exact'] for row in rows]):.1%}", flush=True)

    return rows


def energies(model, b, maps):
    """Log probability the mapper gives to each candidate mapping of each reaction in the batch."""
    log_p = model(b)
    out = []
    for k, candidates in enumerate(maps):
        rows = torch.as_tensor(np.stack(candidates).astype(np.int64), device=log_p.device)
        mask = b["mask_b"][k]

        # the likelihood of a mapping is the sum of the seat log probabilities of the product atoms
        picked = log_p[k].gather(1, rows.T[:mask.sum()]).T
        out.append(picked.sum(1))

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="schneider50k")
    ap.add_argument("--candidates", default="data/candidates.pkl")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--subset", type=int, help="uspto_mit, the reactions among this many random lines of the training file")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0, help="E step, lower means the model commits harder to its current favourite among tied candidates")
    ap.add_argument("--build-only", action="store_true", help="write the candidate file and stop")
    ap.add_argument("--clean", action="store_true", help="schneider50k, train on the reactions of the classification training split only, "
                    "chosen without looking at any recorded map, keep the last epoch, and write the predicted maps of all reactions")
    ap.add_argument("--last-epoch", action="store_true", help="keep the last epoch, so that no recorded map takes part in the choice of the weights")
    ap.add_argument("--write-maps", default="data/predicted_maps_unsupervised.pkl")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = chem.load(f"data/{args.dataset}.pkl")
    train, val, _ = splits(data, "map")

    if args.clean:
        # no recorded map takes part, neither in the choice of the reactions nor in the choice of the epoch
        train = [r for r in data["reactions"] if r["split"] == "train" and len(r["b"]["x"]) <= len(r["a"]["x"])]
        args.tag += "-clean"

    if args.last_epoch:
        args.tag += "-last"

    train = train[:args.limit] if args.limit else train

    if args.subset:
        chosen = set(subset_lines(args.subset).tolist())
        train = [r for r in train if r["id"] in chosen]

    path = Path(args.candidates)
    if path.exists():
        rows = pickle.loads(path.read_bytes())
    else:
        rows = build_candidates(train, path, args.workers)

    if args.build_only:
        return

    by_id = {row["id"]: row for row in rows}
    train = [r for r in train if r["id"] in by_id]

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = chem.Mapper().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    steps = args.epochs * (len(train) // 16 + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 5e-4, total_steps=steps, pct_start=0.1)
    best, state, start = -1.0, None, time.time()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        order = rng.permutation(len(train))
        for k in range(0, len(train), 16):
            batch = [train[i] for i in order[k:k + 16]]
            if not batch:
                continue

            b = chem.collate(batch, device)
            maps = [by_id[r["id"]]["maps"] for r in batch]
            costs = [torch.tensor(by_id[r["id"]]["costs"], device=device, dtype=torch.float32) for r in batch]
            loss = torch.zeros((), device=device)
            for scores, cost, reaction in zip(energies(model, b, maps), costs, batch):
                cheapest = cost <= cost.min() + 1e-6

                # the E step. where the cheapest firing vector is unique the net fixes the target, and where several
                # are equally cheap the model's own posterior decides, so the preference learned on the unambiguous
                # reactions carries over to the ambiguous ones through the shared energy
                with torch.no_grad():
                    weight = torch.where(cheapest, torch.softmax(scores.detach() / args.temperature, 0), torch.zeros_like(scores))
                    weight = weight * cheapest
                    weight = weight / weight.sum().clamp(min=1e-9)

                # the M step raises the likelihood of the weighted candidates under the mapper itself, per product atom.
                # Normalising over the list of candidates instead would give no gradient where the cheapest firing
                # vector is unique
                loss = loss - (weight * scores).sum() / len(reaction["b"]["x"])

            loss = loss / len(batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())

        # the recorded maps are never used for training, only to watch what the model agrees with
        model.eval()
        agree = 0

        with torch.no_grad():
            for k in range(0, min(len(val), 500), 16):
                rs = val[k:k + 16]
                b = chem.collate(rs, device)
                for r, pred in zip(rs, model.decode(model(b), b)):
                    m = token_descent(r, pred.astype(np.int64).copy())
                    agree += same_cgr(r, m, r["target"].astype(np.int64))

        score = agree / min(len(val), 500)
        print(f"epoch {epoch}: loss {np.mean(losses):.4f}  agreement with the recorded validation maps {score:.4f}  ({time.time() - start:.0f}s)", flush=True)

        # with --clean the recorded maps are only watched, the last epoch is kept
        if args.clean or args.last_epoch or score >= best:
            best, state = score, copy.deepcopy(model.state_dict())

    out = Path("results") / ("chem" if args.dataset == "schneider50k" else args.dataset) / "map" / f"npf-unsupervised{args.tag}-{args.seed}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out)
    out.with_suffix(".json").write_text(json.dumps({"task": "map", "model": "npf-unsupervised", "seed": args.seed, "n_train": len(train),
                                                    "params": sum(p.numel() for p in model.parameters()), "train_seconds": time.time() - start,
                                                    "recorded_map_agreement_val": best}, indent=1))
    print(f"wrote {out}")

    if args.clean:
        model.load_state_dict(state)

        with torch.no_grad():
            write_predicted_maps(model, [r for r in data["reactions"] if len(r["b"]["x"]) <= len(r["a"]["x"])], device, args.write_maps)


if __name__ == "__main__":
    main()
