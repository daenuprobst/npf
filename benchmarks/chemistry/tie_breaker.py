"""A learned rate law that orders the firing vectors the net cannot tell apart, trained without any recorded map.

The exact mapper of npf.chem.exact leaves 15 % of reactions with several equally cheap firing vectors, and a uniform
choice among them scores 86.8 % where the curated map is among the optima for 94.1 %. That gap is the only thing a
learned model is still needed for. The training signal comes from the net alone: for a reaction whose cheapest firing
vector is unique the net fixes the target, with no label of any kind, and the energy learned on those reactions orders
the ties of the others.

    uv run python -m benchmarks.chemistry.tie_breaker build --limit 12000   # data/tie_classes_schneider50k.pkl
    uv run python -m benchmarks.chemistry.tie_breaker train                 # results/chem/map/npf-tie-0.pt
    uv run python -m benchmarks.chemistry.tie_breaker evaluate              # results/chem/tie_breaker.json
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
from npf.chem import exact

from .exact_map import load as load_reactions
from .golden import same_cgr

OPTIONS = dict(secondary=(1, 1, 1), labile_h=False, ch_places=True)
CLASSES = Path("data/tie_classes_schneider50k.pkl")
WEIGHTS = Path("results/chem/map/npf-tie-0.pt")


def classes_of(job):
    """The distinct optimal mappings of one reaction, merged by condensed graph. No recorded map is read."""
    reaction, seconds, limit, hint = job

    try:
        maps = exact.optimal_mappings(reaction, limit=limit, seconds=seconds, workers=1, hint=hint, **OPTIONS)
    except Exception:
        maps = None

    if not maps:
        return {"id": reaction["id"], "classes": None}

    distinct = []
    for m in maps:
        if not any(same_cgr(reaction, m, other) for other in distinct):
            distinct.append(m)

    return {"id": reaction["id"], "classes": [m.astype(np.int16) for m in distinct], "complete": len(maps) < limit}


def build(limit, seconds, processes, out=CLASSES):
    reactions = [r for r in chem.load("data/schneider50k.pkl")["reactions"]
                 if r["split"] == "train" and len(r["b"]["x"]) <= len(r["a"]["x"])][:limit]
    known = pickle.loads(Path("data/exact_maps_schneider50k.pkl").read_bytes())
    fits = lambda r: r["id"] in known and len(known[r["id"]]) == len(r["b"]["x"])
    start = time.time()
    rows = []

    with Pool(processes) as pool:
        for row in pool.imap(classes_of, [(r, seconds, 32, known[r["id"]] if fits(r) else None) for r in reactions], chunksize=1):
            rows.append(row)

            if len(rows) % 500 == 0:
                done = [x for x in rows if x["classes"]]
                unique = np.mean([len(x["classes"]) == 1 for x in done]) if done else 0
                print(f"{len(rows)}/{len(reactions)} enumerated {len(done) / len(rows):.3f} unique {unique:.3f} "
                      f"{time.time() - start:.0f}s", flush=True)

    out.write_bytes(pickle.dumps({x["id"]: x for x in rows}))
    done = [x for x in rows if x["classes"]]
    print(f"{len(rows)} reactions, enumerated {len(done) / len(rows):.1%}, "
          f"cheapest firing vector unique for {np.mean([len(x['classes']) == 1 for x in done]):.1%}, wrote {out}")


def train(epochs, seed, device, batch=16):
    """Maximise the likelihood of the net's own unique answer. The recorded maps are never read."""
    rows = pickle.loads(CLASSES.read_bytes())
    data = chem.load("data/schneider50k.pkl")
    pool = [r for r in data["reactions"] if r["id"] in rows and rows[r["id"]]["classes"] is not None]
    unique = [r for r in pool if len(rows[r["id"]]["classes"]) == 1]
    print(f"{len(pool)} reactions enumerated, {len(unique)} with a unique cheapest firing vector, training on those", flush=True)

    # the target of a training reaction is the net's own answer, not a record
    supervised = []
    for r in unique:
        rr = copy.copy(r)
        rr["target"] = rows[r["id"]]["classes"][0].astype(np.int16)
        supervised.append(rr)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = chem.Mapper().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    steps = epochs * (len(supervised) // batch + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 5e-4, total_steps=steps, pct_start=0.1)
    start = time.time()

    for epoch in range(epochs):
        model.train()
        losses = []
        order = rng.permutation(len(supervised))

        for k in range(0, len(supervised), batch):
            rs = [supervised[i] for i in order[k:k + batch]]
            if not rs:
                continue

            b = chem.collate(rs, device)
            loss = chem.Mapper.loss(model(b), b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())

        # no validation against recorded maps, the last epoch is kept
        print(f"epoch {epoch}: loss {np.mean(losses):.4f}  ({time.time() - start:.0f}s)", flush=True)

    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS)
    print(f"wrote {WEIGHTS}")


@torch.no_grad()
def priors(reactions, device, batch=16):
    """Per reaction, the cost of every seat under the learned energy, lower is better."""
    model = chem.Mapper().to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.eval()
    out = {}

    for k in range(0, len(reactions), batch):
        rs = reactions[k:k + batch]
        b = chem.collate(rs, device)
        log_p = model(b).cpu().numpy()
        for row, r in enumerate(rs):
            n_b, n_a = len(r["b"]["x"]), len(r["a"]["x"])
            out[r["id"]] = -log_p[row, :n_b, :n_a]

    return out


def work(job):
    reaction, prior, seconds = job

    try:
        mapping, _, proved = exact.solve(reaction, prior=prior, seconds=seconds, workers=2, **OPTIONS)
    except Exception:
        return {"id": reaction["id"], "correct": False, "proved": False}

    target = reaction["target"].astype(np.int64)
    correct = mapping is not None and same_cgr(reaction, mapping, target)

    return {"id": reaction["id"], "correct": bool(correct), "proved": bool(proved)}


def evaluate(seconds, processes, device, out="results/chem/tie_breaker.json"):
    reactions = load_reactions("golden")
    prior = priors(reactions, device)
    rows = []

    with Pool(processes) as pool:
        for row in pool.imap(work, [(r, prior[r["id"]], seconds) for r in reactions], chunksize=1):
            rows.append(row)

            if len(rows) % 200 == 0:
                print(f"{len(rows)}/{len(reactions)} correct {np.mean([x['correct'] for x in rows]):.4f}", flush=True)

    summary = {"reactions": len(rows), "correct": float(np.mean([x["correct"] for x in rows])),
               "proved": float(np.mean([x["proved"] for x in rows])),
               "correct_when_proved": float(np.mean([x["correct"] for x in rows if x["proved"]]))}
    Path(out).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["build", "train", "evaluate"])
    ap.add_argument("--limit", type=int, default=12000)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--processes", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.phase == "build":
        build(args.limit, args.seconds, args.processes)
    elif args.phase == "train":
        train(args.epochs, args.seed, device)
    else:
        evaluate(args.seconds, max(args.processes // 2, 1), device)
