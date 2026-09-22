"""Re-rank the beam candidates of the token game with a verifier that reads every candidate through the state equation.

The verifier is trained on the candidates of validation reactions, which neither the token game nor its model
selection has seen, and scored on the candidates of the test reactions. Both files are written by
benchmarks.chemistry.experiment (the second one with --dump-validation-beams).

    uv run python -m benchmarks.chemistry.verify results/uspto_mit/beams/npf-deep-0 --dataset uspto_mit
"""

import argparse
import copy
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from npf import chem

from .experiment import USPTO_MIT_TEST_LINES


def groups(path, reactions, need_correct):
    """(reaction, candidates) with candidates = [(edits, log probability, correct)]."""
    out = []
    for rid, candidates in pickle.loads(Path(path).read_bytes()):
        if candidates and (not need_correct or any(ok for _, _, ok in candidates)):
            out.append((reactions[rid], candidates))

    return out


def collate(batch, device):
    """One row per candidate. Returns the precursor tensors repeated per candidate, the candidate edits, the beam
    log probabilities, the correctness flags and the index of the reaction of every row.
    """
    b = chem.collate([r for r, _ in batch], device)
    owner = torch.tensor(
        [k for k, (_, cands) in enumerate(batch) for _ in cands], device=device
    )
    rows = {
        key: value[owner]
        for key, value in b.items()
        if torch.is_tensor(value) and value.shape[:1] == (len(batch),)
    }
    n = rows["ba"].shape[1]
    edits = torch.zeros(len(owner), n, n, dtype=torch.long, device=device)
    flat = [c for _, cands in batch for c in cands]
    for row, (e, _, _) in enumerate(flat):
        if len(e):
            i, j, t = torch.as_tensor(np.asarray(e, dtype=np.int64), device=device).T
            edits[row, i, j] = t + 1
            edits[row, j, i] = t + 1

    log_p = torch.tensor([lp for _, lp, _ in flat], dtype=torch.float32, device=device)
    correct = torch.tensor([ok for _, _, ok in flat], device=device)

    return rows, edits, log_p, correct, owner


def group_logsumexp(x, owner, n):
    top = torch.full((n,), -1e4, device=x.device).scatter_reduce(
        0, owner, x, "amax", include_self=True
    )

    return (
        top
        + torch.zeros(n, device=x.device)
        .scatter_add(0, owner, (x - top[owner]).exp())
        .clamp(min=1e-30)
        .log()
    )


def listwise_loss(score, correct, owner, n):
    """Negative log of the probability mass that a softmax over the candidates of a reaction puts on the correct ones."""
    return (
        group_logsumexp(score, owner, n)
        - group_logsumexp(score.masked_fill(~correct, -1e4), owner, n)
    ).mean()


@torch.no_grad()
def top1(model, data, device, size=16):
    """Number of reactions whose best candidate is correct, before and after re-ranking."""
    model.eval()
    before = after = 0

    for start in range(0, len(data), size):
        batch = data[start : start + size]
        rows, edits, log_p, correct, owner = collate(batch, device)
        score = model(rows, edits, log_p)
        for k in range(len(batch)):
            mine = owner == k
            before += bool(correct[mine][log_p[mine].argmax()])
            after += bool(correct[mine][score[mine].argmax()])

    return before, after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "beams", help="prefix of the beam files, <prefix>-val.pkl and <prefix>-test.pkl"
    )
    ap.add_argument("--dataset", default="schneider50k")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--attention", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    reactions = {r["id"]: r for r in chem.load(f"data/{args.dataset}.pkl")["reactions"]}
    train = groups(args.beams + "-val.pkl", reactions, need_correct=True)
    test = groups(args.beams + "-test.pkl", reactions, need_correct=False)
    order = rng.permutation(len(train))
    held_out, train = [train[i] for i in order[: len(train) // 10]], [
        train[i] for i in order[len(train) // 10 :]
    ]
    print(
        f"{len(train)} training reactions with a correct candidate, {len(held_out)} held out, {len(test)} test reactions",
        flush=True,
    )

    model = chem.Verifier(attention=args.attention).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    steps = args.epochs * (len(train) // 16 + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, 5e-4, total_steps=steps, pct_start=0.1
    )
    best, state, start = -1, None, time.time()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        shuffled = rng.permutation(len(train))
        for k in range(0, len(train), 16):
            batch = [train[i] for i in shuffled[k : k + 16]]
            rows, edits, log_p, correct, owner = collate(batch, device)
            loss = listwise_loss(model(rows, edits, log_p), correct, owner, len(batch))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())

        before, after = top1(model, held_out, device)
        print(
            f"epoch {epoch}: loss {np.mean(losses):.4f}  held-out top-1 {before / len(held_out):.4f} -> "
            f"{after / len(held_out):.4f}  ({time.time() - start:.0f}s)",
            flush=True,
        )

        if after >= best:
            best, state = after, copy.deepcopy(model.state_dict())

    model.load_state_dict(state)
    before, after = top1(model, test, device)
    n = USPTO_MIT_TEST_LINES if args.dataset == "uspto_mit" else len(test)
    result = {
        "beams": args.beams,
        "n_test": n,
        "n_train": len(train),
        "params": sum(p.numel() for p in model.parameters()),
        "train_seconds": time.time() - start,
        "product_top1_beam": before / n,
        "product_top1_verified": after / n,
    }
    out = (
        Path(args.beams).parent.parent / "verifier" / (Path(args.beams).name + ".json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    torch.save(state, out.with_suffix(".pt"))
    print(
        f"test top-1 {before / n:.4f} -> {after / n:.4f} with the verifier ({n} reactions in the denominator)"
    )


if __name__ == "__main__":
    main()
