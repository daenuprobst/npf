"""Train one model on one setting and write its test metrics to results/<task>/<regime>/<model>-<seed>.json

    uv run python -m benchmarks.synthetic.experiment --task transitions --regime petri --model npf --seed 0
"""
import argparse
import copy
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from npf import batching, datasets, models

REGIMES = {
    # main task, which transitions fired, and how often, between two sampled states of a stochastic token game
    # (petri-ode, same task on deterministic continuous token flow, so there is no irreducible noise)
    "transitions": {"graph": dict(max_arity=1), "petri": dict(max_arity=2), "petri-ode": dict(max_arity=2, ode=True)},
    # secondary task, continuous token flow, predict the next state
    "next": {
        "graph-sat": dict(max_arity=1, kind="sat"),
        "petri-sat": dict(max_arity=2, kind="sat"),
        "petri-min": dict(max_arity=2, kind="min"),
    },
}

# number of places, training only ever sees SMALL
SMALL, LARGE = (6, 12), (20, 30)


def make_splits(task, regime):
    cfg = REGIMES[task][regime]

    if task == "transitions":
        make = datasets.make_flow_pairs if cfg.get("ode") else datasets.make_pairs
        more_tokens, longer_gap = (dict(scale=12.0), dict(gap=(5, 8))) if cfg.get("ode") else (dict(tokens=24), dict(gap=(1.0, 2.0)))
        mk = lambda seed, n, places, **kw: make(seed, n, places, cfg["max_arity"], 64, **kw)
        return {
            "train": mk(1, 300, SMALL), "val": mk(2, 40, SMALL), "test": mk(3, 100, SMALL),
            # 2-3x bigger nets
            "test-large": mk(4, 100, LARGE),
            # 3x more tokens
            "test-tokens": mk(5, 100, SMALL, **more_tokens),
            # states further apart than ever seen in training
            "test-gap": mk(6, 100, SMALL, **longer_gap),
        }

    mk = lambda seed, n, places, n_traj, n_steps, **kw: datasets.make_flows(
        seed, n, places, cfg["max_arity"], n_traj, cfg["kind"], n_steps, **kw)

    return {
        "train": mk(1, 300, SMALL, 8, 8), "val": mk(2, 40, SMALL, 8, 8), "test": mk(3, 100, SMALL, 4, 40),
        "test-large": mk(4, 100, LARGE, 4, 40), "test-tokens": mk(5, 100, SMALL, 4, 40, scale=12.0),
    }


def load_splits(task, regime, root):
    # caches are pickles of npf.nets.Net and npf.datasets.Group. the data are regenerated from fixed seeds if missing
    path = Path(root) / "cache" / f"{task}-{regime}-v2.pkl"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(make_splits(task, regime)))

    return pickle.loads(path.read_bytes())


# ------------------------------------------------------------------ evaluation

@torch.no_grad()
def predict(model, groups, device, chunk=20):
    """Model output for every sample of every group, split back per group."""
    out = []

    for i in range(0, len(groups), chunk):
        gs = groups[i:i + chunk]
        flat = model(batching.collate(gs, [np.arange(len(g.m)) for g in gs], device)).double().cpu().numpy()
        sizes = [len(g.m) * (g.net.n_trans if g.sigma is not None else g.net.n_places) for g in gs]
        out += [x.reshape(len(g.m), -1) for g, x in zip(gs, np.split(flat, np.cumsum(sizes)[:-1]))]

    return out


def transition_metrics(groups, preds):
    err, row, ker, tgt, unexplained = [], [], [], [], []
    hit_t = hit_s = consistent = 0

    for g, p in zip(groups, preds):
        e = p - g.sigma

        # part of the error the state equation alone would have removed
        e_row = e @ (g.net.C_pinv @ g.net.C).T
        err.append(e.ravel())
        row.append(e_row.ravel())
        ker.append((e - e_row).ravel())
        tgt.append(g.sigma.ravel())
        r = np.rint(p)
        hit_t += (r == g.sigma).sum()
        hit_s += (r == g.sigma).all(1).sum()
        consistent += (np.abs(g.m + r @ g.net.C.T - g.m_b).max(1) < 0.5).sum()
        unexplained.append(np.abs(g.m + p @ g.net.C.T - g.m_b).sum(1))

    err, row, ker, tgt = map(np.concatenate, (err, row, ker, tgt))
    n_samples = sum(len(g.m) for g in groups)
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)))

    return {
        "rmse": rms(err), "mae": float(np.abs(err).mean()), "r2": float(1 - (err ** 2).sum() / ((tgt - tgt.mean()) ** 2).sum()),
        # error inside im C^T (fixed by the two states) vs inside ker C
        "rmse_row": rms(row), "rmse_ker": rms(ker),
        "acc_transition": float(hit_t / len(err)), "acc_sample": float(hit_s / n_samples),
        # rounded prediction really takes state A to state B
        "consistent": float(consistent / n_samples),
        # |M_A + C sigma_hat - M_B|_1 per sample
        "unexplained_tokens": float(np.concatenate(unexplained).mean()),
        "min_pred": float(min(p.min() for p in preds)),
        "target_mean": float(tgt.mean()), "target_std": float(tgt.std()),
    }


def evaluate_transitions(model, groups, device):
    out = transition_metrics(groups, predict(model, groups, device))

    # no T-invariants, sigma is a function of the two states
    acyclic = [g for g in groups if g.net.n_tinv == 0]
    if acyclic:
        out["rmse_no_tinv"] = transition_metrics(acyclic, predict(model, acyclic, device))["rmse"]

    return out


@torch.no_grad()
def evaluate_next(model, groups, device, horizons=(1, 8, 40), chunk=25):
    sq_err = {h: 0.0 for h in horizons}
    sq_change = dict(sq_err)
    drift = {h: [] for h in horizons}
    negative = total = 0
    most_negative = 0.0
    firing = []
    for i in range(0, len(groups), chunk):
        gs = [datasets.Group(g.net, g.states[:, 0], states=g.states, fired=g.fired) for g in groups[i:i + chunk]]
        b = batching.collate(gs, [np.arange(len(g.m)) for g in gs], device)
        sizes = np.cumsum([g.m.size for g in gs])[:-1]
        m = b.m

        # the hidden layer of NPF is a firing vector, compare it with the true one
        if hasattr(model, "step"):
            sigma = model(b, return_firing=True)[1].double().cpu().numpy()
            for g, s in zip(gs, np.split(sigma, np.cumsum([g.fired[:, 0].size for g in gs])[:-1])):
                firing.append((s.reshape(g.fired[:, 0].shape), g.fired[:, 0], g.net))

        for h in range(1, max(horizons) + 1):
            raw = model(b, m)
            negative += int((raw < -1e-4).sum())
            total += raw.numel()
            most_negative = min(most_negative, float(raw.min()))

            # feeding negative markings back would only hurt the unconstrained models
            m = raw.clamp(min=0)

            if h in horizons:
                for g, x in zip(gs, np.split(raw.double().cpu().numpy(), sizes)):
                    x, x0, xh = x.reshape(len(g.m), -1), g.states[:, 0], g.states[:, h]
                    sq_err[h] += ((x - xh) ** 2).sum()
                    sq_change[h] += ((xh - x0) ** 2).sum()

                    if g.net.X.shape[1]:
                        drift[h] += list(np.linalg.norm((x - x0) @ g.net.X, axis=1) / np.linalg.norm(x0 @ g.net.X, axis=1).clip(1e-9))

    # 1.0 = "nothing changes"
    out = {f"nrmse@{h}": float(np.sqrt(sq_err[h] / sq_change[h])) for h in horizons}

    # relative violation of the conservation laws
    out |= {f"drift@{h}": float(np.mean(drift[h])) for h in horizons}
    out |= {"negative_frac": negative / total, "most_negative": most_negative}

    if firing:
        s, t = (np.concatenate([f[k].ravel() for f in firing]) for k in (0, 1))
        ker = np.concatenate([((p - q) - (p - q) @ (n.C_pinv @ n.C).T).ravel() for p, q, n in firing])
        out |= {"firing_corr": float(np.corrcoef(s, t)[0, 1]), "firing_nrmse": float(np.sqrt(np.mean((s - t) ** 2)) / t.std()),
                "firing_nrmse_ker": float(np.sqrt(np.mean(ker ** 2)) / t.std())}

    return out


# ------------------------------------------------------------------ training

def train(model, task, splits, device, iters, seed, log_every=250):
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=2e-5)
    key = "sigma" if task == "transitions" else "y"
    validate = (lambda: evaluate_transitions(model, splits["val"], device)["rmse"]) if task == "transitions" else \
        (lambda: float(np.sqrt(np.mean(np.concatenate([(p - g.y).ravel() for g, p in zip(splits["val"], predict(model, splits["val"], device))]) ** 2))))
    best, best_state, curve = np.inf, None, []
    for it in range(1, iters + 1):
        gs = [splits["train"][i] for i in rng.choice(len(splits["train"]), 16, replace=False)]
        rows = [rng.choice(len(g.m), 32, replace=False) for g in gs]
        batch, target = batching.collate(gs, rows, device), batching.flat_targets(gs, rows, key, device)
        loss = F.mse_loss(model(batch), target)

        # e.g. learned incidence, the learned net has to explain the recorded firings
        if hasattr(model, "auxiliary_loss"):
            loss = loss + model.auxiliary_loss(batch, target)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if it % log_every == 0 or it == iters:
            model.eval()
            val = validate()
            model.train()
            curve.append((it, loss.item(), val))

            if val < best:
                best, best_state = val, copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(REGIMES), default="transitions")
    ap.add_argument("--regime", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--root", default="results")
    ap.add_argument("--prepare", action="store_true", help="only generate and cache the dataset")
    args = ap.parse_args()

    splits = load_splits(args.task, args.regime, args.root)

    if args.prepare:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    model = models.build(args.model, args.task).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    start, curve = time.time(), []

    if n_params:
        curve = train(model, args.task, splits, device, args.iters, args.seed)

    model.eval()
    evaluate = evaluate_transitions if args.task == "transitions" else evaluate_next
    result = {
        "task": args.task, "regime": args.regime, "model": args.model, "seed": args.seed, "params": n_params,
        "train_seconds": time.time() - start, "curve": curve,
        "metrics": {name: evaluate(model, groups, device) for name, groups in splits.items() if name.startswith("test")},
    }
    out = Path(args.root) / args.task / args.regime / f"{args.model}-{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    head = result["metrics"]["test"]
    print(f"{args.task}/{args.regime} {args.model:>10} seed {args.seed}: " + "  ".join(f"{k}={v:.4g}" for k, v in list(head.items())[:6])
          + f"  ({n_params} params, {result['train_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
