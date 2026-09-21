"""The protocol of the PGNN paper, applied to the transition prediction task. One fixed net, 100 samples, a 70/30
split, 300 epochs of Adam, parameters per net (transductive), a clean and a noisy scenario. This is the setting in
which the paper trains its linear special case (Eq. 13), so that model is included here.

    uv run python -m benchmarks.synthetic.paper_protocol
"""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from npf import batching, datasets, models, nets, simulate
from .experiment import transition_metrics

N_NETS, N_SEEDS, EPOCHS, BATCH, DT = 5, 3, 300, 10, 0.5


def make_net_data(seed, noise):
    rng = np.random.default_rng(seed)
    net = nets.random_net(rng, n_places=7, n_base=4, max_arity=2)
    M0 = rng.integers(0, 9, size=(100, 7)) * (rng.random((100, 7)) > 0.25)
    t_a = rng.uniform(0.0, 0.5, 100)
    MA, MB, sigma = simulate.gillespie_pairs(net, M0, t_a, t_a + DT, rng)

    # the models never see the true states
    observe = lambda M: np.maximum(M + noise * rng.uniform(-1, 1, M.shape), 0.0)
    seen = datasets.Group(net, observe(MA), observe(MB), np.full(100, DT), sigma)
    true = datasets.Group(net, MA, MB, np.full(100, DT), sigma)

    return net, seen, true


def build(name, net, noise):
    if name == "pgnn-linear":
        return models.LinearPGNN(net)

    if name == "npf":
        return models.NPF("transitions", hidden=16, noisy_states=noise > 0)

    # insists on the state equation even though the observed states are noisy
    if name == "npf-hard":
        return models.NPF("transitions", hidden=16)

    return models.build(name, "transitions", hidden=16)


def run(name, net, seen, true, noise, seed, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build(name, net, noise).to(device)
    train, test = np.arange(70), np.arange(70, 100)

    if sum(p.numel() for p in model.parameters()):
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS * len(train) // BATCH, eta_min=1e-4)
        for _ in range(EPOCHS):
            for rows in np.split(rng.permutation(train), len(train) // BATCH):
                loss = F.mse_loss(model(batching.collate([seen], [rows], device)), batching.flat_targets([seen], [rows], "sigma", device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()

    model.eval()

    with torch.no_grad():
        pred = model(batching.collate([seen], [test], device)).double().cpu().numpy().reshape(len(test), -1)

    # scored against the true states
    held_out = datasets.Group(net, true.m[test], true.m_b[test], true.dt[test], true.sigma[test])

    return transition_metrics([held_out], [pred]) | {"params": sum(p.numel() for p in model.parameters())}


def job(args):
    name, noise, net_seed, seed = args
    torch.set_num_threads(1)
    net, seen, true = make_net_data(100 + net_seed, noise)

    return name, noise, run(name, net, seen, true, noise, seed, "cpu")


def main(root="results"):
    jobs = [
        (name, noise, net_seed, seed)
        for noise in (0.0, 1.0)
        for name in ["se-only", "gnn", "pgnn-linear", "pgnn", "pgnn+", "npf"] + (["npf-hard"] if noise else [])
        for net_seed in range(N_NETS) for seed in range(N_SEEDS if name != "se-only" else 1)
    ]
    runs = {}

    # the models are tiny, one CPU core each beats sharing a GPU
    with ProcessPoolExecutor(8) as pool:
        for name, noise, metrics in pool.map(job, jobs):
            runs.setdefault(f"{name}|noise={noise}", []).append(metrics)

    out = {key: {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))) for k in rs[0]} for key, rs in runs.items()}

    # paired comparison on identical (net, seed) runs, how often is NPF the better model?
    for key, rs in runs.items():
        ours = runs[f"npf|noise={key.split('=')[1]}"]
        if len(rs) == len(ours):
            out[key]["npf_wins"] = (float(np.mean([a["rmse"] < b["rmse"] for a, b in zip(ours, rs)])), 0.0)

    for key, m in out.items():
        print(f"{key:>24}: " + "  ".join(f"{k}={m[k][0]:.3f}±{m[k][1]:.3f}" for k in ("rmse", "mae", "rmse_row", "rmse_ker", "consistent", "params")), flush=True)

    path = Path(root) / "paper_protocol.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
