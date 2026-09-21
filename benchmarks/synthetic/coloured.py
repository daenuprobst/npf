"""Coloured tokens. Tokens carry features and transitions compute on them (THEORY.md, P9).

The ground truth, hidden from the models, is a continuous coloured Petri net. Place p holds mass m_p of colour c_p in
R^2. Transition t fires at rate k_t prod_p (m_p / (K_p + m_p))^Pre(p,t) * guard_t(c_in) with the guard
(1 + <c_in, u_t>) / 2, where c_in is the Pre-weighted mean colour of its input places, and it emits tokens of colour
o_t = Rot(theta_t) c_in. Places mix what arrives with what stays,
    d(m_p c_p)/dt = sum_in Pos v_t o_t - sum_out Pre v_t c_p.

npf-coloured  Token game for the masses with a learned colour guard in the rate law, a neural arc expression o_theta
              for the colour of emitted tokens, and the mixing rule that token conservation dictates,
                  c_p <- ((m_p - out_p) c_p + sum_in Pos v_t o_t) / m_p'
              The aggregation is the flow-weighted mean and the update gate is the share of tokens that stayed.
              Neither is learned.
npf-free      Ablation with the same masses, but the colour update is a free function of the same messages.
pgnn+         Generic message passing between places and transitions on the features [mass, colour].

Provable for npf-coloured and any weights. Masses obey P1 and P2, a place without inflow keeps its colour exactly,
and new colours are convex combinations of the old colour and the emitted ones.

    uv run python -m benchmarks.synthetic.coloured    # writes results/coloured.json
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from npf import batching, datasets, nets
from npf.layers import scatter_sum
from npf.models.coloured import ColouredNPF, ColouredPGNN

SMALL, LARGE = (6, 12), (20, 30)
DT, SUBSTEPS = 0.5, 50


def rotate(c, theta):
    cos, sin = np.cos(theta)[..., None], np.sin(theta)[..., None]

    return np.concatenate([cos * c[..., :1] - sin * c[..., 1:], sin * c[..., :1] + cos * c[..., 1:]], -1)


def derivative(net, angles, m, q):
    """m [S, P], q = m * c [S, P, 2] -> dm/dt, dq/dt."""
    K, k = np.exp(net.e[:, 0]), np.exp(net.a[:, 0])
    c = q / np.maximum(m, 1e-9)[..., None]

    # [P, T]
    weight = net.Pre / net.Pre.sum(0, keepdims=True)
    c_in = np.einsum("spd,pt->std", c, weight)
    guard = 0.5 * (1 + (c_in * np.stack([np.cos(angles[:, 0]), np.sin(angles[:, 0])], -1)).sum(-1))
    logu = np.log(np.maximum(m / (K + m), 1e-300))

    # [S, T]
    v = k * np.exp(np.where(net.Pre > 0, logu[:, :, None] * net.Pre, 0.0).sum(1)) * guard
    emitted = rotate(c_in, angles[:, 1])
    dq = np.einsum("st,pt,std->spd", v, net.Pos, emitted) - (v @ net.Pre.T)[..., None] * c

    return v @ net.C.T, dq


def simulate(net, angles, m, c, n_steps):
    q, h = m[..., None] * c, DT / SUBSTEPS
    states = [(m, c)]
    for _ in range(n_steps):
        for _ in range(SUBSTEPS):
            k1 = derivative(net, angles, m, q)
            k2 = derivative(net, angles, m + h / 2 * k1[0], q + h / 2 * k1[1])
            k3 = derivative(net, angles, m + h / 2 * k2[0], q + h / 2 * k2[1])
            k4 = derivative(net, angles, m + h * k3[0], q + h * k3[1])
            m = np.maximum(m + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), 0.0)
            q = q + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])

        states.append((m, q / np.maximum(m, 1e-9)[..., None]))

    return states


def make_split(seed, n_nets, places, n_traj=8, n_steps=8):
    rng = np.random.default_rng(seed)
    groups = []
    for _ in range(n_nets):
        n_places = int(rng.integers(places[0], places[1] + 1))
        net = nets.random_net(rng, n_places, n_base=max(2, int(round(0.6 * n_places))), max_arity=2)

        # guard direction, rotation of the emitted colour
        angles = rng.uniform(-np.pi, np.pi, size=(net.n_trans, 2))
        m0 = rng.uniform(0.3, 4.0, size=(n_traj, n_places))
        phi = rng.uniform(-np.pi, np.pi, size=(n_traj, n_places))
        states = simulate(net, angles, m0, np.stack([np.cos(phi), np.sin(phi)], -1), n_steps)
        groups.append({"net": net, "angles": angles, "m": np.stack([s[0] for s in states], 1), "c": np.stack([s[1] for s in states], 1)})

    return groups


def collate(groups, device, step=None, rng=None, n=32):
    """Batch of (state, next state) pairs, step=None samples n random pairs per net, step=k takes pair k of every trajectory."""
    picked, rows = [], []
    for g in groups:
        S, L = g["m"].shape[0], g["m"].shape[1] - 1
        idx = (rng.integers(0, S, n), rng.integers(0, L, n)) if step is None else (np.arange(S), np.full(S, step))
        picked.append(datasets.Group(g["net"], g["m"][idx], y=g["m"][idx[0], idx[1] + 1]))
        rows.append(np.arange(len(idx[0])))
        g["_pick"] = idx

    b = batching.collate(picked, rows, device)
    flat = lambda key, shift: torch.as_tensor(np.concatenate([g[key][g["_pick"][0], g["_pick"][1] + shift].reshape(-1, *g[key].shape[3:]) for g in groups]),
                                              dtype=torch.float32, device=device)
    b.colour, b.next_colour, b.next_mass = flat("c", 0), flat("c", 1), flat("m", 1)
    b.angles = torch.as_tensor(np.concatenate([np.tile(g["angles"], (len(g["_pick"][0]), 1)) for g in groups]), dtype=torch.float32, device=device)

    return b










def loss_fn(pred, b):
    # the colour of an empty place is undefined
    weight = (b.next_mass > 0.05).float()[:, None]

    return F.mse_loss(pred[0], b.next_mass) + (weight * (pred[1] - b.next_colour) ** 2).sum() / weight.sum().clamp(min=1) / 2


@torch.no_grad()
def evaluate(model, groups, device, horizon=8):
    b = collate(groups, device, step=0)
    m, c = b.m, b.colour
    out = {}

    # places no transition produces into
    no_inflow = scatter_sum(torch.ones_like(b.pos_w), b.pos_p, b.n_places) == 0
    for h in range(1, horizon + 1):
        m, c = model(b, m, c)
        m = m.clamp(min=0)

        if h in (1, horizon):
            target = collate(groups, device, step=h - 1)
            weight = target.next_mass > 0.05
            out[f"mass_nrmse@{h}"] = float(((m - target.next_mass) ** 2).mean().sqrt() / ((target.next_mass - b.m) ** 2).mean().sqrt())
            out[f"colour_rmse@{h}"] = float((((c - target.next_colour) ** 2).sum(1)[weight]).mean().sqrt())

            if h == 1 and no_inflow.any():
                out["colour_change_without_inflow"] = float((c - b.colour)[no_inflow].abs().max())

    return out


def main(root="results", iters=3000, seeds=2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = {"train": make_split(1, 300, SMALL), "val": make_split(2, 40, SMALL), "test": make_split(3, 100, SMALL, n_traj=4),
              "test-large": make_split(4, 60, LARGE, n_traj=4)}
    out = {}
    for name in ("pgnn+", "npf-free", "npf-coloured"):
        for seed in range(seeds):
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            model = (ColouredPGNN() if name == "pgnn+" else ColouredNPF(mixing=name == "npf-coloured")).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=2e-3)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=2e-5)
            for _ in range(iters):
                nets = [splits["train"][i] for i in rng.choice(len(splits["train"]), 16, replace=False)]
                b = collate(nets, device, rng=rng)
                loss = loss_fn(model(b), b)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()

            model.eval()
            row = {k: evaluate(model, v, device) for k, v in splits.items() if k.startswith("test")}
            out.setdefault(name, []).append(row | {"params": sum(p.numel() for p in model.parameters())})
            print(f"{name:>12} seed {seed}: " + "  ".join(f"{k}: " + " ".join(f"{a}={x:.3g}" for a, x in v.items()) for k, v in row.items()), flush=True)

    Path(root).mkdir(parents=True, exist_ok=True)
    Path(root, "coloured.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
