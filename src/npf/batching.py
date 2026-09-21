"""Batching of nets as a disjoint union with one copy of the net per sample."""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Batch:
    # markings A and B, flattened over all copies [NP]
    m: torch.Tensor
    m_b: torch.Tensor

    # place attributes [NP, 1] and transition attributes [NT, 1]
    e: torch.Tensor
    a: torch.Tensor

    # elapsed time broadcast to places and to transitions
    dt_p: torch.Tensor
    dt_t: torch.Tensor
    pre_p: torch.Tensor
    pre_t: torch.Tensor
    pre_w: torch.Tensor
    pos_p: torch.Tensor
    pos_t: torch.Tensor
    pos_w: torch.Tensor
    edge_u: torch.Tensor
    edge_v: torch.Tensor
    edge_t: torch.Tensor
    edge_wu: torch.Tensor
    edge_wv: torch.Tensor
    n_places: int
    n_trans: int

    # dense incidence matrices per net, zero padded, for the state equation projection
    C: torch.Tensor
    C_pinv: torch.Tensor

    # position of every place and transition in the padded [G, S, Pmax] and [G, S, Tmax] layouts
    pad_p: torch.Tensor
    pad_t: torch.Tensor
    pad_shape: tuple

    # orthonormal bases of the P-invariants ker C^T, zero padded [G, Pmax, rmax]
    X: torch.Tensor = None


INDEX_FIELDS = ("pad_p", "pad_t", "pre_p", "pre_t", "pos_p", "pos_t", "edge_u", "edge_v", "edge_t")


def collate(groups, rows, device):
    """groups is a list of Group, rows selects the same number S of samples in every group."""
    S = len(rows[0])
    G, Pmax, Tmax = len(groups), max(g.net.n_places for g in groups), max(g.net.n_trans for g in groups)
    names = ("m", "m_b", "e", "a", "dt_p", "dt_t", "pre_p", "pre_t", "pre_w", "pos_p", "pos_t", "pos_w",
             "edge_u", "edge_v", "edge_t", "edge_wu", "edge_wv", "pad_p", "pad_t")
    cols = {k: [] for k in names}
    C, C_pinv = np.zeros((G, Pmax, Tmax)), np.zeros((G, Tmax, Pmax))
    X = np.zeros((G, Pmax, max(1, max(g.net.X.shape[1] for g in groups))))
    off_p = off_t = 0
    for gi, (g, r) in enumerate(zip(groups, rows)):
        net = g.net
        P, T = net.n_places, net.n_trans
        sp, st = off_p + P * np.arange(S)[:, None], off_t + T * np.arange(S)[:, None]
        dt = g.dt[r] if g.dt is not None else np.zeros(S)
        cols["m"].append(g.m[r].ravel())
        cols["m_b"].append(g.m_b[r].ravel() if g.m_b is not None else np.zeros(S * P))
        cols["e"].append(np.tile(net.e, (S, 1)))
        cols["a"].append(np.tile(net.a, (S, 1)))
        cols["dt_p"].append(np.repeat(dt, P))
        cols["dt_t"].append(np.repeat(dt, T))
        cols["pre_p"].append((net.pre_p[None] + sp).ravel())
        cols["pre_t"].append((net.pre_t[None] + st).ravel())
        cols["pos_p"].append((net.pos_p[None] + sp).ravel())
        cols["pos_t"].append((net.pos_t[None] + st).ravel())
        cols["pre_w"].append(np.tile(net.pre_w, S))
        cols["pos_w"].append(np.tile(net.pos_w, S))
        eu, ev, et, wu, wv = net.edges
        cols["edge_u"].append((eu[None] + sp).ravel())
        cols["edge_v"].append((ev[None] + sp).ravel())
        cols["edge_t"].append((et[None] + st).ravel())
        cols["edge_wu"].append(np.tile(wu, S))
        cols["edge_wv"].append(np.tile(wv, S))
        cols["pad_p"].append(((gi * S + np.arange(S))[:, None] * Pmax + np.arange(P)[None]).ravel())
        cols["pad_t"].append(((gi * S + np.arange(S))[:, None] * Tmax + np.arange(T)[None]).ravel())
        C[gi, :P, :T], C_pinv[gi, :T, :P] = net.C, net.C_pinv
        X[gi, :P, :net.X.shape[1]] = net.X
        off_p += S * P
        off_t += S * T

    tensors = {
        k: torch.as_tensor(np.concatenate(v), dtype=torch.long if k in INDEX_FIELDS else torch.float32, device=device)
        for k, v in cols.items()
    }

    return Batch(
        **tensors, n_places=off_p, n_trans=off_t,
        C=torch.as_tensor(C, dtype=torch.float32, device=device),
        C_pinv=torch.as_tensor(C_pinv, dtype=torch.float32, device=device),
        pad_shape=(G, S, Pmax, Tmax), X=torch.as_tensor(X, dtype=torch.float64, device=device),
    )


def flat_targets(groups, rows, key, device):
    values = np.concatenate([getattr(g, key)[r].ravel() for g, r in zip(groups, rows)])

    return torch.as_tensor(values, dtype=torch.float32, device=device)
