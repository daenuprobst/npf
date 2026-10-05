"""Inputs and targets of the supervised baselines, which see a culture only through its measured rates and knockouts.

A culture is the vector of its measured exchange rates and growth relative to its carbon uptake in glucose
equivalents s, that uptake itself, which exchanges were measured, and which core reactions its knockouts block.
Learners predict net fluxes per unit of s in reaction space, so that every study, whatever it measured, is scored
through its own measurement rows M v with M the signed sum of each measured flux. The same splits as flux.py are
used and hyperparameters are chosen by an inner five fold cross validation on the training cultures only.
"""

import numpy as np

from .. import data
from . import common


def reaction_ids():
    return [r["id"] for r in common.model()["reactions"]]


def rows(inst, index):
    """M with M v the measurements of a culture for net fluxes v in the order of index."""
    M = np.zeros((len(inst.culture.measurements), len(index)))
    for k, (_, coeff, _) in enumerate(inst.culture.measurements):
        for rid, c in coeff.items():
            M[k, index[rid]] = c

    return M


class Features:
    """Feature map fitted on the training cultures, standardised, constant columns dropped, a leading one."""

    def __init__(self, train):
        self.exchanges = sorted({r for i in train for r in i.culture.pinned if r != data.BIOMASS})
        self.blocked = sorted({r for i in train for r in i.culture.knockouts})
        raw = np.array([self.raw(i) for i in train])
        self.mu, self.sd = raw.mean(0), raw.std(0)
        self.keep = self.sd > 1e-9

    def raw(self, inst):
        s = common.substrate_scale(inst)
        pinned = inst.culture.pinned
        rates = [pinned.get(r, 0.0) / s for r in self.exchanges]
        measured = [float(r in pinned) for r in self.exchanges]
        blocked = [float(r in inst.culture.knockouts) for r in self.blocked]

        return np.array([s, inst.growth / s] + rates + measured + blocked)

    def __call__(self, inst):
        z = (self.raw(inst) - self.mu)[self.keep] / self.sd[self.keep]

        return np.concatenate([[1.0], z])


def relative_error(inst, v, index):
    pred = rows(inst, index) @ v

    return float(np.linalg.norm(pred - inst.y) / np.linalg.norm(inst.y))


def inner_folds(train, seed, k=5):
    order = np.random.default_rng(seed).permutation(len(train))

    return [
        ([train[j] for j in order if j not in set(f)], [train[j] for j in f]) for f in np.array_split(order, k)
    ]


def choose(train, grid, fit, seed):
    """The grid point with the lowest mean relative error over the inner folds, fit(cultures, point) -> predict."""
    scores = []
    for point in grid:
        errors = []

        for inner_train, inner_val in inner_folds(train, seed):
            predict = fit(inner_train, point)
            errors += [relative_error(i, predict(i), predict.index) for i in inner_val]

        scores.append(np.mean(errors))

    return grid[int(np.argmin(scores))], scores


def projected(inst, v, index, measured):
    """The flux balance solution closest to the prediction on the reactions seen in training measurements."""
    ids, N = common.reactions(inst)
    keep = [k for k, r in enumerate(ids) if r in measured]
    target = np.array([v[index[ids[k]]] for k in keep])
    sigma = common.maintained(inst, lambda lo: common.project(inst, N[keep], target, lo))

    return None if sigma is None else common.cancel(inst, sigma)


def output(inst, v, index, measured, project):
    """Firing counts of a prediction, signed and unprojected by default, else projected on the flux balance."""
    if project:
        return projected(inst, v, index, measured)

    return common.signed_sigma(inst, {r: v[k] for r, k in index.items()})
