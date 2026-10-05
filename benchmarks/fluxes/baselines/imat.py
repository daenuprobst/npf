"""iMAT (Shlomi et al. 2008 Nat Biotechnol 26 1003) on the culture net, Ishii 2007 cultures only.

As in the sim_intra test of Machado and Herrgard 2014, growth and measured exchanges are fixed, genes above the upper
quartile of the culture count as high and below the lower quartile as low, reaction levels take the minimum over and
and the maximum over or, and epsilon is 1. iMAT maximises the number of high reactions carrying at least epsilon in
either direction plus low reactions carrying none, with the flux bounds of the core model, 1000, as big M. The least
total flux among the optima is taken.

    uv run python -m benchmarks.fluxes.baselines.imat --split all --omics transcripts
"""

import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from . import common, omics

LOW, HIGH, EPSILON, BIG = 0.25, 0.75, 1.0, 1000.0


def imat(inst, levels):
    low, high = omics.quantile(levels, LOW), omics.quantile(levels, HIGH)
    discrete = {g: 1.0 if x > high else -1.0 if x < low else 0.0 for g, x in levels.items()}
    level = omics.reaction_levels(discrete, min, max)
    ids, N = common.reactions(inst)
    level = {r: level.get(r, math.nan) for r in ids}
    on = [k for k, r in enumerate(ids) if not math.isnan(level[r]) and level[r] > 0]
    off = [k for k, r in enumerate(ids) if not math.isnan(level[r]) and level[r] < 0]
    n_t, n_on, n_off = inst.net.n_trans, len(on), len(off)
    v_max, v_min = BIG * (N > 0).any(1), -BIG * (N < 0).any(1)
    C, b = inst.net.C, common.rhs(inst)

    # variables sigma, y+ and y- for the high reactions, y0 for the low ones
    n = n_t + 2 * n_on + n_off
    rows = [LinearConstraint(np.hstack([C, np.zeros((len(b), n - n_t))]), b, b)]
    for j, k in enumerate(on):
        # y+ = 1 forces v >= epsilon, y- = 1 forces v <= -epsilon
        up = np.zeros(n)
        up[:n_t], up[n_t + j] = N[k], v_min[k] - EPSILON
        rows.append(LinearConstraint(up, v_min[k], np.inf))
        down = np.zeros(n)
        down[:n_t], down[n_t + n_on + j] = N[k], v_max[k] + EPSILON
        rows.append(LinearConstraint(down, -np.inf, v_max[k]))

    for j, k in enumerate(off):
        # y0 = 1 forces v = 0
        row = np.zeros(n)
        row[:n_t], row[n_t + 2 * n_on + j] = N[k], v_min[k]
        rows.append(LinearConstraint(row, v_min[k], np.inf))
        row = np.zeros(n)
        row[:n_t], row[n_t + 2 * n_on + j] = N[k], v_max[k]
        rows.append(LinearConstraint(row, -np.inf, v_max[k]))

    integrality = np.concatenate([np.zeros(n_t), np.ones(n - n_t)])
    reward = np.concatenate([np.zeros(n_t), np.ones(n - n_t)])

    def solve(lo):
        bounds = Bounds(np.concatenate([lo, np.zeros(n - n_t)]), np.concatenate([np.full(n_t, BIG), np.ones(n - n_t)]))
        res = milp(-reward, constraints=rows, integrality=integrality, bounds=bounds, options={"time_limit": 120})
        if res.x is None:
            return None

        # least total firing among the solutions that agree with as many levels
        keep = LinearConstraint(reward[None], np.round(reward @ res.x) - 0.5, np.inf)
        total = np.concatenate([np.ones(n_t), np.zeros(n - n_t)])
        res2 = milp(total, constraints=rows + [keep], integrality=integrality, bounds=bounds,
                    options={"time_limit": 120})

        return (res.x if res2.x is None else res2.x)[:n_t]

    sigma = common.maintained(inst, solve)

    return None if sigma is None else common.cancel(inst, sigma)


def main():
    args = omics.parser(__doc__).parse_args()
    omics.evaluate("imat", args, imat, note="iMAT with growth and exchanges fixed, quartile thresholds, epsilon 1")


if __name__ == "__main__":
    main()
