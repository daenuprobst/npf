"""E-Flux2 (Kim, O'Brien, Kim, Lun 2016 PLoS One 11 e0157101) on the core model, Ishii 2007 cultures only.

Reaction levels take the minimum over and and the sum over or, scaled to a maximum of 1, and bound the flux in each
allowed direction, reactions without a level keep the bounds of the template (omics.template). E-Flux2 maximises
growth and then takes the flux of least Euclidean norm at that growth. Its fluxes are in arbitrary units and are
scaled to the measured glucose uptake, the only measured rate it uses besides the knockouts. The measured growth
and secretions are not imposed, so its balance residual is its miss on them. Needs cvxpy.

    uv run --with cvxpy python -m benchmarks.fluxes.baselines.eflux2 --split all --omics transcripts
"""

import math

import numpy as np

from .. import data
from . import common, omics


def bounds(inst, levels):
    ids, S, lb, ub = omics.template(inst)
    level = omics.reaction_levels(levels, min, sum)
    top = max(x for x in level.values() if not math.isnan(x))
    for j, r in enumerate(ids):
        if not math.isnan(level[r]) and r not in inst.culture.knockouts:
            lb[j] = -level[r] / top if lb[j] < 0 else 0.0
            ub[j] = level[r] / top if ub[j] > 0 else 0.0

    return ids, S, lb, ub


def eflux2(inst, levels):
    import cvxpy as cp

    ids, S, lb, ub = bounds(inst, levels)
    growth = ids.index(data.BIOMASS)
    c = np.zeros(len(ids))
    c[growth] = -1.0
    z = common.lp(c, lb, S, np.zeros(len(S)), hi=ub)
    if z is None or z[growth] <= 1e-9:
        return None

    # the flux of least norm at maximal growth
    v = cp.Variable(len(ids))
    finite = np.isfinite(lb)
    constraints = [S @ v == 0, v <= ub, v[finite] >= lb[finite], v[growth] == z[growth]]
    cp.Problem(cp.Minimize(cp.sum_squares(v)), constraints).solve(solver="CLARABEL")

    if v.value is None:
        return None

    return omics.scaled(inst, ids, v.value)


def main():
    args = omics.parser(__doc__).parse_args()
    omics.evaluate("eflux2", args, eflux2, note="E-Flux2, DC template, scaled to measured glucose uptake")


if __name__ == "__main__":
    main()
