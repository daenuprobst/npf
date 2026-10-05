"""SPOT (Kim, O'Brien, Kim, Lun 2016 PLoS One 11 e0157101) on the core model, Ishii 2007 cultures only.

Every allowed direction of a reaction of the template (omics.template) is a non-negative variable. SPOT maximises
the inner product of these fluxes with the reaction levels, minimum over and and sum over or, directions without a
level weigh 0, subject to the steady state and a Euclidean norm of at most 1, the uncentred Pearson correlation
between flux and expression. It uses no growth objective. Its fluxes are scaled to the measured glucose uptake, the
only measured rate it uses besides the knockouts. Needs cvxpy.

    uv run --with cvxpy python -m benchmarks.fluxes.baselines.spot --split all --omics transcripts
"""

import math

import numpy as np

from . import omics


def spot(inst, levels):
    import cvxpy as cp

    ids, S, lb, ub = omics.template(inst)
    level = omics.reaction_levels(levels, min, sum)
    top = max(x for x in level.values() if not math.isnan(x))

    # one column per allowed direction, the backward column carries -S
    cols = [(j, 1.0) for j in range(len(ids)) if ub[j] > 0] + [(j, -1.0) for j in range(len(ids)) if lb[j] < 0]
    D = np.zeros((len(ids), len(cols)))
    for k, (j, s) in enumerate(cols):
        D[j, k] = s

    g = np.array([0.0 if math.isnan(level[ids[j]]) else level[ids[j]] / top for j, _ in cols])
    x = cp.Variable(len(cols))
    problem = cp.Problem(cp.Maximize(g @ x), [S @ D @ x == 0, x >= 0, cp.norm(x, 2) <= 1])
    problem.solve(solver="CLARABEL")

    if x.value is None:
        return None

    return omics.scaled(inst, ids, D @ x.value)


def main():
    args = omics.parser(__doc__).parse_args()
    omics.evaluate("spot", args, spot, note="SPOT, DC template, scaled to measured glucose uptake")


if __name__ == "__main__":
    main()
