"""Regulatory on/off minimisation (Shlomi, Berkman, Ruppin 2005 PNAS 102 7695) with delta 0.03 and epsilon 0.001.

Reference as in moma.py, the pFBA flux of the same culture with the knockouts undone. ROOM minimises the number of
reactions whose net flux leaves the band w +- (delta |w| + epsilon), a mixed integer program with the flux bounds
of the core model, 1000 in either direction, as big M. The least total flux among the optima is taken.

    uv run python -m benchmarks.fluxes.baselines.room --split all
"""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .. import flux
from . import common
from .moma import reference

DELTA, EPSILON, BIG = 0.03, 0.001, 1000.0


def room(inst):
    if not inst.culture.knockouts:
        return common.cancel(inst, flux.pfba(inst))

    N, w = reference(inst)
    w = np.asarray(w)
    n_r, n_t = N.shape
    C, b = inst.net.C, common.rhs(inst)
    hi_v = BIG * (N > 0).any(1)
    lo_v = -BIG * (N < 0).any(1)
    w_up = w + DELTA * np.abs(w) + EPSILON
    w_lo = w - DELTA * np.abs(w) - EPSILON

    # y_r = 1 lets reaction r leave its band, v - y (v_max - w_up) <= w_up and v - y (v_min - w_lo) >= w_lo
    rows = [
        LinearConstraint(np.hstack([C, np.zeros((len(b), n_r))]), b, b),
        LinearConstraint(np.hstack([N, -np.diag(hi_v - w_up)]), -np.inf, w_up),
        LinearConstraint(np.hstack([N, -np.diag(lo_v - w_lo)]), w_lo, np.inf),
    ]
    integrality = np.concatenate([np.zeros(n_t), np.ones(n_r)])

    def solve(lo):
        bounds = Bounds(np.concatenate([lo, np.zeros(n_r)]), np.concatenate([np.full(n_t, BIG), np.ones(n_r)]))
        c = np.concatenate([np.zeros(n_t), np.ones(n_r)])
        res = milp(c, constraints=rows, integrality=integrality, bounds=bounds, options={"time_limit": 120})
        if res.x is None:
            return None

        # least total firing among the solutions with as few changed reactions
        cap = LinearConstraint(c[None], -np.inf, np.round(c @ res.x) + 0.5)
        total = np.concatenate([np.ones(n_t), np.zeros(n_r)])
        res2 = milp(total, constraints=rows + [cap], integrality=integrality, bounds=bounds,
                    options={"time_limit": 120})

        return (res.x if res2.x is None else res2.x)[:n_t]

    sigma = common.maintained(inst, solve)

    return None if sigma is None else common.cancel(inst, sigma)


def main():
    args = common.parser(__doc__).parse_args()
    note = "ROOM to the pFBA wild type at the same measured exchanges and growth, delta 0.03, epsilon 0.001"
    common.evaluate("room", args, lambda train: room, note=note, training_free=True)


if __name__ == "__main__":
    main()
