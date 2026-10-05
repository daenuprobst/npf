"""Linear MOMA, the L1 distance to the wild type (Becker et al. 2007 Nat Protoc 2 727, COBRA toolbox linearMOMA).

Reference as in moma.py, the pFBA flux of the same culture with the knockouts undone. The L1 optimum is not unique,
the least total flux among the optima is taken.

    uv run python -m benchmarks.fluxes.baselines.lmoma --split all
"""

import numpy as np

from .. import flux
from . import common
from .moma import reference


def lmoma(inst):
    if not inst.culture.knockouts:
        return common.cancel(inst, flux.pfba(inst))

    N, w = reference(inst)
    w = np.asarray(w)
    n_t, n_r = N.shape[1], N.shape[0]
    C, b = inst.net.C, common.rhs(inst)

    # variables sigma and d with |N sigma - w| <= d, minimise sum d
    A_eq = np.hstack([C, np.zeros((len(b), n_r))])
    A_ub = np.block([[N, -np.eye(n_r)], [-N, -np.eye(n_r)]])
    b_ub = np.concatenate([w, -w])

    def solve(lo):
        lo = np.concatenate([lo, np.zeros(n_r)])
        c = np.concatenate([np.zeros(n_t), np.ones(n_r)])
        x = common.lp(c, lo, A_eq, b, A_ub, b_ub)
        if x is None:
            return None

        # least total firing among the L1 optima
        keep = np.concatenate([np.zeros(n_t), np.ones(n_r)])[None]
        x2 = common.least_total(lo, A_eq, b, np.vstack([A_ub, keep]), np.append(b_ub, common.tie(c @ x)), n=n_t)

        return (x if x2 is None else x2)[:n_t]

    sigma = common.maintained(inst, solve)

    return None if sigma is None else common.cancel(inst, sigma)


def main():
    args = common.parser(__doc__).parse_args()
    note = "linear MOMA to the pFBA wild type at the same measured exchanges and growth, ties by least total flux"
    common.evaluate("lmoma", args, lambda train: lmoma, note=note, training_free=True)


if __name__ == "__main__":
    main()
