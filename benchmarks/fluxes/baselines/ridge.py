"""Ridge regression from the measured rates and knockouts to the net fluxes, fitted through each measurement matrix.

v = s B x for a culture with features x and carbon uptake s (learn.py). B minimises the squared relative error
sum over cultures of ||M (B x) - y / s||^2 / ||y / s||^2 plus lambda ||B||^2 on all but the intercept, a linear least
squares problem in B solved in closed form. Every study is fitted on its own measured fluxes, lumped or not, and B
has one row per reaction that some training measurement touches. lambda comes from an inner five fold cross
validation on the training cultures. The prediction ignores the net, its balance residual is that of v itself.
With --project it is projected on the flux balance of the culture afterwards (needs cvxpy).

    uv run python -m benchmarks.fluxes.baselines.ridge --split all
    uv run --with cvxpy python -m benchmarks.fluxes.baselines.ridge --split all --project
"""

import numpy as np

from . import common, learn

GRID = [10.0 ** k for k in range(-4, 4)]


class Ridge:
    def __init__(self, train, lam):
        self.feats = learn.Features(train)
        self.index = {r: k for k, r in enumerate(learn.reaction_ids())}
        self.measured = sorted({r for i in train for _, coeff, _ in i.culture.measurements for r in coeff})
        self.cols = [self.index[r] for r in self.measured]
        d, n = len(self.feats(train[0])), len(self.cols)
        G, h = np.zeros((d * n, d * n)), np.zeros(d * n)

        for inst in train:
            x = self.feats(inst)
            M = learn.rows(inst, self.index)[:, self.cols]
            y = inst.y / common.substrate_scale(inst)

            # M B x = (x^T kron M) vec(B), each culture weighted to its relative error
            w = 1.0 / (y @ y)
            G += w * np.kron(np.outer(x, x), M.T @ M)
            h += w * np.kron(x, M.T @ y)

        # the intercept block is barely shrunk, a reaction seen only in a lumped sum still gets a unique value
        penalty = np.full(d * n, lam)
        penalty[:n] = 1e-8
        beta = np.linalg.solve(G + np.diag(penalty), h)
        self.B = beta.reshape(d, n).T

    def __call__(self, inst):
        v = np.zeros(len(self.index))
        v[self.cols] = common.substrate_scale(inst) * (self.B @ self.feats(inst))

        return v


def main():
    ap = common.parser(__doc__)
    ap.add_argument("--project", action="store_true")
    args = ap.parse_args()

    def fit(train):
        lam, scores = learn.choose(train, GRID, Ridge, args.seed)
        model = Ridge(train, lam)

        def predict(inst):
            return learn.output(inst, model(inst), model.index, set(model.measured), args.project)

        predict.info = {"lambda": lam, "inner_rel_error": scores, "params": int(model.B.size)}

        return predict

    name = "ridge+proj" if args.project else "ridge"
    common.evaluate(name, args, fit, note=f"lambda by inner 5-fold CV on the training cultures from {GRID}")


if __name__ == "__main__":
    main()
