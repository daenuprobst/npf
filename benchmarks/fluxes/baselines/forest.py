"""Random forest from the measured rates and knockouts to the net fluxes of all core reactions.

A forest needs the same outputs for every culture, while the studies measure different, partly lumped fluxes. The
target of a training culture is therefore its measurements carried to reaction space, the flux balance solution
closest to them, argmin ||W sigma - y||^2 with the least total flux among ties, as in 13C flux fitting on the core.
Only training measurements enter. The forest predicts v / s from the features of learn.py, leaf size and feature
fraction come from an inner five fold cross validation on the training cultures. Without --project the prediction
ignores the net, with it the prediction is projected on the flux balance of the culture. Needs cvxpy and
scikit-learn.

    uv run --with cvxpy --with scikit-learn python -m benchmarks.fluxes.baselines.forest --split all
    uv run --with cvxpy --with scikit-learn python -m benchmarks.fluxes.baselines.forest --split all --project
"""

import numpy as np

from . import common, learn

GRID = [(leaf, share) for leaf in (1, 3, 10) for share in (1.0, 0.33)]
TREES = 300


class Targets:
    """Net fluxes per unit of carbon uptake fitted to the measurements of a culture, computed once per culture."""

    def __init__(self):
        self.cache, self._index = {}, None

    @property
    def index(self):
        # read on first use, so the module imports without the data
        if self._index is None:
            self._index = {r: k for k, r in enumerate(learn.reaction_ids())}

        return self._index

    def __call__(self, inst):
        if common.key(inst) not in self.cache:
            sigma = common.maintained(inst, lambda lo: common.project(inst, inst.W, inst.y, lo))
            v = np.zeros(len(self.index))

            # the drain of the maintenance place is no reaction of the model, it follows from ATPM
            for r, f in common.net_flux(inst, sigma).items():
                if r in self.index:
                    v[self.index[r]] = f

            self.cache[common.key(inst)] = v / common.substrate_scale(inst)

        return self.cache[common.key(inst)]


TARGETS = Targets()


class Forest:
    def __init__(self, train, point, seed=0):
        leaf, share = point
        self.feats = learn.Features(train)
        self.index = TARGETS.index
        self.measured = {r for i in train for _, coeff, _ in i.culture.measurements for r in coeff}

        # imported here so the module imports without scikit-learn
        from sklearn.ensemble import RandomForestRegressor

        self.model = RandomForestRegressor(
            TREES, min_samples_leaf=leaf, max_features=share, random_state=seed, n_jobs=1
        ).fit(np.array([self.feats(i) for i in train]), np.array([TARGETS(i) for i in train]))

    def __call__(self, inst):
        return common.substrate_scale(inst) * self.model.predict(self.feats(inst)[None])[0]


def main():
    ap = common.parser(__doc__)
    ap.add_argument("--project", action="store_true")
    args = ap.parse_args()

    def fit(train):
        point, scores = learn.choose(train, GRID, lambda t, p: Forest(t, p, args.seed), args.seed)
        model = Forest(train, point, args.seed)

        def predict(inst):
            return learn.output(inst, model(inst), model.index, model.measured, args.project)

        nodes = sum(t.tree_.node_count for t in model.model.estimators_)
        predict.info = {"min_samples_leaf": point[0], "max_features": point[1], "inner_rel_error": scores,
                        "params": int(nodes)}

        return predict

    name = "forest+proj" if args.project else "forest"
    common.evaluate(name, args, fit, note=f"{TREES} trees, leaf size and feature share by inner 5-fold CV from {GRID}")


if __name__ == "__main__":
    main()
