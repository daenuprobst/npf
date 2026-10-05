"""The fibre {sigma >= 0 : C sigma = m_B - m_A} of every culture under the rule of the environment, open by default and
closed with METABOLISM_CLOSE_CARBON=1, facts that App D.4 quotes and no method needs.

Per culture, the dimension of the kernel of C, the unmeasured carbon exchanges that may secrete, the smallest relative
error that any point of the fibre reaches against the 13C fluxes, a diagnostic that reads the test fluxes and trains
nothing, and the share of the substrate carbon that the state equation alone, the I-projection of uniform rates,
secretes through those exchanges.

    uv run python -m benchmarks.fluxes.fibre                               # results/metabolism/fibre.json
    METABOLISM_CLOSE_CARBON=1 uv run python -m benchmarks.fluxes.fibre     # results/metabolism_closed/fibre.json
"""

import json
from pathlib import Path

import numpy as np
import torch

from npf import models

from . import data, flux
from .baselines import common


def facts(inst, sigma, reactions):
    C, n = inst.net.C, inst.net.n_trans

    # carbon atoms per firing of every unmeasured exchange that secretes carbon, carbon dioxide excepted
    carbon = np.zeros(n)
    for t, (rid, sign) in enumerate(zip(inst.reaction, inst.sign)):
        if rid.startswith("EX_") and sign == 1.0 and rid not in inst.culture.pinned and rid not in data.FREE_CARBON:
            (met,) = reactions[rid]["metabolites"]
            carbon[t] = common.carbons(met)

    # the point of the fibre closest to the measured fluxes, the least error any method on the net can reach
    closest = common.project(inst, inst.W, inst.y, np.zeros(n), alpha=1e-6)
    uptake = 6.0 * common.substrate_scale(inst)

    return {
        "culture": common.key(inst),
        "kernel": int(n - np.linalg.matrix_rank(C)),
        "carbon_sinks": int((carbon > 0).sum()),
        "floor": None if closest is None else float(np.linalg.norm(inst.W @ closest - inst.y) / np.linalg.norm(inst.y)),
        "state_equation_sink_carbon": float(carbon @ sigma / uptake),
    }


def main():
    torch.set_num_threads(2)
    insts, _ = data.instances()
    ids, _, boundary = data.vocabulary()
    model = models.build("se-only", "transitions", n_ids=len(ids), n_env=len(boundary) + 1, kl_options=flux.KL).eval()

    with torch.no_grad():
        sigmas = [s.double().numpy() for s in flux.run(model, insts, "cpu")]

    reactions = {r["id"]: r for r in common.model()["reactions"]}
    rows = [facts(i, s, reactions) for i, s in zip(insts, sigmas)]
    out = Path("results/metabolism_closed" if data.CLOSE_CARBON else "results/metabolism") / "fibre.json"
    out.write_text(json.dumps({"closed": data.CLOSE_CARBON, "rows": rows}, indent=1))
    floor = [r["floor"] for r in rows if r["floor"] is not None]
    print(f"{'closed' if data.CLOSE_CARBON else 'open'}: {len(rows)} cultures, kernel median "
          f"{np.median([r['kernel'] for r in rows]):.0f}, floor mean {np.mean(floor):.3f} median {np.median(floor):.3f}, "
          f"state equation alone secretes {100 * np.mean([r['state_equation_sink_carbon'] for r in rows]):.1f} % of the "
          f"substrate carbon through unmeasured exchanges")


if __name__ == "__main__":
    main()
