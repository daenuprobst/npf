"""GraphormerMapper, Nugmanov et al. 2022, the pretrained mapper of chytorch-rxnmap, on unmapped reactions.

It needs an environment of its own with chython and chytorch, so this script imports nothing of npf. Each reaction is
parsed by chython as it is, with no standardisation, mapped by reset_mapping, the attention of the published model
followed by the group and mechanism fixes of chython, and written back as mapped SMILES. The mean attention of the
chosen atom pairs is kept as the confidence. The output and its resumption are those of resumable.py.

    uv run --no-project --python 3.11 --with chytorch-rxnmap==1.4 --with chytorch==1.65 --with chython==1.78 \\
        --with "setuptools<81" python benchmarks/chemistry/baselines/graphormer_mapper.py \\
        data/enzymemap_unmapped.txt results/chem/graphormer_enzymemap.json cpu 4

The device is cpu or cuda, cuda runs in half precision under autocast as chython does, the last argument is the
number of CPU threads of torch.
"""

import os
import sys
import time

from resumable import Run


def main(
    unmapped="data/enzymemap_unmapped.txt",
    out="results/chem/graphormer_enzymemap.json",
    device="cpu",
    threads=4,
):
    run = Run(unmapped, out)
    todo = run.todo()
    if todo:
        # imported here so the script imports without the chython environment
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        import chython
        import torch
        from chython import smiles

        torch.set_num_threads(int(threads))
        chython.torch_device = device

    for rid, reaction in todo:
        start = time.time()

        # reset_mapping returns False instead of a score when chython refuses the reaction
        try:
            r = smiles(reaction)
            score = r.reset_mapping(return_score=True)
            if isinstance(score, bool):
                raise ValueError("chython does not map hypervalent atoms")

            run.add(rid, format(r, "m"), time.time() - start, confidence=score)
        except Exception as e:
            run.add(rid, None, time.time() - start, error=repr(e))

    run.finish()


if __name__ == "__main__":
    main(*sys.argv[1:])
