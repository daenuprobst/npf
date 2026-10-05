"""LocalMapper, Chen et al. 2024, the pretrained mapper of the localmapper package, on unmapped reactions.

It needs an environment of its own with DGL and an older torch, so this script imports nothing of npf. The model is the
default of the package, version 202403. The flag confident, true when the template of the predicted map is one the
chemists accepted in training, is kept as the confidence. The output and its resumption are those of resumable.py.

The DGL wheels on PyPI have no CUDA and take about 2 s per reaction on the CPU, the edge network of the message
passing makes a 256 by 256 matrix for every bond. The CUDA wheel of DGL comes from the wheel index of DGL.

    uv run --no-project --python 3.11 --find-links https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html \\
        --with localmapper==0.1.5 --with "dgl==2.4.0+cu124" --with torch==2.4.0 --with torchdata==0.8.0 \\
        --with pyyaml --with pydantic --with rdkit --with "numpy<2" --with "setuptools<81" \\
        python benchmarks/chemistry/baselines/local_mapper.py \\
        data/enzymemap_unmapped.txt results/chem/localmapper_enzymemap.json cuda 4 8

    uv run --no-project --python 3.11 --with localmapper==0.1.5 --with dgl==2.1.0 --with torch==2.1.2 \\
        --with torchdata==0.7.1 --with pyyaml --with pydantic --with rdkit --with "numpy<2" --with "setuptools<81" \\
        python benchmarks/chemistry/baselines/local_mapper.py \\
        data/enzymemap_unmapped.txt results/chem/localmapper_enzymemap.json cpu 4 32    # PyPI only, CPU

The arguments after the output are the device, the number of CPU threads of torch and the reactions per batch, whose
memory grows with the bonds in the batch.
"""

import os
import sys
import time

from resumable import Run


def main(
    unmapped="data/enzymemap_unmapped.txt",
    out="results/chem/localmapper_enzymemap.json",
    device="cpu",
    threads=4,
    batch=8,
):
    run = Run(unmapped, out)
    todo = run.todo()
    if todo:
        # imported here so the script imports without the LocalMapper environment
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        import torch
        from localmapper import localmapper

        torch.set_num_threads(int(threads))
        mapper = localmapper(device)

    for k in range(0, len(todo), int(batch)):
        part, start = todo[k : k + int(batch)], time.time()

        # one reaction that fails takes its batch down, so the batch is repeated one reaction at a time
        try:
            results = mapper.get_atom_map(
                [smiles for _, smiles in part], return_dict=True
            )
            each = [(result, (time.time() - start) / len(part)) for result in results]
        except Exception:
            each = []
            for _, smiles in part:
                start = time.time()

                try:
                    result = mapper.get_atom_map([smiles], return_dict=True)[0]
                except Exception as e:
                    result = repr(e)

                each.append((result, time.time() - start))

        for (rid, _), (result, seconds) in zip(part, each):
            if isinstance(result, dict):
                run.add(
                    rid,
                    result["mapped_rxn"],
                    seconds,
                    confidence=result["confident"],
                    error="no map returned",
                )
            else:
                run.add(rid, None, seconds, error=result)

    run.finish()


if __name__ == "__main__":
    main(*sys.argv[1:])
