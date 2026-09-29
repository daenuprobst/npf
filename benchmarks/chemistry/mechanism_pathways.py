"""Top-k pathway accuracy on the FlowER mechanism benchmark, from the per-step ranks that `mechanism evaluate` saves.

The metric is the one of FlowER's sequence_evaluation.py (tag 2.0.0). The test steps are grouped by their sequence id,
leaving out the PMechDB and RMechDB steps (ids PM, RS, RC, PC), which have no pathway. Each pathway is a graph over the
SMILES of its steps without atom maps, from its one start to the products that end in a self-loop. A route is a simple
path from the start to such a product, so the terminal self-loop itself is not on it, and the rank of a route is the
worst rank of its steps. A pathway is correct at k when some route has rank at most k. A pathway without a single start
or without a terminal product counts as wrong. When a pathway holds the same step twice, the first one in file order
that is found sets its rank, as in FlowER.

FlowER also credits a prediction that equals another recorded product of the same reactant. Only the rank of each
step's own product is saved here, so that credit is not given, which can only lower the accuracy of NPF.

    uv run python -m benchmarks.chemistry.mechanism_pathways                  # results/mechanism/pathways.json
    uv run python -m benchmarks.chemistry.mechanism_pathways --runs npf-electron --seeds 0
"""

import argparse
import json
from collections import defaultdict
from multiprocessing import Pool

import networkx as nx
import numpy as np
from rdkit import Chem, RDLogger

from . import mechanism as M

RDLogger.DisableLog("rdApp.*")
CURATED = ("PM", "RS", "RC", "PC")


def unmapped(line):
    params = Chem.SmilesParserParams()
    params.removeHs = False
    reaction, sequence = line.strip().rsplit("|", 1)
    sides = []
    for smiles in reaction.split(">>"):
        mol = Chem.MolFromSmiles(smiles, params)
        for atom in mol.GetAtoms():
            atom.ClearProp("molAtomMapNumber")
        once = Chem.MolToSmiles(mol, isomericSmiles=False)
        sides.append(Chem.MolToSmiles(Chem.MolFromSmiles(once, params), isomericSmiles=False))

    return sides[0], sides[1], sequence


def pathway_rank(steps):
    graph, rank = nx.DiGraph(), {}
    for reactant, product, r in steps:
        graph.add_edge(reactant, product)
        if rank.get((reactant, product), np.inf) == np.inf:
            rank[(reactant, product)] = r

    start = [n for n, d in graph.in_degree() if d == 0]
    terminal = list(nx.nodes_with_selfloops(graph))
    if len(start) != 1 or not terminal:
        return np.inf

    best = np.inf
    for t in terminal:
        for path in nx.all_simple_paths(graph, start[0], t):
            best = min(best, max(rank[e] for e in nx.utils.pairwise(path)))

    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="npf,npf-electron", help="result names, npf is the arrow net")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--processes", type=int, default=16)
    args = ap.parse_args()

    lines = (M.SOURCE / "test.txt").read_text().splitlines()
    with Pool(args.processes) as pool:
        steps = pool.map(unmapped, lines, chunksize=500)

    out = {}
    for name in args.runs.split(","):
        for seed in map(int, args.seeds.split(",")):
            saved = np.load(M.RESULTS / f"{name}-{seed}-ranks.npz")
            beam = json.loads((M.RESULTS / f"{name}-{seed}.json").read_text())["beam"]

            # rank 0 means the recorded product was not among the predictions
            ranks = saved["rank"].astype(float)
            ranks[ranks == 0] = np.inf
            groups = defaultdict(list)
            for k, r in zip(saved["step"], ranks):
                reactant, product, sequence = steps[k]
                if sequence not in CURATED:
                    groups[sequence].append((reactant, product, r))

            with Pool(args.processes) as pool:
                worst = np.asarray(pool.map(pathway_rank, list(groups.values()), chunksize=200))

            out[f"{name}-{seed}"] = {"pathways": len(worst), "beam": beam} | {
                f"pathway_top{k}": float(np.mean(worst <= k)) for k in (1, 2, 3, 5, 10) if k <= beam
            }
            print(name, seed, out[f"{name}-{seed}"], flush=True)

    M.RESULTS.mkdir(parents=True, exist_ok=True)
    (M.RESULTS / "pathways.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
