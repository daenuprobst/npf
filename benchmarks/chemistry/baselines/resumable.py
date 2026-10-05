"""Resumable runs of the published atom mappers that need environments of their own, standard library only.

A run reads the tab-separated file of reaction ids and SMILES that benchmarks.chemistry.golden prepare or
benchmarks.chemistry.enzymemap prepare writes. The maps go to the output in the format of rxnmapper_golden.py, which
benchmarks.chemistry.exact_map_report reads, a reaction id to its mapped reaction SMILES, the confidence if the tool
gives one and the seconds it took. A reaction the tool fails on is left out of the output, so the report scores it as
wrong, and its reason goes to the file .failed.json beside the output.

Until every reaction is done the maps are kept in the file .partial.json, rewritten every few hundred reactions, and a
run that finds the two files maps only the reactions in neither. The output appears when the run is complete, so a
queue can test for it. A failure is final, delete its entry from .failed.json to try the reaction again.
"""

import json
import os
import time
from pathlib import Path


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def write(path, data):
    # a run killed while it writes leaves the previous file intact
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


class Run:
    def __init__(self, unmapped, out, every=200):
        self.rows = [
            line.split("\t")
            for line in Path(unmapped).read_text().splitlines()
            if line.strip()
        ]
        self.out = Path(out)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.partial = self.out.with_suffix(".partial.json")
        self.failures = self.out.with_suffix(".failed.json")
        self.mapped = read(self.out) or read(self.partial)
        self.failed = read(self.failures)
        self.every, self.since = int(every), 0
        self.start, self.resumed = time.time(), self.count()

        print(f"{self.resumed} of {len(self.rows)} done before", flush=True)

    def count(self):
        return len(self.mapped) + len(self.failed)

    def todo(self):
        return [
            (rid, smiles)
            for rid, smiles in self.rows
            if rid not in self.mapped and rid not in self.failed
        ]

    def add(self, rid, mapped_rxn, seconds, confidence=None, error=None):
        if mapped_rxn:
            entry = {"mapped_rxn": mapped_rxn}

            if confidence is not None:
                entry["confidence"] = str(confidence)

            entry["seconds"] = round(seconds, 3)
            self.mapped[rid] = entry
        else:
            self.failed[rid] = {
                "error": str(error)[-500:],
                "seconds": round(seconds, 3),
            }

        self.since += 1

        if self.since >= self.every:
            self.save()

    def save(self):
        write(self.failures, self.failed)

        if self.todo():
            write(self.partial, self.mapped)
        else:
            write(self.out, self.mapped)
            self.partial.unlink(missing_ok=True)

        self.since = 0
        new, seconds = self.count() - self.resumed, time.time() - self.start
        print(
            f"{self.count()} of {len(self.rows)}, {len(self.failed)} failed, "
            f"{seconds:.0f}s, {new / max(seconds, 1e-9):.2f} reactions/s",
            flush=True,
        )

    def finish(self):
        if self.since or not self.out.exists():
            self.save()

        print(f"done {len(self.mapped)} of {len(self.rows)}", flush=True)
