"""RDT, the Reaction Decoder Tool of Rahman et al. 2016, on unmapped reactions, one Java process per reaction.

RDT is a Java jar with no batch mode, so every reaction runs the command line tool of the jar in a folder of its own and
the map is the line after SELECTED AAM MAPPING in the text file it writes. A process that runs longer than the time
limit is killed and the reaction counts as failed. This script needs only the standard library, Java and the jar
live in ~/.local/share/pgnn-tools, a portable Temurin 21 and the jar with dependencies of an RDT release on GitHub.
The output and its resumption are those of resumable.py.

    uv run --no-project --python 3.11 python benchmarks/chemistry/baselines/rdt_mapper.py \\
        data/enzymemap_unmapped.txt results/chem/rdt_enzymemap.json 20-23 300

The arguments after the output are the cores and the time limit in seconds. Every core takes one reaction at a time
and its Java process is pinned to it with taskset, since RDT starts several threads per reaction whatever the JVM is
told, and the time limit then counts the time of one core.

The default release is 2.4.1 of 2020, the last of the original line, which returned the same map on repeated runs. The
jar of 3.3.0, the last release of 2026 with a jar, hangs in a race between its threads on a quarter to half of the runs
of some small reactions unless it is pinned to one core, returns different maps on repeated runs even when pinned, and
left product atoms without a precursor atom on 9 of the first 100 reactions of the EnzymeMap 3k set, where it mapped
48 of the 93 scored reactions right and 2.4.1 79. Another jar is the fifth argument, and attempts above 1 as the sixth
run a reaction again after a time out.
"""

import queue
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from resumable import Run

TOOLS = Path.home() / ".local/share/pgnn-tools"
JAVA = TOOLS / "jdk/bin/java"
JAR = TOOLS / "rdt/rdt-2.4.1-jar-with-dependencies.jar"


def cpus(spec):
    """Cores of a taskset list such as 20-23 or 0,2,4."""
    out = []

    for part in str(spec).split(","):
        low, _, high = part.partition("-")
        out += list(range(int(low), int(high or low) + 1))

    return out


def rdt(smiles, core, jar, timeout, memory):
    """The mapped SMILES and None, or None and the reason."""
    task = ["-Q", "SMI", "-q", smiles, "-j", "AAM", "-f", "TEXT", "-p", "r"]

    # one core for the JVM and its garbage collector, RDT sizes some of its thread pools by it
    java = [
        str(JAVA),
        f"-Xmx{memory}",
        "-XX:ActiveProcessorCount=1",
        "-XX:+UseSerialGC",
    ]
    command = ["taskset", "-c", str(core), *java, "-jar", str(jar), *task]

    with tempfile.TemporaryDirectory(prefix="rdt-") as folder:
        try:
            done = subprocess.run(
                command,
                cwd=folder,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None, f"timeout after {timeout}s"

        # the jar of 2.4.1 exits with 1 after it has written the map, so only the file counts
        for path in Path(folder).glob("*_AAM.txt"):
            lines = path.read_text().splitlines()
            if "SELECTED AAM MAPPING" in lines:
                mapped = lines[lines.index("SELECTED AAM MAPPING") + 1].strip()
                if ">>" in mapped:
                    return mapped, None

        return None, f"exit {done.returncode} " + (done.stderr or done.stdout)[-400:]


def main(
    unmapped="data/enzymemap_unmapped.txt",
    out="results/chem/rdt_enzymemap.json",
    cores="20-23",
    timeout=300,
    jar=JAR,
    attempts=1,
    memory="4g",
):
    run = Run(unmapped, out, every=100)
    todo = run.todo()

    print(f"{jar} on cores {cores}, {timeout}s per reaction", flush=True)

    # a worker takes a free core for each reaction and gives it back afterwards
    free = queue.Queue()
    for core in cpus(cores):
        free.put(core)

    def one(smiles):
        core, start = free.get(), time.time()

        try:
            for _ in range(int(attempts)):
                mapped, error = rdt(smiles, core, jar, float(timeout), memory)
                if mapped or not error.startswith("timeout"):
                    break
        finally:
            free.put(core)

        return mapped, error, time.time() - start

    with ThreadPoolExecutor(free.qsize()) as pool:
        jobs = {pool.submit(one, smiles): rid for rid, smiles in todo}
        for job in as_completed(jobs):
            mapped, error, seconds = job.result()
            run.add(jobs[job], mapped, seconds, error=error)

    run.finish()


if __name__ == "__main__":
    main(*sys.argv[1:])
