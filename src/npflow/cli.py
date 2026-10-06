"""The command line of npflow. The atom mapper needs no training, so it is the part of the package that runs as it is
installed: a reaction SMILES, or a file with one per line, gets the atom map numbers of the minimum firing vector.

    npflow map "CC(=O)Cl.CN>>CC(=O)NC"
    npflow map reactions.smi -o mapped.smi --processes 8 --seconds 60

A line of a file may carry an identifier after a tab, which is written back after the mapped reaction. A reaction the
mapper cannot map, because RDKit cannot read it or its product holds atoms no precursor supplies, gives an empty line,
so line k of the output belongs to line k of the input.
"""

import argparse
import functools
import sys
from multiprocessing import Pool
from pathlib import Path


def map_one(line, seconds):
    from .chem.mapping import map_reaction

    smiles, _, rest = line.partition("\t")
    try:
        mapped = map_reaction(smiles.strip(), seconds=seconds) if smiles.strip() else None
    except Exception:
        mapped = None

    return (mapped or "") + (f"\t{rest}" if rest else "")


def map_command(args):
    path = Path(args.reaction)
    lines = path.read_text().splitlines() if path.is_file() else [args.reaction]
    work = functools.partial(map_one, seconds=args.seconds)

    if args.processes > 1 and len(lines) > 1:
        with Pool(args.processes) as pool:
            mapped = pool.map(work, lines, chunksize=8)
    else:
        mapped = [work(line) for line in lines]

    text = "\n".join(mapped) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    else:
        sys.stdout.write(text)

    failed = sum(not m.partition("\t")[0] for m in mapped)
    if failed:
        print(f"{failed} of {len(lines)} reactions could not be mapped", file=sys.stderr)

    return 0 if failed < len(lines) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="npflow", description=__doc__.split("\n\n")[0])
    commands = ap.add_subparsers(dest="command", required=True)

    m = commands.add_parser("map", help="atom-map reactions with the minimum firing vector of the valence net")
    m.add_argument("reaction", help="a reaction SMILES, precursors>>product, or a file with one per line")
    m.add_argument("-o", "--output", help="write the mapped reactions to this file instead of the terminal")
    m.add_argument("--processes", type=int, default=1, help="map the lines of a file in parallel")
    m.add_argument("--seconds", type=float, default=60.0, help="limit on the search per reaction")
    m.set_defaults(run=map_command)

    args = ap.parse_args(argv)

    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
