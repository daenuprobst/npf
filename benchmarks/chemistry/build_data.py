"""Build the reaction data sets.

    uv run python -m benchmarks.chemistry.build_data              # writes data/schneider50k.pkl
    uv run python -m benchmarks.chemistry.build_data uspto-mit    # writes data/uspto_mit.pkl
"""
import sys

from npf import chem

if __name__ == "__main__":
    if "uspto-mit" in sys.argv:
        chem.build_uspto_mit()
    else:
        chem.build()
