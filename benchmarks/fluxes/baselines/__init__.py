"""Published flux inference methods and plain learners on the metabolism benchmark, one method per module.

Every module runs as uv run python -m benchmarks.fluxes.baselines.NAME --split cv and writes the result file of
flux.py to results/metabolism/SPLIT/METHOD-0.json. Modules that solve quadratic programs, sample or fit forests need
the extra packages named in their docstring, passed as uv run --with PACKAGE.
"""
