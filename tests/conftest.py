"""Shared fixtures. Tests that need the data sets are skipped when they are missing. Everything runs on the CPU.

    uv run pytest -q

style_audit.py in this directory is a script and not a test.
"""

import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NPF_DATA", ROOT / "data"))
RESULTS = Path(os.environ.get("NPF_RESULTS", ROOT / "results"))

for entry in (str(ROOT), str(ROOT / "src")):
    if entry in sys.path:
        sys.path.remove(entry)

    sys.path.insert(0, entry)


def pytest_configure(config):
    # the data loaders of the chemistry script fork worker processes while torch threads exist
    config.addinivalue_line(
        "filterwarnings",
        "ignore:This process .* is multi-threaded, use of fork:DeprecationWarning",
    )


@pytest.fixture(scope="session")
def schneider_all():
    """The whole data set, about 220 MB. Tests must not modify the reactions in place."""
    path = DATA / "schneider50k.pkl"
    if not path.exists():
        pytest.skip(f"{path} not found")

    import pickle

    return pickle.loads(path.read_bytes())


@pytest.fixture(scope="session")
def schneider(schneider_all):
    # a fixed spread over the file, mapped and unmapped reactions, train and test
    index = np.linspace(0, len(schneider_all["reactions"]) - 1, 360).astype(int)

    return {
        "reactions": [schneider_all["reactions"][i] for i in index],
        "classes": schneider_all["classes"],
    }


def assert_same_arrays(a, b, what=""):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, f"{what}: shape {a.shape} != {b.shape}"
    assert a.dtype == b.dtype, f"{what}: dtype {a.dtype} != {b.dtype}"
    assert np.array_equal(
        a, b, equal_nan=True
    ), f"{what}: values differ, max abs diff {np.abs(a.astype(float) - b.astype(float)).max()}"
