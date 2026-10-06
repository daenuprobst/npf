"""Samples on one net. A sample is a pair of markings of one run and the time between them."""

from dataclasses import dataclass

import numpy as np

from .nets import Net


@dataclass
class Group:
    """All samples that live on one net."""

    net: Net

    # marking A or the current marking, [S, P]
    m: np.ndarray

    # marking B, [S, P]
    m_b: np.ndarray = None

    # time between the two markings, [S]
    dt: np.ndarray = None

    # identity of every transition in a vocabulary shared by all nets [T], and an environment per sample [S, E]
    t_id: np.ndarray = None
    env: np.ndarray = None
