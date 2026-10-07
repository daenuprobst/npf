"""Petri nets with attributes."""
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import null_space


@dataclass(eq=False)
class Net:
    n_places: int
    n_trans: int

    # arcs from places to transitions as index arrays with weights Pre(p, t)
    pre_p: np.ndarray
    pre_t: np.ndarray
    pre_w: np.ndarray

    # arcs from transitions to places with weights Pos(p, t)
    pos_p: np.ndarray
    pos_t: np.ndarray
    pos_w: np.ndarray

    # static attributes of places [P, 1] and of transitions [T, 1]
    e: np.ndarray
    a: np.ndarray
    Pre: np.ndarray = field(init=False)
    Pos: np.ndarray = field(init=False)
    C: np.ndarray = field(init=False)
    C_pinv: np.ndarray = field(init=False)
    X: np.ndarray = field(init=False)

    def __post_init__(self):
        self.Pre = np.zeros((self.n_places, self.n_trans))
        self.Pos = np.zeros((self.n_places, self.n_trans))
        self.Pre[self.pre_p, self.pre_t] = self.pre_w
        self.Pos[self.pos_p, self.pos_t] = self.pos_w

        # incidence matrix of the state equation m' = m + C sigma
        self.C = self.Pos - self.Pre
        self.C_pinv = np.linalg.pinv(self.C)

        # an orthonormal basis of the P-invariants ker C^T
        self.X = null_space(self.C.T)

