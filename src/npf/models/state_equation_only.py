"""No learning. The non-negative solution of the state equation closest to zero."""
import torch.nn as nn

from ..layers import project_state_equation


class StateEquationOnly(nn.Module):
    def forward(self, b, m=None):
        return project_state_equation(b.m.new_zeros(b.n_trans), b)
