"""No learning. The non-negative solution of the state equation closest to zero."""

import torch.nn as nn

from ..layers import kl_project_state_equation, project_state_equation


class StateEquationOnly(nn.Module):
    """With kl_options the I projection of a uniform prior, the maximum entropy firing counts of the fibre."""

    def __init__(self, kl_options=None):
        super().__init__()
        self.kl_options = kl_options

    def forward(self, b, m=None):
        if self.kl_options is None:
            return project_state_equation(b.m.new_zeros(b.n_trans), b)

        return kl_project_state_equation(b.m.new_ones(b.n_trans), b, **self.kl_options)
