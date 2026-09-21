"""Linear special case of PGNN that its paper trains on one fixed net (Eq. 13). Parameters belong to the net."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearPGNN(nn.Module):
    def __init__(self, net):
        super().__init__()
        mask = torch.as_tensor((net.Pre + net.Pos > 0).T, dtype=torch.float32)
        self.register_buffer("mask", mask)
        self.w_a, self.w_b = nn.Parameter(0.1 * torch.randn_like(mask)), nn.Parameter(0.1 * torch.randn_like(mask))
        self.omega, self.bias = nn.Parameter(torch.ones(len(mask))), nn.Parameter(torch.zeros(len(mask)))

    def forward(self, b, m=None):
        P = self.mask.shape[1]
        flow = b.m.view(-1, P) @ (self.w_a * self.mask).T + b.m_b.view(-1, P) @ (self.w_b * self.mask).T

        return F.relu(self.omega * flow + self.bias).reshape(-1)
