"""Equilibrium layer for reversible nets. Only the place energies are learned."""

import torch
import torch.nn as nn

from ..layers import mlp


class ThermoNPF(nn.Module):
    def __init__(self, hidden=64, newton_steps=40):
        super().__init__()
        self.energy = mlp(1, hidden, 1)
        self.newton_steps = newton_steps

    def forward(self, b, m=None):
        """Equilibrium marking in the compatibility class of b.m."""
        G, S, Pmax, _ = b.pad_shape
        pad = (
            lambda v: v.new_zeros(G * S * Pmax)
            .index_copy_(0, b.pad_p, v)
            .view(G, S, Pmax)
        )
        m0, mask = pad(b.m).double(), pad(torch.ones_like(b.m)).double()
        log_ref = -pad(self.energy(b.e).squeeze(-1)).double()
        X = b.X
        target = torch.einsum("gsp,gpr->gsr", m0, X)
        eye = 1e-9 * torch.eye(X.shape[2], dtype=torch.float64, device=X.device)

        # m* = m_ref exp(X lam) with X a basis of ker C^T. lam minimises the strictly convex dual, whose gradient is
        # X^T (m - m_0) and Hessian X^T diag(m) X, so the minimiser exists and is unique, by Horn and Jackson and Birch
        def newton(lam, ref):
            m = torch.exp(ref + torch.einsum("gpr,gsr->gsp", X, lam)) * mask
            grad = torch.einsum("gsp,gpr->gsr", m, X) - target
            H = torch.einsum("gpr,gsp,gpq->gsrq", X, m, X) + eye

            return torch.linalg.solve(H, grad[..., None])[..., 0], grad

        def dual(lam, ref):
            return (torch.exp(ref + torch.einsum("gpr,gsr->gsp", X, lam)) * mask).sum(
                2
            ) - (lam * target).sum(2)

        lam = torch.zeros_like(target)

        with torch.no_grad():
            ref = log_ref.detach()
            for _ in range(self.newton_steps):
                step, grad = newton(lam, ref)
                if grad.abs().max() < 1e-10:
                    break

                alpha = torch.ones_like(grad[..., :1])
                base, slope = dual(lam, ref), (grad * step).sum(2)
                for _ in range(20):
                    worse = (
                        dual(lam - alpha * step, ref)
                        > base - 1e-4 * alpha[..., 0] * slope
                    )
                    if not worse.any():
                        break

                    alpha = torch.where(worse[..., None], alpha / 2, alpha)

                lam = lam - alpha * step

        # a differentiable Newton step at lam* keeps it and carries its derivative, the implicit function theorem
        lam = lam - newton(lam, log_ref)[0]
        m = torch.exp(log_ref + torch.einsum("gpr,gsr->gsp", X, lam)) * mask

        # remove the rounding error of the conserved totals. X is orthonormal, so this is a projection along X
        m = m + torch.einsum(
            "gpr,gsr->gsp", X, target - torch.einsum("gsp,gpr->gsr", m, X)
        )

        return m.reshape(-1)[b.pad_p].float()
