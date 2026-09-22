"""Reaction classification from the firing vector."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import N_BOND, Encoder, mlp


class Classifier(nn.Module):
    """Reads the firing vector of a reaction through the state equation, without an atom mapping.

    petri False is the generic counterpart that pools both sides and concatenates them. gate weighs precursor
    molecules by the probability that a firing touches them. explicit_firing adds one term per fired transition,
    which needs a mapping, those of the exact mapper. mapped_atoms is its generic counterpart,
    it reads the same mapping as one term per seated atom and never forms a firing vector.
    """

    def __init__(
        self,
        n_classes,
        d=128,
        rounds=3,
        petri=True,
        dropout=0.2,
        gate=True,
        n_firing_types=0,
        explicit_firing=False,
        mapped_atoms=False,
    ):
        super().__init__()
        self.petri, self.gated, self.by_atom = petri, gate and petri, mapped_atoms
        explicit_firing = explicit_firing or mapped_atoms
        self.transition = (
            mlp(4 * d + 2 * N_BOND, 2 * d, 2 * d) if explicit_firing else None
        )

        # auxiliary task, which transition types fired, read linearly from the vector the classifier sees
        self.firing = (
            nn.Linear(
                4 * d * (rounds + 1) + (2 * d if explicit_firing else 0), n_firing_types
            )
            if n_firing_types
            else None
        )

        # the encoder is local on purpose. psi depends on the R-ball of a place only, so untouched places cancel
        self.encoder = Encoder(d, rounds)
        self.atom = nn.ModuleList(mlp(d, 2 * d, d) for _ in range(rounds + 1))
        self.bond = nn.ModuleList(
            mlp(d + N_BOND - 1, 2 * d, d) for _ in range(rounds + 1)
        )
        width = 2 * d * (rounds + 1)
        self.out = nn.Sequential(
            nn.Linear(2 * width + (2 * d if explicit_firing else 0), 4 * d),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d, n_classes),
        )
        self.gate = mlp(2 * d, d, 1)

    def marking(self, depths, bonds, mask, weight):
        """sum of psi over the atom places and the bond places of one side, at every depth of the encoder."""
        adj = F.one_hot(bonds, N_BOND)[..., 1:].to(depths[0].dtype)
        parts = []
        for h, atom, bond in zip(depths, self.atom, self.bond):
            parts.append((atom(h) * (mask * weight)[..., None]).sum(1))

            # h_i + h_j for every bond type at atom i
            ends = (
                torch.einsum("bijt,bjd->bitd", adj, h)
                + h[:, :, None, :] * adj.sum(2)[..., None]
            )
            pair = torch.cat(
                [
                    ends,
                    torch.eye(N_BOND - 1, device=h.device).expand(
                        *ends.shape[:2], -1, -1
                    ),
                ],
                -1,
            )
            parts.append(
                (bond(pair) * ((adj.sum(2) > 0) * weight[..., None])[..., None]).sum(
                    (1, 2)
                )
                / 2
            )

        return torch.cat(parts, -1)

    def forward(self, b, return_gate=False):
        da = self.encoder(b["xa"], b["ba"], b["mask_a"], all_depths=True)
        db = self.encoder(b["xb"], b["bb"], b["mask_b"], all_depths=True)
        ones_b = b["mask_b"].to(da[0].dtype)
        logit = None

        if self.gated:
            # solvents and catalysts have no counterpart in B and would not cancel, so every precursor molecule gets
            # a participation gate computed from the molecule and the product
            member = (
                F.one_hot(b["frag_a"].clamp(min=0), int(b["frag_a"].max()) + 1).to(
                    da[0].dtype
                )
                * b["mask_a"][..., None]
            )
            molecule = (
                torch.einsum("bnf,bnd->bfd", member, da[-1])
                / member.sum(1).clamp(min=1)[..., None]
            )
            product = (db[-1] * ones_b[..., None]).sum(1) / ones_b.sum(1, keepdim=True)
            logit = self.gate(
                torch.cat([molecule, product[:, None].expand_as(molecule)], -1)
            ).squeeze(-1)
            weight = torch.einsum("bnf,bf->bn", member, torch.sigmoid(logit))
            self.gate_logit, self.member = logit, member
        else:
            weight = b["mask_a"].to(da[0].dtype)

        a, p = self.marking(da, b["ba"], b["mask_a"], weight), self.marking(
            db, b["bb"], b["mask_b"], ones_b
        )
        spectators = (
            self.marking(
                da, b["ba"], b["mask_a"], b["mask_a"].to(weight.dtype) - weight
            )
            if self.gated
            else torch.zeros_like(a)
        )

        # state equation readout r = sum_B psi - sum_A psi. with the true correspondence pi it equals the sum of
        # psi_B(i) - psi_A(pi(i)) over atoms within R of a changed place, minus psi_A over unmatched atoms.
        # everything the firing did not touch cancels exactly
        readout = torch.cat([p - a, spectators] if self.petri else [a, p], -1)

        if self.transition is not None:
            readout = torch.cat(
                [
                    readout,
                    (self.seated_atoms if self.by_atom else self.fired_transitions)(
                        b, da[-1], db[-1]
                    ),
                ],
                -1,
            )

        self.firing_prediction = (
            self.firing(readout) if self.firing is not None else None
        )
        out = self.out(readout)

        return (out, weight) if return_gate else out

    def fired_transitions(self, b, ha, hb):
        """Sum over the fired transitions of f(atoms i and j before the firing, their partners after it)."""
        k, i, j = torch.nonzero(torch.triu(b["edits"], 1) > 0, as_tuple=True)

        # partner maps a precursor atom to its product atom
        partner = torch.full_like(b["mask_a"], -1, dtype=torch.long)
        rows = torch.arange(b["target"].shape[1], device=partner.device).expand_as(
            b["target"]
        )
        valid = b["target"] >= 0
        partner[torch.nonzero(valid, as_tuple=True)[0], b["target"][valid]] = rows[
            valid
        ]
        after = lambda atom: torch.where(
            (partner[k, atom] >= 0)[:, None],
            hb[k, partner[k, atom].clamp(min=0)],
            torch.zeros_like(ha[k, atom]),
        )
        x = torch.cat(
            [
                ha[k, i] + ha[k, j],
                ha[k, i] * ha[k, j],
                after(i) + after(j),
                after(i) * after(j),
                F.one_hot(b["ba"][k, i, j], N_BOND).to(ha.dtype),
                F.one_hot(b["edits"][k, i, j] - 1, N_BOND).to(ha.dtype),
            ],
            -1,
        )

        return ha.new_zeros(len(ha), self.transition[-1].out_features).index_add_(
            0, k, self.transition(x)
        )

    def seated_atoms(self, b, ha, hb):
        """Sum over the seated product atoms of f(the atom after the reaction, its seat before it). Same width as a
        fired transition, the bond type slots stay empty."""
        k, i = torch.nonzero((b["target"] >= 0) & b["mask_b"], as_tuple=True)
        after, before = hb[k, i], ha[k, b["target"][k, i]]
        x = torch.cat(
            [
                after + before,
                after * before,
                after - before,
                (after - before) ** 2,
                ha.new_zeros(len(k), 2 * N_BOND),
            ],
            -1,
        )

        return ha.new_zeros(len(ha), self.transition[-1].out_features).index_add_(
            0, k, self.transition(x)
        )

    def firing_loss(self, b):
        has_map = (b["target"] >= 0).any(1)
        if self.firing is None or "hist" not in b or not has_map.any():
            return 0.0

        return F.mse_loss(self.firing_prediction[has_map], b["hist"][has_map])

    def auxiliary_loss(self, b):
        """Where a mapping is available during training, those of the exact mapper, the molecules touched by a firing
        are known."""
        has_map = (b["target"] >= 0).any(1)
        if not self.gated or not has_map.any():
            return 0.0

        kept = torch.zeros_like(b["mask_a"], dtype=torch.float32).scatter_(
            1, b["target"].clamp(min=0), (b["target"] >= 0).float()
        )
        touched = (torch.einsum("bnf,bn->bf", self.member, kept) > 0).float()
        exists = self.member.sum(1) > 0

        return F.binary_cross_entropy_with_logits(
            self.gate_logit[has_map][exists[has_map]], touched[has_map][exists[has_map]]
        )
