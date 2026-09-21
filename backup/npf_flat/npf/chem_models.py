"""Neural Petri Flow on the valence Petri net of a chemical reaction.

The net of one reaction, over the heavy atoms 1..N of the precursors:

    places        B_ij   electron pairs shared by atoms i and j            marking = bond order
                  H_i    hydrogens on atom i (Q_i: its charge)
    transitions   B_ij -> type k   break (k = 0), form (from 0) or re-type the bond between i and j;
                                   every firing moves valence tokens between B_ij and H_i, H_j
    P-invariant   sum_j b_ij + h_i - q_i  is conserved for every atom (holds for 99.85 % of the atoms of
                  Schneider 50k): hydrogens never have to be predicted, they follow from the invariant
    enabling      a marking must stay non-negative: h_i >= 0 is exactly the valence rule
    state eq.     E_B = E_A + C sigma, the Dugundji-Ugi equation B + R = E

All three tasks are questions about the firing vector sigma of this net:
    classify   read sigma out through the state equation: sum_B phi(place) - sum_A phi(place) = C_phi sigma,
               places that no transition touches cancel, no atom mapping needed
    map        the correspondence between the places of B and of A under which sigma is most probable
               (minimum chemical distance with learned transition costs); the relaxation is a sequence of
               Sinkhorn steps, each the equilibrium of a conservative Petri net (npf/theory_checks.py, C)
    forward    play the token game: fire the transitions with the highest learned rates that keep every
               marking non-negative, then B = A + C sigma

`petri=False` gives the PGNN-style counterpart: the same message passing on the same net, but a generic
readout in place of the Petri semantics (pooled embeddings / similarity matching / independent edge labels).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .chem_data import BOND_ORDER, N_ATOM_FEAT

N_BOND = 5  # none, single, double, triple, aromatic


def mlp(n_in, n_hidden, n_out):
    return nn.Sequential(nn.Linear(n_in, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_out))


class PetriLayer(nn.Module):
    """One round of place <-> transition message passing (PGNN Eqs. 9-11) on the valence net: every bond
    place B_ij with tokens links the atom places i and j, with one message function per bond type."""

    def __init__(self, d):
        super().__init__()
        self.rel = nn.Linear(d, (N_BOND - 1) * d, bias=False)
        self.upd = mlp(2 * d, 2 * d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, h, adj):
        B, N, d = h.shape
        m = torch.einsum("bijt,bjtd->bid", adj, self.rel(h).view(B, N, N_BOND - 1, d))
        return self.norm(h + self.upd(torch.cat([h, m], -1)))


class Encoder(nn.Module):
    """`rounds` of message passing along marked bond places; `attention` layers let atoms that are not bonded
    exchange messages as well - these are the neighbourhoods of the (unmarked) form-transitions."""

    def __init__(self, d=128, rounds=4, attention=0, n_extra=0):
        super().__init__()
        self.inp = nn.Linear(N_ATOM_FEAT + n_extra, d)
        self.layers = nn.ModuleList(PetriLayer(d) for _ in range(rounds))
        self.attn = nn.ModuleList(nn.TransformerEncoderLayer(d, 8, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
                                  for _ in range(attention))

    def forward(self, x, bonds, mask, all_depths=False):
        adj = F.one_hot(bonds, N_BOND)[..., 1:].to(x.dtype)
        h = self.inp(x) * mask[..., None]
        depths = [h]
        for k, layer in enumerate(self.layers):
            h = layer(h, adj) * mask[..., None]
            if k < len(self.attn):
                h = self.attn[k](h, src_key_padding_mask=~mask) * mask[..., None]
            depths.append(h)
        return depths if all_depths else h


# ------------------------------------------------------------------ reaction classification

class Classifier(nn.Module):
    """Reads the firing vector out of the state equation. For any function phi of a place and its k-hop
    surroundings, sum_B phi - sum_A phi only receives contributions from places whose marking or surroundings
    were changed by a firing: everything else cancels exactly. The sum runs over atom places and over bond
    places (so formed and broken bonds appear directly), at every message-passing depth.

    Precursor molecules that no transition touches (solvents, salts, catalysts) have no counterpart in B and
    would not cancel. A participation gate per molecule - the probability that a firing touches it, learned
    from the recorded mappings of the training reactions, needing no mapping at test time - weighs them."""

    def __init__(self, n_classes, d=128, rounds=3, petri=True, dropout=0.2, gate=True, n_firing_types=0, explicit_firing=False):
        super().__init__()
        self.petri, self.gated = petri, gate and petri
        # explicit_firing: besides the state-equation readout, read the firing vector itself - one term per re-typed bond
        # place, built from the surroundings of its two atoms before and after (needs a mapping, e.g. from our mapper)
        self.transition = mlp(4 * d + 2 * N_BOND, 2 * d, 2 * d) if explicit_firing else None
        # auxiliary task: which transition types fired, read *linearly* from the same vector the classifier sees
        self.firing = nn.Linear(4 * d * (rounds + 1), n_firing_types) if n_firing_types else None
        self.encoder = Encoder(d, rounds)  # local on purpose: untouched places must cancel
        self.atom = nn.ModuleList(mlp(d, 2 * d, d) for _ in range(rounds + 1))
        self.bond = nn.ModuleList(mlp(d + N_BOND - 1, 2 * d, d) for _ in range(rounds + 1))
        width = 2 * d * (rounds + 1)
        self.out = nn.Sequential(nn.Linear(2 * width + (2 * d if explicit_firing else 0), 4 * d), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(4 * d, n_classes))
        self.gate = mlp(2 * d, d, 1)

    def marking(self, depths, bonds, mask, weight):
        adj = F.one_hot(bonds, N_BOND)[..., 1:].to(depths[0].dtype)  # [B, N, N, types]
        parts = []
        for h, atom, bond in zip(depths, self.atom, self.bond):
            parts.append((atom(h) * (mask * weight)[..., None]).sum(1))
            ends = torch.einsum("bijt,bjd->bitd", adj, h) + h[:, :, None, :] * adj.sum(2)[..., None]  # h_i + h_j per bond type at atom i
            pair = torch.cat([ends, torch.eye(N_BOND - 1, device=h.device).expand(*ends.shape[:2], -1, -1)], -1)
            parts.append((bond(pair) * ((adj.sum(2) > 0) * weight[..., None])[..., None]).sum((1, 2)) / 2)
        return torch.cat(parts, -1)

    def forward(self, b, return_gate=False):
        da = self.encoder(b["xa"], b["ba"], b["mask_a"], all_depths=True)
        db = self.encoder(b["xb"], b["bb"], b["mask_b"], all_depths=True)
        ones_b = b["mask_b"].to(da[0].dtype)
        logit = None
        if self.gated:  # one gate per precursor molecule, from the molecule and the product
            member = F.one_hot(b["frag_a"].clamp(min=0), int(b["frag_a"].max()) + 1).to(da[0].dtype) * b["mask_a"][..., None]  # [B, N, F]
            molecule = torch.einsum("bnf,bnd->bfd", member, da[-1]) / member.sum(1).clamp(min=1)[..., None]
            product = (db[-1] * ones_b[..., None]).sum(1) / ones_b.sum(1, keepdim=True)
            logit = self.gate(torch.cat([molecule, product[:, None].expand_as(molecule)], -1)).squeeze(-1)  # [B, F]
            weight = torch.einsum("bnf,bf->bn", member, torch.sigmoid(logit))
            self.gate_logit, self.member = logit, member
        else:
            weight = b["mask_a"].to(da[0].dtype)
        a, p = self.marking(da, b["ba"], b["mask_a"], weight), self.marking(db, b["bb"], b["mask_b"], ones_b)
        spectators = self.marking(da, b["ba"], b["mask_a"], b["mask_a"].to(weight.dtype) - weight) if self.gated else torch.zeros_like(a)
        readout = torch.cat([p - a, spectators] if self.petri else [a, p], -1)
        if self.transition is not None:
            readout = torch.cat([readout, self.fired_transitions(b, da[-1], db[-1])], -1)
        self.firing_prediction = self.firing(readout) if self.firing is not None else None
        out = self.out(readout)
        return (out, weight) if return_gate else out

    def fired_transitions(self, b, ha, hb):
        """sum over fired transitions t = (B_ij: old -> new) of f(atoms i, j before the firing, their partners after it)."""
        k, i, j = torch.nonzero(torch.triu(b["edits"], 1) > 0, as_tuple=True)
        partner = torch.full_like(b["mask_a"], -1, dtype=torch.long)  # precursor atom -> product atom
        rows = torch.arange(b["target"].shape[1], device=partner.device).expand_as(b["target"])
        valid = b["target"] >= 0
        partner[torch.nonzero(valid, as_tuple=True)[0], b["target"][valid]] = rows[valid]
        after = lambda atom: torch.where((partner[k, atom] >= 0)[:, None], hb[k, partner[k, atom].clamp(min=0)], torch.zeros_like(ha[k, atom]))
        x = torch.cat([ha[k, i] + ha[k, j], ha[k, i] * ha[k, j], after(i) + after(j), after(i) * after(j),
                       F.one_hot(b["ba"][k, i, j], N_BOND).to(ha.dtype), F.one_hot(b["edits"][k, i, j] - 1, N_BOND).to(ha.dtype)], -1)
        return ha.new_zeros(len(ha), self.transition[-1].out_features).index_add_(0, k, self.transition(x))

    def firing_loss(self, b):
        has_map = (b["target"] >= 0).any(1)
        if self.firing is None or "hist" not in b or not has_map.any():
            return 0.0
        return F.mse_loss(self.firing_prediction[has_map], b["hist"][has_map])

    def auxiliary_loss(self, b):
        """Where the recorded mapping is available during training, the molecules touched by a firing are known."""
        has_map = (b["target"] >= 0).any(1)
        if not self.gated or not has_map.any():
            return 0.0
        kept = torch.zeros_like(b["mask_a"], dtype=torch.float32).scatter_(1, b["target"].clamp(min=0), (b["target"] >= 0).float())
        touched = (torch.einsum("bnf,bn->bf", self.member, kept) > 0).float()
        exists = self.member.sum(1) > 0
        return F.binary_cross_entropy_with_logits(self.gate_logit[has_map][exists[has_map]], touched[has_map][exists[has_map]])


# ------------------------------------------------------------------ atom mapping

def sinkhorn(log_s, n_iter=10):
    """Rows: product atoms (mass 1) and a last slack row that takes the precursor atoms which leave;
    columns: precursor atoms (mass 1). Returns the log of the equilibrium assignment."""
    B, R, C = log_s.shape
    log_r = torch.zeros(B, R, device=log_s.device); log_r[:, -1] = np.log(max(C - R + 1, 1))
    u, v = torch.zeros(B, R, device=log_s.device), torch.zeros(B, C, device=log_s.device)
    for _ in range(n_iter):
        u = log_r - torch.logsumexp(log_s + v[:, None, :], 2)
        v = -torch.logsumexp(log_s + u[:, :, None], 1)
    return log_s + u[:, :, None] + v[:, None, :]


class Mapper(nn.Module):
    """Atom mapping = the correspondence between the places of B and of A under which the firing vector is most
    probable. Each round (i) scores every candidate pair by the bonds it keeps under the current correspondence
    (every bond not kept costs a firing: minimum chemical distance with learned costs), (ii) checks the
    correspondence as a net morphism - an atom and its expected partner must agree, and disagreement spreads
    along the bond places - and (iii) relaxes to the equilibrium of the assignment net (Sinkhorn), whose
    P-invariants say that every product atom has exactly one partner and every precursor atom at most one."""

    def __init__(self, d=192, rounds=6, petri=True, consensus=10, equilibrium=True, kept_bonds=True, morphism=True, token_cost=True):
        super().__init__()
        self.petri, self.consensus = petri, consensus if petri else 0
        # ablation switches, one per Petri component (all on = NPF, petri=False = generic readout)
        self.use_equilibrium, self.use_kept, self.use_morphism, self.use_cost = equilibrium, kept_bonds, morphism, token_cost
        self.encoder = Encoder(d, rounds)
        self.q, self.k = nn.Linear(d, d), nn.Linear(d, d)
        self.slack = nn.Parameter(torch.zeros(()))
        # log-rate of keeping a bond of type b' in A as a bond of type b in B (anything else costs a firing)
        self.keep = nn.Parameter(torch.eye(N_BOND - 1))
        self.alpha = nn.Parameter(torch.ones(max(self.consensus, 1)))
        self.token_cost = nn.Parameter(torch.ones(2))  # a hydrogen / a charge that has to move is a firing, too
        self.agree_b, self.agree_a = mlp(3 * d, 2 * d, d), mlp(3 * d, 2 * d, d)
        self.spread_b, self.spread_a = PetriLayer(d), PetriLayer(d)

    def forward(self, b, all_rounds=False):
        ha = self.encoder(b["xa"], b["ba"], b["mask_a"])
        hb = self.encoder(b["xb"], b["bb"], b["mask_b"])
        moved = self.token_cost[0] * (b["h_b"][:, :, None] - b["h_a"][:, None, :]).abs() \
            + self.token_cost[1] * (b["q_b"][:, :, None] - b["q_a"][:, None, :]).abs() if self.petri and self.use_cost else 0.0
        score = lambda: self.q(hb) @ self.k(ha).transpose(1, 2) / ha.shape[-1] ** 0.5 - moved
        allowed = (b["el_b"][:, :, None] == b["el_a"][:, None, :]) & b["mask_a"][:, None, :]  # elements are conserved
        pad_row = ~b["mask_b"][:, :, None]  # padded product rows behave like slack

        def equilibrium(s):
            s = torch.where(allowed, s, torch.full_like(s, -1e4))
            if not self.petri or not self.use_equilibrium:  # generic readout: every product atom picks its most similar precursor atom
                return torch.log_softmax(s, 2)
            s = torch.where(pad_row, torch.zeros_like(s), s)
            return sinkhorn(torch.cat([s, self.slack.expand(len(s), 1, s.shape[2])], 1))[:, :-1]

        rounds = [equilibrium(score())]
        adj_a = F.one_hot(b["ba"], N_BOND)[..., 1:].float()
        adj_b = F.one_hot(b["bb"], N_BOND)[..., 1:].float()
        for t in range(self.consensus):
            p = rounds[-1].exp() * b["mask_b"][:, :, None]
            # expected number of bonds of product atom i that survive if i sits on precursor atom j
            kept = torch.einsum("bikt,bkl,bjlu,tu->bij", adj_b, p, adj_a, self.keep) if self.use_kept else 0.0
            if self.use_morphism:
                partner_b, partner_a = p @ ha, p.transpose(1, 2) @ hb
                hb = hb + self.spread_b(self.agree_b(torch.cat([hb, partner_b, hb - partner_b], -1)), adj_b) * b["mask_b"][..., None]
                ha = ha + self.spread_a(self.agree_a(torch.cat([ha, partner_a, ha - partner_a], -1)), adj_a) * b["mask_a"][..., None]
            rounds.append(equilibrium(score() + self.alpha[t] * kept))
        return rounds if all_rounds else rounds[-1]

    @staticmethod
    def loss(log_p, b):
        """Equivalent atoms are interchangeable on both sides: product atom i may sit on any precursor atom that
        is equivalent to the recorded partner of any product atom equivalent to i."""
        partner = torch.gather(b["sym_a"], 1, b["target"].clamp(min=0))  # class of the recorded precursor atom
        same = (b["sym_b"][:, :, None] == b["sym_b"][:, None, :]) & b["mask_b"][:, None, :]  # [B, i, i']
        same = ((partner[:, None, :, None] == b["sym_a"][:, None, None, :]) & same[..., None]).any(2)  # [B, i, j]
        ll = torch.logsumexp(torch.where(same & b["mask_a"][:, None, :], log_p, torch.full_like(log_p, -1e4)), 2)
        return -(ll * b["mask_b"]).sum() / b["mask_b"].sum()

    @staticmethod
    def decode(log_p, b):
        """One-to-one assignment (Hungarian) per reaction; returns a list of arrays product atom -> precursor atom."""
        out = []
        for lp, mb, ma in zip(log_p.detach().cpu().numpy(), b["mask_b"].cpu().numpy(), b["mask_a"].cpu().numpy()):
            rows, cols = linear_sum_assignment(-lp[mb][:, ma])
            out.append(cols[np.argsort(rows)])
        return out


# ------------------------------------------------------------------ forward prediction

class TokenGame(nn.Module):
    """Forward prediction as a firing sequence of the valence net. One step:
        rates     lambda(t | current marking) for every transition t = (B_ij -> type k), and for STOP
        enabling  t may fire only if the hydrogen places of i and j stay within capacity (the valence rule),
                  and every bond place is re-typed at most once
        firing    the marking is updated by the state equation and the rates are recomputed
    Concurrent transitions have no order (a firing sequence is a trace), so training is order-agnostic: from a
    random subset of the true firings, any remaining enabled one is a correct next step."""

    def __init__(self, d=128, rounds=6, attention=4, max_steps=12, enabling=True, hops=0):
        super().__init__()
        self.max_steps, self.enabling = max_steps, enabling  # enabling=False: ablation without the valence capacities
        self.hops = hops  # > 0: the pair (i, j) also sees its distance in the current marking, in bond places (ring closures)
        self.encoder = Encoder(d, rounds, attention, n_extra=8)
        self.pair = mlp(2 * d + N_BOND + 2 + (hops + 1 if hops else 0), 2 * d, N_BOND)
        self.stop = mlp(d, d, 1)
        self.register_buffer("order", torch.tensor(BOND_ORDER, dtype=torch.float32))

    def rates(self, b, cur, fired):
        """Log-rates [B, N, N, types] (-inf where the transition is not enabled) and the STOP log-rate [B]."""
        hydrogens = b["h_a"] - (self.order[cur] - self.order[b["ba"]]).sum(2)
        touched = fired.any(2, keepdim=True).float()
        x = torch.cat([b["xa"], F.one_hot((hydrogens.round().long() + 2).clamp(0, 6), 7).float(), touched], -1)
        h = self.encoder(x, cur, b["mask_a"])
        same_fragment = (b["frag_a"][:, :, None] == b["frag_a"][:, None, :]).float()[..., None]
        z = [h[:, :, None] + h[:, None, :], h[:, :, None] * h[:, None, :], F.one_hot(cur, N_BOND).float(), same_fragment, fired.float()[..., None]]
        if self.hops:
            z.append(F.one_hot(self.distance(cur), self.hops + 1).float())
        z = torch.cat(z, -1)
        logits = self.pair(z)
        n = cur.shape[1]
        gain = self.order[None, None, None, :] - self.order[cur][..., None]  # tokens the firing adds to B_ij
        room = hydrogens + b["cap_a"] + 0.5  # half a token of slack: aromatic bonds count 1.5
        enabled = (gain <= room[:, :, None, None]) & (gain <= room[:, None, :, None]) if self.enabling else torch.ones_like(gain, dtype=torch.bool)
        valid = (b["mask_a"][:, :, None] & b["mask_a"][:, None, :] & ~fired
                 & torch.triu(torch.ones(n, n, dtype=torch.bool, device=cur.device), 1))[..., None] & ~F.one_hot(cur, N_BOND).bool()
        stop = self.stop((h * b["mask_a"][..., None]).sum(1)).squeeze(-1)
        return logits.masked_fill(~valid, -1e4), enabled, stop

    def distance(self, cur):
        """Number of marked bond places on the shortest path between two atoms, 1..hops (0: farther apart or not connected)."""
        adj = (cur > 0).float()
        reach, dist = adj.clone(), (cur > 0).long()
        for k in range(2, self.hops + 1):
            reach = ((reach @ adj) > 0).float()
            dist = torch.where((dist == 0) & (reach > 0), torch.full_like(dist, k), dist)
        eye = torch.eye(cur.shape[1], dtype=torch.bool, device=cur.device)
        return dist.masked_fill(eye, 0)

    def loss(self, b):
        true = b["edits"] > 0
        keep = torch.rand(len(true), 1, 1, device=true.device)  # a random part of the true firing vector has already fired
        draw = torch.rand(true.shape, device=true.device); draw = torch.triu(draw, 1); draw = draw + draw.transpose(1, 2)
        fired = true & (draw < keep)
        cur = torch.where(fired, b["edits"] - 1, b["ba"])
        logits, enabled, stop = self.rates(b, cur, fired)
        remaining = F.one_hot((b["edits"] - 1).clamp(min=0), N_BOND).bool() & (true & ~fired)[..., None] & (logits > -1e3)
        target = remaining & enabled
        target = torch.where(target.flatten(1).any(1)[:, None, None, None], target, remaining)  # no enabled one left: allow any
        logits = logits.masked_fill(~enabled & ~target, -1e4)
        log_z = torch.logsumexp(torch.cat([logits.flatten(1), stop[:, None]], 1), 1)
        hit = torch.logsumexp(logits.masked_fill(~target, -1e4).flatten(1), 1)
        done = ~remaining.flatten(1).any(1)
        return (log_z - torch.where(done, stop, hit)).mean()

    @torch.no_grad()
    def forward(self, b):
        """Greedy token game; returns the final marking as `edits` [B, N, N] (0 = unchanged, k + 1 = new type k)."""
        cur, fired = b["ba"].clone(), torch.zeros_like(b["ba"], dtype=torch.bool)
        active = torch.ones(len(cur), dtype=torch.bool, device=cur.device)
        n = cur.shape[1]
        for _ in range(self.max_steps):
            logits, enabled, stop = self.rates(b, cur, fired)
            logits = logits.masked_fill(~enabled, -1e4)
            best, where = logits.flatten(1).max(1)
            fire = active & (best > stop)
            if not fire.any():
                break
            i, j, k = where // (n * N_BOND), (where // N_BOND) % n, where % N_BOND
            rows = torch.nonzero(fire).squeeze(1)
            cur[rows, i[rows], j[rows]] = k[rows]; cur[rows, j[rows], i[rows]] = k[rows]
            fired[rows, i[rows], j[rows]] = True; fired[rows, j[rows], i[rows]] = True
            active = fire
        return torch.where(fired, cur + 1, torch.zeros_like(cur))

    @torch.no_grad()
    def beam_search(self, b, width=5):
        """Most probable final markings of ONE reaction (batch of size 1). Firing sequences that differ only in
        the order of their transitions are the same trace and reach the same marking: their probabilities add."""
        n = b["ba"].shape[1]
        beams, finished = {frozenset(): (0.0, b["ba"].clone(), torch.zeros_like(b["ba"], dtype=torch.bool))}, {}
        for _ in range(self.max_steps):
            if not beams:
                break
            keys = list(beams)
            cur = torch.cat([beams[k][1] for k in keys]); fired = torch.cat([beams[k][2] for k in keys])
            rep = {k: v.expand(len(keys), *v.shape[1:]) for k, v in b.items() if torch.is_tensor(v)}
            logits, enabled, stop = self.rates(rep, cur, fired)
            logp = torch.log_softmax(torch.cat([logits.masked_fill(~enabled, -1e4).flatten(1), stop[:, None]], 1), 1)
            top_lp, top_ix = logp.topk(width, 1)
            grown = {}
            for r, key in enumerate(keys):
                for lp, ix in zip(top_lp[r].tolist(), top_ix[r].tolist()):
                    total = beams[key][0] + lp
                    if ix == logp.shape[1] - 1:  # STOP
                        finished[key] = float(np.logaddexp(finished.get(key, -np.inf), total))
                        continue
                    i, j, k = ix // (n * N_BOND), (ix // N_BOND) % n, ix % N_BOND
                    new_key = key | {(i, j, k)}
                    if new_key in grown:
                        grown[new_key] = (float(np.logaddexp(grown[new_key][0], total)), *grown[new_key][1:])
                    else:
                        c, f = cur[r:r + 1].clone(), fired[r:r + 1].clone()
                        c[0, i, j] = c[0, j, i] = k; f[0, i, j] = f[0, j, i] = True
                        grown[new_key] = (total, c, f)
            beams = dict(sorted(grown.items(), key=lambda kv: -kv[1][0])[:width])
        ranked = sorted(finished.items(), key=lambda kv: -kv[1])[:width]
        return [(np.array(sorted(key), dtype=np.int64).reshape(-1, 3), lp) for key, lp in ranked]

    @staticmethod
    def decode(edits, b, reactions):
        out = []
        for e, r in zip(edits.cpu().numpy(), reactions):
            n = len(r["a"]["x"])
            i, j = np.nonzero(np.triu(e[:n, :n], 1))
            out.append(np.stack([i, j, e[i, j] - 1], 1))
        return out


class Forward(nn.Module):
    """One-shot counterpart (generic readout): every bond place is labelled independently."""

    def __init__(self, d=128, rounds=5, attention=3, petri=True):
        super().__init__()
        self.petri = petri
        self.encoder = Encoder(d, rounds, attention)
        self.pair = mlp(2 * d + N_BOND + 1, 2 * d, N_BOND + 1)  # log-rates: unchanged, or re-type to none/1/2/3/aromatic

    def forward(self, b):
        h = self.encoder(b["xa"], b["ba"], b["mask_a"])
        same_fragment = (b["frag_a"][:, :, None] == b["frag_a"][:, None, :]).float()[..., None]
        z = torch.cat([h[:, :, None] + h[:, None, :], h[:, :, None] * h[:, None, :], F.one_hot(b["ba"], N_BOND).float(), same_fragment], -1)
        logits = self.pair(z)
        if self.petri:  # enabling: a transition that would leave the marking of B_ij unchanged does not exist
            logits = logits.masked_fill(F.one_hot(b["ba"] + 1, N_BOND + 1).bool(), -1e4)
        return logits

    @staticmethod
    def loss(logits, b):
        n = logits.shape[1]
        valid = (b["mask_a"][:, :, None] & b["mask_a"][:, None, :]) & torch.triu(torch.ones(n, n, dtype=torch.bool, device=logits.device), 1)
        return F.cross_entropy(logits[valid], b["edits"][valid])

    def decode(self, logits, b, reactions):
        """Most probable firing vector. With Petri semantics it must be enabled: the hydrogen place of every
        atom stays non-negative, h_i - sum_j (order'(ij) - order(ij)) >= 0. A violated atom either drops its
        least likely bond-forming firing or fires its most likely bond-breaking transition, whichever is cheaper."""
        logp = torch.log_softmax(logits.float(), -1).cpu().numpy()
        out = []
        for k, r in enumerate(reactions):
            n = len(r["a"]["x"])
            lp, before = logp[k, :n, :n], np.zeros((n, n), np.int64)
            if len(r["a"]["bonds"]):
                i, j, t = r["a"]["bonds"].T
                before[i, j] = t; before[j, i] = t
            choice = lp.argmax(-1)
            choice = np.triu(choice, 1); choice = choice + choice.T  # one decision per unordered pair
            if self.petri:
                for _ in range(8):
                    after = np.where(choice > 0, choice - 1, before)
                    hydrogens = r["a"]["h"] - (BOND_ORDER[after] - BOND_ORDER[before]).sum(1)
                    bad = np.nonzero(hydrogens < -1e-6)[0]
                    if not len(bad):
                        break
                    i = bad[np.argmin(hydrogens[bad])]
                    gain = BOND_ORDER[after[i]] - BOND_ORDER[before[i]]
                    options = []  # (cost in log-probability, atom j, new choice)
                    for j in np.nonzero(gain > 0)[0]:  # undo a firing that adds tokens to B_ij
                        options.append((lp[i, j, choice[i, j]] - lp[i, j, 0], j, 0))
                    for j in np.nonzero((before[i] > 0) & (choice[i] == 0))[0]:  # fire a transition that removes tokens
                        lower = [c for c in range(1, N_BOND + 1) if BOND_ORDER[c - 1] < BOND_ORDER[before[i, j]]]
                        c = max(lower, key=lambda c: lp[i, j, c])
                        options.append((lp[i, j, 0] - lp[i, j, c], j, c))
                    if not options:
                        break
                    _, j, c = min(options, key=lambda o: o[0])
                    choice[i, j] = choice[j, i] = c
            i, j = np.nonzero(np.triu(choice, 1))
            out.append(np.stack([i, j, choice[i, j] - 1], 1))
        return out
