"""Forward reaction prediction as a firing sequence of the valence net."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import N_BOND, Encoder, mlp
from .featurisation import BOND_ORDER


class TokenGame(nn.Module):
    """Embedded jump chain of a stochastic net with a learned rate law.

    A transition re-types one bond place B_ij. It moves tokens between B_ij and the slack places S_i and S_j, so the
    valence of every atom is a P-invariant. enabling False is the ablation without the valence capacities.
    hops above zero adds the distance of i and j in the current marking as a pair feature.
    """

    def __init__(self, d=128, rounds=6, attention=4, max_steps=12, enabling=True, hops=0, composites=0):
        super().__init__()
        self.max_steps, self.enabling = max_steps, enabling
        self.hops = hops

        # a substitution fires a break on B_ij and a formation on B_jk at once. the tokens the break leaves on the
        # slack place of the shared atom j are the ones the formation takes, so the composite column is
        # delta (e_Bjk - e_Bij - e_Si + e_Sk) and the central atom needs no free valence. every composite is an
        # enabled sequence of the elementary net, so it adds no reachable marking and validity carries over
        self.composites = composites
        self.encoder = Encoder(d, rounds, attention, n_extra=8)
        self.pair = mlp(2 * d + N_BOND + 2 + (hops + 1 if hops else 0), 2 * d, N_BOND)

        if composites:
            self.centre = mlp(3 * d + 2 * N_BOND, 2 * d, 1)

        self.stop = mlp(d, d, 1)

        # a constant added to the log rate of STOP when decoding, fitted on validation reactions. it is part of
        # the rate law, so the model stays the same stochastic net, and it is not a trained parameter
        self.stop_bias = 0.0
        self.register_buffer("order", torch.tensor(BOND_ORDER, dtype=torch.float32))

    def rates(self, b, cur, fired, extra=False):
        """Log rates [B, N, N, types] of all transitions, the enabling mask and the log rate of STOP [B]."""
        # marking of the slack places. every token that entered a bond place of atom i left S_i
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

        # enabling is the valence rule. a transition that adds gain tokens to B_ij takes them from S_i and S_j, so it
        # needs gain <= s_i and gain <= s_j. every reachable marking is then non-negative for all weights
        # aromatic bonds count 1.5, so markings are multiples of one half and the capacity gets half a token of slack
        gain = self.order[None, None, None, :] - self.order[cur][..., None]
        room = hydrogens + b["cap_a"] + 0.5
        enabled = (gain <= room[:, :, None, None]) & (gain <= room[:, None, :, None]) if self.enabling else torch.ones_like(gain, dtype=torch.bool)
        valid = (b["mask_a"][:, :, None] & b["mask_a"][:, None, :] & ~fired
                 & torch.triu(torch.ones(n, n, dtype=torch.bool, device=cur.device), 1))[..., None] & ~F.one_hot(cur, N_BOND).bool()
        stop = self.stop((h * b["mask_a"][..., None]).sum(1)).squeeze(-1) + (0.0 if self.training else self.stop_bias)

        if extra:
            return logits.masked_fill(~valid, -1e4), enabled, stop, h, room

        return logits.masked_fill(~valid, -1e4), enabled, stop

    def substitutions(self, b, cur, fired, logits, enabled, h, room, width):
        """Rates of the substitutions built from the most probable breaks and every formation at the shared atom.

        A substitution breaks B_ij and forms B_jk. The break is enabled whenever the place holds tokens, and the
        formation needs room on S_j and S_k, but the break has just put its tokens on S_j, so S_j only has to hold
        what the two firings do not cancel. Returns the log rates [B, width, N, types] and the breaks (i, j).
        """
        n = cur.shape[1]
        masked = logits.masked_fill(~enabled, -1e4)

        # breaks, a marked bond place re-typed to order zero
        break_score = masked[..., 0].masked_fill(cur == 0, -1e4)
        best = break_score.flatten(1).topk(min(width, n * n), 1)
        bi, bj = best.indices // n, best.indices % n
        rows = torch.arange(len(cur), device=cur.device)[:, None]
        broken = self.order[cur[rows, bi, bj]][..., None, None]

        # formations on any pair that shares the atom j of the break, to any type the elementary rate allows
        form = masked[rows, bj]
        gain = self.order[None, None, None, :] - self.order[cur[rows, bj]][..., None]
        allowed = (gain <= room[:, None, :, None]) & (gain <= room[rows, bj][..., None, None] + broken)

        if not self.enabling:
            allowed = torch.ones_like(gain, dtype=torch.bool)

        centre = torch.cat([h[rows, bi][:, :, None].expand(-1, -1, n, -1),
                            h[rows, bj][:, :, None].expand(-1, -1, n, -1),
                            h[:, None].expand(-1, bi.shape[1], -1, -1),
                            F.one_hot(cur[rows, bi, bj], N_BOND).float()[:, :, None].expand(-1, -1, n, -1),
                            F.one_hot(cur[rows, bj], N_BOND).float()], -1)

        # the composite carries its own rate, it is not the product of the two halves
        score = form + self.centre(centre)
        atoms = torch.arange(n, device=cur.device)
        same = (bj[..., None] == atoms) | (bi[..., None] == atoms)
        bad = ~allowed | (form <= -1e3) | same[..., None] | (best.values <= -1e3)[..., None, None]

        return score.masked_fill(bad, -1e4), bi, bj

    def events(self, b, cur, fired):
        """Log rates of every event at a marking, as one flat vector per reaction.

        An event is an elementary firing, a substitution that fires two at once, or STOP. The layout is
        [N * N * types elementary, width * N * types substitutions, one STOP], so an index decodes to the firings
        it performs.
        """
        if not self.composites:
            logits, enabled, stop = self.rates(b, cur, fired)
            flat = torch.cat([logits.masked_fill(~enabled, -1e4).flatten(1), stop[:, None]], 1)
            return flat, logits, enabled, None, None, None

        logits, enabled, stop, h, room = self.rates(b, cur, fired, extra=True)
        sub, bi, bj = self.substitutions(b, cur, fired, logits, enabled, h, room, self.composites)
        flat = torch.cat([logits.masked_fill(~enabled, -1e4).flatten(1), sub.flatten(1), stop[:, None]], 1)

        return flat, logits, enabled, sub, bi, bj

    def apply_event(self, index, cur, fired, bi, bj):
        """The firings that an event index performs, as a list of (i, j, type) per reaction."""
        n = cur.shape[1]
        elementary = n * n * N_BOND
        out = []
        for r, ix in enumerate(index.tolist()):
            if ix < elementary:
                out.append([(ix // (n * N_BOND), (ix // N_BOND) % n, ix % N_BOND)])
            elif bi is not None and ix < elementary + bi.shape[1] * n * N_BOND:
                c = ix - elementary
                w, k, t = c // (n * N_BOND), (c // N_BOND) % n, c % N_BOND
                i, j = int(bi[r, w]), int(bj[r, w])
                out.append([(i, j, 0), (j, k, t)])
            else:
                out.append([])

        return out

    def distance(self, cur):
        """Number of marked bond places on the shortest path between two atoms, from 1 to hops. 0 if farther apart or not connected."""
        adj = (cur > 0).float()
        reach, dist = adj.clone(), (cur > 0).long()
        for k in range(2, self.hops + 1):
            reach = ((reach @ adj) > 0).float()
            dist = torch.where((dist == 0) & (reach > 0), torch.full_like(dist, k), dist)

        eye = torch.eye(cur.shape[1], dtype=torch.bool, device=cur.device)

        return dist.masked_fill(eye, 0)

    def loss(self, b):
        # the record gives a firing vector but no order, and sequences with the same firing vector reach the same
        # marking. so a random part of the firing vector is fired and any remaining enabled firing is a correct next
        # step. the loss is -log sum_t P(t | m_A + C sigma_F) over these firings, or -log P(STOP) if none remains
        true = b["edits"] > 0
        keep = torch.rand(len(true), 1, 1, device=true.device)
        draw = torch.rand(true.shape, device=true.device)
        draw = torch.triu(draw, 1)
        draw = draw + draw.transpose(1, 2)
        fired = true & (draw < keep)
        cur = torch.where(fired, b["edits"] - 1, b["ba"])
        flat, logits, enabled, sub, bi, bj = self.events(b, cur, fired)
        n = cur.shape[1]
        remaining = F.one_hot((b["edits"] - 1).clamp(min=0), N_BOND).bool() & (true & ~fired)[..., None] & (logits > -1e3)
        target = remaining & enabled

        # a noisy record can leave no enabled firing, then any remaining one is accepted
        target = torch.where(target.flatten(1).any(1)[:, None, None, None], target, remaining)
        hit = [logits.masked_fill(~enabled & ~target, -1e4).masked_fill(~target, -1e4).flatten(1)]

        if sub is not None:
            # a substitution is a correct next event when both of its firings are still to come
            rows = torch.arange(len(cur), device=cur.device)[:, None]
            breaks = remaining[rows, bi, bj, 0]
            forms = remaining[rows, bj]
            hit.append(sub.masked_fill(~(forms & breaks[..., None, None]), -1e4).flatten(1))

        done = ~remaining.flatten(1).any(1)
        log_z = torch.logsumexp(flat, 1)

        return (log_z - torch.where(done, flat[:, -1], torch.logsumexp(torch.cat(hit, 1), 1))).mean()

    @torch.no_grad()
    def forward(self, b):
        """Greedy token game. Returns the final marking as edits [B, N, N], 0 for unchanged and k + 1 for new type k."""
        cur, fired = b["ba"].clone(), torch.zeros_like(b["ba"], dtype=torch.bool)
        active = torch.ones(len(cur), dtype=torch.bool, device=cur.device)
        for _ in range(self.max_steps):
            flat, _, _, _, bi, bj = self.events(b, cur, fired)
            best, where = flat[:, :-1].max(1)
            fire = active & (best > flat[:, -1])
            if not fire.any():
                break

            for r, firings in enumerate(self.apply_event(where, cur, fired, bi, bj)):
                if not fire[r]:
                    continue

                for i, j, k in firings:
                    cur[r, i, j] = cur[r, j, i] = k
                    fired[r, i, j] = fired[r, j, i] = True

            active = fire

        return torch.where(fired, cur + 1, torch.zeros_like(cur))

    @torch.no_grad()
    def beam_search(self, b, width=5):
        """Most probable final markings of one reaction, batch of size 1."""
        # hypotheses are keyed by their firing vector. sequences that differ only in the order of their firings
        # reach the same marking by the state equation, so they merge and their probabilities add
        n = b["ba"].shape[1]
        beams, finished = {frozenset(): (0.0, b["ba"].clone(), torch.zeros_like(b["ba"], dtype=torch.bool))}, {}
        for _ in range(self.max_steps):
            if not beams:
                break

            keys = list(beams)
            cur = torch.cat([beams[k][1] for k in keys])
            fired = torch.cat([beams[k][2] for k in keys])
            rep = {k: v.expand(len(keys), *v.shape[1:]) for k, v in b.items() if torch.is_tensor(v)}
            flat, _, _, _, bi, bj = self.events(rep, cur, fired)
            logp = torch.log_softmax(flat, 1)
            top_lp, top_ix = logp.topk(width, 1)
            grown = {}
            for r, key in enumerate(keys):
                firings = self.apply_event(top_ix[r], cur[r:r + 1].expand(width, -1, -1), fired, bi[r:r + 1].expand(width, -1) if bi is not None else None,
                                           bj[r:r + 1].expand(width, -1) if bj is not None else None)
                for lp, event in zip(top_lp[r].tolist(), firings):
                    total = beams[key][0] + lp

                    # an event with no firing is STOP
                    if not event:
                        finished[key] = float(np.logaddexp(finished.get(key, -np.inf), total))
                        continue

                    new_key = key | set(event)
                    if new_key in grown:
                        grown[new_key] = (float(np.logaddexp(grown[new_key][0], total)), *grown[new_key][1:])
                    else:
                        c, f = cur[r:r + 1].clone(), fired[r:r + 1].clone()
                        for i, j, k in event:
                            c[0, i, j] = c[0, j, i] = k
                            f[0, i, j] = f[0, j, i] = True

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
