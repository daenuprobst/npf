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
    """

    def __init__(self, d=128, rounds=6, attention=4, max_steps=12, enabling=True):
        super().__init__()
        self.max_steps, self.enabling = max_steps, enabling
        self.encoder = Encoder(d, rounds, attention, n_extra=8)
        self.pair = mlp(2 * d + N_BOND + 2, 2 * d, N_BOND)
        self.stop = mlp(d, d, 1)

        # a constant on the log rate of STOP at decoding, zero in every reported run, part of the rate law, not trained
        self.stop_bias = 0.0
        self.register_buffer("order", torch.tensor(BOND_ORDER, dtype=torch.float32))

    def rates(self, b, cur, fired):
        """Log rates [B, N, N, types] of all transitions, the enabling mask and the log rate of STOP [B]."""
        # marking of the slack places. every token that entered a bond place of atom i left S_i
        hydrogens = b["h_a"] - (self.order[cur] - self.order[b["ba"]]).sum(2)
        touched = fired.any(2, keepdim=True).float()
        x = torch.cat(
            [
                b["xa"],
                F.one_hot((hydrogens.round().long() + 2).clamp(0, 6), 7).float(),
                touched,
            ],
            -1,
        )
        h = self.encoder(x, cur, b["mask_a"])
        same_fragment = (b["frag_a"][:, :, None] == b["frag_a"][:, None, :]).float()[
            ..., None
        ]
        z = [
            h[:, :, None] + h[:, None, :],
            h[:, :, None] * h[:, None, :],
            F.one_hot(cur, N_BOND).float(),
            same_fragment,
            fired.float()[..., None],
        ]

        z = torch.cat(z, -1)
        logits = self.pair(z)
        n = cur.shape[1]

        # enabling is the valence rule, a transition that adds gain tokens to B_ij takes them from S_i and S_j, so it
        # needs gain <= s_i and gain <= s_j, and aromatic bonds count 1.5, so the capacity gets half a token of slack
        gain = self.order[None, None, None, :] - self.order[cur][..., None]
        room = hydrogens + b["cap_a"] + 0.5
        enabled = (
            (gain <= room[:, :, None, None]) & (gain <= room[:, None, :, None])
            if self.enabling
            else torch.ones_like(gain, dtype=torch.bool)
        )
        valid = (
            b["mask_a"][:, :, None]
            & b["mask_a"][:, None, :]
            & ~fired
            & torch.triu(torch.ones(n, n, dtype=torch.bool, device=cur.device), 1)
        )[..., None] & ~F.one_hot(cur, N_BOND).bool()
        stop = self.stop((h * b["mask_a"][..., None]).sum(1)).squeeze(-1) + (
            0.0 if self.training else self.stop_bias
        )

        return logits.masked_fill(~valid, -1e4), enabled, stop

    def events(self, b, cur, fired):
        """Log rates of every event at a marking, as one flat vector per reaction, [N * N * types firings, STOP]."""
        logits, enabled, stop = self.rates(b, cur, fired)
        flat = torch.cat(
            [logits.masked_fill(~enabled, -1e4).flatten(1), stop[:, None]], 1
        )

        return flat, logits, enabled

    @staticmethod
    def apply_event(index, n):
        """The firing (i, j, type) of every event index, as a list per reaction, empty for STOP."""
        return [
            (
                [(ix // (n * N_BOND), (ix // N_BOND) % n, ix % N_BOND)]
                if ix < n * n * N_BOND
                else []
            )
            for ix in index.tolist()
        ]

    @staticmethod
    def continuations(options, real, edits, fired, allowed):
        """The firings still to come [B, K, N, N, types] of every vector of a set [B, K, N, N] that contains the fired
        part of edits, and which vectors those are [B, K]. real marks the vectors that are not padding.
        """
        # a vector continues the fired part when it fires every fired place the same way
        agree = real & ((options == edits[:, None]) | ~fired[:, None]).flatten(2).all(2)
        live = (options > 0) & ~fired[:, None] & agree[..., None, None]

        return (
            torch.stack(
                [
                    (options == t + 1) & live & allowed[:, None, ..., t]
                    for t in range(N_BOND)
                ],
                -1,
            ),
            agree,
        )

    def loss(self, b):
        # the record gives a firing vector but no order, so a random part of it is fired and any remaining enabled
        # firing is correct, the loss is -log sum_t P(t | m_A + C sigma_F) or -log P(STOP) if none remains. with a set
        # of vectors one is drawn, a firing is correct when it continues a vector that contains the fired part
        edits, options = b["edits"], b.get("edits_set")
        keep = torch.rand(len(edits), 1, 1, device=edits.device)
        draw = torch.rand(edits.shape, device=edits.device)
        draw = torch.triu(draw, 1)
        draw = draw + draw.transpose(1, 2)

        if options is None:
            options, real = edits[:, None], torch.ones(
                len(edits), 1, dtype=torch.bool, device=edits.device
            )
        else:
            rows = torch.arange(len(edits), device=edits.device)
            edits = options[
                rows, (torch.rand(len(edits), device=edits.device) * b["n_set"]).long()
            ].long()
            real = (
                torch.arange(options.shape[1], device=edits.device)
                < b["n_set"][:, None]
            )

        true = edits > 0
        fired = true & (draw < keep)
        cur = torch.where(fired, edits - 1, b["ba"])
        flat, logits, enabled = self.events(b, cur, fired)
        live, agree = self.continuations(options, real, edits, fired, logits > -1e3)
        remaining = live.any(1)
        target = remaining & enabled
        hit = [logits.masked_fill(~target, -1e4).flatten(1)]
        found = target.flatten(1).any(1)

        # STOP is correct when a vector of the set has no firing left
        done = (agree & ~live.flatten(2).any(2)).any(1)
        hit.append(flat[:, -1:].masked_fill(~done[:, None], -1e4))
        log_z = torch.logsumexp(flat, 1)
        per_state = log_z - torch.logsumexp(torch.cat(hit, 1), 1)

        # a state with no enabled firing left has no correct next event and is left out, else the loss is unbounded
        usable = done | found

        return (per_state * usable).sum() / usable.sum().clamp(min=1)

    @torch.no_grad()
    def forward(self, b):
        """Greedy token game. Returns the final marking as edits [B, N, N], 0 for unchanged and k + 1 for new type k."""
        cur, fired = b["ba"].clone(), torch.zeros_like(b["ba"], dtype=torch.bool)
        active = torch.ones(len(cur), dtype=torch.bool, device=cur.device)
        for _ in range(self.max_steps):
            flat, _, _ = self.events(b, cur, fired)
            best, where = flat[:, :-1].max(1)
            fire = active & (best > flat[:, -1])
            if not fire.any():
                break

            for r, firings in enumerate(self.apply_event(where, cur.shape[1])):
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
        # hypotheses are keyed by their firing vector, orders of the same firings reach the same marking and merge
        n = b["ba"].shape[1]
        beams, finished = {
            frozenset(): (
                0.0,
                b["ba"].clone(),
                torch.zeros_like(b["ba"], dtype=torch.bool),
            )
        }, {}
        for _ in range(self.max_steps):
            if not beams:
                break

            keys = list(beams)
            cur = torch.cat([beams[k][1] for k in keys])
            fired = torch.cat([beams[k][2] for k in keys])
            rep = {
                k: v.expand(len(keys), *v.shape[1:])
                for k, v in b.items()
                if torch.is_tensor(v)
            }
            flat, _, _ = self.events(rep, cur, fired)
            logp = torch.log_softmax(flat, 1)
            top_lp, top_ix = logp.topk(width, 1)
            grown = {}
            for r, key in enumerate(keys):
                firings = self.apply_event(top_ix[r], n)
                for lp, event in zip(top_lp[r].tolist(), firings):
                    # a disabled event has probability zero and enters the top k only when fewer than k are enabled
                    if lp < -1e3:
                        continue

                    total = beams[key][0] + lp

                    # an event with no firing is STOP
                    if not event:
                        finished[key] = float(
                            np.logaddexp(finished.get(key, -np.inf), total)
                        )
                        continue

                    new_key = key | set(event)
                    if new_key in grown:
                        grown[new_key] = (
                            float(np.logaddexp(grown[new_key][0], total)),
                            *grown[new_key][1:],
                        )
                    else:
                        c, f = cur[r : r + 1].clone(), fired[r : r + 1].clone()
                        for i, j, k in event:
                            c[0, i, j] = c[0, j, i] = k
                            f[0, i, j] = f[0, j, i] = True

                        grown[new_key] = (total, c, f)

            beams = dict(sorted(grown.items(), key=lambda kv: -kv[1][0])[:width])

        ranked = sorted(finished.items(), key=lambda kv: -kv[1])[:width]

        return [
            (np.array(sorted(key), dtype=np.int64).reshape(-1, 3), lp)
            for key, lp in ranked
        ]

    @torch.no_grad()
    def beam_search_batch(self, b, width=5):
        """beam_search for a batch of reactions at once, one ranked list per reaction. The hypotheses of all reactions
        share one evaluation of the rate law per step, and their markings stay on the host, so extending a hypothesis
        launches no device work."""
        device, batch = b["ba"].device, b["ba"].shape[0]
        start = b["ba"].cpu().numpy()
        untouched = np.zeros(start.shape[1:], bool)
        beams = [{frozenset(): (0.0, start[r], untouched)} for r in range(batch)]
        finished = [{} for _ in range(batch)]
        static = {
            k: v
            for k, v in b.items()
            if torch.is_tensor(v) and v.dim() and v.shape[0] == batch
        }
        for _ in range(self.max_steps):
            rows = [(r, key) for r in range(batch) for key in beams[r]]
            if not rows:
                break

            item = torch.tensor([r for r, _ in rows], device=device)
            rep = {k: v[item] for k, v in static.items()}
            cur = torch.from_numpy(np.stack([beams[r][key][1] for r, key in rows])).to(
                device
            )
            fired = torch.from_numpy(
                np.stack([beams[r][key][2] for r, key in rows])
            ).to(device)
            flat, _, _ = self.events(rep, cur, fired)
            top_lp, top_ix = torch.log_softmax(flat, 1).topk(width, 1)
            top_lp, top_ix = top_lp.cpu(), top_ix.cpu()
            grown = [{} for _ in range(batch)]
            for h, (r, key) in enumerate(rows):
                base, c0, f0 = beams[r][key]
                firings = self.apply_event(top_ix[h], cur.shape[1])
                for lp, event in zip(top_lp[h].tolist(), firings):
                    # a disabled event has probability zero and enters the top k only when fewer than k are enabled
                    if lp < -1e3:
                        continue

                    total = base + lp

                    # an event with no firing is STOP
                    if not event:
                        finished[r][key] = float(
                            np.logaddexp(finished[r].get(key, -np.inf), total)
                        )
                        continue

                    new_key = key | set(event)
                    if new_key in grown[r]:
                        grown[r][new_key] = (
                            float(np.logaddexp(grown[r][new_key][0], total)),
                        ) + grown[r][new_key][1:]
                    else:
                        c, f = c0.copy(), f0.copy()
                        for i, j, k in event:
                            c[i, j] = c[j, i] = k
                            f[i, j] = f[j, i] = True

                        grown[r][new_key] = (total, c, f)

            beams = [
                dict(sorted(g.items(), key=lambda kv: -kv[1][0])[:width]) for g in grown
            ]

        out = []
        for r in range(batch):
            ranked = sorted(finished[r].items(), key=lambda kv: -kv[1])[:width]
            out.append(
                [
                    (np.array(sorted(key), dtype=np.int64).reshape(-1, 3), lp)
                    for key, lp in ranked
                ]
            )

        return out

    @staticmethod
    def decode(edits, b, reactions):
        out = []
        for e, r in zip(edits.cpu().numpy(), reactions):
            n = len(r["a"]["x"])
            i, j = np.nonzero(np.triu(e[:n, :n], 1))
            out.append(np.stack([i, j, e[i, j] - 1], 1))

        return out
