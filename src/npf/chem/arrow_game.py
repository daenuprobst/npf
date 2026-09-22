"""Elementary mechanistic steps as firing sequences of the arrow net (arrows.py)."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import PetriLayer, mlp

N_ELEMENT = 90
N_ORDER = 4
MAX_ARROWS = 16


class ArrowGame(nn.Module):
    """Embedded jump chain of a stochastic net on the arrow net, with a learned rate law.

    A state is the marking of one elementary step in progress, bond orders cur [B, N, N], lone pairs [B, N] and
    charges [B, N], with the arrows fired so far in fired_a and fired_b [B, N, N]. The events are the arrows a(i, j)
    and b(i, j) and STOP, which ends the step. enabling False is the ablation without octet capacities and charge
    windows, where only lone pairs and bond pairs that exist can move.
    """

    def __init__(self, d=256, rounds=6, attention=6, pair=64, enabling=True, slack=1):
        super().__init__()
        self.enabling, self.slack = enabling, slack
        self.element = nn.Embedding(N_ELEMENT, d)
        self.inp = nn.Linear(d + 19, d)
        self.layers = nn.ModuleList(PetriLayer(d) for _ in range(rounds))
        self.attn = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d, 8, 2 * d, dropout=0.0, batch_first=True, norm_first=True
            )
            for _ in range(attention)
        )

        # additive pair scores, one per arrow kind, from the two atoms and the bond place between them
        self.src, self.dst = nn.Linear(d, 2 * pair), nn.Linear(d, 2 * pair)
        self.place = nn.Embedding(N_ORDER * 4, 2 * pair)
        self.out = nn.Parameter(torch.randn(2, pair) / pair**0.5)
        self.bias = nn.Parameter(torch.zeros(2))
        self.count = nn.Embedding(MAX_ARROWS + 1, d)
        self.stop = mlp(d, d, 1)

    def free(self, s):
        """Free octet slots V_i, what the capacity leaves after lone pairs, bonds and an unpaired electron."""
        return s["cap"] - (s["lone"] + s["odd"] + s["cur"].sum(2))

    def encode(self, s):
        mask = s["mask"]
        x = torch.cat(
            [
                self.element(s["z"].clamp(max=N_ELEMENT - 1)),
                F.one_hot((s["q"] + 3).clamp(0, 6), 7).float(),
                F.one_hot(s["lone"].clamp(0, 4), 5).float(),
                F.one_hot(self.free(s).clamp(0, 4), 5).float(),
                s["odd"][..., None].float(),
                s["touched"][..., None].float(),
            ],
            -1,
        )
        h = self.inp(x) * mask[..., None]
        adj = F.one_hot(s["cur"].clamp(0, 4), 5)[..., 1:].to(h.dtype)
        for k, layer in enumerate(self.layers):
            h = layer(h, adj) * mask[..., None]

            if k < len(self.attn):
                h = self.attn[k](h, src_key_padding_mask=~mask) * mask[..., None]

        return h

    def enabled(self, s):
        """The Petri part. a(i, j) needs a lone pair on i, a free octet slot on j and room on the bond place, b(i, j)
        needs a pair on the bond ij, and both keep every charge within its window widened by slack. An arrow that would
        undo one fired in this step is not enabled. Returns the mask [B, N, N, 2] of the arrows and [B] of STOP.
        """
        lone, q, cur, mask = s["lone"], s["q"], s["cur"], s["mask"]
        n = cur.shape[1]
        pairs = (
            mask[:, :, None]
            & mask[:, None, :]
            & ~torch.eye(n, dtype=torch.bool, device=cur.device)
        )
        room = (s["n_fired"] < MAX_ARROWS)[:, None, None]
        a = (lone >= 1)[:, :, None] & (cur < N_ORDER - 1) & (s["fired_b"] == 0)
        b = (cur >= 1) & (s["fired_a"] == 0)

        # the octet rule is that the target of a has a free slot, the charge windows bound both ends of both kinds
        if self.enabling:
            lo, hi = s["lo"] - self.slack, s["hi"] + self.slack
            a = (
                a
                & (self.free(s) >= 1)[:, None, :]
                & (q + 1 <= hi)[:, :, None]
                & (q - 1 >= lo)[:, None, :]
            )
            b = b & (q - 1 >= lo)[:, :, None] & (q + 1 <= hi)[:, None, :]

        stop = (
            ((q >= s["lo"]) & (q <= s["hi"]) | ~mask).all(1)
            if self.enabling
            else torch.ones_like(mask[:, 0])
        )

        return torch.stack([a & pairs & room, b & pairs & room], -1), stop

    def rates(self, s):
        """Log rates of all arrows [B, N, N, 2] and of STOP [B], with their enabling masks."""
        h = self.encode(s)
        batch, n, _ = h.shape
        p = self.out.shape[1]
        u = self.src(h).view(batch, n, 1, 2, p)
        v = self.dst(h).view(batch, 1, n, 2, p)
        flags = (
            s["cur"].clamp(0, N_ORDER - 1) * 4
            + (s["fired_a"] > 0).long() * 2
            + (s["fired_b"] > 0).long()
        )
        z = F.silu(u + v + self.place(flags).view(batch, n, n, 2, p))
        logits = (z * self.out).sum(-1) + self.bias
        pooled = (h * s["mask"][..., None]).sum(1) + self.count(
            s["n_fired"].clamp(max=MAX_ARROWS)
        )
        enabled, stop_ok = self.enabled(s)

        return logits.float(), enabled, self.stop(pooled).squeeze(-1).float(), stop_ok

    def events(self, s):
        """Log rates of every event as one flat vector per state, [N * N arrows a, N * N arrows b, STOP]."""
        logits, enabled, stop, stop_ok = self.rates(s)
        arrows = logits.masked_fill(~enabled, -1e4).permute(0, 3, 1, 2).flatten(1)

        return torch.cat([arrows, stop.masked_fill(~stop_ok, -1e4)[:, None]], 1)

    def loss(self, s):
        # the record gives the arrows of a step but no order, and every enabled order reaches the same marking. a
        # state is reached by a random enabled part of the arrows, and every remaining arrow after which the rest can
        # still fire is a correct next event. the loss is -log sum of their probabilities, or -log P(STOP) at the end
        flat = self.events(s)
        target = torch.cat(
            [s["target"].permute(0, 3, 1, 2).flatten(1), s["target_stop"][:, None]], 1
        )
        hit = flat.masked_fill(~target, -1e4)

        return (torch.logsumexp(flat, 1) - torch.logsumexp(hit, 1)).mean()

    @torch.no_grad()
    def beam(self, static, width=10, top=10):
        """Most probable end markings of a batch of steps, by beam search over arrows.

        static holds the per step tensors z, odd, cap, lo, hi, mask [B, N] and the start marking cur [B, N, N],
        lone and q [B, N]. Hypotheses with the same firing vector reach the same marking, so they merge and their
        probabilities add. Returns per step a list of (sorted arrows, log probability), best first.
        """
        batch, n = static["mask"].shape
        device = static["mask"].device
        start = {k: static[k].cpu().numpy() for k in ("cur", "lone", "q")}
        live = [
            {(): (0.0, start["cur"][r], start["lone"][r], start["q"][r])}
            for r in range(batch)
        ]
        done = [{} for _ in range(batch)]
        for _ in range(MAX_ARROWS + 1):
            rows = [(r, key) for r in range(batch) for key in live[r]]
            if not rows:
                break

            item = torch.tensor([r for r, _ in rows], device=device)
            s = {k: static[k][item] for k in ("z", "odd", "cap", "lo", "hi", "mask")}
            s["cur"] = torch.from_numpy(
                np.stack([live[r][key][1] for r, key in rows])
            ).to(device)
            s["lone"] = torch.from_numpy(
                np.stack([live[r][key][2] for r, key in rows])
            ).to(device)
            s["q"] = torch.from_numpy(
                np.stack([live[r][key][3] for r, key in rows])
            ).to(device)
            fired = np.zeros((len(rows), 2, n, n), np.int64)
            touched = np.zeros((len(rows), n), bool)
            for h, (r, key) in enumerate(rows):
                for kind, i, j in key:
                    fired[h, kind, i, j] += 1
                    touched[h, [i, j]] = True

            fired = torch.from_numpy(fired).to(device)
            s["fired_a"], s["fired_b"] = fired[:, 0], fired[:, 1]
            s["touched"] = torch.from_numpy(touched).to(device)
            s["n_fired"] = torch.tensor([len(key) for _, key in rows], device=device)
            logp = torch.log_softmax(self.events(s), 1)
            values, index = logp.topk(width, 1)
            grown = [{} for _ in range(batch)]
            for h, (r, key) in enumerate(rows):
                base, cur, lone, q = live[r][key]
                for lp, ix in zip(values[h].tolist(), index[h].tolist()):
                    if lp < -1e3:
                        continue

                    total = base + lp

                    if ix == 2 * n * n:
                        done[r][key] = float(
                            np.logaddexp(done[r].get(key, -np.inf), total)
                        )
                        continue

                    kind, i, j = ix // (n * n), (ix % (n * n)) // n, ix % n
                    new = tuple(sorted(key + ((kind, i, j),)))
                    if new in grown[r]:
                        grown[r][new] = (
                            float(np.logaddexp(grown[r][new][0], total)),
                        ) + grown[r][new][1:]
                        continue

                    sign = 1 if kind == 0 else -1
                    c, l, g = cur.copy(), lone.copy(), q.copy()
                    c[i, j] += sign
                    c[j, i] += sign
                    l[i] -= sign
                    g[i] += sign
                    g[j] -= sign
                    grown[r][new] = (total, c, l, g)

            live = [
                dict(sorted(g.items(), key=lambda kv: -kv[1][0])[:width]) for g in grown
            ]

        return [sorted(d.items(), key=lambda kv: -kv[1])[:top] for d in done]
