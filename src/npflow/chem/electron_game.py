"""Elementary mechanistic steps as firing sequences of the electron net (electron.py)."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .arrow_game import MAX_ARROWS, N_ELEMENT
from .encoder import PetriLayer, mlp

MAX_ELECTRONS = 7

# the four transitions, a lone electron or a pair moving into a bond place or out of one
KINDS = ((0, 1), (0, 2), (1, 1), (1, 2))
N_KIND = len(KINDS)


class ElectronGame(nn.Module):
    """Embedded jump chain of a stochastic net on the electron net, with a learned rate law. A state is the marking of
    a step in progress, bond electrons cur [B, N, N], nonbonding electrons nb and doubled charges q2 [B, N], and the
    arrows fired so far in fired_a and fired_b [B, N, N]. The events are a(i, j, w), b(i, j, w) and STOP. enabling
    False is the ablation without octet capacities and charge windows, where only electrons that exist may move.
    """

    def __init__(self, d=256, rounds=6, attention=6, pair=64, enabling=True, slack=2):
        super().__init__()
        self.enabling, self.slack = enabling, slack
        self.element = nn.Embedding(N_ELEMENT, d)
        self.inp = nn.Linear(d + 33, d)
        self.layers = nn.ModuleList(
            PetriLayer(d, MAX_ELECTRONS - 1) for _ in range(rounds)
        )
        self.attn = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d, 8, 2 * d, dropout=0.0, batch_first=True, norm_first=True
            )
            for _ in range(attention)
        )

        # additive pair scores, one per transition, from the two atoms and the bond place between them
        self.src = nn.Linear(d, N_KIND * pair)
        self.dst = nn.Linear(d, N_KIND * pair)
        self.place = nn.Embedding(MAX_ELECTRONS * 4, N_KIND * pair)
        self.out = nn.Parameter(torch.randn(N_KIND, pair) / pair**0.5)
        self.bias = nn.Parameter(torch.zeros(N_KIND))
        self.count = nn.Embedding(MAX_ARROWS + 1, d)
        self.stop = mlp(d, d, 1)

    def free(self, s):
        """Free octet slots V_i in electrons, what the capacity leaves after the nonbonding and bonding electrons."""
        return 2 * s["cap"] - (s["nb"] + s["cur"].sum(2))

    def encode(self, s):
        mask = s["mask"]
        x = torch.cat(
            [
                self.element(s["z"].clamp(max=N_ELEMENT - 1)),
                F.one_hot((s["q2"] + 6).clamp(0, 12), 13).float(),
                F.one_hot(s["nb"].clamp(0, 8), 9).float(),
                F.one_hot(self.free(s).clamp(0, 8), 9).float(),
                (s["nb"] % 2)[..., None].float(),
                s["touched"][..., None].float(),
            ],
            -1,
        )
        h = self.inp(x) * mask[..., None]
        adj = F.one_hot(s["cur"].clamp(0, 6), 7)[..., 1:].to(h.dtype)
        for k, layer in enumerate(self.layers):
            h = layer(h, adj) * mask[..., None]

            if k < len(self.attn):
                h = self.attn[k](h, src_key_padding_mask=~mask) * mask[..., None]

        return h

    def enabled(self, s):
        """Masks [B, N, N, 4] of the arrows and [B] of STOP. a(i, j, w) needs w nonbonding electrons on i, w free
        slots on j and room on the bond place, b(i, j, w) needs w electrons on the bond, both keep every charge
        within its widened window, and an arrow that would undo one fired in this step is not enabled.
        """
        nb, q2, cur, mask = s["nb"], s["q2"], s["cur"], s["mask"]
        n = cur.shape[1]
        pairs = (
            mask[:, :, None]
            & mask[:, None, :]
            & ~torch.eye(n, dtype=torch.bool, device=cur.device)
        )
        room = (s["n_fired"] < MAX_ARROWS)[:, None, None]
        free = self.free(s)
        lo, hi = s["lo"] - self.slack, s["hi"] + self.slack
        out = []

        for kind, w in KINDS:
            if kind == 0:
                ok = (nb >= w)[:, :, None] & (cur + w <= 2 * 3) & (s["fired_b"] == 0)

                # the octet rule is that the head of a has free slots, the charge windows bound both ends
                if self.enabling:
                    ok = (
                        ok
                        & (free >= w)[:, None, :]
                        & (q2 + w <= hi)[:, :, None]
                        & (q2 - w >= lo)[:, None, :]
                    )
            else:
                ok = (cur >= w) & (s["fired_a"] == 0)

                if self.enabling:
                    ok = ok & (q2 - w >= lo)[:, :, None] & (q2 + w <= hi)[:, None, :]

            out.append(ok & pairs & room)

        # a step ends with every charge inside its window and whole again, and with whole pairs on every bond place, as
        # no molecule holds a one-electron bond
        stop = (
            (((q2 >= s["lo"]) & (q2 <= s["hi"]) & (q2 % 2 == 0)) | ~mask).all(1)
            & (cur % 2 == 0).flatten(1).all(1)
            if self.enabling
            else torch.ones_like(mask[:, 0])
        )

        return torch.stack(out, -1), stop

    def rates(self, s):
        """Log rates of all arrows [B, N, N, 4] and of STOP [B], with their enabling masks."""
        h = self.encode(s)
        batch, n, _ = h.shape
        p = self.out.shape[1]
        u = self.src(h).view(batch, n, 1, N_KIND, p)
        v = self.dst(h).view(batch, 1, n, N_KIND, p)
        flags = (
            s["cur"].clamp(0, MAX_ELECTRONS - 1) * 4
            + (s["fired_a"] > 0).long() * 2
            + (s["fired_b"] > 0).long()
        )
        z = F.silu(u + v + self.place(flags).view(batch, n, n, N_KIND, p))
        logits = (z * self.out).sum(-1) + self.bias
        pooled = (h * s["mask"][..., None]).sum(1) + self.count(
            s["n_fired"].clamp(max=MAX_ARROWS)
        )
        enabled, stop_ok = self.enabled(s)

        return logits.float(), enabled, self.stop(pooled).squeeze(-1).float(), stop_ok

    def events(self, s):
        """Log rates of every event as one flat vector per state, the four transitions over N * N then STOP."""
        logits, enabled, stop, stop_ok = self.rates(s)
        arrows = logits.masked_fill(~enabled, -1e4).permute(0, 3, 1, 2).flatten(1)

        return torch.cat([arrows, stop.masked_fill(~stop_ok, -1e4)[:, None]], 1)

    def loss(self, s):
        # the record gives the arrows of a step but no order, so a state is a random enabled part of them and every
        # remaining arrow after which the rest can still fire is correct, the loss is -log of their total probability
        flat = self.events(s)
        target = torch.cat(
            [s["target"].permute(0, 3, 1, 2).flatten(1), s["target_stop"][:, None]], 1
        )
        hit = flat.masked_fill(~target, -1e4)

        return (torch.logsumexp(flat, 1) - torch.logsumexp(hit, 1)).mean()

    @torch.no_grad()
    def beam(self, static, width=10, top=10):
        """Most probable end markings of a batch of steps by beam search over arrows. static holds z, cap, lo, hi,
        mask [B, N], cur [B, N, N], nb and q2 [B, N]. Hypotheses with the same firing vector reach the same marking
        and merge. Returns per step a list of (sorted arrows, log probability), best first.
        """
        batch, n = static["mask"].shape
        device = static["mask"].device
        start = {k: static[k].cpu().numpy() for k in ("cur", "nb", "q2")}
        live = [
            {(): (0.0, start["cur"][r], start["nb"][r], start["q2"][r])}
            for r in range(batch)
        ]
        done = [{} for _ in range(batch)]
        for _ in range(MAX_ARROWS + 1):
            rows = [(r, key) for r in range(batch) for key in live[r]]
            if not rows:
                break

            item = torch.tensor([r for r, _ in rows], device=device)
            s = {k: static[k][item] for k in ("z", "cap", "lo", "hi", "mask")}
            for name, k in (("cur", 1), ("nb", 2), ("q2", 3)):
                s[name] = torch.from_numpy(
                    np.stack([live[r][key][k] for r, key in rows])
                ).to(device)

            fired = np.zeros((len(rows), 2, n, n), np.int64)
            touched = np.zeros((len(rows), n), bool)
            for h, (r, key) in enumerate(rows):
                for kind, i, j, w in key:
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
                base, cur, nb, q2 = live[r][key]
                for lp, ix in zip(values[h].tolist(), index[h].tolist()):
                    if lp < -1e3:
                        continue

                    total = base + lp

                    if ix == N_KIND * n * n:
                        done[r][key] = float(
                            np.logaddexp(done[r].get(key, -np.inf), total)
                        )
                        continue

                    slot, rest = ix // (n * n), ix % (n * n)
                    kind, w = KINDS[slot]
                    i, j = rest // n, rest % n
                    new = tuple(sorted(key + ((kind, i, j, w),)))
                    if new in grown[r]:
                        grown[r][new] = (
                            float(np.logaddexp(grown[r][new][0], total)),
                        ) + grown[r][new][1:]
                        continue

                    sign = w if kind == 0 else -w
                    c, e, g = cur.copy(), nb.copy(), q2.copy()
                    c[i, j] += sign
                    c[j, i] += sign
                    e[i] -= sign
                    g[i] += sign
                    g[j] -= sign
                    grown[r][new] = (total, c, e, g)

            live = [
                dict(sorted(g.items(), key=lambda kv: -kv[1][0])[:width]) for g in grown
            ]

        return [sorted(d.items(), key=lambda kv: -kv[1])[:top] for d in done]
