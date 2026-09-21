"""Models for two tasks on attributed Petri nets.

  task="transitions"  given two states A and B, predict the firing counts sigma of every transition
  task="next"         given a state, predict the state one step ahead

All models see the same inputs (markings, static place/transition attributes, elapsed time, arcs with
weights and direction) and use the same number of propagation rounds.

  npf        Neural Petri Flow (ours). The learned object is the local rate law lambda_theta of a
             transition, in the product form that Petri / reaction-network theory prescribes
             (log-linear over the input places). Everything else is Petri net semantics:
               next         token game  m <- m + C v  with demand = rate * dt, conflict resolution, min
               transitions  expected counts = rate law integrated along a token-game path from A to B,
                            times a local posterior correction; then the state equation M_B = M_A + C sigma
                            is imposed exactly (Poisson-weighted projection) and occupancies are refined
  npf-mlp    same, but the rate law is a generic set function of the input places (no product form)
  npf-prior  same, but without the local posterior correction: rate law + Petri semantics only
  npf-1pass  same, but without the occupancy refinement (one rate-law pass, one projection)
  pgnn       the paper's multimodal message passing, Eqs. (9)-(12), with the signed incidence-weighted
             aggregation of Fig. 3d / Eq. (13)
  pgnn-eq10  same, but Eq. (10) read literally (a place only hears its incoming transitions)
  pgnn+      strengthened PGNN: learned role-specific messages in both directions
  pgnn+se    ablation: pgnn+ with the state equation on top (projection / flux readout), no rate law
  gnn        classic message passing, Eqs. (3)-(6), on the place graph
  se-only    no learning: minimum-norm non-negative solution of the state equation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_MARK = 3  # marking features
N_PLACE_FEAT = 2 * N_MARK + 1 + 1 + 1  # features of m, features of m_b, m_b - m, e, dt
N_TRANS_FEAT = 1 + 1  # a, dt


def mlp(n_in, n_hidden, n_out):
    return nn.Sequential(
        nn.Linear(n_in, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_out)
    )


def scatter_sum(src, index, n):
    return src.new_zeros((n,) + src.shape[1:]).index_add_(0, index, src)


def marking_features(m):
    m = m.clamp(min=0)
    return torch.stack([m / 4.0, torch.log1p(m), torch.log(m + 0.01)], -1)  # log m: rates are log-linear in markings


def place_features(b, m):
    return torch.cat([marking_features(m), marking_features(b.m_b), ((b.m_b - m) / 4.0)[:, None], b.e, b.dt_p[:, None]], -1)


def trans_features(b):
    return torch.cat([b.a, b.dt_t[:, None]], -1)


def apply_incidence(v, b):
    """C v on the batched net: tokens produced minus tokens consumed by the firing vector v."""
    produced = scatter_sum(b.pos_w * v[b.pos_t], b.pos_p, b.n_places)
    consumed = scatter_sum(b.pre_w * v[b.pre_t], b.pre_p, b.n_places)
    return produced - consumed


def project_state_equation(sigma, b, weighted=True, n_iter=6, obs_var=None):
    """Firing vector closest to `sigma` that satisfies the state equation M_B = M_A + C sigma.

    weighted: treat sigma as the mean of independent Poisson counts (variance = mean) and condition on
    the exact linear observation C sigma = dM. The posterior mean is a generalised least-squares step
        sigma + W C^T (C W C^T)^+ (dM - C sigma),   W = diag(sigma),
    i.e. one solve with the Petri Laplacian C W C^T. The orthogonal projector C^+ then removes what is
    left of the residual; alternating with sigma >= 0 keeps counts non-negative. Only the component in
    im C^T is ever dictated by the data, the T-invariant component (ker C) stays with the estimator.

    obs_var: variance of the noise on the observed markings. The state equation then only holds up to
    that noise, the solve becomes the Kalman update (C W C^T + obs_var I)^-1 and nothing is forced."""
    G, S, Pmax, Tmax = b.pad_shape
    dm = sigma.new_zeros(G * S * Pmax).index_copy_(0, b.pad_p, b.m_b - b.m).view(G, S, Pmax)
    s = sigma.new_zeros(G * S * Tmax).index_copy_(0, b.pad_t, sigma).view(G, S, Tmax)
    if weighted:
        C = b.C.double()[:, None]  # [G, 1, P, T]
        w = s.double() + 1e-3
        ridge = 1e-6 if obs_var is None else obs_var.double()
        laplacian = (C * w[:, :, None, :]) @ C.transpose(-1, -2) + ridge * torch.eye(Pmax, dtype=torch.float64, device=s.device)
        residual = dm.double() - torch.einsum("gst,gpt->gsp", s.double(), b.C.double())
        y = torch.linalg.solve(laplacian, residual)
        s = s + (w * torch.einsum("gsp,gpt->gst", y, b.C.double())).to(s.dtype)
    if obs_var is not None:
        return s.clamp(min=0).reshape(-1)[b.pad_t]
    for i in range(n_iter):
        if i:
            s = s.clamp(min=0)
        residual = dm - torch.einsum("gst,gpt->gsp", s, b.C)
        s = s + torch.einsum("gsp,gtp->gst", residual, b.C_pinv)
    return s.reshape(-1)[b.pad_t]


class NPF(nn.Module):
    """The learned object is the rate law of a transition as a function of its *input* places (locality),
    zero when an input place is empty (enabling). With product_form the rate is
        lambda_t = exp( beta(a_t) + sum_{p in •t} g(m_p, e_p, Pre(p, t)) ),
    a product of learned place activities - generalised mass action. The same rate law serves both tasks;
    inferring transitions between two observed states adds a local posterior correction (see __init__)."""

    def __init__(self, task, hidden=64, rounds=4, product_form=True, noisy_states=False, refine=True, posterior=True):
        super().__init__()
        self.task, self.rounds, self.product_form, self.refine = task, rounds, product_form, refine
        self.log_obs_var = nn.Parameter(torch.tensor(-2.0)) if noisy_states else None
        self.damping = nn.Parameter(torch.tensor(0.0))  # sigmoid -> 0.5
        # The rate law is a forward model and only sees input places. Given BOTH end states, the expected
        # count of a transition also depends on what arrived at its output places: a local posterior
        # correction exp(c_t) of the rate-law prior, initialised at c_t = 0.
        self.posterior = posterior and task == "transitions"
        if self.posterior:
            self.post_in, self.post_out = mlp(N_PLACE_FEAT, hidden, hidden), mlp(N_PLACE_FEAT, hidden, hidden)
            self.post = mlp(2 * hidden + 1, hidden, 1)
            nn.init.zeros_(self.post[-1].weight); nn.init.zeros_(self.post[-1].bias)
        self.arc = mlp(N_MARK + 1 + 1, hidden, 1 if product_form else hidden)
        self.base = mlp(1, hidden, 1) if product_form else mlp(hidden + 1, hidden, 1)

    def rate(self, m, b):
        x = torch.cat([marking_features(m)[b.pre_p], b.e[b.pre_p], b.pre_w[:, None]], -1)
        pooled = scatter_sum(self.arc(x), b.pre_t, b.n_trans)
        log_rate = self.base(b.a) + pooled if self.product_form else self.base(torch.cat([pooled, b.a], -1))
        tokens = (m[b.pre_p] / b.pre_w).clamp(0.0, 1.0)
        enabled = tokens.new_ones(b.n_trans).scatter_reduce(0, b.pre_t, tokens, reduce="amin")
        return torch.exp(log_rate.squeeze(-1).clamp(max=8.0)) * enabled

    def infer_transitions(self, m, b):
        """Expected counts are the rate law integrated along the path from A to B. The path is latent:
          path            play the fluid token game from A with the learned rates (fast transitions move
                          their tokens early), then pin the end of the path to the observed state B
          rate law        sigma~ = dt * mean over the path of lambda(occupancy)       (local, learned)
          state equation  sigma^ = sigma~ + W C^T (C W C^T)^+ (dM - C sigma~)         (global, exact)
          occupancy       a place whose consumers must fire more than predicted was fuller than assumed;
                          consumers of one place share it in proportion to their rates (free choice),
                          so a place that is empty in A and B but had tokens passing through is recovered
        The last two steps are iterated; the state equation holds exactly after every iteration."""
        obs_var = None if self.log_obs_var is None else self.log_obs_var.exp()
        has_consumer = b.pre_w.new_zeros(b.n_places).index_add_(0, b.pre_p, torch.ones_like(b.pre_w)) > 0
        states = [m]
        for _ in range(self.rounds):
            states.append(states[-1] + apply_incidence(self.step(states[-1], b, b.dt_t), b))
        miss = b.m_b - states[-1]
        path = [((states[k] + states[k + 1]) / 2 + (k + 0.5) / self.rounds * miss).clamp(min=0) + 0.05 * has_consumer
                for k in range(self.rounds)]
        scale = torch.ones_like(m)
        for i in range(self.rounds if self.refine else 1):
            prior = b.dt_t * sum(self.rate(x * scale, b) for x in path) / self.rounds
            if self.posterior:
                if i == 0:  # sees the two states, never the elapsed time: time enters only through the rate law
                    x = place_features(b, m)[:, :-1]
                    z_in = scatter_sum(self.post_in(torch.cat([x[b.pre_p], b.pre_w[:, None]], -1)), b.pre_t, b.n_trans)
                    z_out = scatter_sum(self.post_out(torch.cat([x[b.pos_p], b.pos_w[:, None]], -1)), b.pos_t, b.n_trans)
                    correction = torch.exp(1.5 * torch.tanh(self.post(torch.cat([z_in, z_out, b.a], -1)).squeeze(-1)))
                prior = prior * correction
            sigma = project_state_equation(prior, b, obs_var=obs_var)
            needed = scatter_sum(b.pre_w * sigma[b.pre_t], b.pre_p, b.n_places)
            predicted = scatter_sum(b.pre_w * prior[b.pre_t], b.pre_p, b.n_places)
            scale = scale * ((needed.clamp(min=0) + 0.1) / (predicted + 0.1)) ** torch.sigmoid(self.damping)
        return sigma

    def step(self, m, b, dt=1.0):
        """One round of the token game. Conflict resolution: a place facing total demand D hands out
        m (1 - exp(-D / m)) tokens, shared in proportion to the demands - exact for first-order consumption,
        never more than it holds. Synchronisation: a transition fires as far as its scarcest input allows."""
        d = self.rate(m, b) * dt / self.rounds
        x = scatter_sum(b.pre_w * d[b.pre_t], b.pre_p, b.n_places) / m.clamp(min=1e-30)
        served = torch.where(x < 1e-4, 1.0 - x / 2, -torch.expm1(-x) / x.clamp(min=1e-4))
        return d * served.new_ones(b.n_trans).scatter_reduce(0, b.pre_t, served[b.pre_p], reduce="amin")

    def forward(self, b, m=None, return_firing=False):
        m = b.m if m is None else m
        if self.task == "transitions":
            return self.infer_transitions(m, b)
        sigma = 0.0
        for _ in range(self.rounds):  # tied weights: depth is time, every P-invariant is conserved exactly
            v = self.step(m, b)
            m, sigma = m + apply_incidence(v, b), sigma + v
        return (m, sigma) if return_firing else m


class PGNN(nn.Module):
    def __init__(self, task, hidden=64, rounds=4, aggregate="incidence", state_equation=False):
        super().__init__()
        self.task, self.aggregate, self.state_equation = task, aggregate, state_equation
        self.enc = mlp(N_PLACE_FEAT, hidden, hidden)
        n_msg = 2 * hidden if aggregate == "learned" else hidden
        self.layers = nn.ModuleList()
        for i in range(rounds + 1):  # the last entry only computes the transition messages that are read out
            layer = nn.ModuleDict({
                "phi_in": mlp(hidden + 1, hidden, hidden),  # input places  •t with w_{•t,t}
                "phi_out": mlp(hidden + 1, hidden, hidden),  # output places t• with w_{t•,t}
                "psi": mlp(2 * hidden + N_TRANS_FEAT, hidden, hidden),  # Eq. (9)
            })
            if i < rounds:
                layer["upd"] = mlp(hidden + n_msg, hidden, hidden)  # Eq. (11)
                if aggregate == "learned":
                    layer["from_in"] = mlp(hidden + 1, hidden, hidden)
                    layer["from_out"] = mlp(hidden + 1, hidden, hidden)
            self.layers.append(layer)
        self.out = mlp(hidden, hidden, 1)  # Eq. (12)

    def message(self, layer, h, b):
        z_in = scatter_sum(layer["phi_in"](torch.cat([h[b.pre_p], b.pre_w[:, None]], -1)), b.pre_t, b.n_trans)
        z_out = scatter_sum(layer["phi_out"](torch.cat([h[b.pos_p], b.pos_w[:, None]], -1)), b.pos_t, b.n_trans)
        return layer["psi"](torch.cat([z_in, z_out, trans_features(b)], -1))

    def forward(self, b, m=None):
        m = b.m if m is None else m
        h = self.enc(place_features(b, m))
        for layer in self.layers[:-1]:
            m_t = self.message(layer, h, b)
            if self.aggregate == "incidence":  # m_p = sum_t C[p, t] m_t   (Fig. 3d, Eq. 13)
                m_p = scatter_sum(b.pos_w[:, None] * m_t[b.pos_t], b.pos_p, b.n_places) \
                    - scatter_sum(b.pre_w[:, None] * m_t[b.pre_t], b.pre_p, b.n_places)
            elif self.aggregate == "incoming":  # Eq. (10) read literally: only t in •p
                m_p = scatter_sum(b.pos_w[:, None] * m_t[b.pos_t], b.pos_p, b.n_places)
            else:
                m_p = torch.cat([
                    scatter_sum(layer["from_in"](torch.cat([m_t[b.pos_t], b.pos_w[:, None]], -1)), b.pos_p, b.n_places),
                    scatter_sum(layer["from_out"](torch.cat([m_t[b.pre_t], b.pre_w[:, None]], -1)), b.pre_p, b.n_places),
                ], -1)
            h = h + layer["upd"](torch.cat([h, m_p], -1))
        if self.task == "next" and not self.state_equation:
            return m + self.out(h).squeeze(-1)
        flow = self.out(self.message(self.layers[-1], h, b)).squeeze(-1)  # the paper reads m_t as network flow
        if self.task == "next":
            return m + apply_incidence(flow, b)
        return project_state_equation(F.softplus(flow), b) if self.state_equation else F.softplus(flow)


class GNN(nn.Module):
    def __init__(self, task, hidden=64, rounds=4):
        super().__init__()
        self.task = task
        self.enc = mlp(N_PLACE_FEAT, hidden, hidden)
        n_edge = 2 + N_TRANS_FEAT
        self.layers = nn.ModuleList(
            nn.ModuleDict({
                "fwd": mlp(2 * hidden + n_edge, hidden, hidden),
                "bwd": mlp(2 * hidden + n_edge, hidden, hidden),
                "upd": mlp(3 * hidden, hidden, hidden),
            })
            for _ in range(rounds)
        )
        self.pool_in, self.pool_out = mlp(hidden + 1, hidden, hidden), mlp(hidden + 1, hidden, hidden)
        self.out = mlp(2 * hidden + N_TRANS_FEAT if task == "transitions" else hidden, hidden, 1)

    def forward(self, b, m=None):
        m = b.m if m is None else m
        h = self.enc(place_features(b, m))
        edge = torch.cat([b.edge_wu[:, None], b.edge_wv[:, None], trans_features(b)[b.edge_t]], -1)
        for layer in self.layers:
            x = torch.cat([h[b.edge_u], h[b.edge_v], edge], -1)
            m_in = scatter_sum(layer["fwd"](x), b.edge_v, b.n_places)
            m_out = scatter_sum(layer["bwd"](x), b.edge_u, b.n_places)
            h = h + layer["upd"](torch.cat([h, m_in, m_out], -1))
        if self.task == "next":
            return m + self.out(h).squeeze(-1)
        z_in = scatter_sum(self.pool_in(torch.cat([h[b.pre_p], b.pre_w[:, None]], -1)), b.pre_t, b.n_trans)
        z_out = scatter_sum(self.pool_out(torch.cat([h[b.pos_p], b.pos_w[:, None]], -1)), b.pos_t, b.n_trans)
        return F.softplus(self.out(torch.cat([z_in, z_out, trans_features(b)], -1)).squeeze(-1))


class LinearPGNN(nn.Module):
    """The special case the paper trains in its experiments (Eq. 13, Fig. 3d-e): on one fixed net, the flow
    of a transition is a linear function of the signals on its incident places, m_t = omega_t * sum_p w_pt x_p,
    with learnable flow rates omega (one per transition) and conversion weights Pre/Pos (one per arc and
    signal) on the known incidence pattern, followed by a ReLU. Transductive: parameters belong to the net."""

    def __init__(self, net):
        super().__init__()
        mask = torch.as_tensor((net.Pre + net.Pos > 0).T, dtype=torch.float32)  # [T, P]
        self.register_buffer("mask", mask)
        self.w_a, self.w_b = nn.Parameter(0.1 * torch.randn_like(mask)), nn.Parameter(0.1 * torch.randn_like(mask))
        self.omega, self.bias = nn.Parameter(torch.ones(len(mask))), nn.Parameter(torch.zeros(len(mask)))

    def forward(self, b, m=None):
        P = self.mask.shape[1]
        flow = b.m.view(-1, P) @ (self.w_a * self.mask).T + b.m_b.view(-1, P) @ (self.w_b * self.mask).T
        return F.relu(self.omega * flow + self.bias).reshape(-1)


class StateEquationOnly(nn.Module):
    def forward(self, b, m=None):
        return project_state_equation(b.m.new_zeros(b.n_trans), b)


def build(name, task, hidden=64, rounds=4):
    """`name@k` overrides the number of rounds, e.g. npf@16 plays the token game with 16 finer time steps."""
    if "@" in name:
        name, rounds = name.split("@")[0], int(name.split("@")[1])
    if name in ("npf", "npf-mlp", "npf-1pass", "npf-prior"):
        return NPF(task, hidden, rounds, product_form=name != "npf-mlp", refine=name != "npf-1pass", posterior=name != "npf-prior")
    if name == "gnn":
        return GNN(task, hidden, rounds)
    if name == "se-only":
        return StateEquationOnly()
    aggregate = {"pgnn": "incidence", "pgnn-eq10": "incoming", "pgnn+": "learned", "pgnn+se": "learned"}[name]
    return PGNN(task, hidden, rounds, aggregate=aggregate, state_equation=name == "pgnn+se")
