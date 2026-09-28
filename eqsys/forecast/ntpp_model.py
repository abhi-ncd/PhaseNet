"""Neural temporal point process in the style of RECAST (Dascher-Cousineau et al., 2023, GRL).

A GRU encodes the event history, one step per event with features (log inter-event time,
magnitude above Mc). From the hidden state after event i, a mixture of Weibull distributions gives
the density of the time to the next event. Magnitudes follow Gutenberg-Richter with the training
b-value (as in RECAST), so only the temporal part is learned, and the log-likelihood is directly
comparable with the temporal ETAS and Poisson models.

Log-likelihood of events in [a, b) given all earlier events:
    sum_i log f(tau_i | h_{i-1})  - log S(a - t_prev | h_prev)  + log S(b - t_last | h_last)
where the first correction conditions on no event between the last pre-window event and a, and
the last term is the probability of no further event before b.
"""

import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..catalog import b_value
from .temporal import sample_gr

logger = logging.getLogger(__name__)

EPS = 1e-8


class RecastNet(nn.Module):
    def __init__(self, hidden=32, n_mix=8):
        super().__init__()
        self.gru = nn.GRU(input_size=2, hidden_size=hidden, batch_first=True)
        self.h0 = nn.Parameter(torch.zeros(hidden))
        self.head = nn.Linear(hidden, 3 * n_mix)

    def encode(self, x):
        """x: (1, N, 2) -> hidden states (N + 1, H); row i is the state after i events."""
        h, _ = self.gru(x, self.h0.view(1, 1, -1))
        return torch.cat([self.h0.view(1, -1), h[0]], dim=0)

    def step(self, x, h):
        """x: (B, 2) features of one new event per sequence, h: (B, H) -> new states (B, H)."""
        out, _ = self.gru(x.unsqueeze(1), h.unsqueeze(0).contiguous())
        return out[:, 0]

    def mixture(self, h):
        logits, log_shape, log_scale = self.head(h).chunk(3, dim=-1)
        log_w = F.log_softmax(logits, dim=-1)
        shape = F.softplus(log_shape) + 0.05
        scale = torch.exp(log_scale.clamp(-15, 15))
        return log_w, shape, scale

    @staticmethod
    def log_survival_components(tau, shape, scale):
        return -((tau.unsqueeze(-1) / scale) ** shape)

    def log_pdf(self, tau, h):
        log_w, k, lam = self.mixture(h)
        tau_ = tau.clamp_min(EPS).unsqueeze(-1)
        log_f = torch.log(k) - torch.log(lam) + (k - 1) * (torch.log(tau_) - torch.log(lam)) - (tau_ / lam) ** k
        return torch.logsumexp(log_w + log_f, dim=-1)

    def log_survival(self, tau, h):
        log_w, k, lam = self.mixture(h)
        return torch.logsumexp(log_w + self.log_survival_components(tau.clamp_min(0.0), k, lam), dim=-1)


class NeuralTPP:
    name = "NTPP (RECAST-style)"

    def __init__(self, hidden=32, n_mix=8, epochs=200, lr=1e-3, seed=0, patience=30, **_):
        self.hidden, self.n_mix, self.epochs, self.lr, self.seed, self.patience = hidden, n_mix, epochs, lr, seed, patience

    # ------------------------------------------------------------ features
    def features(self, t, m):
        tau = np.diff(np.r_[t[0] - self.tau0, t])
        x = np.stack([(np.log(np.maximum(tau, 1e-6)) - self.log_tau_mean) / self.log_tau_std, m - self.mc], axis=-1)
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    def window_ll(self, H, t, a, b):
        """Log-likelihood of events in [a, b); H[i] is the state after i events of the full catalog t."""
        t_t = torch.tensor(t, dtype=torch.float64)
        i0 = int(np.searchsorted(t, a, side="left"))
        i1 = int(np.searchsorted(t, b, side="left"))
        n = i1 - i0
        ll = torch.zeros((), dtype=torch.float32)
        prev_t = t_t[i0 - 1] if i0 > 0 else torch.tensor(a, dtype=torch.float64)
        if n > 0:
            prev = torch.cat([prev_t.view(1), t_t[i0:i1 - 1]])
            tau = (t_t[i0:i1] - prev).float()
            ll = ll + self.net.log_pdf(tau, H[i0:i1]).sum()
        if i0 > 0:  # condition on no event between the last pre-window event and a
            ll = ll - self.net.log_survival((torch.tensor(a) - prev_t).float().view(1), H[i0:i0 + 1]).sum()
        # no further event before b (if the window is empty this pairs with the conditioning term above)
        last_t = t_t[i1 - 1] if i1 > 0 else torch.tensor(a, dtype=torch.float64)
        ll = ll + self.net.log_survival((b - last_t).float().view(1), H[i1:i1 + 1]).sum()
        return ll, n

    # ------------------------------------------------------------ interface
    def fit(self, split):
        torch.manual_seed(self.seed)
        self.mc, self.delta_m = split.mc, split.delta_m
        t_tr, m_tr = split.window("train")
        tau_tr = np.diff(t_tr)
        self.tau0 = float(np.median(tau_tr)) if len(tau_tr) else 1.0
        log_tau = np.log(np.maximum(tau_tr, 1e-6))
        self.log_tau_mean, self.log_tau_std = float(log_tau.mean()), float(log_tau.std() + 1e-6)
        self.b, _, _ = b_value(m_tr, self.mc, self.delta_m)
        self.beta = self.b * np.log(10)

        self.net = RecastNet(self.hidden, self.n_mix)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        # training uses events up to the end of the training window; validation sees train + valid
        end_train = int(np.searchsorted(split.t, split.t_train[1]))
        end_valid = int(np.searchsorted(split.t, split.t_valid[1]))
        x_train = self.features(split.t[:end_train], split.m[:end_train])
        x_valid = self.features(split.t[:end_valid], split.m[:end_valid])
        best, best_state, bad = -np.inf, None, 0
        for epoch in range(self.epochs):
            self.net.train()
            opt.zero_grad()
            H = self.net.encode(x_train)
            ll, n = self.window_ll(H, split.t[:end_train], *split.t_train)
            (-ll / max(n, 1)).backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
            opt.step()

            self.net.eval()
            with torch.no_grad():
                H = self.net.encode(x_valid)
                vll, vn = self.window_ll(H, split.t[:end_valid], *split.t_valid)
                vll = vll.item() / max(vn, 1)
            if vll > best:
                best, best_state, bad = vll, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
            if epoch % 20 == 0:
                logger.info(f"NTPP epoch {epoch}: train LL/event {ll.item() / max(n, 1):.3f}, valid LL/event {vll:.3f}")
        self.net.load_state_dict(best_state)
        self.net.eval()
        logger.info(f"NTPP best validation LL/event {best:.3f}")
        return self

    def log_likelihood(self, t, m, a, b):
        end = int(np.searchsorted(t, b))
        with torch.no_grad():
            H = self.net.encode(self.features(t[:end], m[:end]))
            ll, n = self.window_ll(H, t[:end], a, b)
        return float(ll), n

    def hidden_states(self, t, m):
        with torch.no_grad():
            return self.net.encode(self.features(t, m))

    @torch.no_grad()
    def simulate_counts(self, t, m, d0, d1, n_sim, rng, H=None, max_steps=20000):
        if H is None:
            H = self.hidden_states(t, m)
        i = int(np.searchsorted(t, d0))
        h = H[i].expand(n_sim, -1).contiguous()
        t_last = np.full(n_sim, t[i - 1] if i > 0 else d0)
        lower = np.full(n_sim, d0 - t_last[0] if i > 0 else 0.0)
        counts = np.zeros(n_sim, dtype=int)
        alive = np.ones(n_sim, dtype=bool)
        for _ in range(max_steps):
            idx = np.where(alive)[0]
            if len(idx) == 0:
                break
            log_w, k, lam = self.net.mixture(h[idx])
            lo = torch.tensor(lower[idx], dtype=torch.float32)
            # component weights given survival past `lower`, then an exact truncated Weibull draw
            logits = log_w + self.net.log_survival_components(lo, k, lam)
            comp = torch.distributions.Categorical(logits=logits).sample()
            kk = k.gather(1, comp[:, None])[:, 0].double().numpy()
            ll = lam.gather(1, comp[:, None])[:, 0].double().numpy()
            u = rng.random(len(idx))
            tau = ll * ((lower[idx] / ll) ** kk - np.log(u)) ** (1 / kk)
            t_new = t_last[idx] + tau
            inside = t_new < d1
            alive[idx[~inside]] = False
            idx_in = idx[inside]
            if len(idx_in) == 0:
                break
            counts[idx_in] += 1
            m_new = sample_gr(len(idx_in), self.mc, self.delta_m, self.beta, rng)
            log_tau = (np.log(np.maximum(tau[inside], 1e-6)) - self.log_tau_mean) / self.log_tau_std
            x = torch.tensor(np.stack([log_tau, m_new - self.mc], axis=-1), dtype=torch.float32)
            h[idx_in] = self.net.step(x, h[idx_in])
            t_last[idx_in] = t_new[inside]
            lower[idx_in] = 0.0
        return counts
