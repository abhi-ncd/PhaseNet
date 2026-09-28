"""Temporal point-process baselines: homogeneous Poisson and temporal ETAS.

Temporal ETAS (Ogata, 1988) conditional intensity, with times in days:
    lambda(t) = mu + sum_{t_j < t} k * exp(alpha * (m_j - Mc)) * (t - t_j + c)^(-p)
The triggering kernel is truncated after `max_lag` days so that likelihood evaluation scales
with the number of event pairs inside that lag instead of N^2.

Every model exposes the same interface so evaluate.py can compare them:
    fit(split)
    log_likelihood(t, m, a, b) -> (sum of log-likelihood over events in [a, b), number of events)
    simulate_counts(t, m, d0, d1, n_sim, rng) -> number of events in [d0, d1) per simulation
where t, m hold the full catalog; only events before the scored window's end are used as history.
"""

import logging

import numpy as np
from scipy.optimize import minimize

from ..catalog import b_value

logger = logging.getLogger(__name__)


def sample_gr(n, mc, delta_m, beta, rng, m_max=9.0):
    """Gutenberg-Richter magnitudes above the bin edge mc - delta_m/2, truncated at m_max."""
    m0 = mc - delta_m / 2
    u = rng.random(n)
    return m0 - np.log(1 - u * (1 - np.exp(-beta * (m_max - m0)))) / beta


class PoissonModel:
    name = "Poisson"

    def fit(self, split):
        t, _ = split.window("train")
        a, b = split.t_train
        self.mu = len(t) / (b - a)
        return self

    def log_likelihood(self, t, m, a, b):
        n = int(np.sum((t >= a) & (t < b)))
        return n * np.log(self.mu) - self.mu * (b - a), n

    def simulate_counts(self, t, m, d0, d1, n_sim, rng):
        return rng.poisson(self.mu * (d1 - d0), size=n_sim)


class TemporalETAS:
    name = "ETAS"
    param_names = ["log10_mu", "log10_k", "alpha", "log10_c", "p"]
    bounds = [(-6, 3), (-6, 1), (0.0, 5.0), (-6, 0), (1.001, 3.0)]

    def __init__(self, max_lag=365.0, m_max=9.0):
        self.max_lag = max_lag
        self.m_max = m_max

    # ------------------------------------------------------------ kernel pieces
    @staticmethod
    def omori_integral(lo, hi, c, p):
        """Integral of (s + c)^(-p) from lo to hi (elementwise, p > 1)."""
        return ((lo + c) ** (1 - p) - (hi + c) ** (1 - p)) / (p - 1)

    def _pairs(self, t, targets):
        """Flat (target, parent) index arrays for parents within max_lag before each target."""
        lo = np.searchsorted(t, t[targets] - self.max_lag, side="left")
        counts = targets - lo
        tgt = np.repeat(np.arange(len(targets)), counts)
        start = np.repeat(lo - np.cumsum(np.r_[0, counts[:-1]]), counts)
        par = start + np.arange(counts.sum())
        return tgt, par

    def _prepare(self, t, m, a, b):
        targets = np.where((t >= a) & (t < b))[0]
        tgt, par = self._pairs(t, targets)
        dt = t[targets][tgt] - t[par]
        sources = np.where((t < b) & (t >= a - self.max_lag))[0]
        lo = np.maximum(a - t[sources], 0.0)
        hi = np.minimum(b - t[sources], self.max_lag)
        ok = hi > lo
        return {
            "n": len(targets), "tgt": tgt, "dt": dt, "m_par": m[par] - self.mc,
            "m_src": m[sources][ok] - self.mc, "lo": lo[ok], "hi": hi[ok], "T": b - a,
        }

    def _log_likelihood(self, theta, d):
        log10_mu, log10_k, alpha, log10_c, p = theta
        mu, k, c = 10**log10_mu, 10**log10_k, 10**log10_c
        trig = k * np.exp(alpha * d["m_par"]) * (d["dt"] + c) ** (-p)
        lam = mu + np.bincount(d["tgt"], weights=trig, minlength=d["n"])
        integral = mu * d["T"] + np.sum(k * np.exp(alpha * d["m_src"]) * self.omori_integral(d["lo"], d["hi"], c, p))
        return np.sum(np.log(lam)) - integral

    # ------------------------------------------------------------ interface
    def fit(self, split, n_starts=4, seed=0):
        self.mc, self.delta_m = split.mc, split.delta_m
        a, b = split.t_train
        d = self._prepare(split.t, split.m, a, b)
        rng = np.random.default_rng(seed)
        mu0 = np.log10(max(d["n"], 1) / d["T"] / 2)
        best = None
        for i in range(n_starts):
            x0 = np.array([mu0, -1.5, 1.5, -2.5, 1.1]) if i == 0 else np.array(
                [rng.uniform(lo, hi) for lo, hi in self.bounds])
            res = minimize(lambda th: -self._log_likelihood(th, d), x0, method="L-BFGS-B", bounds=self.bounds)
            if best is None or res.fun < best.fun:
                best = res
        self.theta = best.x
        self.params = dict(zip(self.param_names, map(float, best.x)))
        train_m = split.window("train")[1]
        self.b, _, _ = b_value(train_m, self.mc, self.delta_m)
        self.beta = self.b * np.log(10)
        logger.info(f"ETAS parameters: {self.params}, b={self.b:.2f}, branching ratio={self.branching_ratio():.2f}")
        return self

    def branching_ratio(self):
        """Expected number of direct aftershocks per event (GR magnitudes, truncated kernel)."""
        _, log10_k, alpha, log10_c, p = self.theta
        k, c = 10**log10_k, 10**log10_c
        if self.beta <= alpha:
            return np.inf
        mean_prod = self.beta / (self.beta - alpha)
        return float(k * mean_prod * self.omori_integral(0.0, self.max_lag, c, p))

    def log_likelihood(self, t, m, a, b):
        d = self._prepare(t, m, a, b)
        return float(self._log_likelihood(self.theta, d)), d["n"]

    def _sample_times(self, lo, hi, rng):
        _, _, _, log10_c, p = self.theta
        c = 10**log10_c
        A, B = (lo + c) ** (1 - p), (hi + c) ** (1 - p)
        u = rng.random(len(lo))
        return (A - u * (A - B)) ** (1 / (1 - p)) - c

    def simulate_counts(self, t, m, d0, d1, n_sim, rng, max_events=100000):
        log10_mu, log10_k, alpha, log10_c, p = self.theta
        mu, k, c = 10**log10_mu, 10**log10_k, 10**log10_c
        hist = (t < d0) & (t >= d0 - self.max_lag)
        th, mh = t[hist], m[hist]
        lo, hi = d0 - th, np.minimum(d1 - th, self.max_lag)
        ok = hi > lo
        th, mh, lo, hi = th[ok], mh[ok], lo[ok], hi[ok]
        expected_hist = k * np.exp(alpha * (mh - self.mc)) * self.omori_integral(lo, hi, c, p)

        counts = np.zeros(n_sim, dtype=int)
        for s in range(n_sim):
            n_bg = rng.poisson(mu * (d1 - d0))
            times = [d0 + rng.random(n_bg) * (d1 - d0)]
            n_off = rng.poisson(expected_hist)
            idx = np.repeat(np.arange(len(n_off)), n_off)
            times.append(th[idx] + self._sample_times(lo[idx], hi[idx], rng))
            gen_t = np.concatenate(times)
            total = len(gen_t)
            while len(gen_t) and total < max_events:
                gen_m = sample_gr(len(gen_t), self.mc, self.delta_m, self.beta, rng, self.m_max)
                span = np.minimum(d1 - gen_t, self.max_lag)
                n_off = rng.poisson(k * np.exp(alpha * (gen_m - self.mc)) * self.omori_integral(0.0, span, c, p))
                idx = np.repeat(np.arange(len(n_off)), n_off)
                gen_t = gen_t[idx] + self._sample_times(np.zeros(len(idx)), span[idx], rng)
                total += len(gen_t)
            counts[s] = min(total, max_events)
        return counts
