"""Forecasting models: exact likelihoods, ETAS parameter recovery, NTPP consistency."""

import numpy as np
import pandas as pd
import pytest
from scipy import integrate

from eqsys.forecast.data import Split
from eqsys.forecast.ntpp_model import NeuralTPP, RecastNet
from eqsys.forecast.temporal import PoissonModel, TemporalETAS, sample_gr

MC, DM = 2.0, 0.1
TRUE_THETA = np.array([np.log10(0.5), np.log10(0.01), 1.8, np.log10(0.01), 1.15])  # mu=0.5/day


def simulate_etas(theta, T, rng, max_lag=365.0):
    """Branching-process simulation of the temporal ETAS model used by TemporalETAS."""
    model = TemporalETAS(max_lag=max_lag)
    model.theta = theta
    log10_mu, log10_k, alpha, log10_c, p = theta
    mu, k, c = 10**log10_mu, 10**log10_k, 10**log10_c
    beta = 1.0 * np.log(10)
    gen_t = rng.random(rng.poisson(mu * T)) * T
    all_t, all_m = [], []
    while len(gen_t):
        gen_m = sample_gr(len(gen_t), MC, DM, beta, rng)
        all_t.append(gen_t)
        all_m.append(gen_m)
        span = np.minimum(T - gen_t, max_lag)
        n = rng.poisson(k * np.exp(alpha * (gen_m - MC)) * model.omori_integral(0.0, span, c, p))
        idx = np.repeat(np.arange(len(n)), n)
        gen_t = gen_t[idx] + model._sample_times(np.zeros(len(idx)), span[idx], rng)
    t, m = np.concatenate(all_t), np.concatenate(all_m)
    order = np.argsort(t)
    return t[order], m[order]


def make_split(t, m, T):
    cat = pd.DataFrame({"time": pd.Timestamp("2000-01-01") + pd.to_timedelta(t, unit="D"), "magnitude": m,
                        "latitude": 0.0, "longitude": 0.0, "depth_km": 5.0})
    return Split(pd.Timestamp("2000-01-01"), MC, DM, t, m, cat, (0, 0.6 * T), (0.6 * T, 0.8 * T), (0.8 * T, T))


@pytest.fixture(scope="module")
def etas_split():
    rng = np.random.default_rng(3)
    T = 3000.0
    t, m = simulate_etas(TRUE_THETA, T, rng)
    return make_split(t, m, T)


def test_poisson_log_likelihood_is_analytic():
    t = np.sort(np.random.default_rng(0).random(500) * 100)
    split = make_split(t, np.full(500, 2.5), 100.0)
    model = PoissonModel().fit(split)
    ll, n = model.log_likelihood(t, None, 80.0, 100.0)
    mu = np.sum(t < 60) / 60.0
    assert ll == pytest.approx(n * np.log(mu) - mu * 20.0)


def test_etas_likelihood_matches_numerical_integral():
    rng = np.random.default_rng(1)
    t, m = simulate_etas(TRUE_THETA, 200.0, rng)
    model = TemporalETAS()
    model.theta, model.mc = TRUE_THETA, MC
    ll, _ = model.log_likelihood(t, m, 50.0, 200.0)

    mu, k, alpha, c, p = 10 ** TRUE_THETA[0], 10 ** TRUE_THETA[1], TRUE_THETA[2], 10 ** TRUE_THETA[3], TRUE_THETA[4]

    def lam(s):
        prev = t < s
        return mu + np.sum(k * np.exp(alpha * (m[prev] - MC)) * (s - t[prev] + c) ** (-p))

    sel = (t >= 50) & (t < 200)
    edges = np.r_[50.0, t[sel], 200.0]
    integral = sum(integrate.quad(lam, a, b, limit=200)[0] for a, b in zip(edges[:-1], edges[1:]) if b > a)
    expected = np.sum([np.log(lam(s)) for s in t[sel]]) - integral
    assert ll == pytest.approx(expected, rel=1e-3)


def test_etas_recovers_parameters(etas_split):
    model = TemporalETAS().fit(etas_split, n_starts=2)
    a, b = etas_split.t_train
    d = model._prepare(etas_split.t, etas_split.m, a, b)
    # maximum likelihood: fitted parameters are at least as likely as the true ones
    assert model._log_likelihood(model.theta, d) >= model._log_likelihood(TRUE_THETA, d) - 1e-6
    true = TemporalETAS()
    true.theta, true.beta = TRUE_THETA, model.beta
    assert model.branching_ratio() == pytest.approx(true.branching_ratio(), abs=0.15)
    assert 10 ** model.params["log10_mu"] == pytest.approx(0.5, rel=0.3)
    assert model.b == pytest.approx(1.0, abs=0.1)

    # ETAS should beat Poisson out of sample on clustered data
    pois = PoissonModel().fit(etas_split)
    ll_e, n = model.log_likelihood(etas_split.t, etas_split.m, *etas_split.t_test)
    ll_p, _ = pois.log_likelihood(etas_split.t, etas_split.m, *etas_split.t_test)
    assert ll_e > ll_p

    # simulated daily counts are centred on a sensible value on a quiet day
    counts = model.simulate_counts(etas_split.t, etas_split.m, 2500.0, 2501.0, 200, np.random.default_rng(0))
    assert counts.mean() >= 0.5 * 10 ** model.params["log10_mu"]


def test_weibull_mixture_is_a_density():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    net = RecastNet(hidden=8, n_mix=4)
    h = torch.randn(1, 8)
    grid = torch.logspace(-8, 4, 20000)
    pdf = torch.exp(net.log_pdf(grid, h.expand(len(grid), -1))).detach().numpy()
    mass = np.trapz(pdf, grid.numpy())
    surv_end = torch.exp(net.log_survival(grid[-1:], h)).item()
    assert mass + surv_end == pytest.approx(1.0, abs=0.02)


def test_ntpp_window_likelihood_is_additive(etas_split):
    model = NeuralTPP(hidden=8, n_mix=4, epochs=5, lr=1e-2).fit(etas_split)
    a, b = etas_split.t_test
    mid = (a + b) / 2
    ll_ab, n_ab = model.log_likelihood(etas_split.t, etas_split.m, a, b)
    ll_1, n_1 = model.log_likelihood(etas_split.t, etas_split.m, a, mid)
    ll_2, n_2 = model.log_likelihood(etas_split.t, etas_split.m, mid, b)
    assert n_ab == n_1 + n_2
    assert ll_ab == pytest.approx(ll_1 + ll_2, rel=1e-4, abs=1e-2)

    counts = model.simulate_counts(etas_split.t, etas_split.m, a, a + 1.0, 50, np.random.default_rng(0))
    assert counts.shape == (50,) and counts.min() >= 0
