"""Stage 7: fit forecasting models and evaluate them on the held-out test window.

Metrics (following Stockman et al., 2024 and CSEP practice):
- test log-likelihood per event and information gain per event over the Poisson model
  (all three are temporal point processes over the same events, so the numbers are comparable)
- daily number tests: for every forecast day, simulated event counts give the quantiles
  delta1 = P(N_sim >= N_obs) and delta2 = P(N_sim <= N_obs); a day fails if either is < 0.025
- a spatio-temporal ETAS map forecast for the first `horizon_days` of the test window

Writes <run_dir>/forecast/report.json, daily_counts.csv and figures.
"""

import json
import logging

import numpy as np
import pandas as pd

from ..config import path
from .data import load_catalog, make_split
from .ntpp_model import NeuralTPP
from .temporal import PoissonModel, TemporalETAS

logger = logging.getLogger(__name__)


def number_test_quantiles(sim_counts, n_obs):
    return float(np.mean(sim_counts >= n_obs)), float(np.mean(sim_counts <= n_obs))


def daily_evaluation(models, split, horizon, n_sim, seed=0):
    rng = np.random.default_rng(seed)
    a, b = split.t_test
    starts = np.arange(a, b - horizon + 1e-9, horizon)
    ntpp_H = {m.name: m.hidden_states(split.t, split.m) for m in models if isinstance(m, NeuralTPP)}
    rows = []
    for d0 in starts:
        d1 = d0 + horizon
        n_obs = int(np.sum((split.t >= d0) & (split.t < d1)))
        row = {"day_start": split.origin + pd.Timedelta(days=float(d0)), "n_observed": n_obs}
        for model in models:
            kw = {"H": ntpp_H[model.name]} if model.name in ntpp_H else {}
            sims = model.simulate_counts(split.t, split.m, d0, d1, n_sim, rng, **kw)
            d_1, d_2 = number_test_quantiles(sims, n_obs)
            row[f"{model.name}|mean"] = float(sims.mean())
            row[f"{model.name}|q05"] = float(np.quantile(sims, 0.05))
            row[f"{model.name}|q95"] = float(np.quantile(sims, 0.95))
            row[f"{model.name}|fail"] = bool(min(d_1, d_2) < 0.025)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_daily(daily, models, fname):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(models), 1, figsize=(8, 2.4 * len(models)), sharex=True, sharey=True)
    for ax, model in zip(np.atleast_1d(axes), models):
        ax.fill_between(daily["day_start"], daily[f"{model.name}|q05"] + 0.1, daily[f"{model.name}|q95"] + 0.1,
                        step="post", alpha=0.3, label="90% forecast interval")
        ax.step(daily["day_start"], daily[f"{model.name}|mean"] + 0.1, where="post", lw=1, label="forecast mean")
        ax.plot(daily["day_start"], daily["n_observed"] + 0.1, "k.", ms=3, label="observed")
        ax.set_yscale("log")
        ax.set_ylabel("events + 0.1")
        fail = daily[f"{model.name}|fail"].mean()
        ax.set_title(f"{model.name}: number test fails on {100 * fail:.1f}% of windows", fontsize=9)
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def run(config):
    fc = config["forecast"]
    df = load_catalog(config)
    split = make_split(df, config)
    t_test, m_test = split.window("test")
    if len(split.window("train")[0]) < 20 or len(t_test) == 0:
        raise ValueError("forecasting needs at least 20 training events and 1 test event above Mc; "
                         "use a longer catalog (forecast.source: reference) or lower forecast.min_magnitude")

    models = [
        PoissonModel().fit(split),
        TemporalETAS(max_lag=fc.get("max_lag_days", 365.0)).fit(split),
        NeuralTPP(**fc["ntpp"]).fit(split),
    ]

    report = {"mc": split.mc, "num_events": {k: int(len(split.window(k)[0])) for k in ["train", "valid", "test"]},
              "windows": {k: [str(split.origin + pd.Timedelta(days=x)) for x in getattr(split, f"t_{k}")]
                          for k in ["train", "valid", "test"]},
              "etas_parameters": models[1].params, "etas_branching_ratio": models[1].branching_ratio(),
              "b_value_train": models[1].b, "test": {}}
    ll_poisson = None
    for model in models:
        ll, n = model.log_likelihood(split.t, split.m, *split.t_test)
        if model.name == "Poisson":
            ll_poisson = ll
        report["test"][model.name] = {"log_likelihood": ll, "ll_per_event": ll / n,
                                      "information_gain_per_event_vs_poisson": (ll - ll_poisson) / n}
        logger.info(f"{model.name}: test LL/event {ll / n:.3f}, IG vs Poisson {(ll - ll_poisson) / n:+.3f}")

    daily = daily_evaluation(models, split, fc["horizon_days"], fc.get("n_simulations_daily", 100))
    daily.to_csv(path(config, "forecast", "daily_counts.csv"), index=False)
    for model in models:
        report["test"][model.name]["number_test_fail_fraction"] = float(daily[f"{model.name}|fail"].mean())
    plot_daily(daily, models, path(config, "forecast", "daily_forecasts.png"))

    if fc.get("spatial_etas", True):
        from . import etas_model

        try:
            report["spatial_etas"] = etas_model.run(split, config)
        except Exception as e:
            logger.exception("spatio-temporal ETAS failed")
            report["spatial_etas"] = {"error": str(e)}

    with open(path(config, "forecast", "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    logger.info("forecast report:\n" + json.dumps(report["test"], indent=2, default=float))
    return report
