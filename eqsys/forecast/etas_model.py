"""Spatio-temporal ETAS map forecast with the `etas` package (Mizrahi, Nandan & Wiemer, 2021).

Parameters are inverted on the training + validation windows, then `n_simulations` catalogs are
simulated for the `horizon_days` following the end of the validation window (the start of the
test window). The result is summarised as expected counts per 0.1 degree cell and the probability of
at least one event above several magnitude thresholds, and checked with a catalog-based number
test (pyCSEP convention) against what was observed.
"""

import datetime as dt
import logging

import numpy as np
import pandas as pd

from ..config import path
from .data import forecast_region

logger = logging.getLogger(__name__)


def region_polygon(region):
    r = region
    return np.array([
        [r["minlatitude"], r["minlongitude"]],
        [r["maxlatitude"], r["minlongitude"]],
        [r["maxlatitude"], r["maxlongitude"]],
        [r["minlatitude"], r["maxlongitude"]],
    ])


def fit_and_simulate(split, config):
    from etas.inversion import ETASParameterCalculation
    from etas.simulation import ETASSimulation

    fc = config["forecast"]
    cat = split.catalog[["time", "latitude", "longitude", "magnitude"]].copy()
    cat.index.name = "id"
    start = split.origin
    t_end = split.origin + pd.Timedelta(days=split.t_valid[1])
    metadata = {
        "catalog": cat,
        "auxiliary_start": start,
        # the first year only acts as sources, so early targets are not missing their triggers
        "timewindow_start": start + pd.Timedelta(days=365),
        "timewindow_end": t_end,
        "mc": split.mc,
        "m_ref": split.mc,
        "delta_m": split.delta_m,
        "coppersmith_multiplier": 100,
        "shape_coords": region_polygon(forecast_region(config)),
        "name": config["name"],
    }
    inversion = ETASParameterCalculation(metadata)
    inversion.prepare()
    theta = inversion.invert()
    logger.info(f"spatio-temporal ETAS parameters: {theta}")

    sim = ETASSimulation(inversion, m_max=9.0)
    sim.prepare()
    catalogs = sim.simulate_to_df(forecast_n_days=fc["horizon_days"], n_simulations=fc["n_simulations"],
                                  m_threshold=split.mc)
    return inversion, catalogs, t_end


def summarise(catalogs, observed, n_sim, region, mc, cell=0.1):
    n_sim_counts = catalogs.groupby("catalog_id").size().reindex(range(n_sim), fill_value=0).values
    n_obs = len(observed)
    summary = {
        "n_observed": int(n_obs),
        "n_forecast_mean": float(n_sim_counts.mean()),
        "n_forecast_q05_q95": [float(np.quantile(n_sim_counts, 0.05)), float(np.quantile(n_sim_counts, 0.95))],
        # catalog-based N-test quantiles (Savran et al., 2020)
        "n_test_delta1": float(np.mean(n_sim_counts >= n_obs)),
        "n_test_delta2": float(np.mean(n_sim_counts <= n_obs)),
    }
    for m in [mc, 4.0, 5.0, 6.0]:
        if m < mc:
            continue
        hits = catalogs[catalogs["magnitude"] >= m].groupby("catalog_id").size()
        summary[f"P(at least one M>={m:g})"] = float(len(hits) / n_sim)

    lat_edges = np.arange(region["minlatitude"], region["maxlatitude"] + cell, cell)
    lon_edges = np.arange(region["minlongitude"], region["maxlongitude"] + cell, cell)
    grid, _, _ = np.histogram2d(catalogs["latitude"], catalogs["longitude"], [lat_edges, lon_edges])
    return summary, grid / n_sim, lat_edges, lon_edges


def plot_map(rate, lat_edges, lon_edges, observed, fname, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig, ax = plt.subplots(figsize=(5.5, 5))
    positive = rate[rate > 0]
    norm = LogNorm(vmin=max(positive.min(), 1e-4), vmax=positive.max()) if len(positive) else None
    pc = ax.pcolormesh(lon_edges, lat_edges, np.where(rate > 0, rate, np.nan), cmap="viridis", norm=norm)
    fig.colorbar(pc, ax=ax, label="expected events per cell")
    if len(observed):
        ax.scatter(observed["longitude"], observed["latitude"], s=8, c="r", marker="x", label="observed")
        ax.legend(loc="lower left", fontsize=8)
    ax.set_aspect(1 / np.cos(np.deg2rad(np.mean(lat_edges))))
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def run(split, config):
    fc = config["forecast"]
    inversion, catalogs, t_start = fit_and_simulate(split, config)
    t_stop = t_start + pd.Timedelta(days=fc["horizon_days"])
    obs = split.catalog[(split.catalog["time"] >= t_start) & (split.catalog["time"] < t_stop)]
    region = forecast_region(config)
    summary, rate, lat_edges, lon_edges = summarise(catalogs, obs, fc["n_simulations"], region, split.mc)
    summary.update({
        "forecast_start": str(t_start), "forecast_end": str(t_stop),
        "parameters": {k: float(v) for k, v in inversion.theta.items()},
    })
    catalogs.drop(columns=["geometry"], errors="ignore").to_csv(path(config, "forecast", "etas_simulations.csv"))
    plot_map(rate, lat_edges, lon_edges, obs, path(config, "forecast", "etas_map.png"),
             f"ETAS forecast M>={split.mc:g}, {t_start:%Y-%m-%d} + {fc['horizon_days']} d")
    return summary
