"""Stage 3: phase association with GaMMA (Zhu et al., 2022, JGR).

GaMMA treats picks at all stations as samples from a Bayesian Gaussian mixture whose components
are earthquakes, and jointly fits origin time, hypocentre and (with amplitudes) magnitude.

Reads  <run_dir>/picks.csv, stations.csv
Writes <run_dir>/events_gamma.csv   event_index, time, x_km, y_km, z_km, latitude, longitude, depth_km, magnitude, ...
       <run_dir>/picks_gamma.csv    picks.csv plus event_index (-1 = unassociated) and gamma_score
"""

import logging

import numpy as np
import pandas as pd

from .config import path, projection

logger = logging.getLogger(__name__)


def region_bounds_km(config):
    r = config["region"]
    proj = projection(config)
    lons = [r["minlongitude"], r["maxlongitude"], r["minlongitude"], r["maxlongitude"]]
    lats = [r["minlatitude"], r["minlatitude"], r["maxlatitude"], r["maxlatitude"]]
    x, y = proj(lons, lats)
    return (float(min(x)), float(max(x))), (float(min(y)), float(max(y)))


def gamma_config(config, stations):
    from gamma.utils import estimate_eps

    ac = config["associate"]
    xlim, ylim = region_bounds_km(config)
    zlim = tuple(ac["z_range_km"])
    gc = {
        "dims": ["x(km)", "y(km)", "z(km)"],
        "use_dbscan": ac["use_dbscan"],
        "use_amplitude": ac["use_amplitude"],
        "method": ac["method"],
        "oversample_factor": ac["oversample_factor"],
        "vel": dict(ac["vel"]),
        "x(km)": xlim,
        "y(km)": ylim,
        "z(km)": zlim,
        "bfgs_bounds": ((xlim[0] - 1, xlim[1] + 1), (ylim[0] - 1, ylim[1] + 1), (0, zlim[1] + 1), (None, None)),
        "dbscan_min_samples": ac["dbscan_min_samples"],
        "min_picks_per_eq": ac["min_picks_per_eq"],
        "min_p_picks_per_eq": ac["min_p_picks_per_eq"],
        "min_s_picks_per_eq": ac["min_s_picks_per_eq"],
        "max_sigma11": ac["max_sigma11"],
        "max_sigma22": ac["max_sigma22"],
        "max_sigma12": ac["max_sigma12"],
        "ncpu": ac["ncpu"],
    }
    gc["dbscan_eps"] = ac["dbscan_eps"] or float(estimate_eps(stations, gc["vel"]["p"]))
    return gc


def to_gamma_inputs(picks, stations, use_amplitude):
    st = stations.rename(columns={"station_id": "id", "x_km": "x(km)", "y_km": "y(km)", "z_km": "z(km)"})
    st = st[["id", "x(km)", "y(km)", "z(km)"]]
    pk = picks[picks["station_id"].isin(st["id"])].copy()
    if use_amplitude:
        pk = pk[np.isfinite(pk["phase_amplitude"]) & (pk["phase_amplitude"] > 0)]
    pk = pk.reset_index().rename(columns={"index": "pick_index"})
    gp = pd.DataFrame(
        {
            "id": pk["station_id"],
            "timestamp": pd.to_datetime(pk["phase_time"]),
            "type": pk["phase_type"].str.lower(),
            "prob": pk["phase_score"],
        }
    )
    if use_amplitude:
        gp["amp"] = pk["phase_amplitude"]
    return gp, st, pk["pick_index"].to_numpy()


def associate(picks, stations, config):
    from gamma.utils import association

    gc = gamma_config(config, stations.rename(columns={"x_km": "x(km)", "y_km": "y(km)", "z_km": "z(km)"}))
    gp, st, original_index = to_gamma_inputs(picks, stations, gc["use_amplitude"])
    logger.info(f"associating {len(gp)} picks on {gp['id'].nunique()} stations (eps={gc['dbscan_eps']:.1f}s)")
    events, assignment = association(gp, st, gc, method=gc["method"])

    events = pd.DataFrame(events)
    picks = picks.copy()
    picks["event_index"] = -1
    picks["gamma_score"] = 0.0
    if len(events) == 0:
        logger.warning("GaMMA associated no events")
        return events, picks

    proj = projection(config)
    events = events.rename(columns={"x(km)": "x_km", "y(km)": "y_km", "z(km)": "z_km"})
    lon, lat = proj(events["x_km"].values, events["y_km"].values, inverse=True)
    events["longitude"] = np.round(lon, 5)
    events["latitude"] = np.round(lat, 5)
    events["depth_km"] = events["z_km"]
    if not gc["use_amplitude"]:
        events["magnitude"] = np.nan
    events = events.sort_values("time").reset_index(drop=True)

    assign = pd.DataFrame(assignment, columns=["gamma_pick", "event_index", "gamma_score"])
    rows = original_index[assign["gamma_pick"].to_numpy()]
    picks.loc[rows, "event_index"] = assign["event_index"].to_numpy()
    picks.loc[rows, "gamma_score"] = assign["gamma_score"].to_numpy()
    return events, picks


def run(config):
    picks = pd.read_csv(path(config, "picks.csv"))
    stations = pd.read_csv(path(config, "stations.csv"), keep_default_na=False, dtype={"location": str})
    events, picks = associate(picks, stations, config)
    events.to_csv(path(config, "events_gamma.csv"), index=False)
    picks.to_csv(path(config, "picks_gamma.csv"), index=False)
    logger.info(f"GaMMA: {len(events)} events, {(picks.event_index >= 0).sum()} of {len(picks)} picks associated")
    return events, picks
