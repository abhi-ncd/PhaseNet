"""Stage 4: absolute earthquake location with ADLoc (AI4EPS).

ADLoc locates each GaMMA event with RANSAC over its picks (robust to wrong associations), using
travel times from a 2D eikonal solution of a layered 1D velocity model. It is iterated a few times
while updating P and S station-term corrections (mean inlier residual per station).

Reads  <run_dir>/events_gamma.csv, picks_gamma.csv, stations.csv
Writes <run_dir>/events_adloc.csv   event_index, time, latitude, longitude, depth_km, adloc_score,
                                    adloc_residual_time, num_picks, ...
       <run_dir>/picks_adloc.csv    associated picks with residual_time and mask (1 = inlier)
       <run_dir>/stations_adloc.csv stations with station_term_time_p / station_term_time_s
"""

import logging

import numpy as np
import pandas as pd

from .associate import region_bounds_km
from .config import path, projection

logger = logging.getLogger(__name__)

PHASE_INDEX = {"P": 0, "S": 1}


def adloc_config(config, stations):
    lc = config["locate"]
    xlim, ylim = region_bounds_km(config)
    zmax = lc["z_range_km"][1]
    # travel times use depth relative to the station, so the grid must also cover station elevations
    zspan = zmax + max(0.0, -float(stations["z_km"].min())) + lc["h"]
    vm = lc["velocity_model"]
    ac = {
        "xlim_km": xlim,
        "ylim_km": ylim,
        "zlim_km": (0.0, zspan),
        "bfgs_bounds": ((xlim[0] - 1, xlim[1] + 1), (ylim[0] - 1, ylim[1] + 1), (0, zmax + 1), (None, None)),
        "vel": {0: vm["P"][0], 1: vm["S"][0]},  # only used without eikonal tables
        "use_amplitude": False,
        "min_picks": lc["min_picks"],
        "min_picks_ratio": lc["min_picks_ratio"],
        "min_p_picks": lc["min_p_picks"],
        "min_s_picks": lc["min_s_picks"],
        "max_residual_time": lc["max_residual_time"],
        "min_score": lc["min_score"],
        "ncpu": lc["ncpu"],
        "proj": projection(config),
    }
    eikonal = {
        "vel": {"Z": list(vm["Z"]), "P": list(vm["P"]), "S": list(vm["S"])},
        "h": lc["h"],
        "xlim_km": ac["xlim_km"],
        "ylim_km": ac["ylim_km"],
        "zlim_km": ac["zlim_km"],
    }
    return ac, eikonal


def prepare_inputs(events, picks, stations):
    stations = stations.copy().reset_index(drop=True)
    stations["idx_sta"] = np.arange(len(stations))

    picks = picks[picks["event_index"] >= 0].copy()
    picks = picks[picks["station_id"].isin(stations["station_id"])]
    picks["phase_time"] = pd.to_datetime(picks["phase_time"])
    picks["phase_type"] = picks["phase_type"].str.upper().map(PHASE_INDEX)

    events = events.copy()
    events["time"] = pd.to_datetime(events["time"])
    events["idx_eve"] = np.arange(len(events))
    mapping = dict(zip(events["event_index"], events["idx_eve"]))
    picks["idx_eve"] = picks["event_index"].map(mapping)
    picks = picks.dropna(subset=["idx_eve"])
    picks["idx_eve"] = picks["idx_eve"].astype(int)
    picks = picks.merge(stations[["station_id", "idx_sta"]], on="station_id")
    return events, picks, stations


def update_station_terms(picks, stations):
    inliers = picks[picks["mask"] == 1]
    for phase, col in [(0, "station_term_time_p"), (1, "station_term_time_s")]:
        mean_res = inliers[inliers["phase_type"] == phase].groupby("idx_sta")["residual_time"].mean()
        idx = stations["idx_sta"].map(mean_res).fillna(0.0)
        stations[col] = stations[col] + idx.values
    return stations


def locate(events, picks, stations, config, iterations=3):
    from adloc.eikonal2d import init_eikonal2d
    from adloc.sacloc2d import ADLoc
    from adloc.utils import invert_location

    ac, eikonal = adloc_config(config, stations)
    eikonal = init_eikonal2d(eikonal)
    events, picks, stations = prepare_inputs(events, picks, stations)
    stations["station_term_time_p"] = 0.0
    stations["station_term_time_s"] = 0.0
    if len(events) == 0 or len(picks) == 0:
        return pd.DataFrame(), picks, stations

    events_init = events[["idx_eve", "event_index", "x_km", "y_km", "z_km", "time"]]
    located, picks_out = None, picks
    for it in range(iterations):
        estimator = ADLoc(ac, eikonal=eikonal)
        picks_out, located = invert_location(picks, stations, ac, estimator, events_init=events_init, iter=it)
        if located is None:
            logger.warning("ADLoc located no events")
            return pd.DataFrame(), picks, stations
        stations = update_station_terms(picks_out, stations)
        events_init = located[["idx_eve", "event_index", "x_km", "y_km", "z_km", "time"]]

    located = located.merge(events[["idx_eve", "magnitude"]].rename(columns={"magnitude": "magnitude_gamma"}), on="idx_eve")
    located = located.drop(columns=["idx_eve"]).sort_values("time").reset_index(drop=True)
    for c in ["latitude", "longitude"]:
        located[c] = located[c].round(5)
    located["depth_km"] = located["depth_km"].round(3)
    return located, picks_out, stations


def run(config):
    events = pd.read_csv(path(config, "events_gamma.csv"))
    picks = pd.read_csv(path(config, "picks_gamma.csv"))
    stations = pd.read_csv(path(config, "stations.csv"), keep_default_na=False, dtype={"location": str})
    located, picks_out, stations = locate(events, picks, stations, config)
    located.to_csv(path(config, "events_adloc.csv"), index=False)
    if len(picks_out):
        picks_out = picks_out.copy()
        picks_out["phase_type"] = picks_out["phase_type"].map({0: "P", 1: "S"})
    picks_out.to_csv(path(config, "picks_adloc.csv"), index=False)
    stations.to_csv(path(config, "stations_adloc.csv"), index=False)
    logger.info(f"ADLoc: located {len(located)} of {len(events)} events")
    return located
