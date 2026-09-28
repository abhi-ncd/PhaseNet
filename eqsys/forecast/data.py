"""Forecasting catalogs: fetch or load, apply completeness cut, and split chronologically."""

import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from obspy import UTCDateTime

from ..catalog import mc_maxc, obspy_catalog_to_df
from ..config import path

logger = logging.getLogger(__name__)


def forecast_region(config):
    return config["forecast"].get("region") or config["region"]


def fetch_reference_catalog(config):
    """Download the forecasting catalog year by year (FDSN services cap the events per query)."""
    from obspy.clients.fdsn import Client

    fc = config["forecast"]
    r = forecast_region(config)
    client = Client(fc["client"])
    t0, t_end = UTCDateTime(fc["starttime"]), UTCDateTime(fc["endtime"])
    frames = []
    while t0 < t_end:
        t1 = min(UTCDateTime(t0.year + 1, 1, 1), t_end)
        try:
            cat = client.get_events(
                starttime=t0, endtime=t1,
                minlatitude=r["minlatitude"], maxlatitude=r["maxlatitude"],
                minlongitude=r["minlongitude"], maxlongitude=r["maxlongitude"],
                minmagnitude=fc["min_magnitude"], orderby="time-asc",
            )
            frames.append(obspy_catalog_to_df(cat))
            logger.info(f"{t0.year}: {len(cat)} events")
        except Exception as e:
            if "No data" in str(e) or "204" in str(e):
                logger.info(f"{t0.year}: 0 events")
            else:
                raise
        t0 = t1
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)
    return df.sort_values("time").drop_duplicates(subset=["time", "latitude", "longitude"]).reset_index(drop=True)


def load_catalog(config, refresh=False):
    fc = config["forecast"]
    if fc["source"] == "pipeline":
        df = pd.read_csv(path(config, "catalog.csv"))
    else:
        cache = path(config, "forecast", "input_catalog.csv")
        if os.path.exists(cache) and not refresh:
            df = pd.read_csv(cache)
        else:
            df = fetch_reference_catalog(config)
            df.to_csv(cache, index=False)
    df["time"] = pd.to_datetime(df["time"])
    df = df.dropna(subset=["magnitude"]).sort_values("time").reset_index(drop=True)
    return df[["time", "latitude", "longitude", "depth_km", "magnitude"]]


@dataclass
class Split:
    """Event times in days since `origin`; magnitudes above `mc` only."""

    origin: pd.Timestamp
    mc: float
    delta_m: float
    t: np.ndarray
    m: np.ndarray
    catalog: pd.DataFrame
    t_train: tuple
    t_valid: tuple
    t_test: tuple

    def window(self, name):
        a, b = getattr(self, f"t_{name}")
        sel = (self.t >= a) & (self.t < b)
        return self.t[sel], self.m[sel]


def make_split(df, config):
    fc = config["forecast"]
    origin = pd.Timestamp(fc["starttime"])
    bounds = [pd.Timestamp(fc[k]) for k in ["starttime", "train_end", "valid_end", "endtime"]]
    if not (bounds[0] < bounds[1] < bounds[2] < bounds[3]):
        raise ValueError("forecast windows must satisfy starttime < train_end < valid_end < endtime")
    days = [(b - origin).total_seconds() / 86400.0 for b in bounds]

    train = df[(df["time"] >= bounds[0]) & (df["time"] < bounds[1])]
    mc = fc["mc"]
    if mc is None:
        mc = mc_maxc(train["magnitude"], fc["delta_m"], correction=0.2)
    mc = float(mc)
    keep = df[(df["magnitude"] >= mc - fc["delta_m"] / 2 - 1e-9) & (df["time"] >= bounds[0]) & (df["time"] < bounds[3])]
    keep = keep.reset_index(drop=True)
    t = (keep["time"] - origin).dt.total_seconds().values / 86400.0
    # nudge identical times apart so inter-event times are strictly positive
    t = t + np.arange(len(t)) * 1e-9
    split = Split(origin, mc, fc["delta_m"], t, keep["magnitude"].values.astype(float), keep,
                  (days[0], days[1]), (days[1], days[2]), (days[2], days[3]))
    for name in ["train", "valid", "test"]:
        logger.info(f"{name}: {len(split.window(name)[0])} events with M >= {mc}")
    return split
