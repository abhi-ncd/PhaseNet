"""Stage 5: local magnitude ML from simulated Wood-Anderson amplitudes.

For every located event and every station with inlier picks, the horizontal traces are cut from
the P pick to after the S pick, the instrument response is removed to displacement, a Wood-Anderson
seismometer (T0 = 0.8 s, h = 0.7, static gain 2080) is simulated, and the peak amplitude A (mm) gives
a per-channel magnitude
    ML = log10(A) + a * log10(r / 100) + b * (r - 100) + c
with hypocentral distance r in km. The defaults are Hutton & Boore (1987) for southern California
(a=1.110, b=0.00189, c=3.0); other regions should set their own coefficients in the config.
The event magnitude is the median over channels, with a scaled MAD as its uncertainty.

Reads  <run_dir>/events_adloc.csv, picks_adloc.csv, stations.csv, stations.xml, waveforms/
Writes <run_dir>/events_mag.csv, station_magnitudes.csv
"""

import glob
import logging
import os
from functools import lru_cache

import numpy as np
import obspy
import pandas as pd
from obspy import UTCDateTime

from .config import path

logger = logging.getLogger(__name__)

HUTTON_BOORE_1987 = {"a": 1.110, "b": 0.00189, "c": 3.0}
PAD_SECONDS = 10.0  # extra data on both sides of the window, discarded after filtering


def wood_anderson_paz(gain=2080.0, period=0.8, damping=0.7):
    w0 = 2 * np.pi / period
    re, im = -damping * w0, w0 * np.sqrt(1 - damping**2)
    return {"poles": [complex(re, im), complex(re, -im)], "zeros": [0j, 0j], "gain": 1.0, "sensitivity": gain}


def local_magnitude(amplitude_mm, distance_km, a=1.110, b=0.00189, c=3.0):
    amplitude_mm = np.asarray(amplitude_mm, dtype=float)
    r = np.asarray(distance_km, dtype=float)
    return np.log10(amplitude_mm) + a * np.log10(r / 100.0) + b * (r - 100.0) + c


def robust_event_magnitude(station_mags):
    m = np.asarray(station_mags, dtype=float)
    m = m[np.isfinite(m)]
    if len(m) == 0:
        return np.nan, np.nan
    med = np.median(m)
    mad = 1.4826 * np.median(np.abs(m - med)) if len(m) > 1 else np.nan
    return med, mad


class WaveformIndex:
    """Finds the 3-component chunk files of a station that overlap a time window."""

    def __init__(self, wave_dir, chunk_seconds):
        self.chunk_seconds = chunk_seconds
        self.files = {}
        for f in glob.glob(os.path.join(wave_dir, "*", "*.mseed")):
            chunk = UTCDateTime(os.path.basename(os.path.dirname(f)))
            sid = os.path.splitext(os.path.basename(f))[0]
            self.files.setdefault(sid, []).append((chunk, f))

    def find(self, station_id, t0, t1):
        return [f for c, f in self.files.get(station_id, []) if c < t1 and c + self.chunk_seconds > t0]


@lru_cache(maxsize=128)
def _read(fname):
    return obspy.read(fname)


def cut(files, t0, t1):
    st = obspy.Stream()
    for f in files:
        st += _read(f).slice(t0, t1).copy()
    if len(st):
        st.merge(fill_value=0)
    return st


def wood_anderson_amplitudes(st, inventory, paz, t0=None, t1=None):
    """Peak Wood-Anderson amplitude (mm) of every horizontal channel, measured within [t0, t1]."""
    out = {}
    for tr in st:
        if tr.stats.channel[-1] in "ZU" or tr.stats.npts < 50:
            continue
        tr = tr.copy()
        nyq = tr.stats.sampling_rate / 2
        try:
            tr.detrend("demean").taper(0.05)
            tr.remove_response(inventory=inventory, output="DISP", pre_filt=(0.1, 0.2, 0.8 * nyq, 0.9 * nyq))
            tr.simulate(paz_simulate=paz, simulate_sensitivity=True)
        except Exception as e:
            logger.debug(f"{tr.id}: {e}")
            continue
        if t0 is not None:
            tr.trim(t0, t1)
        if tr.stats.npts == 0:
            continue
        out[tr.id] = float(np.max(np.abs(tr.data))) * 1e3  # m -> mm
    return out


def measure_event(event, picks, stations, index, inventory, config, paz, coeffs):
    mc = config["magnitude"]
    origin = UTCDateTime(pd.Timestamp(event["time"]).to_pydatetime())
    rows = []
    for sid, pk in picks.groupby("station_id"):
        if sid not in stations.index:
            continue
        sta = stations.loc[sid]
        r = np.sqrt(
            (event["x_km"] - sta["x_km"]) ** 2 + (event["y_km"] - sta["y_km"]) ** 2 + (event["z_km"] - sta["z_km"]) ** 2
        )
        if r > mc["max_distance_km"] or r < 1e-3:
            continue
        tp = pk.loc[pk["phase_type"] == "P", "phase_time"]
        ts = pk.loc[pk["phase_type"] == "S", "phase_time"]
        t_p = UTCDateTime(tp.iloc[0]) if len(tp) else None
        t_s = UTCDateTime(ts.iloc[0]) if len(ts) else None
        if t_s is None:
            t_s = origin + 1.73 * ((t_p - origin) if t_p is not None else r / 6.0)
        t0 = (t_p if t_p is not None else origin) - mc["pre_p_seconds"]
        t1 = t_s + mc["post_s_seconds"]
        pad = PAD_SECONDS
        st = cut(index.find(sid, t0 - pad, t1 + pad), t0 - pad, t1 + pad)
        for cha, amp in wood_anderson_amplitudes(st, inventory, paz, t0, t1).items():
            if amp > 0:
                ml = float(local_magnitude(amp, r, **coeffs))
                rows.append({"event_index": event["event_index"], "channel_id": cha, "station_id": sid,
                             "distance_km": round(float(r), 3), "wa_amplitude_mm": amp, "magnitude": round(ml, 3)})
    return rows


def compute_magnitudes(events, picks, stations, inventory, config):
    mc = config["magnitude"]
    coeffs = {k: mc.get(k, v) for k, v in HUTTON_BOORE_1987.items()}
    paz = wood_anderson_paz(mc["wood_anderson_gain"])
    index = WaveformIndex(path(config, "waveforms"), config["download"]["chunk_seconds"])
    stations = stations.set_index("station_id")
    picks = picks[picks["mask"] == 1] if "mask" in picks.columns else picks
    by_event = dict(tuple(picks.groupby("event_index")))

    station_rows = []
    for _, event in events.iterrows():
        pk = by_event.get(event["event_index"])
        if pk is not None:
            station_rows += measure_event(event, pk, stations, index, inventory, config, paz, coeffs)
    station_mags = pd.DataFrame(station_rows)

    events = events.copy()
    events["magnitude"] = np.nan
    events["magnitude_uncertainty"] = np.nan
    events["magnitude_num_stations"] = 0
    if len(station_mags):
        for eid, g in station_mags.groupby("event_index"):
            nsta = g["station_id"].nunique()
            if nsta < mc["min_stations"]:
                continue
            ml, mad = robust_event_magnitude(g["magnitude"])
            sel = events["event_index"] == eid
            events.loc[sel, "magnitude"] = round(ml, 2)
            events.loc[sel, "magnitude_uncertainty"] = round(mad, 2) if np.isfinite(mad) else np.nan
            events.loc[sel, "magnitude_num_stations"] = nsta
    events["magnitude_type"] = "ML"
    return events, station_mags


def run(config):
    events = pd.read_csv(path(config, "events_adloc.csv"))
    picks = pd.read_csv(path(config, "picks_adloc.csv"))
    stations = pd.read_csv(path(config, "stations.csv"), keep_default_na=False, dtype={"location": str})
    inventory = obspy.read_inventory(path(config, "stations.xml"))
    events, station_mags = compute_magnitudes(events, picks, stations, inventory, config)
    events.to_csv(path(config, "events_mag.csv"), index=False)
    station_mags.to_csv(path(config, "station_magnitudes.csv"), index=False)
    logger.info(f"ML computed for {events['magnitude'].notna().sum()} of {len(events)} events")
    return events
