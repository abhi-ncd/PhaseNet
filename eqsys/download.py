"""Stage 1: download continuous waveforms and station metadata from FDSN services.

Stations are selected from the providers' inventories (one instrument per station, chosen by
`channel_priorities`, optionally limited to the `max_stations` nearest the region centre), then
waveforms are requested one station and one time chunk at a time with a timeout and retries, so a
dropped connection costs one request instead of stalling the whole download. Existing files are
skipped, so an interrupted download can simply be rerun.

Outputs in <run_dir>:
    stations.xml        StationXML with responses for the selected channels (for response removal)
    stations.csv        one row per PhaseNet station id (NET.STA.LOC.CH, e.g. CI.CCC..HH)
    waveforms/<chunk>/<station_id>.mseed   3-component files, one per station per time chunk
    mseed_list.csv      `fname` column relative to waveforms/, input for phasenet/predict.py
"""

import fnmatch
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import obspy
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.geodetics import gps2dist_azimuth

from .config import center, path, projection

logger = logging.getLogger(__name__)


def fetch_inventory(config):
    dl, r = config["download"], config["region"]
    inventories = {}
    for provider in dl["providers"]:
        try:
            inventories[provider] = Client(provider, timeout=dl.get("timeout", 120)).get_stations(
                starttime=UTCDateTime(config["starttime"]),
                endtime=UTCDateTime(config["endtime"]),
                network=dl["network"], station=dl["station"], location=dl["location"], channel=dl["channel"],
                minlatitude=r["minlatitude"], maxlatitude=r["maxlatitude"],
                minlongitude=r["minlongitude"], maxlongitude=r["maxlongitude"],
                level="response",
            )
        except Exception as e:
            logger.warning(f"station query to {provider} failed: {e}")
    return inventories


def select_instruments(inventories, config):
    """One (provider, net, sta, loc, band+instrument) per station, by channel then location priority."""
    dl = config["download"]
    lat0, lon0 = center(config)
    chosen = {}
    for provider, inv in inventories.items():
        for net in inv:
            for sta in net:
                key = (net.code, sta.code)
                if key in chosen:
                    continue  # the first provider that serves a station wins
                codes = {(c.location_code, c.code) for c in sta.channels}
                locations = dl["location_priorities"] + sorted({l for l, _ in codes} - set(dl["location_priorities"]))
                # candidates in priority order; take the first with three components, else the fullest
                candidates = []
                for pattern in dl["channel_priorities"]:
                    for loc in locations:
                        comps = sorted(c for l, c in codes if l == loc and fnmatch.fnmatch(c, pattern))
                        if comps:
                            candidates.append((len({c[-1] for c in comps}), loc, comps[0][:-1]))
                if not candidates:
                    continue
                full = [c for c in candidates if c[0] >= 3]
                _, loc, instrument = full[0] if full else max(candidates, key=lambda c: c[0])
                pick = (loc, instrument)
                dist = gps2dist_azimuth(lat0, lon0, sta.latitude, sta.longitude)[0]
                chosen[key] = {"provider": provider, "network": net.code, "station": sta.code,
                               "location": pick[0], "instrument": pick[1], "distance_m": dist}
    rows = sorted(chosen.values(), key=lambda x: x["distance_m"])
    if dl.get("max_stations"):
        rows = rows[: dl["max_stations"]]
    return rows


def selected_inventory(inventories, selection):
    out = obspy.Inventory()
    for s in selection:
        inv = inventories[s["provider"]].select(network=s["network"], station=s["station"],
                                                location=s["location"], channel=s["instrument"] + "?")
        out += inv
    return out


def time_chunks(config):
    t0, t_end = UTCDateTime(config["starttime"]), UTCDateTime(config["endtime"])
    step = config["download"]["chunk_seconds"]
    while t0 < t_end:
        yield t0, min(t0 + step, t_end)
        t0 += step


def chunk_label(t):
    return UTCDateTime(t).strftime("%Y%m%dT%H%M%S")


def download_one(client, s, t0, t1, out, retries=3):
    if os.path.exists(out):
        return out
    for attempt in range(retries):
        try:
            st = client.get_waveforms(s["network"], s["station"], s["location"] or "--", s["instrument"] + "?", t0, t1)
            st.merge(fill_value=0)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            st.write(out + ".part", format="MSEED")
            os.replace(out + ".part", out)  # never leave a truncated file that a rerun would skip
            return out
        except FDSNNoDataException:
            return None
        except Exception as e:
            logger.warning(f"{s['network']}.{s['station']} {t0}: attempt {attempt + 1} failed: {e}")
            time.sleep(2 * (attempt + 1))
    return None


def download_waveforms(config, selection):
    dl = config["download"]
    wave_dir = path(config, "waveforms")
    clients = {p: Client(p, timeout=dl.get("timeout", 120)) for p in {s["provider"] for s in selection}}
    jobs = []
    for t0, t1 in time_chunks(config):
        for s in selection:
            sid = f"{s['network']}.{s['station']}.{s['location']}.{s['instrument']}"
            jobs.append((clients[s["provider"]], s, t0, t1, os.path.join(wave_dir, chunk_label(t0), f"{sid}.mseed")))
    with ThreadPoolExecutor(max_workers=dl.get("threads", 4)) as pool:
        results = list(pool.map(lambda j: download_one(*j), jobs))
    fnames = sorted(os.path.relpath(f, wave_dir) for f in results if f)
    pd.DataFrame({"fname": fnames}).to_csv(path(config, "mseed_list.csv"), index=False)
    logger.info(f"wrote {len(fnames)} of {len(jobs)} station-chunk files")
    return fnames


def build_station_table(inventory, proj):
    """One row per PhaseNet station id (NET.STA.LOC.<band+instrument>), matching `tr.id[:-1]`."""
    rows = {}
    for net in inventory:
        for sta in net:
            for cha in sta:
                sid = f"{net.code}.{sta.code}.{cha.location_code}.{cha.code[:-1]}"
                if sid in rows:
                    rows[sid]["component"] = "".join(sorted(set(rows[sid]["component"] + cha.code[-1])))
                    continue
                rows[sid] = {
                    "station_id": sid,
                    "network": net.code,
                    "station": sta.code,
                    "location": cha.location_code,
                    "instrument": cha.code[:-1],
                    "component": cha.code[-1],
                    "latitude": cha.latitude if cha.latitude is not None else sta.latitude,
                    "longitude": cha.longitude if cha.longitude is not None else sta.longitude,
                    "elevation_m": cha.elevation if cha.elevation is not None else sta.elevation,
                    "depth_m": cha.depth or 0.0,
                }
    df = pd.DataFrame(list(rows.values()))
    if df.empty:
        return df
    x, y = proj(df["longitude"].values, df["latitude"].values)
    df["x_km"] = np.round(x, 3)
    df["y_km"] = np.round(y, 3)
    # depth positive down: sensor sits `depth_m` below the surface at `elevation_m`
    df["z_km"] = np.round((-df["elevation_m"] + df["depth_m"]) / 1e3, 4)
    return df.sort_values("station_id").reset_index(drop=True)


def run(config):
    inventories = fetch_inventory(config)
    selection = select_instruments(inventories, config)
    if not selection:
        raise RuntimeError("no stations found; check region, time window, network and channel settings")
    logger.info(f"selected {len(selection)} stations: " +
                ", ".join(f"{s['network']}.{s['station']}.{s['instrument']}" for s in selection))
    inventory = selected_inventory(inventories, selection)
    inventory.write(path(config, "stations.xml"), format="STATIONXML")
    fnames = download_waveforms(config, selection)
    station_ids = {os.path.splitext(os.path.basename(f))[0] for f in fnames}
    df = build_station_table(inventory, projection(config))
    df = df[df["station_id"].isin(station_ids)]
    df.to_csv(path(config, "stations.csv"), index=False)
    logger.info(f"wrote {len(df)} stations to stations.csv")
