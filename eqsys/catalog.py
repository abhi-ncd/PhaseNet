"""Stage 6: final catalog export, quality control and validation against a reference catalog.

- Exports catalog.csv and catalog.xml (QuakeML).
- Fetches a reference catalog (default USGS ComCat) for the same region and time window, matches
  events one-to-one within `match_time_s` and `match_distance_km`, and reports recall, the fraction
  of pipeline events that match, and location / origin-time / magnitude residuals.
- Estimates magnitude of completeness Mc (maximum curvature + correction, Wiemer & Wyss 2000) and
  the b-value (Aki 1965 / Utsu maximum likelihood with bin correction).
- Saves QC figures to <run_dir>/qc/.
"""

import json
import logging

import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth

from .config import path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- statistics


def mc_maxc(magnitudes, delta_m=0.1, correction=0.2):
    """Magnitude of completeness by maximum curvature (mode of the non-cumulative FMD) + correction."""
    m = np.asarray(magnitudes, dtype=float)
    m = m[np.isfinite(m)]
    if len(m) == 0:
        return np.nan
    bins = np.round(m / delta_m) * delta_m
    values, counts = np.unique(np.round(bins, 6), return_counts=True)
    return float(np.round(values[np.argmax(counts)] + correction, 6))


def b_value(magnitudes, mc, delta_m=0.1):
    """Aki-Utsu maximum-likelihood b-value with Shi & Bolt (1982) uncertainty."""
    m = np.asarray(magnitudes, dtype=float)
    m = m[np.isfinite(m) & (m >= mc - delta_m / 2 - 1e-9)]
    n = len(m)
    if n < 2:
        return np.nan, np.nan, n
    mean = m.mean()
    b = np.log10(np.e) / (mean - (mc - delta_m / 2))
    sigma = 2.3 * b**2 * np.sqrt(np.sum((m - mean) ** 2) / (n * (n - 1)))
    return float(b), float(sigma), n


# ---------------------------------------------------------------- reference catalog and matching


def fetch_reference(config):
    from obspy.clients.fdsn import Client

    cc = config["catalog"]
    r = config["region"]
    try:
        cat = Client(cc["reference_client"]).get_events(
            starttime=UTCDateTime(config["starttime"]),
            endtime=UTCDateTime(config["endtime"]),
            minlatitude=r["minlatitude"],
            maxlatitude=r["maxlatitude"],
            minlongitude=r["minlongitude"],
            maxlongitude=r["maxlongitude"],
            minmagnitude=cc["reference_min_magnitude"],
        )
    except Exception as e:
        logger.warning(f"reference catalog query failed: {e}")
        return pd.DataFrame(columns=["time", "latitude", "longitude", "depth_km", "magnitude", "magnitude_type"])
    return obspy_catalog_to_df(cat)


def obspy_catalog_to_df(cat):
    rows = []
    for ev in cat:
        o = ev.preferred_origin() or ev.origins[0]
        m = ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None)
        rows.append(
            {
                "time": pd.Timestamp(o.time.datetime),
                "latitude": o.latitude,
                "longitude": o.longitude,
                "depth_km": (o.depth or 0.0) / 1e3,
                "magnitude": m.mag if m else np.nan,
                "magnitude_type": m.magnitude_type if m else None,
            }
        )
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True) if rows else pd.DataFrame(rows)


def match_catalogs(detected, reference, max_dt=3.0, max_dist_km=15.0):
    """One-to-one matching; pairs are accepted greedily in order of increasing time difference."""
    if len(detected) == 0 or len(reference) == 0:
        return pd.DataFrame(columns=["det", "ref", "dt_s", "dist_km"])
    td = pd.to_datetime(detected["time"]).values.astype("datetime64[ns]").astype(np.int64) / 1e9
    tr = pd.to_datetime(reference["time"]).values.astype("datetime64[ns]").astype(np.int64) / 1e9
    order = np.argsort(td)
    td_sorted = td[order]
    candidates = []
    for j, t in enumerate(tr):
        lo, hi = np.searchsorted(td_sorted, [t - max_dt, t + max_dt])
        for k in range(lo, hi):
            i = order[k]
            dist = gps2dist_azimuth(
                reference["latitude"].iloc[j], reference["longitude"].iloc[j],
                detected["latitude"].iloc[i], detected["longitude"].iloc[i],
            )[0] / 1e3
            if dist <= max_dist_km:
                candidates.append((abs(td[i] - t), i, j, td[i] - t, dist))
    candidates.sort()
    used_d, used_r, pairs = set(), set(), []
    for _, i, j, dt, dist in candidates:
        if i in used_d or j in used_r:
            continue
        used_d.add(i)
        used_r.add(j)
        pairs.append({"det": i, "ref": j, "dt_s": dt, "dist_km": dist})
    return pd.DataFrame(pairs, columns=["det", "ref", "dt_s", "dist_km"])


def validation_report(detected, reference, pairs, min_ref_magnitude=None):
    report = {"num_detected": int(len(detected)), "num_reference": int(len(reference)), "num_matched": int(len(pairs))}
    if len(reference):
        report["recall"] = len(pairs) / len(reference)
    if len(detected):
        report["fraction_detected_matched"] = len(pairs) / len(detected)
    if min_ref_magnitude is not None and len(reference):
        big = reference.index[reference["magnitude"] >= min_ref_magnitude]
        if len(big):
            report[f"recall_M{min_ref_magnitude:g}+"] = float(pairs["ref"].isin(big).sum() / len(big))
    if len(pairs):
        d = detected.iloc[pairs["det"].values].reset_index(drop=True)
        r = reference.iloc[pairs["ref"].values].reset_index(drop=True)
        report["median_abs_origin_time_s"] = float(np.median(np.abs(pairs["dt_s"])))
        report["median_epicentral_km"] = float(np.median(pairs["dist_km"]))
        report["fraction_epicentral_lt_5km"] = float(np.mean(pairs["dist_km"] < 5))
        report["median_abs_depth_km"] = float(np.median(np.abs(d["depth_km"] - r["depth_km"])))
        dm = (d["magnitude"] - r["magnitude"]).dropna()
        if len(dm):
            report["median_magnitude_residual"] = float(np.median(dm))
            report["median_abs_magnitude_residual"] = float(np.median(np.abs(dm)))
    return report


# ---------------------------------------------------------------- export and plots


def to_quakeml(events, fname):
    from obspy.core.event import Catalog, Event, Magnitude, Origin, ResourceIdentifier

    cat = Catalog()
    for _, e in events.iterrows():
        origin = Origin(
            time=UTCDateTime(pd.Timestamp(e["time"]).to_pydatetime()),
            latitude=e["latitude"],
            longitude=e["longitude"],
            depth=e["depth_km"] * 1e3,
        )
        ev = Event(resource_id=ResourceIdentifier(f"smi:eqsys/event/{int(e['event_index'])}"), origins=[origin])
        if np.isfinite(e.get("magnitude", np.nan)):
            ev.magnitudes = [Magnitude(mag=e["magnitude"], magnitude_type=e.get("magnitude_type", "ML"),
                                       origin_id=origin.resource_id)]
        cat.append(ev)
    cat.write(fname, format="QUAKEML")


def qc_plots(events, reference, mc, b, config):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    qc = path(config, "qc")
    dm = config["catalog"]["delta_m"]
    m = events["magnitude"].dropna().values

    fig, ax = plt.subplots(figsize=(5, 4))
    if len(m):
        edges = np.arange(np.floor(m.min() / dm) * dm, m.max() + 2 * dm, dm) - dm / 2
        counts, _ = np.histogram(m, edges)
        centers = edges[:-1] + dm / 2
        ax.semilogy(centers, counts, "o", ms=3, label="non-cumulative")
        ax.semilogy(centers, counts[::-1].cumsum()[::-1], "s", ms=3, label="cumulative")
        if np.isfinite(mc) and np.isfinite(b):
            n_mc = np.sum(m >= mc - dm / 2)
            xx = np.linspace(mc, m.max(), 20)
            ax.semilogy(xx, n_mc * 10 ** (-b * (xx - mc)), "k--", label=f"b={b:.2f}, Mc={mc:.1f}")
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Number of events")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{qc}/frequency_magnitude.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    if len(reference):
        ax.scatter(reference["longitude"], reference["latitude"], s=6, c="0.6", label=f"reference ({len(reference)})")
    ax.scatter(events["longitude"], events["latitude"], s=3, c="C3", label=f"this pipeline ({len(events)})")
    r = config["region"]
    ax.set_xlim(r["minlongitude"], r["maxlongitude"])
    ax.set_ylim(r["minlatitude"], r["maxlatitude"])
    ax.set_aspect(1 / np.cos(np.deg2rad((r["minlatitude"] + r["maxlatitude"]) / 2)))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{qc}/map.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for df, label, c in [(reference, "reference", "0.5"), (events, "this pipeline", "C3")]:
        if len(df):
            t = pd.to_datetime(df["time"]).sort_values()
            ax.step(t, np.arange(1, len(t) + 1), where="post", c=c, label=label)
    ax.set_ylabel("Cumulative number")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(f"{qc}/cumulative.png", dpi=150)
    plt.close(fig)


def run(config):
    cc = config["catalog"]
    events = pd.read_csv(path(config, "events_mag.csv"))
    events = events.sort_values("time").reset_index(drop=True)
    cols = ["event_index", "time", "latitude", "longitude", "depth_km", "magnitude", "magnitude_type",
            "magnitude_uncertainty", "magnitude_num_stations", "magnitude_gamma", "num_picks",
            "adloc_score", "adloc_residual_time"]
    events = events[[c for c in cols if c in events.columns]]
    events.to_csv(path(config, "catalog.csv"), index=False)
    to_quakeml(events, path(config, "catalog.xml"))

    mc = mc_maxc(events["magnitude"], cc["delta_m"], cc["mc_correction"])
    b, b_sigma, n_b = b_value(events["magnitude"], mc, cc["delta_m"]) if np.isfinite(mc) else (np.nan, np.nan, 0)

    reference = fetch_reference(config)
    reference.to_csv(path(config, "reference_catalog.csv"), index=False)
    pairs = match_catalogs(events, reference, cc["match_time_s"], cc["match_distance_km"])
    pairs.to_csv(path(config, "qc", "matches.csv"), index=False)
    report = validation_report(events, reference, pairs, min_ref_magnitude=1.5)
    report.update({"mc": mc, "b_value": b, "b_value_sigma": b_sigma, "num_events_above_mc": n_b})
    if len(reference):
        mc_ref = mc_maxc(reference["magnitude"], cc["delta_m"], cc["mc_correction"])
        report["mc_reference"] = mc_ref
    with open(path(config, "qc", "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    qc_plots(events, reference, mc, b, config)
    logger.info("QC report:\n" + json.dumps(report, indent=2, default=float))
    return report
