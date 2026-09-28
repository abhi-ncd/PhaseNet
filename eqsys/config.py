"""Load and validate the pipeline YAML config, filling in defaults."""

import copy
import os
from pathlib import Path

import yaml
from obspy import UTCDateTime

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULTS = {
    "download": {
        "providers": ["SCEDC", "IRIS"],
        "network": "*",
        "station": "*",
        "location": "*",
        "channel": "HH?,BH?,EH?,HN?",
        "channel_priorities": ["HH[ZNE12]", "BH[ZNE12]", "EH[ZNE12]", "HN[ZNE12]"],
        "location_priorities": ["", "00", "10", "01", "02"],
        "chunk_seconds": 3600,  # PhaseNet runs a whole file in one pass; keep files short to bound memory
        "minimum_interstation_distance_km": 1.0,
        "max_stations": None,
    },
    "pick": {
        "model_dir": "model/190703-214543",
        "python": None,  # interpreter for phasenet/predict.py; defaults to the current one
        "batch_size": 1,
        "min_p_prob": 0.3,
        "min_s_prob": 0.3,
        "highpass_filter": 0.0,
    },
    "associate": {
        "vel": {"p": 6.0, "s": 3.47},
        "use_amplitude": True,
        "method": "BGMM",
        "oversample_factor": 5,
        "use_dbscan": True,
        "dbscan_eps": None,  # None -> estimated from station spacing
        "dbscan_min_samples": 3,
        "min_picks_per_eq": 8,
        "min_p_picks_per_eq": 4,
        "min_s_picks_per_eq": 2,
        "max_sigma11": 2.0,  # s
        "max_sigma22": 1.0,  # log10 amplitude
        "max_sigma12": 1.0,
        "z_range_km": [0, 30],
        "ncpu": 2,
    },
    "locate": {
        # layered 1D model: depths of layer tops (km) with P and S velocities (km/s)
        "velocity_model": {"Z": [0.0, 5.5, 16.0, 32.0], "P": [5.5, 6.3, 6.7, 7.8], "S": None},
        "vp_vs_ratio": 1.73,
        "h": 1.0,  # eikonal grid spacing (km)
        "min_picks": 6,
        "min_picks_ratio": 0.5,
        "min_p_picks": 3,
        "min_s_picks": 2,
        "max_residual_time": 1.0,
        "min_score": 0.5,
        "z_range_km": [0, 30],
        "ncpu": 2,
    },
    "magnitude": {
        "pre_p_seconds": 1.0,
        "post_s_seconds": 10.0,
        "min_stations": 2,
        "max_distance_km": 300.0,
        "wood_anderson_gain": 2080.0,
    },
    "catalog": {
        "reference_client": "USGS",
        "reference_min_magnitude": None,
        "match_time_s": 3.0,
        "match_distance_km": 15.0,
        "mc_correction": 0.2,
        "delta_m": 0.1,
    },
    "forecast": {
        "source": "reference",  # "reference" (long FDSN catalog) or "pipeline" (catalog built by stages 1-6)
        "client": "USGS",
        "starttime": "2000-01-01",
        "train_end": "2016-01-01",
        "valid_end": "2019-01-01",
        "endtime": "2021-01-01",
        "min_magnitude": 2.5,
        "mc": None,  # None -> estimated from the training window
        "delta_m": 0.1,
        "horizon_days": 1,
        "n_simulations": 200,
        "ntpp": {"hidden": 32, "n_mix": 8, "epochs": 200, "lr": 1e-3, "seq_len": 64, "seed": 0},
    },
}


def _merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path):
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    return build_config(user)


def build_config(user):
    for key in ["name", "region", "starttime", "endtime"]:
        if key not in user:
            raise ValueError(f"config is missing required key '{key}'")
    region = user["region"]
    for key in ["minlatitude", "maxlatitude", "minlongitude", "maxlongitude"]:
        if key not in region:
            raise ValueError(f"config.region is missing '{key}'")
    if region["minlatitude"] >= region["maxlatitude"] or region["minlongitude"] >= region["maxlongitude"]:
        raise ValueError("config.region min values must be smaller than max values")

    config = _merge(DEFAULTS, user)
    config["starttime"] = str(UTCDateTime(config["starttime"]))
    config["endtime"] = str(UTCDateTime(config["endtime"]))
    if UTCDateTime(config["starttime"]) >= UTCDateTime(config["endtime"]):
        raise ValueError("config.starttime must be before config.endtime")

    run_dir = Path(config.get("run_dir") or f"runs/{config['name']}")
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    config["run_dir"] = str(run_dir)

    model_dir = Path(config["pick"]["model_dir"])
    if not model_dir.is_absolute():
        config["pick"]["model_dir"] = str(REPO_ROOT / model_dir)

    vm = config["locate"]["velocity_model"]
    if vm.get("S") is None:
        vm["S"] = [v / config["locate"]["vp_vs_ratio"] for v in vm["P"]]
    if not (len(vm["Z"]) == len(vm["P"]) == len(vm["S"])):
        raise ValueError("locate.velocity_model Z, P and S must have the same length")
    return config


def center(config):
    r = config["region"]
    return (r["minlatitude"] + r["maxlatitude"]) / 2, (r["minlongitude"] + r["maxlongitude"]) / 2


def projection(config):
    """Local azimuthal-equidistant projection (km) centred on the region."""
    from pyproj import Proj

    lat0, lon0 = center(config)
    return Proj(f"+proj=aeqd +lon_0={lon0} +lat_0={lat0} +units=km")


def path(config, *parts):
    p = os.path.join(config["run_dir"], *parts)
    os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
    return p
