"""GaMMA association and ADLoc location recover known synthetic earthquakes."""

import numpy as np
import pandas as pd
import pytest

from eqsys.associate import associate
from eqsys.config import projection
from eqsys.locate import locate

VP, VS = 6.0, 6.0 / 1.73


@pytest.fixture
def synthetic(base_config):
    rng = np.random.default_rng(0)
    proj = projection(base_config)
    xs, ys = np.meshgrid(np.linspace(-35, 35, 5), np.linspace(-35, 35, 5))
    stations = pd.DataFrame({"station_id": [f"XX.S{i:02d}..HH" for i in range(xs.size)],
                             "x_km": xs.ravel(), "y_km": ys.ravel(), "z_km": 0.0})
    lon, lat = proj(stations["x_km"].values, stations["y_km"].values, inverse=True)
    stations["longitude"], stations["latitude"] = lon, lat

    t0 = pd.Timestamp("2019-07-10T00:00:00")
    sources = pd.DataFrame({"x_km": [-10.0, 12.0, 0.0, 20.0], "y_km": [5.0, -15.0, 20.0, 18.0],
                            "z_km": [8.0, 5.0, 12.0, 6.0], "t_s": [10.0, 50.0, 95.0, 140.0]})
    picks = []
    for true_event, src in sources.iterrows():
        d = np.sqrt((stations["x_km"] - src.x_km) ** 2 + (stations["y_km"] - src.y_km) ** 2 + src.z_km**2)
        for phase, v in [("P", VP), ("S", VS)]:
            t = src.t_s + d / v + rng.normal(0, 0.05, len(d))
            for sid, ti in zip(stations["station_id"], t):
                picks.append({"station_id": sid, "phase_time": (t0 + pd.Timedelta(seconds=float(ti))).isoformat(),
                              "phase_score": 0.9, "phase_type": phase, "phase_amplitude": 1e-5,
                              "true_event": true_event})
    picks = pd.DataFrame(picks).sort_values("phase_time").reset_index(drop=True)

    cfg = base_config
    cfg["associate"].update({"use_amplitude": False, "vel": {"p": VP, "s": VS}, "ncpu": 1, "z_range_km": [0, 20]})
    cfg["locate"].update({"velocity_model": {"Z": [0.0, 40.0], "P": [VP, VP], "S": [VS, VS]},
                          "ncpu": 1, "z_range_km": [0, 20]})
    return cfg, stations, picks, sources


def nearest_error(found, sources):
    errs = []
    for _, s in sources.iterrows():
        d = np.sqrt((found["x_km"] - s.x_km) ** 2 + (found["y_km"] - s.y_km) ** 2)
        errs.append(d.min())
    return np.array(errs)


def test_gamma_then_adloc_recover_sources(synthetic):
    config, stations, picks, sources = synthetic
    events, picks_out = associate(picks, stations, config)
    assert len(events) == len(sources)
    assoc = picks_out[picks_out["event_index"] >= 0]
    assert (assoc["phase_type"] == "P").sum() == (picks["phase_type"] == "P").sum()
    assert len(assoc) > 0.8 * len(picks)
    # every associated pick belongs to the right earthquake (one GaMMA event per true source)
    assert assoc.groupby("event_index")["true_event"].nunique().max() == 1
    assert assoc.groupby("true_event")["event_index"].nunique().max() == 1
    assert nearest_error(events, sources).max() < 3.0

    located, _, _ = locate(events, picks_out, stations, config, iterations=1)
    assert len(located) == len(sources)
    assert nearest_error(located, sources).max() < 1.5
    for _, s in sources.iterrows():
        nearest = located.iloc[int(np.argmin(np.hypot(located["x_km"] - s.x_km, located["y_km"] - s.y_km)))]
        assert abs(nearest["z_km"] - s.z_km) < 2.5
