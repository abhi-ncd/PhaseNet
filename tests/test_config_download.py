import os

import numpy as np
import pytest
from obspy.core.inventory import Channel, Inventory, Network, Station

from eqsys.config import build_config, projection
from eqsys.download import build_station_table, select_instruments, time_chunks


def test_defaults_and_derived_values(base_config):
    vm = base_config["locate"]["velocity_model"]
    assert np.allclose(vm["S"], np.array(vm["P"]) / 1.73)
    assert os.path.isabs(base_config["pick"]["model_dir"])
    assert base_config["download"]["chunk_seconds"] == 3600


@pytest.mark.parametrize(
    "patch",
    [
        {"region": {"minlatitude": 36, "maxlatitude": 35, "minlongitude": -118, "maxlongitude": -117}},
        {"starttime": "2019-07-11", "endtime": "2019-07-10"},
    ],
)
def test_invalid_config_rejected(patch):
    cfg = {
        "name": "x",
        "region": {"minlatitude": 35, "maxlatitude": 36, "minlongitude": -118, "maxlongitude": -117},
        "starttime": "2019-07-10",
        "endtime": "2019-07-11",
    }
    cfg.update(patch)
    with pytest.raises(ValueError):
        build_config(cfg)


def make_station(code, lat, lon, channels):
    chans = [Channel(code=c, location_code=loc, latitude=lat, longitude=lon, elevation=0.0, depth=0.0)
             for loc, c in channels]
    return Station(code, lat, lon, 0.0, channels=chans)


def test_select_instruments_prefers_complete_high_rate_channels(base_config):
    stations = [
        make_station("A", 35.5, -117.5, [("", "HHZ"), ("", "HHN"), ("", "HHE"), ("", "EHZ")]),
        make_station("B", 35.9, -117.9, [("", "HHZ"), ("", "EHZ"), ("", "EHN"), ("", "EHE")]),  # HH incomplete
        make_station("C", 35.6, -117.6, [("10", "HHZ"), ("10", "HH1"), ("10", "HH2")]),
        make_station("D", 35.5, -117.5, [("", "LHZ")]),  # nothing usable
    ]
    inv = Inventory(networks=[Network(code="CI", stations=stations)])
    sel = select_instruments({"SCEDC": inv}, base_config)
    got = {s["station"]: (s["location"], s["instrument"]) for s in sel}
    assert got == {"A": ("", "HH"), "B": ("", "EH"), "C": ("10", "HH")}
    assert [s["station"] for s in sel] == ["A", "C", "B"]  # sorted by distance from the centre

    base_config["download"]["max_stations"] = 2
    assert len(select_instruments({"SCEDC": inv}, base_config)) == 2


def test_time_chunks_cover_window(base_config):
    base_config["endtime"] = "2019-07-10T02:30:00"
    chunks = list(time_chunks(base_config))
    assert len(chunks) == 3 and chunks[-1][1] - chunks[-1][0] == 1800


def test_station_table_matches_phasenet_ids(base_config):
    chans = [Channel(code=c, location_code="", latitude=35.5, longitude=-117.5, elevation=1000.0, depth=0.0)
             for c in ["HHE", "HHN", "HHZ"]]
    chans.append(Channel(code="HNZ", location_code="", latitude=35.5, longitude=-117.5, elevation=1000.0, depth=0.0))
    inv = Inventory(networks=[Network(code="CI", stations=[Station("CCC", 35.5, -117.5, 1000.0, channels=chans)])])
    df = build_station_table(inv, projection(base_config))
    assert sorted(df["station_id"]) == ["CI.CCC..HH", "CI.CCC..HN"]
    row = df.set_index("station_id").loc["CI.CCC..HH"]
    assert row["component"] == "ENZ"
    assert row["z_km"] == pytest.approx(-1.0)  # 1 km above sea level -> negative depth
    assert abs(row["x_km"]) < 1e-6 and abs(row["y_km"]) < 1e-6  # region centre
