import numpy as np
import pandas as pd
import pytest
from scipy import signal

from eqsys.catalog import b_value, match_catalogs, mc_maxc
from eqsys.magnitude import local_magnitude, robust_event_magnitude, wood_anderson_paz


def test_local_magnitude_reference_point():
    # By definition ML 3 gives 1 mm on a Wood-Anderson instrument at 100 km.
    assert local_magnitude(1.0, 100.0) == pytest.approx(3.0)
    # Hutton & Boore 1987 at 17 km: -log A0 = 1.110 log10(0.17) + 0.00189 (17 - 100) + 3
    expected = np.log10(0.5) + 1.110 * np.log10(0.17) + 0.00189 * (17 - 100) + 3.0
    assert local_magnitude(0.5, 17.0) == pytest.approx(expected)


def test_wood_anderson_response():
    paz = wood_anderson_paz()
    b, a = signal.zpk2tf(paz["zeros"], paz["poles"], paz["gain"] * paz["sensitivity"])
    w, h = signal.freqs(b, a, worN=2 * np.pi * np.array([0.1, 1.25, 20.0]))
    # displacement response: ~2080 well above the 1.25 Hz corner, strongly reduced below it
    assert abs(h[2]) == pytest.approx(2080, rel=0.01)
    assert abs(h[0]) < 0.01 * 2080
    assert 0.5 * 2080 < abs(h[1]) < 2080


def test_robust_event_magnitude():
    ml, mad = robust_event_magnitude([2.0, 2.1, 2.2, 5.0, np.nan])
    assert ml == pytest.approx(2.15)
    assert mad < 0.3


def test_mc_and_b_value_on_synthetic_gr():
    rng = np.random.default_rng(1)
    b_true, mc_true = 1.0, 2.0
    complete = mc_true - 0.05 + rng.exponential(1 / (b_true * np.log(10)), 20000)
    incomplete = rng.uniform(0.5, 2.0, 3000)
    m = np.round(np.r_[complete, incomplete], 1)
    mc = mc_maxc(m, 0.1, correction=0.0)
    assert mc == pytest.approx(2.0, abs=0.11)
    b, sigma, n = b_value(m, 2.0, 0.1)
    assert b == pytest.approx(b_true, abs=3 * sigma + 0.02)


def test_match_catalogs_one_to_one():
    t0 = pd.Timestamp("2019-07-10")
    ref = pd.DataFrame({"time": [t0, t0 + pd.Timedelta(seconds=60)], "latitude": [35.5, 35.6],
                        "longitude": [-117.5, -117.6], "depth_km": [5, 5], "magnitude": [2.0, 3.0]})
    det = pd.DataFrame({"time": [t0 + pd.Timedelta(seconds=1), t0 + pd.Timedelta(seconds=1.5),
                                 t0 + pd.Timedelta(seconds=200)],
                        "latitude": [35.51, 35.9, 35.6], "longitude": [-117.5, -117.5, -117.6]})
    pairs = match_catalogs(det, ref, max_dt=3, max_dist_km=15)
    assert list(zip(pairs["det"], pairs["ref"])) == [(0, 0)]  # det 1 is 44 km away, det 2 is 140 s late
