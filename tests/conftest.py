import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def base_config(tmp_path):
    from eqsys.config import build_config

    return build_config(
        {
            "name": "test",
            "run_dir": str(tmp_path / "run"),
            "region": {"minlatitude": 35.0, "maxlatitude": 36.0, "minlongitude": -118.0, "maxlongitude": -117.0},
            "starttime": "2019-07-10T00:00:00",
            "endtime": "2019-07-10T01:00:00",
        }
    )
