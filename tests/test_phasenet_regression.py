"""PhaseNet still produces sensible picks on the bundled Ridgecrest demo waveforms.

Guards against the model silently breaking (e.g. a layer shim that zeroes activations), which
would otherwise only show up as an empty catalog several stages later.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_demo_picks(tmp_path):
    data_list = tmp_path / "list.csv"
    data_list.write_text("fname\nCCC.mseed\n")
    subprocess.run(
        [sys.executable, str(REPO / "phasenet" / "predict.py"), "--model_dir", str(REPO / "model" / "190703-214543"),
         "--data_dir", str(REPO / "demo" / "mseed"), "--data_list", str(data_list), "--format", "mseed",
         "--amplitude", "--batch_size", "1", "--result_dir", str(tmp_path)],
        check=True, cwd=REPO, capture_output=True,
    )
    picks = pd.read_csv(tmp_path / "picks.csv")
    # 7 hours of CI.CCC during the 2019-07-04 Ridgecrest foreshock sequence: >1000 confident picks
    assert (picks["phase_type"] == "P").sum() > 500
    assert (picks["phase_type"] == "S").sum() > 500
    assert picks["phase_score"].median() > 0.5
    assert picks["phase_amplitude"].gt(0).all()
