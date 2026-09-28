"""Stage 2: P/S picking with PhaseNet, by running the existing phasenet/predict.py CLI.

Reads  <run_dir>/mseed_list.csv, waveforms/, stations.xml
Writes <run_dir>/picks.csv with columns
    station_id, phase_time, phase_score, phase_type, phase_amplitude, phase_index, begin_time, file_name
`phase_amplitude` is peak ground velocity in m/s, because the response sensitivity is removed.
"""

import logging
import os
import subprocess
import sys

import pandas as pd

from .config import REPO_ROOT, path

logger = logging.getLogger(__name__)


def predict_command(config):
    pk = config["pick"]
    return [
        pk["python"] or sys.executable,
        str(REPO_ROOT / "phasenet" / "predict.py"),
        f"--model_dir={pk['model_dir']}",
        f"--data_dir={path(config, 'waveforms')}",
        f"--data_list={path(config, 'mseed_list.csv')}",
        "--format=mseed",
        "--amplitude",
        f"--response_xml={path(config, 'stations.xml')}",
        f"--batch_size={pk['batch_size']}",
        f"--min_p_prob={pk['min_p_prob']}",
        f"--min_s_prob={pk['min_s_prob']}",
        f"--highpass_filter={pk['highpass_filter']}",
        f"--result_dir={path(config, 'phasenet')}",
        "--result_fname=picks",
    ]


def clean_picks(df):
    df = df.drop(columns=[c for c in ["phase_amp"] if c in df.columns])
    df["phase_type"] = df["phase_type"].str.upper()
    df = df.sort_values("phase_time").reset_index(drop=True)
    return df


def run(config):
    cmd = predict_command(config)
    logger.info("running " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    raw = os.path.join(path(config, "phasenet"), "picks.csv")
    df = clean_picks(pd.read_csv(raw))
    df.to_csv(path(config, "picks.csv"), index=False)
    logger.info(
        f"{len(df)} picks ({(df.phase_type == 'P').sum()} P, {(df.phase_type == 'S').sum()} S) "
        f"on {df.station_id.nunique()} stations"
    )
    return df
