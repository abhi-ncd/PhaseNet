"""eqsys: earthquake detection, cataloging and forecasting pipeline built around PhaseNet.

Stages (each reads the previous stage's files from ``<run_dir>``):
    download -> pick -> associate -> locate -> magnitude -> catalog -> forecast
"""

__version__ = "0.1.0"
