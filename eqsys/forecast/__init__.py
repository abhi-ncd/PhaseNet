"""Probabilistic earthquake forecasting on a catalog.

Temporal models (Poisson, ETAS, neural point process) are compared by exact log-likelihood on a
held-out test window and by daily number tests, following Stockman et al. (2024, earthquakeNPP).
A spatio-temporal ETAS model (Mizrahi et al., 2021, `etas` package) gives map forecasts.
"""
