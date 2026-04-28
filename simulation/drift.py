from typing import Optional

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

from simulation.volSurface import VolatilitySurface


def get_HJM_drifts(
        local_vs: VolatilitySurface,
        degrees: list[int],
        timeline: Optional[list[pd.Timestamp]] = None,
        tenors: Optional[list[int]] = None,
        Musiela: bool = True,
        vol_surface: Optional[dict] = None,
) -> np.ndarray:
    """
    Risk-neutral HJM drift on the Musiela grid:

        α(t, x) = Σ_j σ_j(t, x) * ∫_0^x σ_j(t, u) du           (+ ∂f/∂x if Musiela)

    σ_j is the polynomial fit of the j-th annualized vol loading (rate / sqrt(year)),
    so the integral is taken over tenor expressed in YEARS — matching the units in
    which dt is later supplied to the simulator.

    `vol_surface`, if supplied, must be the dict returned by
    `VolatilitySurface.polyfit(timeline, degrees=...)` — passing it lets callers
    that already have it (e.g. MCSimulation) avoid recomputing the polynomial fits.

    Returns: array of shape (n_dates, n_tenors).
    """
    if timeline is None:
        timeline = local_vs.timeline
    if tenors is None:
        tenors = local_vs.tenors
    if vol_surface is None:
        vol_surface = local_vs.polyfit(timeline, degrees=degrees)

    tenors_yr = np.asarray(tenors, dtype=float) / 12.0
    n_tenors = len(tenors_yr)
    n_factors = local_vs.n_factors

    drift_curve = np.empty((len(timeline), n_tenors), dtype=float)

    for d_idx, t in enumerate(timeline):
        drifts = np.zeros(n_tenors, dtype=float)

        for j in range(n_factors):
            sigma_j = np.asarray(vol_surface[t][j], dtype=float)
            # cumulative_trapezoid with initial=0 gives ∫_0^{x_k} σ du for each k,
            # collapsing the previous O(n²) per-tenor trapezoid loop to a single pass.
            integrals = cumulative_trapezoid(sigma_j, tenors_yr, initial=0.0)
            drifts += sigma_j * integrals

        if Musiela:
            f = np.asarray(local_vs.windowed_fwds[t].iloc[-1], dtype=float)
            drifts += np.gradient(f, tenors_yr)

        drift_curve[d_idx] = drifts

    return drift_curve
