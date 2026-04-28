"""
Forward (out-of-sample) Monte Carlo simulation of the forward-rate curve f(t, x)
under HJM with Musiela parametrization:

    df(t, x) = [∂f/∂x + α(t, x)] dt + Σ_j σ_j(x) dW_j(t)
    α(t, x) = Σ_j σ_j(x) · ∫_0^x σ_j(u) du

Vol loadings σ_j(x) are calibrated once (today) and held constant over the projection
horizon — the standard time-homogeneous HJM convention. The convexity drift α(x) is
precomputed; the Musiela aging term ∂f/∂x is path-dependent and refreshed each step.

Distinct from `simulation.MonteCarlo.MCSimulation`, which replays along the historical
timeline using each day's locally calibrated vol — useful for backtesting, not for
forward projection of an option book.
"""
from typing import Optional

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

from simulation.volSurface import VolatilitySurface


class HJMForwardSimulator:
    def __init__(
            self,
            f0: np.ndarray,
            tenors_m: np.ndarray,
            vol_loadings: np.ndarray,
            seed: Optional[int] = None,
    ):
        """
        f0:           (n_tenors,)            initial forward curve f(0, x)
        tenors_m:     (n_tenors,)            tenor grid in months
        vol_loadings: (n_tenors, n_factors)  σ_j(x) in annualized units (rate/sqrt(year))
        """
        f0 = np.asarray(f0, dtype=float)
        tenors_m = np.asarray(tenors_m, dtype=int)
        vol_loadings = np.asarray(vol_loadings, dtype=float)

        if f0.shape != (len(tenors_m),):
            raise ValueError(f"f0 shape {f0.shape} doesn't match tenors {tenors_m.shape}.")
        if vol_loadings.shape[0] != len(tenors_m):
            raise ValueError(
                f"vol_loadings rows ({vol_loadings.shape[0]}) must equal n_tenors ({len(tenors_m)})."
            )
        if not np.all(np.diff(tenors_m) > 0):
            raise ValueError("tenors_m must be strictly increasing.")

        self.f0 = f0
        self.tenors_m = tenors_m
        self.tenors_yr = tenors_m.astype(float) / 12.0
        self.vol_loadings = vol_loadings
        self.n_tenors, self.n_factors = vol_loadings.shape
        self.rng = np.random.default_rng(seed)

        # Time-homogeneous HJM convexity drift α(x). cumulative_trapezoid keeps it O(n).
        self._convex_drift = np.zeros(self.n_tenors)
        for j in range(self.n_factors):
            sigma_j = self.vol_loadings[:, j]
            integral = cumulative_trapezoid(sigma_j, self.tenors_yr, initial=0.0)
            self._convex_drift += sigma_j * integral

    @classmethod
    def from_volatility_surface(
            cls,
            vs: VolatilitySurface,
            degrees: list[int],
            date: Optional[pd.Timestamp] = None,
            seed: Optional[int] = None,
    ) -> "HJMForwardSimulator":
        """
        Calibrate from a `VolatilitySurface` snapshot. Uses the polyfit-smoothed
        loadings at `date` (defaults to the most recent timeline date) and the forward
        curve as of that date as the initial condition.
        """
        if date is None:
            date = vs.timeline[-1]
        fitted = vs.localVols[date].polyfit(degrees)['fittedVols']
        vol_loadings = np.asarray(fitted).T  # (n_tenors, n_factors)
        f0 = np.asarray(vs.windowed_fwds[date].iloc[-1], dtype=float)
        tenors_m = np.asarray(vs.tenors, dtype=int)
        return cls(f0=f0, tenors_m=tenors_m, vol_loadings=vol_loadings, seed=seed)

    def simulate(
            self,
            dt: float,
            n_steps: int,
            n_paths: int,
            Musiela: bool = True,
    ) -> np.ndarray:
        """
        Project f(t, x) forward over [0, n_steps · dt] years.

        Returns paths of shape (n_paths, n_steps + 1, n_tenors), with
        paths[:, 0, :] = f0.

        Memory: only the path tensor itself is allocated (no per-step Brownian
        cache), so peak ~ n_paths · (n_steps + 1) · n_tenors · 8 B.
        """
        if dt <= 0 or n_steps <= 0 or n_paths <= 0:
            raise ValueError("dt, n_steps, n_paths must all be positive.")

        sqrt_dt = float(np.sqrt(dt))
        paths = np.empty((n_paths, n_steps + 1, self.n_tenors), dtype=float)
        paths[:, 0, :] = self.f0

        for s in range(n_steps):
            f_curr = paths[:, s, :]

            # Brownian factor noise → tenor-space diffusion via vol_loadings.
            dW = self.rng.normal(size=(n_paths, self.n_factors)) * sqrt_dt
            diffusion = dW @ self.vol_loadings.T  # (n_paths, n_tenors)

            if Musiela:
                df_dx = np.gradient(f_curr, self.tenors_yr, axis=1)
                drift_step = (self._convex_drift + df_dx) * dt
            else:
                drift_step = self._convex_drift * dt

            paths[:, s + 1, :] = f_curr + drift_step + diffusion

        return paths
