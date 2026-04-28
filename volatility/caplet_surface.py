"""
Caplet/floorlet vol surfaces for the closed-form (Black/Bachelier) pricers.

Distinct from `volatility.iv_pca` and `volatility.pca_result`, which model the
*forward-rate process* vol used by the HJM Monte Carlo simulator. This module is
about *option-implied* vols quoted per (expiry, strike) — the inputs Black/Bachelier
formulas consume. Wire to market quotes, or to MC-implied vols later.
"""
from typing import Optional

import numpy as np


class FlatCapletVolSurface:
    """
    Single constant vol across all expiries and strikes — useful for testing.

    `sigma` units depend on `model`:
      - 'lognormal' (Black): RELATIVE vol, e.g. 0.30 = 30%.
      - 'normal'   (Bachelier): ABSOLUTE rate vol, e.g. 0.01 = 100 bp / √yr.

    Conversion at strike F: σ_normal ≈ σ_lognormal · F.
    """

    def __init__(self, sigma: float, model: str = "lognormal"):
        if model not in ("lognormal", "normal"):
            raise ValueError(f"model must be 'lognormal' or 'normal', got {model!r}.")
        self.sigma = float(sigma)
        self.model = model

    def capfloor_vol(self, T: float, accrual: float, strike: float, model: str) -> float:
        if model != self.model:
            raise ValueError(f"surface stores {self.model!r} vol; pricer asked for {model!r}.")
        return self.sigma


class TermStructureCapletVolSurface:
    """
    ATM term-structure: one vol per expiry, no skew. Linearly interpolated in expiry,
    flat-extrapolated outside [T_min, T_max]. Strike is ignored.
    """

    def __init__(self, expiries_yr: np.ndarray, sigmas: np.ndarray, model: str = "lognormal"):
        if model not in ("lognormal", "normal"):
            raise ValueError(f"model must be 'lognormal' or 'normal', got {model!r}.")
        expiries_yr = np.asarray(expiries_yr, dtype=float)
        sigmas = np.asarray(sigmas, dtype=float)
        if expiries_yr.shape != sigmas.shape or expiries_yr.ndim != 1:
            raise ValueError("expiries_yr and sigmas must be 1-D arrays of the same length.")
        order = np.argsort(expiries_yr)
        self._T = expiries_yr[order]
        self._sigma = sigmas[order]
        self.model = model

    def capfloor_vol(self, T: float, accrual: float, strike: float, model: str) -> float:
        if model != self.model:
            raise ValueError(f"surface stores {self.model!r} vol; pricer asked for {model!r}.")
        # np.interp does flat extrapolation by default — appropriate for ATM term structure.
        return float(np.interp(T, self._T, self._sigma))


class GridCapletVolSurface:
    """
    2-D grid: vol = f(expiry, strike). Bilinear interp inside the grid, flat extrapolation
    outside. Strike axis lets the pricer query a smile/skew at each expiry.
    """

    def __init__(
            self,
            expiries_yr: np.ndarray,
            strikes: np.ndarray,
            sigmas: np.ndarray,
            model: str = "lognormal",
    ):
        if model not in ("lognormal", "normal"):
            raise ValueError(f"model must be 'lognormal' or 'normal', got {model!r}.")
        expiries_yr = np.asarray(expiries_yr, dtype=float)
        strikes = np.asarray(strikes, dtype=float)
        sigmas = np.asarray(sigmas, dtype=float)
        if sigmas.shape != (len(expiries_yr), len(strikes)):
            raise ValueError(
                f"sigmas shape {sigmas.shape} must equal (len(expiries), len(strikes))="
                f"({len(expiries_yr)}, {len(strikes)})."
            )
        i = np.argsort(expiries_yr); j = np.argsort(strikes)
        self._T = expiries_yr[i]
        self._K = strikes[j]
        self._sigma = sigmas[np.ix_(i, j)]
        self.model = model

    def capfloor_vol(self, T: float, accrual: float, strike: float, model: str) -> float:
        if model != self.model:
            raise ValueError(f"surface stores {self.model!r} vol; pricer asked for {model!r}.")
        # 1-D linear interp along strike at every expiry knot, then along expiry.
        per_expiry = np.array([np.interp(strike, self._K, row) for row in self._sigma])
        return float(np.interp(T, self._T, per_expiry))
