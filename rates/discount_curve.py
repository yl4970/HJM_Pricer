"""
Pricing-time view of a single-date discount curve.

Wraps the bootstrapped DFs from `rates.forward_curve.bootstrap_discount_factors` and
exposes the two methods the closed-form pricers consume:

    df(t_yr)                     → P(0, t)
    forward(t_start, t_end)      → simply-compounded F over [t_start, t_end]

Interpolation: cubic spline on log(DF), same convention as the instantaneous-forward
curve construction, so DF and forward are mutually consistent.
"""
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from data.term_data import TermStructureData


class DiscountCurve:
    def __init__(self, knot_tenors_yr: np.ndarray, knot_dfs: np.ndarray):
        knot_tenors_yr = np.asarray(knot_tenors_yr, dtype=float)
        knot_dfs = np.asarray(knot_dfs, dtype=float)
        if not np.all(np.diff(knot_tenors_yr) > 0):
            raise ValueError("knot_tenors_yr must be strictly increasing.")
        if np.any(knot_dfs <= 0):
            raise ValueError("DFs must be strictly positive.")
        # Anchor df(0) ≡ 1 — without this, natural-BC extrapolation lets df(0)
        # drift away from 1 and biases every short-end forward.
        if knot_tenors_yr[0] > 0:
            knot_tenors_yr = np.concatenate([[0.0], knot_tenors_yr])
            knot_dfs = np.concatenate([[1.0], knot_dfs])
        self._knot_yr = knot_tenors_yr
        self._spline = CubicSpline(knot_tenors_yr, np.log(knot_dfs), bc_type="natural")

    @classmethod
    def from_tsd(cls, df_tsd: TermStructureData, date) -> "DiscountCurve":
        """Build from a row of `bootstrap_discount_factors` output."""
        tenors_yr = np.asarray(df_tsd.tenors, dtype=float) / 12.0
        row = df_tsd.to_dataframe().loc[pd.Timestamp(date)].values
        return cls(tenors_yr, row)

    def df(self, t_yr: float) -> float:
        return float(np.exp(self._spline(float(t_yr))))

    def forward(self, t_start_yr: float, t_end_yr: float) -> float:
        """Simply-compounded forward rate F(0; s, e) = (P(s)/P(e) - 1) / (e - s)."""
        if t_end_yr <= t_start_yr:
            raise ValueError(f"t_end ({t_end_yr}) must be > t_start ({t_start_yr}).")
        return (self.df(t_start_yr) / self.df(t_end_yr) - 1.0) / (t_end_yr - t_start_yr)
