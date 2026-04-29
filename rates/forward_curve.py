from datetime import date as dt
from typing import Optional
import warnings

import numpy as np
from scipy.interpolate import CubicSpline, RBFInterpolator

from data.loader import TermStructureLoader
from data.term_data import TermStructureData
from data.registry import DEFAULT_TERM_STRUCTURE_LOADER

warnings.filterwarnings("ignore")

SHORT_END_MONTHS = 12  # tenors <= 12M treated as continuously-compounded zero rates
COUPON_FREQ_MONTHS = 6  # UST semi-annual coupon convention


def _bootstrap_one_date(
        tenors_m: np.ndarray,
        par_yields: np.ndarray,
) -> np.ndarray:
    """
    Bootstrap discount factors at the quoted market tenors for a single date.

    Conventions:
      - tenors <= SHORT_END_MONTHS: treat the par yield as a continuously-compounded
        zero rate, so P(t) = exp(-y * t).
      - tenors > SHORT_END_MONTHS: par yield on a semi-annual coupon UST. Solve
        P(T) from
            1 = (Y/2) * sum_{c < T} P(c) + (1 + Y/2) * P(T)
        where coupon dates fall every 6M. DFs at intermediate coupon dates that
        are not market knots are interpolated linearly in log(DF) using the
        already-bootstrapped points.

    Returns DFs aligned with the input tenors_m.
    """
    tenors_m = np.asarray(tenors_m, dtype=int)
    par_yields = np.asarray(par_yields, dtype=float)

    if not np.all(np.diff(tenors_m) > 0):
        raise ValueError("Tenors must be strictly increasing for bootstrap.")

    tenors_yr = tenors_m.astype(float) / 12.0
    n = len(tenors_m)
    dfs = np.empty(n)

    # Working store of all known DFs, keyed by tenor in months (int).
    known: dict[int, float] = {}

    for i, (t_m, t_yr, y) in enumerate(zip(tenors_m, tenors_yr, par_yields)):
        if t_m <= SHORT_END_MONTHS:
            P = float(np.exp(-y * t_yr))
            dfs[i] = P
            known[int(t_m)] = P
            continue

        # Long end: bootstrap with semi-annual coupons.
        coupon_months = np.arange(COUPON_FREQ_MONTHS, int(t_m), COUPON_FREQ_MONTHS)

        sum_intermediate = 0.0
        for cm in coupon_months:
            cm = int(cm)
            if cm not in known:
                ks = np.array(sorted(known.keys()))
                lnPs = np.log(np.array([known[m] for m in ks]))
                # log-linear interp; flat extrapolation if cm sits outside known range
                known[cm] = float(np.exp(np.interp(cm, ks, lnPs)))
            sum_intermediate += known[cm]

        P_T = (1.0 - (y / 2.0) * sum_intermediate) / (1.0 + y / 2.0)
        dfs[i] = P_T
        known[int(t_m)] = P_T

    return dfs


def bootstrap_discount_factors(tsd: TermStructureData) -> TermStructureData:
    """
    Bootstrap discount factors at the quoted tenors for every date in the TSD.
    Output tenors are kept in months to stay consistent with the rest of the library.
    """
    tenors_m = np.asarray(tsd.tenors, dtype=int)
    out = np.empty_like(tsd.values, dtype=float)
    for i, row in enumerate(tsd.values):
        out[i] = _bootstrap_one_date(tenors_m, row)

    return TermStructureData(time=tsd.time, tenors=tenors_m, values=out)


def instantaneous_forwards_from_dfs(
        knot_tenors_m: np.ndarray,
        knot_dfs: np.ndarray,
        target_tenors_m: np.ndarray,
        interp_method: str = "cubic_spline",
) -> np.ndarray:
    """
    Interpolate log(DF) and differentiate to get instantaneous forwards.
 
    f(0, t) = -d/dt ln P(0, t)
 
    Anchors at (t=0, df=1) by definition.
 
    Parameters
    ----------
    interp_method : str
        'cubic_spline' — natural cubic spline on log(DF). Analytical derivative
                          via spline.derivative(). C² smooth forwards.
        'rbf'          — radial basis function (thin-plate spline) on log(DF).
    """
    knot_yr = np.asarray(knot_tenors_m, dtype=float) / 12.0
    knot_dfs = np.asarray(knot_dfs, dtype=float)
    if knot_yr[0] > 0:
        knot_yr = np.concatenate([[0.0], knot_yr])
        knot_dfs = np.concatenate([[1.0], knot_dfs])
 
    log_dfs = np.log(knot_dfs)
    target_yr = np.asarray(target_tenors_m, dtype=float) / 12.0
 
    if interp_method == "cubic_spline":
        spline = CubicSpline(knot_yr, log_dfs, bc_type="natural")
        return -spline.derivative()(target_yr)
 
    elif interp_method == "rbf":
        rbf = RBFInterpolator(
            knot_yr.reshape(-1, 1),
            log_dfs,
            kernel="thin_plate_spline",
        )
        target_log_dfs = rbf(target_yr.reshape(-1, 1)).ravel()
        return -np.gradient(target_log_dfs, target_yr)
 
    else:
        raise ValueError(
            f"Unknown interp_method '{interp_method}'. "
            f"Supported: 'cubic_spline', 'rbf'."
        )


def build_forward_curve(
        tsd: TermStructureData,
        target_tenors_m: Optional[np.ndarray] = None,
) -> TermStructureData:
    """
    Pipeline: par yields → bootstrap DFs at market knots → cubic-spline log(DF)
    → instantaneous forwards on the target monthly grid.
    """
    tenors_m = np.asarray(tsd.tenors, dtype=int)
    if target_tenors_m is None:
        target_tenors_m = np.arange(int(tenors_m[0]), int(tenors_m[-1]) + 1)
    target_tenors_m = np.asarray(target_tenors_m, dtype=int)

    n_dates = len(tsd.time)
    fwd = np.empty((n_dates, len(target_tenors_m)))

    for i, row in enumerate(tsd.values):
        knot_dfs = _bootstrap_one_date(tenors_m, row)
        fwd[i] = instantaneous_forwards_from_dfs(tenors_m, knot_dfs, target_tenors_m)

    return TermStructureData(
        time=tsd.time,
        tenors=target_tenors_m,
        values=fwd,
    )


class ForwardCurve:
    def __init__(
            self,
            loader: TermStructureLoader = DEFAULT_TERM_STRUCTURE_LOADER,
            sdate: dt = None,
            edate: dt = None,
    ):
        self.loader = loader
        self.sdate = sdate
        self.edate = edate
        self.curve: Optional[TermStructureData] = None

    def compute(self) -> TermStructureData:
        if self.curve is None:
            par_yields = self.loader.load(self.sdate, self.edate)
            self.curve = build_forward_curve(par_yields)
        return self.curve
