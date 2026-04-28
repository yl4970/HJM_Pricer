from math import log, sqrt

from instruments.capsfloors import CapFloor
from pricers._helpers import norm_cdf


def black_call(F: float, K: float, sigma: float, T: float) -> float:
    if sigma <= 0 or T <= 0:
        return max(F - K, 0.0)
    if F <= 0 or K <= 0:
        # Black is undefined for non-positive forwards/strikes; caller should pick Bachelier.
        raise ValueError(f"Black model requires positive F and K (got F={F}, K={K}).")
    s = sigma * sqrt(T)
    d1 = (log(F / K) + 0.5 * s * s) / s
    d2 = d1 - s
    return F * norm_cdf(d1) - K * norm_cdf(d2)


def black_put(F: float, K: float, sigma: float, T: float) -> float:
    # Put-call parity on the forward (undiscounted): C - P = F - K, so P = C - F + K.
    return black_call(F, K, sigma, T) - F + K


class CapFloorBlackEngine:
    def __init__(self, discount_curve, forward_curve, vol_surface):
        self.discount = discount_curve    # must have .df(t_yr)
        self.forward = forward_curve      # must have .forward(t_start_yr, t_end_yr)
        self.vols = vol_surface           # must have .capfloor_vol(T, accrual, strike, model)

    def price(self, inst: CapFloor) -> float:
        pv = 0.0
        for cf in inst.schedule:
            F = self.forward.forward(cf.start, cf.end)
            P = self.discount.df(cf.pay_date)
            T = cf.fixing_time
            K = inst.strike
            sigma = self.vols.capfloor_vol(T, cf.accrual, strike=K, model="lognormal")

            opt = black_call(F, K, sigma, T) if inst.sign == 1 else black_put(F, K, sigma, T)
            pv += inst.notional * P * cf.accrual * opt
        return pv
