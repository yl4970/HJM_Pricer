from instruments.capsfloors import CapFloor, CashFlow
from math import log, sqrt, erf

def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))

def black_call(F, K, sigma, T):
    if sigma <= 0 or T <= 0:
        return max(F - K, 0.0)
    s = sigma * sqrt(T)
    d1 = (log(F/K) + 0.5*s*s) / s
    d2 = d1 - s
    return F * norm_cdf(d1) - K * norm_cdf(d2)

class CapFloorBlackEngine:
    def __init__(self, discount_curve, forward_curve, vol_surface):
        self.discount = discount_curve    # must have .df(t)
        self.forward = forward_curve      # must have .forward(t0, t1)
        self.vols = vol_surface           # must have .capfloor_vol(t_expiry, accrual, strike, model)

    def price(self, inst: CapFloor) -> float:
        pv = 0.0
        for cf in inst.schedule:
            F = self.forward.forward(cf.start, cf.end)
            P = self.discount.df(cf.pay_date)
            T = cf.fixing_time
            K = inst.strike
            a = cf.accrual
            sigma = self.vols.capfloor_vol(T, cf.accrual, strike=K, model="lognormal")

            call = black_call(F, K, sigma, T)
            # Cap = call; Floor = call - (K - F)
            opt = call if inst.sign == 1 else call - (K - F)

            pv += inst.notional * P * a * opt
        return pv