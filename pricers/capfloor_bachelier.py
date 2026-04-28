from math import sqrt

from instruments.capsfloors import CapFloor
from pricers._helpers import norm_cdf, norm_pdf


def bachelier_call(F: float, K: float, sigma: float, T: float) -> float:
    """
    Normal-model (Bachelier) call on a forward rate.
        C = (F - K) Φ(d) + σ√T φ(d),   d = (F - K) / (σ√T)
    Sigma is in absolute-rate units (e.g. 0.01 = 100 bp/yr), unlike Black's relative vol.
    Bachelier handles negative or zero forwards/strikes natively, which is why it's the
    standard for low-/negative-rate environments.
    """
    if sigma <= 0 or T <= 0:
        return max(F - K, 0.0)
    s = sigma * sqrt(T)
    d = (F - K) / s
    return (F - K) * norm_cdf(d) + s * norm_pdf(d)


def bachelier_put(F: float, K: float, sigma: float, T: float) -> float:
    # Put-call parity on the forward (undiscounted): C - P = F - K, so P = C - F + K.
    return bachelier_call(F, K, sigma, T) - F + K


class CapFloorBachelierEngine:
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
            sigma = self.vols.capfloor_vol(T, cf.accrual, strike=K, model="normal")

            opt = bachelier_call(F, K, sigma, T) if inst.sign == 1 else bachelier_put(F, K, sigma, T)
            pv += inst.notional * P * cf.accrual * opt
        return pv
