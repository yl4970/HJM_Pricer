"""
Monte Carlo cap/floor pricer driven by HJM-simulated forward-rate paths.

Pricing setup (risk-neutral, bank-account numeraire):

    PV = E_Q[ exp(-∫_0^{T_pay} r(s) ds) · max(sign · (L - K), 0) ]

where r(s) = f(s, 0) is the short rate (proxied by the shortest tenor on the
Musiela grid), and L = L(T_fix; T_start, T_end) is the simply-compounded forward
recovered from the simulated curve at the fixing time:

    P(T_fix; T_end) = exp(-∫_0^{T_end - T_start} f(T_fix, x) dx)
    L = (1 / P - 1) / (T_end - T_start)

Assumes cashflow `fixing_time` and `pay_date` align to the simulation step grid,
and that accrual length `T_end - T_start` is a whole multiple of the tenor spacing.
"""
from typing import Optional

import numpy as np

from instruments.capsfloors import CapFloor
from simulation.hjm_forward import HJMForwardSimulator


class CapFloorMCEngine:
    def __init__(self, simulator: HJMForwardSimulator):
        self.simulator = simulator

    def price(
            self,
            inst: CapFloor,
            n_paths: int = 5000,
            steps_per_year: int = 12,
            return_se: bool = False,
    ):
        T_max = max(cf.pay_date for cf in inst.schedule)
        dt = 1.0 / steps_per_year
        n_steps = int(np.ceil(T_max * steps_per_year)) + 1  # +1 buffer for rounding

        paths = self.simulator.simulate(
            dt=dt, n_steps=n_steps, n_paths=n_paths, Musiela=True,
        )
        df_paths = self._bank_account_dfs(paths, dt)

        pv = np.zeros(n_paths)
        for cf in inst.schedule:
            fix_idx = int(round(cf.fixing_time * steps_per_year))
            pay_idx = int(round(cf.pay_date * steps_per_year))

            L = self._simply_compounded_fwd(paths, fix_idx, cf.end - cf.start)
            payoff = np.maximum(inst.sign * (L - inst.strike), 0.0)
            pv += inst.notional * cf.accrual * df_paths[:, pay_idx] * payoff

        mean = float(pv.mean())
        se = float(pv.std(ddof=1) / np.sqrt(n_paths))
        return {'pv': mean, 'se': se} if return_se else mean

    def _simply_compounded_fwd(
            self,
            paths: np.ndarray,
            fix_idx: int,
            delta_yr: float,
    ) -> np.ndarray:
        """
        L = (1 / P(T_fix; T_end) − 1) / Δ from the simulated curve.
        Augments the [0, 1/12] gap by flat-extrapolating f(t, 0) ≈ f(t, x_min);
        bias is O(slope · 1/12), small for monthly grids.
        """
        sim = self.simulator
        delta_months = int(round(delta_yr * 12))
        if delta_months < 1 or delta_months > sim.n_tenors:
            raise ValueError(
                f"accrual {delta_yr} yr → {delta_months}M is outside tenor grid (1..{sim.n_tenors}M)."
            )

        f_first = paths[:, fix_idx, :1]                                   # f(T_fix, x_min) as f(T_fix, 0)
        f_grid = paths[:, fix_idx, :delta_months]                         # (n_paths, delta_months)
        f_aug = np.concatenate([f_first, f_grid], axis=1)                 # (n_paths, delta_months + 1)
        x_aug = np.concatenate([[0.0], sim.tenors_yr[:delta_months]])     # (delta_months + 1,)

        integral = np.trapz(f_aug, x_aug, axis=1)
        P = np.exp(-integral)
        return (1.0 / P - 1.0) / delta_yr

    def _bank_account_dfs(self, paths: np.ndarray, dt: float) -> np.ndarray:
        """
        DF(t_n) = exp(-∫_0^{t_n} r(s) ds), trapezoid in time, with r(s) ≈ f(s, x_min).
        Returns (n_paths, n_steps + 1) with DF(t_0) = 1.
        """
        short = paths[:, :, 0]                                   # (n_paths, n_steps + 1)
        integrand = 0.5 * (short[:, :-1] + short[:, 1:]) * dt    # (n_paths, n_steps)
        cum = np.zeros_like(short)
        cum[:, 1:] = np.cumsum(integrand, axis=1)
        return np.exp(-cum)
