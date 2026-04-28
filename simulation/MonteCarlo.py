import typing

import numpy as np

from simulation.drift import get_HJM_drifts as get_drift
from simulation.volSurface import VolatilitySurface


class MCSimulation:
    def __init__(self, VS: VolatilitySurface, seed: typing.Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.VS = VS
        self.timeline = VS.timeline
        self.tenors = VS.tenors
        self.n_factors = VS.n_factors

    def drifts(self, degrees: list[int], vol_surface: typing.Optional[dict] = None):
        return get_drift(
            local_vs=self.VS,
            degrees=degrees,
            timeline=self.timeline,
            tenors=self.tenors,
            vol_surface=vol_surface,
        )

    def _year_fractions(self) -> np.ndarray:
        """
        dt[i] = business days between timeline[i] and timeline[i+1], expressed as
        a year fraction using that calendar year's bday count. The forward-curve
        timeline is already business-day indexed, so consecutive entries are usually
        one bday apart; this still does the right thing across weekends/holidays.
        """
        dates = np.array([t.date() for t in self.VS.timeline], dtype='datetime64[D]')
        bdays = np.busday_count(dates[:-1], dates[1:]).astype(float)
        denom = np.array([self.VS.bdays_dict[t.year] for t in self.VS.timeline[1:]], dtype=float)
        return bdays / denom

    def sim(self, degrees: list[int], paths: int = 1):
        self.dt = self._year_fractions()
        self.sqrt_dt = np.sqrt(self.dt)
        n_steps = len(self.dt)
        n_tenors = len(self.tenors)

        # Compute polyfits once; share with drift to avoid duplicate work.
        vol_surface = self.VS.polyfit(self.timeline, degrees=degrees)
        simulate_drifts = self.drifts(degrees, vol_surface=vol_surface)[1:]
        vol_tensor = np.stack([
            np.asarray(vol_surface[t]).T for t in self.timeline[1:]
        ])  # (n_steps, n_tenors, n_factors)

        dW = self.rng.normal(scale=1.0, size=(paths, n_steps, self.n_factors))

        drift_term = simulate_drifts[np.newaxis, :, :] * self.dt[np.newaxis, :, np.newaxis]
        vol_dW_term = np.einsum(
            'tnf, ptf -> ptn',
            vol_tensor,
            dW * self.sqrt_dt[np.newaxis, :, np.newaxis],
        )

        increments = drift_term + vol_dW_term
        paths_array = np.zeros((paths, n_steps + 1, n_tenors))
        paths_array[:, 1:, :] = np.cumsum(increments, axis=1)

        sim_forward_curve = self.VS.windowed_fwds_df.to_numpy()[np.newaxis, :, :] + paths_array

        return paths_array, sim_forward_curve
