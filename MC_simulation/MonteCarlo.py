import numpy as np
import typing
from pandas import bdate_range as bdate_range

from MC_simulation.drift import get_HJM_drifts as get_drift
from MC_simulation.volSurface import VolatilitySurface

class MCSimulation:
    def __init__(self, VS:VolatilitySurface,seed: typing.Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.VS = VS
        self.timeline = VS.timeline
        self.tenors = VS.tenors
        self.n_factors = VS.n_factors
    
    def drifts(self, degrees:list[int]):
        return get_drift(
            timeline=self.timeline, 
            tenors=self.tenors,
            local_vs=self.VS,
            degrees=degrees
                         )

    def sim(self, degrees:list[int], paths:int=1):

        self.dt = np.diff([t.toordinal() / len(
                            bdate_range(
                            start=f'{t.year}-01-01', 
                            end = f'{t.year}-12-31'
                            )) for t in self.VS.timeline])
        self.sqrt_dt = np.sqrt(self.dt)
        n_steps = len(self.dt)
        n_tenors = len(self.tenors)

        simulate_drifts = self.drifts(degrees)[1:]
        vol_tensor = np.stack([
            np.array(self.VS.localVols[t].polyfit(degrees=degrees)['fittedVols']).T  # ensure (n_tenors, n_factors)
            for t in self.timeline[1:]
        ])  # shape: (n_steps, n_tenors, n_factors)


        # Generate Brownian increments: shape (paths, n_steps, n_factors)
        dW = self.rng.normal(scale=1.0, size=(paths, n_steps, self.n_factors))

        # Calculate increments: drift * dt + vol * dW summed over factors
        # vol_tensor: (n_steps, n_tenors, n_factors)
        # dW: (paths, n_steps, n_factors)
        # increments: (paths, n_steps, n_tenors)
        drift_term = simulate_drifts[np.newaxis, :, :] * self.dt[np.newaxis, :, np.newaxis] # (1, n_steps, n_tenors)
        vol_dW_term =  np.einsum(
            'tnf, ptf -> ptn',
            vol_tensor,                               # (n_steps, n_tenors, n_factors)
            dW * self.sqrt_dt[np.newaxis, :, np.newaxis]   # (paths, n_steps, n_factors)
        )

        increments = drift_term + vol_dW_term # (paths, n_steps, n_tenors)

        # Initialize array for simulated paths: include initial time point zero
        paths_array = np.zeros((paths, n_steps+1, n_tenors))

        # Cumulative sum along time axis to get full paths
        paths_array[:, 1:, :] = np.cumsum(increments, axis=1)

        # Add cumulative increments to the initial forward curve
        sim_forward_curve = self.VS.windowed_fwds_df.to_numpy()[np.newaxis, :, :] + paths_array

        return paths_array, sim_forward_curve
