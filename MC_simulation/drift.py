import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from MC_simulation.volSurface import VolatilitySurface

def get_HJM_drifts(
        timeline:list[pd.Timestamp]=None, 
        tenors:list[int]=np.linspace(1,360,360), 
        local_vs:VolatilitySurface=None, 
        degrees:list[int]=None,
        Musiela:bool=True
        ) -> dict[pd.Timestamp, list[float]]:
    """
    Compute a dictionary of drift vectors across the given timeline and tenor structure.
    """
    drift_curve = []
    vol_surface = local_vs.polyfit(degrees)
    for t in timeline:
        drifts = np.zeros(len(local_vs.tenors), dtype=complex)
        for i in range(local_vs.n_factors):
            vol_points = vol_surface[t].fittedVols[i]
            integrals = np.array([
                trapezoid(
                    vol_surface[t].fittedVols[i][:tenor],
                    tenors[:tenor]
                    ) for tenor in tenors
                         ])
            drifts += vol_points * integrals
        drift_curve.append(drifts)
    drift_curve = np.array(drift_curve)

    if Musiela:
        drift_curve += np.gradient(
            local_vs.windowed_forward_curves_df, 
            tenors, 
            axis=1
            )
    return drift_curve