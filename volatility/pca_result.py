import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PCAResult:
    V: np.ndarray
    s: np.ndarray
    bdays_in_year: int
    tenors: list[int]

    vols_daily: Optional[np.ndarray] = field(default=None, init=False)
    vols_annually: Optional[np.ndarray] = field(default=None, init=False)

    def __post_init__(self):
        # s is the eigenvalue of the per-step (daily) sample covariance of dF.
        # Loadings carry units of [rate/sqrt(time)]; annualization scales by sqrt(bdays).
        self.vols_daily = self.V * np.sqrt(self.s)
        self.vols_annually = self.V * np.sqrt(self.s * self.bdays_in_year)

    def __repr__(self):
        return f"Principal component volatility(shape={self.vols_annually.shape}, matrix={self.vols_annually[:3]}...)"

    def polyfit(self, degrees: list[int] = None) -> dict:
        n_factors = self.V.shape[1]
        if len(degrees) != n_factors:
            raise ValueError(f'Expected {n_factors} degrees, got {len(degrees)}.')
        if any(d < 0 for d in degrees):
            raise ValueError('Expected all non-negative degrees.')

        # Fit each factor's annualized vol loading across tenors so downstream
        # consumers (drift integral, MC vol_tensor) see σ_j(x) in proper units.
        params, fitted = [], []
        for index, sigma_j in enumerate(self.vols_annually.T):
            p = np.polyfit(self.tenors, sigma_j, degrees[index])
            params.append(p)
            fitted.append(np.polyval(p, self.tenors))
        return {'params': params, 'fittedVols': fitted}
