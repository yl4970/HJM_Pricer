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
        self.vols_daily = self.V * np.sqrt(self.s)
        self.vols_annually = self.vols_daily * self.bdays_in_year

    def __repr__(self):
        return f"Principal component volatility(shape={self.vols_annually.shape}, matrix={self.vols_annually[:3]}...)"
    
    def polyfit(self, degrees:list[int]=None)->dict:
        if len(degrees) != self.V.shape[1]:
            raise ValueError(f'Expected {self.n_factors} degrees, got {len(degrees)}.')
        if any(d<0 for d in degrees):
            raise ValueError(f'Expected all non-negative degrees.')
        
        self.polyparams = []
        self.polyfittedVols = []
        for index, v in enumerate(self.V.T):
            params = np.polyfit(self.tenors, v, degrees[index])
            self.polyparams.append(params)
            self.polyfittedVols.append(np.polyval(params, self.tenors))
        
        return {'params': self.polyparams, 'fittedVols': self.polyfittedVols}
