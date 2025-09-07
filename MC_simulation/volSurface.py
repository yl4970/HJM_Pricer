import numpy as np
import pandas as pd
from typing import Optional, Any, Callable, ParamSpec, Concatenate, Iterable
from functools import wraps
from dataclasses import dataclass, field
from tqdm import tqdm

from alt_volSurface.volatility import PCAResult

DateKey = pd.Timestamp
P = ParamSpec("P")

def vectorize_over_dates(func:Callable[Concatenate[object, DateKey, P], tuple]):
    @wraps(func)
    def wrapper(self, dates:Iterable[object], **kwargs:P.kwargs):
        sample_output = func(self, dates[0], **kwargs)
        n_keys = 1
        if isinstance(sample_output, dict):
            keys = sample_output.keys()
            n_keys = len(keys)
        output_dict = [dict() for _ in range(n_keys)]

        for date in tqdm(dates):
            output = func(self, date, **kwargs)
            if isinstance(output, dict):
                for dct, out in zip(output_dict, output):
                    dct[date] = output[out]
            else:
                output_dict[0][date] = output
    
        return output_dict[0] if n_keys==1 else tuple(output_dict)
    return wrapper

@dataclass(slots=True)
class VolatilitySurface:
    forward_curves: pd.DataFrame = None
    localVol_window_months: int = None
    n_factors: int = None

    _full_timeline: Optional[list[DateKey]] = field(default=None, init=False)
    tenors: Optional[list[int]] = field(default=None, init=False)
    timeline: Optional[list[DateKey]] = field(default=None, init=False)
    bdays_dict: Optional[dict[int, int]] = field(default=None, init=False)
    windowed_bdays: Optional[dict[DateKey, int]] = field(default=None, init=False)
    windowed_fwds: Optional[dict[DateKey, pd.DataFrame]] = field(default=None, init=False)
    windowed_fwds_df: pd.DataFrame = field(default=None, init=False)
    localVols: Optional[dict[DateKey, PCAResult]] = field(default=None, init=False)

    @property
    def window(self) -> int:
        return self.localVol_window_months

    def __post_init__(self):
        self._full_timeline = self.forward_curves.index
        self.tenors = list(self.forward_curves.columns)
        self.windowed_bdays = self._get_bdays_within_window(self._full_timeline)
        self._get_windowed_timeline()
        self.windowed_fwds_df = self.forward_curves.loc[self.timeline]
        self._get_bdays_dict()
        self._check_if_nobs_deficient()

    def build(self):
        self.windowed_fwds = self._get_fwds_within_window(self.timeline)
        self.localVols = self._pca(self.timeline)

    def _get_windowed_timeline(self):
        timeline_start = min(self.windowed_bdays.keys())
        windowed_timeline_start = timeline_start + pd.offsets.BusinessDay(n=self.windowed_bdays[timeline_start])
        self.timeline = [i for i in self._full_timeline if i>= windowed_timeline_start]

    def _check_if_nobs_deficient(self):
        n_obs = len(self.timeline)
        min_windowed_bdays = min(self.windowed_bdays.items())[1]
        if min_windowed_bdays > n_obs:
            raise ValueError(f'Rolling window ({min_windowed_bdays}) exceeds available history ({n_obs}).')

    # @vectorize_over_dates
    # def _get_localVol(self, DateKey, windowed_fwds) -> pd.DataFrame:
    #     return Volatility(windowed_fwds[DateKey], self.n_factors)
    
    @vectorize_over_dates
    def _get_fwds_within_window(self, DateKey) -> pd.DataFrame:
        start = DateKey + pd.DateOffset(months=self.window * (-1))
        if start + pd.tseries.offsets.BDay(0) != start:
            start += pd.tseries.offsets.BDay(-1)
        return self.forward_curves.loc[start : DateKey]
    
    @vectorize_over_dates
    def _get_bdays_within_window(self, DateKey) -> int:
        return len(pd.date_range(DateKey+pd.DateOffset(months=self.window * (-1)), DateKey, freq='B'))
    
    @vectorize_over_dates
    def polyfit(self, DateKey, degrees:tuple[int]=None) -> dict:
        return self.localVols[DateKey].polyfit(degrees)['fittedVols']

    def _get_bdays_dict(self):
        self.bdays_dict = {
            year: len(
                pd.bdate_range(start=f'{year}-1-1', end=f'{year}-12-31')) for year in 
            list(range(self.timeline[0].year, self.timeline[-1].year+1))
        }

    @vectorize_over_dates
    def _pca(self, DateKey):
        df = self.windowed_fwds[DateKey]
        dX = np.diff(df, axis=0)
        dX -= dX.mean(axis=0, keepdims=True)
        m, n = dX.shape
        if m < n:
            M = 1/(m-1) * (dX @ dX.T)
            s, U = np.linalg.eigh(M.astype(float))
            idx = s.argsort()[::-1][:self.n_factors]
            s, U = s[idx], U[:, idx]

            eps = 1e-18
            V = (dX.T @ U) / np.sqrt((m-1) * np.maximum(s, eps))
        else:
            s, V = np.linalg.eigh(np.cov(dX))

        return PCAResult(V=V, s=s, bdays_in_year=self.bdays_dict[DateKey.year], tenors=self.tenors)
