import numpy as np
import pandas as pd
from typing import Optional, Any, Callable, ParamSpec, Concatenate, Iterable
from functools import wraps
from dataclasses import dataclass, field
from tqdm import tqdm

from volatility.pca_result import PCAResult

DateKey = pd.Timestamp
P = ParamSpec("P")


def _top_k_eigh(symmetric: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    s, U = np.linalg.eigh(symmetric)
    idx = s.argsort()[::-1][:k]
    return s[idx], U[:, idx]

def vectorize_over_dates(func:Callable[Concatenate[object, DateKey, P], tuple]):
    @wraps(func)
    def wrapper(self, dates:Iterable[object], **kwargs:P.kwargs):
        if len(dates) == 0:
            return {}

        def _store(stores, date, output):
            if isinstance(output, dict):
                for dct, key in zip(stores, output):
                    dct[date] = output[key]
            else:
                stores[0][date] = output

        # Probe the first date once so we can size the output containers, then
        # iterate over the remainder. This avoids re-running func on dates[0].
        first = func(self, dates[0], **kwargs)
        n_keys = len(first) if isinstance(first, dict) else 1
        stores = [dict() for _ in range(n_keys)]
        _store(stores, dates[0], first)

        for date in tqdm(dates[1:]):
            _store(stores, date, func(self, date, **kwargs))

        return stores[0] if n_keys == 1 else tuple(stores)
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
            # Time-domain trick: eig of the (m,m) Gram dominates when tenors > history.
            gram = (dX @ dX.T).astype(float) / (m - 1)
            s, U = _top_k_eigh(gram, self.n_factors)
            eps = 1e-18
            V = (dX.T @ U) / np.sqrt((m - 1) * np.maximum(s, eps))
        else:
            s, V = _top_k_eigh(np.cov(dX, rowvar=False), self.n_factors)

        # Sign stabilization across rolling windows: anchor the sign at the
        # max-magnitude tenor so polyfit and the HJM drift don't flip between dates.
        for k in range(V.shape[1]):
            if V[np.argmax(np.abs(V[:, k])), k] < 0:
                V[:, k] *= -1

        return PCAResult(V=V, s=s, bdays_in_year=self.bdays_dict[DateKey.year], tenors=self.tenors)
    