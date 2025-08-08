import numpy as np
import pandas as pd
from datetime import datetime as dt
from typing import List, Dict
from functools import cached_property
from collections import namedtuple
from MC_simulation.volatility import Volatility as vol, FitResult as FR

class VolatilitySurface:
    def __init__(self, 
                 forward_curves:pd.DataFrame=None, 
                 n_factors:int=3, 
                 window_months:int=3):
        
        self.forward_curves_df = forward_curves
        self.stationary_fwds_df = self.forward_curves_df.diff().dropna()
        self.tenors = list(self.stationary_fwds_df.columns)
        self.n_factors = n_factors
        self.window_months = window_months    

        self._get_windowed_timeline()
        self._get_localVols()

    def __repr__(self):
        return f"{self.__class__.__name__}(tenors={len(self.tenors)}, curve_start={self.timeline[0].strftime('%Y-%m-%d')}, curve_end={self.timeline[-1].strftime('%Y-%m-%d')}, rolling_window={self.window_months}mon, pca_n_components={self.n_factors}.)" 
    
    def _get_windowed_timeline(self):
        # Make sure window is able to fit in the dataset timeline length
        try: 
            self.pd_Btimeline = pd.bdate_range(
                start=self.forward_curves_df.index[0], 
                end=self.forward_curves_df.index[-1] - pd.DateOffset(months=self.window_months)
                ).tolist()
            self.timeline = [i for i in self.pd_Btimeline if i in self.forward_curves_df.index]
            self.windowed_forward_curves_df = self.forward_curves_df.loc[self.timeline]
            
    
        except: raise ValueError("Window size exceeds the time range of the provided DataFrame of forward curves.")

    def _get_localVols(self) -> Dict[pd.Timestamp, vol]:
        self.localVols= {}
        for t in self.timeline:
            windowed_bdays = self._get_BDays_in_window(t, self.window_months)
            windowed_df = self._get_fwds_in_window(self.stationary_fwds_df, t, windowed_bdays)
            windowed_cov_matrix = self._get_annualized_cov(windowed_df)
            self.localVols[t] = vol(self.tenors, windowed_cov_matrix, self.n_factors)
        return self.localVols
    
    def polyfit(self, 
                degrees:List[int]=None
                ) -> Dict[pd.Timestamp, FR]:
        self.fittedLocalVols = {}
        for t in self.timeline:
            self.fittedLocalVols[t] = self.localVols[t].polyfit(degrees)
        return self.fittedLocalVols

    @staticmethod
    def _get_BDays_in_window(
        t:pd.Timestamp=None, 
        window_months:int=3
        ) -> int:
        return len(pd.date_range(t, t+pd.DateOffset(months=window_months), freq='B'))

    @staticmethod
    def _get_fwds_in_window(
        forward_curves_df:pd.DataFrame=None, 
        t:pd.Timestamp=None, 
        num_BDays:int=66) -> pd.DataFrame:
        return forward_curves_df.loc[t:t+pd.tseries.offsets.BDay(num_BDays)]

    @staticmethod
    def _get_annualized_cov(
        stationary_fwds_df:pd.DataFrame=None
        ) -> np.ndarray:
        year = stationary_fwds_df.index[0].year
        bdays_in_year = len(pd.date_range(start=dt(year,1,1), end=dt(year,12,31)))
        return np.cov(stationary_fwds_df.T) * bdays_in_year
  