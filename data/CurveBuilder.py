import pandas as pd
import numpy as np

from datetime import datetime as dt
from scipy.interpolate import interp1d
from functools import cached_property
import warnings
warnings.filterwarnings("ignore")

from data.TsyYieldLoader import data_loader as dl

class CurveBuilder:
    def __init__(self, start_date: dt, end_date: dt, decimals: bool = True):
        self.yields = dl(start_date, end_date, decimals)
        
    @cached_property
    def forwards(self) -> pd.DataFrame:
        x = self.yields.columns
        t = list(range(x[0], x[-1]+1))
        fwd_curves = {}

        for i in self.yields.index:
            f_interp = interp1d(x, self.yields.loc[pd.to_datetime(i), :])
            z = f_interp(t)
            fwd_curve = np.concatenate(([z[0]], np.diff(z*t)/np.diff(t)))
            fwd_curves[i] = fwd_curve

        fwd_df = pd.DataFrame(fwd_curves).T
        fwd_df.columns += 1
        fwd_df = fwd_df.sort_index()
        return fwd_df
    