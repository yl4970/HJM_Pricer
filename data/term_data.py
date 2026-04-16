from dataclasses import dataclass

import pandas as pd
import numpy as np

@dataclass(frozen=True)
class TermStructureData:
    time: np.ndarray
    tenors: np.ndarray
    values: np.ndarray

    def __post_init__(self):

        # check legal shape of data object
        if self.values.shape !=(len(self.time), len(self.tenors)):
            raise ValueError(
                f"Inconsistent shapes: values {self.values.shape}, "
                f"time {len(self.time)}, tenors {len(self.tenors)}."
            )
        
        # check legal tenor formats
        if not np.all(np.diff(self.tenors) > 0):
            raise ValueError("Tenors must be strictly increasing.")
        
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.values,
            index=self.time, 
            columns=self.tenors
        )
    
    def __repr__(self):
        text_sdate = np.datetime_as_string([self.time[0]], unit='D')[0]
        text_edate = np.datetime_as_string([self.time[-1]], unit='D')[0]
        return f"time=({text_sdate}, {text_edate}); tenors={len(self.tenors)}M; values.shape={self.values.shape}."
