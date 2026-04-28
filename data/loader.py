from abc import ABC, abstractmethod

import pandas_datareader.data as web
from datetime import date as dt
import pandas as pd
import numpy as np

import logging
from utils.logging import setup_logger
from .term_data import TermStructureData
from utils.util import TM, DEFAULT_DATA_SOURCE, DEFAULT_DATA_TYPE
logger = setup_logger(__name__)

# ====================================================================================================
#                          Loader to download data through various methods 
# ====================================================================================================
class TermStructureLoader(ABC):
    @abstractmethod
    def load(self, sdate:dt, edate:dt, DataFreq: str) -> TermStructureData:
        pass

class FREDtsdLoader(TermStructureLoader):
    """
    Load yield curve data from FRED.
    """
    def __init__(self, 
                data_source: str = DEFAULT_DATA_SOURCE,
                data_type: str = DEFAULT_DATA_TYPE,
                overwrite_source: str = None,
                overwrite_type: str = None
        ):
        self._base_source = data_source
        self._base_type = data_type
        self._overwrite_source = overwrite_source
        self._overwrite_type = overwrite_type
        self._ow_flag = False if not self._overwrite_source and not self._overwrite_type else True
        self._ow_name = None

    def load(self, sdate:dt, edate:dt, DataFreq: str = "D") -> TermStructureData:
        """
        sdate: start date
        edate: end date
        """
        if sdate is None or edate is None:
            raise ValueError("Both sdate and edate must be provided.")

        if sdate >= edate:
            raise ValueError(
                f"sdate must be earlier than edate, got sdate={sdate}, edate={edate}"
            )
        
        base_tenor_map = TM[DataFreq]["%s_%s" % (self._base_source, self._base_type)]

        if self._ow_flag:
            self._ow_name = "%s_%s" % (DEFAULT_DATA_SOURCE, self._overwrite_type)
            if not self._overwrite_source:
                logger.warning(
                    f"Unspecified data source to overwrite, setting to default data source: "
                    f"{DEFAULT_DATA_SOURCE}."
                )
                self._ow_name = "%s_%s" % (DEFAULT_DATA_SOURCE, self._overwrite_type)
            elif not self._overwrite_type:
                logger.warning(
                    "Unspecified data type to overwrite; will not overwrite."
                )
                self._ow_flag, self._ow_name = False, None

        overwrite_tm = None
        if self._ow_flag:
            try:
                overwrite_tm = TM[DataFreq][self._ow_name]
            except KeyError:
                logger.warning(
                    "Invalid data source and data type combo to overwrite. "
                    f"Got {self._ow_name}, need {list(TM[DataFreq].keys())}, "
                    "unable to overwrite."
                )
                self._ow_flag, self._ow_name = False, None

        ow_tmin, ow_tmax = (-1,1e9)
        if self._ow_flag and overwrite_tm is not None:
            ow_tmin, ow_tmax = (
                min(overwrite_tm.values()), 
                max(overwrite_tm.values())
            )

        tenor_to_series = {
            tenor: sid 
            for (sid, tenor) in base_tenor_map.items()
            if not (ow_tmin <= tenor <= ow_tmax)
            }

        if self._ow_flag and overwrite_tm is not None:
            for sid, tenor in overwrite_tm.items():
                tenor_to_series[tenor] = sid
        
        series_id = list(tenor_to_series.values())
        new_tenor_map = {sid: tenor for (tenor, sid) in tenor_to_series.items()}
        df = web.DataReader(series_id, self._base_source, sdate, edate)
        df = df.dropna() 
        df = df/100

        df = df.rename(columns=new_tenor_map)
        df = df.sort_index()

        tenors = np.array(sorted(df.columns))
        df = df[tenors]
        
        time = pd.to_datetime(df.index).to_numpy()
        values = df.values.astype(float)

        return TermStructureData(
            time=time,
            tenors=tenors,
            values=values
            )
        
