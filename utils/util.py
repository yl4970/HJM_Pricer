import numpy as np
from datetime import date as dt

DEFAULT_DATA_SOURCE = "fred"
DEFAULT_DATA_TYPE = "UST"

TM = { #nested tenor map

    "D": {
        "fred_UST": {
            "DGS1MO": 1,
            "DGS3MO": 3,
            "DGS6MO": 6,
            "DGS1": 12,
            "DGS2": 24,
            "DGS3": 36,
            "DGS5": 60,
            "DGS7": 84,
            "DGS10": 120,
            "DGS20": 240,
            "DGS30": 360
        },

        "fred_TBill": {
            "DTB4WK": 4/52,
            "DTB3": 3,
            "DTB6": 6,
            "DTB1YR": 12
        }
    },

    "W": {
        "fred_UST": {
            "WGS1MO": 1,
            "WGS3MO": 3,
            "WGS6MO": 6,
            "WGS1YR": 12,
            "WGS2YR": 24,
            "WGS3YR": 36,
            "WGS5YR": 60,
            "WGS7YR": 84,
            "WGS10YR": 120,
            "WGS20YR": 240,
            "WGS30YR": 360
        },

        "fred_TBill": {
            "WTB4WK": 4/52,
            "WTB3MS": 3,
            "WTB6MS": 6,
            "WTB1YR": 12
        }
    },

    "M": {
        "fred_UST": {
            "GS1M": 1,
            "GS3M": 3,
            "GS6M": 6,
            "GS1": 12,
            "GS2": 24,
            "GS3": 36,
            "GS5": 60,
            "GS7": 84,
            "GS10": 120,
            "GS20": 240,
            "GS30": 360
        },

        "fred_TBill": {
            "TB4WK": 4/52,
            "TB3MS": 3, 
            "TB6MS": 6,
            "TB1YR": 12
        }
    },

    "Y": {
        "fred_UST": {
            "RIFLGFCM01NA": 1,
            "RIFLGFCM03NA": 3,
            "RIFLGFCM06NA": 6,
            "RIFLGFCY01NA": 12,
            "RIFLGFCY02NA": 24,
            "RIFLGFCY03NA": 36,
            "RIFLGFCY05NA": 60,
            "RIFLGFCY07NA": 84,
            "RIFLGFCY10NA": 120,
            "RIFLGFCY20NA": 240,
            "RIFLGFCY30NA": 360
        },

        "fred_TBill": {
            "RIFSGFSW04NA": 4/52,
            "RIFSGFSM03NA": 3,
            "RIFSGFSM06NA": 6,
            "RIFSGFSY01NA": 12
        }
    }
}


def getDataFreq(
        time_horizon: list | np.ndarray,
 ) -> str:
    """
    Infer the dominant data frequency from the time index of forward curves.
    Returns one of {'D', 'W', 'M', 'Y'}.
    """
    if len(time_horizon) < 2:
        raise ValueError("At least two time points are required to infer frequency.")
    
    # Compute average spacing in days
    deltas = np.diff(time_horizon).astype("timedelta64[D]").astype(int)
    avg_days = int(np.round(deltas.mean()))

    if avg_days <= 2:
        return "D"
    elif avg_days <= 10:
        return "W"
    elif avg_days <= 40:
        return "M"
    else:
        return "Y"

ANNUALIZE_FACTOR = {
    "D": 252, # daily data
    "W": 52,  # weekly data
    "M": 12,  # monthly data
    "Y": 1    # annual data
}


MIN_PCA_K, MAX_PCA_K = 3, 5