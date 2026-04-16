import numpy as np

from data.term_data import TermStructureData as tsd

import logging
from utils.logging import setup_logger
from utils.util import getDataFreq
from .iv_pca import iv_pca

logger = setup_logger(__name__)
IV_METHODS = {
    "pca": iv_pca
}

def getIV(
        fwd_curves: tsd, 
        method: str
):
    dF = np.diff(fwd_curves.values, axis=0)
    freq = getDataFreq(fwd_curves.time)

    try:
        iv_func = IV_METHODS[method]
    except KeyError:
        raise ValueError(
            f"Unknown method to estaimte implied vol'{method}'."
            f"Registered methods: {list(IV_METHODS.keys())}."
        )

    logger.info("Estimating implied volatility", extra={"method": method})

    return iv_func(dF, freq)

    