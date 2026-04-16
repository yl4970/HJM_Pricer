import logging

VERBOSE = 15
logging.addLevelName(VERBOSE, "VERBOSE")

def verbose(self, msg, *args, **kwargs):
    if self.isEnabledFor(VERBOSE):
        self._log(VERBOSE, msg, args, **kwargs)

logging.Logger.verbose = verbose

def setup_logger(
        name: str,
        level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(name)

    # prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(level)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False 

    return logger

