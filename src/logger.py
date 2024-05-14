import logging

LOGFILE = 'out.log'

logname2level = {
    'CRITICAL': logging.CRITICAL,
    'FATAL': logging.FATAL,
    'ERROR': logging.ERROR,
    'WARN': logging.WARNING,
    'WARNING': logging.WARNING,
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG,
    'NOTSET': logging.NOTSET,
}

class LoggerConfig:
    def __init__(self, name: str, level: str = 'INFO', verbose: bool = False) -> None:
        self.name = name
        self.level = logname2level.get(level)
        self.verbose = verbose

    def get(self):
        logger = logging.getLogger(self.name)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        if logger.hasHandlers():
            logger.handlers.clear()

        # Create file handler
        fh = logging.FileHandler(filename=LOGFILE, encoding="utf-8", mode='w')
        fh.setLevel(self.level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        if self.verbose:
            # create console handler
            ch = logging.StreamHandler()
            ch.setLevel(self.level)
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        log_filter = logging.Filter(self.name)
        logger.addFilter(log_filter)
        logger.setLevel(self.level)
        logger.propagate = False
        return logger
