import logging

class LoggerGenerator:
    """
    Class for generating logger.

    Parameters
    ----------
    log_filename : str, optional
        File name of the log file. Default is 'logfile.log'.

    Examples
    --------
    >>> from pcl_monitor.utils._logger import LoggerGenerator
    >>> class TestLogger:
    ...     def __init__(self):
    ...         self.LOGGER = LoggerGenerator().generate(self.__class__.__name__)
    ...     def clear_log(self):
    ...         self.LOGGER.handlers.clear()
    >>> tg = TestLogger()
    >>> tg.LOGGER.info("Test Logger output.")
    >>> tg.clear_log()
    """

    def __init__(self, log_filename=None):
        if not log_filename:
            self.log_filename = 'logfile.log'
        else:
            self.log_filename = log_filename

        console_formatter = logging.Formatter(
            'Module:%(filename)20s -- %(name)s.%(funcName)s() -- %(levelname)s:%(message)s'
        )
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(console_formatter)
        self.console_handler = console_handler

        file_formatter = logging.Formatter(
            '%(asctime)s -- Module:%(filename)20s -- %(name)s.%(funcName)s() -- %(levelname)s:%(message)s'
        )
        file_handler = logging.FileHandler(self.log_filename)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        self.file_handler = file_handler

    def generate(self, name):
        logger = logging.getLogger(name)
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == self.file_handler.baseFilename
                   for h in logger.handlers):
            logger.addHandler(self.file_handler)
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            logger.addHandler(self.console_handler)
        logger.setLevel(logging.DEBUG)
        return logger