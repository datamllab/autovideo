import logging

def _get_logger(log_path=None):
    if log_path is None:
        shandle = logging.StreamHandler()
    else:
        shandle = logging.FileHandler(log_path, 'w')
    shandle.setFormatter(
        logging.Formatter(
            '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
            '%(message)s'))
    logger = logging.getLogger('autovideo')
    logger.propagate = False
    logger.addHandler(shandle)
    logger.setLevel(logging.INFO)

    return logger

logger = _get_logger()

def set_log_path(log_path):
    global logger
    logger = _get_logger(log_path)
