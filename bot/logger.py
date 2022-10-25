import logging


def setup_logger() -> logging.Logger:
    logger = logging.getLogger('bot')
    logger.setLevel(logging.INFO)
    stream_handler = _setup_stream_handler()
    logger.addHandler(stream_handler)
    return logger


def _setup_stream_handler() -> logging.StreamHandler:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s')
    handler.setFormatter(formatter)
    return handler
