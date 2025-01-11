import logging
from typing import Optional


def setup_logger(name: str,
                 level: int = logging.INFO,
                 log_file: Optional[str] = None
                 ) -> logging.Logger:
    """
    Sets up a logger with the specified name, logging level, and optional file logging.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (optional)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# Configure main logger and other specific loggers
LOG_FILE = "app.log"
main_logger = setup_logger(
    "main_logger", level=logging.INFO, log_file=LOG_FILE)
embed_logger = setup_logger("embed_contexts", level=logging.WARNING)
