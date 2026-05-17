"""
src/utils/logger.py — Centralized Logger
=========================================
Provides a single reusable logger used by every module
so all pipeline output is consistently formatted.
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger with console output.
    Call this at the top of every module:
        log = get_logger(__name__)
    """
    logger = logging.getLogger(name)

    if not logger.handlers:                      # avoid duplicate handlers
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
