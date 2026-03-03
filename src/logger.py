"""
logger.py – Centralised logging configuration for the Interview Predictor.

Import and use the get_logger() factory in any module to get a consistently
formatted logger without repeating basicConfig boilerplate everywhere.

Usage:
    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Starting training …")
"""

import logging
import sys
from pathlib import Path

# Default log file inside the project root
_DEFAULT_LOG_FILE = Path(__file__).resolve().parent.parent / "app.log"

_LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s – %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_handlers_configured = False


def _configure_root(log_file: Path | None = _DEFAULT_LOG_FILE,
                    level: int = logging.INFO) -> None:
    """Configure the root logger with a stream handler (and optionally a file handler)."""
    global _handlers_configured
    if _handlers_configured:
        return

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(sh)

    # File handler (silently skip if we cannot write)
    if log_file is not None:
        try:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
            root.addHandler(fh)
        except OSError:
            root.warning(f"Could not open log file at {log_file}. File logging disabled.")

    _handlers_configured = True


def get_logger(name: str,
               level: int = logging.INFO,
               log_file: Path | None = _DEFAULT_LOG_FILE) -> logging.Logger:
    """
    Return a named logger, ensuring the root handler is configured once.

    Parameters
    ----------
    name     : Logger name (typically __name__ of the calling module)
    level    : Logging level (default: INFO)
    log_file : Path to a log file; pass None to disable file logging

    Returns
    -------
    logging.Logger
    """
    _configure_root(log_file=log_file, level=level)
    return logging.getLogger(name)
