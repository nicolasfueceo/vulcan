import sys
import logging

from loguru import logger

from src.utils.logging_utils import InterceptHandler
from src.utils.run_utils import get_run_log_file


def setup_logging(log_level: str = "INFO") -> None:
    """Set up Loguru to be the main logging system."""
    # Remove default handler to avoid duplicate logs
    logger.remove()

    # Add a console sink
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=False,
    )

    # Add a file sink for the main pipeline log
    log_file = get_run_log_file()
    logger.add(
        log_file,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="10 days",
        enqueue=True,  # Make logging non-blocking
        backtrace=True,
        diagnose=True,
    )

    # Intercept standard logging messages
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)


def get_logger(name: str):
    """Get a logger with the specified name (compatible with loguru)."""
    return logger.bind(name=name)
