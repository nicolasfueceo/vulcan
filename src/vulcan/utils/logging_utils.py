#!/usr/bin/env python
"""Logging utilities for VULCAN."""

import logging
import logging.handlers
from pathlib import Path

from vulcan.schemas import LoggingConfig

# Configure structlog if it's the primary logging system
# This is a basic setup; actual structlog configuration might be more complex
# and exist elsewhere (e.g., in get_vulcan_logger)


def get_root_logger() -> logging.Logger:
    """Get the root logger."""
    return logging.getLogger()


def setup_experiment_file_logging(
    experiment_log_file: Path,
    config: LoggingConfig,
    # vulcan_config: VulcanConfig # Might be needed for global log level
) -> None:
    """Sets up a rotating file handler for the current experiment.

    Args:
        experiment_log_file: Path to the dedicated log file for the experiment.
        config: The logging configuration object from VulcanConfig.
    """
    logger = get_root_logger()

    # Determine log level
    log_level_str = config.level.upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Create formatter
    # If structlog is used and configured to process standard library logs,
    # its formatting will apply. Otherwise, this basic formatter is used for the file.
    formatter = logging.Formatter(
        config.format
        if config.format
        else "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create a rotating file handler
    try:
        # Convert max_file_size from string like "10MB" to bytes
        size_str = config.max_file_size.upper()
        if size_str.endswith("MB"):
            max_bytes = int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith("KB"):
            max_bytes = int(size_str[:-2]) * 1024
        elif size_str.endswith("B"):
            max_bytes = int(size_str[:-1])
        else:
            max_bytes = 10 * 1024 * 1024  # Default to 10MB if format is unknown
    except ValueError:
        max_bytes = 10 * 1024 * 1024  # Default in case of parsing error

    file_handler = logging.handlers.RotatingFileHandler(
        filename=experiment_log_file,
        maxBytes=max_bytes,
        backupCount=config.backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)  # Set level for this handler

    # Add the handler to the root logger
    # This ensures that logs from any module using standard logging (or structlog routing to it)
    # will also go to this file, respecting the handler's level.
    logger.addHandler(file_handler)

    # It might also be necessary to ensure the root logger's level is at least
    # as verbose as the most verbose handler. For example, if root is WARNING,
    # an INFO handler won't get INFO messages.
    if logger.level == logging.NOTSET or logger.level > log_level:
        logger.setLevel(log_level)  # Set root logger level if it's too restrictive

    # Initial log to confirm setup
    # Use a direct logger instance to avoid issues if structlog isn't fully configured yet
    initial_setup_logger = logging.getLogger(__name__)
    initial_setup_logger.info(
        f"Experiment file logging configured. Outputting to: {experiment_log_file}"
    )
    initial_setup_logger.info(f"File log level set to: {log_level_str}")


# Example of a more general console logger setup (might exist in get_vulcan_logger or main)
# def setup_console_logging(level: str = "INFO"):
#     logger = get_root_logger()
#     console_handler = logging.StreamHandler(sys.stdout)
#     log_level_val = getattr(logging, level.upper(), logging.INFO)
#     console_handler.setLevel(log_level_val)
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     console_handler.setFormatter(formatter)
#     if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
#         logger.addHandler(console_handler)
#     if logger.level == logging.NOTSET or logger.level > log_level_val:
#        logger.setLevel(log_level_val)
