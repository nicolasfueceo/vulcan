"""Logging utilities for VULCAN system."""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional

import structlog
from rich.console import Console
from rich.logging import RichHandler

from vulcan.types import LoggingConfig

# Constants
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
RICH_FORMAT = "%(message)s"


def setup_logging(config: LoggingConfig) -> None:
    """Setup structured logging with Rich console output.

    Args:
        config: Logging configuration.
    """
    # Clear existing handlers
    logging.getLogger().handlers.clear()

    # Setup console handler with Rich
    console = Console(stderr=True)
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=True,
        markup=True,
        rich_tracebacks=True,
    )
    console_handler.setFormatter(logging.Formatter(RICH_FORMAT))

    # Setup file handler if configured
    handlers = [console_handler]
    if config.file:
        file_handler = _create_file_handler(config)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, config.level.upper()),
        handlers=handlers,
        format=config.format,
    )

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
            if config.structured
            else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def _create_file_handler(config: LoggingConfig) -> logging.Handler:
    """Create rotating file handler.

    Args:
        config: Logging configuration.

    Returns:
        Configured file handler.
    """
    log_file = Path(config.file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Parse file size
    max_bytes = _parse_file_size(config.max_file_size)

    handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=max_bytes,
        backupCount=config.backup_count,
        encoding="utf-8",
    )

    formatter = logging.Formatter(config.format)
    handler.setFormatter(formatter)

    return handler


def _parse_file_size(size_str: str) -> int:
    """Parse file size string to bytes.

    Args:
        size_str: Size string like "10MB", "1GB", etc.

    Returns:
        Size in bytes.
    """
    size_str = size_str.upper().strip()

    # Extract number and unit
    if size_str.endswith("KB"):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith("MB"):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith("GB"):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        # Assume bytes
        return int(size_str)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name.

    Returns:
        Configured logger instance.
    """
    return structlog.get_logger(name)


class VulcanLogger:
    """Context-aware logger for VULCAN operations."""

    def __init__(self, name: str) -> None:
        """Initialize logger with name.

        Args:
            name: Logger name.
        """
        self._logger = get_logger(name)
        self._context = {}

    def bind(self, **kwargs) -> "VulcanLogger":
        """Bind context to logger.

        Args:
            **kwargs: Context key-value pairs.

        Returns:
            New logger instance with bound context.
        """
        new_logger = VulcanLogger(self._logger._context.get("logger", "vulcan"))
        new_logger._logger = self._logger.bind(**kwargs)
        new_logger._context = {**self._context, **kwargs}
        return new_logger

    def bind_experiment(
        self, experiment_id: str, experiment_name: Optional[str] = None
    ) -> "VulcanLogger":
        """Bind experiment context.

        Args:
            experiment_id: Experiment ID.
            experiment_name: Optional experiment name.

        Returns:
            Logger with experiment context.
        """
        context = {"experiment_id": experiment_id}
        if experiment_name:
            context["experiment_name"] = experiment_name
        return self.bind(**context)

    def bind_node(self, node_id: str) -> "VulcanLogger":
        """Bind MCTS node context.

        Args:
            node_id: Node ID.

        Returns:
            Logger with node context.
        """
        return self.bind(node_id=node_id)

    def bind_feature(self, feature_name: str) -> "VulcanLogger":
        """Bind feature context.

        Args:
            feature_name: Feature name.

        Returns:
            Logger with feature context.
        """
        return self.bind(feature_name=feature_name)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self._logger.error(message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self._logger.critical(message, **kwargs)


def get_vulcan_logger(name: str) -> VulcanLogger:
    """Get a VULCAN context-aware logger.

    Args:
        name: Logger name.

    Returns:
        VULCAN logger instance.
    """
    return VulcanLogger(name)
