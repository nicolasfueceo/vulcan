import logging
import sys

from src.utils.run_utils import format_log_message, get_run_logs_dir


class RunContextFormatter(logging.Formatter):
    """Custom formatter that includes run context in log messages."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with run context."""
        record.msg = format_log_message(str(record.msg))
        return super().format(record)


def setup_logging(log_level: int = logging.INFO) -> None:
    """Set up logging configuration with run-specific paths."""
    # Create run-specific log directory
    log_dir = get_run_logs_dir()
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = RunContextFormatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler
    log_file = log_dir / "pipeline.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_formatter = RunContextFormatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)
