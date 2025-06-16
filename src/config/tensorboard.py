import subprocess
from loguru import logger
from typing import Optional

from torch.utils.tensorboard import SummaryWriter

from src.utils.run_utils import get_run_tensorboard_dir


def start_tensorboard() -> None:
    """Start TensorBoard in the background for a global (non-run-specific) log directory."""
    log_dir = "runtime/tensorboard_global"
    try:
        subprocess.Popen(
            ["tensorboard", "--logdir", str(log_dir), "--port", "6006"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        logger.warning(f"Could not start TensorBoard: {e}")


def get_tensorboard_writer() -> SummaryWriter:
    """Get a TensorBoard writer for the global (non-run-specific) TensorBoard log directory."""
    log_dir = "runtime/tensorboard_global"
    return SummaryWriter(log_dir=str(log_dir))


def log_metric(
    writer: SummaryWriter, tag: str, value: float, step: Optional[int] = None
) -> None:
    """Log a metric to TensorBoard."""
    writer.add_scalar(tag, value, step)


def log_metrics(
    writer: SummaryWriter, metrics: dict, step: Optional[int] = None
) -> None:
    """Log multiple metrics to TensorBoard."""
    for tag, value in metrics.items():
        log_metric(writer, tag, value, step)


def log_hyperparams(writer: SummaryWriter, hparams: dict) -> None:
    """Log hyperparameters to TensorBoard."""
    writer.add_hparams(hparams, {})
