# src/utils/decorators.py
import time
from functools import wraps

from loguru import logger


def agent_run_decorator(agent_name: str):
    """
    A decorator to log the duration of an agent's run method and write it to TensorBoard.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            logger.info(f"{agent_name} started.")
            start_time = time.time()

            result = func(self, *args, **kwargs)

            end_time = time.time()
            duration = end_time - start_time

            if hasattr(self, "writer") and self.writer is not None:
                run_count = getattr(self, "run_count", 0)
                self.writer.add_scalar("run_duration_seconds", duration, run_count)

            logger.info(f"{agent_name} finished in {duration:.2f} seconds.")
            return result

        return wrapper

    return decorator
