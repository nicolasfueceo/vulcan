from functools import wraps
from typing import Callable

def catch_and_log(logger):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Exception in {func.__name__}: {e}")
                return None
        return wrapper
    return decorator
