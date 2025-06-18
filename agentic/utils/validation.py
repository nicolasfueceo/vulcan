from functools import wraps
from typing import Callable, List

def require_keys(keys: List[str]):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(d, *args, **kwargs):
            missing = [k for k in keys if k not in d]
            if missing:
                raise KeyError(f"Missing required keys: {missing}")
            return func(d, *args, **kwargs)
        return wrapper
    return decorator
