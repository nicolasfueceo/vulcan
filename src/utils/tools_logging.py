from functools import wraps
from datetime import datetime
import inspect

# This wrapper assumes the tool function signature includes session_state or can be passed one.
def log_tool_call(tool_func, session_state, tool_name=None):
    name = tool_name or tool_func.__name__
    @wraps(tool_func)
    def wrapper(*args, **kwargs):
        input_args = inspect.getcallargs(tool_func, *args, **kwargs)
        # Remove session_state from args for logging clarity
        input_args_log = {k: v for k, v in input_args.items() if k != 'session_state'}
        logger = getattr(session_state, 'run_logger', None)
        agent = kwargs.get('agent', None)
        start_time = datetime.utcnow().isoformat() + 'Z'
        try:
            output = tool_func(*args, **kwargs)
            if logger:
                logger.log_tool_call(
                    tool_name=name,
                    input_args=input_args_log,
                    output=output,
                    agent=agent,
                    extra={"start_time": start_time, "success": True}
                )
            return output
        except Exception as e:
            if logger:
                logger.log_tool_call(
                    tool_name=name,
                    input_args=input_args_log,
                    output=str(e),
                    agent=agent,
                    extra={"start_time": start_time, "success": False}
                )
            raise
    return wrapper
