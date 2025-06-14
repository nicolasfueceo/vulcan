# src/utils/logging_utils.py
import logging
from typing import Any, Dict

from loguru import logger




class InterceptHandler(logging.Handler):
    """
    A handler to intercept standard logging messages and redirect them to loguru.
    """

    def emit(self, record):
        # Get corresponding Loguru level if it exists.
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where the logged message originated.
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )





def log_agent_context(context: Dict[str, Any]) -> None:
    """Log the context passed to an agent."""
    logger.info(f"Context received: {context}")


def log_agent_response(response: Dict[str, Any]) -> None:
    """Log the response from an agent."""
    logger.info(f"Response generated: {response}")


def log_agent_error(error: Exception) -> None:
    """Log an error that occurred in an agent."""
    logger.error(f"Error occurred: {str(error)}")


def log_llm_prompt(prompt: str) -> None:
    """Log the prompt sent to the LLM."""
    logger.info(f"📤 LLM PROMPT:\n{'-' * 50}\n{prompt}\n{'-' * 50}")


def log_llm_response(response: str) -> None:
    """Log the response from the LLM."""
    logger.info(f"📥 LLM RESPONSE:\n{'-' * 50}\n{response}\n{'-' * 50}")


def log_tool_call(tool_name: str, tool_args: Dict[str, Any]) -> None:
    """Log a tool call being made."""
    logger.info(f"🔧 TOOL CALL: {tool_name} with args: {tool_args}")


def log_tool_result(tool_name: str, result: Any) -> None:
    """Log the result of a tool call."""
    logger.info(f"🔧 TOOL RESULT from {tool_name}: {result}")



