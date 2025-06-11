# src/utils/logging_utils.py
import logging
import sys
from pathlib import Path
from typing import Any, Dict

from loguru import logger

# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Agent colors mapping
AGENT_COLORS = {
    "DataAnalysisAgent": "cyan",
    "HypothesisAgent": "green",
    "ReasoningAgent": "yellow",
    "FeatureIdeationAgent": "magenta",
    "FeatureRealizationAgent": "blue",
    "OptimizationAgent": "red",
    "EvaluationAgent": "white",
    "ReflectionAgent": "bright_blue",
    "ResearchAgent": "bright_green",
}


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


def setup_agent_logger(agent_name: str) -> None:
    """Set up logging for a specific agent with its own color and file."""
    # Remove default handler
    logger.remove()

    # Add console handler with agent-specific color
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        f"<{AGENT_COLORS.get(agent_name, 'white')}>{agent_name}</{AGENT_COLORS.get(agent_name, 'white')}> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True,
    )

    # Add file handler for this agent
    logger.add(
        LOGS_DIR / f"{agent_name.lower()}.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="1 day",
        retention="7 days",
    )


def log_agent_context(agent_name: str, context: Dict[str, Any]) -> None:
    """Log the context passed to an agent."""
    logger.info(f"Context received: {context}")


def log_agent_response(agent_name: str, response: Dict[str, Any]) -> None:
    """Log the response from an agent."""
    logger.info(f"Response generated: {response}")


def log_agent_error(agent_name: str, error: Exception) -> None:
    """Log an error that occurred in an agent."""
    logger.error(f"Error occurred: {str(error)}")


def log_llm_prompt(agent_name: str, prompt: str) -> None:
    """Log the prompt sent to the LLM."""
    logger.info(f"📤 LLM PROMPT:\n{'-' * 50}\n{prompt}\n{'-' * 50}")


def log_llm_response(agent_name: str, response: str) -> None:
    """Log the response from the LLM."""
    logger.info(f"📥 LLM RESPONSE:\n{'-' * 50}\n{response}\n{'-' * 50}")


def log_tool_call(agent_name: str, tool_name: str, tool_args: Dict[str, Any]) -> None:
    """Log a tool call being made."""
    logger.info(f"🔧 TOOL CALL: {tool_name} with args: {tool_args}")


def log_tool_result(agent_name: str, tool_name: str, result: Any) -> None:
    """Log the result of a tool call."""
    logger.info(f"🔧 TOOL RESULT from {tool_name}: {result}")


# Example usage in an agent:
"""
from src.utils.logging_utils import setup_agent_logger, log_agent_context, log_agent_response

class MyAgent:
    def __init__(self):
        setup_agent_logger(self.__class__.__name__)
    
    def run(self, context):
        log_agent_context(self.__class__.__name__, context)
        # ... agent logic ...
        log_agent_response(self.__class__.__name__, response)
"""
