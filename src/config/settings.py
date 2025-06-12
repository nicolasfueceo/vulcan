"""
Configuration settings for the VULCAN project.
Contains database paths, LLM configurations, and other global constants.
"""

from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
PROMPTS_DIR = SRC_DIR / "prompts"
LOGS_DIR = ROOT_DIR / "logs"
DATA_DIR = ROOT_DIR / "data"
RUN_DIR = ROOT_DIR / "runtime" / "runs"

# Database configuration
DB_PATH = str(DATA_DIR / "goodreads_curated.duckdb")

# LLM Configuration - Default configuration that can be used across agents
# This will be overridden by the orchestrator with actual API keys and config lists
LLM_CONFIG = {
    "config_list": [],  # Will be populated by orchestrator from OAI_CONFIG_LIST.json
    "cache_seed": None,
    "temperature": 0.7,
    "timeout": 120,
}

# Agent configuration
MAX_CONSECUTIVE_AUTO_REPLY = 10
CODE_EXECUTION_TIMEOUT = 120

# Plotting configuration
PLOT_DPI = 300
PLOT_STYLE = "default"
PLOT_PALETTE = "husl"

# OpenAI configuration
OPENAI_MODEL_VISION = "gpt-4o"
OPENAI_MODEL_TEXT = "gpt-4o-mini"
OPENAI_MAX_TOKENS = 1000

# Database connection settings
DB_READ_ONLY = False  # Allow writes for temporary views
DB_TIMEOUT = 30

# Insight Discovery settings
INSIGHT_AGENTS_CONFIG_PATH = ROOT_DIR / "config" / "OAI_CONFIG_LIST.json"
INSIGHT_MAX_TURNS = 20
INSIGHT_MAX_CONSECUTIVE_AUTO_REPLY = 5

# Add other settings as needed
