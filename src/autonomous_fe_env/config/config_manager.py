"""
Configuration manager for VULCAN autonomous feature engineering system.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages configuration settings for the VULCAN system.

    Supports loading from YAML and JSON files, environment variables,
    and provides default configurations.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to configuration file (YAML or JSON)
        """
        self.config: Dict[str, Any] = {}
        self.config_path = config_path

        # Load default configuration
        self._load_defaults()

        # Load from file if provided
        if config_path:
            self.load_from_file(config_path)

        # Override with environment variables
        self._load_from_env()

    def _load_defaults(self) -> None:
        """Load default configuration values."""
        self.config = {
            "data_source": {
                "type": "sql",
                "db_path": "data/goodreads.db",
                "reviews_table": "reviews",
                "books_table": "books",
                "users_table": "users",
                "splits": {
                    "directory": "data/splits",
                    "files": {
                        "train": "train.csv",
                        "test": "test.csv",
                        "validation": "validation.csv",
                    },
                    "id_column": "user_id",
                    "entity_type": "user",
                },
            },
            "mcts": {
                "max_iterations": 50,
                "exploration_factor": 1.414,
                "reward_discount": 1.0,
                "max_depth": 10,
                "agent_failure_strategy": "skip_node",
            },
            "evaluation": {
                "n_clusters": 5,
                "random_state": 42,
                "sample_size": 1000,
                "metric": "silhouette",
            },
            "agents": {
                "feature_agent": {"type": "feature", "mode": "predefined"},
                "reflection_agent": {
                    "type": "reflection",
                    "reflection_types": [
                        "feature_proposal",
                        "feature_evaluation",
                        "strategy",
                    ],
                },
            },
            "reflection": {
                "memory_dir": "memory",
                "max_entries": 100,
                "templates_dir": "prompts/reflection",
            },
            "sandbox": {
                "timeout": 30,
                "max_memory": 104857600,  # 100MB
            },
            "llm": {"model": "gpt-4", "temperature": 0.7, "max_tokens": 2000},
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
            "state_dir": "state",
            "results_dir": "results",
            "reflection_interval": 5,
        }

    def load_from_file(self, config_path: str) -> bool:
        """
        Load configuration from a file.

        Args:
            config_path: Path to configuration file

        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(config_path):
            logger.warning(f"Configuration file not found: {config_path}")
            return False

        try:
            with open(config_path, "r") as f:
                if config_path.endswith(".yaml") or config_path.endswith(".yml"):
                    file_config = yaml.safe_load(f)
                elif config_path.endswith(".json"):
                    file_config = json.load(f)
                else:
                    logger.error(
                        f"Unsupported configuration file format: {config_path}"
                    )
                    return False

            # Merge with existing configuration
            self._deep_merge(self.config, file_config)
            logger.info(f"Configuration loaded from {config_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            return False

    def _load_from_env(self) -> None:
        """Load configuration overrides from environment variables."""
        # Database configuration
        if os.getenv("VULCAN_DB_PATH"):
            self.config["data_source"]["db_path"] = os.getenv("VULCAN_DB_PATH")

        # MCTS configuration
        if os.getenv("VULCAN_MAX_ITERATIONS"):
            try:
                self.config["mcts"]["max_iterations"] = int(
                    os.getenv("VULCAN_MAX_ITERATIONS")
                )
            except ValueError:
                logger.warning("Invalid VULCAN_MAX_ITERATIONS value")

        # LLM configuration
        if os.getenv("OPENAI_API_KEY"):
            self.config["llm"]["api_key"] = os.getenv("OPENAI_API_KEY")

        if os.getenv("VULCAN_LLM_MODEL"):
            self.config["llm"]["model"] = os.getenv("VULCAN_LLM_MODEL")

        # Logging level
        if os.getenv("VULCAN_LOG_LEVEL"):
            self.config["logging"]["level"] = os.getenv("VULCAN_LOG_LEVEL")

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """
        Deep merge two dictionaries.

        Args:
            base: Base dictionary to merge into
            override: Dictionary with override values
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: Configuration key (supports dot notation, e.g., 'mcts.max_iterations')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split(".")
        config = self.config

        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set the value
        config[keys[-1]] = value

    def save_to_file(self, config_path: str) -> bool:
        """
        Save current configuration to a file.

        Args:
            config_path: Path to save configuration to

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(config_path), exist_ok=True)

            with open(config_path, "w") as f:
                if config_path.endswith(".yaml") or config_path.endswith(".yml"):
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
                elif config_path.endswith(".json"):
                    json.dump(self.config, f, indent=2)
                else:
                    logger.error(
                        f"Unsupported configuration file format: {config_path}"
                    )
                    return False

            logger.info(f"Configuration saved to {config_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving configuration to {config_path}: {e}")
            return False

    def get_config(self) -> Dict[str, Any]:
        """
        Get the complete configuration dictionary.

        Returns:
            Complete configuration dictionary
        """
        return self.config.copy()

    def validate(self) -> bool:
        """
        Validate the current configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        errors = []

        # Validate required sections
        required_sections = ["data_source", "mcts", "evaluation"]
        for section in required_sections:
            if section not in self.config:
                errors.append(f"Missing required configuration section: {section}")

        # Validate data source configuration
        if "data_source" in self.config:
            data_config = self.config["data_source"]
            if "db_path" not in data_config:
                errors.append("Missing data_source.db_path")

        # Validate MCTS configuration
        if "mcts" in self.config:
            mcts_config = self.config["mcts"]
            if "max_iterations" not in mcts_config:
                errors.append("Missing mcts.max_iterations")
            elif (
                not isinstance(mcts_config["max_iterations"], int)
                or mcts_config["max_iterations"] <= 0
            ):
                errors.append("mcts.max_iterations must be a positive integer")

        # Validate evaluation configuration
        if "evaluation" in self.config:
            eval_config = self.config["evaluation"]
            if "n_clusters" not in eval_config:
                errors.append("Missing evaluation.n_clusters")
            elif (
                not isinstance(eval_config["n_clusters"], int)
                or eval_config["n_clusters"] <= 0
            ):
                errors.append("evaluation.n_clusters must be a positive integer")

        if errors:
            for error in errors:
                logger.error(f"Configuration validation error: {error}")
            return False

        logger.info("Configuration validation passed")
        return True

    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"ConfigManager(config_path={self.config_path})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"ConfigManager(config_path={self.config_path}, sections={list(self.config.keys())})"
