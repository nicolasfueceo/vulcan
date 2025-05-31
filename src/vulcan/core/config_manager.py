"""Configuration management for VULCAN system."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import structlog

from vulcan.types import VulcanConfig

logger = structlog.get_logger(__name__)

# Constants
DEFAULT_CONFIG_FILE = "config/vulcan.yaml"
ENV_CONFIG_PATH = "VULCAN_CONFIG_PATH"


class ConfigManager:
    """Manages VULCAN configuration with validation and environment support."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize configuration manager.

        Args:
            config_path: Path to configuration file. If None, uses default or env var.
        """
        self._config_path = self._resolve_config_path(config_path)
        self._config: Optional[VulcanConfig] = None
        self._load_config()

    def _resolve_config_path(self, config_path: Optional[Union[str, Path]]) -> Path:
        """Resolve configuration file path from various sources."""
        if config_path:
            return Path(config_path)

        # Check environment variable
        env_path = os.getenv(ENV_CONFIG_PATH)
        if env_path:
            return Path(env_path)

        # Use default
        return Path(DEFAULT_CONFIG_FILE)

    def _load_config(self) -> None:
        """Load configuration from file."""
        try:
            if self._config_path.exists():
                self._config = VulcanConfig.from_yaml(self._config_path)
                logger.info(
                    "Configuration loaded successfully",
                    config_path=str(self._config_path),
                )
            else:
                logger.warning(
                    "Configuration file not found, using defaults",
                    config_path=str(self._config_path),
                )
                self._config = VulcanConfig()
                self._save_default_config()

        except Exception as e:
            logger.error(
                "Failed to load configuration, using defaults",
                error=str(e),
                config_path=str(self._config_path),
            )
            self._config = VulcanConfig()

    def _save_default_config(self) -> None:
        """Save default configuration to file."""
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            self._config.to_yaml(self._config_path)
            logger.info(
                "Default configuration saved",
                config_path=str(self._config_path),
            )
        except Exception as e:
            logger.error(
                "Failed to save default configuration",
                error=str(e),
                config_path=str(self._config_path),
            )

    @property
    def config(self) -> VulcanConfig:
        """Get current configuration."""
        if self._config is None:
            raise RuntimeError("Configuration not loaded")
        return self._config

    def reload(self) -> None:
        """Reload configuration from file."""
        logger.info("Reloading configuration")
        self._load_config()

    def update_config(self, **kwargs: Any) -> None:
        """Update configuration with new values."""
        if self._config is None:
            raise RuntimeError("Configuration not loaded")

        self._config = self._config.update(**kwargs)
        logger.info("Configuration updated", updates=kwargs)

    def save_config(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """Save current configuration to file."""
        if self._config is None:
            raise RuntimeError("Configuration not loaded")

        save_path = Path(output_path) if output_path else self._config_path
        self._config.to_yaml(save_path)
        logger.info("Configuration saved", config_path=str(save_path))

    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        if self._config is None:
            raise RuntimeError("Configuration not loaded")
        return self._config.to_dict()

    def validate_config(self) -> bool:
        """Validate current configuration."""
        try:
            if self._config is None:
                return False

            # Perform additional validation checks
            self._validate_paths()
            self._validate_api_config()
            self._validate_llm_config()

            logger.info("Configuration validation passed")
            return True

        except Exception as e:
            logger.error("Configuration validation failed", error=str(e))
            return False

    def _validate_paths(self) -> None:
        """Validate file paths in configuration."""
        # Validate database paths exist or can be created
        data_config = self.config.data
        for db_path in [
            data_config.train_db,
            data_config.test_db,
            data_config.validation_db,
        ]:
            db_file = Path(db_path)
            if not db_file.exists():
                logger.warning("Database file not found", path=db_path)

    def _validate_api_config(self) -> None:
        """Validate API configuration."""
        api_config = self.config.api
        if api_config.enabled:
            if not (1000 <= api_config.port <= 65535):
                raise ValueError(f"Invalid API port: {api_config.port}")

    def _validate_llm_config(self) -> None:
        """Validate LLM configuration."""
        llm_config = self.config.llm
        if llm_config.provider == "openai":
            api_key = os.getenv(llm_config.api_key_env)
            if not api_key:
                logger.warning(
                    "OpenAI API key not found in environment",
                    env_var=llm_config.api_key_env,
                )
