# tests/unit/test_core/test_config_manager.py
import tempfile
from pathlib import Path

import pytest
import yaml  # Added for creating temporary config files

from vulcan.core.config_manager import ConfigManager
from vulcan.types import (
    VulcanConfig,
)

# Assuming WORKSPACE_ROOT is defined in conftest.py or can be derived
# For now, let's derive it here for self-containment of this snippet,
# but ideally it's a fixture.
TEST_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def sample_config_path() -> Path:
    return TEST_WORKSPACE_ROOT / "tests" / "test_data" / "sample_config.yaml"


@pytest.fixture
def default_config_path() -> Path:
    # Corrected path to default config after directory restructuring
    return TEST_WORKSPACE_ROOT / "configs" / "config.yaml"


@pytest.fixture
def temp_config_file() -> Path:
    """Create a temporary YAML config file for testing updates."""
    initial_data = {
        "experiment": {
            "name": "initial_name",
            "output_dir": "outputs/initial",
            "max_generations": 10,
        },
        "llm": {"provider": "openai", "model_name": "gpt-3", "temperature": 0.7},
        "data": {"sample_size": 100},
        "mcts": {},
        "evaluation": {},
        "api": {},
        "logging": {},
    }
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as tmpfile:
        yaml.dump(initial_data, tmpfile)
        tmp_path = Path(tmpfile.name)
    yield tmp_path
    tmp_path.unlink()  # Clean up


def test_load_sample_config(sample_config_path: Path):
    """Test loading a valid YAML configuration file."""
    assert sample_config_path.exists(), "Sample config file should exist for test."

    manager = ConfigManager(config_path=str(sample_config_path))
    config = manager.config

    assert isinstance(config, VulcanConfig)
    assert config.experiment.name == "test_experiment"
    assert config.llm.model_name == "gpt-test-model"
    assert config.data.sample_size == 100
    assert config.logging.level == "DEBUG"


def test_load_default_config(default_config_path: Path):
    """Test loading the default YAML configuration file."""
    # Ensure the default config file actually exists where expected
    # If not, this test might need to create a dummy one or be skipped.
    if not default_config_path.exists():
        pytest.skip(
            f"Default config file not found at {default_config_path}, skipping test."
        )

    manager = ConfigManager(config_path=str(default_config_path))
    config = manager.config
    assert isinstance(config, VulcanConfig)
    assert config.experiment is not None
    assert config.llm is not None


def test_config_manager_update_config_method(temp_config_file: Path):
    """Test ConfigManager.update_config() method."""
    manager = ConfigManager(config_path=str(temp_config_file))
    initial_name = manager.config.experiment.name
    initial_model = manager.config.llm.model_name

    assert initial_name == "initial_name"
    assert initial_model == "gpt-3"

    updates = {
        "experiment": {"name": "updated_by_manager"},
        "llm": {"model_name": "gpt-4-manager", "temperature": 0.88},
        "data": {"sample_size": 5000},
    }
    manager.update_config(**updates)
    updated_config = manager.config

    assert updated_config.experiment.name == "updated_by_manager"
    assert updated_config.llm.model_name == "gpt-4-manager"
    assert updated_config.llm.temperature == 0.88
    assert updated_config.data.sample_size == 5000


def test_vulcan_config_update_method():
    """Test VulcanConfig.update() method directly."""
    initial_dict = {
        "experiment": {
            "name": "initial_name",
            "output_dir": "outputs/initial",
            "max_generations": 5,
        },
        "llm": {"provider": "openai", "model_name": "gpt-3", "temperature": 0.7},
        "data": {"sample_size": 1000},
        "mcts": {},
        "evaluation": {},
        "api": {},
        "logging": {},
    }
    config = VulcanConfig(**initial_dict)

    updates = {
        "experiment": {"name": "updated_name_direct"},
        "llm": {"model_name": "gpt-4-direct", "temperature": 0.99},
        "data": {"sample_size": 2000},
        "new_top_level_key": "new_value",  # Test adding a new top-level key
    }

    new_config_after_update = config.update(**updates)

    assert new_config_after_update.experiment.name == "updated_name_direct"
    assert new_config_after_update.llm.model_name == "gpt-4-direct"
    assert new_config_after_update.llm.temperature == 0.99
    assert new_config_after_update.data.sample_size == 2000
    assert config.experiment.name == "initial_name"  # Original should be unchanged
    assert config.llm.model_name == "gpt-3"

    # Check behavior for extra fields. Pydantic V1 default is to ignore them.
    # If VulcanConfig is set to forbid extra fields, this update itself would raise a ValidationError.
    # If VulcanConfig is set to allow them, hasattr would be True.
    assert not hasattr(new_config_after_update, "new_top_level_key"), \
        "Extra top-level keys should be ignored by default in Pydantic V1 unless explicitly allowed."


# More tests can be added for:
# - ConfigManager handling of missing configuration files (it creates a default one)
# - ConfigManager saving configuration (save_config method)
# - ConfigManager reloading configuration (reload method)
# - VulcanConfig validation for specific field constraints (Pydantic handles most, but custom validators too)
# - Environment variable overrides in ConfigManager (if VULCAN_CONFIG_PATH is used)

# tests/unit/test_core/test_config_manager.py
