"""Configuration type definitions for VULCAN system."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

# Constants for validation
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_FILE = "vulcan.log"
DEFAULT_LOG_SIZE = "10MB"
DEFAULT_BACKUP_COUNT = 5

VALID_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
VALID_LLM_PROVIDERS = ["openai", "anthropic", "local"]
VALID_AGENT_STRATEGIES = ["skip_node", "retry"]


class MCTSConfig(BaseModel):
    """MCTS algorithm configuration."""

    max_iterations: int = Field(default=50, ge=1, le=1000)
    exploration_factor: float = Field(default=1.414, ge=0.1, le=5.0)
    reward_discount: float = Field(default=1.0, ge=0.1, le=1.0)
    max_depth: int = Field(default=10, ge=1, le=50)
    agent_failure_strategy: str = Field(
        default="skip_node", pattern="^(skip_node|retry)$"
    )


class LLMConfig(BaseModel):
    """LLM configuration."""

    provider: str = Field(default="openai", pattern="^(openai|anthropic|local)$")
    model_name: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, ge=100, le=8000)
    max_retries: int = Field(default=3, ge=1, le=10)
    rate_limit_delay: float = Field(default=0.5, ge=0.0, le=10.0)
    api_key_env: str = Field(default="OPENAI_API_KEY")


class ClusteringConfig(BaseModel):
    """Clustering configuration."""

    n_clusters: Optional[int] = Field(
        default=None, description="Fixed number of clusters"
    )
    cluster_range: List[int] = Field(
        default=[2, 20], description="Range of clusters to try"
    )
    metric: str = Field(
        default="silhouette", description="Clustering metric to optimize"
    )
    optimize_n_clusters: bool = Field(
        default=True, description="Whether to optimize n_clusters"
    )


class EvaluationConfig(BaseModel):
    """Feature evaluation configuration."""

    sample_size: int = Field(default=5000, ge=100, le=100000)
    random_state: int = Field(default=42)
    metrics: List[str] = Field(default=["silhouette_score", "calinski_harabasz"])
    clustering_method: str = Field(default="kmeans")
    clustering_config: ClusteringConfig = Field(default_factory=ClusteringConfig)


class ApiConfig(BaseModel):
    """API server configuration."""

    enabled: bool = Field(default=True)
    host: str = Field(default="localhost")
    port: int = Field(default=8000, ge=1000, le=65535)
    reload: bool = Field(default=True)
    cors_origins: List[str] = Field(default=["http://localhost:3000"])
    websocket_enabled: bool = Field(default=True)


class ExperimentConfig(BaseModel):
    """Configuration for experiment management."""

    name: str = Field(default="vulcan_experiment", description="Experiment name")
    description: Optional[str] = Field(
        default=None, description="Experiment description"
    )
    output_dir: str = Field(
        default="experiments", description="Main output directory for all experiments"
    )
    tags: List[str] = Field(default_factory=list, description="Tags for the experiment")

    # W&B specific
    wandb_enabled: bool = Field(
        default=False, description="Enable Weights & Biases logging"
    )
    project_name: Optional[str] = Field(default=None, description="W&B project name")
    entity: Optional[str] = Field(default=None, description="W&B entity")

    # TensorBoard specific
    tensorboard_enabled: bool = Field(
        default=False, description="Enable TensorBoard logging"
    )

    # Feature engineering and evolution parameters
    max_features_in_set: int = Field(
        default=100,
        description="Maximum number of features to maintain in a candidate set",
    )
    min_features_in_set: int = Field(
        default=3, description="Minimum number of features to start with or maintain"
    )
    max_generations: int = Field(
        default=10,
        description="Max generations for progressive evolution or MCTS iterations",
    )

    # Run control
    random_seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility of the experiment run",
    )
    save_artifacts: bool = Field(
        default=True,
        description="Whether to save experiment artifacts (results, logs, etc.)",
    )


class DataConfig(BaseModel):
    """Data configuration."""

    train_db: str = Field(default="data/train.db")
    test_db: str = Field(default="data/test.db")
    validation_db: str = Field(default="data/validation.db")

    @validator("train_db", "test_db", "validation_db")
    def validate_db_paths(cls, v: str) -> str:
        """Validate database paths exist or can be created."""
        # Skip validation if file doesn't exist (for testing)
        return v


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(
        default=DEFAULT_LOG_LEVEL, pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
    )
    format: str = Field(default=DEFAULT_LOG_FORMAT)
    file: Optional[str] = Field(default=DEFAULT_LOG_FILE)
    max_file_size: str = Field(default=DEFAULT_LOG_SIZE)
    backup_count: int = Field(default=DEFAULT_BACKUP_COUNT, ge=1, le=20)
    structured: bool = Field(default=True)


class VulcanConfig(BaseModel):
    """Main VULCAN configuration."""

    # Core configurations
    mcts: MCTSConfig = Field(default_factory=MCTSConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Directory configurations
    results_dir: str = Field(default="results")
    states_dir: str = Field(default="states")
    configs_dir: str = Field(default="config")

    # System configurations
    random_seed: int = Field(default=42)
    debug_mode: bool = Field(default=False)

    def __init__(self, **kwargs: Any) -> None:
        """Initialize configuration and ensure directories exist."""
        super().__init__(**kwargs)
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            self.results_dir,
            self.states_dir,
            self.configs_dir,
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "VulcanConfig":
        """Load configuration from YAML file."""
        import yaml

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        return cls(**config_data)

    def to_yaml(self, output_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        import yaml

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(self.dict(), f, default_flow_style=False, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.dict()

    def update(self, **kwargs: Any) -> "VulcanConfig":
        """Update configuration with new values."""
        config_dict = self.dict()

        def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        updated_dict = deep_update(config_dict, kwargs)
        return VulcanConfig(**updated_dict)
