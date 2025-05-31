"""Experiment tracking utilities for VULCAN system."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from vulcan.types import ExperimentConfig, ExperimentResult, FeatureEvaluation

logger = structlog.get_logger(__name__)

# Constants
WANDB_PROJECT_PREFIX = "vulcan"
ARTIFACT_TYPES = {
    "feature_code": "code",
    "reflection": "text",
    "experiment_state": "model",
    "learning_curve": "dataset",
}


class ExperimentTracker:
    """Tracks experiments with Weights & Biases and local storage."""

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize experiment tracker.

        Args:
            config: Experiment tracking configuration.
        """
        self.config = config
        self._wandb_run = None
        self._experiment_id: Optional[str] = None
        self._metrics_history: List[Dict[str, Any]] = []

        if config.wandb_enabled:
            self._initialize_wandb()

    def _initialize_wandb(self) -> None:
        """Initialize Weights & Biases."""
        try:
            import wandb

            # Check if already initialized
            if wandb.run is not None:
                logger.warning("W&B run already active, finishing previous run")
                wandb.finish()

            logger.info("Initializing Weights & Biases tracking")

        except ImportError:
            logger.warning("Weights & Biases not installed, disabling W&B tracking")
            self.config.wandb_enabled = False

    def start_experiment(
        self,
        experiment_id: str,
        experiment_name: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Start tracking a new experiment.

        Args:
            experiment_id: Unique experiment identifier.
            experiment_name: Human-readable experiment name.
            config_dict: Experiment configuration dictionary.
        """
        self._experiment_id = experiment_id
        self._metrics_history.clear()

        if self.config.wandb_enabled:
            self._start_wandb_run(experiment_id, experiment_name, config_dict)

        logger.info(
            "Started experiment tracking",
            experiment_id=experiment_id,
            experiment_name=experiment_name,
        )

    def _start_wandb_run(
        self,
        experiment_id: str,
        experiment_name: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Start Weights & Biases run."""
        try:
            import wandb

            # Prepare run configuration
            run_name = experiment_name or f"experiment_{experiment_id[:8]}"
            tags = self.config.tags.copy()
            tags.append(f"experiment_id:{experiment_id}")

            # Start run
            self._wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=run_name,
                tags=tags,
                config=config_dict or {},
                reinit=True,
            )

            logger.info("W&B run started", run_id=self._wandb_run.id)

        except Exception as e:
            logger.error("Failed to start W&B run", error=str(e))
            self.config.wandb_enabled = False

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics for the current experiment.

        Args:
            metrics: Dictionary of metric name-value pairs.
            step: Optional step number.
        """
        if not self._experiment_id:
            logger.warning("No active experiment, skipping metrics logging")
            return

        # Add timestamp
        metrics_with_timestamp = {
            **metrics,
            "timestamp": time.time(),
        }

        if step is not None:
            metrics_with_timestamp["step"] = step

        # Store locally
        self._metrics_history.append(metrics_with_timestamp)

        # Log to W&B
        if self.config.wandb_enabled and self._wandb_run:
            try:
                self._wandb_run.log(metrics, step=step)
            except Exception as e:
                logger.error("Failed to log metrics to W&B", error=str(e))

        logger.debug("Logged metrics", metrics=metrics, step=step)

    def log_feature_evaluation(self, evaluation: FeatureEvaluation) -> None:
        """Log feature evaluation results.

        Args:
            evaluation: Feature evaluation result.
        """
        metrics = {
            "feature_score": evaluation.score,
            "evaluation_time": evaluation.evaluation_time,
            "iteration": evaluation.iteration,
        }

        # Add detailed metrics
        if evaluation.metrics.silhouette_score is not None:
            metrics["silhouette_score"] = evaluation.metrics.silhouette_score

        if evaluation.metrics.calinski_harabasz is not None:
            metrics["calinski_harabasz"] = evaluation.metrics.calinski_harabasz

        if evaluation.metrics.davies_bouldin is not None:
            metrics["davies_bouldin"] = evaluation.metrics.davies_bouldin

        self.log_metrics(metrics, step=evaluation.iteration)

    def log_mcts_iteration(
        self,
        iteration: int,
        best_score: float,
        nodes_explored: int,
        current_depth: int,
    ) -> None:
        """Log MCTS iteration progress.

        Args:
            iteration: Current iteration number.
            best_score: Best score achieved so far.
            nodes_explored: Number of nodes explored.
            current_depth: Current tree depth.
        """
        metrics = {
            "mcts/best_score": best_score,
            "mcts/nodes_explored": nodes_explored,
            "mcts/current_depth": current_depth,
            "mcts/iteration": iteration,
        }

        self.log_metrics(metrics, step=iteration)

    def save_artifact(
        self,
        artifact_path: str,
        artifact_name: str,
        artifact_type: str = "file",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save artifact to experiment tracking.

        Args:
            artifact_path: Path to the artifact file.
            artifact_name: Name for the artifact.
            artifact_type: Type of artifact.
            metadata: Optional metadata dictionary.
        """
        if not self._experiment_id:
            logger.warning("No active experiment, skipping artifact save")
            return

        if self.config.wandb_enabled and self._wandb_run:
            try:
                import wandb

                artifact = wandb.Artifact(
                    name=artifact_name,
                    type=artifact_type,
                    metadata=metadata or {},
                )

                artifact.add_file(artifact_path)
                self._wandb_run.log_artifact(artifact)

                logger.info(
                    "Artifact saved to W&B",
                    artifact_name=artifact_name,
                    artifact_type=artifact_type,
                )

            except Exception as e:
                logger.error("Failed to save artifact to W&B", error=str(e))

    def save_feature_code(self, feature_name: str, code: str) -> None:
        """Save feature code as artifact.

        Args:
            feature_name: Name of the feature.
            code: Feature code.
        """
        if not self._experiment_id:
            return

        # Save to local file
        artifacts_dir = Path("artifacts") / self._experiment_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        code_file = artifacts_dir / f"{feature_name}.py"
        with open(code_file, "w", encoding="utf-8") as f:
            f.write(code)

        # Save as artifact
        self.save_artifact(
            artifact_path=str(code_file),
            artifact_name=f"feature_code_{feature_name}",
            artifact_type=ARTIFACT_TYPES["feature_code"],
            metadata={"feature_name": feature_name},
        )

    def save_reflection(self, reflection_id: str, content: str) -> None:
        """Save LLM reflection as artifact.

        Args:
            reflection_id: Unique reflection identifier.
            content: Reflection content.
        """
        if not self._experiment_id:
            return

        # Save to local file
        artifacts_dir = Path("artifacts") / self._experiment_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        reflection_file = artifacts_dir / f"reflection_{reflection_id}.txt"
        with open(reflection_file, "w", encoding="utf-8") as f:
            f.write(content)

        # Save as artifact
        self.save_artifact(
            artifact_path=str(reflection_file),
            artifact_name=f"reflection_{reflection_id}",
            artifact_type=ARTIFACT_TYPES["reflection"],
            metadata={"reflection_id": reflection_id},
        )

    def finish_experiment(self, result: ExperimentResult) -> None:
        """Finish experiment tracking.

        Args:
            result: Final experiment result.
        """
        if not self._experiment_id:
            return

        # Log final metrics
        final_metrics = {
            "final/best_score": result.best_score,
            "final/total_iterations": result.total_iterations,
            "final/execution_time": result.execution_time,
        }

        self.log_metrics(final_metrics)

        # Finish W&B run
        if self.config.wandb_enabled and self._wandb_run:
            try:
                self._wandb_run.finish()
                logger.info("W&B run finished")
            except Exception as e:
                logger.error("Failed to finish W&B run", error=str(e))

        logger.info(
            "Experiment tracking finished",
            experiment_id=self._experiment_id,
            best_score=result.best_score,
        )

        self._experiment_id = None
        self._wandb_run = None

    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get metrics history for current experiment.

        Returns:
            List of metrics dictionaries.
        """
        return self._metrics_history.copy()
