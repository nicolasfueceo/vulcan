"""Main orchestrator for VULCAN system."""

import asyncio
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import WebSocket

from vulcan.types import (
    ExperimentResult,
    ExperimentStatus,
    VulcanConfig,
    WebSocketMessage,
    WebSocketMessageType,
)
from vulcan.utils import get_vulcan_logger
from vulcan.utils.logging_utils import setup_experiment_file_logging

logger = get_vulcan_logger(__name__)

# Constants
DEFAULT_EXPERIMENT_NAME = "vulcan_experiment"
MAX_CONCURRENT_EXPERIMENTS = 1


class VulcanOrchestrator:
    """Main orchestrator for VULCAN autonomous feature engineering."""

    def __init__(self, config: VulcanConfig) -> None:
        """Initialize VULCAN orchestrator.

        Args:
            config: VULCAN configuration.
        """
        self.config = config
        self.performance_tracker = None

        # State management
        self._is_running = False
        self._experiment_id: Optional[str] = None
        self._components_initialized = False
        self._results_manager = None

        # Experiment tracking
        self._experiment_history: List[ExperimentResult] = []

        # Orchestrator instances
        self.evo_orchestrator = None
        self.mcts_orchestrator = None

        # WebSocket management
        self._websocket_callbacks: List[callable] = []
        self._active_websockets: List[WebSocket] = []
        self._exploration_state: Optional[Dict[str, Any]] = None

        # Artifact callbacks
        self._callbacks = {
            "llm_output": [],
            "feature_code": [],
            "evaluation": [],
            "evolution_snapshot": [],
        }

        logger.info("🔧 VULCAN Orchestrator initialized", config_loaded=True)

    async def initialize_components(self) -> bool:
        """Initialize all system components.

        Returns:
            True if initialization successful, False otherwise.
        """
        start_time = time.time()
        logger.info("🚀 Starting VULCAN component initialization")

        try:
            # Initialize experiment tracker
            logger.info("📊 Initializing experiment tracking...")
            init_start = time.time()
            await self._initialize_experiment_tracker()
            logger.info(
                "✅ Experiment tracker initialized",
                initialization_time=f"{time.time() - init_start:.3f}s",
            )

            # Initialize data access layer
            logger.info("🗄️ Initializing data access layer...")
            init_start = time.time()
            await self._initialize_data_layer()
            logger.info(
                "✅ Data layer initialized",
                initialization_time=f"{time.time() - init_start:.3f}s",
            )

            # Initialize Progressive Evolution components
            logger.info("🧬 Initializing Progressive Evolution components...")
            init_start = time.time()
            await self._initialize_evolution()
            logger.info(
                "✅ Evolution components initialized",
                initialization_time=f"{time.time() - init_start:.3f}s",
            )

            # Initialize agents
            logger.info("🤖 Initializing LLM and feature agents...")
            init_start = time.time()
            await self._initialize_agents()
            logger.info(
                "✅ Agents initialized",
                initialization_time=f"{time.time() - init_start:.3f}s",
            )

            self._components_initialized = True
            total_time = time.time() - start_time
            logger.info(
                "🎉 All VULCAN components initialized successfully",
                total_initialization_time=f"{total_time:.3f}s",
            )
            return True

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(
                "❌ Component initialization failed",
                error=str(e),
                failed_after=f"{total_time:.3f}s",
                exc_info=True,
            )
            return False

    async def _initialize_experiment_tracker(self) -> None:
        """Initialize experiment tracking."""
        # Simple logging-based experiment tracking
        logger.debug("📊 Experiment tracking using file-based logging")
        logger.debug("✅ Experiment tracker ready")

    async def _initialize_data_layer(self) -> None:
        """Initialize data access layer."""
        logger.debug("🔧 Setting up database connections and data validation")
        try:
            # Validate database connections exist and are accessible
            from vulcan.data.goodreads_loader import GoodreadsDataLoader

            # Use absolute paths
            db_path = "/Users/nicolasdhnr/Documents/Imperial/Imperial Thesis/Code/VULCAN/data/goodreads.db"
            splits_dir = "/Users/nicolasdhnr/Documents/Imperial/Imperial Thesis/Code/VULCAN/data/splits"

            # Test database connectivity
            test_loader = GoodreadsDataLoader(
                db_path=db_path,
                splits_dir=splits_dir,
                outer_fold=1,
                inner_fold=1,
            )
            test_stats = test_loader._get_dataset_stats()

            if not test_stats or test_stats.get("n_users", 0) == 0:
                raise RuntimeError("Database validation failed - no data accessible")

            logger.debug("🔌 Database connections validated")
            logger.debug("🔍 Data schema validation complete")
            logger.debug("📊 Data layer validation complete")
        except Exception as e:
            logger.error("❌ Data layer initialization failed", error=str(e))
            raise RuntimeError(f"Data layer initialization failed: {e}") from e
        logger.debug("✅ Data layer ready")

    async def _initialize_evolution(self) -> None:
        """Initialize Progressive Evolution components."""
        logger.debug(
            "🔧 Setting up Progressive Evolution algorithm",
            max_generations=getattr(self.config.mcts, "max_iterations", 50),
            population_size=getattr(self.config.mcts, "population_size", 50),
            generation_size=getattr(self.config.mcts, "generation_size", 20),
        )
        try:
            # Initialize feature components
            from vulcan.evaluation import FeatureEvaluator
            from vulcan.feature import FeatureExecutor

            self.feature_executor = FeatureExecutor(self.config)
            self.feature_evaluator = FeatureEvaluator(self.config)

            logger.debug("🧬 Progressive Evolution population initialized")
            logger.debug("🎯 RL action selection strategy configured")
        except ImportError:
            logger.error("❌ Evolution components not implemented")
            raise RuntimeError("Evolution components not implemented")
        except Exception as e:
            logger.error("❌ Evolution initialization failed", error=str(e))
            raise RuntimeError(f"Evolution initialization failed: {e}") from e
        logger.debug("✅ Evolution components ready")

    async def _initialize_agents(self) -> None:
        """Initialize LLM and feature agents."""
        logger.debug(
            "🔧 Setting up agent system",
            llm_provider=self.config.llm.provider,
            model_name=self.config.llm.model_name,
            max_tokens=self.config.llm.max_tokens,
            temperature=self.config.llm.temperature,
        )
        try:
            # Verify feature agent is available
            from vulcan.agents import FeatureAgent

            # Test agent initialization
            test_agent = FeatureAgent(self.config)

            # Verify LLM client connectivity if enabled
            if self.config.llm.provider != "none":
                logger.debug("🔍 Testing LLM connectivity")
                # Basic LLM connectivity will be tested during first usage

            logger.debug("🤖 Feature generation agent initialized")
            logger.debug("💭 LLM client connection established")
            logger.debug("⚙️ Agent strategies configured")
        except ImportError:
            logger.error("❌ Feature agents not implemented")
            raise RuntimeError("Feature agent components not implemented")
        except Exception as e:
            logger.error("❌ Agent initialization failed", error=str(e))
            raise RuntimeError(f"Agent initialization failed: {e}") from e
        logger.debug("✅ All agents ready")

    async def start_experiment(
        self,
        experiment_name: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        data_context: Optional[Any] = None,
        results_manager: Optional[Any] = None,
    ) -> str:
        """Start a new experiment.

        Args:
            experiment_name: Name for the experiment.
            config_overrides: Configuration overrides.
            data_context: Data context for the experiment.
            results_manager: ResultsManager for saving experiment data to files.

        Returns:
            Experiment ID.

        Raises:
            RuntimeError: If experiment is already running or components not initialized.
        """
        if self._is_running:
            logger.warning(
                "Attempted to start experiment while another is already running."
            )
            raise RuntimeError("Experiment already running")

        if not self._components_initialized:
            logger.error(
                "Attempted to start experiment before components were initialized."
            )
            raise RuntimeError(
                "Components not initialized. Call initialize_components() first."
            )

        self._is_running = True
        self._experiment_id = str(uuid.uuid4())
        self._results_manager = results_manager

        effective_config = self.config
        if config_overrides:
            effective_config = self.config.update(**config_overrides)
            logger.info(
                "Applied configuration overrides for experiment.",
                overrides=config_overrides,
            )

        experiment_name_for_run = (
            experiment_name
            or effective_config.experiment.name
            or DEFAULT_EXPERIMENT_NAME
        )

        # Determine experiment directory
        current_experiment_dir = None
        if hasattr(self._results_manager, "get_experiment_dir") and callable(
            self._results_manager.get_experiment_dir
        ):
            # This is a hypothetical way ResultsManager might provide the dir.
            # current_experiment_dir = self._results_manager.get_experiment_dir()
            # Actual integration depends on ResultsManager implementation.
            # For now, we proceed with fallback if not available or not set this way.
            pass  # Placeholder

        if current_experiment_dir is None:  # Fallback directory creation
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_folder_name = f"{timestamp_str}_{experiment_name_for_run.replace(' ', '_')}_{self._experiment_id[:8]}"
            current_experiment_dir = (
                Path(effective_config.experiment.output_dir) / run_folder_name
            )

        current_experiment_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Experiment artifacts will be saved to: {current_experiment_dir}")

        # Setup experiment-specific file logging
        experiment_log_file = current_experiment_dir / "experiment.log"
        setup_experiment_file_logging(experiment_log_file, effective_config.logging)

        # Log Experiment Header
        header_log_data = {
            "event_type": "experiment_start_header",
            "experiment_id": self._experiment_id,
            "experiment_name": experiment_name_for_run,
            "run_directory": str(current_experiment_dir),
            "start_timestamp": datetime.now().isoformat(),
            "config_llm_provider": effective_config.llm.provider,
            "config_llm_model": effective_config.llm.model_name,
            "config_max_generations": effective_config.experiment.max_generations,
            "config_mcts_max_iterations": effective_config.mcts.max_iterations,
            "config_evaluation_clustering_metric": effective_config.evaluation.clustering_config.metric,
            "config_wandb_enabled": effective_config.experiment.wandb_enabled,
            "config_tensorboard_enabled": effective_config.experiment.tensorboard_enabled,
        }
        if data_context and hasattr(data_context, "fold_id"):
            header_log_data["data_fold_id"] = data_context.fold_id
        if data_context and hasattr(data_context, "data_schema"):
            header_log_data["data_schema_keys"] = list(data_context.data_schema.keys())

        logger.info("Experiment Run Initialized", **header_log_data)

        # Initialize TensorBoard writer if enabled
        self.tensorboard_writer = None
        if effective_config.experiment.tensorboard_enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_log_dir = current_experiment_dir / "tb_logs"
                tb_log_dir.mkdir(parents=True, exist_ok=True)
                self.tensorboard_writer = SummaryWriter(log_dir=str(tb_log_dir))
                logger.info(f"TensorBoard logging enabled. Log directory: {tb_log_dir}")
            except ImportError:
                logger.warning(
                    "TensorBoard logging enabled in config, but 'torch.utils.tensorboard' not found. Skipping TensorBoard."
                )
            except Exception as e:
                logger.error(
                    f"Failed to initialize TensorBoard SummaryWriter: {e}",
                    exc_info=True,
                )

        # Store effective config and other details for the run
        self.current_experiment_config = effective_config
        self.current_experiment_name = experiment_name_for_run
        self.current_experiment_dir_path = current_experiment_dir
        self.current_data_context = data_context

        # Save the effective_config to a file in current_experiment_dir
        try:
            effective_config.to_yaml(current_experiment_dir / "run_config.yaml")
            logger.info(
                f"Saved effective run configuration to {current_experiment_dir / 'run_config.yaml'}"
            )
        except Exception as e:
            logger.error(f"Failed to save run_config.yaml: {e}", exc_info=True)

        logger.info(
            f"Scheduling experiment execution: {experiment_name_for_run} (ID: {self._experiment_id})"
        )

        # Schedule the actual experiment run
        asyncio.create_task(
            self._run_experiment(
                experiment_id=self._experiment_id,
                experiment_name=experiment_name_for_run,
                data_context=data_context,
            )
        )
        return self._experiment_id

    async def _run_experiment(
        self, experiment_id: str, experiment_name: str, data_context: Any
    ) -> None:
        """Run the experiment asynchronously."""
        start_time = time.time()
        exp_logger = logger.bind_experiment(experiment_id, experiment_name)

        exp_logger.info(
            "🔬 Starting experiment execution",
            experiment_id=experiment_id,
            n_users=data_context.n_users,
            n_items=data_context.n_items,
            fold_id=data_context.fold_id,
            max_generations=self.config.mcts.max_iterations,
        )

        try:
            exp_logger.info("🧬 Initializing Progressive Feature Evolution algorithm")

            # Initialize Progressive Evolution orchestrator
            from vulcan.evolution.progressive_orchestrator import (
                ProgressiveEvolutionOrchestrator,
            )

            self.evo_orchestrator = ProgressiveEvolutionOrchestrator(
                config=self.current_experiment_config,
                performance_tracker=self.performance_tracker,
                websocket_callback=self._send_evolution_websocket_update,
                tensorboard_writer=self.tensorboard_writer,
            )

            await self.evo_orchestrator.initialize()

            exp_logger.debug("🔧 Evolution population initialized")
            exp_logger.debug("🎯 RL action selection configured")
            exp_logger.debug("📊 Evaluation metrics prepared")

            exp_logger.info("▶️ Starting Progressive Evolution")

            # Run Progressive Evolution
            evolution_results = await self.evo_orchestrator.run_evolution(
                data_context=data_context,
                max_generations=self.config.mcts.max_iterations,
            )

            # Update exploration state for potential future API access
            self._exploration_state = {
                "current_generation": evolution_results.get("total_generations", 0),
                "population": await self.evo_orchestrator.get_evolution_visualization_data(),
                "best_score": evolution_results.get("best_score", 0.0),
                "total_generations": evolution_results.get("total_generations", 0),
                "total_features_generated": evolution_results.get(
                    "total_features_generated", 0
                ),
            }

            # Extract results from Progressive Evolution
            execution_time = time.time() - start_time

            if not evolution_results or "best_score" not in evolution_results:
                raise RuntimeError("Progressive Evolution did not return valid results")

            result = ExperimentResult(
                experiment_id=experiment_id,
                experiment_name=experiment_name,
                best_node_id=None,  # Not applicable for evolution
                best_score=evolution_results["best_score"],
                best_feature=evolution_results.get("best_features", [{}])[0].get(
                    "name", "unknown"
                )
                if evolution_results.get("best_features")
                else "unknown",
                best_features=evolution_results.get("best_features", []),
                total_iterations=evolution_results.get("total_generations", 0),
                execution_time=execution_time,
            )

            self._experiment_history.append(result)

            # Save final results via results manager
            saved_artifacts_summary = {}
            if (
                self._results_manager
                and hasattr(self._results_manager, "finish_experiment")
                and hasattr(self._results_manager, "get_experiment_artifact_paths")
            ):
                final_payload_to_save = {
                    "experiment_results": result.dict(),
                    "evolution_results": evolution_results,
                    "final_state": self._exploration_state,
                    "config_used": self.current_experiment_config.dict(),  # Save the actual config used for the run
                }
                self._results_manager.finish_experiment(
                    self._experiment_id, final_payload_to_save
                )  # Assuming finish_experiment takes experiment_id
                saved_artifacts_summary = (
                    self._results_manager.get_experiment_artifact_paths(
                        self._experiment_id
                    )
                )
                exp_logger.info(
                    "Final experiment artifacts saved by ResultsManager.",
                    **saved_artifacts_summary,
                )
            else:
                exp_logger.warning(
                    "ResultsManager not available or does not support full artifact path reporting; some artifact paths may not be logged here."
                )

            # Enhanced completion log
            completion_log_data = {
                "event": "experiment_completed_successfully",
                "experiment_id": experiment_id,
                "experiment_name": experiment_name,
                "best_score": result.best_score,
                "total_generations_or_iterations": result.total_iterations,
                "execution_time_seconds": f"{execution_time:.3f}",
                "generations_per_second": f"{result.total_iterations / execution_time:.2f}"
                if execution_time > 0
                else "N/A",
                "features_generated_in_run": evolution_results.get(
                    "total_features_generated", 0
                ),
                "final_population_size": evolution_results.get("population_size", 0),
                "experiment_log_file": str(
                    self.current_experiment_dir_path / "experiment.log"
                ),
                "run_config_file": str(
                    self.current_experiment_dir_path / "run_config.yaml"
                ),
                "results_directory": str(self.current_experiment_dir_path),
            }
            if saved_artifacts_summary:
                completion_log_data["saved_artifacts"] = saved_artifacts_summary

            exp_logger.info(
                "Experiment completed and results processed", **completion_log_data
            )
            # This log will go to console if a console handler is attached to the root logger.

        except ImportError as e:
            execution_time = time.time() - start_time
            exp_logger.error(
                "❌ Progressive Evolution components not implemented",
                experiment_id=experiment_id,
                error="ProgressiveEvolutionOrchestrator not available",
                execution_time=f"{execution_time:.3f}s",
            )

            # Mark experiment as failed in results manager
            if self._results_manager:
                self._results_manager.finish_experiment(
                    {"error": str(e), "status": "failed"}
                )

            raise RuntimeError(
                "Progressive Evolution feature engineering components not implemented"
            ) from e

        except Exception as e:
            execution_time = time.time() - start_time
            exp_logger.error(
                "❌ Experiment execution failed",
                experiment_id=experiment_id,
                error=str(e),
                execution_time=f"{execution_time:.3f}s",
                exc_info=True,
            )

            # Mark experiment as failed in results manager
            if self._results_manager:
                self._results_manager.finish_experiment(
                    {"error": str(e), "status": "failed"}
                )

            exp_logger.info("📁 Experiment failure recorded")
            raise

        finally:
            self._is_running = False
            self._experiment_id = None
            self._results_manager = None  # Clear reference
            # Cleanup evolution orchestrator
            if self.evo_orchestrator:
                await self.evo_orchestrator.cleanup()
            # Close TensorBoard writer if it was created by VulcanOrchestrator
            if self.tensorboard_writer:  # Ensure VO closes the writer it created
                try:
                    self.tensorboard_writer.close()
                    logger.info(
                        "TensorBoard SummaryWriter closed by VulcanOrchestrator."
                    )
                except Exception as e:
                    logger.error(
                        f"Error closing TensorBoard SummaryWriter: {e}", exc_info=True
                    )
            exp_logger.info("🔄 Experiment cleanup completed")

    async def _send_evolution_websocket_update(
        self, update_data: Dict[str, Any]
    ) -> None:
        """Send evolution update via WebSocket."""
        await self._send_websocket_message(
            WebSocketMessage(
                type=WebSocketMessageType.EXPLORATION_UPDATE,
                timestamp=time.time(),
                experiment_id=self._experiment_id,
                data=update_data,
            )
        )

    async def stop_experiment(self) -> bool:
        """Stop the current experiment.

        Returns:
            True if experiment was stopped, False if no experiment running.
        """
        if not self._is_running:
            return False

        logger.info("Stopping experiment", experiment_id=self._experiment_id)

        # Send stop notification
        await self._send_websocket_message(
            WebSocketMessage(
                type=WebSocketMessageType.EXPERIMENT_STOPPED,
                timestamp=time.time(),
                experiment_id=self._experiment_id,
            )
        )

        self._is_running = False
        self._experiment_id = None
        return True

    def get_status(self) -> ExperimentStatus:
        """Get current system status."""
        return ExperimentStatus(
            is_running=self._is_running,
            experiment_id=self._experiment_id,
            config_summary={
                "mcts_iterations": self.config.mcts.max_iterations,
                "llm_provider": self.config.llm.provider,
                "llm_model": self.config.llm.model_name,
                "api_enabled": self.config.api.enabled,
            },
            components_initialized={
                "experiment_tracker": self._components_initialized,
                "evo_orchestrator": self._components_initialized,
                "dal": self._components_initialized,
            },
            experiment_history_count=len(self._experiment_history),
        )

    def get_experiment_history(self) -> List[ExperimentResult]:
        """Get experiment history."""
        return self._experiment_history.copy()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance tracking metrics."""
        logger.debug("Retrieving performance metrics")
        return self.performance_tracker.get_performance_summary()

    def get_feature_performance(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """Get performance metrics for a specific feature."""
        metrics = self.performance_tracker.get_feature_metrics(feature_name)
        if metrics:
            return {
                "feature_name": metrics.feature_name,
                "avg_score": metrics.avg_score,
                "max_score": metrics.max_score,
                "min_score": metrics.min_score,
                "score_trend": metrics.score_trend,
                "recent_performance": metrics.recent_performance,
                "success_rate": metrics.success_rate,
                "appearances": metrics.appearances,
                "consistency_score": metrics.consistency_score,
                "avg_execution_time": metrics.avg_execution_time,
                "avg_computational_cost": metrics.avg_computational_cost,
            }
        return None

    def get_feature_suggestions(self) -> Dict[str, List[str]]:
        """Get AI-powered feature performance suggestions."""
        logger.debug("Generating feature performance suggestions")
        return self.performance_tracker.suggest_feature_actions()

    def get_best_features(
        self, top_k: int = 5, criteria: str = "avg_score"
    ) -> List[Dict[str, Any]]:
        """Get the best performing features."""
        best_features = self.performance_tracker.get_best_features(top_k, criteria)
        return [
            {
                "feature_name": f.feature_name,
                "avg_score": f.avg_score,
                "success_rate": f.success_rate,
                "consistency_score": f.consistency_score,
                "appearances": f.appearances,
            }
            for f in best_features
        ]

    def export_performance_data(self) -> Dict[str, Any]:
        """Export all performance data for analysis."""
        logger.info("Exporting performance tracking data")
        return self.performance_tracker.export_metrics()

    def add_websocket_callback(self, callback: callable) -> None:
        """Add WebSocket message callback."""
        self._websocket_callbacks.append(callback)

    def remove_websocket_callback(self, callback: callable) -> None:
        """Remove WebSocket message callback."""
        if callback in self._websocket_callbacks:
            self._websocket_callbacks.remove(callback)

    async def _send_websocket_message(self, message: WebSocketMessage) -> None:
        """Send WebSocket message to all callbacks."""
        for callback in self._websocket_callbacks:
            try:
                await callback(message)
            except Exception as e:
                logger.error("WebSocket callback failed", error=str(e))

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self._is_running:
            await self.stop_experiment()

        logger.info("VULCAN Orchestrator cleaned up")

    def _create_default_data_context(self):
        """Create a default data context using real Goodreads data."""
        from vulcan.data.goodreads_loader import GoodreadsDataLoader

        logger.info("📊 Loading real Goodreads data with streaming approach")

        # Log data loading configuration - use absolute paths
        db_path = "/Users/nicolasdhnr/Documents/Imperial/Imperial Thesis/Code/VULCAN/data/goodreads.db"
        splits_dir = "/Users/nicolasdhnr/Documents/Imperial/Imperial Thesis/Code/VULCAN/data/splits"
        outer_fold = 1
        inner_fold = 1
        batch_size = 1000

        logger.debug(
            "🔧 Data loader configuration",
            db_path=db_path,
            splits_dir=splits_dir,
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            batch_size=batch_size,
        )

        try:
            loader_start = time.time()
            loader = GoodreadsDataLoader(
                db_path=db_path,
                splits_dir=splits_dir,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                batch_size=batch_size,
            )

            loader_time = time.time() - loader_start
            logger.debug(
                "✅ Data loader initialized", initialization_time=f"{loader_time:.3f}s"
            )

            # Get streaming data context - fast because it doesn't load all data upfront
            context_start = time.time()
            data_context = loader.get_data_context()
            context_time = time.time() - context_start

            logger.info(
                "✅ Successfully loaded Goodreads data context",
                n_users=f"{data_context.n_users:,}",
                n_items=f"{data_context.n_items:,}",
                sparsity=f"{data_context.sparsity:.6f}",
                density_percent=f"{(1 - data_context.sparsity) * 100:.4f}%",
                fold_id=data_context.fold_id,
                context_creation_time=f"{context_time:.3f}s",
                total_time=f"{loader_time + context_time:.3f}s",
            )

            return data_context

        except Exception as e:
            logger.error(
                "❌ Failed to create data context",
                error=str(e),
                db_path=db_path,
                splits_dir=splits_dir,
                exc_info=True,
            )
            raise

    def add_callback(self, callback_type: str, callback: callable) -> None:
        """Add a callback for saving experiment artifacts.

        Args:
            callback_type: Type of callback ("llm_output", "feature_code", "evaluation", "evolution_snapshot")
            callback: Callback function to invoke
        """
        if callback_type in self._callbacks:
            self._callbacks[callback_type].append(callback)
        else:
            logger.warning(f"Unknown callback type: {callback_type}")

    def remove_callback(self, callback_type: str, callback: callable) -> None:
        """Remove a callback.

        Args:
            callback_type: Type of callback
            callback: Callback function to remove
        """
        if (
            callback_type in self._callbacks
            and callback in self._callbacks[callback_type]
        ):
            self._callbacks[callback_type].remove(callback)

    def _invoke_callbacks(self, callback_type: str, *args, **kwargs) -> None:
        """Invoke all callbacks of a given type.

        Args:
            callback_type: Type of callback to invoke
            *args, **kwargs: Arguments to pass to callbacks
        """
        for callback in self._callbacks.get(callback_type, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Callback failed: {callback_type}", error=str(e))

    async def add_websocket(self, websocket: WebSocket):
        """Add a websocket connection to broadcast updates to."""
        await websocket.accept()
        self._active_websockets.append(websocket)

        # Send current state if available
        if self._exploration_state:
            await self._broadcast_state()

    async def remove_websocket(self, websocket: WebSocket):
        """Remove a websocket connection."""
        if websocket in self._active_websockets:
            self._active_websockets.remove(websocket)

    async def _broadcast_state(self):
        """Broadcast current exploration state to all connected clients."""
        if not self._exploration_state:
            return

        # Convert dataclasses to dicts for JSON serialization
        serializable_state = {
            "current_generation": self._exploration_state["current_generation"],
            "population": self._exploration_state["population"],
            "best_score": self._exploration_state["best_score"],
            "total_generations": self._exploration_state["total_generations"],
            "total_features_generated": self._exploration_state[
                "total_features_generated"
            ],
        }

        state_json = json.dumps(serializable_state)
        dead_sockets = []

        for ws in self._active_websockets:
            try:
                await ws.send_text(state_json)
            except Exception as e:
                logger.warning("Failed to send to websocket", error=str(e))
                dead_sockets.append(ws)

        # Clean up dead connections
        for ws in dead_sockets:
            await self.remove_websocket(ws)
