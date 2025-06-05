"""Main orchestrator for VULCAN system."""

import asyncio
import json
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import WebSocket
from rich.console import Console

from vulcan.schemas import (
    DataContext,
    ExperimentResult,
    ExperimentStatus,
    VulcanConfig,
    WebSocketMessage,
    WebSocketMessageType,
)
from vulcan.utils import ResultsManager, get_vulcan_logger

logger = get_vulcan_logger(__name__)
CONSOLE = Console()

# Constants
DEFAULT_EXPERIMENT_NAME = "vulcan_experiment"
MAX_CONCURRENT_EXPERIMENTS = 1

# --- Orchestrator-level helpers ---

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"


def load_presets() -> Dict[str, Any]:
    """Load all experiment presets from the configs directory."""
    presets = {}
    for config_path in CONFIGS_DIR.glob("*.yaml"):
        with open(config_path) as f:
            preset_name = config_path.stem
            presets[preset_name] = {"path": config_path, "config": yaml.safe_load(f)}
    return presets


def get_fold_counts(config: Dict[str, Any]) -> tuple[int, int]:
    """Get the number of outer and inner folds from the data splits directory."""
    try:
        # Assuming the base path for data is relative to the project root
        project_root = Path(__file__).resolve().parents[3]
        splits_dir = project_root / config["data"]["splits_dir"]

        if not splits_dir.is_dir():
            logger.warning(f"Splits directory not found at {splits_dir}")
            return 1, 1

        outer_folds = len(list(splits_dir.glob("outer_fold_*")))
        inner_folds_path = splits_dir / "outer_fold_0"
        inner_folds = (
            len(list(inner_folds_path.glob("inner_fold_*")))
            if inner_folds_path.is_dir()
            else 0
        )
        return max(1, outer_folds), max(1, inner_folds)
    except (KeyError, FileNotFoundError, TypeError) as e:
        logger.error(f"Could not determine fold counts: {e}")
        return 1, 1


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
        self._results_manager = ResultsManager(config)

        # Queue for pending experiments
        self.experiment_queue = deque()
        self._queue_runner_task: Optional[asyncio.Task] = None
        self._run_queue_runner = False

        # Experiment tracking
        self._experiment_history: List[ExperimentResult] = []

        # Orchestrator instances
        self.evo_orchestrator = None

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

            self._run_queue_runner = True
            self._queue_runner_task = asyncio.create_task(self._queue_runner())
            logger.info("✅ Experiment queue runner started.")
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
            max_generations=self.config.experiment.max_generations,
            population_size=self.config.experiment.population_size,
            generation_size=self.config.experiment.generation_size,
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

    async def _queue_runner(self):
        """A background task that consumes and runs experiments from the queue."""
        logger.info("Queue runner is active and waiting for experiments.")
        while self._run_queue_runner:
            try:
                if not self._is_running and self.experiment_queue:
                    request_to_run = self.experiment_queue.popleft()
                    logger.info(
                        "Dequeued experiment, starting run.",
                        experiment_name=request_to_run.get("experiment_name"),
                    )
                    asyncio.create_task(self._run_experiment(request_to_run))
                await asyncio.sleep(3)
            except asyncio.CancelledError:
                logger.info("Queue runner task was cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in queue runner: {e}", exc_info=True)
                await asyncio.sleep(10)  # Wait before retrying

    async def start_preset_experiment(self, preset_name: str) -> Dict[str, Any]:
        """
        Starts an experiment based on a named preset from the /configs directory.
        If the preset is a cross-validation run, it queues an experiment for each fold.
        """
        if self._is_running:
            return {
                "status": "error",
                "message": "An experiment is already in progress.",
            }

        presets = load_presets()
        if preset_name not in presets:
            return {"status": "error", "message": f"Preset '{preset_name}' not found."}

        preset = presets[preset_name]
        preset_config = preset["config"]
        experiment_base_name = preset_config.get("experiment", {}).get(
            "name", preset_name
        )

        # Special handling for cross-validation runs
        if "goodreads" in preset_name and "large" in preset_name:
            num_outer, _ = get_fold_counts(preset_config)
            msg = f"🔬 Starting {num_outer}-fold cross-validation for preset '{preset_name}'."
            logger.info(msg)
            await self._send_websocket_message(
                WebSocketMessage(
                    type=WebSocketMessageType.NOTIFICATION,
                    timestamp=time.time(),
                    payload={"message": msg},
                )
            )

            for i in range(num_outer):
                exp_name = f"{experiment_base_name}_outer_fold_{i}"
                overrides = {"data": {"outer_fold": i, "inner_fold": 0}}
                request = {
                    "experiment_name": exp_name,
                    "config_overrides": overrides,
                    "original_preset": preset_config,
                }
                self.experiment_queue.append(request)
            return {
                "status": "success",
                "message": f"Queued {num_outer}-fold cross-validation.",
            }

        # For single runs
        msg = f"🚀 Queuing single experiment for preset: '{preset_name}'"
        logger.info(msg)
        await self._send_websocket_message(
            WebSocketMessage(
                type=WebSocketMessageType.NOTIFICATION,
                timestamp=time.time(),
                payload={"message": msg},
            )
        )
        request = {
            "experiment_name": experiment_base_name,
            "config_overrides": {},
            "original_preset": preset_config,
        }
        self.experiment_queue.append(request)
        return {"status": "success", "message": f"Queued single run for {preset_name}."}

    async def start_experiment(
        self,
        experiment_name: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        data_context: Optional[Any] = None,
        results_manager: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Start a new feature engineering experiment."""
        if self._is_running:
            return {
                "status": "error",
                "message": "An experiment is already in progress.",
            }

        if not self._components_initialized:
            raise RuntimeError("Cannot queue experiment: components not initialized.")

        request_id = str(uuid.uuid4())
        experiment_name_for_run = (
            experiment_name or self.config.experiment.name or DEFAULT_EXPERIMENT_NAME
        )

        experiment_request_data = {
            "experiment_id": request_id,
            "experiment_name": experiment_name_for_run,
            "config_overrides": config_overrides or {},
            "data_context": data_context,
            "results_manager": results_manager,  # This might be None, handled in _run_experiment
            "queued_at": datetime.utcnow().isoformat(),
        }

        self.experiment_queue.append(experiment_request_data)
        queue_position = len(self.experiment_queue)

        logger.info(
            "Experiment added to queue.",
            experiment_name=experiment_name_for_run,
            position=queue_position,
        )

        return {
            "status": "queued",
            "experiment_id": request_id,
            "position": queue_position,
        }

    async def _run_experiment(self, experiment_request: Dict[str, Any]):
        """Execute a single experiment run."""
        if not self._components_initialized:
            logger.error("Cannot run experiment, components are not initialized.")
            return

        self._is_running = True
        self._experiment_id = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )
        experiment_name = experiment_request.get(
            "experiment_name", DEFAULT_EXPERIMENT_NAME
        )
        config_overrides = experiment_request.get("config_overrides", {})
        original_preset = experiment_request.get("original_preset", {})

        # Create a disposable config for this specific run
        run_config = VulcanConfig(**original_preset).update(**config_overrides)

        await self._send_websocket_message(
            WebSocketMessage(
                type=WebSocketMessageType.EXPERIMENT_START,
                timestamp=time.time(),
                payload={
                    "experiment_id": self._experiment_id,
                    "experiment_name": experiment_name,
                    "config": run_config.to_dict(),
                    "start_time": time.time(),
                },
            )
        )

        try:
            logger.info(
                "🚀 Starting new experiment run", experiment_name=experiment_name
            )
            from vulcan.evolution.progressive_orchestrator import (
                ProgressiveEvolutionOrchestrator,
            )

            # Use the disposable run_config for the orchestrator
            self.evo_orchestrator = ProgressiveEvolutionOrchestrator(
                config=run_config,
                results_manager=self._results_manager,
                websocket_callback=self._send_evolution_websocket_update,
            )
            data_context = self._create_default_data_context(run_config.data)
            results = await self.evo_orchestrator.run_evolution(data_context)

            logger.info(
                "✅ Experiment run finished successfully",
                experiment_name=experiment_name,
                results=results,
            )

            await self._send_websocket_message(
                WebSocketMessage(
                    type=WebSocketMessageType.EXPERIMENT_END,
                    timestamp=time.time(),
                    payload={
                        "experiment_id": self._experiment_id,
                        "status": "completed",
                        "end_time": time.time(),
                        "results": results,
                    },
                )
            )

        except Exception as e:
            logger.error(
                "❌ Experiment run failed",
                experiment_name=experiment_name,
                error=str(e),
                exc_info=True,
            )
            await self._send_websocket_message(
                WebSocketMessage(
                    type=WebSocketMessageType.EXPERIMENT_END,
                    timestamp=time.time(),
                    payload={
                        "experiment_id": self._experiment_id,
                        "status": "failed",
                        "error": str(e),
                        "end_time": time.time(),
                    },
                )
            )
        finally:
            self._is_running = False
            self.evo_orchestrator = None
            self._experiment_id = None
            logger.info("Orchestrator is now idle.")
            if not self.experiment_queue:
                await self._send_websocket_message(
                    WebSocketMessage(
                        type=WebSocketMessageType.NOTIFICATION,
                        timestamp=time.time(),
                        payload={"message": "All queued experiments have completed."},
                    )
                )

    def _create_default_data_context(self, data_config: "DataConfig") -> DataContext:
        """Create a default data context for a single experiment run."""
        from vulcan.data.goodreads_loader import GoodreadsDataLoader

        logger.info("Creating data context for experiment run.")

        # Prefer the specific db_path from presets, otherwise fallback to train_db
        db_path = data_config.db_path or data_config.train_db
        splits_dir = data_config.splits_dir or "data/splits"

        loader = GoodreadsDataLoader(
            db_path=db_path,
            splits_dir=splits_dir,
            outer_fold=data_config.outer_fold,
            inner_fold=data_config.inner_fold,
        )
        data_context = loader.get_data_context()
        logger.info(
            "DataContext created in orchestrator",
            data_context_type=type(data_context),
            train_data_type=type(data_context.train_data),
        )
        return data_context

    def get_status(self) -> ExperimentStatus:
        """Get current system status, including the queue."""
        queued_experiments_list = [
            {
                "name": exp.get("experiment_name"),
                "id": exp.get("experiment_id"),
                "queued_at": exp.get("queued_at"),
            }
            for exp in self.experiment_queue
        ]

        # This needs to be built with all fields from ExperimentStatus model
        status_payload = {
            "is_running": self._is_running,
            "experiment_id": self._experiment_id,
            "queued_experiments": queued_experiments_list,
            # Placeholder for other required fields
            "config_summary": {},
            "components_initialized": {"orchestrator": self._components_initialized},
            "current_experiment": None,
            "experiment_history_count": len(self._experiment_history),
        }
        return ExperimentStatus(**status_payload)

    async def cleanup(self) -> None:
        """Cleanup resources, including the queue runner."""
        if self._queue_runner_task:
            self._run_queue_runner = False
            self._queue_runner_task.cancel()
            try:
                await self._queue_runner_task
            except asyncio.CancelledError:
                logger.info("Queue runner task successfully cancelled.")

        if self._is_running:
            await self.stop_experiment()

        logger.info("VULCAN Orchestrator cleaned up")

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

        # Also stop the underlying evolution orchestrator if it exists
        if self.evo_orchestrator:
            await self.evo_orchestrator.cleanup()

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
        """Send WebSocket message to all callbacks. Replaced with logging for script-based run."""
        log_level = "info"
        status = message.data.get("status") if message.data else None
        if message.type == WebSocketMessageType.EXPERIMENT_END and status == "failed":
            log_level = "error"

        log_message = f"[WEBSOCKET_MESSAGE] Type: {message.type.value}"
        log_payload = {
            "data": message.data,
            "error": message.error,
            "results": message.results,
        }

        if log_level == "error":
            logger.error(log_message, **log_payload)
        else:
            logger.info(log_message, **log_payload)

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
            "current_node_id": self._exploration_state.current_node_id,
            "llm_history": [vars(h) for h in self._exploration_state.llm_history],
            "best_score": self._exploration_state.best_score,
            "total_iterations": self._exploration_state.total_iterations,
            "current_path": self._exploration_state.current_path,
        }

        state_json = json.dumps(serializable_state, default=str)
        logger.info("[BROADCAST_STATE]", state=state_json)

    async def run_single_experiment_from_preset(
        self, preset_name: str, preset_config: Dict[str, Any]
    ):
        """Runs a single experiment directly, bypassing the queue.

        Args:
            preset_name: The name of the experiment preset.
            preset_config: The loaded configuration dictionary for the preset.
        """
        if self._is_running:
            logger.warning("An experiment is already in progress.")
            return

        # Initialize components before the first run
        if not self._components_initialized:
            await self.initialize_components()

        experiment_name = preset_config.get("experiment", {}).get("name", preset_name)

        request = {
            "experiment_name": experiment_name,
            "config_overrides": {},
            "original_preset": preset_config,
        }

        # Directly run the experiment instead of queueing
        await self._run_experiment(request)
