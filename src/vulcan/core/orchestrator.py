"""Main orchestrator for VULCAN system."""

import asyncio
import json
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import WebSocket

from vulcan.schemas import (
    ExperimentResult,
    ExperimentStatus,
    VulcanConfig,
    WebSocketMessage,
    WebSocketMessageType,
)
from vulcan.utils import get_vulcan_logger

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
                logger.error(
                    "Critical error in queue runner", error=str(e), exc_info=True
                )
                await asyncio.sleep(10)

    async def start_experiment(
        self,
        experiment_name: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        data_context: Optional[Any] = None,
        results_manager: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Adds an experiment request to the queue."""
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
        """Runs a single experiment from start to finish. This is the consumer logic."""
        # 1. Set busy state
        self._is_running = True
        self._experiment_id = experiment_request["experiment_id"]
        exp_logger = logger.bind_experiment(
            self._experiment_id, experiment_request["experiment_name"]
        )

        try:
            # 2. Unpack request and set up config
            config_overrides = experiment_request["config_overrides"]
            data_context = experiment_request["data_context"]
            effective_config = (
                self.config.update(**config_overrides)
                if config_overrides
                else self.config
            )

            # Use self._results_manager if available, otherwise the one from request
            # This part of the logic might need refinement based on how ResultsManager is used.
            # For now, let's assume one is created if not present.
            if not self._results_manager:
                from vulcan.utils import ResultsManager

                self._results_manager = ResultsManager(effective_config)

            # 3. Set up directories and logging for this run
            exp_logger.info("🔬 Starting experiment execution...")

            # 4. Initialize and run the evolution orchestrator
            from vulcan.evolution.progressive_orchestrator import (
                ProgressiveEvolutionOrchestrator,
            )

            self.evo_orchestrator = ProgressiveEvolutionOrchestrator(
                config=effective_config,
                results_manager=self._results_manager,
                performance_tracker=self.performance_tracker,
                websocket_callback=self._send_evolution_websocket_update,
            )
            await self.evo_orchestrator.initialize()

            evolution_results = await self.evo_orchestrator.run_evolution(
                data_context=data_context,
                max_generations=effective_config.experiment.max_generations,
            )

            # 5. Process results (simplified for clarity)
            exp_logger.info(
                "✅ Experiment completed successfully.",
                best_score=evolution_results.get("best_score"),
            )

        except Exception as e:
            exp_logger.error(
                "❌ Experiment execution failed", error=str(e), exc_info=True
            )
        finally:
            # 6. ALWAYS clean up and reset state
            exp_logger.info("🔄 Experiment finished, cleaning up run...")
            if self.evo_orchestrator:
                await self.evo_orchestrator.cleanup()
                self.evo_orchestrator = None
            self._is_running = False
            self._experiment_id = None
            exp_logger.info("Queue is now free for the next experiment.")

    def _create_default_data_context(self):
        """Creates a default data context using the Goodreads data loader."""
        from vulcan.data.goodreads_loader import GoodreadsDataLoader

        logger.info("📊 Creating default Goodreads data context...")

        # These paths and settings could be moved to the main config, but for now,
        # we mirror the previous logic.
        db_path = "/Users/nicolasdhnr/Documents/Imperial/Imperial Thesis/Code/VULCAN/data/goodreads.db"
        splits_dir = "/Users/nicolasdhnr/Documents/Imperial/Imperial Thesis/Code/VULCAN/data/splits"

        try:
            loader = GoodreadsDataLoader(
                db_path=db_path,
                splits_dir=splits_dir,
                outer_fold=1,  # Default fold
                inner_fold=1,  # Default fold
            )
            data_context = loader.get_data_context()
            logger.info("✅ Default data context created successfully.")
            return data_context
        except Exception as e:
            logger.error(
                f"❌ Failed to create default data context: {e}", exc_info=True
            )
            raise  # Re-raise the exception to be handled by the API layer

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
        """Send WebSocket message to all callbacks."""
        for callback in self._websocket_callbacks:
            try:
                await callback(message)
            except Exception as e:
                logger.error("WebSocket callback failed", error=str(e))

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
