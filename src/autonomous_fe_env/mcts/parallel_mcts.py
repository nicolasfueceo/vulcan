"""
Parallel MCTS implementation for feature engineering.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from .mcts_node import MCTSNode
from .mcts_orchestrator import MCTSOrchestrator

logger = logging.getLogger(__name__)


class ParallelMCTS:
    """
    Parallel implementation of Monte Carlo Tree Search for feature engineering.

    Supports running multiple MCTS processes concurrently to explore
    different parts of the search space simultaneously.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the parallel MCTS system.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.parallel_config = config.get("parallel_mcts", {})

        # Parallel execution parameters
        self.num_workers = self.parallel_config.get("num_workers", 4)
        self.worker_iterations = self.parallel_config.get("worker_iterations", 10)
        self.synchronization_interval = self.parallel_config.get(
            "synchronization_interval", 5
        )
        self.use_processes = self.parallel_config.get("use_processes", False)

        # Shared state
        self.global_best_node: Optional[MCTSNode] = None
        self.global_best_score: float = -float("inf")
        self.worker_results: List[Dict[str, Any]] = []

        logger.info(f"Initialized ParallelMCTS with {self.num_workers} workers")

    async def run_parallel_search(
        self, orchestrators: List[MCTSOrchestrator]
    ) -> MCTSNode:
        """
        Run parallel MCTS search using multiple orchestrators.

        Args:
            orchestrators: List of MCTS orchestrators to run in parallel

        Returns:
            Best node found across all parallel searches
        """
        if len(orchestrators) != self.num_workers:
            raise ValueError(
                f"Number of orchestrators ({len(orchestrators)}) must match num_workers ({self.num_workers})"
            )

        logger.info(f"Starting parallel MCTS search with {self.num_workers} workers")
        start_time = time.time()

        if self.use_processes:
            # Use process-based parallelism
            results = await self._run_with_processes(orchestrators)
        else:
            # Use thread-based parallelism
            results = await self._run_with_threads(orchestrators)

        # Aggregate results
        best_node = self._aggregate_results(results)

        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"Parallel MCTS search completed in {duration:.2f} seconds")
        logger.info(f"Best node found: {best_node}")

        return best_node

    async def _run_with_threads(
        self, orchestrators: List[MCTSOrchestrator]
    ) -> List[MCTSNode]:
        """
        Run MCTS searches using thread-based parallelism.

        Args:
            orchestrators: List of MCTS orchestrators

        Returns:
            List of best nodes from each worker
        """
        logger.info("Running parallel MCTS with threads")

        # Create tasks for each orchestrator
        tasks = []
        for i, orchestrator in enumerate(orchestrators):
            task = asyncio.create_task(
                self._run_worker_async(orchestrator, worker_id=i)
            )
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and extract successful results
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Worker {i} failed with exception: {result}")
            else:
                successful_results.append(result)

        return successful_results

    async def _run_with_processes(
        self, orchestrators: List[MCTSOrchestrator]
    ) -> List[MCTSNode]:
        """
        Run MCTS searches using process-based parallelism.

        Args:
            orchestrators: List of MCTS orchestrators

        Returns:
            List of best nodes from each worker
        """
        logger.info("Running parallel MCTS with processes")

        # Note: Process-based parallelism is more complex due to serialization requirements
        # For now, we'll use a simplified approach with ThreadPoolExecutor
        # In a full implementation, this would use ProcessPoolExecutor with proper serialization

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit tasks to the executor
            futures = []
            for i, orchestrator in enumerate(orchestrators):
                future = executor.submit(self._run_worker_sync, orchestrator, i)
                futures.append(future)

            # Collect results
            results = []
            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Worker {i} failed with exception: {e}")

            return results

    async def _run_worker_async(
        self, orchestrator: MCTSOrchestrator, worker_id: int
    ) -> MCTSNode:
        """
        Run a single MCTS worker asynchronously.

        Args:
            orchestrator: MCTS orchestrator for this worker
            worker_id: Unique identifier for this worker

        Returns:
            Best node found by this worker
        """
        logger.info(f"Starting worker {worker_id}")

        try:
            # Run the orchestrator with limited iterations
            original_max_iterations = orchestrator.max_iterations
            orchestrator.max_iterations = self.worker_iterations

            # Run the search
            best_node = orchestrator.run()

            # Restore original max iterations
            orchestrator.max_iterations = original_max_iterations

            logger.info(
                f"Worker {worker_id} completed with best score: {best_node.score:.4f}"
            )
            return best_node

        except Exception as e:
            logger.error(f"Worker {worker_id} failed: {e}")
            raise

    def _run_worker_sync(
        self, orchestrator: MCTSOrchestrator, worker_id: int
    ) -> MCTSNode:
        """
        Run a single MCTS worker synchronously.

        Args:
            orchestrator: MCTS orchestrator for this worker
            worker_id: Unique identifier for this worker

        Returns:
            Best node found by this worker
        """
        logger.info(f"Starting worker {worker_id}")

        try:
            # Run the orchestrator with limited iterations
            original_max_iterations = orchestrator.max_iterations
            orchestrator.max_iterations = self.worker_iterations

            # Run the search
            best_node = orchestrator.run()

            # Restore original max iterations
            orchestrator.max_iterations = original_max_iterations

            logger.info(
                f"Worker {worker_id} completed with best score: {best_node.score:.4f}"
            )
            return best_node

        except Exception as e:
            logger.error(f"Worker {worker_id} failed: {e}")
            raise

    def _aggregate_results(self, results: List[MCTSNode]) -> MCTSNode:
        """
        Aggregate results from parallel workers.

        Args:
            results: List of best nodes from each worker

        Returns:
            Overall best node
        """
        if not results:
            raise ValueError("No results to aggregate")

        # Find the best node across all workers
        best_node = max(results, key=lambda node: node.score)

        logger.info(f"Aggregated {len(results)} results")
        logger.info(f"Best score across all workers: {best_node.score:.4f}")

        # Store aggregation statistics
        scores = [node.score for node in results]
        self.worker_results = [
            {
                "worker_id": i,
                "score": node.score,
                "features": [f.name for f in node.state_features],
                "visits": node.visits,
                "value": node.value,
            }
            for i, node in enumerate(results)
        ]

        logger.info(
            f"Score statistics - Min: {min(scores):.4f}, Max: {max(scores):.4f}, Avg: {sum(scores) / len(scores):.4f}"
        )

        return best_node

    def get_worker_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about worker performance.

        Returns:
            Dictionary with worker statistics
        """
        if not self.worker_results:
            return {"error": "No worker results available"}

        scores = [result["score"] for result in self.worker_results]

        return {
            "num_workers": len(self.worker_results),
            "best_score": max(scores),
            "worst_score": min(scores),
            "average_score": sum(scores) / len(scores),
            "score_std": (
                sum((s - sum(scores) / len(scores)) ** 2 for s in scores) / len(scores)
            )
            ** 0.5,
            "worker_results": self.worker_results,
        }

    def create_worker_orchestrators(
        self, base_orchestrator: MCTSOrchestrator, num_workers: Optional[int] = None
    ) -> List[MCTSOrchestrator]:
        """
        Create multiple orchestrator instances for parallel execution.

        Args:
            base_orchestrator: Base orchestrator to clone
            num_workers: Number of workers to create (defaults to self.num_workers)

        Returns:
            List of orchestrator instances
        """
        if num_workers is None:
            num_workers = self.num_workers

        orchestrators = []

        for i in range(num_workers):
            # Create a new orchestrator with the same configuration
            # but potentially different random seeds for diversity
            worker_config = self.config.copy()

            # Modify random seed for diversity
            if "random_seed" in worker_config:
                worker_config["random_seed"] = worker_config["random_seed"] + i
            else:
                worker_config["random_seed"] = 42 + i

            # Create new orchestrator
            orchestrator = MCTSOrchestrator(worker_config)

            # Set up with the same components as the base orchestrator
            if base_orchestrator.dal:
                orchestrator.setup(
                    dal=base_orchestrator.dal,
                    agent=base_orchestrator.agent,
                    evaluator=base_orchestrator.evaluator,
                    reflection_engine=base_orchestrator.reflection_engine,
                )

            orchestrators.append(orchestrator)

        logger.info(f"Created {len(orchestrators)} worker orchestrators")
        return orchestrators
