"""Results manager for saving experiment data to files."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from vulcan.types import VulcanConfig


class ResultsManager:
    """Manages saving and loading experiment results to/from files."""

    def __init__(self, config: VulcanConfig, base_dir: Optional[str] = None):
        """Initialize results manager.

        Args:
            config: VULCAN configuration
            base_dir: Base directory for results (defaults to ./results)
        """
        self.config = config
        self.logger = structlog.get_logger(__name__)

        # Setup results directory
        if base_dir:
            self.results_dir = Path(base_dir)
        else:
            self.results_dir = Path("results")

        self.results_dir.mkdir(exist_ok=True)

        # Current experiment tracking
        self.current_experiment_id: Optional[str] = None
        self.current_experiment_dir: Optional[Path] = None
        self.current_metadata: Dict[str, Any] = {}

    def start_experiment(self, experiment_id: str, metadata: Dict[str, Any]) -> str:
        """Start a new experiment and create its directory.

        Args:
            experiment_id: Unique experiment identifier
            metadata: Experiment metadata (algorithm, config, etc.)

        Returns:
            Path to experiment directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{timestamp}_{experiment_id}"

        self.current_experiment_id = experiment_id
        self.current_experiment_dir = self.results_dir / experiment_name
        self.current_experiment_dir.mkdir(exist_ok=True)

        # Save metadata
        self.current_metadata = {
            **metadata,
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "status": "running",
        }

        self._save_metadata()

        # Create initial empty data files
        self._save_experiment_data(
            {
                "nodes": [],
                "edges": [],
                "best_node_id": None,
                "stats": {
                    "total_nodes": 0,
                    "max_depth": 0,
                    "best_score": 0.0,
                    "iterations_completed": 0,
                    "avg_branching_factor": 0.0,
                },
                "generation_history": [],
                "action_rewards": {"generate_new": [], "mutate_existing": []},
                "best_candidate": None,
                "decision_logs": [],
                "llm_interactions": [],
            }
        )

        self.logger.info(
            "Started experiment",
            experiment_id=experiment_id,
            experiment_dir=str(self.current_experiment_dir),
        )

        return str(self.current_experiment_dir)

    def update_experiment_data(self, data: Dict[str, Any]) -> None:
        """Update experiment data and save to file.

        Args:
            data: Updated experiment data
        """
        if not self.current_experiment_dir:
            self.logger.warning("No active experiment to update")
            return

        self._save_experiment_data(data)

        # Update metadata with latest stats
        if "stats" in data:
            self.current_metadata.update(
                {
                    "last_update": datetime.now().isoformat(),
                    "iterations_completed": data["stats"].get(
                        "iterations_completed", 0
                    ),
                    "best_score": data["stats"].get("best_score", 0.0),
                }
            )
            self._save_metadata()

    def add_decision_log(self, decision_log: Dict[str, Any]) -> None:
        """Add a decision log entry.

        Args:
            decision_log: Decision log entry
        """
        if not self.current_experiment_dir:
            return

        # Load current data, add decision log, save back
        current_data = self._load_experiment_data()
        if "decision_logs" not in current_data:
            current_data["decision_logs"] = []
        current_data["decision_logs"].append(decision_log)
        self._save_experiment_data(current_data)

    def add_llm_interaction(self, llm_interaction: Dict[str, Any]) -> None:
        """Add an LLM interaction entry.

        Args:
            llm_interaction: LLM interaction entry
        """
        if not self.current_experiment_dir:
            return

        # Load current data, add LLM interaction, save back
        current_data = self._load_experiment_data()
        if "llm_interactions" not in current_data:
            current_data["llm_interactions"] = []
        current_data["llm_interactions"].append(llm_interaction)
        self._save_experiment_data(current_data)

    def finish_experiment(self, final_results: Optional[Dict[str, Any]] = None) -> None:
        """Mark experiment as finished.

        Args:
            final_results: Final experiment results
        """
        if not self.current_experiment_dir:
            return

        self.current_metadata.update(
            {"status": "completed", "end_time": datetime.now().isoformat()}
        )

        if final_results:
            self.current_metadata.update(final_results)

        self._save_metadata()

        self.logger.info(
            "Finished experiment",
            experiment_id=self.current_experiment_id,
            experiment_dir=str(self.current_experiment_dir),
        )

        self.current_experiment_id = None
        self.current_experiment_dir = None
        self.current_metadata = {}

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all available experiments.

        Returns:
            List of experiment metadata
        """
        experiments = []

        for exp_dir in self.results_dir.iterdir():
            if exp_dir.is_dir():
                metadata_file = exp_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                        experiments.append(metadata)
                    except Exception as e:
                        self.logger.warning(
                            "Failed to load experiment metadata",
                            experiment_dir=str(exp_dir),
                            error=str(e),
                        )

        # Sort by start time (newest first)
        experiments.sort(key=lambda x: x.get("start_time", ""), reverse=True)
        return experiments

    def load_experiment_data(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """Load experiment data by name.

        Args:
            experiment_name: Name of experiment directory

        Returns:
            Experiment data or None if not found
        """
        exp_dir = self.results_dir / experiment_name
        if not exp_dir.exists():
            return None

        return self._load_experiment_data(exp_dir)

    def get_latest_experiment_data(self) -> Optional[Dict[str, Any]]:
        """Get data from the most recent experiment.

        Returns:
            Latest experiment data or None
        """
        experiments = self.list_experiments()
        if not experiments:
            return None

        latest_exp = experiments[0]
        return self.load_experiment_data(latest_exp["experiment_name"])

    def _save_metadata(self) -> None:
        """Save experiment metadata to file."""
        if not self.current_experiment_dir:
            return

        metadata_file = self.current_experiment_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(self.current_metadata, f, indent=2)

    def _save_experiment_data(self, data: Dict[str, Any]) -> None:
        """Save experiment data to file.

        Args:
            data: Experiment data to save
        """
        if not self.current_experiment_dir:
            return

        data_file = self.current_experiment_dir / "experiment_data.json"
        with open(data_file, "w") as f:
            json.dump(data, f, indent=2)

    def _load_experiment_data(self, exp_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Load experiment data from file.

        Args:
            exp_dir: Experiment directory (uses current if None)

        Returns:
            Experiment data
        """
        if exp_dir is None:
            exp_dir = self.current_experiment_dir

        if not exp_dir:
            return {}

        data_file = exp_dir / "experiment_data.json"
        if not data_file.exists():
            return {}

        try:
            with open(data_file) as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(
                "Failed to load experiment data", data_file=str(data_file), error=str(e)
            )
            return {}

    def cleanup_old_experiments(self, keep_count: int = 10) -> None:
        """Clean up old experiment directories.

        Args:
            keep_count: Number of recent experiments to keep
        """
        experiments = self.list_experiments()

        if len(experiments) <= keep_count:
            return

        # Remove oldest experiments
        to_remove = experiments[keep_count:]
        for exp in to_remove:
            exp_dir = self.results_dir / exp["experiment_name"]
            if exp_dir.exists():
                import shutil

                shutil.rmtree(exp_dir)
                self.logger.info(
                    "Removed old experiment", experiment_name=exp["experiment_name"]
                )
