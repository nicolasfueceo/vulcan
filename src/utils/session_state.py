import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb

from src.schemas.models import Hypothesis, Insight
from src.utils.run_utils import get_run_dir


class SessionState:
    """Manages the state and artifacts of a single pipeline run."""

    def __init__(self, run_dir: Optional[Path] = None):
        self.run_dir = run_dir or get_run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize default state
        self.insights: List[Insight] = []
        self.hypotheses: List[Hypothesis] = []

        # Additional state for complete pipeline management
        self.prioritized_hypotheses: List[Dict] = []
        self.candidate_features: List[Dict] = []
        self.best_params: Dict = {}
        self.best_rmse: Optional[float] = None
        self.bo_history: Dict = {}
        self.reflections: List[Dict] = []

        # Run counters for agents
        self.ideation_run_count: int = 0
        self.feature_realization_run_count: int = 0
        self.reflection_run_count: int = 0

        # Load existing state if available
        self._load_from_disk()

        self.db_path = "data/goodreads_curated.duckdb"
        try:
            # Connect in read-write mode to allow for TEMP view creation
            self.conn = duckdb.connect(database=self.db_path, read_only=False)
            print(f"Successfully connected to {self.db_path} in read-write mode.")
        except Exception as e:
            print(f"FATAL: Failed to connect to database at {self.db_path}: {e}")
            self.conn = None
            raise e

    def _load_from_disk(self):
        """Loads existing session state from disk if available."""
        state_file = self.run_dir / "session_state.json"
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    data = json.load(f)

                # Load insights and hypotheses using proper model classes
                if "insights" in data:
                    self.insights = [Insight(**insight) for insight in data["insights"]]
                if "hypotheses" in data:
                    self.hypotheses = [
                        Hypothesis(**hypothesis) for hypothesis in data["hypotheses"]
                    ]

                # Load simple state fields
                self.prioritized_hypotheses = data.get("prioritized_hypotheses", [])
                self.candidate_features = data.get("candidate_features", [])
                self.best_params = data.get("best_params", {})
                self.best_rmse = data.get("best_rmse")
                self.bo_history = data.get("bo_history", {})
                self.reflections = data.get("reflections", [])
                self.set_state("features", data.get("features", {}))
                self.set_state("metrics", data.get("metrics", {}))
                self.set_state("models", data.get("models", {}))

                # Load run counters
                self.ideation_run_count = data.get("ideation_run_count", 0)
                self.feature_realization_run_count = data.get(
                    "feature_realization_run_count", 0
                )
                self.reflection_run_count = data.get("reflection_run_count", 0)

                print(
                    f"Loaded existing session state with {len(self.insights)} insights and {len(self.hypotheses)} hypotheses."
                )
            except Exception as e:
                print(
                    f"Warning: Failed to load existing session state: {e}. Starting with fresh state."
                )
        else:
            print("No existing session state found. Starting with fresh state.")

    def add_insight(self, insight: Insight):
        self.insights.append(insight)
        self.save_to_disk()
        print(f"Added and saved new insight: '{insight.title}'")

    def finalize_hypotheses(self, hypotheses: List[Hypothesis]):
        self.hypotheses.extend(hypotheses)
        self.save_to_disk()
        print(f"Finalized and saved {len(hypotheses)} hypotheses.")

    # Prioritized hypotheses management
    def set_prioritized_hypotheses(self, hypotheses: List[Dict]):
        self.prioritized_hypotheses = hypotheses
        self.save_to_disk()

    def get_prioritized_hypotheses(self) -> List[Dict]:
        return self.prioritized_hypotheses

    # Candidate features management
    def set_candidate_features(self, features: List[Dict]):
        self.candidate_features = features
        self.save_to_disk()

    def get_candidate_features(self) -> List[Dict]:
        return self.candidate_features

    # Optimization results management
    def set_best_params(self, params: Dict):
        self.best_params = params
        self.save_to_disk()

    def get_best_params(self) -> Dict:
        return self.best_params

    def set_best_rmse(self, rmse: float):
        self.best_rmse = rmse
        self.save_to_disk()

    def get_best_rmse(self) -> Optional[float]:
        return self.best_rmse

    def set_bo_history(self, history: Dict):
        self.bo_history = history
        self.save_to_disk()

    def get_bo_history(self) -> Dict:
        return self.bo_history

    # Reflections management
    def add_reflection(self, reflection: Dict):
        self.reflections.append(reflection)
        self.save_to_disk()

    def get_reflections(self) -> List[Dict]:
        return self.reflections

    # Run counters management
    def increment_ideation_run_count(self) -> int:
        self.ideation_run_count += 1
        self.save_to_disk()
        return self.ideation_run_count

    def get_ideation_run_count(self) -> int:
        return self.ideation_run_count

    def increment_feature_realization_run_count(self) -> int:
        self.feature_realization_run_count += 1
        self.save_to_disk()
        return self.feature_realization_run_count

    def get_feature_realization_run_count(self) -> int:
        return self.feature_realization_run_count

    def increment_reflection_run_count(self) -> int:
        self.reflection_run_count += 1
        self.save_to_disk()
        return self.reflection_run_count

    def get_reflection_run_count(self) -> int:
        return self.reflection_run_count

    # Feature, metric, and model storage
    def store_feature(self, feature_name: str, feature_data: Dict):
        """Store feature data in the session state."""
        features = self.get_state("features", {})
        features[feature_name] = feature_data
        self.set_state("features", features)

    def get_feature(self, feature_name: str) -> Optional[Dict]:
        """Get feature data from the session state."""
        features = self.get_state("features", {})
        return features.get(feature_name)

    def store_metric(self, metric_name: str, metric_data: Dict):
        """Store metric data in the session state."""
        metrics = self.get_state("metrics", {})
        metrics[metric_name] = metric_data
        self.set_state("metrics", metrics)

    def get_metric(self, metric_name: str) -> Optional[Dict]:
        """Get metric data from the session state."""
        metrics = self.get_state("metrics", {})
        return metrics.get(metric_name)

    def store_model(self, model_name: str, model_data: Dict):
        """Store model data in the session state."""
        models = self.get_state("models", {})
        models[model_name] = model_data
        self.set_state("models", models)

    def get_model(self, model_name: str) -> Optional[Dict]:
        """Get model data from the session state."""
        models = self.get_state("models", {})
        return models.get(model_name)

    # Generic get/set methods for backward compatibility and any additional state
    def get_state(self, key: str, default: Any = None) -> Any:
        """Generic getter for any state attribute."""
        return getattr(self, key, default)

    def set_state(self, key: str, value: Any):
        """Generic setter for any state attribute."""
        setattr(self, key, value)
        self.save_to_disk()

    def get_final_insight_report(self) -> str:
        """Returns a string report of all insights generated."""
        if not self.insights:
            return "No insights were generated during this run."

        report = "--- INSIGHTS REPORT ---\n\n"
        for i, insight in enumerate(self.insights, 1):
            report += f"Insight {i}: {insight.title}\n"
            report += f"  Finding: {insight.finding}\n"
            if insight.source_representation:
                report += f"  Source: {insight.source_representation}\n"
            if insight.supporting_code:
                report += f"  Code:\n```\n{insight.supporting_code}\n```\n"
            if insight.plot_path:
                report += f"  Plot: {insight.plot_path}\n"
            report += "\n"
        return report

    def get_final_hypotheses(self) -> Optional[List[Hypothesis]]:
        """Returns the final list of vetted hypotheses."""
        return self.hypotheses if self.hypotheses else None

    def vision_tool(self, image_path: str, prompt: str) -> str:
        """
        Analyzes an image file using OpenAI's GPT-4o vision model.
        This tool automatically resolves the path relative to the run's output directory.
        """
        import base64
        import os

        from openai import OpenAI

        try:
            full_path = self.get_run_output_dir() / image_path
            if not full_path.exists():
                logger.error(f"Vision tool failed: File not found at {full_path}")
                return f"ERROR: File not found at '{image_path}'. Please ensure the file was saved correctly."

            # Initialize OpenAI client
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            # Read and encode the image
            with open(full_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1000,
            )
            return response.choices[0].message.content
        except ImportError:
            return "ERROR: OpenAI library is not installed. Please install it with `pip install openai`."
        except Exception as e:
            logger.error(f"Vision tool failed with an unexpected error: {e}")
            return f"ERROR: An unexpected error occurred while analyzing the image: {e}"

    def save_to_disk(self):
        """Saves the current session state to disk."""
        output = {
            "insights": [i.model_dump() for i in self.insights],
            "hypotheses": [h.model_dump() for h in self.hypotheses],
            "prioritized_hypotheses": self.prioritized_hypotheses,
            "candidate_features": self.candidate_features,
            "best_params": self.best_params,
            "best_rmse": self.best_rmse,
            "bo_history": self.bo_history,
            "reflections": self.reflections,
            "features": self.get_state("features", {}),
            "metrics": self.get_state("metrics", {}),
            "models": self.get_state("models", {}),
            "ideation_run_count": self.ideation_run_count,
            "feature_realization_run_count": self.feature_realization_run_count,
            "reflection_run_count": self.reflection_run_count,
        }
        output_path = self.run_dir / "session_state.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=4)

    def close_connection(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            print("Database connection closed.")

    def reconnect(self):
        """Reopens the database connection."""
        try:
            self.conn = duckdb.connect(database=self.db_path, read_only=False)
            print(f"Successfully reconnected to {self.db_path} in read-write mode.")
        except Exception as e:
            print(f"FATAL: Failed to reconnect to database at {self.db_path}: {e}")
            self.conn = None
            raise e

    @property
    def db_connection(self):
        """Returns the current database connection."""
        return self.conn
