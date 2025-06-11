import json
from pathlib import Path
from typing import List, Optional

import duckdb

from src.utils.run_utils import get_run_dir
from src.utils.schemas_v2 import Hypothesis, Insight


class SessionState:
    """Manages the state and artifacts of a single pipeline run."""

    def __init__(self, run_dir: Optional[Path] = None):
        self.run_dir = run_dir or get_run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.insights: List[Insight] = []
        self.hypotheses: List[Hypothesis] = []

        db_path = "data/goodreads_curated.duckdb"
        try:
            # Connect in read-write mode to allow for TEMP view creation
            self.conn = duckdb.connect(database=db_path, read_only=False)
            print(f"Successfully connected to {db_path} in read-write mode.")
        except Exception as e:
            print(f"FATAL: Failed to connect to database at {db_path}: {e}")
            self.conn = None
            raise e

    def add_insight(self, insight: Insight):
        self.insights.append(insight)
        self.save_to_disk()
        print(f"Added and saved new insight: '{insight.title}'")

    def finalize_hypotheses(self, hypotheses: List[Hypothesis]):
        self.hypotheses.extend(hypotheses)
        self.save_to_disk()
        print(f"Finalized and saved {len(hypotheses)} hypotheses.")

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

    def save_to_disk(self):
        """Saves the current session state to disk."""
        output = {
            "insights": [i.model_dump() for i in self.insights],
            "hypotheses": [h.model_dump() for h in self.hypotheses],
        }
        output_path = self.run_dir / "session_state.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=4)

    def close_connection(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            print("Database connection closed.")
