# -*- coding: utf-8 -*-
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt

from src.config.settings import DB_PATH
from src.schemas.models import Hypothesis, Insight
from src.utils.run_utils import get_run_dir

logger = logging.getLogger(__name__)


def run_sql_query(query: str) -> str:
    """
    Executes a read-only SQL query against the database and returns the result as a markdown string.
    This tool should be used for all SELECT queries.
    """
    try:
        with duckdb.connect(database=str(DB_PATH), read_only=True) as conn:
            df = conn.execute(query).fetchdf()
            if df.empty:
                return "Query executed successfully, but returned no results."
            return df.to_markdown(index=False)
    except Exception as e:
        logger.error(f"SQL query failed: {query} | Error: {e}")
        return f"ERROR: SQL query failed: {e}"


def get_table_sample(table_name: str, n_samples: int = 5) -> str:
    """Retrieves a random sample of rows from a specified table in the database."""
    return run_sql_query(f'SELECT * FROM "{table_name}" USING SAMPLE {n_samples} ROWS;')


def save_plot(filename: str):
    """Saves the current matplotlib figure to the run-local 'plots' directory."""
    plots_dir = get_run_dir() / "plots"
    plots_dir.mkdir(exist_ok=True)
    basename = Path(filename).name
    if not basename.lower().endswith(".png"):
        basename += ".png"
    path = plots_dir / basename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    abs_path = path.resolve()
    print(f"PLOT_SAVED:{abs_path}")
    return str(abs_path)


def create_analysis_view(view_name: str, sql_query: str, rationale: str):
    """Creates a permanent view for analysis, documents it, and tracks it for cleanup."""
    with duckdb.connect(database=str(DB_PATH), read_only=False) as write_conn:
        existing_views = [v[0] for v in write_conn.execute("SHOW TABLES;").fetchall()]
        actual_name = view_name
        version = 2
        while actual_name in existing_views:
            actual_name = f"{view_name}_v{version}"
            version += 1
        if actual_name != view_name:
            print(
                f"View '{view_name}' already exists. Creating '{actual_name}' instead."
            )
        full_sql = f"CREATE OR REPLACE VIEW {actual_name} AS ({sql_query})"
        write_conn.execute(full_sql)
        views_file = get_run_dir() / "generated_views.json"
        views_data = (
            json.load(open(views_file)) if views_file.exists() else {"views": []}
        )
        views_data["views"].append(
            {
                "name": actual_name,
                "original_name": view_name,
                "sql": sql_query,
                "rationale": rationale,
                "created_at": datetime.now().isoformat(),
            }
        )
        with open(views_file, "w") as f:
            json.dump(views_data, f, indent=2)
        print(f"VIEW_CREATED:{actual_name}")
        return f"Successfully created view: {actual_name}"


def cleanup_analysis_views(run_dir: Path):
    """Cleans up any database views created during a run."""
    views_file = run_dir / "generated_views.json"
    if not views_file.exists():
        print("No views to clean up.")
        return

    try:
        with open(views_file, "r") as f:
            views_data = json.load(f)

        views_to_drop = [view["name"] for view in views_data["views"]]

        if not views_to_drop:
            print("No views to clean up.")
            return

        with duckdb.connect(database=DB_PATH, read_only=False) as conn:
            for view_name in views_to_drop:
                try:
                    conn.execute(f"DROP VIEW IF EXISTS {view_name};")
                    print(f"Successfully dropped view: {view_name}")
                except Exception as e:
                    print(f"Warning: Could not drop view {view_name}: {e}")
        # Optionally remove the tracking file after cleanup
        # views_file.unlink()
    except Exception as e:
        print(f"Error during view cleanup: {e}")


def get_add_insight_tool(session_state):
    """Returns a function that can be used as an AutoGen tool to add insights."""

    def add_insight_to_report(
        title: str,
        finding: str,
        source_representation: str,
        supporting_code: str = None,
        plot_path: str = None,
        plot_interpretation: str = None,
    ) -> str:
        """
        Adds a structured insight to the session report.

        Args:
            title: A concise, descriptive title for the insight
            finding: The detailed finding or observation
            source_representation: The name of the SQL View or Graph used for analysis
            supporting_code: The exact SQL or Python code used to generate the finding
            plot_path: The path to the plot that visualizes the finding
            plot_interpretation: LLM-generated analysis of what the plot shows

        Returns:
            Confirmation message
        """
        insight = Insight(
            title=title,
            finding=finding,
            source_representation=source_representation,
            supporting_code=supporting_code,
            plot_path=plot_path,
            plot_interpretation=plot_interpretation,
        )

        session_state.add_insight(insight)
        logger.info(f"Insight '{insight.title}' added.")
        return f"Successfully added insight: '{title}' to the report."

    return add_insight_to_report


def get_finalize_hypotheses_tool(session_state):
    """Returns a function that can be used as an AutoGen tool to finalize hypotheses."""

    def finalize_hypotheses(hypotheses_data: list) -> str:
        """
        Finalizes the list of vetted hypotheses.

        Args:
            hypotheses_data: List of dictionaries containing hypothesis information
            Each dict should have: id, description, strategic_critique, feasibility_critique

        Returns:
            Confirmation message
        """
        hypotheses = []
        for h_data in hypotheses_data:
            hypothesis = Hypothesis(
                id=h_data["id"],
                description=h_data["description"],
                strategic_critique=h_data["strategic_critique"],
                feasibility_critique=h_data["feasibility_critique"],
            )
            hypotheses.append(hypothesis)

        session_state.finalize_hypotheses(hypotheses)
        logger.info(f"Finalized {len(hypotheses)} hypotheses.")
        return f"Successfully finalized {len(hypotheses)} hypotheses."

    return finalize_hypotheses


def vision_tool(image_path: str, prompt: str) -> str:
    """Analyzes an image file using OpenAI's GPT-4o vision model."""
    import base64
    from pathlib import Path

    from openai import OpenAI

    try:
        # Try to resolve path relative to current working directory
        full_path = Path(image_path)
        if not full_path.exists():
            # Try relative to run directory if available
            run_dir = get_run_dir()
            full_path = run_dir / image_path

        if not full_path.exists():
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
        return f"ERROR: An unexpected error occurred while analyzing the image: {e}"
