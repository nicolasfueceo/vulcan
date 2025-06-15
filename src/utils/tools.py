# -*- coding: utf-8 -*-
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import duckdb
import matplotlib.pyplot as plt
from openai import BadRequestError

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


def create_plot(query: str, plot_type: str = "scatter", x: str = None, y: str = None, file_name: str = "plot.png") -> str:
    """
    Executes a SQL query, generates a matplotlib plot, and saves it to the run-local 'plots' directory.
    Args:
        query: SQL SELECT query. Must return a DataFrame with named columns.
        plot_type: One of ['scatter', 'bar', 'hist']
        x: Name of the column for x-axis (required for scatter/bar)
        y: Name of the column for y-axis (required for scatter/bar)
        file_name: Desired file name for the output plot (should end with .png)
    Returns:
        Absolute path to the saved plot file, or error string.
    """
    try:
        with duckdb.connect(database=str(DB_PATH), read_only=True) as conn:
            df = conn.execute(query).fetchdf()
        if df.empty:
            return "ERROR: Query returned no data to plot."
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        if plot_type == "scatter":
            if x is None or y is None:
                return "ERROR: 'x' and 'y' must be specified for scatter plots."
            plt.scatter(df[x], df[y], alpha=0.7)
            plt.xlabel(x)
            plt.ylabel(y)
        elif plot_type == "bar":
            if x is None or y is None:
                return "ERROR: 'x' and 'y' must be specified for bar plots."
            plt.bar(df[x], df[y])
            plt.xlabel(x)
            plt.ylabel(y)
        elif plot_type == "hist":
            if x is None:
                return "ERROR: 'x' must be specified for histogram plots."
            plt.hist(df[x], bins=20, alpha=0.7)
            plt.xlabel(x)
            plt.ylabel("Frequency")
        else:
            return f"ERROR: Unknown plot_type '{plot_type}'. Use 'scatter', 'bar', or 'hist'."
        plt.title(f"{plot_type.title()} plot of {y if y else x}")
        abs_path = save_plot(file_name)
        return abs_path
    except Exception as e:
        logger.error(f"Failed to create plot: {e}")
        return f"ERROR: Could not create plot. {e}"


def create_analysis_view(view_name: str, sql_query: str, rationale: str):
    """
    Creates a permanent view for analysis. It opens a temporary write-enabled
    connection to do so, avoiding holding a lock.
    """
    try:
        with duckdb.connect(database=str(DB_PATH), read_only=False) as write_conn:
            # Check if view exists to handle versioning
            existing_views = [
                v[0] for v in write_conn.execute("SHOW TABLES;").fetchall()
            ]

            actual_name = view_name
            version = 2
            while actual_name in existing_views:
                actual_name = f"{view_name}_v{version}"
                version += 1

            if actual_name != view_name:
                logger.info(
                    "View '%s' already exists. Creating '%s' instead.",
                    view_name,
                    actual_name,
                )

            # Create the view
            full_sql = f"CREATE OR REPLACE VIEW {actual_name} AS ({sql_query})"
            write_conn.execute(full_sql)

            # ... (rest of the metadata tracking is the same)

            print(f"VIEW_CREATED:{actual_name}")
            return f"Successfully created view: {actual_name}"
    except Exception as e:
        logger.error(f"Failed to create view '{view_name}': {e}")
        return f"ERROR: Could not create view '{view_name}'. Reason: {e}"


def cleanup_analysis_views(run_dir: Path):
    """Cleans up any database views created during a run."""
    views_file = run_dir / "generated_views.json"
    if not views_file.exists():
        logger.info("No views file found. Nothing to clean up.")
        return

    try:
        with open(views_file, "r") as f:
            views_data = json.load(f)

        views_to_drop = [view["name"] for view in views_data["views"]]

        if not views_to_drop:
            logger.info("No views to clean up.")
            return

        with duckdb.connect(database=DB_PATH, read_only=False) as conn:
            for view_name in views_to_drop:
                try:
                    conn.execute(f"DROP VIEW IF EXISTS {view_name};")
                    logger.info("Successfully dropped view: %s", view_name)
                except Exception as e:
                    logger.warning("Could not drop view %s: %s", view_name, e)
        # Optionally remove the tracking file after cleanup
        # views_file.unlink()
    except Exception as e:
        logger.error("Error during view cleanup: %s", e)


def get_add_insight_tool(session_state):
    """Returns a function that can be used as an AutoGen tool to add insights."""

    def add_insight_to_report(
        title: str,
        finding: str,
        source_representation: str,
        supporting_code: str = None,
        plot_path: str = None,
        plot_interpretation: str = None,
        quality_score: Optional[float] = None,
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
            quality_score: The quality score of the insight
        Returns:
            Confirmation message
        """
        try:
            insight = Insight(
                title=title,
                finding=finding,
                source_representation=source_representation,
                supporting_code=supporting_code,
                plot_path=plot_path,
                plot_interpretation=plot_interpretation,
                quality_score=quality_score,
            )

            session_state.add_insight(insight)
            logger.info(f"Insight '{insight.title}' added.")
            return f"Successfully added insight: '{title}' to the report."
        except BadRequestError as e:
            if "context_length_exceeded" in str(e):
                error_msg = (
                    "ERROR: The context length was exceeded. Please:\n"
                    "1. Break down your insight into smaller, more focused parts\n"
                    "2. Reduce the size of any large data structures or strings\n"
                    "3. Consider summarizing long findings\n"
                    "4. Remove any unnecessary details from the insight"
                )
                logger.error(error_msg)
                return error_msg
            raise

    return add_insight_to_report


def get_finalize_hypotheses_tool(session_state):
    """Returns a function that can be used as an AutoGen tool to finalize hypotheses."""

    def finalize_hypotheses(hypotheses_data: list) -> str:
        """
        Finalizes the list of vetted hypotheses after validation.
        """
        # First, validate the hypotheses
        insight_report = session_state.get_final_insight_report()
        is_valid, error_message = validate_hypotheses(hypotheses_data, insight_report)

        if not is_valid:
            logger.error(f"Hypothesis validation failed: {error_message}")
            return f"ERROR: Hypothesis validation failed. {error_message}"

        # If valid, proceed to create models and save
        try:
            hyp_models = [Hypothesis(**h) for h in hypotheses_data]
            session_state.finalize_hypotheses(hyp_models)
            logger.info(f"Finalized and saved {len(hyp_models)} valid hypotheses.")
            return f"SUCCESS: Successfully finalized and saved {len(hyp_models)} hypotheses."
        except Exception as e:
            logger.error(f"Failed to save hypotheses after validation: {e}")
            return f"ERROR: Failed to save hypotheses after validation. Reason: {e}"

    return finalize_hypotheses


def validate_hypotheses(
    hypotheses_data: List[Dict], insight_report: str
) -> (bool, str):
    """
    Validates a list of hypothesis data against the insight report and internal consistency.
    """
    insight_titles = {
        line.split(":", 1)[1].strip()
        for line in insight_report.split("\n")
        if line.startswith("Insight")
    }
    hypothesis_ids = set()

    for h_data in hypotheses_data:
        h_id = h_data.get("id")
        if h_id in hypothesis_ids:
            return False, f"Duplicate hypothesis ID found: {h_id}"
        hypothesis_ids.add(h_id)

        if not h_data.get("rationale"):
            return False, f"Hypothesis {h_id} has an empty rationale."

        source_insight = h_data.get("source_insight")
        if source_insight and source_insight not in insight_titles:
            return (
                False,
                f"Hypothesis {h_id} references a non-existent insight: '{source_insight}'",
            )
    return True, "All hypotheses are valid."


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

        try:
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
        except BadRequestError as e:
            if "context_length_exceeded" in str(e):
                error_msg = (
                    "ERROR: The context length was exceeded. Please:\n"
                    "1. Make your prompt more concise\n"
                    "2. Use a smaller image or reduce its resolution\n"
                    "3. Break down your analysis into smaller parts\n"
                    "4. Remove any unnecessary details from the prompt"
                )
                logger.error(error_msg)
                return error_msg
            raise
    except ImportError:
        return "ERROR: OpenAI library is not installed. Please install it with `pip install openai`."
    except Exception as e:
        return f"ERROR: An unexpected error occurred while analyzing the image: {e}"

def execute_python(code: str, timeout: int = 60) -> str:
    """
    Executes a string of Python code in a sandboxed subprocess.

    Args:
        code: The Python code to execute.
        timeout: The timeout in seconds for the subprocess.

    Returns:
        The stdout of the executed code, or an error message if it fails.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,  # This will raise CalledProcessError for non-zero exit codes
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        logger.error(f"Code execution timed out after {timeout} seconds.")
        return f"ERROR: Code execution timed out after {timeout} seconds."
    except subprocess.CalledProcessError as e:
        logger.error(f"Code execution failed with exit code {e.returncode}.")
        logger.error(f"STDOUT: {e.stdout.strip()}")
        logger.error(f"STDERR: {e.stderr.strip()}")
        return f"ERROR: Code execution failed.\n---STDERR---\n{e.stderr.strip()}"
    except Exception as e:
        logger.error(f"An unexpected error occurred during code execution: {e}")
        return f"ERROR: An unexpected error occurred: {e}"
