# -*- coding: utf-8 -*-
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import duckdb
import matplotlib.pyplot as plt
from openai import BadRequestError

from src.config.settings import DB_PATH
from src.schemas.models import Hypothesis, Insight
from src.utils.run_utils import get_run_dir

logger = logging.getLogger(__name__)


def compute_summary_stats(table_or_view: str, limit: int = 10000) -> str:
    """
    Computes comprehensive summary statistics for all columns in a DuckDB table or view.
    - Numerical: mean, median, mode, std, min, max, skewness, kurtosis, percentiles, missing count/ratio.
    - Categorical: unique count, top frequencies, mode, missing count/ratio.
    Returns a markdown-formatted report.
    """
    import numpy as np
    import pandas as pd

    try:
        with duckdb.connect(database=str(DB_PATH), read_only=True) as conn:
            # Sample up to limit rows for efficiency
            df = conn.execute(f'SELECT * FROM "{table_or_view}" LIMIT {limit}').fetchdf()
        if df.empty:
            return f"No data in {table_or_view}."
        report = f"# Summary Statistics for `{table_or_view}`\n"
        for col in df.columns:
            report += f"\n## Column: `{col}`\n"
            series = df[col]
            n_missing = series.isnull().sum()
            missing_ratio = n_missing / len(series)
            report += f"- Missing: {n_missing} ({missing_ratio:.2%})\n"
            if pd.api.types.is_numeric_dtype(series):
                desc = series.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
                report += "- Type: Numerical\n"
                report += f"- Count: {desc['count']}\n"
                report += f"- Mean: {desc['mean']:.4f}\n"
                report += f"- Std: {desc['std']:.4f}\n"
                report += f"- Min: {desc['min']}\n"
                report += f"- 5th pct: {desc.get('5%', np.nan)}\n"
                report += f"- 25th pct: {desc.get('25%', np.nan)}\n"
                report += f"- Median: {desc['50%']}\n"
                report += f"- 75th pct: {desc.get('75%', np.nan)}\n"
                report += f"- 95th pct: {desc.get('95%', np.nan)}\n"
                report += f"- Max: {desc['max']}\n"
                mode = series.mode().iloc[0] if not series.mode().empty else "N/A"
                report += f"- Mode: {mode}\n"
            else:
                report += "- Type: Categorical\n"
                report += "- # Unique: {}\n".format(series.nunique())
                mode = series.mode().iloc[0] if not series.mode().empty else "N/A"
                report += "- Mode: {}\n".format(mode)
                top_freq = series.value_counts().head(5)
                report += "- Top Values:\n"
                for idx, val in enumerate(top_freq.index):
                    report += "    - {}: {}\n".format(val, top_freq.iloc[idx])
            report += "---\n"
        return truncate_output_to_word_limit(report, 1000)
    except duckdb.Error as e:
        logger.error("Failed to compute summary stats for %s: %s", table_or_view, e)
        return "ERROR: Could not compute summary stats for {}: {}".format(table_or_view, e)


def truncate_output_to_word_limit(text: str, word_limit: int = 1000) -> str:
    """
    Truncate the output to a maximum number of words, appending a message if truncation occurred.
    """
    words = text.split()
    if len(words) > word_limit:
        truncated = " ".join(words[:word_limit])
        return (
            truncated
            + f"\n\n---\n[Output truncated to {word_limit} words. Please refine your query or request a smaller subset if needed.]"
        )
    return text


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
            return truncate_output_to_word_limit(df.to_markdown(index=False), 1000)
    except duckdb.Error as e:
        logger.error("SQL query failed: %s | Error: %s", query, e)
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


def create_plot(
    query: str,
    plot_type: str = "scatter",
    x: Union[str, None] = None,
    y: Union[str, None] = None,
    file_name: str = "plot.png",
    analysis_prompt: Union[str, None] = None,
) -> dict:
    """
    Executes a SQL query, generates a matplotlib plot, saves it, and analyzes it using vision_tool.
    Args:
        query: SQL SELECT query. Must return a DataFrame with named columns.
        plot_type: One of ['scatter', 'bar', 'hist']
        x: Name of the column for x-axis (required for scatter/bar)
        y: Name of the column for y-axis (required for scatter/bar)
        file_name: Desired file name for the output plot (should end with .png)
        analysis_prompt: Prompt for vision_tool analysis (optional)
    Returns:
        Dict with keys: 'plot_path', 'vision_analysis', and 'error' if any.
    """
    try:
        with duckdb.connect(database=str(DB_PATH), read_only=True) as conn:
            df = conn.execute(query).fetchdf()
        if df.empty:
            return {"error": "Query returned no data to plot."}
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        if plot_type == "scatter":
            if x is None or y is None:
                return {"error": "'x' and 'y' must be specified for scatter plots."}
            plt.scatter(df[x], df[y], alpha=0.7)
            plt.xlabel(x)
            plt.ylabel(y)
        elif plot_type == "bar":
            if x is None or y is None:
                return {"error": "'x' and 'y' must be specified for bar plots."}
            plt.bar(df[x], df[y])
            plt.xlabel(x)
            plt.ylabel(y)
        elif plot_type == "hist":
            if x is None:
                return {"error": "'x' must be specified for histogram plots."}
            plt.hist(df[x], bins=20, alpha=0.7)
            plt.xlabel(x)
            plt.ylabel("Frequency")
        else:
            return {"error": f"Unknown plot_type '{plot_type}'. Use 'scatter', 'bar', or 'hist'."}
        plt.title(f"{plot_type.title()} plot of {y if y else x}")
        abs_path = save_plot(file_name)
        # Automatically call vision_tool
        from src.utils.tools import vision_tool

        if analysis_prompt is None:
            analysis_prompt = "Analyze this plot and summarize key trends and anomalies."
        vision_result = vision_tool(abs_path, analysis_prompt)
        return {"plot_path": abs_path, "vision_analysis": vision_result}
    except Exception as e:
        logger.error(f"Failed to create plot: {e}")
        return {"error": f"Could not create plot. {e}"}


def create_analysis_view(view_name: str, sql_query: str, rationale: str):
    """
    Creates a permanent view for analysis. It opens a temporary write-enabled
    connection to do so, avoiding holding a lock.
    """
    try:
        with duckdb.connect(database=str(DB_PATH), read_only=False) as write_conn:
            # Check if view exists to handle versioning
            existing_views = [v[0] for v in write_conn.execute("SHOW TABLES;").fetchall()]

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
    except duckdb.Error as e:
        logger.error("DuckDB error during view cleanup: %s", e)
    except OSError as e:
        logger.error("OS error during view cleanup: %s", e)
    except Exception as e:
        # This is intentionally broad to ensure all unexpected errors during cleanup are logged and do not crash the orchestrator.
        logger.error("Unexpected error during view cleanup: %s", e)


def get_add_insight_tool(session_state):
    """Returns a function that can be used as an AutoGen tool to add insights."""

    def add_insight_to_report(
        title: str,
        finding: str,
        source_representation: str,
        reasoning_trace: List[str],
        supporting_code: Union[None, str] = None,
        plot_path: Union[None, str] = None,
        plot_interpretation: Union[None, str] = None,
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
            reasoning_trace: Step-by-step reasoning chain for this insight (required)
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
                reasoning_trace=reasoning_trace,
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


def get_add_to_central_memory_tool(session_state):
    """Returns a function that agents can call to add notes to the shared central memory."""
    from datetime import datetime

    def add_to_central_memory(
        note: str, reasoning: str, agent: str, metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """Appends a structured entry to ``session_state.central_memory`` and persists it.

        Args:
            note: The core note or finding.
            reasoning: Short rationale explaining the significance of the note.
            agent: Name or identifier of the calling agent.
            metadata: Optional dict with extra context (e.g., related tables, feature names).
        Returns:
            Confirmation string on success.
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "agent": agent,
            "note": note,
            "reasoning": reasoning,
        }
        if metadata:
            # Ensure all metadata values are strings for serialization
            entry["metadata"] = {str(k): str(v) for k, v in metadata.items()}

        session_state.central_memory.append(entry)
        # Persist session state
        session_state.save_to_disk()
        logger.info("Central memory updated by %s", agent)
        return f"Note added to central memory by {agent}."

    return add_to_central_memory


def get_finalize_hypotheses_tool(session_state):
    """Returns a function that can be used as an AutoGen tool to finalize hypotheses."""

    def finalize_hypotheses(hypotheses_data: list) -> str:
        """
        Finalizes the list of vetted hypotheses after validation.
        """
        # Step 1: Instantiate Hypothesis models (assigns IDs, validates fields)
        try:
            hyp_models = [Hypothesis(**h) for h in hypotheses_data]
        except Exception as e:
            logger.error(f"Failed to instantiate Hypothesis models: {e}")
            return f"ERROR: Failed to instantiate Hypothesis models. Reason: {e}"

        # Step 2: Validate for duplicate IDs and empty rationales
        ids = set()
        for h in hyp_models:
            if h.id in ids:
                logger.error(f"Duplicate hypothesis ID found: {h.id}")
                return f"ERROR: Hypothesis validation failed. Duplicate hypothesis ID found: {h.id}"
            ids.add(h.id)
            if not h.rationale:
                logger.error(f"Hypothesis {h.id} has an empty rationale.")
                return f"ERROR: Hypothesis validation failed. Hypothesis {h.id} has an empty rationale."

        # Step 3: Save to session state
        try:
            session_state.finalize_hypotheses(hyp_models)
            logger.info(f"Finalized and saved {len(hyp_models)} valid hypotheses.")
            return f"SUCCESS: Successfully finalized and saved {len(hyp_models)} hypotheses."
        except Exception as e:
            logger.error(f"Failed to save hypotheses after validation: {e}")
            return f"ERROR: Failed to save hypotheses after validation. Reason: {e}"

    return finalize_hypotheses


def validate_hypotheses(hypotheses_data: List[Dict], insight_report: str) -> Tuple[bool, str]:
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


def get_save_features_tool(session_state):
    """Returns a function that can be used as an AutoGen tool to save features to the session state."""
    def save_features(features_data: list) -> str:
        """
        Saves a list of features (as dicts) to session_state.features.
        """
        try:
            features_dict = {f.get('name', f'feature_{i}'): f for i, f in enumerate(features_data)}
            session_state.set_state("features", features_dict)
            logger.info(f"Saved {len(features_dict)} features to session state.")
            return f"SUCCESS: Successfully saved {len(features_dict)} features to session state."
        except Exception as e:
            logger.error(f"Failed to save features: {e}")
            return f"ERROR: Failed to save features. Reason: {e}"
    return save_features
    from pathlib import Path

    from openai import OpenAI

    try:
        # Robust path resolution
        full_path = Path(image_path)
        logger.info(
            f"vision_tool: Received image_path='{image_path}' (absolute? {full_path.is_absolute()})"
        )
        if not full_path.is_absolute():
            # Try CWD first
            if not full_path.exists():
                # Try run_dir/plots/image_path
                run_dir = get_run_dir()
                candidate = run_dir / "plots" / image_path
                logger.info(f"vision_tool: Trying run_dir/plots: '{candidate}'")
                if candidate.exists():
                    full_path = candidate
        if not full_path.exists():
            logger.error(f"vision_tool: File not found at '{full_path}' (original: '{image_path}')")
            return f"ERROR: File not found at '{image_path}'. Please ensure the file was saved correctly."
        logger.info(
            f"vision_tool: Using resolved image path: '{full_path}' (exists: {full_path.exists()})"
        )

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
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
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
        return (
            "ERROR: OpenAI library is not installed. Please install it with `pip install openai`."
        )
    except Exception as e:
        logger.error("Unexpected error during image analysis: %s", e)
        return f"ERROR: An unexpected error occurred while analyzing the image: {e}"


def _execute_python_run_code(pipe, code, run_dir):
    # Headless plotting
    import matplotlib

    matplotlib.use("Agg")
    from pathlib import Path

    import duckdb
    import matplotlib.pyplot as plt

    from src.config.settings import DB_PATH
    from src.utils.tools import get_table_sample

    # Save plot helper using provided run_dir
    def save_plot(filename: str):
        try:
            plots_dir = Path(run_dir) / "plots"
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
        except Exception as e:
            print(f"ERROR: Could not save plot: {e}")
            return None

    # Dummy add_insight_to_report for now
    def add_insight_to_report(title, finding, supporting_evidence, confidence):
        print(
            "INSIGHT_ADDED: {'title': title, 'finding': finding, 'supporting_evidence': supporting_evidence, 'confidence': confidence}"
        )

    # Provide a real DuckDB connection for the code
    conn = duckdb.connect(database=str(DB_PATH), read_only=False)
    # If in future you want to expose CV folds or other context, load and inject here.
    local_ns = {
        "save_plot": save_plot,
        "get_table_sample": get_table_sample,
        "conn": conn,
        "add_insight_to_report": add_insight_to_report,
        "__builtins__": __builtins__,
    }
    import contextlib
    import io
    import traceback

    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, local_ns, local_ns)
        conn.close()
        pipe.send(stdout.getvalue().strip())
    except Exception as e:
        tb = traceback.format_exc()
        pipe.send(f"ERROR: An unexpected error occurred: {e}\n{tb}")


def execute_python(code: str, timeout: int = 60) -> str:
    """
    Executes a string of Python code in a controlled, headless, and time-limited environment with injected helper functions.
    Injected helpers: save_plot, get_table_sample, conn (DuckDB connection), add_insight_to_report, etc.
    - Plots are always generated in headless mode (matplotlib 'Agg').
    - Each call is stateless: agents must reload data in each code block.
    - If code execution exceeds the timeout, it is forcibly terminated.
    Args:
        code: The Python code to execute.
        timeout: The timeout in seconds for the subprocess.
    Returns:
        The stdout of the executed code, or an error message if it fails.
    """
    import multiprocessing

    run_dir = str(get_run_dir())
    parent_conn, child_conn = multiprocessing.Pipe()
    p = multiprocessing.Process(target=_execute_python_run_code, args=(child_conn, code, run_dir))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return f"ERROR: Code execution timed out after {timeout} seconds."
    if parent_conn.poll():
        return parent_conn.recv()
    return "ERROR: No output returned from code execution."
