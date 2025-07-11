# -*- coding: utf-8 -*-
import base64
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
from openai import BadRequestError, OpenAI

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
    Executes an SQL query against the database and returns the result as a markdown .
    """
    try:
        with duckdb.connect(database=str(DB_PATH), read_only=False) as conn:
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


def create_analysis_view(view_name: str, sql_query: str, rationale: str, session_state=None):
    """
    Creates a permanent view for analysis. It opens a temporary write-enabled
    connection to do so, avoiding holding a lock.
    Logs all arguments, versioning, success, and failure.
    """
    logger.info(
        f"[TOOL CALL] create_analysis_view called with arguments: view_name={view_name}, sql_query={sql_query}, rationale={rationale}, session_state_present={session_state is not None}"
    )
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
                    f"[TOOL INFO] View '{view_name}' already exists. Creating '{actual_name}' instead."
                )

            # Remove trailing semicolon (and whitespace) if present
            cleaned_sql_query = sql_query.rstrip().rstrip(';').rstrip()
            if cleaned_sql_query != sql_query.rstrip():
                logger.warning(f"[TOOL WARNING] Trailing semicolon removed from SQL query for view '{actual_name}'.")
            # Create the view
            full_sql = f"CREATE OR REPLACE VIEW {actual_name} AS ({cleaned_sql_query})"
            write_conn.execute(full_sql)
            logger.info(f"[TOOL SUCCESS] Created view {actual_name} with query: {cleaned_sql_query}")
            if session_state is not None and hasattr(session_state, "log_view_creation"):
                session_state.log_view_creation(actual_name, sql_query, rationale)
            print(f"VIEW_CREATED:{actual_name}")
            return f"Successfully created view: {actual_name}"
    except Exception as e:
        logger.error(f"[TOOL ERROR] Failed to create view {view_name}: {e}")
        return f"ERROR: Failed to create view '{view_name}'. Reason: {e}"


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
    """
    Returns a function that can be used as an AutoGen tool to finalize hypotheses.

    TOOL DESCRIPTION FOR AGENTS:
    ------------------------------------------------------------
    finalize_hypotheses(hypotheses_data: list) -> str

    This tool is used to submit the final list of hypotheses for the current discovery round. Each hypothesis MUST be a dictionary with the following structure:
        {
            "summary": <str, required, non-empty>,
            "rationale": <str, required, non-empty>,
            "id": <str, optional, will be auto-generated if omitted>
        }
    - The "summary" is a concise, one-sentence statement of the hypothesis.
    - The "rationale" is a clear explanation of why this hypothesis is useful and worth testing.
    - The "id" field is optional; if not provided, it will be auto-generated.
    - All fields must be strings. Empty or missing required fields will cause the tool to fail.
    - The tool will return an explicit error message if any item does not match the schema, or if any required field is missing or invalid.
    - If your call fails, read the error message carefully and correct your output to match the schema contract exactly.

    Example valid call:
        finalize_hypotheses([
            {"summary": "Users who review more books tend to give higher ratings.", "rationale": "Observed a positive correlation in the sample."},
            {"summary": "Standalone books are rated higher than series books.", "rationale": "Series books have more variance and lower means in ratings."}
        ])
    ------------------------------------------------------------
    """

    def finalize_hypotheses(hypotheses_data: list) -> str:
        """
        Validates and finalizes the list of vetted hypotheses. Each item in the list MUST
        conform to the Hypothesis schema (must include non-empty 'summary', 'rationale', and 'depends_on').
        - If any item is missing required fields or has an empty value, the tool will fail with a detailed error message.
        - If the call fails, carefully read the error and correct your output to match the schema contract.
        """
        logger.info(f"[TOOL CALL] finalize_hypotheses called with {len(hypotheses_data)} items.")
        validated_hypotheses = []
        # --- DB schema validation for depends_on ---
        # Get DB schema (tables and columns)
        import duckdb
        db_path = getattr(session_state, "db_path", None) or DB_PATH
        # Gather schema info once for DRY validation
        with duckdb.connect(database=str(db_path), read_only=True) as conn:
            tables = set(row[0] for row in conn.execute("SHOW TABLES").fetchall())
            table_columns = {
                t: set(row[1] for row in conn.execute(f"PRAGMA table_info('{t}')").fetchall())
                for t in tables
            }
        for i, h_data in enumerate(hypotheses_data):
            try:
                hypothesis = Hypothesis(**h_data)
            except Exception as e:
                error_message = (
                    f"[SCHEMA VALIDATION ERROR] Hypothesis at index {i} failed validation.\n"
                    f"Input: {h_data}\n"
                    f"Error: {e}\n"
                    "==> ACTION REQUIRED: Each hypothesis must be a dictionary with non-empty string fields 'summary', 'rationale', and a non-empty list 'depends_on'. 'id' is optional.\n"
                    "Please correct your output to match the schema contract exactly."
                )
                logger.error(f"[TOOL ERROR] {error_message}")
                return error_message
            # DRY: Use helper for depends_on validation
            depends_on = getattr(hypothesis, "depends_on", None)
            if depends_on:
                valid, dep_error = _validate_depends_on_schema(
                    depends_on, tables, table_columns, "Hypothesis", i
                )
                if not valid:
                    logger.error(f"[TOOL ERROR] {dep_error}")
                    return dep_error or "[DEPENDENCY VALIDATION ERROR] Unknown error."
            validated_hypotheses.append(hypothesis)
        try:
            session_state.finalize_hypotheses(validated_hypotheses)
            success_message = (
                f"SUCCESS: Successfully validated and saved {len(validated_hypotheses)} hypotheses."
            )
            logger.info(f"[TOOL SUCCESS] {success_message}")
            return success_message
        except Exception as e:
            error_message = (
                f"[INTERNAL ERROR] Failed to save hypotheses after validation. Reason: {e}"
            )
            logger.error(f"[TOOL ERROR] {error_message}")
            return error_message

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
    """
    Analyzes an image file using OpenAI's GPT-4o vision model.
    Args:
        image_path (str): Path to the image file (absolute or relative).
        prompt (str): Prompt for the vision model.
    Returns:
        str: Model response, or error message.
    """
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
                max_tokens=2000,
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
    except Exception as e:
        logger.error(f"An unexpected error occurred in vision_tool: {e}", exc_info=True)
        return f"ERROR: An unexpected error occurred: {e}"


def get_save_features_tool(session_state):
    """Returns a function that can be used as an AutoGen tool to save features to the session state."""

    def save_features(features_data: list) -> str:
        """
        Saves a list of features (as dicts) to session_state.features.
        """
        try:
            features_dict = {f.get("name", f"feature_{i}"): f for i, f in enumerate(features_data)}
            session_state.set_state("features", features_dict)
            logger.info(f"Saved {len(features_dict)} features to session state.")
            return f"SUCCESS: Successfully saved {len(features_dict)} features to session state."
        except Exception as e:
            logger.error(f"Failed to save features: {e}")
            return f"ERROR: Failed to save features. Reason: {e}"

    return save_features


def _execute_python_run_code(pipe, code, run_dir, session_state=None):
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

    # Create save_candidate_features function if session_state is available
    def save_candidate_features(candidate_features_data):
        if session_state is None:
            print("ERROR: save_candidate_features called but no session_state available")
            return "ERROR: No session state available"
        
        try:
            # Use the same logic as the registered tool
            from src.utils.tools import get_save_candidate_features_tool
            tool_func = get_save_candidate_features_tool(session_state)
            result = tool_func(candidate_features_data)
            print(f"SUCCESS: Saved {len(candidate_features_data)} candidate features")
            return result
        except Exception as e:
            print(f"ERROR: Failed to save candidate features: {e}")
            return f"ERROR: {e}"

    # Provide a real DuckDB connection for the code
    conn = duckdb.connect(database=str(DB_PATH), read_only=False)
    # Always import matplotlib and seaborn for agent code
    import matplotlib.pyplot as plt
    import seaborn as sns
    # If in future you want to expose CV folds or other context, load and inject here.
    local_ns = {
        "save_plot": save_plot,
        "get_table_sample": get_table_sample,
        "save_candidate_features": save_candidate_features,
        "conn": conn,
        "__builtins__": __builtins__,
        "plt": plt,
        "sns": sns,
    }
    import contextlib
    import io
    import traceback

    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, local_ns, local_ns)
        pipe.send(stdout.getvalue().strip())
    except Exception as e:
        tb = traceback.format_exc()
        pipe.send(f"ERROR: An unexpected error occurred: {e}\n{tb}")
    finally:
        conn.close()


def execute_python(code: str, timeout: int = 300, session_state=None) -> str:
    """
    NOTE: A pre-configured DuckDB connection object named `conn` is already provided in the execution environment. DO NOT create your own connection using duckdb.connect(). Use the provided `conn` for all SQL operations (e.g., conn.sql(...)).

    NOTE: After every major code block or SQL result, you should print the result using `print('!!!', result)` so outputs are clearly visible in logs and debugging is easier.

    NOTE: Variable context is NOT retained across runs. Each execution of this tool must be self contained, even if it means redeclaring variables.
    Executes a string of Python code in a controlled, headless, and time-limited environment with injected helper functions.
    Args:
        code: Python code to execute
        timeout: Maximum time (seconds) to allow execution (default: 300)
        session_state: Optional session state to make save_candidate_features available
    Returns:
        The stdout of the executed code, or an error message if it fails.
    """
    import multiprocessing

    run_dir = str(get_run_dir())
    parent_conn, child_conn = multiprocessing.Pipe()
    p = multiprocessing.Process(target=_execute_python_run_code, args=(child_conn, code, run_dir, session_state))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return f"ERROR: Code execution timed out after {timeout} seconds."
    if parent_conn.poll():
        return parent_conn.recv()
    return "ERROR: No output returned from code execution."


def _validate_depends_on_schema(depends_on, tables, table_columns, entity_label, idx):
    import re

    for dep in depends_on:
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*$", dep):
            return (
                False,
                f"[DEPENDENCY VALIDATION ERROR] {entity_label} at index {idx} has invalid depends_on entry: '{dep}'.\n"
                f"Each depends_on entry must be fully qualified as 'table.column'.\n"
                f"Tables available: {sorted(tables)}\n"
                f"Please correct your output to match the schema contract.",
            )
        table, column = dep.split(".")
        if table not in tables:
            return (
                False,
                f"[DEPENDENCY VALIDATION ERROR] {entity_label} at index {idx} references table '{table}' which does not exist.\n"
                f"Available tables: {sorted(tables)}\n"
                f"Please correct your output to match the actual database schema.",
            )
        if column not in table_columns.get(table, set()):
            return (
                False,
                f"[DEPENDENCY VALIDATION ERROR] {entity_label} at index {idx} references column '{column}' in table '{table}' which does not exist.\n"
                f"Available columns in '{table}': {sorted(table_columns[table])}\n"
                f"Please correct your output to match the actual database schema.",
            )
    return (True, None)


def get_save_candidate_features_tool(session_state):
    """
    Returns a function to save candidate features, now with schema validation.
    The tool validates that each depends_on entry is fully qualified and exists in the DB.
    """
    from src.schemas.models import CandidateFeature

    def save_candidate_features(candidate_features_data: list) -> str:
        """
        Validates and saves a list of candidate feature specifications to the session state.
        Each feature MUST conform to the CandidateFeature schema.
        Additionally, each depends_on entry must be a fully qualified column name (table.column), and both the table and column must exist in the database.
        """
        import duckdb

        logger.info(
            f"[TOOL CALL] save_candidate_features called with {len(candidate_features_data)} items."
        )
        validated_features = []
        db_path = getattr(session_state, "db_path", None)
        if not db_path:
            error_message = "[INTERNAL ERROR] No db_path found in session_state."
            logger.error(error_message)
            return error_message
        # Gather schema info
        with duckdb.connect(database=str(db_path), read_only=False) as conn:
            tables = set(row[0] for row in conn.execute("SHOW TABLES").fetchall())
            table_columns = {
                t: set(row[1] for row in conn.execute(f"PRAGMA table_info('{t}')").fetchall())
                for t in tables
            }
        for i, f_data in enumerate(candidate_features_data):
            try:
                feature = CandidateFeature(**f_data)
            except Exception as e:
                error_message = (
                    f"[SCHEMA VALIDATION ERROR] CandidateFeature at index {i} failed validation.\n"
                    f"Input: {f_data}\n"
                    f"Error: {e}\n"
                    "==> ACTION REQUIRED: Each candidate feature must match the schema contract exactly.\n"
                    "Please correct your output."
                )
                logger.error(f"[TOOL ERROR] {error_message}")
                return error_message
            # DRY: Use helper for depends_on validation
            valid, dep_error = _validate_depends_on_schema(
                feature.depends_on, tables, table_columns, "CandidateFeature", i
            )
            if not valid:
                logger.error(f"[TOOL ERROR] {dep_error}")
                return dep_error or "[DEPENDENCY VALIDATION ERROR] Unknown error."
            validated_features.append(feature)
        try:
            session_state.set_candidate_features([f.model_dump() for f in validated_features])
            success_message = f"SUCCESS: Successfully validated and saved {len(validated_features)} candidate features."
            logger.info(f"[TOOL SUCCESS] {success_message}")
            return success_message
        except Exception as e:
            error_message = (
                f"ERROR: Failed to save candidate features after validation. Reason: {e}"
            )
            logger.error(f"[TOOL ERROR] {error_message}")
            return error_message
