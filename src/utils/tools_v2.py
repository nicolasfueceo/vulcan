import os
import json
import duckdb
import networkx as nx
import pandas as pd
from PIL import Image
import google.generativeai as genai
from google.api_core import exceptions
import matplotlib.pyplot as plt

from src.utils.schemas_v2 import Insight, Hypothesis
from src.utils.session_state import SessionState
import subprocess
from pathlib import Path
from typing import Callable, Dict, List

from src.utils.run_utils import get_run_dir

# Configure the Gemini API key
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


class DatabaseTools:
    """A class to encapsulate database-related tools that share a connection."""

    def __init__(self, session_state: SessionState):
        if not session_state.conn:
            raise ValueError("Database connection not found in SessionState.")
        self.conn = session_state.conn

    def get_db_schema(self) -> str:
        """Returns the schema of all tables in the connected database."""
        try:
            tables = self.conn.execute("SHOW TABLES;").fetchall()
            schema_info = "Database Schema:\n\n"
            for table_name in [t[0] for t in tables]:
                schema_info += f"Table: {table_name}\n"
                columns = self.conn.execute(
                    f"PRAGMA table_info('{table_name}');"
                ).fetchall()
                for col in columns:
                    schema_info += f"  - {col[1]} ({col[2]})\n"
                schema_info += "\n"
            return schema_info
        except Exception as e:
            return f"Error getting schema: {e}"

    def create_sql_view(self, view_name: str, sql_query: str) -> str:
        """Creates or replaces a temporary SQL view for the current session."""
        try:
            if ";" in sql_query and not sql_query.strip().endswith(";"):
                return "Error: Multiple SQL statements are not allowed."
            self.conn.execute(f"CREATE OR REPLACE TEMP VIEW {view_name} AS {sql_query}")
            return f"Successfully created temporary SQL View: '{view_name}'"
        except Exception as e:
            return f"Error creating SQL View: {e}"

    def create_graph_from_query(
        self, graph_name: str, node_query: str, edge_query: str
    ) -> str:
        """Creates a NetworkX graph from node and edge queries."""
        try:
            node_df = self.conn.execute(node_query).fetchdf()
            edge_df = self.conn.execute(edge_query).fetchdf()
            if edge_df.shape[1] < 2:
                return "Error: Edge query must return at least two columns."

            G = nx.from_pandas_edgelist(
                edge_df,
                source=edge_df.columns[0],
                target=edge_df.columns[1],
                edge_attr=True,
                create_using=nx.DiGraph(),
            )

            if not node_df.empty:
                node_id_col = node_df.columns[0]
                for _, row in node_df.iterrows():
                    node_id = row[node_id_col]
                    if G.has_node(node_id):
                        for col, value in row.items():
                            if col != node_id_col:
                                G.nodes[node_id][col] = value

            graph_path = f"runtime/graphs/{graph_name}.gml"
            os.makedirs(os.path.dirname(graph_path), exist_ok=True)
            nx.write_gml(G, graph_path)
            return (
                f"Successfully created graph '{graph_name}' and saved to {graph_path}"
            )
        except Exception as e:
            return f"Error creating graph: {e}"

    def execute_python_with_db_connection(self, code: str) -> str:
        """Executes Python code with access to the shared DB connection `conn`."""
        try:
            import sys
            from io import StringIO

            old_stdout = sys.stdout
            redirected_output = sys.stdout = StringIO()
            exec(code, {"conn": self.conn, "pd": pd, "plt": plt, "nx": nx})
            sys.stdout = old_stdout
            return redirected_output.getvalue()
        except Exception as e:
            return f"Error executing Python code: {e}"


# --- Python and Vision Tools ---

_SAVE_PLOT_HELPER = """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import uuid

def save_plot(plt_object, filename=None):
    if filename is None:
        filename = f"plot_{uuid.uuid4().hex}.png"
    
    plots_dir = Path("./plots") 
    plots_dir.mkdir(exist_ok=True)
    
    path = plots_dir / filename
    plt_object.savefig(path)
    print(f"PLOT_PATH:{path.resolve()}")
    plt_object.close()
"""


def python_tool(code: str) -> str:
    """Executes Python code in an isolated process within the run's directory."""

    # Preamble to provide a database connection and the plot helper
    preamble = f"""
import json
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.database import get_db_connection
{_SAVE_PLOT_HELPER}

with get_db_connection() as conn:
"""
    # The user code will be indented to run inside the `with` block
    indented_code = "    " + code.replace("\n", "\n    ")

    code_to_run = f"{preamble}\n{indented_code}"
    run_dir = get_run_dir()
    script_path = run_dir / "temp_script.py"

    with open(script_path, "w") as f:
        f.write(code_to_run)

    try:
        result = subprocess.run(
            ["python", str(script_path.resolve())],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
            cwd=str(run_dir),
        )

        output = f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}"

        return output

    except subprocess.CalledProcessError as e:
        return f"Python script failed:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
    finally:
        if script_path.exists():
            script_path.unlink()


def vision_tool(image_path: str, prompt: str) -> str:
    """Analyzes an image file using a vision model and returns a description."""
    if not genai.context.api_key:
        return "Error: GOOGLE_API_KEY is not configured. Vision tool is disabled."
    try:
        # The path can be absolute as returned by save_plot
        p = Path(image_path)
        if not p.exists():
            # Or relative to the run dir
            p = get_run_dir() / image_path
            if not p.exists():
                return f"Error: Image file not found at '{image_path}'"

        vision_model = genai.GenerativeModel("gemini-pro-vision")
        image = Image.open(p)
        response = vision_model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"Error during vision analysis: {e}"


# --- State Management Tool ---


def get_add_insight_tool(session_state: SessionState) -> Callable:
    """Returns a tool function that is bound to the current session state."""

    def add_insight_to_report(
        title: str,
        finding: str,
        source_representation: str,
        supporting_code: str = None,
        plot_path: str = None,
        plot_interpretation: str = None,
    ) -> str:
        """Logs a structured insight to the session report. Must be called after each finding."""
        try:
            insight = Insight(
                title=title,
                finding=finding,
                source_representation=source_representation,
                supporting_code=supporting_code,
                plot_path=plot_path,
                plot_interpretation=plot_interpretation,
            )
            session_state.add_insight(insight)
            return f"Successfully added insight: '{title}'"
        except Exception as e:
            return f"Error: Could not add insight. Details: {e}"

    return add_insight_to_report


def get_finalize_hypotheses_tool(session_state: SessionState) -> Callable:
    """Returns a tool function that is bound to the current session state."""

    def finalize_hypotheses(hypotheses: List[Dict]) -> str:
        """
        Finalizes and saves the vetted list of hypotheses, concluding the strategy session.
        This tool should only be called by the HypothesisAgent when the team is in agreement.
        """
        if not isinstance(hypotheses, list) or not hypotheses:
            return "Error: You must provide a non-empty list of hypotheses."

        try:
            for h_dict in hypotheses:
                hyp = Hypothesis.parse_obj(h_dict)
                session_state.add_hypothesis(hyp)

            success_message = (
                f"Successfully finalized and saved {len(hypotheses)} hypotheses."
            )
            return f"SUCCESS: {success_message} You can now terminate the conversation."

        except Exception as e:
            return f"Error: Could not finalize hypotheses. Input must be a list of valid Hypothesis objects. Details: {e}"

    return finalize_hypotheses
