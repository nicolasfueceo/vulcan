import json
import subprocess
from pathlib import Path

import duckdb

from src.utils.run_utils import get_run_dir

# Database path constant
DB_PATH = "/Users/nicolasdhnr/Documents/VULCAN/data/goodreads_curated.duckdb"

# The preamble provides the helper functions and the database connection
# that will be available to the agent's code.
_PYTHON_TOOL_PREAMBLE = f"""
import pandas as pd
import duckdb
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os
import json
from datetime import datetime

# Use absolute path to the database with read-write access (orchestrator connection is closed)
DB_PATH = "{DB_PATH}"

# Set better plotting defaults
plt.style.use('default')
sns.set_palette("husl")

def save_plot(filename: str):
    '''Saves the current matplotlib plot to a file in the run's plot directory with optimized settings.'''
    plots_dir = Path("./plots") # It will be executed from the run directory
    plots_dir.mkdir(exist_ok=True)
    
    # Ensure the filename ends with .png, but don't add it if it already does.
    if not filename.lower().endswith('.png'):
        filename += '.png'
        
    path = plots_dir / filename
    
    # Apply better plot formatting
    plt.tight_layout()
    
    # Auto-set reasonable axis limits if not already set
    ax = plt.gca()
    if hasattr(ax, 'get_xlim'):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # Only adjust if using default limits
        if xlim == (0.0, 1.0) or any(abs(x) > 1e6 for x in xlim):
            # Get data bounds and add small margin
            try:
                lines = ax.get_lines()
                if lines:
                    xdata = np.concatenate([line.get_xdata() for line in lines])
                    if len(xdata) > 0:
                        x_margin = (np.max(xdata) - np.min(xdata)) * 0.05
                        ax.set_xlim(np.min(xdata) - x_margin, np.max(xdata) + x_margin)
            except:
                pass
    
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close() # Close the plot to free memory
    print(f"PLOT_SAVED:{{path.resolve()}}") # Use a special token to signal a plot was saved
    return str(path.resolve())

def create_analysis_view(view_name: str, sql_query: str):
    '''Creates a permanent view for analysis and tracks it for cleanup.'''
    # Use a non-read-only connection just for view creation
    with duckdb.connect(database=str(DB_PATH), read_only=False) as write_conn:
        # Create the view
        full_sql = f"CREATE OR REPLACE VIEW {{view_name}} AS {{sql_query}}"
        write_conn.execute(full_sql)
        
        # Track the view for cleanup
        views_file = Path("./generated_views.json")
        if views_file.exists():
            with open(views_file, 'r') as f:
                views_data = json.load(f)
        else:
            views_data = {{"views": []}}
        
        views_data["views"].append({{
            "name": view_name,
            "sql": full_sql,
            "created_at": datetime.now().isoformat()
        }})
        
        with open(views_file, 'w') as f:
            json.dump(views_data, f, indent=2)
            
        print(f"VIEW_CREATED:{{view_name}}")
        return f"Successfully created view: {{view_name}}"

def analyze_and_plot(data_df, title="Data Analysis", x_col=None, y_col=None, plot_type="auto"):
    '''Helper function to both analyze data numerically AND create bounded visualizations.'''
    print(f"\\n=== NUMERICAL ANALYSIS: {{title}} ===")
    print(f"Dataset shape: {{data_df.shape}}")
    print(f"\\nSummary statistics:")
    print(data_df.describe())
    
    if len(data_df.columns) > 1:
        print(f"\\nCorrelation matrix:")
        numeric_cols = data_df.select_dtypes(include=[np.number])
        if len(numeric_cols.columns) > 1:
            print(numeric_cols.corr())
    
    # Create visualization with bounded axes
    plt.figure(figsize=(10, 6))
    
    if plot_type == "scatter" and x_col and y_col:
        plt.scatter(data_df[x_col], data_df[y_col], alpha=0.6)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        # Set bounded limits
        x_min, x_max = data_df[x_col].min(), data_df[y_col].max()
        y_min, y_max = data_df[y_col].min(), data_df[y_col].max()
        x_margin = (x_max - x_min) * 0.05
        y_margin = (y_max - y_min) * 0.05
        plt.xlim(x_min - x_margin, x_max + x_margin)
        plt.ylim(y_min - y_margin, y_max + y_margin)
    elif plot_type == "hist" or (plot_type == "auto" and len(data_df.columns) == 1):
        col = data_df.columns[0] if not x_col else x_col
        data_df[col].hist(bins=30, alpha=0.7)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        # Set bounded x limits
        x_min, x_max = data_df[col].min(), data_df[col].max()
        x_margin = (x_max - x_min) * 0.05
        plt.xlim(x_min - x_margin, x_max + x_margin)
    
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    return title.lower().replace(" ", "_").replace(":", "")

# The user's code will run inside this 'with' block with read-write access
with duckdb.connect(database=str(DB_PATH), read_only=False) as conn:
"""


def execute_python(code: str) -> str:
    """
    Executes a string of Python code in a sandboxed environment with a pre-configured
    database connection (`conn`) and helper functions (`save_plot`).

    Args:
        code: The string of Python code to execute.

    Returns:
        A string containing the captured STDOUT and STDERR from the execution.
    """

    # Always use the standard preamble for now (simplified approach)
    indented_code = "    " + code.replace("\n", "\n    ")
    full_script = f"{_PYTHON_TOOL_PREAMBLE}\n{indented_code}"

    # Use the run-specific directory for sandboxing
    run_dir = get_run_dir()
    script_path = run_dir / "temp_agent_script.py"

    with open(script_path, "w") as f:
        f.write(full_script)

    try:
        # Execute the script in a new process for isolation
        result = subprocess.run(
            ["python", str(script_path)],
            capture_output=True,
            text=True,
            check=True,  # This will raise CalledProcessError if the script fails (exit code != 0)
            timeout=120,  # Add a timeout to prevent runaway processes
            cwd=str(run_dir),  # Set the working directory to the run directory
        )
        output = f"STDOUT:\n{result.stdout}"
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        return output

    except subprocess.CalledProcessError as e:
        # This block catches script failures (e.g., Python exceptions)
        return f"EXECUTION FAILED:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
    except subprocess.TimeoutExpired:
        return "EXECUTION FAILED: Code execution timed out after 120 seconds."
    finally:
        # Clean up the temporary script
        if script_path.exists():
            script_path.unlink()


def cleanup_analysis_views(run_dir: Path):
    """Cleans up any database views created during a run."""
    views_file = run_dir / "generated_views.json"
    if not views_file.exists():
        print("No views to clean up.")
        return

    with open(views_file, "r") as f:
        views_data = json.load(f)

    views_to_drop = [view["name"] for view in views_data["views"]]

    if not views_to_drop:
        print("No views to clean up.")
        return

    try:
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
