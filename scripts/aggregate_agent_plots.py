# scripts/aggregate_agent_plots.py
"""
Aggregates all files from each run_{id}/plots directory in /runtime/runs into report/agent_plots,
prefixing filenames with their run id to avoid collisions. Robust to missing or empty plots directories.
"""
import os
import shutil
from pathlib import Path

def aggregate_agent_plots(
    runs_dir: Path = Path("/Users/nicolasdhnr/Documents/VULCAN/runtime/runs"),
    output_dir: Path = Path("/Users/nicolasdhnr/Documents/VULCAN/report/agent_plots")
):
    output_dir.mkdir(parents=True, exist_ok=True)
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
        plots_dir = run_dir / "plots"
        if not plots_dir.exists() or not plots_dir.is_dir():
            continue
        for plot_file in plots_dir.iterdir():
            if not plot_file.is_file():
                continue
            dest_filename = f"{run_dir.name}_{plot_file.name}"
            dest_path = output_dir / dest_filename
            shutil.copy2(plot_file, dest_path)
            print(f"Copied {plot_file} -> {dest_path}")

if __name__ == "__main__":
    aggregate_agent_plots()
