#!/usr/bin/env python3
"""
VULCAN Experiment Queue Runner

This script reads a list of experiment jobs from a YAML file (default: experiment_queue.yaml),
sorts them by priority, and runs them sequentially using the VulcanOrchestrator.

Usage:
    python3 scripts/run_queue.py [--jobs <path_to_jobs_file.yaml>] [--output_dir <base_output_directory>]

Job File Format (YAML):
  jobs:
    - name: "Experiment_Name_1"
      config_file: "path/to/specific_config.yaml"  # Optional
      overrides:  # Optional
        llm:
          model_name: "gpt-3.5-turbo"
        experiment:
          max_generations: 5
      priority: 1  # Optional, lower is higher

If 'config_file' is omitted, 'config/config.yaml' is used as the base.
Overrides are applied on top of the loaded configuration.
Jobs without priority or with the same priority are run in file order.
"""

import argparse
import asyncio
import copy
import datetime
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Ensure the src directory is in the Python path
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(WORKSPACE_ROOT / "src"))
sys.path.append(str(WORKSPACE_ROOT))  # For experiment_queue.yaml at root

from vulcan.core.config_manager import ConfigManager
from vulcan.core.orchestrator import VulcanOrchestrator
from vulcan.data.goodreads_loader import GoodreadsDataLoader  # For data_context loading
from vulcan.types import VulcanConfig

# --- Default Paths ---
DEFAULT_JOBS_FILE = WORKSPACE_ROOT / "experiment_queue.yaml"
DEFAULT_BASE_CONFIG_PATH = WORKSPACE_ROOT / "configs" / "config.yaml"
DEFAULT_QUEUE_OUTPUT_DIR = WORKSPACE_ROOT / "experiments" / "queue_runs"


# --- Logger Setup ---
def setup_queue_logger(log_dir: Path, log_level: int = logging.INFO) -> logging.Logger:
    """Sets up a logger for the queue runner."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / "queue_run.log"

    logger = logging.getLogger("VulcanQueueRunner")
    logger.setLevel(log_level)

    # Prevent duplicate handlers if called multiple times (e.g. in tests)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - [Queue] - %(levelname)s - %(message)s")
    )
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - [Queue] - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    return logger


# Global logger instance for the queue
queue_logger: Optional[logging.Logger] = None


# --- Data Loading (Adapted from run_ablation.py) ---
def load_data_context_for_job(
    config: VulcanConfig,
    base_data_path: Path,
    data_config_overrides: Optional[
        Dict[str, Any]
    ] = None,  # Not used here, but kept for signature similarity
) -> Any:
    """
    Loads the data context required for an experiment based on its config.
    """
    global queue_logger
    queue_logger.info("Loading data context for job...")

    # Use data paths and fold info directly from the job's config object
    db_path = (
        base_data_path / "goodreads.db"
    )  # Assuming goodreads.db is the primary source
    splits_dir = base_data_path / "splits"

    # Ensure data config attributes exist, using defaults if necessary
    outer_fold = getattr(config.data, "outer_fold", 0)
    inner_fold = getattr(config.data, "inner_fold", 0)
    sample_size = getattr(
        config.data, "sample_size", None
    )  # Or a default like 1000 if None is problematic

    loader = GoodreadsDataLoader(
        db_path=str(db_path),
        splits_dir=str(splits_dir),
        outer_fold=outer_fold,
        inner_fold=inner_fold,
    )

    data_context = loader.get_data_context(sample_size=sample_size)  # Pass sample_size
    queue_logger.info(
        f"Data context loaded. N Users: {getattr(data_context, 'n_users', 'N/A')}, N Items: {getattr(data_context, 'n_items', 'N/A')}"
    )
    return data_context


# --- Main Experiment Execution Logic (Adapted from run_ablation.py) ---
async def run_single_job(
    job_config: VulcanConfig,
    job_name: str,
    queue_run_output_dir: Path,  # Base output for this queue run
    data_context: Any,
) -> Dict[str, Any]:
    """
    Runs a single experiment job.
    """
    global queue_logger
    queue_logger.info(f"--- Starting Job: {job_name} ---")

    # The job_config is already tailored for this specific job.
    # Override the output directory to be within the queue_run_output_dir
    # The VulcanOrchestrator will create a timestamped subfolder within this.
    job_config.experiment.output_dir = str(queue_run_output_dir)
    # The experiment name in config is less important here as VO uses job_name for folder.
    job_config.experiment.name = (
        job_name  # Ensure this is used if VO relies on it for sub-folder.
    )

    Path(job_config.experiment.output_dir).mkdir(parents=True, exist_ok=True)

    orchestrator = VulcanOrchestrator(job_config)
    job_summary = {
        "job_name": job_name,
        "status": "UNKNOWN",
        "output_dir": "N/A",
        "best_score": None,
        "error": None,
    }

    try:
        await orchestrator.initialize_components()
    except Exception as e:
        queue_logger.error(
            f"Failed to initialize orchestrator components for job {job_name}: {e}",
            exc_info=True,
        )
        job_summary["status"] = "FAILED_INITIALIZATION"
        job_summary["error"] = str(e)
        return job_summary

    try:
        queue_logger.info(
            f"Launching VulcanOrchestrator.start_experiment for job {job_name}..."
        )
        exp_id = await orchestrator.start_experiment(
            experiment_name=job_name,  # This name is used for the timestamped sub-folder
            data_context=data_context,
        )
        queue_logger.info(
            f"Job {job_name} (Experiment ID: {exp_id}) started. Waiting for completion..."
        )

        while orchestrator.get_status().is_running:
            await asyncio.sleep(10)  # Poll every 10 seconds
            queue_logger.debug(f"Job {job_name} is still running...")

        queue_logger.info(f"Job {job_name} (Experiment ID: {exp_id}) finished.")

        # Retrieve results
        history = orchestrator.get_experiment_history()
        run_result = None
        if history:
            for res in reversed(history):
                if res.experiment_id == exp_id:
                    run_result = res
                    break

        if (
            hasattr(orchestrator, "current_experiment_dir_path")
            and orchestrator.current_experiment_dir_path
        ):
            job_summary["output_dir"] = str(orchestrator.current_experiment_dir_path)
        else:  # Fallback if path not directly available
            # Construct path based on convention: queue_run_output_dir / (timestamped job_name folder)
            # This is harder to get exactly without VO providing it. For now, log N/A or base.
            job_summary["output_dir"] = (
                str(job_config.experiment.output_dir)
                + f"/{job_name}_[timestamp_unknown]"
            )

        if run_result:
            queue_logger.info(
                f"Results for job {job_name}: Score={run_result.best_score}, Features={len(run_result.best_features)}"
            )
            job_summary.update(
                {
                    "status": "SUCCEEDED",
                    "best_score": run_result.best_score,
                    "num_best_features": len(run_result.best_features),
                    "execution_time_sec": run_result.execution_time,
                    "total_generations_iterations": run_result.total_iterations,
                }
            )
        else:
            queue_logger.warning(
                f"Could not find result for experiment_id {exp_id} in history for job {job_name}. Check logs in output directory."
            )
            job_summary["status"] = "COMPLETED_NO_RESULT_IN_HISTORY"

    except Exception as e:
        queue_logger.error(
            f"Job {job_name} failed during execution: {e}", exc_info=True
        )
        job_summary["status"] = "FAILED_EXECUTION"
        job_summary["error"] = str(e)
        if (
            hasattr(orchestrator, "current_experiment_dir_path")
            and orchestrator.current_experiment_dir_path
        ):
            job_summary["output_dir"] = str(orchestrator.current_experiment_dir_path)

    finally:
        if orchestrator:
            queue_logger.info(f"Cleaning up orchestrator for job {job_name}...")
            await orchestrator.cleanup()
        queue_logger.info(
            f"--- Finished Job: {job_name} --- Output: {job_summary['output_dir']}"
        )

    return job_summary


async def main_queue(args):
    """Main function to process the experiment queue."""
    global queue_logger

    # Setup base output directory for this queue run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    queue_run_name = f"JobQueueRun_{timestamp}"

    # Prefer user-specified output_dir, then default, then fallback.
    if args.output_dir:
        base_queue_output_dir = Path(args.output_dir)
    else:
        base_queue_output_dir = DEFAULT_QUEUE_OUTPUT_DIR

    queue_run_output_dir = base_queue_output_dir / queue_run_name
    queue_run_output_dir.mkdir(parents=True, exist_ok=True)

    queue_logger = setup_queue_logger(queue_run_output_dir)  # Initialize logger here

    jobs_file_path = Path(args.jobs_file or DEFAULT_JOBS_FILE)
    if not jobs_file_path.exists():
        queue_logger.error(f"Jobs file not found: {jobs_file_path}")
        sys.exit(1)

    try:
        with open(jobs_file_path) as f:
            jobs_data = yaml.safe_load(f)
        if not jobs_data or "jobs" not in jobs_data:
            queue_logger.error(
                f"Invalid jobs file format. Expected a 'jobs' list. File: {jobs_file_path}"
            )
            sys.exit(1)
        job_list = jobs_data["jobs"]
    except Exception as e:
        queue_logger.error(
            f"Error parsing jobs file {jobs_file_path}: {e}", exc_info=True
        )
        sys.exit(1)

    # Sort jobs: by priority (lower is higher), then by original order for ties.
    # Assign a very high default priority for jobs missing it to sort them last, preserving their order.
    max_priority_val = float("inf")
    job_list_with_indices = list(
        enumerate(job_list)
    )  # Keep original index for stable sort

    job_list_with_indices.sort(
        key=lambda item: (item[1].get("priority", max_priority_val), item[0])
    )

    sorted_job_list = [job_def for _, job_def in job_list_with_indices]

    num_jobs = len(sorted_job_list)
    queue_logger.info(
        f"Loaded {num_jobs} jobs from {jobs_file_path}. Output directory for this run: {queue_run_output_dir}"
    )
    queue_logger.info("Execution order after priority sorting:")
    for i, job_def in enumerate(sorted_job_list):
        queue_logger.info(
            f"  {i + 1}. {job_def['name']} (Priority: {job_def.get('priority', 'N/A')})"
        )

    # Load the absolute base configuration once
    # This is config/config.yaml. Individual jobs can override this.
    base_vulcan_config_path = Path(args.base_config or DEFAULT_BASE_CONFIG_PATH)
    if not base_vulcan_config_path.exists():
        queue_logger.error(
            f"Base VULCAN configuration file not found: {base_vulcan_config_path}"
        )
        # Fallback: try to create a default config object if file is missing (might be risky)
        queue_logger.warning("Attempting to use default VulcanConfig object as base.")
        base_vulcan_config = VulcanConfig()
    else:
        try:
            base_config_manager = ConfigManager(
                config_path=str(base_vulcan_config_path)
            )
            base_vulcan_config = base_config_manager.config
            queue_logger.info(
                f"Loaded base VULCAN configuration from: {base_vulcan_config_path}"
            )
        except Exception as e:
            queue_logger.error(
                f"Error loading base VULCAN configuration from {base_vulcan_config_path}: {e}",
                exc_info=True,
            )
            sys.exit(1)

    # Data path for loading data context (can be overridden by job's base_path if needed)
    # TODO: Make this more flexible if jobs can specify different base data paths.
    # For now, assume one common data root.
    data_root_path = WORKSPACE_ROOT / "data"
    if (
        hasattr(base_vulcan_config.data, "base_path")
        and base_vulcan_config.data.base_path
    ):
        configured_base_path = Path(base_vulcan_config.data.base_path)
        data_root_path = (
            configured_base_path
            if configured_base_path.is_absolute()
            else WORKSPACE_ROOT / configured_base_path
        )

    succeeded_jobs = 0
    failed_jobs = 0
    job_summaries: List[Dict[str, Any]] = []

    for i, job_def in enumerate(sorted_job_list):
        job_name = job_def["name"]
        queue_logger.info(f"--- Preparing Job {i + 1}/{num_jobs}: {job_name} ---")

        current_job_config = copy.deepcopy(base_vulcan_config)

        # 1. Load job-specific config file if provided
        if "config_file" in job_def and job_def["config_file"]:
            job_config_path_str = job_def["config_file"]
            job_config_path = Path(job_config_path_str)
            if not job_config_path.is_absolute():
                job_config_path = WORKSPACE_ROOT / job_config_path_str

            if job_config_path.exists():
                try:
                    job_specific_manager = ConfigManager(
                        config_path=str(job_config_path)
                    )
                    current_job_config = job_specific_manager.config  # Overwrites base
                    queue_logger.info(
                        f"Loaded job-specific config for '{job_name}' from: {job_config_path}"
                    )
                except Exception as e:
                    queue_logger.error(
                        f"Error loading config file {job_config_path} for job '{job_name}': {e}. Using previously loaded base/default.",
                        exc_info=True,
                    )
            else:
                queue_logger.warning(
                    f"Job-specific config file {job_config_path} not found for '{job_name}'. Using previously loaded base/default."
                )

        # 2. Apply overrides
        if "overrides" in job_def and job_def["overrides"]:
            try:
                # The VulcanConfig.update method handles nested updates
                current_job_config = current_job_config.update(**job_def["overrides"])
                queue_logger.info(f"Applied overrides for job '{job_name}'.")
            except Exception as e:
                queue_logger.error(
                    f"Error applying overrides for job '{job_name}': {e}. Configuration might be inconsistent.",
                    exc_info=True,
                )

        # Load data context for this specific job's configuration (folds, sample_size)
        try:
            # It's important that current_job_config.data reflects the correct outer/inner folds for this job
            job_data_context = load_data_context_for_job(
                current_job_config, data_root_path
            )
        except FileNotFoundError as e_data:
            queue_logger.error(
                f"Data loading failed for job '{job_name}': Missing split files. Error: {e_data}. Skipping job.",
                exc_info=True,
            )
            job_summary = {
                "job_name": job_name,
                "status": "FAILED_DATA_LOADING",
                "error": str(e_data),
                "output_dir": "N/A",
            }
            job_summaries.append(job_summary)
            failed_jobs += 1
            continue  # Skip to next job
        except Exception as e_data_other:
            queue_logger.error(
                f"Data loading failed for job '{job_name}' with an unexpected error: {e_data_other}. Skipping job.",
                exc_info=True,
            )
            job_summary = {
                "job_name": job_name,
                "status": "FAILED_DATA_LOADING",
                "error": str(e_data_other),
                "output_dir": "N/A",
            }
            job_summaries.append(job_summary)
            failed_jobs += 1
            continue

        job_run_start_time = time.time()
        job_result = await run_single_job(
            job_config=current_job_config,
            job_name=job_name,
            queue_run_output_dir=queue_run_output_dir,
            data_context=job_data_context,
        )
        job_run_duration = time.time() - job_run_start_time
        job_result["queue_run_duration_sec"] = job_run_duration
        job_summaries.append(job_result)

        if (
            "SUCCEEDED" in job_result["status"] or "COMPLETED" in job_result["status"]
        ):  # Consider COMPLETED_NO_RESULT_IN_HISTORY as a form of success
            succeeded_jobs += 1
            queue_logger.info(
                f"Job '{job_name}' completed. Status: {job_result['status']}. Score: {job_result.get('best_score', 'N/A')}. Time: {job_run_duration:.2f}s. Results: {job_result['output_dir']}"
            )
        else:
            failed_jobs += 1
            queue_logger.error(
                f"Job '{job_name}' failed. Status: {job_result['status']}. Error: {job_result.get('error', 'Unknown')}. Time: {job_run_duration:.2f}s. Partial results (if any): {job_result['output_dir']}"
            )

        # Small delay between jobs if needed
        await asyncio.sleep(5)

    queue_logger.info("===== Queue Run Summary =====")
    queue_logger.info(f"Total jobs processed: {num_jobs}")
    queue_logger.info(f"Succeeded: {succeeded_jobs}")
    queue_logger.info(f"Failed: {failed_jobs}")
    queue_logger.info("Detailed job results:")
    for summary in job_summaries:
        queue_logger.info(
            f"  - Job: {summary['job_name']}, Status: {summary['status']}, "
            f"Score: {summary.get('best_score', 'N/A')}, Output: {summary['output_dir']}, "
            f"Error: {summary.get('error', 'None')}"
        )
    queue_logger.info(f"All queue run outputs are under: {queue_run_output_dir}")
    queue_logger.info("Queue execution finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VULCAN Experiment Queue Runner")
    parser.add_argument(
        "--jobs",
        type=str,
        default=None,  # Will use DEFAULT_JOBS_FILE if None
        help=f"Path to the YAML jobs file. (Default: {DEFAULT_JOBS_FILE})",
    )
    parser.add_argument(
        "--base_config",
        type=str,
        default=None,  # Will use DEFAULT_BASE_CONFIG_PATH
        help=f"Path to the base VULCAN YAML configuration file. (Default: {DEFAULT_BASE_CONFIG_PATH})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,  # Will use DEFAULT_QUEUE_OUTPUT_DIR
        help=f"Base directory for all queue run outputs. A timestamped subfolder will be created here. (Default: {DEFAULT_QUEUE_OUTPUT_DIR})",
    )

    cli_args = parser.parse_args()

    try:
        asyncio.run(main_queue(cli_args))
    except KeyboardInterrupt:
        if queue_logger:
            queue_logger.info("Queue execution interrupted by user.")
        else:
            print("Queue execution interrupted by user.")
        sys.exit(0)
    except Exception as e:
        if queue_logger:
            queue_logger.critical(
                f"Unhandled critical error in queue runner: {e}", exc_info=True
            )
        else:
            print(f"Unhandled critical error in queue runner: {e}")
        sys.exit(1)
