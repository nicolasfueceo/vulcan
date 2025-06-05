#!/usr/bin/env python3
"""
Script to run ablation studies for the VULCAN feature engineering pipeline.

This script allows for systematic variation of key components and parameters
to assess their impact on performance. It can run multiple experiments
sequentially, save their results in uniquely named folders, and provide
a summary of outcomes.

Usage:
    python3 scripts/run_ablation.py [--study <study_name>] [--base_config <path_to_config.yaml>]

Examples:
    python3 scripts/run_ablation.py --study model_ablation
    python3 scripts/run_ablation.py --study all
    python3 scripts/run_ablation.py --study prompt_memory --base_config configs/custom_base.yaml

Supported studies:
    - model_ablation: Vary LLM models.
    - prompt_memory_ablation: Vary prompt memory length/format.
    - clustering_ablation: Vary clustering algorithms.
    - scoring_ablation: Vary feature evaluation scoring methods.
    - all: Run all defined ablation studies sequentially.
"""

import argparse
import asyncio
import copy
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure the src directory is in the Python path
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
sys.path.append(str(WORKSPACE_ROOT / "src"))

from vulcan.core.config_manager import ConfigManager
from vulcan.core.orchestrator import VulcanOrchestrator
from vulcan.data.goodreads_loader import (
    GoodreadsDataLoader,  # Assuming this is the data loader
)
from vulcan.types import VulcanConfig

# Configure basic logging for the script itself
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Default base config path - adjust if your default config is elsewhere
DEFAULT_BASE_CONFIG_PATH = WORKSPACE_ROOT / "configs" / "config.yaml"
DEFAULT_ABLATION_OUTPUT_DIR = WORKSPACE_ROOT / "experiments" / "ablations"
DEFAULT_ABLATION_RESULTS_DIR = WORKSPACE_ROOT / "ablation_results"


# --- Data Loading (Placeholder - adapt to your project) ---
def load_data_context(
    config: VulcanConfig,
    base_data_path: Path,
    data_config_overrides: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Loads the data context required for an experiment.
    This is a placeholder and might need significant adjustments
    based on how your project handles data loading and splitting (e.g., for folds).
    """
    logger.info("Loading data context...")

    # Example: using GoodreadsDataLoader. Adapt as needed.
    # These paths might come from config or be fixed.
    db_path = base_data_path / "goodreads.db"  # Example
    splits_dir = base_data_path / "splits"  # Example

    data_loader_params = {
        "db_path": str(db_path),
        "splits_dir": str(splits_dir),
        "outer_fold": config.data.outer_fold,  # Assuming these exist in your VulcanConfig.data
        "inner_fold": config.data.inner_fold,
    }
    if data_config_overrides:
        data_loader_params.update(data_config_overrides)

    loader = GoodreadsDataLoader(
        db_path=data_loader_params["db_path"],
        splits_dir=data_loader_params["splits_dir"],
        outer_fold=data_loader_params["outer_fold"],
        inner_fold=data_loader_params["inner_fold"],
    )

    sample_size = config.data.sample_size  # Assuming this exists
    if data_config_overrides and "sample_size" in data_config_overrides:
        sample_size = data_config_overrides["sample_size"]

    data_context = loader.get_data_context(sample_size=sample_size)
    logger.info(
        f"Data context loaded. N Users: {getattr(data_context, 'n_users', 'N/A')}, N Items: {getattr(data_context, 'n_items', 'N/A')}"
    )
    return data_context


# --- Experiment Execution ---
async def run_single_experiment(
    base_config: VulcanConfig,
    experiment_name: str,
    config_overrides: Dict[str, Any],
    ablation_study_name: str,
    output_base_dir: Path,
    data_context: Any,
) -> Optional[Dict[str, Any]]:
    """
    Runs a single experiment with the given configuration.
    """
    logger.info(
        f"--- Starting Experiment: {experiment_name} (Study: {ablation_study_name}) ---"
    )

    run_config = copy.deepcopy(base_config)

    # Apply specific overrides for this experiment
    # The config_overrides is a dictionary like: {"llm.model_name": "new_model", "llm.provider": "openai"}
    if config_overrides:
        for path_str, value in config_overrides.items():
            try:
                obj = run_config
                parts = path_str.split(".")
                for i, part in enumerate(parts):
                    if i == len(parts) - 1:
                        if hasattr(obj, part):
                            setattr(obj, part, value)
                            logger.debug(f"Set config: {path_str} = {value}")
                        else:
                            logger.warning(
                                f"Attribute {part} not found in {obj} for path {path_str}"
                            )
                            # Optionally raise error or handle dicts if config can be mixed Pydantic/dict
                    else:
                        if hasattr(obj, part):
                            obj = getattr(obj, part)
                        else:
                            logger.warning(
                                f"Intermediate attribute {part} not found in {obj} for path {path_str}"
                            )
                            # Break or handle dicts
                            break
            except Exception as e:
                logger.error(
                    f"Failed to apply config override {path_str}={value}: {e}",
                    exc_info=True,
                )
                # Decide if a single failed override should stop the experiment
                # return None # Or collect errors

    # Override output directory to group ablation runs
    run_config.experiment.output_dir = str(output_base_dir / ablation_study_name)
    Path(run_config.experiment.output_dir).mkdir(parents=True, exist_ok=True)

    # Setup main VULCAN logging (console and file for the experiment)
    # The orchestrator will set up its own file logger.
    # setup_logging(run_config.logging) # This might be for the script itself, orchestrator handles experiment logs.

    orchestrator = VulcanOrchestrator(run_config)

    # Initialize components (this was shown in cli.py)
    # This part needs to be confirmed with VulcanOrchestrator's API.
    # It might not need explicit results_manager or data_context here if passed to start_experiment.
    try:
        await orchestrator.initialize_components()  # Add data_context if needed by init
    except Exception as e:
        logger.error(
            f"Failed to initialize orchestrator components for {experiment_name}: {e}",
            exc_info=True,
        )
        return None

    experiment_summary = None
    try:
        logger.info(f"Launching orchestrator.start_experiment for {experiment_name}...")
        # The orchestrator internally handles naming with timestamps.
        # We provide the descriptive base name.
        # The `results_manager` is handled internally by VulcanOrchestrator or needs to be passed.
        # Based on prior use, it seems internally managed if not given.
        exp_id = await orchestrator.start_experiment(
            experiment_name=experiment_name,
            # config_overrides are already applied to run_config
            data_context=data_context,
        )
        logger.info(
            f"Experiment {experiment_name} (ID: {exp_id}) started. Waiting for completion..."
        )

        # Wait for the experiment to complete
        while orchestrator.get_status().is_running:
            await asyncio.sleep(5)  # Poll every 5 seconds
            # Add a timeout maybe?
            logger.debug(f"Experiment {experiment_name} is still running...")

        logger.info(f"Experiment {experiment_name} (ID: {exp_id}) finished.")

        # Retrieve results
        # The orchestrator.get_experiment_history() returns a list of ExperimentResult
        history = orchestrator.get_experiment_history()
        if history:
            # Assuming the last experiment in history is the one just run
            # This could be risky if orchestrator is reused without reset across script runs (not the case here per experiment)
            # Or, better, find by exp_id if orchestrator stores it that way.
            # For now, relying on last entry for this specific orchestrator instance.
            # Let's find the result by experiment_id to be safer.
            # However, start_experiment returns an ID, but get_experiment_history stores ExperimentResult objects.
            # Need to check how to match them or if history is reset/scoped.
            # The current orchestrator's history will only contain the experiments it ran.

            # The `_run_experiment` in orchestrator creates ExperimentResult and appends to self._experiment_history
            # So the last one should be correct FOR THIS orchestrator instance.

            # Let's check how the experiment_id from start_experiment relates to ExperimentResult.experiment_id
            # VulcanOrchestrator sets self._experiment_id, which is used in ExperimentResult.

            # Find the result corresponding to exp_id
            # This might be complicated if orchestrator runs multiple things without clearing history.
            # For now, assuming the LAST history entry for THIS orchestrator instance corresponds to THIS run.
            run_result = None
            for res in reversed(history):  # Check newest first
                if res.experiment_id == exp_id:
                    run_result = res
                    break

            if run_result:
                logger.info(
                    f"Results for {experiment_name}: Score={run_result.best_score}, Features={len(run_result.best_features)}"
                )
                experiment_summary = {
                    "experiment_name": experiment_name,
                    "ablation_study": ablation_study_name,
                    "config_overrides_applied": config_overrides,
                    "best_score": run_result.best_score,
                    "num_best_features": len(run_result.best_features),
                    "execution_time_sec": run_result.execution_time,
                    "total_generations_iterations": run_result.total_iterations,
                    # The run_config.experiment.output_dir is the base for the study,
                    # The actual folder will have a timestamp. Need to get this from orchestrator or results.
                    # The `current_experiment_dir_path` in orchestrator is the one.
                    "output_dir": str(orchestrator.current_experiment_dir_path)
                    if hasattr(orchestrator, "current_experiment_dir_path")
                    else "N/A",
                }
            else:
                logger.warning(
                    f"Could not find result for experiment_id {exp_id} in history for {experiment_name}."
                )
                # Try to load from results.json as a fallback
                # This needs to know the exact path, which orchestrator.current_experiment_dir_path should provide
                if hasattr(orchestrator, "current_experiment_dir_path"):
                    results_json_path = (
                        orchestrator.current_experiment_dir_path / "results.json"
                    )  # Or whatever the file is named
                    if results_json_path.exists():
                        try:
                            with open(results_json_path) as f:
                                saved_results = json.load(f)
                            # Extract relevant metrics from saved_results
                            # This depends on the structure of results.json
                            # Example:
                            best_score_from_file = saved_results.get(
                                "experiment_results", {}
                            ).get("best_score")
                            if best_score_from_file is not None:
                                experiment_summary = {
                                    "experiment_name": experiment_name,
                                    "ablation_study": ablation_study_name,
                                    "config_overrides_applied": config_overrides,
                                    "best_score": best_score_from_file,
                                    "output_dir": str(
                                        orchestrator.current_experiment_dir_path
                                    ),
                                    # Add other fields as available
                                }
                                logger.info(
                                    f"Loaded results from {results_json_path} for {experiment_name}"
                                )
                            else:
                                logger.error(
                                    f"results.json found for {experiment_name} but key fields missing."
                                )
                        except Exception as e_json:
                            logger.error(
                                f"Error loading results.json for {experiment_name}: {e_json}"
                            )
                    else:
                        logger.error(
                            f"results.json not found for {experiment_name} at {results_json_path}"
                        )
                else:
                    logger.error(
                        f"Cannot determine output directory for {experiment_name} to load results.json."
                    )

        else:
            logger.warning(
                f"No history found after running experiment {experiment_name}."
            )

    except Exception as e:
        logger.error(f"Experiment {experiment_name} failed: {e}", exc_info=True)
        # Store partial failure info if possible
        experiment_summary = {
            "experiment_name": experiment_name,
            "ablation_study": ablation_study_name,
            "status": "FAILED",
            "error": str(e),
            "config_overrides_applied": config_overrides,
            "output_dir": str(orchestrator.current_experiment_dir_path)
            if hasattr(orchestrator, "current_experiment_dir_path")
            else "N/A",
        }
    finally:
        if orchestrator:  # Ensure orchestrator exists before cleanup
            logger.info(f"Cleaning up orchestrator for experiment {experiment_name}...")
            await orchestrator.cleanup()
        logger.info(f"--- Finished Experiment: {experiment_name} ---")

    return experiment_summary


# --- Ablation Study Definitions ---
def get_llm_model_variations(base_config: VulcanConfig) -> List[Dict[str, Any]]:
    """Defines LLM model variations for ablation."""
    variations = []
    # Example: ensure these models are supported by your config and LLM enum/provider logic
    # Also, ensure provider is correctly set (e.g., config.llm.provider)
    supported_models = {
        "gpt-4": {"provider": "openai"},
        "gpt-3.5-turbo": {"provider": "openai"},
        "claude-3-opus-20240229": {
            "provider": "anthropic"
        },  # Example, verify actual name
        # "claude-v1": {"provider": "anthropic"}, # Replace with actual supported Claude model
    }

    for model_name, model_details in supported_models.items():
        # This assumes VulcanConfig has llm.model_name and llm.provider
        # We need to create two overrides: one for model_name, one for provider.
        # The run_single_experiment needs to handle multiple overrides or a dict of overrides.
        # For now, let's make it a list of single overrides. This will run one experiment per override.
        # This is not ideal. Better to pass a dict of overrides like {'llm.model_name': 'x', 'llm.provider': 'y'}
        # The current run_single_experiment's config_overrides expects {"path": "key.path", "value": val}
        # This needs to be generalized. Let's assume for now we make multiple calls or adjust run_single_experiment.

        # For now, let's assume `config_overrides` in `run_single_experiment` can take a list of these dicts
        # or the update logic is smarter.
        # Let's design `config_overrides` to be a DICT of path -> value

        variations.append(
            {
                "name": f"model_{model_name.replace('-', '_').replace('.', '_')}",
                "config_changes": {
                    "llm.model_name": model_name,
                    "llm.provider": model_details["provider"],
                },
            }
        )
    return variations


def get_prompt_memory_variations(base_config: VulcanConfig) -> List[Dict[str, Any]]:
    """Defines prompt memory variations. Requires changes in FeatureAgent and config."""
    variations = []
    # Config path: llm.prompt_memory_type -> Literal["full", "limited", "none"]
    # Config path: llm.prompt_memory_max_features -> Optional[int]

    variations.append(
        {
            "name": "prompt_full_memory",
            "config_changes": {"llm.prompt_memory_type": "full"},
        }
    )
    variations.append(
        {
            "name": "prompt_limited_memory_5",
            "config_changes": {
                "llm.prompt_memory_type": "limited",
                "llm.prompt_memory_max_features": 5,
            },
        }
    )
    variations.append(
        {
            "name": "prompt_limited_memory_10",
            "config_changes": {
                "llm.prompt_memory_type": "limited",
                "llm.prompt_memory_max_features": 10,
            },
        }
    )
    variations.append(
        {
            "name": "prompt_no_memory",
            "config_changes": {"llm.prompt_memory_type": "none"},
        }
    )
    # logger.warning("Prompt memory ablation not yet fully implemented. Requires config and FeatureAgent changes.")
    # Commenting out warning as we will implement this.
    return variations


def get_clustering_variations(base_config: VulcanConfig) -> List[Dict[str, Any]]:
    """Defines clustering algorithm variations. Requires changes in evaluator and config."""
    variations = []
    # Config path: evaluation.clustering_config.algorithm -> Literal["kmeans", "hierarchical", "dbscan"]
    # Note: EvaluationConfig also has `clustering_method` - ensure clarity on which is used.
    # This targets the newly added `algorithm` field in `ClusteringConfig`.

    algorithms = [
        "kmeans",
        "hierarchical",
        "dbscan",
    ]  # Add more if supported by implementation
    for alg in algorithms:
        variations.append(
            {
                "name": f"cluster_{alg}",
                "config_changes": {"evaluation.clustering_config.algorithm": alg},
            }
        )

    # logger.warning("Clustering ablation not yet fully implemented. Requires config and evaluator changes.")
    return variations


def get_scoring_variations(base_config: VulcanConfig) -> List[Dict[str, Any]]:
    """Defines scoring method variations. Requires changes in evaluator and config."""
    variations = []
    # Config path: evaluation.scoring_mode -> Literal["cluster", "recommender"]
    # Config path for cluster metrics: evaluation.clustering_config.metric -> str (e.g., "silhouette", "calinski_harabasz", etc.)

    # Variation 1: Different cluster metrics (when scoring_mode = "cluster")
    # Ensure these metrics are supported by your evaluation logic.
    cluster_metrics = (
        base_config.evaluation.metrics
    )  # Use metrics defined in base_config as a source
    if not cluster_metrics:  # Fallback if not defined
        cluster_metrics = ["silhouette", "calinski_harabasz", "davies_bouldin"]

    for metric in cluster_metrics:
        variations.append(
            {
                "name": f"scoring_cluster_{metric.replace('_score', '')}",  # Cleaner name
                "config_changes": {
                    "evaluation.scoring_mode": "cluster",
                    "evaluation.clustering_config.metric": metric,
                },
            }
        )

    # Variation 2: Switch to recommender score (if implemented)
    variations.append(
        {
            "name": "scoring_recommender_mode",  # Generic name, specific metric handled by recommender logic
            "config_changes": {"evaluation.scoring_mode": "recommender"},
        }
    )
    # logger.warning("Scoring ablation not yet fully implemented. Requires config and evaluator changes.")
    return variations


# --- Summary and Plotting ---
def save_summary_results(
    results: List[Dict[str, Any]], study_name: str, output_dir: Path
):
    """Saves summary of ablation study to a CSV file."""
    if not results:
        logger.warning(f"No results to save for study {study_name}.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / f"{study_name}_summary.csv"

    # Dynamically determine headers from the first result, maintaining order if possible
    # Fallback if results is empty or items are malformed.
    headers = (
        list(results[0].keys()) if results and isinstance(results[0], dict) else []
    )
    if not headers:  # Fallback static headers if dynamic fails
        headers = ["experiment_name", "best_score", "status", "error"]

    try:
        with open(summary_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for result in results:
                # Ensure all results have all headers, fill with N/A if missing
                row_to_write = {header: result.get(header, "N/A") for header in headers}
                writer.writerow(row_to_write)
        logger.info(f"Ablation summary saved to: {summary_file}")
    except OSError as e:
        logger.error(f"Failed to write summary CSV {summary_file}: {e}")
    except Exception as e_csv:
        logger.error(
            f"An unexpected error occurred while writing summary CSV {summary_file}: {e_csv}"
        )


def plot_comparison(
    results: List[Dict[str, Any]],
    study_name: str,
    metric_to_plot: str,
    output_dir: Path,
):
    """Generates a bar plot comparing a given metric across experiment variations."""
    if not results:
        logger.warning(f"No results to plot for study {study_name}.")
        return

    try:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")  # Use non-interactive backend

        # Filter out failed experiments or those missing the metric
        valid_results = [
            r
            for r in results
            if r.get("status") != "FAILED"
            and metric_to_plot in r
            and r[metric_to_plot] is not None
        ]
        if not valid_results:
            logger.warning(
                f"No valid data to plot for metric '{metric_to_plot}' in study {study_name}."
            )
            return

        labels = [r["experiment_name"] for r in valid_results]
        values = [float(r[metric_to_plot]) for r in valid_results]  # Ensure numeric

        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, values)
        plt.ylabel(metric_to_plot.replace("_", " ").title())
        plt.xlabel("Experiment Variation")
        plt.title(
            f"{study_name.replace('_', ' ').title()} Comparison: {metric_to_plot.replace('_', ' ').title()}"
        )
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Add labels on bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval,
                f"{yval:.3f}",
                va="bottom" if yval >= 0 else "top",
                ha="center",
            )

        plot_file = output_dir / f"{study_name}_{metric_to_plot}_comparison.png"
        plt.savefig(plot_file)
        plt.close()
        logger.info(f"Comparison plot saved to: {plot_file}")

    except ImportError:
        logger.warning(
            "Matplotlib not installed. Skipping plot generation. `pip install matplotlib` to enable plots."
        )
    except Exception as e:
        logger.error(f"Failed to generate plot for {study_name}: {e}", exc_info=True)


# --- Main Orchestration ---
async def run_ablation_study(
    study_name: str,
    get_variations_func: callable,
    base_config: VulcanConfig,
    ablation_output_dir: Path,
    results_summary_dir: Path,
    data_context: Any,
):
    """Runs a specific ablation study (e.g., model ablation)."""
    logger.info(f"===== Starting Ablation Study: {study_name} =====")
    variations = get_variations_func(base_config)
    if not variations:
        logger.warning(f"No variations defined for study: {study_name}. Skipping.")
        return []

    all_results_for_study = []
    for variation_params in variations:
        exp_name = (
            f"abl_{study_name}_{variation_params['name']}"  # Prefixed to avoid clashes
        )

        # Create a deep copy of the base config for this specific run
        current_run_base_config = copy.deepcopy(base_config)

        # Apply general config overrides for this variation
        # The `run_single_experiment` needs to be adapted to take this structure.
        # Let's refine run_single_experiment to accept a dictionary of path:value for overrides.
        # This will be handled inside run_single_experiment.

        # The `config_changes` is a dict like {"llm.model_name": "gpt-4", "llm.provider": "openai"}
        # The `run_single_experiment` current `config_overrides` is a single path-value pair.
        # This needs adjustment.
        # For now, I will assume run_single_experiment is adapted or I will adapt it.
        # Let's assume for now that we pass the whole `variation_params['config_changes']` dict.
        # And run_single_experiment will iterate through it.

        result = await run_single_experiment(
            base_config=current_run_base_config,  # Pass the copied config
            experiment_name=exp_name,
            # This needs to be a dict of {"config.path.key": value}
            config_overrides=variation_params["config_changes"],
            ablation_study_name=study_name,
            output_base_dir=ablation_output_dir,  # e.g., experiments/ablations/
            data_context=data_context,
        )
        if result:
            all_results_for_study.append(result)
        else:
            # Add a placeholder for failed/skipped runs to keep track
            all_results_for_study.append(
                {
                    "experiment_name": exp_name,
                    "ablation_study": study_name,
                    "status": "SKIPPED_OR_ERROR_PREVENTING_RUN",
                    # Add config info if available from variation_params
                    "config_details": variation_params["config_changes"],
                }
            )

        # Small delay to ensure logs are flushed and resources (like GPU if used) are released.
        await asyncio.sleep(10)

    if all_results_for_study:
        save_summary_results(all_results_for_study, study_name, results_summary_dir)
        # Plot comparison for a primary metric, e.g., best_score
        plot_comparison(
            all_results_for_study, study_name, "best_score", results_summary_dir
        )
        # Potentially plot other metrics like execution_time_sec if relevant
        plot_comparison(
            all_results_for_study, study_name, "execution_time_sec", results_summary_dir
        )

    logger.info(f"===== Finished Ablation Study: {study_name} =====")
    return all_results_for_study


async def main(args):
    """Main function to orchestrate ablation studies."""

    # Load base configuration
    base_config_path = Path(args.base_config or DEFAULT_BASE_CONFIG_PATH)
    if not base_config_path.exists():
        logger.error(f"Base configuration file not found: {base_config_path}")
        sys.exit(1)

    try:
        config_manager = ConfigManager(config_path=str(base_config_path))
        base_config = config_manager.config
        logger.info(f"Loaded base configuration from: {base_config_path}")
    except Exception as e:
        logger.error(f"Error loading base configuration: {e}", exc_info=True)
        sys.exit(1)

    # Prepare output directories
    ablation_output_dir = Path(args.ablation_output_dir or DEFAULT_ABLATION_OUTPUT_DIR)
    results_summary_dir = Path(args.results_summary_dir or DEFAULT_ABLATION_RESULTS_DIR)
    ablation_output_dir.mkdir(parents=True, exist_ok=True)
    results_summary_dir.mkdir(parents=True, exist_ok=True)

    # Load Data Context (once, if shared across all ablations)
    # If data context needs to vary per ablation, this needs to move inside study loop.
    # For now, assume a single data context for all.
    # The path to data should be configurable, e.g., from base_config.data.base_path
    # Defaulting to WORKSPACE_ROOT / "data" for now.
    base_data_path = (
        WORKSPACE_ROOT / "data"
    )  # TODO: Make this configurable from base_config
    if (
        hasattr(base_config, "data")
        and hasattr(base_config.data, "base_path")
        and base_config.data.base_path
    ):
        base_data_path = Path(base_config.data.base_path)
        if not base_data_path.is_absolute():
            base_data_path = WORKSPACE_ROOT / base_config.data.base_path

    try:
        # Small sample size for quick testing, configure via base_config for real runs
        # This should be part of the base_config.data.sample_size
        # Let's make it tiny for initial tests of the script itself.
        # Override for dev: base_config.data.sample_size = 100
        # base_config.experiment.max_generations = 2 # For quick test
        # base_config.mcts.max_iterations = 2 # For quick test (if mcts is used by progressive)

        data_context = load_data_context(base_config, base_data_path)
    except Exception as e:
        logger.error(f"Failed to load data context: {e}", exc_info=True)
        sys.exit(1)

    studies_to_run = {
        "model_ablation": get_llm_model_variations,
        "prompt_memory_ablation": get_prompt_memory_variations,
        "clustering_ablation": get_clustering_variations,
        "scoring_ablation": get_scoring_variations,
    }

    all_study_results = {}

    requested_study = args.study.lower()

    if requested_study == "all":
        for study_name, get_variations_func in studies_to_run.items():
            results = await run_ablation_study(
                study_name,
                get_variations_func,
                base_config,
                ablation_output_dir,
                results_summary_dir,
                data_context,
            )
            all_study_results[study_name] = results
    elif requested_study in studies_to_run:
        results = await run_ablation_study(
            requested_study,
            studies_to_run[requested_study],
            base_config,
            ablation_output_dir,
            results_summary_dir,
            data_context,
        )
        all_study_results[requested_study] = results
    else:
        logger.error(
            f"Unknown study: {args.study}. Supported studies are: {list(studies_to_run.keys()) + ['all']}"
        )
        sys.exit(1)

    logger.info("===== All Ablation Studies Completed =====")
    # Optionally, print a final summary of all studies or save a master summary file
    # For now, individual CSVs and plots are generated per study.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VULCAN ablation studies.")
    parser.add_argument(
        "--study",
        type=str,
        default="all",
        help=f"Name of the ablation study to run. Options: {list(['all'] + list(k for k in ['model_ablation', 'prompt_memory_ablation', 'clustering_ablation', 'scoring_ablation']))}. Default is 'all'.",
    )
    parser.add_argument(
        "--base_config",
        type=str,
        default=None,  # Will use DEFAULT_BASE_CONFIG_PATH if None
        help="Path to the base YAML configuration file.",
    )
    parser.add_argument(
        "--ablation_output_dir",
        type=str,
        default=None,  # Will use DEFAULT_ABLATION_OUTPUT_DIR
        help="Base directory to save experiment outputs for ablation runs.",
    )
    parser.add_argument(
        "--results_summary_dir",
        type=str,
        default=None,  # Will use DEFAULT_ABLATION_RESULTS_DIR
        help="Directory to save summary CSVs and plots for ablation studies.",
    )

    # Add any other relevant global arguments, e.g., data path if not in config

    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("Ablation script interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled error in ablation script: {e}", exc_info=True)
        sys.exit(1)
