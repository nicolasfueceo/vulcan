import os
import sqlite3
from pathlib import Path

import pandas as pd
import yaml


def print_header(title):
    print("\\n" + "=" * 80)
    print(f"⚙️  {title}")
    print("=" * 80)


def get_dataset_stats(db_path: str) -> dict:
    """Gets basic statistics from the database."""
    print(f"Analyzing database at: {db_path}")
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Database not found at {db_path}. Please ensure the path is correct."
        )

    with sqlite3.connect(db_path) as conn:
        n_users = pd.read_sql("SELECT COUNT(DISTINCT user_id) FROM reviews", conn).iloc[
            0, 0
        ]
        n_items = pd.read_sql("SELECT COUNT(DISTINCT book_id) FROM books", conn).iloc[
            0, 0
        ]
        n_ratings = pd.read_sql("SELECT COUNT(*) FROM reviews", conn).iloc[0, 0]

    stats = {
        "n_users": n_users,
        "n_items": n_items,
        "n_ratings": n_ratings,
        "density": n_ratings / (n_users * n_items) if (n_users * n_items) > 0 else 0,
    }
    print("Database Stats:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  - {key}: {value:.6f}")
        else:
            print(f"  - {key}: {value:,}")
    return stats


def generate_base_config(db_path: str) -> dict:
    """Generates the base configuration dictionary."""
    return {
        "api": {
            "host": "localhost",
            "port": 8000,
            "reload": True,
        },
        "llm": {
            "provider": "local",
            "model_name": "mock_model",
            "temperature": 0.5,
        },
        "data": {
            "db_path": db_path,
            "splits_dir": "data/splits",
            "outer_fold": 1,
            "inner_fold": 1,
            "sample_size": 10000,  # A reasonable sample for faster runs
        },
        "logging": {
            "level": "INFO",
        },
        "evaluation": {
            "sample_size": 5000,
            "scoring_mode": "cluster",
        },
        "experiment": {
            "output_dir": "experiments",
            "tensorboard_enabled": True,
            "wandb_enabled": False,  # Disable W&B by default for local runs
        },
    }


def create_experiment_configs(base_config: dict, stats: dict, output_dir: Path):
    """Creates a suite of YAML configuration files for an ablation study."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating experiment configs in: {output_dir}")

    # 1. Baseline Configuration
    baseline_exp = {
        "name": "Baseline_Medium",
        "max_generations": 50,
        "population_size": 50,
        "generation_size": 10,
        "mutation_rate": 0.3,
        "ucb_exploration_constant": 1.4,
    }
    config_baseline = base_config.copy()
    config_baseline["experiment"].update(baseline_exp)
    save_config(config_baseline, output_dir / "01_baseline.yaml")

    # 2. High Exploration
    high_explore_exp = baseline_exp.copy()
    high_explore_exp["name"] = "Ablation_High_Exploration"
    high_explore_exp["ucb_exploration_constant"] = 2.5  # High exploration
    config_high_explore = base_config.copy()
    config_high_explore["experiment"].update(high_explore_exp)
    save_config(config_high_explore, output_dir / "02_high_exploration.yaml")

    # 3. High Exploitation (Low Exploration)
    low_explore_exp = baseline_exp.copy()
    low_explore_exp["name"] = "Ablation_High_Exploitation"
    low_explore_exp["ucb_exploration_constant"] = 0.5  # Low exploration
    config_low_explore = base_config.copy()
    config_low_explore["experiment"].update(low_explore_exp)
    save_config(config_low_explore, output_dir / "03_high_exploitation.yaml")

    # 4. Large Population
    large_pop_exp = baseline_exp.copy()
    large_pop_exp["name"] = "Ablation_Large_Population"
    large_pop_exp["population_size"] = 100  # Double population
    large_pop_exp["generation_size"] = 20
    large_pop_exp["max_generations"] = 25  # Halve generations to keep compute similar
    config_large_pop = base_config.copy()
    config_large_pop["experiment"].update(large_pop_exp)
    save_config(config_large_pop, output_dir / "04_large_population.yaml")

    # 5. High Mutation Rate
    high_mutation_exp = baseline_exp.copy()
    high_mutation_exp["name"] = "Ablation_High_Mutation"
    high_mutation_exp["mutation_rate"] = 0.6  # High mutation
    config_high_mutation = base_config.copy()
    config_high_mutation["experiment"].update(high_mutation_exp)
    save_config(config_high_mutation, output_dir / "05_high_mutation.yaml")

    print(f"✅ Generated {len(output_dir.glob('*.yaml'))} configuration files.")


def save_config(config_dict: dict, path: Path):
    """Saves a dictionary as a YAML file."""
    with open(path, "w") as f:
        yaml.dump(config_dict, f, sort_keys=False, default_flow_style=False, indent=2)
    print(f"   - Saved {path.name}")


def main():
    print_header("VULCAN Configuration Setup Script")
    project_root = Path(__file__).resolve().parent.parent
    db_path = str(project_root / "data" / "goodreads.db")
    configs_output_dir = project_root / "configs"

    try:
        stats = get_dataset_stats(db_path)
        base_config = generate_base_config(db_path)
        create_experiment_configs(base_config, stats, configs_output_dir)
        print(
            "\\nSetup complete. You can now run 'scripts/queue_final_experiments.py' to queue these experiments."
        )
    except Exception as e:
        print(f"\\n❌ An error occurred during setup: {e}")


if __name__ == "__main__":
    main()
