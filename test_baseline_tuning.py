"""Test baseline hyperparameter tuning."""

import asyncio
import json

from vulcan.data.goodreads_loader import GoodreadsDataLoader
from vulcan.evaluation import RecommendationEvaluator
from vulcan.types import VulcanConfig


async def test_baseline_tuning():
    """Test baseline hyperparameter tuning process."""

    # Create config
    config = VulcanConfig()

    # Create evaluator
    evaluator = RecommendationEvaluator(config)
    await evaluator.initialize()

    # Load data
    print("Loading Goodreads data...")
    loader = GoodreadsDataLoader(
        db_path="/Users/nicolasdhnr/Documents/Imperial/Imperial Thesis/Code/VULCAN/data/goodreads.db",
        splits_dir="/Users/nicolasdhnr/Documents/Imperial/Imperial Thesis/Code/VULCAN/data/splits",
        outer_fold=1,
        inner_fold=1,
        batch_size=1000,
    )
    data_context = loader.get_data_context()

    # Run pre-tuning
    print("\n🔧 Starting baseline hyperparameter tuning...")
    baseline_scores = await evaluator.pretune_baselines(
        data_context,
        force_retune=True,  # Force to see the process
    )

    print("\n📊 Baseline scores after tuning:")
    for model, score in baseline_scores.items():
        print(f"  {model}: {score:.4f}")

    # Load and display tuned hyperparameters
    try:
        with open("baseline_hyperparams.json") as f:
            tuned_params = json.load(f)

        print("\n🎯 Tuned hyperparameters:")
        for model, params in tuned_params.items():
            print(f"\n{model}:")
            for param, value in params.items():
                print(f"  {param}: {value}")
    except Exception as e:
        print(f"Could not load hyperparameters: {e}")

    # Test that cached loading works
    print("\n🔄 Testing cached hyperparameter loading...")
    evaluator2 = RecommendationEvaluator(config)
    await evaluator2.initialize()

    if evaluator2.hyperparams_tuned:
        print("✅ Successfully loaded cached hyperparameters")
    else:
        print("❌ Failed to load cached hyperparameters")


if __name__ == "__main__":
    asyncio.run(test_baseline_tuning())
