#!/usr/bin/env python3
"""
Test script for cluster-based recommendation evaluation.

This script verifies that the new evaluation system:
1. Properly clusters users based on features
2. Evaluates intra-cluster recommendation performance
3. Compares against baseline methods
"""

import asyncio
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

from baselines.baseline_evaluator import BaselineEvaluator
from vulcan.core.config_manager import ConfigManager
from vulcan.data.goodreads_loader import GoodreadsDataLoader
from vulcan.evaluation.cluster_recommendation_evaluator import (
    ClusterRecommendationEvaluator,
)
from vulcan.types import (
    FeatureDefinition,
    FeatureSet,
    FeatureType,
    FeatureValue,
    MCTSAction,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_evaluation_system():
    """Test the cluster-based recommendation evaluation."""

    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.config
    logger.info("Configuration loaded")

    # Load data
    loader = GoodreadsDataLoader(
        db_path="data/goodreads.db",
        splits_dir="data/splits",
        outer_fold=1,
        inner_fold=1,
    )
    data_context = loader.get_data_context(sample_size=1000)
    logger.info(
        f"Data loaded: {data_context.n_users} users, {data_context.n_items} items"
    )

    # Initialize baselines
    baseline_evaluator = BaselineEvaluator(db_path="data/goodreads.db", config={})
    baseline_evaluator.load_data(sample_size=1000)
    baseline_evaluator.fit_baselines()
    baseline_scores = baseline_evaluator.evaluate_baselines(k=10)
    logger.info(f"Baseline scores: {baseline_scores}")

    # Initialize evaluator
    evaluator = ClusterRecommendationEvaluator(
        config, baselines=baseline_evaluator.baselines
    )
    await evaluator.initialize()

    # Create test features
    test_features = [
        FeatureDefinition(
            name="User Average Rating",
            feature_type=FeatureType.CODE_BASED,
            description="Average rating per user",
            code="result = df.groupby('user_id')['rating'].mean()",
            dependencies=["user_id", "rating"],
            computational_cost=1.0,
        ),
        FeatureDefinition(
            name="User Rating Count",
            feature_type=FeatureType.CODE_BASED,
            description="Number of ratings per user",
            code="result = df.groupby('user_id')['rating'].count()",
            dependencies=["user_id", "rating"],
            computational_cost=1.0,
        ),
        FeatureDefinition(
            name="User Rating Variance",
            feature_type=FeatureType.CODE_BASED,
            description="Rating variance per user",
            code="result = df.groupby('user_id')['rating'].var().fillna(0)",
            dependencies=["user_id", "rating"],
            computational_cost=1.5,
        ),
    ]

    # Create feature set
    feature_set = FeatureSet(features=test_features, action_taken=MCTSAction.ADD)

    # Execute features (simplified for testing)
    feature_results = {}

    # Get sample data from streaming context
    train_batch = data_context.get_sample_batch("train", max_records=1000)
    train_df = pd.DataFrame(train_batch)

    logger.info(f"Loaded train batch with {len(train_df)} records")

    for feature in test_features:
        # Create mock feature values using the actual data
        user_ids = train_df["user_id"].unique()[:100]  # Test with subset

        if feature.name == "User Average Rating":
            values = train_df.groupby("user_id")["rating"].mean()
        elif feature.name == "User Rating Count":
            values = train_df.groupby("user_id")["rating"].count()
        else:  # User Rating Variance
            values = train_df.groupby("user_id")["rating"].var().fillna(0)

        feature_results[feature.name] = [
            FeatureValue(
                user_id=uid, feature_name=feature.name, value=values.get(uid, 0.0)
            )
            for uid in user_ids
        ]

    logger.info(f"Generated feature results for {len(feature_results)} features")

    # Evaluate features
    evaluation = await evaluator.evaluate_feature_set(
        feature_set=feature_set,
        feature_results=feature_results,
        data_context=data_context,
        iteration=1,
    )

    # Display results
    print("\n" + "=" * 60)
    print("CLUSTER-BASED RECOMMENDATION EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nOverall Score: {evaluation.overall_score:.4f}")

    print("\nClustering Metrics:")
    print(f"  Silhouette Score: {evaluation.metrics.silhouette_score:.4f}")
    print(f"  Calinski-Harabasz: {evaluation.metrics.calinski_harabasz:.2f}")
    print(f"  Davies-Bouldin: {evaluation.metrics.davies_bouldin:.4f}")

    print("\nRecommendation Metrics:")
    print(f"  Precision@10: {evaluation.metrics.precision_at_10:.4f}")
    print(f"  Recall@10: {evaluation.metrics.recall_at_10:.4f}")
    print(f"  NDCG@10: {evaluation.metrics.ndcg_at_10:.4f}")

    print("\nCluster Analysis:")
    print(f"  Optimal Clusters: {evaluation.metrics.num_clusters}")
    print(f"  Cluster Coverage: {evaluation.metrics.cluster_coverage:.2%}")
    print(
        f"  Intra-cluster Similarity: {evaluation.metrics.intra_cluster_similarity:.4f}"
    )

    print("\nPerformance vs Baselines:")
    print(f"  Improvement: {evaluation.metrics.improvement_over_baseline:.2%}")

    print("\nBaseline Comparison:")
    for name, score in baseline_scores.items():
        print(f"  {name}: {score:.4f}")

    print("\n" + "=" * 60)

    # Test with different feature combinations
    print("\nTesting with single feature...")
    single_feature_set = FeatureSet(
        features=[test_features[0]],  # Just average rating
        action_taken=MCTSAction.ADD,
    )

    single_evaluation = await evaluator.evaluate_feature_set(
        feature_set=single_feature_set,
        feature_results={test_features[0].name: feature_results[test_features[0].name]},
        data_context=data_context,
        iteration=2,
    )

    print(f"Single Feature Score: {single_evaluation.overall_score:.4f}")
    print(f"Single Feature Clusters: {single_evaluation.metrics.num_clusters}")

    return evaluation


if __name__ == "__main__":
    asyncio.run(test_evaluation_system())
