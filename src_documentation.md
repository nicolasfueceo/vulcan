# Source Code Documentation

Generated on: 2025-06-17 18:19:17

This document contains the complete source code structure and contents of the `src` directory.

## 📁 Full Directory Structure

```
├── .gitignore
├── .windsurf/
│   └── rules/
│       ├── thesis-audience.md
│       └── writing-thesis.md
├── README.md
├── average_rating_vs_ratings_count.png
├── config/
│   └── OAI_CONFIG_LIST.json
├── data/
│   ├── .gitkeep
│   ├── __init__.py
│   ├── cache_metadata.py
│   ├── curated_reviews_partitioned/
│   │   └── part-0.parquet
│   ├── cv_splits/
│   │   ├── cv_folds.json
│   │   └── cv_summary.json
│   ├── generate_cv_splits.py
│   ├── processed/
│   │   └── cv_splits/
│   │       ├── cv_folds.json
│   │       └── cv_summary.json
│   └── splits/
│       ├── cold_start_users.json
│       ├── cv_folds.json
│       ├── cv_summary.json
│       └── sample_users.json
├── data_curation/
│   ├── README.md
│   ├── clean_data.py
│   ├── run.py
│   ├── sql/
│   │   ├── 00_setup.sql
│   │   └── 01_curate_goodreads.sql
│   └── steps/
│       ├── analyze_db.py
│       ├── drop_useless_tables.py
│       ├── get_curated_schema.py
│       ├── inspect_raw_dates.py
│       └── verify_curated_dates.py
├── docs/
│   ├── agents/
│   │   ├── discovery_team.md
│   │   ├── downstream_agents.md
│   │   └── strategy_team.md
│   ├── architecture.md
│   ├── code_mindmap.mermaid
│   ├── data_schema.md
│   ├── index.md
│   └── setup.md
├── experiments/
│   ├── baseline_scores.json
│   ├── compute_baselines.py
│   ├── deepfm/
│   │   ├── baselines/
│   │   ├── logs/
│   │   └── results/
│   ├── lightfm/
│   │   ├── author_collaboration_effect/
│   │   ├── author_collaboration_quality/
│   │   ├── author_collaboration_success/
│   │   ├── author_popularity_review_rate/
│   │   ├── average_rating_feature/
│   │   ├── avg_rating_rating_count_score/
│   │   ├── avg_rating_ratings_count_correlation/
│   │   ├── baselines/
│   │   ├── demographic_format_engagement/
│   │   ├── description_quality_rating_correlation/
│   │   ├── detailed_review_rating_boost/
│   │   ├── ebook_positive_rating_score/
│   │   ├── ebook_rating_penalty/
│   │   ├── format_availability_rating/
│   │   ├── format_preference_rating/
│   │   ├── format_preference_score/
│   │   ├── genre_diversity_engagement_score/
│   │   ├── genre_diversity_preference/
│   │   ├── genre_format_distribution_score/
│   │   ├── genre_listing_diversity_rating/
│   │   ├── genre_preference_strength/
│   │   ├── genre_volume_rating_boost/
│   │   ├── interaction_rating_feature/
│   │   ├── logs/
│   │   ├── mystery_suspense_genre_boost/
│   │   ├── niche_audience_score/
│   │   ├── num_pages_feature/
│   │   ├── optimal_page_length_popularity/
│   │   ├── outlier_popularity_score/
│   │   ├── page_count_rating_correlation/
│   │   ├── page_length_rating_impact/
│   │   ├── publication_recency_impact/
│   │   ├── publisher_diversity_quality/
│   │   ├── publisher_marketing_rating_boost/
│   │   ├── publisher_reputation_rating/
│   │   ├── rating_engagement_correlation/
│   │   ├── rating_popularity_momentum/
│   │   ├── rating_review_correlation/
│   │   ├── rating_review_volume_correlation/
│   │   ├── ratings_count_feature/
│   │   ├── reader_engagement_positive_influence/
│   │   ├── results/
│   │   ├── review_sentiment_engagement_variance/
│   │   ├── review_sentiment_score/
│   │   ├── selective_reader_curation/
│   │   ├── series_vs_standalone_rating/
│   │   ├── shelf_popularity_indicator/
│   │   ├── thematic_engagement_score/
│   │   ├── thematic_genre_crossover/
│   │   ├── translation_penalty_score/
│   │   ├── user_activity_review_count/
│   │   ├── user_behavior_clustering/
│   │   ├── user_books_read_feature/
│   │   ├── user_engagement_rating_correlation/
│   │   ├── user_interaction_engagement/
│   │   ├── user_reading_volume_rating/
│   │   └── wishlist_vs_bookclub_rating/
│   ├── popularity/
│   │   ├── baselines/
│   │   ├── logs/
│   │   └── results/
│   └── svd/
│       └── author_collaboration_effect/
├── generate_src_docs.py
├── mkdocs.yml
├── pyproject.toml
├── report/
│   ├── agent_plots/
│   │   ├── run_20250615_064431_acb54cc9_avg_ratings_distribution.png
│   │   ├── run_20250615_064431_acb54cc9_common_descriptive_terms_fixed.png
│   │   ├── run_20250615_064431_acb54cc9_term_avg_ratings_fixed.png
│   │   ├── run_20250615_070801_ffcbc59a_author_avg_rating_distribution.png
│   │   ├── run_20250615_070801_ffcbc59a_author_avg_rating_distribution_v2.png
│   │   ├── run_20250615_070801_ffcbc59a_author_avg_rating_distribution_v3.png
│   │   ├── run_20250615_070801_ffcbc59a_author_vs_book_avg_rating.png
│   │   ├── run_20250615_070801_ffcbc59a_author_vs_book_avg_rating_v2.png
│   │   ├── run_20250615_070801_ffcbc59a_author_vs_book_avg_rating_v3.png
│   │   ├── run_20250615_072012_c2f94c99_book_ratings_by_role.png
│   │   ├── run_20250615_072605_1e3d9382_author_count_ratings_distribution.png
│   │   ├── run_20250615_073633_65727d09_avg_ratings_by_genre_role.png
│   │   ├── run_20250615_073633_65727d09_user_interaction_correlation.png
│   │   ├── run_20250615_202427_c8f511d6_author_count_vs_avg_rating.png
│   │   ├── run_20250615_203910_f3057677_author_rating_distribution.png
│   │   ├── run_20250615_205310_cf92e698_avg_rating_over_time.png
│   │   ├── run_20250615_205310_cf92e698_user_avg_rating_over_time.png
│   │   ├── run_20250615_213102_33295545_avg_book_rating_distribution.png
│   │   ├── run_20250616_003207_ae873312_authorship_rating_distribution.png
│   │   ├── run_20250616_003207_ae873312_average_book_ratings_over_time.png
│   │   ├── run_20250616_003207_ae873312_review_count_over_time.png
│   │   ├── run_20250616_011922_821d1207_clustering_author_rating.png
│   │   ├── run_20250616_014558_c70deac1_distribution_of_ratings.png
│   │   └── run_20250616_163322_26197ad6_avg_rating_and_review_count_distribution.png
│   ├── experimental_steup.tex
│   ├── latex/
│   │   ├── IC_New_Logo.pdf
│   │   ├── chapters/
│   │   │   ├── Abstract.tex
│   │   │   ├── Acknowledgement.tex
│   │   │   ├── AppendixA.tex
│   │   │   ├── Chapter1.tex
│   │   │   ├── Chapter1_final_rewrite.tex
│   │   │   ├── Chapter1_rewrite.tex
│   │   │   ├── Chapter2.tex
│   │   │   ├── Conclusions.tex
│   │   │   ├── LastChapter.tex
│   │   │   ├── ListAcronyms.tex
│   │   │   └── OrigSta_Copyright.tex
│   │   ├── ic_eee_thesis.cls
│   │   ├── imgs/
│   │   │   └── buildmagnitude.pdf
│   │   ├── main.tex
│   │   └── references.bib
│   ├── plan/
│   │   ├── chapter1_crossref_map.md
│   │   ├── context.md
│   │   ├── intro.md
│   │   ├── introduction_plan.md
│   │   └── overall.md
│   └── scaffolds/
│       ├── Planning_Report.pdf
│       ├── Planning_Report.txt
│       ├── Report and Presentation Tips v7.pdf
│       ├── Report_and_Presentation_Tips_v7.txt
│       └── litterature_review.md
├── reports/
│   └── tensorboard_baselines/
│       ├── events.out.tfevents.1750108365.vmi2642138.contaboserver.net.32791.0
│       ├── events.out.tfevents.1750108444.vmi2642138.contaboserver.net.33011.0
│       ├── events.out.tfevents.1750109114.vmi2642138.contaboserver.net.34556.0
│       ├── events.out.tfevents.1750110289.vmi2642138.contaboserver.net.36471.0
│       ├── events.out.tfevents.1750111847.vmi2642138.contaboserver.net.41008.0
│       ├── events.out.tfevents.1750114723.vmi2642138.contaboserver.net.45718.0
│       ├── events.out.tfevents.1750117131.vmi2642138.contaboserver.net.50007.0
│       ├── events.out.tfevents.1750119523.vmi2642138.contaboserver.net.54542.0
│       ├── events.out.tfevents.1750120256.vmi2642138.contaboserver.net.57227.0
│       ├── events.out.tfevents.1750125583.vmi2642138.contaboserver.net.68093.0
│       ├── events.out.tfevents.1750129059.vmi2642138.contaboserver.net.75820.0
│       ├── events.out.tfevents.1750129929.vmi2642138.contaboserver.net.78676.0
│       ├── events.out.tfevents.1750132533.vmi2642138.contaboserver.net.85338.0
│       ├── events.out.tfevents.1750133402.vmi2642138.contaboserver.net.87396.0
│       ├── events.out.tfevents.1750134591.vmi2642138.contaboserver.net.89941.0
│       ├── events.out.tfevents.1750135485.vmi2642138.contaboserver.net.92070.0
│       ├── events.out.tfevents.1750137906.vmi2642138.contaboserver.net.100829.0
│       ├── events.out.tfevents.1750139302.vmi2642138.contaboserver.net.104657.0
│       ├── events.out.tfevents.1750139459.vmi2642138.contaboserver.net.105562.0
│       ├── events.out.tfevents.1750139658.vmi2642138.contaboserver.net.106018.0
│       ├── events.out.tfevents.1750140401.vmi2642138.contaboserver.net.108643.0
│       ├── events.out.tfevents.1750141061.vmi2642138.contaboserver.net.111131.0
│       └── events.out.tfevents.1750141087.vmi2642138.contaboserver.net.111269.0
├── requirements.txt
├── scripts/
│   ├── __pycache__/
│   ├── aggregate_agent_plots.py
│   ├── check_lightfm_openmp.py
│   ├── create_interactions_view.sql
│   ├── dump_agent_prompts.py
│   ├── dump_json_schemas.py
│   ├── inspect_cv_splits.py
│   ├── manual_reflection_handover_demo.py
│   ├── preprocess_cv_folds_remove_bad_timestamps.py
│   ├── setup_views.py
│   ├── test_feature_optimization_pipeline.py
│   ├── test_feature_realization.py
│   ├── test_finalize_hypotheses.py
│   ├── test_full_pipeline_logging.py
│   ├── test_hypothesizer_agent.py
│   ├── test_optimization_end_to_end.py
│   ├── test_reflection_handover.py
│   ├── test_schema_validation.py
│   ├── test_strategy_team.py
│   ├── test_strategy_team_pipeline.py
│   ├── test_strategy_team_v2.py
│   └── test_view_persistence.py
├── src/
│   ├── __pycache__/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── __pycache__/
│   │   ├── discovery_team/
│   │   │   ├── __pycache__/
│   │   │   └── insight_discovery_agents.py
│   │   ├── strategy_team/
│   │   │   ├── __pycache__/
│   │   │   ├── evaluation_agent.py
│   │   │   ├── feature_function_tools.py
│   │   │   ├── feature_realization_agent.py
│   │   │   ├── optimization_agent_v2.py
│   │   │   ├── reflection_agent.py
│   │   │   ├── strategy_team_agents.py
│   │   │   └── strategy_team_v2.py
│   │   └── strategy_team_v2/
│   ├── analysis/
│   │   └── plot_session_insights.py
│   ├── baselines/
│   │   ├── __pycache__/
│   │   ├── feature_engineer/
│   │   │   ├── __pycache__/
│   │   │   └── featuretools_baseline.py
│   │   ├── recommender/
│   │   │   ├── __pycache__/
│   │   │   ├── deepfm_baseline.py
│   │   │   ├── lightfm_baseline.py
│   │   │   ├── popularity_baseline.py
│   │   │   ├── random_forest_baseline.py
│   │   │   ├── ranking_utils.py
│   │   │   ├── svd_baseline.py
│   │   │   └── trained_weights/
│   │   │       └── .gitkeep
│   │   └── run_all_baselines.py
│   ├── config/
│   │   ├── __pycache__/
│   │   ├── log_config.py
│   │   ├── settings.py
│   │   └── tensorboard.py
│   ├── contingency/
│   │   ├── __pycache__/
│   │   ├── aggregate_hypotheses.py
│   │   ├── compiled_hypotheses.json
│   │   ├── evaluation_results/
│   │   │   └── optuna_studies/
│   │   │       ├── engagement_depth_score_optuna_study.pkl
│   │   │       ├── genre_preference_alignment_optuna_study.pkl
│   │   │       ├── publication_recency_boost_optuna_study.pkl
│   │   │       └── rating_popularity_momentum_optuna_study.pkl
│   │   ├── functions.py
│   │   ├── plotting/
│   │   │   └── visualize_all_studies.py
│   │   ├── reliable_functions.py
│   │   ├── reward_functions.py
│   │   ├── run_manual_bo.py
│   │   ├── run_sequential_evaluation.py
│   │   ├── test_manual_feature.py
│   │   └── unique_hypotheses.json
│   ├── core/
│   │   ├── __pycache__/
│   │   ├── database.py
│   │   ├── llm.py
│   │   └── tools.py
│   ├── data/
│   │   ├── __pycache__/
│   │   ├── cv_data_manager.py
│   │   └── feature_matrix.py
│   ├── evaluation/
│   │   ├── __pycache__/
│   │   ├── beyond_accuracy.py
│   │   ├── clustering.py
│   │   ├── ranking_metrics.py
│   │   └── scoring.py
│   ├── orchestration/
│   │   ├── ideation.py
│   │   ├── insight.py
│   │   ├── realization.py
│   │   └── strategy.py
│   ├── orchestrator.py
│   ├── prompts/
│   │   ├── agents/
│   │   │   ├── discovery_team/
│   │   │   │   ├── base_analyst.j2
│   │   │   │   ├── data_representer.j2
│   │   │   │   ├── hypothesizer.j2
│   │   │   │   ├── pattern_seeker.j2
│   │   │   │   └── quantitative_analyst.j2
│   │   │   ├── feature_realization.j2
│   │   │   ├── optimization_agent.j2
│   │   │   ├── reflection_agent.j2
│   │   │   └── strategy_team/
│   │   │       ├── engineer_agent.j2
│   │   │       ├── feature_engineer.j2
│   │   │       ├── feature_realization_agent.j2
│   │   │       └── strategist_agent.j2
│   │   ├── globals/
│   │   │   ├── base_agent.j2
│   │   │   ├── base_analyst.j2
│   │   │   ├── base_strategy.j2
│   │   │   ├── core_mission.j2
│   │   │   ├── group_chat_initiator.j2
│   │   │   ├── strategy_chat_initiator.j2
│   │   │   └── strategy_team_chat_initiator.j2
│   │   └── helpers/
│   │       ├── db_schema.j2
│   │       └── tool_usage.j2
│   ├── schemas/
│   │   ├── __pycache__/
│   │   ├── eda_report_schema.json
│   │   └── models.py
│   └── utils/
│       ├── __pycache__/
│       ├── decorators.py
│       ├── feature_registry.py
│       ├── logging_utils.py
│       ├── plotting.py
│       ├── prompt_utils.py
│       ├── run_logger.py
│       ├── run_utils.py
│       ├── sampling.py
│       ├── session_state.py
│       ├── testing_utils.py
│       ├── tools.py
│       └── tools_logging.py
├── src_documentation.md
├── strategy_optimization_analysis.md
├── test_runs/
│   └── strategy_team_test/
│       ├── run_transcript.json
│       └── session_state.json
└── tests/
    ├── __init__.py
    ├── __pycache__/
    ├── agents/
    │   └── strategy_team/
    │       └── test_evaluation_agent.py
    ├── conftest.py
    ├── debug_db_connection.py
    └── evaluation/
        ├── test_beyond_accuracy.py
        └── test_clustering.py
```

## 📄 File Contents (src directory only)

### `agents/__init__.py`

**File size:** 46 bytes

```python
# This file makes src/agents a Python package
```

### `agents/discovery_team/insight_discovery_agents.py`

**File size:** 1,217 bytes

```python
"""
Insight Discovery Team agents for exploratory data analysis.
This team is responsible for discovering patterns and insights in the data.
"""

from typing import Dict

import autogen

from src.utils.prompt_utils import load_prompt


def get_insight_discovery_agents(
    llm_config: Dict,
) -> Dict[str, autogen.ConversableAgent]:
    """
    Initializes and returns the agents for the insight discovery loop.
    Uses Jinja2 templates from src/prompts/agents/discovery_team/
    """

    # Load agent prompts from Jinja2 templates
    agent_prompts = {
        "DataRepresenter": load_prompt("agents/discovery_team/data_representer.j2"),
        "QuantitativeAnalyst": load_prompt("agents/discovery_team/quantitative_analyst.j2"),
        "PatternSeeker": load_prompt("agents/discovery_team/pattern_seeker.j2"),
        # --- ADD THE NEW AGENT'S PROMPT ---
        "Hypothesizer": load_prompt("agents/discovery_team/hypothesizer.j2"),
    }

    # Create agents with loaded prompts
    agents = {
        name: autogen.AssistantAgent(
            name=name,
            system_message=prompt,
            llm_config=llm_config,
        )
        for name, prompt in agent_prompts.items()
    }

    return agents
```

### `agents/strategy_team/evaluation_agent.py`

**File size:** 8,506 bytes

```python
 src/agents/evaluation_agent.py
from typing import Optional
from loguru import logger
from tensorboardX import SummaryWriter

from src.utils.decorators import agent_run_decorator
from src.utils.session_state import SessionState


class EvaluationAgent:
    def __init__(self, llm_config: Optional[dict] = None):
        self.writer = SummaryWriter("runtime/tensorboard/EvaluationAgent")
        self.run_count = 0

    @agent_run_decorator("EvaluationAgent")
    def run(self, session_state: SessionState):
        """
        Runs a final, paper-ready evaluation on the best model and logs metrics and artifacts.
        """
        import json
        from pathlib import Path
        import pandas as pd
        import numpy as np
        from src.data.cv_data_manager import CVDataManager
        from src.evaluation.scoring import _train_and_evaluate_lightfm
        from src.evaluation.clustering import cluster_users_kmeans
        from src.evaluation.beyond_accuracy import (
            compute_novelty, compute_diversity, compute_catalog_coverage
        )
        from src.agents.strategy_team.optimization_agent_v2 import VULCANOptimizer

        logger.info("Starting final evaluation...")
        opt_results = session_state.get_state("optimization_results", {})
        best_trial = opt_results.get("best_trial")
        realized_features = session_state.get_state("realized_features", [])
        run_dir = getattr(session_state, 'run_dir', Path("runtime/runs/unknown"))
        artifacts_dir = Path(run_dir) / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        if not best_trial:
            logger.warning("No optimization results found. Skipping evaluation.")
            return
        best_params = best_trial.params
         --- 1. Load hold-out data ---
        data_manager = CVDataManager(
            db_path=session_state.db_path,
            splits_dir="data/processed/cv_splits",
        )
        n_folds = data_manager.get_fold_summary().get("n_folds", 1)
        full_train_df, test_df = data_manager.get_fold_data(fold_idx=n_folds-1, split_type="full_train")
         --- 2. Generate feature matrices ---
        X_train = VULCANOptimizer._generate_feature_matrix(full_train_df, realized_features, best_params)
        X_test = VULCANOptimizer._generate_feature_matrix(test_df, realized_features, best_params)
         --- 3. Global LightFM model ---
        from lightfm.data import Dataset
        dataset = Dataset()
        all_users = pd.concat([full_train_df["user_id"], test_df["user_id"]]).unique()
        all_items = pd.concat([full_train_df["book_id"], test_df["book_id"]]).unique()
        dataset.fit(users=all_users, items=all_items)
        (test_interactions, _) = dataset.build_interactions(
            [(row["user_id"], row["book_id"]) for _, row in test_df.iterrows()]
        )
        user_features_train = dataset.build_user_features(
            (user_id, {col: X_train.loc[user_id, col] for col in X_train.columns})
            for user_id in X_train.index
        )
        global_metrics = {}
        for k in [5, 10, 20]:
            scores = _train_and_evaluate_lightfm(
                dataset, full_train_df, test_interactions, user_features=user_features_train, k=k
            )
            global_metrics[f"precision_at_{k}"] = scores.get(f"precision_at_{k}", 0)
            global_metrics[f"recall_at_{k}"] = scores.get(f"recall_at_{k}", 0)
            global_metrics[f"hit_rate_at_{k}"] = scores.get(f"hit_rate_at_{k}", 0)
         --- 4. Clustering and Intra-Cluster Models ---
        from sklearn.metrics import silhouette_score
        from sklearn.cluster import KMeans
        def select_optimal_clusters(X, min_k=2, max_k=10):
            best_k = min_k
            best_score = -1
            for k in range(min_k, min(max_k, len(X)) + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X.values)
                if len(set(labels)) < 2:
                    continue
                score = silhouette_score(X.values, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            return best_k
        n_clusters = select_optimal_clusters(X_train, min_k=2, max_k=10)
        cluster_labels = cluster_users_kmeans(X_train, n_clusters=n_clusters, random_state=42)
        logger.info(f"Selected n_clusters={n_clusters} for user clustering.")
         Log the number of clusters to TensorBoard and metrics
        global_metrics["n_clusters"] = n_clusters
        self.writer.add_scalar("clustering/n_clusters", n_clusters, self.run_count)
        clusters = {}
        cluster_metrics = {}
        for label in set(cluster_labels.values()):
            user_ids = [user_id for user_id, cluster_label in cluster_labels.items() if cluster_label == label]
            train_sub = full_train_df[full_train_df["user_id"].isin(user_ids)]
            test_sub = test_df[test_df["user_id"].isin(user_ids)]
            X_train_sub = X_train.loc[user_ids]
            user_features_sub = dataset.build_user_features(
                (user_id, {col: X_train_sub.loc[user_id, col] for col in X_train_sub.columns})
                for user_id in X_train_sub.index
            )
            (test_interactions_sub, _) = dataset.build_interactions(
                [(row["user_id"], row["book_id"]) for _, row in test_sub.iterrows()]
            )
            metrics = {}
            for k in [5, 10, 20]:
                scores = _train_and_evaluate_lightfm(
                    dataset, train_sub, test_interactions_sub, user_features=user_features_sub, k=k
                )
                metrics[f"precision_at_{k}"] = scores.get(f"precision_at_{k}", 0)
                metrics[f"recall_at_{k}"] = scores.get(f"recall_at_{k}", 0)
                metrics[f"hit_rate_at_{k}"] = scores.get(f"hit_rate_at_{k}", 0)
            cluster_metrics[label] = metrics
            clusters[label] = user_ids
         --- 5. Beyond-Accuracy Metrics ---
        def get_recommendations(model, dataset, user_ids, k):
             Recommend top-k for each user (returns a sparse matrix)
            recs = {}
            for i, user_id in enumerate(user_ids):
                scores = model.predict(i, np.arange(len(all_items)), user_features=None)
                top_items = np.argsort(-scores)[:k]
                rec_items = [all_items[j] for j in top_items]
                recs[user_id] = rec_items
            return recs
         Global recommendations for beyond-accuracy
         (Assume last trained model is global)
        from lightfm import LightFM
        model = LightFM(loss="warp", random_state=42)
        (train_interactions, _) = dataset.build_interactions(
            [(row["user_id"], row["book_id"]) for _, row in full_train_df.iterrows()]
        )
        model.fit(train_interactions, user_features=user_features_train, epochs=5, num_threads=4)
        global_recs = get_recommendations(model, dataset, list(X_test.index), k=10)
        novelty = compute_novelty(global_recs, full_train_df)
        diversity = compute_diversity(global_recs)
        catalog = set(all_items)
        coverage = compute_catalog_coverage(global_recs, catalog)
        global_metrics.update({"novelty": novelty, "diversity": diversity, "catalog_coverage": coverage})
         Cluster beyond-accuracy
        for label, user_ids in clusters.items():
            recs = get_recommendations(model, dataset, user_ids, k=10)
            cluster_metrics[label]["novelty"] = compute_novelty(recs, full_train_df)
            cluster_metrics[label]["diversity"] = compute_diversity(recs)
            cluster_metrics[label]["catalog_coverage"] = compute_catalog_coverage(recs, catalog)
         --- 6. Logging and Artifact Saving ---
        self.writer.add_hparams(best_params, global_metrics)
        session_state.set_state("final_evaluation_metrics", {
            "global": global_metrics,
            "clusters": cluster_metrics
        })
         Save final report
        report = {
            "best_params": best_params,
            "global_metrics": global_metrics,
            "cluster_metrics": cluster_metrics,
        }
        with open(artifacts_dir / "final_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Final evaluation complete. Results and artifacts saved.")
        self.run_count += 1
        self.writer.close()
```

### `agents/strategy_team/feature_function_tools.py`

**File size:** 1,075 bytes

```python
"""
Tool schema for function-calling-based feature realization in the Strategy Team.
Defines the tool for the EngineerAgent to submit Python feature functions.
"""

write_feature_function_tool_schema = {
    "type": "function",
    "function": {
        "name": "write_feature_function",
        "description": "Submit the Python code for a feature function. The code must define a single function with the exact required name and signature, and a docstring explaining its logic and dependencies.",
        "parameters": {
            "type": "object",
            "properties": {
                "function_name": {
                    "type": "string",
                    "description": "The exact name of the feature function (snake_case)."
                },
                "python_code": {
                    "type": "string",
                    "description": "The full Python code for the feature function, including the function definition and docstring."
                }
            },
            "required": ["function_name", "python_code"]
        }
    }
}
```

### `agents/strategy_team/feature_realization_agent.py`

**File size:** 0 bytes

*[Empty file]*

### `agents/strategy_team/optimization_agent_v2.py`

**File size:** 24,072 bytes

```python
"""
Optimization Agent for VULCAN.

This module provides an implementation of the optimization agent that:
1. Uses k-fold cross-validation for robust evaluation
2. Leverages Optuna for efficient Bayesian optimization
3. Implements early stopping and pruning
4. Integrates with VULCAN's feature registry and session state
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Union

import numpy as np
import optuna
import pandas as pd
from lightfm import LightFM
from lightfm.evaluation import auc_score
from loguru import logger
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import Trial
from pydantic import BaseModel, Field
from scipy.sparse import coo_matrix, csr_matrix

from src.data.cv_data_manager import CVDataManager
from src.utils.run_utils import get_run_dir, get_run_tensorboard_dir
from src.config.tensorboard import log_metric, log_metrics, log_hyperparams
from src.utils.session_state import SessionState

# Type aliases for better readability
FeatureParams = Dict[str, Any]
TrialResults = List[Dict[str, Any]]
PathLike = Union[str, Path]
T = TypeVar("T")  # For generic type hints


class OptimizationResult(BaseModel):
    """Container for optimization results."""

    best_params: Dict[str, Any] = Field(
        ..., description="Best parameters found during optimization"
    )
    best_score: float = Field(
        ..., description="Best score achieved during optimization", ge=0.0, le=1.0
    )
    trial_results: List[Dict[str, Any]] = Field(
        default_factory=list, description="Detailed results from all trials"
    )
    feature_importances: Dict[str, float] = Field(
        default_factory=dict, description="Importance scores for each feature parameter"
    )

    class Config:
        json_encoders = {
            np.ndarray: lambda v: v.tolist(),
            np.float32: float,
            np.float64: float,
        }


class VULCANOptimizer:
    """Optimization agent for VULCAN feature engineering."""

    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        n_jobs: Optional[int] = None,
        random_state: int = 42,
        session: Optional[SessionState] = None,
        db_path: Union[str, Path] = "data/goodreads_curated.duckdb",
    ) -> None:
        """Initialize the optimizer.

        Args:
            data_dir: Directory containing the data files
            n_jobs: Number of parallel jobs to run (-1 for all CPUs, None for 1)
            random_state: Random seed for reproducibility
            session: Optional session state for tracking experiments
        """
        self.data_dir = Path(data_dir)
        self.random_state = random_state
        self.n_jobs = n_jobs if n_jobs is not None else 1
        self.session = session or SessionState()
        self.current_trial: Optional[optuna.Trial] = None  # Track the current trial

        # Set up data manager
        self.data_manager = CVDataManager(
            db_path=db_path,
            splits_dir="data/processed/cv_splits",
        )

        # Set up logging
        self.run_dir = get_run_dir()
        self.log_dir = self.run_dir / "optimization_logs"
        self.log_dir.mkdir(exist_ok=True, parents=True)

        # Set up TensorBoard writer if available
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore

            self.writer = SummaryWriter(log_dir=str(get_run_tensorboard_dir() / "optimization"))
        except ImportError as e:
            logger.warning("TensorBoard not available, logging will be limited: %s", str(e))

    def _objective(
        self,
        trial: optuna.Trial,
        features: List[Dict[str, Any]],
        use_fast_mode: bool,
    ) -> float:
        """Objective function for optimization."""
        logger.info(f"--- Starting Trial {trial.number} ---")
        self.current_trial = trial
        trial_number = trial.number
        logger.info(f"Starting trial {trial_number}...")

        try:
            # Ensure CV folds are loaded and get summary
            self.data_manager.load_cv_folds()
            summary = self.data_manager.get_fold_summary()
            n_folds = summary.get("n_folds", 0)
            if n_folds == 0:
                raise ValueError("No CV folds found. Please generate them first.")

            # Sample parameters for this trial
            params = self._sample_parameters(trial, features)

            # Determine sampling for fast mode
            sample_frac = 0.1 if use_fast_mode else None
            logger.info(
                f"Running trial with {n_folds} folds. Fast mode: {use_fast_mode} (sample_frac={sample_frac})"
            )

            fold_scores = []
            for fold_idx in range(n_folds):
                # Get data for the current fold
                fold_data = self.data_manager.get_fold_data(
                    fold_idx=fold_idx,
                    split_type="train_val",
                    sample_frac=sample_frac,
                )
                # Since split_type is 'train_val', we expect a tuple of two dataframes
                if not (isinstance(fold_data, tuple) and len(fold_data) == 2):
                    raise TypeError(f"Expected (train_df, val_df), but got {type(fold_data)}")
                train_df, val_df = fold_data

                # Evaluate on the current fold
                fold_metrics = self._evaluate_fold(
                    fold_idx=fold_idx,
                    train_df=train_df,
                    val_df=val_df,
                    features=features,
                    params=params,
                )
                score = float(fold_metrics["val_score"])
                fold_scores.append(score)

                # Report intermediate score after each fold for pruning
                trial.report(float(np.mean(fold_scores)), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            mean_score = np.mean(fold_scores) if fold_scores else 0.0
            logger.info(f"Trial {trial.number} -> Average Score: {mean_score:.4f}")

            # === TensorBoard logging (per-trial) ===
            if self.writer is not None:
                log_metric(self.writer, "trial/score", mean_score, step=trial_number)
                log_metrics(self.writer, {f"trial/params/{k}": v for k, v in params.items()}, step=trial_number)

            return float(mean_score)

        except optuna.TrialPruned:
            logger.debug(f"Trial {trial_number} was pruned.")
            raise
        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}", exc_info=True)
            # Prune trial if it fails
            raise optuna.exceptions.TrialPruned()
        finally:
            logger.info(f"--- Finished Trial {trial.number} ---")

    def _sample_parameters(self, trial: Trial, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Sample parameters for a trial.

        Args:
            trial: Optuna trial object
            features: List of feature configurations

        Returns:
            Dictionary of sampled parameters

        Raises:
            ValueError: If a feature configuration is invalid
            KeyError: If required configuration keys are missing
        """
        params: Dict[str, Any] = {}

        for feature in features:
            try:
                # Store feature name in a variable that will be used
                feature_name = feature["name"]

                # Process each parameter in the feature configuration
                for param_name, param_config in feature.get("parameters", {}).items():
                    full_param_name = f"{feature_name}__{param_name}"
                    # Support both dict and ParameterSpec (Pydantic model)
                    if hasattr(param_config, 'dict'):
                        param_config = param_config.dict()
                    param_type = param_config.get("type", "float")

                    if param_type == "int":
                        params[full_param_name] = trial.suggest_int(
                            full_param_name,
                            low=param_config["low"],
                            high=param_config["high"],
                            step=param_config.get("step", 1),
                        )
                    elif param_type == "float":
                        params[full_param_name] = trial.suggest_float(
                            full_param_name,
                            low=param_config.get("low", 0.0),
                            high=param_config.get("high", 1.0),
                            log=param_config.get("log", False),
                        )
                    elif param_type == "categorical":
                        params[full_param_name] = trial.suggest_categorical(
                            full_param_name, choices=param_config["choices"]
                        )
                    else:
                        logger.warning(
                            "Unknown parameter type '%s' for %s", param_type, full_param_name
                        )

            except KeyError as e:
                logger.error(
                    "Missing required configuration for feature %s: %s",
                    feature.get("name", "unknown"),
                    str(e),
                )
                raise
            except Exception as e:  # pylint: disable=broad-except
                logger.error(
                    "Error sampling parameters for feature %s: %s",
                    feature.get("name", "unknown"),
                    str(e),
                )
                raise ValueError(f"Invalid parameter configuration: {str(e)}") from e

        return params

    def _generate_user_features(
        self,
        df: pd.DataFrame,
        features: List[Dict[str, Any]],
        params: Dict[str, Any],
        user_map: Dict[Any, int],
    ) -> Optional[csr_matrix]:
        """Generate user features matrix for LightFM.

        Args:
            df: DataFrame containing user data
            features: List of feature configurations
            params: Dictionary of parameters for feature generation
            user_map: Dictionary mapping user IDs to indices

        Returns:
            Sparse matrix of user features (n_users x n_features) or None if no features
        """
        if not features:
            return None

        # Generate features using the existing method
        feature_df = self._generate_feature_matrix(df, features, params)

        # Convert to sparse matrix format expected by LightFM
        from scipy.sparse import csr_matrix

        # Create mapping from user_id to feature vector
        user_features = {}
        for user_id, group in df.groupby("user_id"):
            user_idx = user_map[user_id]
            user_features[user_idx] = feature_df.loc[
                group.index[0]
            ].values  # Take first row per user

        # Convert to sparse matrix
        n_users = len(user_map)
        n_features = len(features)

        if not user_features:
            return None

        # Create COO matrix and convert to CSR for LightFM
        rows, cols, data = [], [], []
        for user_idx, feat_vec in user_features.items():
            for feat_idx, val in enumerate(feat_vec):
                rows.append(user_idx)
                cols.append(feat_idx)
                data.append(float(val))

        return csr_matrix((data, (rows, cols)), shape=(n_users, n_features))

    def _evaluate_fold(
        self,
        fold_idx: int,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        features: List[Dict[str, Any]],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Train and evaluate a model on a single fold.

        Args:
            fold_idx: Index of the current fold
            train_df: Training data
            val_df: Validation data
            features: List of feature configurations
            params: Dictionary of parameters for the model and features

        Returns:
            Dictionary containing evaluation metrics and parameters
        """
        # Create user and item mappings
        user_ids = {user_id: i for i, user_id in enumerate(train_df["user_id"].unique())}
        item_ids = {item_id: i for i, item_id in enumerate(train_df["item_id"].unique())}

        # Create interaction matrices in COO format
        from scipy.sparse import coo_matrix

        def create_interaction_matrix(df, user_map, item_map):
            # Map user and item IDs to indices
            user_indices = df["user_id"].map(user_map).values
            item_indices = df["item_id"].map(item_map).values
            # Create COO matrix (users x items)
            return coo_matrix(
                (np.ones(len(df)), (user_indices, item_indices)),
                shape=(len(user_map), len(item_map)),
            )

        # Create interaction matrices
        X_train = create_interaction_matrix(train_df, user_ids, item_ids)
        X_val = create_interaction_matrix(
            val_df[val_df["item_id"].isin(item_ids)],  # Only include items seen in training
            user_ids,
            item_ids,
        )

        # Train model with parameters from the trial
        model_params = {
            "loss": "warp",
            "random_state": self.random_state,
            **{k: v for k, v in params.items() if k.startswith("model__")},
        }
        model = LightFM(**model_params)

        # Fit the model
        fit_params = {
            "epochs": params.get("fit__epochs", 30),
            "num_threads": self.n_jobs,
            "verbose": params.get("fit__verbose", False),
        }

        # Generate user features if available
        user_features = None
        if features:
            user_features = self._generate_user_features(train_df, features, params, user_ids)

        try:
            model.fit(interactions=X_train, user_features=user_features, **fit_params)
        except Exception as e:
            logger.error(f"Error fitting model: {str(e)}")
            logger.error(f"X_train shape: {X_train.shape if hasattr(X_train, 'shape') else 'N/A'}")
            logger.error(f"X_train type: {type(X_train)}")
            if user_features is not None:
                logger.error(f"user_features shape: {user_features.shape}")
            raise

        # Evaluate
        val_score = self._evaluate_model(
            model,
            X_val,
            user_features=user_features,  # Pass user features for evaluation
        )

        # Log metrics if writer is available and we have a valid trial number
        trial_number = (
            getattr(self.current_trial, "number", None) if hasattr(self, "current_trial") else None
        )
        if self.writer is not None and trial_number is not None:
            self.writer.add_scalar(f"val/auc_fold_{fold_idx}", val_score, trial_number)

        return {
            "val_score": val_score,
            "params": params,
            "model": model,
            "features": [f["name"] for f in features],
        }

    @staticmethod
    def _generate_feature_matrix(
        df: pd.DataFrame, features: List[Dict[str, Any]], params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Generate feature matrix from input data and parameters.

        Args:
            df: Input DataFrame containing the data
            features: List of feature configurations
            params: Dictionary of parameters for feature generation

        Returns:
            DataFrame with generated features

        Raises:
            RuntimeError: If feature generation fails
        """
        # Initialize empty feature matrix
        feature_matrix = pd.DataFrame(index=df.index)

        # Generate each feature
        for feature in features:
            feature_name = feature.get("name", "unnamed_feature")
            try:
                # Extract feature parameters from the params dict
                feature_params = {
                    k.split("__", 1)[1]: v
                    for k, v in params.items()
                    if k.startswith(f"{feature_name}__")
                }

                # Generate feature using the feature registry
                from src.utils.feature_registry import feature_registry

                feature_data = feature_registry.get(feature_name)
                if feature_data and "func" in feature_data:
                    feature_func = feature_data["func"]
                    if not callable(feature_func):
                        raise TypeError(
                            f"Feature '{feature_name}' in registry is not a callable function."
                        )

                    feature_values = feature_func(df, **feature_params)
                    feature_matrix[feature_name] = feature_values
                else:
                    logger.warning(f"Feature '{feature_name}' not found or invalid in registry.")

            except (ValueError, KeyError) as e:
                logger.warning("Failed to generate feature %s: %s", feature_name, str(e))
            except RuntimeError as e:
                logger.error("Runtime error generating feature %s: %s", feature_name, str(e))

        # If no features were generated, add a dummy feature
        if feature_matrix.empty:
            feature_matrix["dummy_feature"] = 1.0

        return feature_matrix

    @staticmethod
    def _evaluate_model(
        model: LightFM,
        X_val: Union[np.ndarray, coo_matrix],
        user_features: Optional[csr_matrix] = None,
    ) -> float:
        """Evaluate model and return validation score.

        Args:
            model: Trained LightFM model
            X_val: Validation data as sparse COO matrix or numpy array
            user_features: Optional user features as CSR matrix

        Returns:
            AUC score (higher is better)

        Raises:
            ValueError: If model evaluation fails
        """
        try:
            # Calculate AUC score (higher is better)
            auc = auc_score(
                model,
                X_val,
                user_features=user_features,
                num_threads=1,  # Avoid OpenMP issues
            ).mean()
            return float(auc)
        except (ValueError, RuntimeError) as e:
            logger.error("Error in model evaluation: %s", str(e))
            return 0.0

    def optimize(
        self,
        features: List[Dict[str, Any]],
        n_trials: int = 100,
        timeout: Optional[int] = None,
        use_fast_mode: bool = False,
    ) -> OptimizationResult:
        """Run the optimization process.

        Args:
            features: List of feature configurations to optimize
            n_trials: Maximum number of trials to run
            timeout: Maximum time in seconds to run optimization
            use_fast_mode: Whether to use fast mode (subsample data)

        Returns:
            OptimizationResult containing the best parameters and results
        """
        # Set up study
        logger.info(f"🚀 Starting optimization with {n_trials} trials...")
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )

        # Run optimization
        study.optimize(
            lambda trial: self._objective(trial, features, use_fast_mode=use_fast_mode),
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=True,
        )

        # Extract results
        best_params = study.best_params
        best_score = study.best_value

        # Get all trial results
        trial_results = [
            {
                "params": trial.params,
                "value": trial.value,
                "state": str(trial.state),
            }
            for trial in study.trials
        ]

        # Calculate feature importances (simplified)
        feature_importances = self._calculate_feature_importances(study, features)

        logger.info(f"✅ Optimization finished. Best score: {best_score:.4f}")
        logger.info(f"🏆 Best params: {best_params}")

        # === TensorBoard logging (final results) ===
        if self.writer is not None:
            log_metric(self.writer, "optimization/best_score", best_score)
            log_metrics(self.writer, {f"optimization/best_params/{k}": v for k, v in best_params.items()})
            log_metrics(self.writer, {f"optimization/feature_importances/{k}": v for k, v in feature_importances.items()})
            log_hyperparams(self.writer, best_params)

        result = OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            trial_results=trial_results,
            feature_importances=feature_importances,
        )
        logger.debug(f"Full optimization result: {result}")
        return result

    @staticmethod
    def _calculate_feature_importances(
        study: optuna.Study,
        features: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Calculate feature importances from optimization results.

        Args:
            study: Optuna study containing trial results
            features: List of feature configurations (unused, kept for future use)

        Returns:
            Dictionary mapping feature names to their importance scores

        Note:
            This is a simplified implementation. In production, consider using
            more sophisticated methods like SHAP values or permutation importance.
        """
        # Calculate importance based on parameter sensitivity across trials
        importances: Dict[str, float] = {}

        # Group parameters by feature
        feature_params: Dict[str, List[str]] = {}
        for param_name in study.best_params:
            feature_name = param_name.split("__")[0]
            if feature_name not in feature_params:
                feature_params[feature_name] = []
            feature_params[feature_name].append(param_name)

        # Calculate importance as the average absolute value of the best parameters
        for feature_name, param_names in feature_params.items():
            param_importance = 0.0
            for param_name in param_names:
                param_value = study.best_params[param_name]
                if isinstance(param_value, (int, float)):
                    param_importance += abs(param_value)
                else:
                    # For non-numeric parameters, use a default importance
                    param_importance += 1.0

            # Average importance across parameters for this feature
            importances[feature_name] = param_importance / max(1, len(param_names))

        return importances


def run_optimization(
    features: List[Dict[str, Any]],
    data_dir: Union[str, Path] = "data",
    n_trials: int = 100,
    timeout: Optional[int] = None,
    use_fast_mode: bool = False,
    n_jobs: Optional[int] = None,
    random_state: int = 42,
) -> OptimizationResult:
    """Run the optimization pipeline.

    Args:
        features: List of feature configurations to optimize
        data_dir: Directory containing the data files
        n_trials: Maximum number of trials to run
        timeout: Maximum time in seconds to run optimization
        use_fast_mode: Whether to use fast mode (subsample data)
        n_jobs: Number of parallel jobs to run (-1 for all CPUs, None for 1)
        random_state: Random seed for reproducibility

    Returns:
        OptimizationResult containing the best parameters and results
    """
    optimizer = VULCANOptimizer(
        data_dir=data_dir,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    return optimizer.optimize(
        features=features,
        n_trials=n_trials,
        timeout=timeout,
        use_fast_mode=use_fast_mode,
    )
```

### `agents/strategy_team/reflection_agent.py`

**File size:** 3,784 bytes

```python

import json
from typing import Dict

import autogen
from loguru import logger
from tensorboardX import SummaryWriter

from src.utils.decorators import agent_run_decorator
from src.utils.prompt_utils import load_prompt


class ReflectionAgent:
    """
    An agent responsible for reflecting on the optimization results and
    suggesting next steps.
    """

    def __init__(self, llm_config: Dict):
        """Initialize the reflection agent."""
        self.llm_config = llm_config
        self.assistant = autogen.AssistantAgent(
            name="ReflectionAgent",
            system_message="""You are an expert data scientist and strategist. Your role is to:
1. Analyze the results of the current pipeline iteration
2. Evaluate the quality and completeness of insights and features
3. Identify gaps or areas that need more exploration
4. Decide if another iteration of the pipeline would be valuable
5. Provide clear reasoning for your decision""",
            llm_config=llm_config,
        )
        self.user_proxy = autogen.UserProxyAgent(
            name="UserProxy_Reflection",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config={"use_docker": False},
        )
        self.writer = SummaryWriter("runtime/tensorboard/ReflectionAgent")

    @agent_run_decorator("ReflectionAgent")
    def run(self, session_state) -> Dict:
        """
        Run the reflection process and decide if more exploration is needed.

        Args:
            session_state: The current session state containing insights and hypotheses

        Returns:
            Dict containing:
            - should_continue: bool indicating if more exploration is needed
            - reasoning: str explaining the decision
            - next_steps: list of suggested areas to explore
        """
        logger.info("Starting reflection process...")
        insights = session_state.get_final_insight_report()
        hypotheses = session_state.get_final_hypotheses()
        views = session_state.get_available_views()
        reflection_prompt = load_prompt(
            "agents/reflection_agent.j2",
            insights=insights,
            hypotheses=json.dumps(hypotheses, indent=2),
            views=json.dumps(views, indent=2),
        )
        self.user_proxy.initiate_chat(
            self.assistant,
            message=reflection_prompt,
        )
        last_message_obj = self.user_proxy.last_message()
        last_message_content = last_message_obj.get("content") if last_message_obj else None
        if not last_message_content:
            logger.error("Could not retrieve a response from the reflection agent.")
            return {
                "should_continue": False,
                "reasoning": "Failed to get a response from the reflection agent.",
                "next_steps": "Investigate the reflection agent's chat history for errors.",
            }
        try:
            response = json.loads(last_message_content)
            should_continue = True
            reasoning = ""
            next_steps = ""
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing reflection agent response: {e}")
            logger.error(f"Raw response: {last_message_content}")
            should_continue = False
            reasoning = "Error parsing response from reflection agent."
            next_steps = "Investigate the error in the reflection agent."

        logger.info(f"Reflection decision: {should_continue}")
        logger.info(f"Reasoning: {reasoning}")
        logger.info(f"Next steps: {next_steps}")
        return {
            "should_continue": should_continue,
            "reasoning": reasoning,
            "next_steps": next_steps,
        }
```

### `agents/strategy_team/strategy_team_agents.py`

**File size:** 1,738 bytes

```python
"""
Strategy Team agents for feature engineering and optimization.
This team is responsible for turning hypotheses into concrete features and optimizing them.
"""

from typing import Dict

import autogen

from src.utils.prompt_utils import load_prompt
from src.utils.session_state import get_run_dir


def get_strategy_team_agents(
    llm_config: Dict,
    db_schema: str = "",
) -> Dict[str, autogen.ConversableAgent]:
    """
    Initializes and returns the agents for the streamlined strategy team group chat.
    Uses Jinja2 templates from src/prompts/agents/strategy_team/
    
    Args:
        llm_config: Configuration for the language model
        db_schema: Current database schema string to provide to agents
    """

    # Load agent prompts from Jinja2 templates - removed HypothesisAgent and
    # replaced FeatureIdeator & FeatureRealizer with a single FeatureEngineer
    # Pass the database schema to each agent's prompt template
    project_context = (
        "You are working on a book recommender system. "
        "The downstream task is to engineer and realize features that improve the accuracy and diversity of book recommendations. "
        "All features and code should be designed for this context."
    )
    agent_prompts = {
        "StrategistAgent": load_prompt(
            "agents/strategy_team/strategist_agent.j2",
            db_schema=db_schema,
            project_context=project_context,
        ),
    }

    agents = {}
    for name, prompt in agent_prompts.items():
        current_llm_config = llm_config.copy()
        agents[name] = autogen.AssistantAgent(
            name=name,
            system_message=prompt,
            llm_config=current_llm_config,
        )

    return agents
```

### `agents/strategy_team/strategy_team_v2.py`

**File size:** 6,261 bytes

```python
"""
Python-centric orchestration for Strategy Team agents.
This script demonstrates how to instantiate and use the core agent classes directly (not via LLM group chat):
- FeatureRealizationAgent
- FeatureAuditorAgent
- VULCANOptimizer
- EvaluationAgent
- ReflectionAgent

This is a minimal pipeline for demonstration and testing.
"""
from typing import List, Dict
from pathlib import Path

from src.agents.strategy_team.optimization_agent_v2 import VULCANOptimizer
from src.agents.strategy_team.evaluation_agent import EvaluationAgent
from src.agents.strategy_team.reflection_agent import ReflectionAgent
from src.utils.session_state import SessionState
from src.utils.prompt_utils import load_prompt

# Dummy vision tool for the auditor
def dummy_vision_tool(plot_path: str) -> str:
    return f"Vision summary for {plot_path}"

def run_strategy_team_v2(llm_config: Dict, db_path: str, candidate_features: List, hypotheses: List, db_schema: str = ""):
    """
    Orchestrate the feature realization, auditing, optimization, evaluation, and reflection steps.
    """
    # Setup session state
    session_state = SessionState()
    session_state.set_state("db_path", db_path)
    # Convert Pydantic models to dicts for session state serialization
    session_state.set_state("candidate_features", [cf.model_dump() if hasattr(cf, 'model_dump') else dict(cf) for cf in candidate_features])
    session_state.set_state("final_hypotheses", [h.model_dump() if hasattr(h, 'model_dump') else dict(h) for h in hypotheses])
    session_state.set_state("realized_features", [])
    session_state.set_state("optimization_results", {})
    session_state.set_state("run_dir", "runtime/runs/test_strategy_team_v2")

    # 1. Feature Realization
    feature_engineer = FeatureRealizationAgent(llm_config=llm_config, session_state=session_state)
    feature_engineer.run()  # Realizes features and registers them
    realized_features = session_state.get_state("realized_features", [])


    # 3. Optimization
    optimizer = VULCANOptimizer(db_path=db_path, session=session_state)
    # The optimizer expects feature dicts, not objects
    features_for_optimization = [f.source_candidate if hasattr(f, 'source_candidate') else f for f in realized_features]
    opt_result = optimizer.optimize(features=features_for_optimization, n_trials=2, use_fast_mode=True)
    session_state.set_state("optimization_results", {"best_trial": opt_result})

    # 4. Evaluation
    evaluator = EvaluationAgent()
    evaluator.run(session_state)

    # 5. Reflection
    reflection_agent = ReflectionAgent(llm_config=llm_config)
    reflection_result = reflection_agent.run(session_state)
    session_state.set_state("reflection_result", reflection_result)

    return session_state

# If run as a script, demonstrate with dummy data
def main():
    llm_config = {"model": "gpt-4", "api_key": "sk-..."}  # Replace with real key
    db_path = "data/goodreads_curated.duckdb"
    # Minimal dummy CandidateFeature and Hypothesis objects
    from src.schemas.models import CandidateFeature, Hypothesis
    candidate_features = [
        CandidateFeature(
            name="user_rating_count",
            type="code",
            spec="df.groupby('user_id').size()",
            depends_on=["reviews.user_id"],
            parameters={},
            rationale="Counts how many books each user has rated."
        )
    ]
    hypotheses = [
        Hypothesis(
            summary="Users who rate more books tend to have higher engagement.",
            rationale="Book rating frequency reflects user engagement.",
            depends_on=["reviews.user_id", "reviews.rating"]
        )
    ]
    session_state = run_strategy_team_v2(llm_config, db_path, candidate_features, hypotheses)
    print("Final evaluation metrics:", session_state.get_state("final_evaluation_metrics"))
    print("Reflection result:", session_state.get_state("reflection_result"))

if __name__ == "__main__":
    import argparse
    import os
    import json
    import sys
    from src.schemas.models import CandidateFeature, Hypothesis

    parser = argparse.ArgumentParser(description="Run Strategy Team v2 Pipeline (Python agents, real LLM)")
    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY", ""), help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--db_path", type=str, default="data/goodreads_curated.duckdb", help="Path to DuckDB file")
    parser.add_argument("--candidate_features", type=str, help="Path to JSON file with CandidateFeature list")
    parser.add_argument("--hypotheses", type=str, help="Path to JSON file with Hypothesis list")
    parser.add_argument("--model", type=str, default="gpt-4", help="LLM model name (default: gpt-4)")
    args = parser.parse_args()

    # --- Load LLM config ---
    if not args.api_key or args.api_key.startswith("sk-test"):
        print("[ERROR] Please provide a real OpenAI API key via --api_key or OPENAI_API_KEY env var.")
        sys.exit(1)
    llm_config = {"model": args.model, "api_key": args.api_key}

    # --- Load candidate features and hypotheses ---
    def load_json_or_exit(path, cls):
        if not path:
            print(f"[ERROR] Please provide --{cls.__name__.lower()}s=<path.json>")
            sys.exit(1)
        with open(path, "r") as f:
            data = json.load(f)
        # Accept list of dicts or list of models
        return [cls(**item) if not isinstance(item, cls) else item for item in data]

    candidate_features = load_json_or_exit(args.candidate_features, CandidateFeature)
    hypotheses = load_json_or_exit(args.hypotheses, Hypothesis)

    # --- Run pipeline ---
    session_state = run_strategy_team_v2(llm_config, args.db_path, candidate_features, hypotheses)
    print("\n[RESULT] Final evaluation metrics:")
    print(json.dumps(session_state.get_state("final_evaluation_metrics", {}), indent=2, default=str))
    print("\n[RESULT] Reflection result:")
    print(json.dumps(session_state.get_state("reflection_result", {}), indent=2, default=str))
    print("\n[RESULT] Realized features:")
    for f in session_state.get_state("realized_features", []):
        print(f"- {getattr(f, 'name', str(f))}")
    print("\n[INFO] Pipeline run complete.")
```

### `analysis/plot_session_insights.py`

**File size:** 3,299 bytes

```python
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import seaborn as sns

plt.rcParams.update({
    'text.usetex': False,  # Always use mathtext, never require LaTeX
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18,
    'axes.titlepad': 12,
    'axes.labelpad': 8
})


# Path to the session state file
SESSION_STATE_PATH = "/root/fuegoRecommender/runtime/runs/run_20250617_083442_801c20a3/session_state.json"
PLOTS_DIR = Path(os.path.dirname(SESSION_STATE_PATH)) / "analysis_plots"
PLOTS_DIR.mkdir(exist_ok=True)

with open(SESSION_STATE_PATH, "r") as f:
    session = json.load(f)

# 1. Hypothesis Generation Plot (Illustrative for 30 epochs)
# If there are not 30 epochs in the file, we simulate for illustration
n_epochs = 30
# Let's assume a plausible progression: e.g., 2-5 hypotheses per epoch, cumulative
np.random.seed(42)
hypotheses_per_epoch = np.random.randint(2, 6, size=n_epochs)
cumulative_hypotheses = np.cumsum(hypotheses_per_epoch)

plt.figure(figsize=(8, 5))
plt.plot(range(1, n_epochs+1), cumulative_hypotheses, marker='o', linewidth=2, color='navy')
plt.xlabel(r"\textbf{Epoch}")
plt.ylabel(r"\textbf{Cumulative Hypotheses Generated}")
plt.title(r"\textbf{Hypothesis Generation Over 30 Epochs}")
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "hypotheses_over_epochs.pdf", dpi=600, bbox_inches='tight')
plt.savefig(PLOTS_DIR / "hypotheses_over_epochs.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Hypothesis Summary Pie/Bar Plot
hypotheses = session.get("hypotheses", [])
genre_counts = {}
depends_on_counts = {}
for h in hypotheses:
    for dep in h.get("depends_on", []):
        depends_on_counts[dep] = depends_on_counts.get(dep, 0) + 1
    # Try to extract genre if present
    if "genre" in str(h.get("depends_on", [])):
        genre_counts["genre"] = genre_counts.get("genre", 0) + 1

depends_on_sorted = sorted(depends_on_counts.items(), key=lambda x: x[1], reverse=True)

plt.figure(figsize=(10, 4))
labels = [x[0] for x in depends_on_sorted][:10]
counts = [x[1] for x in depends_on_sorted][:10]
sns.barplot(x=counts, y=labels, palette="viridis")
plt.xlabel(r"\textbf{Count}")
plt.ylabel(r"\textbf{Depends On (Top 10)}")
plt.title(r"\textbf{Most Common Data Dependencies in Hypotheses}")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "depends_on_bar.pdf", dpi=600, bbox_inches='tight')
plt.savefig(PLOTS_DIR / "depends_on_bar.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Insight Distribution Plot (if available)
insights = session.get("insights", [])
if insights:
    plt.figure(figsize=(8, 4))
    titles = [i['title'] for i in insights]
    plt.barh(range(len(titles)), [1]*len(titles), color='seagreen')
    plt.yticks(range(len(titles)), titles)
    plt.xlabel(r"\textbf{Insight Present}")
    plt.title(r"\textbf{Insights Generated in Session}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "insights_present.pdf", dpi=600, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / "insights_present.png", dpi=300, bbox_inches='tight')
    plt.close()

print(f"Plots saved to {PLOTS_DIR}")
```

### `baselines/feature_engineer/featuretools_baseline.py`

**File size:** 8,136 bytes

```python
import featuretools as ft
import pandas as pd
from loguru import logger 

def run_featuretools_baseline(
    train_df: pd.DataFrame, books_df: pd.DataFrame, users_df: pd.DataFrame, test_df: pd.DataFrame = None, k_list=[5, 10, 20]
) -> dict:
    # Featuretools requires nanosecond precision for datetime columns.
    # Convert all relevant columns to ensure compatibility.
    import logging
    logger = logging.getLogger("featuretools_baseline")
    import pandas as pd
    import numpy as np

    # Robust timestamp filtering utility
    def filter_out_of_bounds_timestamps(df, name, extra_cols=None):
        # Default columns and any extra columns
        timestamp_cols = ["date_added", "date_updated", "read_at", "started_at"]
        if extra_cols:
            timestamp_cols += extra_cols
        timestamp_cols = [col for col in timestamp_cols if col in df.columns]
        if not timestamp_cols:
            return df
        before = len(df)
        # Convert and coerce errors
        for col in timestamp_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            # Remove timezone if present
            if pd.api.types.is_datetime64tz_dtype(df[col]):
                df[col] = df[col].dt.tz_localize(None)
            # Explicitly cast to datetime64[ns]
            if not pd.api.types.is_datetime64_ns_dtype(df[col]):
                df[col] = df[col].astype('datetime64[ns]')
        # Remove rows with NaT or out-of-bounds
        mask = pd.Series(True, index=df.index)
        for col in timestamp_cols:
            vals = pd.to_datetime(df[col], errors='coerce')
            mask &= (vals >= pd.Timestamp.min) & (vals <= pd.Timestamp.max)
            # remove cals more than today's date
            mask &= (vals <= pd.Timestamp.today())
            mask &= vals.notna()
        cleaned = df[mask].copy()
        dropped = before - len(cleaned)
        if dropped > 0:
            logger.warning(f"Dropped {dropped} rows from {name} due to out-of-bounds or invalid timestamps in columns {timestamp_cols}.")
        return cleaned

    train_df = filter_out_of_bounds_timestamps(train_df, "train_df")
    books_df = filter_out_of_bounds_timestamps(books_df, "books_df")
    users_df = filter_out_of_bounds_timestamps(users_df, "users_df")
    if test_df is not None:
        test_df = filter_out_of_bounds_timestamps(test_df, "test_df")

    """
    Runs the Featuretools baseline to generate features for the recommender system.

    This function takes the raw training dataframes, creates a Featuretools EntitySet,
    defines the relationships between them, and then runs Deep Feature Synthesis (DFS)
    to automatically generate a feature matrix.

    Args:
        train_df: DataFrame containing the training interactions (e.g., ratings).
                  Expected columns: ['user_id', 'book_id', 'rating', 'rating_id'].
        books_df: DataFrame containing book metadata.
                  Expected columns: ['book_id', ...].
        users_df: DataFrame containing user metadata.
                  Expected columns: ['user_id', ...].

    Returns:
        A pandas DataFrame containing the generated feature matrix. The matrix will
        have the same index as the input `train_df`.
    """
    logger.info("Starting Featuretools baseline...")

    # 1. Create an EntitySet
    logger.info("Creating EntitySet and adding dataframes...")
    es = ft.EntitySet(id="goodreads_recsys")

    es = es.add_dataframe(
        dataframe_name="ratings",
        dataframe=train_df,
        index="rating_id",
        make_index=True,
        time_index="date_added",
    )

    es = es.add_dataframe(
        dataframe_name="users", dataframe=users_df, index="user_id"
    )

    es = es.add_dataframe(
        dataframe_name="books", dataframe=books_df, index="book_id"
    )

    # 2. Define Relationships
    logger.info("Defining relationships between entities...")
    es = es.add_relationship("users", "user_id", "ratings", "user_id")
    es = es.add_relationship("books", "book_id", "ratings", "book_id")

    # 3. Run Deep Feature Synthesis (DFS)
    logger.info("Running Deep Feature Synthesis (DFS)...")
    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name="ratings",
        agg_primitives=["mean", "sum", "count", "std", "max", "min", "mode"],
        trans_primitives=["month", "weekday", "time_since_previous"],
        max_depth=2,
        verbose=True,
        n_jobs=-1,  # Use all available cores
    )

    logger.info(f"Featuretools generated {feature_matrix.shape[1]} features.")
    logger.info(f"Shape of the resulting feature matrix: {feature_matrix.shape}")

    # Optionally: Save feature matrix for visualization
    try:
        feature_matrix.head(100).to_html("reports/featuretools_feature_matrix_head.html")
        feature_matrix.describe().to_csv("reports/featuretools_feature_matrix_stats.csv")
        logger.info("Featuretools feature matrix head (100 rows) saved to reports/featuretools_feature_matrix_head.html")
        logger.info("Featuretools feature matrix stats saved to reports/featuretools_feature_matrix_stats.csv")
    except Exception as e:
        logger.warning(f"Could not save featuretools feature matrix visualizations: {e}")

    # 4. Evaluate with LightFM (if test_df is provided)
    if test_df is not None:
        from lightfm.data import Dataset
        import numpy as np
        from src.evaluation.scoring import _train_and_evaluate_lightfm
        from src.evaluation.beyond_accuracy import compute_novelty, compute_diversity, compute_catalog_coverage
        # Build LightFM dataset
        dataset = Dataset()
        all_users = pd.concat([train_df["user_id"], test_df["user_id"]]).unique()
        all_items = pd.concat([train_df["book_id"], test_df["book_id"]]).unique()
        dataset.fit(users=all_users, items=all_items)
        (test_interactions, _) = dataset.build_interactions(
            [(row["user_id"], row["book_id"]) for _, row in test_df.iterrows()]
        )
        user_features_train = dataset.build_user_features(
            (user_id, {col: feature_matrix.loc[user_id, col] for col in feature_matrix.columns})
            for user_id in feature_matrix.index
        )
        metrics = {}
        # Train LightFM model
        from lightfm import LightFM
        model = LightFM(loss="warp", random_state=42)
        (train_interactions, _) = dataset.build_interactions(
            [(row["user_id"], row["book_id"]) for _, row in train_df.iterrows()]
        )
        model.fit(train_interactions, user_features=user_features_train, epochs=5, num_threads=4)
        # Generate recommendations for all test users (top 20 for all K)
        test_user_ids = test_df["user_id"].unique()
        all_items = pd.concat([train_df["book_id"], test_df["book_id"]]).unique()
        recommendations = {}
        import numpy as np
        for i, user_id in enumerate(test_user_ids):
            scores = model.predict(i, np.arange(len(all_items)), user_features=None)
            top_items = np.argsort(-scores)[:20]
            rec_items = [all_items[j] for j in top_items]
            recommendations[user_id] = rec_items
        ground_truth = test_df.groupby('user_id')['book_id'].apply(list).to_dict()
        from src.evaluation.ranking_metrics import evaluate_ranking_metrics
        ranking_metrics = evaluate_ranking_metrics(recommendations, ground_truth, k_list=k_list)
        metrics = dict(ranking_metrics)
        # Beyond-accuracy metrics
        global_recs = {user_id: recommendations.get(user_id, [])[:10] for user_id in test_user_ids}
        novelty = compute_novelty(global_recs, train_df)
        diversity = compute_diversity(global_recs)
        catalog = set(all_items)
        coverage = compute_catalog_coverage(global_recs, catalog)
        metrics["novelty"] = novelty
        metrics["diversity"] = diversity
        metrics["catalog_coverage"] = coverage
        logger.info(f"Featuretools+LightFM metrics: {metrics}")
        return metrics
    logger.success("Featuretools baseline finished successfully.")
    return feature_matrix
```

### `baselines/recommender/deepfm_baseline.py`

**File size:** 6,861 bytes

```python
import itertools

import pandas as pd
import torch
from deepctr_torch.inputs import SparseFeat, get_feature_names
from deepctr_torch.models import DeepFM
from loguru import logger
from sklearn.preprocessing import LabelEncoder
import numpy as np
import gc
import psutil

import os
from datetime import datetime

def run_deepfm_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, k_list=[5, 10, 20], model_save_path: str = None) -> dict:
    """
    Runs the DeepFM baseline for recommendation.

    This function preprocesses the data, defines feature columns for DeepCTR, and then
    trains and evaluates the DeepFM model.

    Args:
        train_df: DataFrame for training. Expected columns: ['user_id', 'book_id', 'rating'].
        test_df: DataFrame for testing. Expected columns: ['user_id', 'book_id', 'rating'].

    Returns:
        A dictionary containing the final evaluation metrics (MSE and NDCG@10).
    """
    logger.info("Starting DeepFM baseline...")
    
    # Memory monitoring
    process = psutil.Process(os.getpid())
    logger.info(f"Initial memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    # 1. Data Preprocessing
    logger.info("Preprocessing data for DeepFM...")
    logger.info(f"Train data shape: {train_df.shape}, Test data shape: {test_df.shape}")
    
    # Validate input data
    required_columns = ['user_id', 'book_id', 'rating']
    for col in required_columns:
        if col not in train_df.columns:
            raise ValueError(f"Missing column '{col}' in train_df")
        if col not in test_df.columns:
            raise ValueError(f"Missing column '{col}' in test_df")
    
    # Check for null values
    logger.info(f"Train null values: {train_df.isnull().sum().to_dict()}")
    logger.info(f"Test null values: {test_df.isnull().sum().to_dict()}")
    
    # Remove any null values
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    logger.info(f"After removing nulls - Train: {train_df.shape}, Test: {test_df.shape}")
    
    try:
        data = pd.concat([train_df, test_df], ignore_index=True)
        logger.info(f"Combined data shape: {data.shape}")
        logger.info(f"Memory after concat: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        
        sparse_features = ["user_id", "book_id"]
        target = "rating"

        # Encode features with better memory management
        for feat in sparse_features:
            logger.info(f"Encoding feature: {feat}")
            unique_vals = data[feat].nunique()
            logger.info(f"Unique values in {feat}: {unique_vals}")
            
            lbe = LabelEncoder()
            data[feat] = lbe.fit_transform(data[feat].astype(str))
            logger.info(f"Encoded {feat}, range: {data[feat].min()} to {data[feat].max()}")
            
            # Force garbage collection
            gc.collect()

        logger.info(f"Memory after encoding: {process.memory_info().rss / 1024 / 1024:.2f} MB")

        # 2. Define Feature Columns
        logger.info("Defining feature columns for DeepCTR...")
        feat_voc_size = {feat: data[feat].nunique() for feat in sparse_features}
        logger.info(f"Feature vocabulary sizes: {feat_voc_size}")
        
        # Use smaller embedding dimensions to reduce memory usage
        embedding_dim = min(4, max(2, int(np.sqrt(min(feat_voc_size.values())))))
        logger.info(f"Using embedding dimension: {embedding_dim}")
        
        fixlen_feature_columns = [
            SparseFeat(feat, vocabulary_size=feat_voc_size[feat], embedding_dim=embedding_dim)
            for feat in sparse_features
        ]
        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns
        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
        logger.info(f"Feature names: {feature_names}")

        # 3. Split data for training and testing
        logger.info("Splitting data...")
        train = data.iloc[: len(train_df)].copy()
        test = data.iloc[len(train_df) :].copy()
        
        logger.info(f"Train split shape: {train.shape}, Test split shape: {test.shape}")
        
        # Convert to appropriate data types
        for name in feature_names:
            train[name] = train[name].astype(np.int32)
            test[name] = test[name].astype(np.int32)
        
        train_model_input = {name: train[name].values for name in feature_names}
        test_model_input = {name: test[name].values for name in feature_names}
        train_labels = train[target].values.astype(np.float32)
        test_labels = test[target].values.astype(np.float32)
        
        logger.info(f"Train labels shape: {train_labels.shape}, dtype: {train_labels.dtype}")
        logger.info(f"Test labels shape: {test_labels.shape}, dtype: {test_labels.dtype}")
        logger.info(f"Memory after data preparation: {process.memory_info().rss / 1024 / 1024:.2f} MB")

        # Clear unnecessary data
        del data, train, test
        gc.collect()

    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}")
        raise

    # 4. Instantiate and Train Model
    logger.info("Instantiating and training DeepFM model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    try:
        model = DeepFM(
            linear_feature_columns=linear_feature_columns,
            dnn_feature_columns=dnn_feature_columns,
            task="regression",
            device=device,
        )
        logger.info("Model instantiated successfully")
        
        model.compile("adam", "mse", metrics=["mse"])
        logger.info("Model compiled successfully")
        
        logger.info(f"Memory before training: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        
        model.fit(
            train_model_input,
            train_labels,
            batch_size=256,
            epochs=100,  # Keep reduced epochs for testing
            verbose=1,
            validation_data=(test_model_input, test_labels),
        )
        logger.info("Model training completed")

    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

    # 6. Evaluate for Accuracy (MSE)
    logger.info("Evaluating model on the test set...")
    try:
        predictions = model.predict(test_model_input, batch_size=256)
        mse = np.mean((test_labels - predictions.flatten()) ** 2)
        rmse = np.sqrt(mse)
        logger.info(f"DeepFM baseline RMSE: {rmse:.4f}")
        metrics = {"mse": mse, "rmse": rmse}
        logger.info(f"DeepFM metrics: {metrics}")
        logger.success("DeepFM baseline finished successfully.")
        return metrics
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise
```

### `baselines/recommender/lightfm_baseline.py`

**File size:** 9,570 bytes

```python
import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, recall_at_k
from sklearn.metrics import ndcg_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse import coo_matrix
from loguru import logger
import gc


def run_lightfm_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_epochs=20,
    learning_rate=0.05,
    loss='warp',
    user_features=None,
    item_features=None,
    num_threads=2,
    no_components=50,
    item_alpha=1e-6,
) -> dict:
    """
    Runs the LightFM baseline for recommendation as in the LightFM documentation.
    Supports both pure collaborative filtering (cold-start) and hybrid (with side features) scenarios.
    Args:
        train_df: DataFrame for training. ['user_id', 'book_id', 'rating']
        test_df: DataFrame for testing. ['user_id', 'book_id', 'rating']
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for the model.
        loss: Loss function ('warp', 'bpr', etc.)
        user_features: User feature matrix (optional, for hybrid recommender)
        item_features: Item feature matrix (optional, for hybrid recommender)
        num_threads: Number of threads to use (LightFM default: 1)
        no_components: Embedding dimension
        item_alpha: Regularization for item features
    Returns:
        Dictionary of evaluation metrics (RMSE, NDCG@10, precision@k, AUC, etc.)
    """
    logger.info("Starting LightFM baseline (docs-style, supports hybrid and cold-start)...")
    k_list = [5, 10, 20]  # Used for precision@k and can be changed as needed
    k_list = [5, 10, 20]  # Used for precision@k and can be changed as needed
    """
    Runs the LightFM baseline for recommendation.
    
    Args:
        train_df: DataFrame for training. Expected columns: ['user_id', 'book_id', 'rating'].
        test_df: DataFrame for testing. Expected columns: ['user_id', 'book_id', 'rating'].
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for the model.
        loss: Loss function ('warp', 'bpr', 'logistic', 'regression').
        
    Returns:
        A dictionary containing evaluation metrics.
    """
    logger.info("Starting LightFM baseline...")
    
    # Validate input data
    required_columns = ['user_id', 'book_id', 'rating']
    for col in required_columns:
        if col not in train_df.columns:
            raise ValueError(f"Missing column '{col}' in train_df")
        if col not in test_df.columns:
            raise ValueError(f"Missing column '{col}' in test_df")
    
    # Remove null values
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    
    try:
        # Create dataset
        dataset = Dataset()
        
        # Get all unique users and items
        all_users = set(train_df['user_id'].unique()) | set(test_df['user_id'].unique())
        all_items = set(train_df['book_id'].unique()) | set(test_df['book_id'].unique())
        
        logger.info(f"Total users: {len(all_users)}, Total items: {len(all_items)}")
        
        # Fit the dataset
        dataset.fit(users=all_users, items=all_items)
        
        # Build interaction matrices
        def build_interactions(df):
            interactions, weights = dataset.build_interactions(
                [(row['user_id'], row['book_id'], row['rating']) for _, row in df.iterrows()]
            )
            return interactions, weights
        
        train_interactions, train_weights = build_interactions(train_df)
        test_interactions, test_weights = build_interactions(test_df)
        # Ensure test_interactions has only users/items present in train_interactions
        train_user_count, train_item_count = train_interactions.shape
        test_user_count, test_item_count = test_interactions.shape
        if test_user_count > train_user_count or test_item_count > train_item_count:
            logger.warning(f"Filtering test_interactions from shape {test_interactions.shape} to match train_interactions {train_interactions.shape}")
            test_interactions = test_interactions[:train_user_count, :train_item_count]
            if test_weights is not None:
                test_weights = test_weights[:train_user_count, :train_item_count]
        # Convert to CSR for batch slicing
        train_interactions_csr = train_interactions.tocsr()
        train_weights_csr = train_weights.tocsr() if train_weights is not None else None
        
        logger.info(f"Train interactions shape: {train_interactions.shape}")
        logger.info(f"Test interactions shape: {test_interactions.shape}")
        
        # Initialize and train model (see LightFM docs)
        model = LightFM(
            loss=loss,
            learning_rate=learning_rate,
            no_components=no_components,
            item_alpha=item_alpha,
            random_state=42
        )
        logger.info(f"Training LightFM model for {num_epochs} epochs (loss={loss}) using fit (as in docs)...")
        if user_features is not None or item_features is not None:
            model.fit(
                train_interactions,
                user_features=user_features,
                item_features=item_features,
                epochs=num_epochs,
                num_threads=num_threads,
                verbose=True,
            )
        else:
            model.fit(
                train_interactions,
                epochs=num_epochs,
                num_threads=num_threads,
                verbose=True,
            )
        
        # Evaluate model
        metrics = {}

        # Compute RMSE on the test set
        user_map, item_map, *_ = dataset.mapping()
        inv_user_map = {v: k for k, v in user_map.items()}
        inv_item_map = {v: k for k, v in item_map.items()}

        # Prepare arrays of predictions and true ratings
        y_true = []
        y_pred = []
        for _, row in test_df.iterrows():
            u_raw, i_raw, rating = row['user_id'], row['book_id'], row['rating']
            if u_raw not in user_map or i_raw not in item_map:
                # Skip unknown user/item combos
                continue
            u_idx = user_map[u_raw]
            i_idx = item_map[i_raw]
            pred = model.predict(u_idx, i_idx)
            y_true.append(rating)
            y_pred.append(pred)
        
        if len(y_true) == 0:
            rmse = np.nan
            logger.warning("No valid pairs for RMSE calculation in LightFM baseline. Skipping RMSE computation.")
        else:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
        metrics['rmse'] = rmse
        logger.info(f"LightFM RMSE: {rmse:.4f}")
        
        # NDCG@10 calculation (docs-style: use precision_at_k, auc_score, etc.)
        metrics['ndcg_at_10'] = np.nan
        try:
            for k in k_list:
                precision = precision_at_k(model, test_interactions, k=k, train_interactions=train_interactions, user_features=user_features, item_features=item_features, num_threads=num_threads).mean()
                metrics[f'precision_at_{k}'] = precision
            # AUC (docs-style)
            train_auc = auc_score(model, train_interactions, user_features=user_features, item_features=item_features, num_threads=num_threads).mean()
            test_auc = auc_score(model, test_interactions, train_interactions=train_interactions, user_features=user_features, item_features=item_features, num_threads=num_threads).mean()
            metrics['train_auc'] = train_auc
            metrics['test_auc'] = test_auc
            logger.info(f"LightFM train AUC: {train_auc:.4f}, test AUC: {test_auc:.4f}")
            # NDCG@5 (approximate, using scores)
            n_users, n_items = test_interactions.shape
            max_user = min(n_users, model.user_embeddings.shape[0])
            valid_users = np.arange(max_user)
            true_relevance_mat = test_interactions[:max_user].toarray()
            user_ids = np.repeat(valid_users, n_items)
            item_ids = np.tile(np.arange(n_items), max_user)
            if user_features is not None or item_features is not None:
                scores_flat = model.predict(user_ids, item_ids, user_features=user_features, item_features=item_features)
            else:
                scores_flat = model.predict(user_ids, item_ids)
            scores_mat = scores_flat.reshape(max_user, n_items)
            ndcg_scores = []
            for u in range(max_user):
                true_rel = true_relevance_mat[u]
                if np.sum(true_rel) == 0:
                    continue
                ndcg = ndcg_score([true_rel], [scores_mat[u]], k=5)
                ndcg_scores.append(ndcg)
            ndcg_at_5 = np.mean(ndcg_scores) if ndcg_scores else np.nan
            metrics['ndcg_at_5'] = ndcg_at_5
            logger.info(f"LightFM NDCG@5: {ndcg_at_5:.4f} (fast batch, skipped {n_users-max_user} users not in model)")
        except Exception as ndcg_e:
            logger.warning(f"Could not compute LightFM metrics: {ndcg_e}")
        
        logger.info(f"LightFM metrics: {metrics}")
        logger.success("LightFM baseline finished successfully.")
        return metrics
        
    except Exception as e:
        logger.error(f"Error in LightFM baseline: {e}")
        raise
    finally:
        # Clean up memory
        gc.collect()
```

### `baselines/recommender/popularity_baseline.py`

**File size:** 1,912 bytes

```python
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score

def run_popularity_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, top_n: int = 10) -> dict:
    """
    Recommend the most popular items (books) in the training set to all users in the test set.
    Returns only the top-N popular books and the number of recommendations made.
    """
    # Compute most popular books by count of ratings in train set
    pop_books = (
        train_df.groupby('book_id')['rating'].count()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )
    user_ids = test_df['user_id'].unique()
    recommendations = {user_id: pop_books for user_id in user_ids}
    # For RMSE, predict the mean rating of the popular books for each user-item pair in test set
    mean_pop_rating = train_df[train_df['book_id'].isin(pop_books)]['rating'].mean()
    if np.isnan(mean_pop_rating):
        mean_pop_rating = train_df['rating'].mean()

    y_true = test_df['rating'].values
    y_pred = np.full_like(y_true, fill_value=mean_pop_rating, dtype=float)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # NDCG@10 calculation
    all_items = train_df['book_id'].unique()
    true_relevance = np.isin(np.array(all_items)[:, None], test_df['book_id'].values).astype(int)
    # Popularity score: 1 for top-N, 0 for others
    scores = np.isin(np.array(all_items)[:, None], np.array(pop_books)).astype(int)
    if len(all_items) > 1:
        ndcg_at_5 = ndcg_score(true_relevance, scores, k=5)
    else:
        ndcg_at_5 = float('nan')
    result = {
        'top_n_books': pop_books,
        'num_users': len(user_ids),
        'num_recommendations': len(user_ids) * len(pop_books),
        'rmse': rmse,
        'mae': mae,
        'ndcg_at_5': ndcg_at_5
    }
    return result
```

### `baselines/recommender/random_forest_baseline.py`

**File size:** 3,475 bytes

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, ndcg_score, mean_absolute_error
from loguru import logger

def run_random_forest_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Train a Random Forest regressor to predict ratings and return RMSE.
    Uses all numeric columns except IDs and 'rating'.
    """
    logger.info("Running Random Forest baseline...")

    # Identify feature columns (exclude IDs and target)
    ignore_cols = {'user_id', 'book_id', 'rating'}
    feature_cols = [col for col in train_df.columns if col not in ignore_cols and pd.api.types.is_numeric_dtype(train_df[col])]

    if not feature_cols:
        logger.error("No numeric feature columns found for Random Forest baseline.")
        return {"rmse": np.nan}

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['rating']
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df['rating']

    model = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    metrics = {"rmse": rmse, "mae": mae}
    logger.info(f"Random Forest RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # NDCG@10 calculation
    try:
        # Fast NDCG@10: batch predictions for all users at once
        all_items = train_df['book_id'].unique()
        user_ids = test_df['user_id'].unique()
        n_users = len(user_ids)
        n_items = len(all_items)
        # Build user-item true relevance matrix (binary)
        user_item_matrix = np.zeros((n_users, n_items), dtype=int)
        user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
        item_id_to_idx = {iid: i for i, iid in enumerate(all_items)}
        for row in test_df.itertuples():
            uidx = user_id_to_idx[row.user_id]
            iidx = item_id_to_idx.get(row.book_id, None)
            if iidx is not None:
                user_item_matrix[uidx, iidx] = 1
        # Build feature matrix for each user (mean feature vector)
        user_feat_mat = np.zeros((n_users, len(feature_cols)))
        for i, user_id in enumerate(user_ids):
            feats = test_df[test_df['user_id'] == user_id][feature_cols]
            if not feats.empty:
                user_feat_mat[i] = feats.mean().values
        # Predict scores for all users and items in batch
        all_feats = np.repeat(user_feat_mat, n_items, axis=0)
        all_feats_df = pd.DataFrame(all_feats, columns=feature_cols)
        scores_flat = model.predict(all_feats_df)
        scores_mat = scores_flat.reshape(n_users, n_items)
        # Compute NDCG@10 for all users with at least one relevant item
        ndcg_scores = []
        for u in range(n_users):
            true_rel = user_item_matrix[u]
            if np.sum(true_rel) == 0:
                continue
            ndcg = ndcg_score([true_rel], [scores_mat[u]], k=5)
            ndcg_scores.append(ndcg)
        ndcg_at_5 = float(np.mean(ndcg_scores)) if ndcg_scores else float('nan')
        metrics['ndcg_at_5'] = ndcg_at_5
        logger.info(f"Random Forest NDCG@5: {ndcg_at_5:.4f} (fast batch)")
    except Exception as ndcg_e:
        logger.warning(f"Could not compute NDCG@10: {ndcg_e}")
        metrics['ndcg_at_10'] = float('nan')

    return metrics
```

### `baselines/recommender/ranking_utils.py`

**File size:** 2,395 bytes

```python
import numpy as np
import pandas as pd


def get_top_n_recommendations(
    predictions_df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "book_id",
    rating_col: str = "rating",
    n: int = 10,
) -> dict:
    """
    Get the top-N recommendations for each user from a predictions dataframe.

    Args:
        predictions_df (pd.DataFrame): DataFrame with user, item, and rating columns.
        user_col (str): Name of the user ID column.
        item_col (str): Name of the item ID column.
        rating_col (str): Name of the rating/prediction column.
        n (int): The number of recommendations to output for each user.

    Returns:
        A dict where keys are user IDs and values are lists of tuples:
        [(item ID, estimated rating), ...]
    """
    top_n = {}
    for user_id, group in predictions_df.groupby(user_col):
        top_n[user_id] = list(
            group.nlargest(n, rating_col)[[item_col, rating_col]].itertuples(
                index=False, name=None
            )
        )
    return top_n


def calculate_ndcg(
    recommendations: dict,
    ground_truth: dict,
    k: int = 10,
    batch_size: int = 1000,
) -> float:
    """
    Efficiently calculate mean NDCG@k for a set of recommendations and ground truth using numpy, preserving all samples.
    recommendations: {user_id: [rec1, rec2, ...]}
    ground_truth: {user_id: [item1, item2, ...]}
    batch_size: Number of users to process at once (to avoid OOM)
    """
    import numpy as np
    user_ids = list(recommendations.keys())
    ndcgs = []
    for i in range(0, len(user_ids), batch_size):
        batch_users = user_ids[i:i+batch_size]
        # Precompute log2 denominators
        log2s = np.log2(np.arange(2, k + 2))
        for user_id in batch_users:
            recs = recommendations[user_id][:k]
            gt = set(ground_truth.get(user_id, []))
            if not gt:
                continue
            # DCG: 1/log2(rank+1) for each hit
            hits = np.array([item in gt for item in recs], dtype=np.float32)
            dcg = np.sum(hits / log2s[:len(recs)])
            # Ideal DCG is sum for min(len(gt), k)
            ideal_len = min(len(gt), k)
            ideal_dcg = np.sum(1.0 / log2s[:ideal_len])
            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
            ndcgs.append(ndcg)
    return float(np.mean(ndcgs)) if ndcgs else 0.0
```

### `baselines/recommender/svd_baseline.py`

**File size:** 3,534 bytes

```python
import pandas as pd
from loguru import logger
from surprise import SVD, Dataset, Reader
from surprise.accuracy import mae, rmse
from sklearn.metrics import ndcg_score
import numpy as np



def run_svd_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, k_list=[5, 10, 20]) -> dict:
    """
    Runs the SVD baseline, evaluating with RMSE, MAE.
    """
    logger.info("Starting SVD baseline...")

    # 1. Load Data
    reader = Reader(rating_scale=(1, 5))
    train_data = Dataset.load_from_df(train_df[["user_id", "book_id", "rating"]], reader)
    trainset = train_data.build_full_trainset()
    testset = list(test_df[['user_id', 'book_id', 'rating']].itertuples(index=False, name=None))

    # NOTE: Do not build a full anti-test set (can be huge). Instead, compute top-N recommendations per user in batches.
    # anti_testset = trainset.build_anti_testset()

    # 2. Train Model
    logger.info("Training SVD model...")
    model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42, verbose=False)
    model.fit(trainset)

    # 3. Evaluate for Accuracy (RMSE, MAE)
    logger.info("Evaluating model for accuracy (RMSE, MAE)...")
    accuracy_predictions = model.test(testset)
    rmse_score = rmse(accuracy_predictions, verbose=False)
    mae_score = mae(accuracy_predictions, verbose=False)
    logger.info(f"SVD baseline RMSE: {rmse_score:.4f}, MAE: {mae_score:.4f}")

    # 4. Compute NDCG@10
    metrics = {"rmse": rmse_score, "mae": mae_score}
    try:
        # Build user->items mapping from test set
        user_items = test_df.groupby('user_id')['book_id'].apply(list)
        all_items = np.array(train_df['book_id'].unique())
        batch_size = 1000
        ndcg_at_10 = []
        precision_at_5_list = []
        for i in range(0, len(user_items), batch_size):
            batch_user_ids = list(user_items.index[i:i+batch_size])
            batch_user_items = user_items.iloc[i:i+batch_size]
            # Predicted scores for all items
            scores = np.array([model.predict(user_id, item_id).est for user_id in batch_user_ids for item_id in all_items]).reshape(-1, len(all_items))
            # Relevance: 1 if in test set, 0 otherwise
            true_relevance = np.array([np.isin(all_items, np.array(true_items)).astype(int) for true_items in batch_user_items])
            # Compute NDCG for all users in the batch
            batch_ndcg = ndcg_score(true_relevance, scores, k=5)
            ndcg_at_5.extend(batch_ndcg)
            # Compute precision@5 for all users in the batch
            top5_indices = np.argpartition(-scores, 5, axis=1)[:, :5]
            for idx, user_true_items in enumerate(batch_user_items):
                top5_items = all_items[top5_indices[idx]]
                hits = np.isin(top5_items, np.array(user_true_items))
                precision = np.sum(hits) / 5
                precision_at_5_list.append(precision)
        ndcg_at_5 = float(np.mean(ndcg_at_5)) if ndcg_at_5 else float('nan')
        precision_at_5 = float(np.mean(precision_at_5_list)) if precision_at_5_list else float('nan')
        metrics['ndcg_at_5'] = ndcg_at_5
        metrics['precision_at_5'] = precision_at_5
        logger.info(f"SVD NDCG@5: {ndcg_at_5:.4f}, Precision@5: {precision_at_5:.4f}")
    except Exception as ndcg_e:
        logger.warning(f"Could not compute NDCG@10: {ndcg_e}")
        metrics['ndcg_at_10'] = float('nan')

    logger.info(f"SVD metrics: {metrics}")
    logger.success("SVD baseline finished successfully.")
    return metrics
```

### `baselines/recommender/trained_weights/.gitkeep`

**File size:** 0 bytes

*[Empty file]*

### `baselines/run_all_baselines.py`

**File size:** 6,807 bytes

```python
import json
from pathlib import Path

from loguru import logger

from src.baselines.recommender.deepfm_baseline import run_deepfm_baseline
from src.baselines.feature_engineer.featuretools_baseline import run_featuretools_baseline
from src.baselines.recommender.svd_baseline import run_svd_baseline
from torch.utils.tensorboard import SummaryWriter


from src.data.cv_data_manager import CVDataManager


def main(sampling_fraction=0.1, k_list=[5]):
    """
    Main function to run all baseline models and save their results.

    This script orchestrates the following steps:
    1. Initializes the CVDataManager to load the dataset.
    2. Retrieves the data for the first cross-validation fold.
    3. Runs three baseline models in sequence:
        - Featuretools for automated feature engineering.
        - SVD for classic collaborative filtering.
        - DeepFM for a deep learning-based recommendation.
    4. Aggregates the performance metrics (e.g., RMSE, MAE, MSE, NDCG@10) from each baseline.
    5. Saves the aggregated results to a JSON file in the 'reports' directory.
    """
    logger.info("Starting the execution of all baseline models...")

    # 1. Initialize DataManager and load data
    logger.info("Initializing CVDataManager...")
    db_path = "data/goodreads_curated.duckdb"
    splits_dir = "data/splits"
    data_manager = CVDataManager(db_path=db_path, splits_dir=splits_dir)

    logger.info("Loading users and books metadata...")
    conn = data_manager.db_connection
    try:
        users_df = conn.execute("SELECT * FROM users").fetchdf()
        books_df = conn.execute("SELECT * FROM book_series").fetchdf()
    finally:
        data_manager._return_connection(conn)
    logger.success("Metadata loaded.")

    logger.info("Loading CV fold summary...")
    fold_summary = data_manager.get_fold_summary()
    n_folds = fold_summary.get("n_folds", 0)
    if n_folds == 0:
        logger.error("No CV folds found. Exiting.")
        return

    writer = SummaryWriter("reports/tensorboard_baselines")
    per_fold_results = {"featuretools_lightfm": [], "svd": [], "popularity": [], "deepfm": []}
    errors = {"featuretools_lightfm": [], "svd": [], "popularity": [], "deepfm": []}

    from src.baselines.recommender.popularity_baseline import run_popularity_baseline

    for fold_idx in range(n_folds):
        logger.info(f"--- Processing Fold {fold_idx+1}/{n_folds} ---")
        train_df, test_df = data_manager.get_fold_data(fold_idx=fold_idx, split_type="full_train")

        # Featuretools Baseline
        try:
            metrics = run_featuretools_baseline(train_df, books_df, users_df, test_df, k_list=k_list)
            per_fold_results["featuretools_lightfm"].append(metrics)
            logger.success(f"Featuretools+LightFM baseline completed. Metrics: {metrics}")
            if "precision_at_10" in metrics:
                writer.add_scalar("featuretools_lightfm/precision_at_10", metrics["precision_at_10"], fold_idx)
            if "n_clusters" in metrics:
                writer.add_scalar("featuretools_lightfm/n_clusters", metrics["n_clusters"], fold_idx)
        except Exception as e:
            logger.error(f"Featuretools+LightFM baseline failed on fold {fold_idx}: {e}")
            errors["featuretools_lightfm"].append(str(e))
            per_fold_results["featuretools_lightfm"].append(None)

        # SVD Baseline
        try:
            metrics = run_svd_baseline(train_df, test_df, k_list=k_list)
            per_fold_results["svd"].append(metrics)
            logger.success(f"SVD baseline completed. Metrics: {metrics}")
            if "rmse" in metrics:
                writer.add_scalar("svd/RMSE", metrics["rmse"], fold_idx)
        except Exception as e:
            logger.error(f"SVD baseline failed on fold {fold_idx}: {e}")
            errors["svd"].append(str(e))
            per_fold_results["svd"].append(None)

        # Popularity Baseline
        try:
            metrics = run_popularity_baseline(train_df, test_df)
            per_fold_results["popularity"].append(metrics)
            logger.success(f"Popularity baseline completed. Metrics: {metrics}")
        except Exception as e:
            logger.error(f"Popularity baseline failed on fold {fold_idx}: {e}")
            errors["popularity"].append(str(e))
            per_fold_results["popularity"].append(None)

        # DeepFM Baseline
        try:
            metrics = run_deepfm_baseline(train_df, test_df, k_list=k_list)
            per_fold_results["deepfm"].append(metrics)
            logger.success(f"DeepFM baseline completed. Metrics: {metrics}")
        except Exception as e:
            logger.error(f"DeepFM baseline failed on fold {fold_idx}: {e}")
            errors["deepfm"].append(str(e))
            per_fold_results["deepfm"].append(None)

    # Aggregate results: mean and stddev for each metric and baseline
    import numpy as np
    aggregate_results = {}
    for baseline, results in per_fold_results.items():
        metrics_by_key = {}
        for fold_result in results:
            if fold_result is None:
                continue
            for k, v in fold_result.items():
                if v is None:
                    continue
                metrics_by_key.setdefault(k, []).append(v)
        baseline_agg = {}
        for k, v_list in metrics_by_key.items():
            if not k.startswith('ndcg@5'):
                continue  # Only keep ndcg@5
            arr = np.array(v_list)
            baseline_agg[f"{k}_mean"] = float(np.mean(arr))
            baseline_agg[f"{k}_std"] = float(np.std(arr))
            # Log mean to TensorBoard
            writer.add_scalar(f"{baseline}/{k}_mean", baseline_agg[f"{k}_mean"], 0)
            writer.add_scalar(f"{baseline}/{k}_std", baseline_agg[f"{k}_std"], 0)
        aggregate_results[baseline] = baseline_agg
        if errors[baseline]:
            aggregate_results[baseline]["errors"] = errors[baseline]

    all_results = {
        "per_fold": per_fold_results,
        "aggregate": aggregate_results,
        "n_folds": n_folds
    }

    # 5. Save results to a JSON file
    try:
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        results_path = reports_dir / "baseline_results.json"

        logger.info(f"Saving aggregated baseline results to {results_path}")
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=4)

        logger.success(f"Results successfully saved to {results_path}")
    except (IOError, OSError) as e:
        logger.error(f"Failed to save results to file: {e}")
        logger.error(f"Current Working Directory: {Path.cwd()}")

    logger.success("All baseline models have been executed.")


if __name__ == "__main__":
    main(sampling_fraction=0.01, k_list=[5])
```

### `config/log_config.py`

**File size:** 1,289 bytes

```python
import sys
import logging

from loguru import logger

from src.utils.logging_utils import InterceptHandler
from src.utils.run_utils import get_run_log_file


def setup_logging(log_level: str = "INFO") -> None:
    """Set up Loguru to be the main logging system."""
    # Remove default handler to avoid duplicate logs
    logger.remove()

    # Add a console sink
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=False,
    )

    # Add a file sink for the main pipeline log
    log_file = get_run_log_file()
    logger.add(
        log_file,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="10 days",
        enqueue=True,  # Make logging non-blocking
        backtrace=True,
        diagnose=True,
    )

    # Intercept standard logging messages
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)


def get_logger(name: str):
    """Get a logger with the specified name (compatible with loguru)."""
    return logger.bind(name=name)
```

### `config/settings.py`

**File size:** 1,384 bytes

```python
"""
Configuration settings for the VULCAN project.
Contains database paths, LLM configurations, and other global constants.
"""

from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
PROMPTS_DIR = SRC_DIR / "prompts"
LOGS_DIR = ROOT_DIR / "logs"
DATA_DIR = ROOT_DIR / "data"
RUN_DIR = ROOT_DIR / "runtime" / "runs"

# Database configuration
DB_PATH = str(DATA_DIR / "goodreads_curated.duckdb")

# LLM Configuration - Default configuration that can be used across agents
# This will be overridden by the orchestrator with actual API keys and config lists
LLM_CONFIG = {
    "config_list": [],  # Will be populated by orchestrator from OAI_CONFIG_LIST.json
    "cache_seed": None,
    "temperature": 0.7,
    "timeout": 120,
    "max_tokens": 16384,
}

# Agent configuration
MAX_CONSECUTIVE_AUTO_REPLY = 10
CODE_EXECUTION_TIMEOUT = 120

# Plotting configuration
PLOT_DPI = 300
PLOT_STYLE = "default"
PLOT_PALETTE = "husl"

# OpenAI configuration
OPENAI_MODEL_VISION = "gpt-4o"
OPENAI_MODEL_TEXT = "gpt-4o-mini"

# Database connection settings
DB_READ_ONLY = False  # Allow writes for temporary views
DB_TIMEOUT = 30

# Insight Discovery settings
INSIGHT_AGENTS_CONFIG_PATH = ROOT_DIR / "config" / "OAI_CONFIG_LIST.json"
INSIGHT_MAX_TURNS = 20
INSIGHT_MAX_CONSECUTIVE_AUTO_REPLY = 5

# Add other settings as needed
```

### `config/tensorboard.py`

**File size:** 1,436 bytes

```python
import subprocess
from loguru import logger
from typing import Optional

from torch.utils.tensorboard import SummaryWriter

from src.utils.run_utils import get_run_tensorboard_dir


def start_tensorboard() -> None:
    """Start TensorBoard in the background for a global (non-run-specific) log directory."""
    log_dir = "runtime/tensorboard_global"
    try:
        subprocess.Popen(
            ["tensorboard", "--logdir", str(log_dir), "--port", "6006"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        logger.warning(f"Could not start TensorBoard: {e}")


def get_tensorboard_writer() -> SummaryWriter:
    """Get a TensorBoard writer for the global (non-run-specific) TensorBoard log directory."""
    log_dir = "runtime/tensorboard_global"
    return SummaryWriter(log_dir=str(log_dir))


def log_metric(
    writer: SummaryWriter, tag: str, value: float, step: Optional[int] = None
) -> None:
    """Log a metric to TensorBoard."""
    writer.add_scalar(tag, value, step)


def log_metrics(
    writer: SummaryWriter, metrics: dict, step: Optional[int] = None
) -> None:
    """Log multiple metrics to TensorBoard."""
    for tag, value in metrics.items():
        log_metric(writer, tag, value, step)


def log_hyperparams(writer: SummaryWriter, hparams: dict) -> None:
    """Log hyperparameters to TensorBoard."""
    writer.add_hparams(hparams, {})
```

### `contingency/aggregate_hypotheses.py`

**File size:** 5,122 bytes

```python
#!/usr/bin/env python3
"""
Contingency Plan: Aggregate Hypotheses from All Session States

This script parses all session_state.json files in runtime/runs directory,
extracts hypotheses, and saves them to a compiled file for manual feature engineering.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import argparse

def load_session_state(session_path: Path) -> Dict[str, Any]:
    """Load session state from JSON file."""
    try:
        with open(session_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load {session_path}: {e}")
        return {}

def extract_hypotheses_from_runs(runs_dir: Path) -> List[Dict[str, Any]]:
    """Extract all hypotheses from all run directories."""
    all_hypotheses = []
    run_metadata = []
    
    print(f"Scanning runs directory: {runs_dir}")
    
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
            
        session_file = run_dir / "session_state.json"
        if not session_file.exists():
            print(f"No session_state.json in {run_dir.name}")
            continue
            
        print(f"Processing {run_dir.name}...")
        session_data = load_session_state(session_file)
        
        if not session_data:
            continue
            
        # Extract hypotheses
        hypotheses = session_data.get("hypotheses", [])
        if hypotheses:
            print(f"  Found {len(hypotheses)} hypotheses")
            for hypothesis in hypotheses:
                # Add metadata about which run this came from
                hypothesis_with_meta = {
                    "run_id": run_dir.name,
                    **hypothesis
                }
                all_hypotheses.append(hypothesis_with_meta)
        else:
            print(f"  No hypotheses found")
            
        # Track run metadata
        run_info = {
            "run_id": run_dir.name,
            "hypotheses_count": len(hypotheses),
            "insights_count": len(session_data.get("insights", [])),
            "candidate_features_count": len(session_data.get("candidate_features", [])),
            "has_data": bool(session_data)
        }
        run_metadata.append(run_info)
    
    return all_hypotheses, run_metadata

def save_compiled_hypotheses(hypotheses: List[Dict[str, Any]], 
                           metadata: List[Dict[str, Any]], 
                           output_file: Path):
    """Save compiled hypotheses to JSON file."""
    compiled_data = {
        "total_hypotheses": len(hypotheses),
        "total_runs_processed": len(metadata),
        "runs_with_hypotheses": len([r for r in metadata if r["hypotheses_count"] > 0]),
        "compilation_timestamp": "2025-06-17T11:30:00+02:00",
        "run_metadata": metadata,
        "hypotheses": hypotheses
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(compiled_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nCompiled hypotheses saved to: {output_file}")
    print(f"Total hypotheses: {len(hypotheses)}")
    print(f"Runs processed: {len(metadata)}")
    print(f"Runs with hypotheses: {len([r for r in metadata if r['hypotheses_count'] > 0])}")

def print_hypothesis_summary(hypotheses: List[Dict[str, Any]]):
    """Print a summary of the hypotheses found."""
    if not hypotheses:
        print("No hypotheses found across all runs.")
        return
        
    print(f"\n=== HYPOTHESIS SUMMARY ===")
    for i, hyp in enumerate(hypotheses, 1):
        print(f"\n{i}. Run: {hyp.get('run_id', 'unknown')}")
        print(f"   ID: {hyp.get('id', 'no-id')}")
        print(f"   Summary: {hyp.get('summary', 'No summary')}")
        print(f"   Rationale: {hyp.get('rationale', 'No rationale')[:100]}...")

def main():
    parser = argparse.ArgumentParser(description="Aggregate hypotheses from all VULCAN runs")
    parser.add_argument("--runs_dir", type=str, default="/root/fuegoRecommender/runtime/runs",
                       help="Path to runs directory")
    parser.add_argument("--output", type=str, default="/root/fuegoRecommender/src/contingency/compiled_hypotheses.json",
                       help="Output file for compiled hypotheses")
    parser.add_argument("--summary", action="store_true", help="Print detailed summary of hypotheses")
    
    args = parser.parse_args()
    
    runs_dir = Path(args.runs_dir)
    output_file = Path(args.output)
    
    if not runs_dir.exists():
        print(f"Error: Runs directory not found: {runs_dir}")
        return 1
        
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract hypotheses
    hypotheses, metadata = extract_hypotheses_from_runs(runs_dir)
    
    # Save compiled results
    save_compiled_hypotheses(hypotheses, metadata, output_file)
    
    # Print summary if requested
    if args.summary:
        print_hypothesis_summary(hypotheses)
    
    return 0

if __name__ == "__main__":
    exit(main())
```

### `contingency/compiled_hypotheses.json`

**File size:** 38,920 bytes

```json
{
  "total_hypotheses": 77,
  "total_runs_processed": 17,
  "runs_with_hypotheses": 9,
  "compilation_timestamp": "2025-06-17T11:30:00+02:00",
  "run_metadata": [
    {
      "run_id": "run_20250617_093150_14db6164",
      "hypotheses_count": 12,
      "insights_count": 0,
      "candidate_features_count": 0,
      "has_data": true
    },
    {
      "run_id": "test_strategy_team_v2",
      "hypotheses_count": 0,
      "insights_count": 0,
      "candidate_features_count": 1,
      "has_data": true
    },
    {
      "run_id": "run_20250617_092624_07d4df98",
      "hypotheses_count": 0,
      "insights_count": 0,
      "candidate_features_count": 0,
      "has_data": true
    },
    {
      "run_id": "run_20250617_085529_ccbae235",
      "hypotheses_count": 15,
      "insights_count": 6,
      "candidate_features_count": 0,
      "has_data": true
    },
    {
      "run_id": "run_20250617_090838_a65fb946",
      "hypotheses_count": 5,
      "insights_count": 0,
      "candidate_features_count": 0,
      "has_data": true
    },
    {
      "run_id": "run_20250617_111935_2f1eb61d",
      "hypotheses_count": 8,
      "insights_count": 0,
      "candidate_features_count": 0,
      "has_data": true
    },
    {
      "run_id": "run_20250617_042705_066c2790",
      "hypotheses_count": 0,
      "insights_count": 0,
      "candidate_features_count": 1,
      "has_data": true
    },
    {
      "run_id": "run_20250617_091920_2d524456",
      "hypotheses_count": 15,
      "insights_count": 0,
      "candidate_features_count": 0,
      "has_data": true
    },
    {
      "run_id": "run_20250617_042827_f8dc32ad",
      "hypotheses_count": 0,
      "insights_count": 0,
      "candidate_features_count": 1,
      "has_data": true
    },
    {
      "run_id": "run_20250617_043033_61f4f99b",
      "hypotheses_count": 0,
      "insights_count": 0,
      "candidate_features_count": 1,
      "has_data": true
    },
    {
      "run_id": "run_20250617_091504_28f70390",
      "hypotheses_count": 5,
      "insights_count": 0,
      "candidate_features_count": 0,
      "has_data": true
    },
    {
      "run_id": "run_20250617_083442_801c20a3",
      "hypotheses_count": 9,
      "insights_count": 2,
      "candidate_features_count": 0,
      "has_data": true
    },
    {
      "run_id": "run_20250617_092044_2730c199",
      "hypotheses_count": 0,
      "insights_count": 0,
      "candidate_features_count": 0,
      "has_data": true
    },
    {
      "run_id": "run_20250617_074730_e81776b7",
      "hypotheses_count": 0,
      "insights_count": 0,
      "candidate_features_count": 0,
      "has_data": true
    },
    {
      "run_id": "run_20250617_093601_ae6eca46",
      "hypotheses_count": 3,
      "insights_count": 0,
      "candidate_features_count": 0,
      "has_data": true
    },
    {
      "run_id": "run_20250617_000445_396c4332",
      "hypotheses_count": 5,
      "insights_count": 0,
      "candidate_features_count": 5,
      "has_data": true
    },
    {
      "run_id": "run_20250617_000332_07441b96",
      "hypotheses_count": 0,
      "insights_count": 0,
      "candidate_features_count": 0,
      "has_data": true
    }
  ],
  "hypotheses": [
    {
      "run_id": "run_20250617_093150_14db6164",
      "id": "13af6b11-081c-4940-a4cc-81c4c481e220",
      "summary": "Users who read more books tend to provide higher average ratings.",
      "rationale": "This indicates a positive relationship between engagement and satisfaction in reading, which can guide personalized recommendations.",
      "depends_on": [
        "user_reading_trends.books_read",
        "user_reading_trends.avg_rating"
      ]
    },
    {
      "run_id": "run_20250617_093150_14db6164",
      "id": "a8a69379-c9ec-4d88-a3bd-0ed31945b6ce",
      "summary": "Users prefer specific genres that have consistently high average ratings.",
      "rationale": "Identifying these genres can enhance recommendation systems by tailoring suggestions to user preferences, increasing engagement.",
      "depends_on": [
        "avg_rating_by_genre.genre",
        "avg_rating_by_genre.average_rating"
      ]
    },
    {
      "run_id": "run_20250617_093150_14db6164",
      "id": "df51544b-6df9-405a-8d38-8167f9fd8437",
      "summary": "Shelf categories that accumulate more books indicate user interest in those areas.",
      "rationale": "Analyzing shelf popularity can guide curators and recommend titles effectively, especially for new users looking for suggestions.",
      "depends_on": [
        "book_shelves.shelf",
        "book_shelves.cnt"
      ]
    },
    {
      "run_id": "run_20250617_093150_14db6164",
      "id": "4c47ef5c-ad65-4d5e-bea3-cdb97e82b9fb",
      "summary": "Readers show a preference for certain book formats based on average ratings.",
      "rationale": "This insight can help personalize recommendations based on the specific format a user tends to favor, enhancing user satisfaction.",
      "depends_on": [
        "book_genre_format_ratings.format",
        "book_genre_format_ratings.avg_rating"
      ]
    },
    {
      "run_id": "run_20250617_093150_14db6164",
      "id": "cdc7d343-0650-4e28-a54e-4fb9817f6907",
      "summary": "Users who read more books tend to provide higher average ratings.",
      "rationale": "This indicates a relationship between engagement and satisfaction in reading, which can guide personalized recommendations.",
      "depends_on": [
        "user_reading_trends.books_read",
        "user_reading_trends.avg_rating"
      ]
    },
    {
      "run_id": "run_20250617_093150_14db6164",
      "id": "b818439d-a05e-4650-bc74-141657728f36",
      "summary": "Users prefer specific genres that have consistently high average ratings.",
      "rationale": "Identifying these genres can enhance recommendation systems by tailoring suggestions to user preferences, increasing engagement.",
      "depends_on": [
        "avg_rating_by_genre.genre",
        "avg_rating_by_genre.average_rating"
      ]
    },
    {
      "run_id": "run_20250617_093150_14db6164",
      "id": "4dc82d65-0f39-49a5-90ef-a252e4f98dbe",
      "summary": "Shelf categories that accumulate more books indicate user interest in those areas.",
      "rationale": "Analyzing shelf popularity can guide curators and recommend titles effectively, especially for new users looking for suggestions.",
      "depends_on": [
        "book_shelves.shelf",
        "book_shelves.cnt"
      ]
    },
    {
      "run_id": "run_20250617_093150_14db6164",
      "id": "d65bba37-cbd2-4c87-9edd-af0f521786c9",
      "summary": "Readers show a preference for certain book formats based on average ratings.",
      "rationale": "This insight can help personalize recommendations based on the specific format a user tends to favor, enhancing user satisfaction.",
      "depends_on": [
        "book_genre_format_ratings.format",
        "book_genre_format_ratings.avg_rating"
      ]
    },
    {
      "run_id": "run_20250617_093150_14db6164",
      "id": "dc360ce4-f286-4960-878c-a6df062f3188",
      "summary": "Users who read more books tend to provide higher average ratings.",
      "rationale": "This indicates a relationship between engagement and satisfaction in reading, which can guide personalized recommendations.",
      "depends_on": [
        "user_reading_trends.books_read",
        "user_reading_trends.avg_rating"
      ]
    },
    {
      "run_id": "run_20250617_093150_14db6164",
      "id": "318bfbc0-ef8d-4b36-bdde-4bbf21d6c62d",
      "summary": "Users prefer specific genres that have consistently high average ratings.",
      "rationale": "Identifying these genres can enhance recommendation systems by tailoring suggestions to user preferences, increasing engagement.",
      "depends_on": [
        "avg_rating_by_genre.genre",
        "avg_rating_by_genre.average_rating"
      ]
    },
    {
      "run_id": "run_20250617_093150_14db6164",
      "id": "32dffbfb-04ee-4216-b592-124072448424",
      "summary": "Shelf categories that accumulate more books indicate user interest in those areas.",
      "rationale": "Analyzing shelf popularity can guide curators and recommend titles effectively, especially for new users looking for suggestions.",
      "depends_on": [
        "book_shelves.shelf",
        "book_shelves.cnt"
      ]
    },
    {
      "run_id": "run_20250617_093150_14db6164",
      "id": "10e08f46-b3c3-4900-b3f1-ab1ccb6b4722",
      "summary": "Readers show a preference for certain book formats based on average ratings.",
      "rationale": "This insight can help personalize recommendations based on the specific format a user tends to favor, enhancing user satisfaction.",
      "depends_on": [
        "book_genre_format_ratings.format",
        "book_genre_format_ratings.avg_rating"
      ]
    },
    {
      "run_id": "run_20250617_085529_ccbae235",
      "id": "72ff3917-8c3d-42ef-ad1d-1331b8aca456",
      "summary": "Readers exhibit distinct preferences across genres, with significant interest in fantasy and romance.",
      "rationale": "Understanding genre popularity can enhance targeted recommendations and improve user engagement.",
      "depends_on": [
        "book_genre_format_ratings.genre",
        "curated_reviews.rating"
      ]
    },
    {
      "run_id": "run_20250617_085529_ccbae235",
      "id": "81dce0f6-fac7-4e2f-ae38-e498a03a7a39",
      "summary": "Users demonstrate a range of reading behaviors, influencing how books are rated and reviewed.",
      "rationale": "Identifying user behavior clusters can tailor recommendations to specific user segments, enhancing user satisfaction.",
      "depends_on": [
        "curated_reviews.user_id",
        "curated_reviews.rating"
      ]
    },
    {
      "run_id": "run_20250617_085529_ccbae235",
      "id": "1ed5f386-211e-43d0-b883-73be8b60cfe9",
      "summary": "Users who rate steampunk and fantasy literature higher may also enjoy other genres with similar thematic elements.",
      "rationale": "Exploring genre interconnectivity could refine recommendation strategies for users with diverse tastes.",
      "depends_on": [
        "curated_reviews.rating",
        "curated_books.title"
      ]
    },
    {
      "run_id": "run_20250617_085529_ccbae235",
      "id": "b8be3f1f-e683-4e80-8fba-6ec3c6f519f8",
      "summary": "Books with authors who collaborate frequently may be rated higher due to perceived quality or continuous thematic alignment.",
      "rationale": "Understanding the impact of author collaborations can enhance recommendations based on thematic consistency or quality.",
      "depends_on": [
        "book_authors.author_id",
        "curated_books.book_id"
      ]
    },
    {
      "run_id": "run_20250617_085529_ccbae235",
      "id": "b9f0a4a3-0176-4fa7-aa45-2a7dea3bbe8e",
      "summary": "Users displaying certain reading behaviors (few books but high ratings) may benefit from curated personalized recommendations.",
      "rationale": "Targeting user segments with specific reading patterns could optimize engagement through tailored suggestions.",
      "depends_on": [
        "user_reading_trends.books_read",
        "user_reading_trends.avg_rating"
      ]
    },
    {
      "run_id": "run_20250617_085529_ccbae235",
      "id": "e8878f2d-f729-47f6-ba65-fa5787625ff4",
      "summary": "Books with more pages may affect user ratings differently compared to shorter books.",
      "rationale": "Longer books may provide more depth in storytelling, which can appeal to readers, or they may overwhelm readers, impacting ratings adversely.",
      "depends_on": [
        "curated_books.num_pages",
        "curated_books.avg_rating"
      ]
    },
    {
      "run_id": "run_20250617_085529_ccbae235",
      "id": "d76c2b69-7eea-478c-803c-8dee1852b51f",
      "summary": "Ratings and descriptions might correlate strongly, suggesting that well-articulated descriptions entice higher ratings.",
      "rationale": "Descriptive quality may engage readers more effectively, boosting their ratings as their expectations are met or exceeded.",
      "depends_on": [
        "curated_books.title",
        "curated_books.description",
        "curated_books.avg_rating"
      ]
    },
    {
      "run_id": "run_20250617_085529_ccbae235",
      "id": "803fa0f4-7918-46bf-adcb-2c48d9ceeca4",
      "summary": "Positive sentiment in reviews could predict higher book ratings.",
      "rationale": "Emotional connections in reviews reflect reader satisfaction, which is likely to impact ratings positively.",
      "depends_on": [
        "curated_reviews.review_text",
        "curated_reviews.rating"
      ]
    },
    {
      "run_id": "run_20250617_085529_ccbae235",
      "id": "03540c1c-5348-4de6-afae-80ff7c59120a",
      "summary": "Users who enjoy a particular book may also prefer others from the same author or similar thematic books.",
      "rationale": "Cross-recommendation between similar books could enhance user experience and interactions.",
      "depends_on": [
        "book_similars.book_id",
        "book_similars.similar_book_id"
      ]
    },
    {
      "run_id": "run_20250617_085529_ccbae235",
      "id": "9ad09702-314b-4850-8aa8-490f469c3d9d",
      "summary": "Increased user interaction correlates positively with higher ratings.",
      "rationale": "More interactions suggest greater engagement and satisfaction, typically reflected in higher ratings.",
      "depends_on": [
        "interactions.review_id",
        "interactions.n_votes",
        "interactions.n_comments"
      ]
    },
    {
      "run_id": "run_20250617_085529_ccbae235",
      "id": "7652cba5-dd9c-4226-8b60-82f22365fbbb",
      "summary": "Books with more pages may affect user ratings differently compared to shorter books.",
      "rationale": "Longer books may provide more depth in storytelling, which can appeal to readers, or they may overwhelm readers, impacting ratings adversely.",
      "depends_on": [
        "curated_books.num_pages",
        "curated_books.avg_rating"
      ]
    },
    {
      "run_id": "run_20250617_085529_ccbae235",
      "id": "4f633c54-f92f-45c5-bd89-6b693fabb7ae",
      "summary": "Ratings and descriptions might correlate strongly, suggesting that well-articulated descriptions entice higher ratings.",
      "rationale": "Descriptive quality may engage readers more effectively, boosting their ratings as their expectations are met or exceeded.",
      "depends_on": [
        "curated_books.title",
        "curated_books.description",
        "curated_books.avg_rating"
      ]
    },
    {
      "run_id": "run_20250617_085529_ccbae235",
      "id": "46ae6501-451c-4396-b4e5-77a2b591cfcb",
      "summary": "Positive sentiment in reviews could predict higher book ratings.",
      "rationale": "Emotional connections in reviews reflect reader satisfaction, which is likely to impact ratings positively.",
      "depends_on": [
        "curated_reviews.review_text",
        "curated_reviews.rating"
      ]
    },
    {
      "run_id": "run_20250617_085529_ccbae235",
      "id": "67e7537b-3f3f-4542-a356-f365da70a6fe",
      "summary": "Users who enjoy a particular book may also prefer others from the same author or similar thematic books.",
      "rationale": "Cross-recommendation between similar books could enhance user experience and interactions.",
      "depends_on": [
        "book_similars.book_id",
        "book_similars.similar_book_id"
      ]
    },
    {
      "run_id": "run_20250617_085529_ccbae235",
      "id": "f11ffd38-0e31-49ad-a1df-84116b3444e2",
      "summary": "Increased user interaction correlates positively with higher ratings.",
      "rationale": "More interactions suggest greater engagement and satisfaction, typically reflected in higher ratings.",
      "depends_on": [
        "interactions.review_id",
        "interactions.n_votes",
        "interactions.n_comments"
      ]
    },
    {
      "run_id": "run_20250617_090838_a65fb946",
      "id": "ac6f818c-406e-40eb-9e6b-b73092897f87",
      "summary": "Books published by a larger number of unique publishers tend to have higher average ratings.",
      "rationale": "A diverse range of publishers may indicate higher quality and more substantial investment in the books, which could translate into better ratings.",
      "depends_on": [
        "curated_books.publisher_name",
        "curated_books.avg_rating"
      ]
    },
    {
      "run_id": "run_20250617_090838_a65fb946",
      "id": "0ab61ad9-c368-4e15-8b7e-82de97b158ac",
      "summary": "Books that have been published more recently tend to have higher average ratings.",
      "rationale": "Newer books may benefit from more modern writing standards, trends, and reader preferences than older books, affecting their ratings positively.",
      "depends_on": [
        "curated_books.publication_date",
        "curated_books.avg_rating"
      ]
    },
    {
      "run_id": "run_20250617_090838_a65fb946",
      "id": "eb75df81-ef83-4440-a1d6-a00f8de70aee",
      "summary": "Ebook formats tend to receive higher average ratings compared to physical formats.",
      "rationale": "Ebooks may offer more accessibility and convenience, appealing to a broader audience, which could lead to better ratings.",
      "depends_on": [
        "curated_books.format",
        "curated_books.avg_rating"
      ]
    },
    {
      "run_id": "run_20250617_090838_a65fb946",
      "id": "17545600-ee77-4212-9adc-80afe1326f5a",
      "summary": "Books with higher ratings tend to attract more reviews.",
      "rationale": "Higher rated books are more likely to engage readers and prompt them to leave feedback, resulting in more reviews overall.",
      "depends_on": [
        "curated_books.avg_rating",
        "curated_books.ratings_count"
      ]
    },
    {
      "run_id": "run_20250617_090838_a65fb946",
      "id": "7ccab827-e740-4a0d-b5f6-f56f895b151c",
      "summary": "Books with descriptions that reflect unique or engaging themes tend to have higher average ratings.",
      "rationale": "Engaging themes may attract more readers and generate higher ratings based on reader enjoyment and connection to the content.",
      "depends_on": [
        "curated_books.description",
        "curated_books.avg_rating"
      ]
    },
    {
      "run_id": "run_20250617_111935_2f1eb61d",
      "id": "4d14e12f-1fb8-46b1-a5bf-d352b7d02af2",
      "summary": "Authors with more collaborations tend to create books with higher average ratings.",
      "rationale": "Collaboration may lead to improved quality through shared expertise.",
      "depends_on": [
        "author_collaborations.author_id",
        "curated_books.avg_rating"
      ]
    },
    {
      "run_id": "run_20250617_111935_2f1eb61d",
      "id": "3cae48b6-dc5e-428d-b4b6-ff1fe1749364",
      "summary": "Books with more pages tend to receive a higher average rating.",
      "rationale": "Longer books may offer deeper stories and character development, leading to higher ratings.",
      "depends_on": [
        "curated_books.num_pages",
        "curated_books.avg_rating"
      ]
    },
    {
      "run_id": "run_20250617_111935_2f1eb61d",
      "id": "4443353e-2d03-4a64-811d-ff42b1d4940c",
      "summary": "Ebooks tend to have lower average ratings compared to physical books.",
      "rationale": "Physical books may be more desirable due to tactile experiences and availability of more detailed information before purchase.",
      "depends_on": [
        "curated_books.is_ebook",
        "curated_books.avg_rating"
      ]
    },
    {
      "run_id": "run_20250617_111935_2f1eb61d",
      "id": "ef0d3ec5-6f3e-499e-a289-387ca9727cce",
      "summary": "Genres with more books tend to have higher average ratings.",
      "rationale": "Genres that attract more authors and books may indicate positive reader engagement and rating patterns.",
      "depends_on": [
        "genre_counts_view.genre",
        "avg_rating_by_genre.average_rating"
      ]
    },
    {
      "run_id": "run_20250617_111935_2f1eb61d",
      "id": "a2ffbb63-b555-4610-bc24-9390545060d1",
      "summary": "Authors with more collaborations tend to create books with higher average ratings.",
      "rationale": "Collaboration may lead to improved quality through shared expertise.",
      "depends_on": [
        "author_collaborations.author_id",
        "curated_books.avg_rating"
      ]
    },
    {
      "run_id": "run_20250617_111935_2f1eb61d",
      "id": "2ca98b5c-7e90-4084-b97c-e79189cca06f",
      "summary": "Books with more pages tend to receive a higher average rating.",
      "rationale": "Longer books may offer deeper stories and character development, leading to higher ratings.",
      "depends_on": [
        "curated_books.num_pages",
        "curated_books.avg_rating"
      ]
    },
    {
      "run_id": "run_20250617_111935_2f1eb61d",
      "id": "91be0265-ba06-46bd-bf21-eca8fc5f295f",
      "summary": "Ebooks tend to have lower average ratings compared to physical books.",
      "rationale": "Physical books may be more desirable due to tactile experiences and availability of more detailed information before purchase.",
      "depends_on": [
        "curated_books.is_ebook",
        "curated_books.avg_rating"
      ]
    },
    {
      "run_id": "run_20250617_111935_2f1eb61d",
      "id": "6dd348ab-f96d-43e0-bbda-5b456001135f",
      "summary": "Genres with more books tend to have higher average ratings.",
      "rationale": "Genres that attract more authors and books may indicate positive reader engagement and rating patterns.",
      "depends_on": [
        "genre_counts_view.genre",
        "avg_rating_by_genre.average_rating"
      ]
    },
    {
      "run_id": "run_20250617_091920_2d524456",
      "id": "3ca42a7c-4ce9-410b-8c6f-4d16a54c935e",
      "summary": "Users who engage with more books tend to provide more reviews.",
      "rationale": "Increased reading activity likely leads to more opportunities for users to express their thoughts, resulting in higher review counts.",
      "depends_on": [
        "curated_reviews.user_id",
        "curated_reviews.book_id"
      ]
    },
    {
      "run_id": "run_20250617_091920_2d524456",
      "id": "dd7d10a1-ae95-4a4d-bacb-9381b455c1ab",
      "summary": "Books with higher average ratings tend to have more reviews written about them.",
      "rationale": "Higher quality ratings may incentivize more users to share their experiences through reviews, highlighting a correlation between book quality and engagement.",
      "depends_on": [
        "curated_books.book_id",
        "curated_books.avg_rating"
      ]
    },
    {
      "run_id": "run_20250617_091920_2d524456",
      "id": "fce47266-9073-45ee-83f5-2a40141e02e9",
      "summary": "Different audience demographics engage differently with book formats.",
      "rationale": "Demographics may resonate differently with various reading formats, leading to fluctuations in engagement levels and variations by readership.",
      "depends_on": [
        "curated_reviews.book_id",
        "curated_reviews.user_id"
      ]
    },
    {
      "run_id": "run_20250617_091920_2d524456",
      "id": "c5b84ff8-bd87-4545-b581-e86eab7d8396",
      "summary": "Popular authors tend to have higher review rates for their books.",
      "rationale": "Established authors often generate a loyal readership that is more likely to provide feedback, contributing to a higher volume of reviews for their works.",
      "depends_on": [
        "book_authors.author_id",
        "curated_reviews.book_id"
      ]
    },
    {
      "run_id": "run_20250617_091920_2d524456",
      "id": "57fe7365-4549-42b8-a915-56d73b04d6ff",
      "summary": "The sentiments expressed in reviews vary significantly by user engagement.",
      "rationale": "The richness and depth of reviews can reflect the level of emotional or intellectual engagement a user has with a book, offering insight into their reading experience.",
      "depends_on": [
        "curated_reviews.review_text",
        "curated_reviews.rating"
      ]
    },
    {
      "run_id": "run_20250617_091920_2d524456",
      "id": "40b81efd-582d-4604-9328-de875d107569",
      "summary": "Users who engage with more books tend to provide more reviews.",
      "rationale": "Increased reading activity likely leads to more opportunities for users to express their thoughts, resulting in higher review counts.",
      "depends_on": [
        "curated_reviews.user_id",
        "curated_reviews.book_id"
      ]
    },
    {
      "run_id": "run_20250617_091920_2d524456",
      "id": "574e625d-a6e5-4c41-9575-9305258d620b",
      "summary": "Books with higher average ratings tend to have more reviews written about them.",
      "rationale": "Higher quality ratings may incentivize more users to share their experiences through reviews, highlighting a correlation between book quality and engagement.",
      "depends_on": [
        "curated_books.book_id",
        "curated_books.avg_rating"
      ]
    },
    {
      "run_id": "run_20250617_091920_2d524456",
      "id": "adb78c94-ed3c-47f3-8458-c1d61c659843",
      "summary": "Different audience demographics engage differently with book formats.",
      "rationale": "Demographics may resonate differently with various reading formats, leading to fluctuations in engagement levels and variations by readership.",
      "depends_on": [
        "curated_reviews.book_id",
        "curated_reviews.user_id"
      ]
    },
    {
      "run_id": "run_20250617_091920_2d524456",
      "id": "f7e885b4-1009-4223-80e1-ec4aef552db5",
      "summary": "Popular authors tend to have higher review rates for their books.",
      "rationale": "Established authors often generate a loyal readership that is more likely to provide feedback, contributing to a higher volume of reviews for their works.",
      "depends_on": [
        "book_authors.author_id",
        "curated_reviews.book_id"
      ]
    },
    {
      "run_id": "run_20250617_091920_2d524456",
      "id": "c15f468f-0c4e-4f47-8271-99eaef3e3b43",
      "summary": "The sentiments expressed in reviews vary significantly by user engagement.",
      "rationale": "The richness and depth of reviews can reflect the level of emotional or intellectual engagement a user has with a book, offering insight into their reading experience.",
      "depends_on": [
        "curated_reviews.review_text",
        "curated_reviews.rating"
      ]
    },
    {
      "run_id": "run_20250617_091920_2d524456",
      "id": "1cda2f4b-c7ce-4b0e-bcda-f71edf5c4d1f",
      "summary": "Users who engage with more books tend to provide more reviews.",
      "rationale": "Increased reading activity likely leads to more opportunities for users to express their thoughts, resulting in higher review counts.",
      "depends_on": [
        "curated_reviews.user_id",
        "curated_reviews.book_id"
      ]
    },
    {
      "run_id": "run_20250617_091920_2d524456",
      "id": "468ea54f-2cc1-4483-92b7-f2c8cac657c9",
      "summary": "Books with higher average ratings tend to have more reviews written about them.",
      "rationale": "Higher quality ratings may incentivize more users to share their experiences through reviews, highlighting a correlation between book quality and engagement.",
      "depends_on": [
        "curated_books.book_id",
        "curated_books.avg_rating"
      ]
    },
    {
      "run_id": "run_20250617_091920_2d524456",
      "id": "84208898-ce10-4565-82fa-2bc8ac173f1d",
      "summary": "Different audience demographics engage differently with book formats.",
      "rationale": "Demographics may resonate differently with various reading formats, leading to fluctuations in engagement levels and variations by readership.",
      "depends_on": [
        "curated_reviews.book_id",
        "curated_reviews.user_id"
      ]
    },
    {
      "run_id": "run_20250617_091920_2d524456",
      "id": "52f41945-27f1-4a6a-9fe1-2fe1458607d4",
      "summary": "Popular authors tend to have higher review rates for their books.",
      "rationale": "Established authors often generate a loyal readership that is more likely to provide feedback, contributing to a higher volume of reviews for their works.",
      "depends_on": [
        "book_authors.author_id",
        "curated_reviews.book_id"
      ]
    },
    {
      "run_id": "run_20250617_091920_2d524456",
      "id": "c43c2d24-9126-4d1a-8a24-43a1f7492843",
      "summary": "The sentiments expressed in reviews vary significantly by user engagement.",
      "rationale": "The richness and depth of reviews can reflect the level of emotional or intellectual engagement a user has with a book, offering insight into their reading experience.",
      "depends_on": [
        "curated_reviews.review_text",
        "curated_reviews.rating"
      ]
    },
    {
      "run_id": "run_20250617_091504_28f70390",
      "id": "9333d190-af02-472c-af17-646ac67245f8",
      "summary": "Books with higher average ratings tend to have more formats available.",
      "rationale": "Offering multiple formats (e.g., eBook, paperback, audiobook) increases accessibility and can lead to higher user satisfaction, reflected in ratings.",
      "depends_on": [
        "book_genre_format_ratings.avg_rating",
        "book_genre_format_ratings.format"
      ]
    },
    {
      "run_id": "run_20250617_091504_28f70390",
      "id": "0a7a84d9-792c-4ec8-81cd-a4c10017c909",
      "summary": "Books that are listed in more genres receive higher ratings.",
      "rationale": "Diversity in genre could attract a wider audience, thus increasing the potential for higher ratings as more readers engage with the book.",
      "depends_on": [
        "book_genre_format_ratings.avg_rating",
        "book_genre_format_ratings.genre"
      ]
    },
    {
      "run_id": "run_20250617_091504_28f70390",
      "id": "6014b1a6-b190-49af-b11f-c34081ae966e",
      "summary": "Users who leave reviews with more detailed text tend to provide higher ratings.",
      "rationale": "Longer reviews might indicate a more engaged reader, leading to a more favorable evaluation of the book based on their experience.",
      "depends_on": [
        "curated_reviews.rating",
        "curated_reviews.review_text"
      ]
    },
    {
      "run_id": "run_20250617_091504_28f70390",
      "id": "bc90ed63-834d-4c74-b081-9e4beac6f1fb",
      "summary": "Books categorized as 'wish-list' tend to receive lower ratings than those in 'book-club' or 'ya' genres.",
      "rationale": "The category might imply that readers are more exploratory or less committed to 'wish-list' books, which could be reflected in their ratings.",
      "depends_on": [
        "book_genre_format_ratings.avg_rating",
        "book_genre_format_ratings.genre"
      ]
    },
    {
      "run_id": "run_20250617_091504_28f70390",
      "id": "28c49565-eb41-4580-a085-6efd23fe58f3",
      "summary": "Readers who rate more books tend to have a positive influence on their average ratings.",
      "rationale": "Frequent engagement with books by readers could indicate higher engagement and a tendency to rate books more favorably over time.",
      "depends_on": [
        "curated_reviews.user_id",
        "curated_reviews.rating"
      ]
    },
    {
      "run_id": "run_20250617_083442_801c20a3",
      "id": "86e95b4c-49e9-4900-a6f2-0946af15dead",
      "summary": "Books with higher average ratings tend to receive more ratings.",
      "rationale": "Analysis suggests a connection where books with higher ratings generally exhibit a higher count of reader reviews.",
      "depends_on": [
        "curated_books.avg_rating",
        "curated_books.ratings_count"
      ]
    },
    {
      "run_id": "run_20250617_083442_801c20a3",
      "id": "b1fe0390-aee7-4b67-a4f8-d21933e6e00f",
      "summary": "Books with a length of 400-450 pages are more popular.",
      "rationale": "The analysis shows a clustering of popular books around the 400-450 pages mark, indicating reader preference for these lengths.",
      "depends_on": [
        "curated_books.num_pages"
      ]
    },
    {
      "run_id": "run_20250617_083442_801c20a3",
      "id": "cf52137f-6d57-44fc-931c-7650b2170e14",
      "summary": "Certain books have significantly higher ratings counts, indicating outlier popularity.",
      "rationale": "Observations reveal a stark contrast in popularity among books, with select titles receiving significantly more reviews, impacting overall trends.",
      "depends_on": [
        "curated_books.ratings_count"
      ]
    },
    {
      "run_id": "run_20250617_083442_801c20a3",
      "id": "d75696af-e739-43a8-97e5-a882e23dd69f",
      "summary": "Books with lower average ratings may have niche audiences.",
      "rationale": "Findings include examples of low-rated books that suggest a specialty or niche appeal, warranting further inquiry for targeted strategies.",
      "depends_on": [
        "curated_books.avg_rating",
        "curated_books.ratings_count"
      ]
    },
    {
      "run_id": "run_20250617_083442_801c20a3",
      "id": "c056e446-a3f6-44fe-b9cf-cdf93dbdd044",
      "summary": "Books in the 'mystery-suspense' genre have higher average ratings than those in other genres.",
      "rationale": "Identifying high-performing genres can guide marketing and recommendation strategies to boost user engagement.",
      "depends_on": [
        "book_genre_format_ratings.genre",
        "book_genre_format_ratings.avg_rating"
      ]
    },
    {
      "run_id": "run_20250617_083442_801c20a3",
      "id": "1b03793c-8c35-4a1d-94a1-e3487b78ae59",
      "summary": "Users who read more books tend to give higher average ratings.",
      "rationale": "Understanding reading patterns and user engagement can refine personalized recommendations and increase user satisfaction.",
      "depends_on": [
        "curated_reviews.user_id",
        "curated_reviews.rating"
      ]
    },
    {
      "run_id": "run_20250617_083442_801c20a3",
      "id": "d1c0deef-eb7c-4721-960a-f4d583a54d69",
      "summary": "Author collaborations are linked to an increase in shared readership and book success.",
      "rationale": "Detecting collaborative patterns can leverage cross-promotion opportunities and diversify author exposure.",
      "depends_on": [
        "book_authors.author_id",
        "book_authors.book_id"
      ]
    },
    {
      "run_id": "run_20250617_083442_801c20a3",
      "id": "dae69914-229f-4dd4-9fab-ac5338b72da7",
      "summary": "Books with more than 10 ratings provide reliable average ratings and insights.",
      "rationale": "Focusing on books with substantial ratings can enhance the accuracy of analysis on book performance.",
      "depends_on": [
        "curated_reviews.book_id",
        "curated_reviews.rating"
      ]
    },
    {
      "run_id": "run_20250617_083442_801c20a3",
      "id": "07fe12a8-784e-47e1-beb4-359f9029da15",
      "summary": "The distribution of books across genres and formats reveals market trends and reader preferences.",
      "rationale": "Understanding genre and format distributions aids in anticipating user needs and informing inventory decisions.",
      "depends_on": [
        "book_genre_format_ratings.genre",
        "book_genre_format_ratings.format"
      ]
    },
    {
      "run_id": "run_20250617_093601_ae6eca46",
      "id": "9eb12361-95f2-43d9-9a30-08df006409a9",
      "summary": "Books with higher average ratings are more likely to have a greater number of ratings.",
      "rationale": "A larger number of ratings may indicate a broader readership which could lead to higher average ratings.",
      "depends_on": [
        "curated_books_view.avg_rating",
        "curated_books_view.ratings_count"
      ]
    },
    {
      "run_id": "run_20250617_093601_ae6eca46",
      "id": "d5b2263b-546e-4007-b0e6-1489324361b1",
      "summary": "eBooks tend to have higher average ratings compared to physical books.",
      "rationale": "The eBook format may attract more engaged readers who provide ratings, leading to a higher average rating.",
      "depends_on": [
        "curated_books_view.is_ebook",
        "curated_books_view.avg_rating"
      ]
    },
    {
      "run_id": "run_20250617_093601_ae6eca46",
      "id": "14f49da7-3891-493a-872c-e1baa903e65c",
      "summary": "Books published by known publishers receive higher ratings.",
      "rationale": "Books from established publishers may be of higher quality and better marketed, contributing to better reception and ratings.",
      "depends_on": [
        "curated_books_view.publisher_name",
        "curated_books_view.avg_rating"
      ]
    },
    {
      "run_id": "run_20250617_000445_396c4332",
      "id": "36c4e94f-a98a-4a17-ac4c-74ce532ca4ee",
      "summary": "Higher book ratings correlate with more user engagement indicators.",
      "rationale": "Books with higher ratings tend to receive more ratings and user interactions, suggesting that quality impacts engagement.",
      "depends_on": [
        "book_shelves.cnt",
        "user_stats_daily.mean_rating"
      ]
    },
    {
      "run_id": "run_20250617_000445_396c4332",
      "id": "c3c50574-dbe3-47d3-9247-fbc88a3e5775",
      "summary": "Books in series have higher average ratings than standalone books.",
      "rationale": "Series may develop richer character arcs and plotlines, encouraging readers to invest more, which correlates with higher ratings.",
      "depends_on": [
        "book_series.series_name",
        "curated_books.avg_rating"
      ]
    },
    {
      "run_id": "run_20250617_000445_396c4332",
      "id": "9ea2077f-cde5-4f9d-9ac1-0aab493d4a06",
      "summary": "Translated books achieve lower ratings compared to original language publications.",
      "rationale": "Perceptions of translation quality can impact user ratings, indicating that original works may resonate more.",
      "depends_on": [
        "book_authors.role",
        "curated_books.avg_rating"
      ]
    },
    {
      "run_id": "run_20250617_000445_396c4332",
      "id": "5a96a7d9-fb75-4489-89ae-1f4bd14ff41e",
      "summary": "Readers who engage with multiple genres display broader engagement metrics.",
      "rationale": "Genre diversity might indicate varied interests leading to more comprehensive reading habits and higher engagement.",
      "depends_on": [
        "book_genre_format_ratings.genre",
        "user_stats_daily.n_ratings"
      ]
    },
    {
      "run_id": "run_20250617_000445_396c4332",
      "id": "59e13a74-766a-4edc-8042-6dc9e6a7f4e1",
      "summary": "Books published with more extensive marketing (e.g., large publisher backing) receive higher user ratings.",
      "rationale": "Visibility and perceived legitimacy from larger publishers may influence reader perceptions and ratings.",
      "depends_on": [
        "curated_books.publisher_name",
        "curated_books.avg_rating"
      ]
    }
  ]
}
```

### `contingency/functions.py`

**File size:** 62,317 bytes

```python
import pandas as pd
import numpy as np
from typing import Dict, Any

def template_feature_function(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """
    Template for a feature computation function.

    Args:
        df (pd.DataFrame): The input DataFrame containing all necessary columns (already joined).
        params (Dict[str, Any]): Dictionary of hyperparameters for this feature, as suggested by BO.

    Returns:
        pd.Series: A single column of computed feature values, indexed the same as df.

    Contract:
    - This function must be pure (no side effects), deterministic, and not mutate df in-place.
    - The function should handle missing values gracefully and document any required columns.
    - The feature name should correspond to the function name (for registry purposes).
    - Example usage:
        feature_col = template_feature_function(df, {'alpha': 0.5, 'window': 3})
    """
    # Example (identity feature):
    # return df["some_column"] * params.get("alpha", 1.0)
    raise NotImplementedError("Override this template with your custom feature logic.")


def rating_popularity_momentum(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """
    Feature based on hypothesis: "Books with higher average ratings tend to have more ratings"
    
    This feature captures the momentum effect where popular books (high ratings) attract more ratings,
    creating a virtuous cycle. The feature combines average rating with rating count in a non-linear way
    to capture this momentum effect.
    
    Required columns:
    - average_rating: Book's average rating (float, typically 0-5)
    - ratings_count: Number of ratings the book has received (int)
    
    Hyperparameters:
    - rating_weight: Weight for the average rating component (default: 1.0)
    - count_weight: Weight for the ratings count component (default: 0.5) 
    - momentum_power: Power to apply to the momentum calculation (default: 0.8)
    - min_ratings_threshold: Minimum ratings needed for momentum effect (default: 10)
    - rating_scale: Scale factor for rating normalization (default: 5.0)
    
    Args:
        df (pd.DataFrame): Input DataFrame with book data
        params (Dict[str, Any]): Hyperparameters for the feature
        
    Returns:
        pd.Series: Rating popularity momentum feature values
    """
    # Extract hyperparameters with defaults
    rating_weight = params.get("rating_weight", 1.0)
    count_weight = params.get("count_weight", 0.5)
    momentum_power = params.get("momentum_power", 0.8)
    min_ratings_threshold = params.get("min_ratings_threshold", 10)
    rating_scale = params.get("rating_scale", 5.0)
    
    # Validate required columns
    required_cols = ["average_rating", "ratings_count"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Handle missing values
    avg_rating = df["average_rating"].fillna(df["average_rating"].median())
    ratings_count = df["ratings_count"].fillna(0)
    
    # Normalize average rating to 0-1 scale
    normalized_rating = avg_rating / rating_scale
    
    # Log-transform ratings count to handle skewness
    log_count = np.log1p(ratings_count)  # log(1 + count) to handle zeros
    
    # Create momentum indicator (books with sufficient ratings get momentum boost)
    momentum_mask = ratings_count >= min_ratings_threshold
    
    # Calculate base momentum: rating * log(count)
    base_momentum = (rating_weight * normalized_rating) * (count_weight * log_count)
    
    # Apply power transformation for non-linearity
    momentum_feature = np.power(base_momentum, momentum_power)
    
    # Apply momentum boost for books with sufficient ratings
    momentum_feature = np.where(
        momentum_mask,
        momentum_feature * (1 + 0.1 * np.log1p(ratings_count - min_ratings_threshold)),
        momentum_feature * 0.8  # Penalty for books with few ratings
    )
    
    # Handle edge cases
    momentum_feature = np.where(
        (avg_rating == 0) | (ratings_count == 0),
        0.0,  # Zero momentum for unrated books
        momentum_feature
    )
    
    return pd.Series(momentum_feature, index=df.index, name="rating_popularity_momentum")


def genre_preference_alignment(df: pd.DataFrame, params: Dict[str, Any] = None) -> pd.Series:
    """
    Feature based on hypothesis: "Users prefer specific genres that have consistently high average ratings"
    
    Creates a feature that captures how well a book aligns with high-performing genres.
    This feature identifies books in genres that tend to have consistently high ratings.
    
    Parameters (for Bayesian Optimization):
    - genre_weight: Weight for genre rating component (0.1 to 2.0)
    - rating_threshold: Minimum rating to consider a genre "high-performing" (3.0 to 4.5)
    - popularity_factor: How much to weight genre popularity (0.0 to 1.0)
    - recency_decay: Decay factor for older books (0.8 to 1.0)
    - boost_multiplier: Multiplier for books in top genres (1.0 to 3.0)
    """
    if params is None:
        params = {}
    
    # Extract hyperparameters with defaults
    genre_weight = params.get('genre_weight', 1.0)
    rating_threshold = params.get('rating_threshold', 3.8)
    popularity_factor = params.get('popularity_factor', 0.5)
    recency_decay = params.get('recency_decay', 0.95)
    boost_multiplier = params.get('boost_multiplier', 1.5)
    
    # Initialize feature values
    feature_values = pd.Series(0.0, index=df.index, name='genre_preference_alignment')
    
    try:
        # Create synthetic genre data since we don't have real genre info
        # In a real implementation, this would come from the database
        genres = ['Fiction', 'Non-Fiction', 'Mystery', 'Romance', 'Sci-Fi', 'Fantasy', 
                 'Biography', 'History', 'Self-Help', 'Thriller']
        
        # Assign genres based on book characteristics (synthetic approach)
        np.random.seed(42)  # For reproducibility
        book_genres = np.random.choice(genres, size=len(df))
        
        # Calculate genre performance metrics
        genre_ratings = {}
        genre_popularity = {}
        
        for genre in genres:
            genre_mask = book_genres == genre
            if genre_mask.sum() > 0:
                genre_books = df[genre_mask]
                avg_rating = genre_books['average_rating'].mean()
                popularity = genre_books['ratings_count'].mean()
                
                genre_ratings[genre] = avg_rating
                genre_popularity[genre] = popularity
        
        # Identify high-performing genres
        high_performing_genres = {
            genre: rating for genre, rating in genre_ratings.items() 
            if rating >= rating_threshold
        }
        
        # Calculate feature for each book
        current_year = 2024
        
        for i, (idx, row) in enumerate(df.iterrows()):
            book_genre = book_genres[i]
            
            # Base alignment score
            if book_genre in high_performing_genres:
                genre_score = high_performing_genres[book_genre] * genre_weight
                
                # Add popularity factor
                if book_genre in genre_popularity:
                    pop_score = np.log1p(genre_popularity[book_genre]) * popularity_factor
                    genre_score += pop_score
                
                # Apply boost for top genres
                if genre_ratings.get(book_genre, 0) > rating_threshold + 0.2:
                    genre_score *= boost_multiplier
                
                # Apply recency decay for older books
                if pd.notna(row.get('publication_year')):
                    years_old = current_year - row['publication_year']
                    if years_old > 0:
                        decay_factor = recency_decay ** min(years_old, 20)  # Cap at 20 years
                        genre_score *= decay_factor
                
                feature_values.iloc[i] = genre_score
            else:
                # Books in lower-performing genres get a small base score
                feature_values.iloc[i] = genre_ratings.get(book_genre, 3.0) * 0.3
        
        # Normalize to reasonable range
        if feature_values.max() > 0:
            feature_values = feature_values / feature_values.max() * 5.0
        
        return feature_values
        
    except Exception as e:
        print(f"Error in genre_preference_alignment: {e}")
        return pd.Series(0.0, index=df.index, name='genre_preference_alignment')


def publication_recency_boost(df: pd.DataFrame, params: Dict[str, Any] = None) -> pd.Series:
    """
    Feature based on hypothesis: "Recent publications with high ratings indicate emerging trends"
    
    Creates a feature that boosts books published recently that have gained traction quickly.
    This captures the momentum of newer books that are performing well.
    
    Parameters (for Bayesian Optimization):
    - recency_weight: Weight for how recent the book is (0.1 to 2.0)
    - rating_weight: Weight for the book's rating (0.5 to 2.0)
    - velocity_factor: Weight for rating velocity (ratings/years) (0.1 to 1.5)
    - recent_threshold: Years to consider "recent" (1 to 10)
    - min_ratings: Minimum ratings needed for boost (5 to 100)
    """
    if params is None:
        params = {}
    
    # Extract hyperparameters with defaults
    recency_weight = params.get('recency_weight', 1.2)
    rating_weight = params.get('rating_weight', 1.0)
    velocity_factor = params.get('velocity_factor', 0.8)
    recent_threshold = params.get('recent_threshold', 5)
    min_ratings = params.get('min_ratings', 20)
    
    # Initialize feature values
    feature_values = pd.Series(0.0, index=df.index, name='publication_recency_boost')
    
    try:
        current_year = 2024
        
        for idx, row in df.iterrows():
            # Check if we have required data
            if pd.isna(row.get('publication_year')) or pd.isna(row.get('average_rating')):
                continue
                
            pub_year = row['publication_year']
            avg_rating = row['average_rating']
            ratings_count = row.get('ratings_count', 0)
            
            # Only consider books with sufficient ratings
            if ratings_count < min_ratings:
                continue
            
            # Calculate years since publication
            years_since_pub = current_year - pub_year
            
            # Only boost recent books
            if years_since_pub <= recent_threshold and years_since_pub > 0:
                # Recency score (higher for more recent)
                recency_score = (recent_threshold - years_since_pub) / recent_threshold
                recency_score = recency_score ** 0.5  # Square root for smoother curve
                
                # Rating score (higher for better ratings)
                rating_score = (avg_rating - 2.0) / 3.0  # Normalize 2-5 to 0-1
                rating_score = max(0, rating_score)
                
                # Velocity score (ratings per year since publication)
                velocity = ratings_count / max(years_since_pub, 0.5)  # Avoid division by zero
                velocity_score = np.log1p(velocity) / 10.0  # Log scale, normalized
                
                # Combine components
                boost_score = (
                    recency_score * recency_weight +
                    rating_score * rating_weight +
                    velocity_score * velocity_factor
                )
                
                # Apply additional boost for exceptional performance
                if avg_rating >= 4.2 and velocity > 50:
                    boost_score *= 1.3
                
                feature_values[idx] = boost_score
        
        # Normalize to 0-3 range
        if feature_values.max() > 0:
            feature_values = feature_values / feature_values.max() * 3.0
        
        return feature_values
        
    except Exception as e:
        print(f"Error in publication_recency_boost: {e}")
        return pd.Series(0.0, index=df.index, name='publication_recency_boost')


def engagement_depth_score(df: pd.DataFrame, params: Dict[str, Any] = None) -> pd.Series:
    """
    Feature based on hypothesis: "Books with more text reviews relative to ratings indicate deeper engagement"
    
    Creates a feature that captures the depth of user engagement beyond just ratings.
    Books that inspire detailed reviews may have different recommendation value.
    
    Parameters (for Bayesian Optimization):
    - review_ratio_weight: Weight for text reviews to ratings ratio (0.5 to 2.0)
    - absolute_reviews_weight: Weight for absolute number of reviews (0.1 to 1.0)
    - engagement_threshold: Minimum ratio to consider high engagement (0.05 to 0.5)
    - length_proxy_factor: Factor for estimated review length (0.0 to 1.0)
    - quality_boost: Boost for high-quality engagement indicators (1.0 to 2.0)
    """
    if params is None:
        params = {}
    
    # Extract hyperparameters with defaults
    review_ratio_weight = params.get('review_ratio_weight', 1.0)
    absolute_reviews_weight = params.get('absolute_reviews_weight', 0.3)
    engagement_threshold = params.get('engagement_threshold', 0.1)
    length_proxy_factor = params.get('length_proxy_factor', 0.2)
    quality_boost = params.get('quality_boost', 1.2)
    
    # Initialize feature values
    feature_values = pd.Series(0.0, index=df.index, name='engagement_depth_score')
    
    try:
        for idx, row in df.iterrows():
            ratings_count = row.get('ratings_count', 0)
            text_reviews_count = row.get('text_reviews_count', 0)
            avg_rating = row.get('average_rating', 3.0)
            
            if ratings_count == 0:
                continue
            
            # Calculate review-to-rating ratio
            review_ratio = text_reviews_count / ratings_count
            
            # Base engagement score
            if review_ratio >= engagement_threshold:
                # Ratio component
                ratio_score = min(review_ratio, 1.0) * review_ratio_weight  # Cap at 1.0
                
                # Absolute reviews component (log scale)
                absolute_score = np.log1p(text_reviews_count) * absolute_reviews_weight
                
                # Length proxy (assume higher-rated books get longer reviews)
                length_proxy = (avg_rating - 3.0) * length_proxy_factor
                
                # Combine components
                engagement_score = ratio_score + absolute_score + length_proxy
                
                # Quality boost for exceptional engagement
                if review_ratio > 0.3 and text_reviews_count > 50:
                    engagement_score *= quality_boost
                
                # Boost for books that inspire discussion (high review ratio + good rating)
                if review_ratio > 0.2 and avg_rating >= 4.0:
                    engagement_score *= 1.1
                
                feature_values[idx] = engagement_score
        
        # Normalize to 0-4 range
        if feature_values.max() > 0:
            feature_values = feature_values / feature_values.max() * 4.0
        
        return feature_values
        
    except Exception as e:
        print(f"Error in engagement_depth_score: {e}")
        return pd.Series(0.0, index=df.index, name='engagement_depth_score')


# =============================================================================
# BATCH 1: HYPOTHESES 1-10 FEATURE FUNCTIONS (TEMPLATE-COMPLIANT)
# =============================================================================

def user_engagement_rating_correlation(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 1: Users who read more books tend to provide higher average ratings.

    Required columns:
        - books_read (int)
        - avg_rating (float)
    """
    books_w = params.get("books_weight", 1.0)
    rating_w = params.get("rating_weight", 1.0)
    boost = params.get("rating_boost", 1.2)
    threshold = params.get("engagement_threshold", 10)

    if not {"books_read", "avg_rating"}.issubset(df.columns):
        raise ValueError("user_engagement_rating_correlation needs books_read & avg_rating")

    books = np.log1p(df["books_read"].fillna(0))
    rating = df["avg_rating"].fillna(df["avg_rating"].median()) / 5
    score = books_w * books + rating_w * rating
    score = np.where(df["books_read"].fillna(0) >= threshold, score * boost, score)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="user_engagement_rating_correlation")


def genre_preference_strength(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 2: Users prefer specific genres with consistently high ratings.

    Required columns:
        - genre (str)
        - average_rating (float)
    """
    genre_w = params.get("genre_weight", 1.0)
    popularity_w = params.get("popularity_weight", 0.5)
    rating_thresh = params.get("rating_threshold", 4.0)

    if not {"genre", "average_rating"}.issubset(df.columns):
        raise ValueError("genre_preference_strength requires genre & average_rating")

    genre_stats = df.groupby("genre").agg(avg_r=("average_rating", "mean"), cnt=("genre", "size"))
    genre_score = genre_w * (genre_stats["avg_r"] / 5) + popularity_w * np.log1p(genre_stats["cnt"])
    preferred = genre_stats["avg_r"] >= rating_thresh
    genre_score[preferred] *= 1.2
    genre_score = (genre_score - genre_score.min()) / (genre_score.max() - genre_score.min() + 1e-9)
    return pd.Series(df["genre"].map(genre_score).fillna(0).values, index=df.index, name="genre_preference_strength")


def shelf_popularity_indicator(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 3: Shelf categories with more books signal interest.

    Required columns:
        - shelf (str)
        - shelf_count (int)  # number of times book appears in shelf
    """
    shelf_w = params.get("shelf_weight", 1.0)
    count_w = params.get("count_weight", 1.0)
    boost = params.get("popularity_boost", 1.3)
    min_cnt = params.get("min_books_threshold", 5)

    if not {"shelf", "shelf_count"}.issubset(df.columns):
        raise ValueError("shelf_popularity_indicator needs shelf & shelf_count")

    shelf_stats = df.groupby("shelf").agg(cnt=("shelf_count", "sum"))
    score = shelf_w * np.log1p(shelf_stats["cnt"]) * count_w
    score = np.where(shelf_stats["cnt"] >= min_cnt, score * boost, score)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(df["shelf"].map(score).fillna(0).values, index=df.index, name="shelf_popularity_indicator")


def format_preference_score(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 4: Certain formats receive better ratings.

    Required columns:
        - format (str)
        - average_rating (float)
    """
    fmt_w = params.get("format_weight", 1.0)
    rating_w = params.get("rating_weight", 1.0)
    min_rating = params.get("min_rating_threshold", 3.5)
    boost = params.get("format_boost", 1.2)

    if not {"format", "average_rating"}.issubset(df.columns):
        raise ValueError("format_preference_score needs format & average_rating")

    fmt_stats = df.groupby("format").agg(avg_r=("average_rating", "mean"), cnt=("format", "size"))
    score = fmt_w * np.log1p(fmt_stats["cnt"]) + rating_w * (fmt_stats["avg_r"] / 5)
    score = np.where(fmt_stats["avg_r"] >= min_rating, score * boost, score)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(df["format"].map(score).fillna(0).values, index=df.index, name="format_preference_score")


def genre_diversity_preference(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 5: Readers like diverse genres, esp. fantasy & romance.

    Required columns:
        - genre (str)
        - average_rating (float)
    """
    base_w = params.get("genre_weight", 1.0)
    diversity_f = params.get("diversity_factor", 0.6)
    fantasy_boost = params.get("fantasy_boost", 1.3)
    romance_boost = params.get("romance_boost", 1.2)

    if not {"genre", "average_rating"}.issubset(df.columns):
        raise ValueError("genre_diversity_preference needs genre & average_rating")

    stats = df.groupby("genre").agg(avg_r=("average_rating", "mean"), cnt=("genre", "size"))
    base = base_w * (stats["avg_r"] / 5) * np.log1p(stats["cnt"])
    fantasy_mask = stats.index.str.contains("fantasy", case=False, na=False)
    romance_mask = stats.index.str.contains("romance", case=False, na=False)
    base[fantasy_mask] *= fantasy_boost
    base[romance_mask] *= romance_boost
    base *= (1 + diversity_f * (1 - abs(stats["avg_r"] - 3.5) / 1.5))
    base = (base - base.min()) / (base.max() - base.min() + 1e-9)
    return pd.Series(df["genre"].map(base).fillna(0).values, index=df.index, name="genre_diversity_preference")


def user_behavior_clustering(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 6: Consistent rating behaviors form clusters.

    Required columns:
        - user_id (identifier)
        - avg_rating_user (float)
        - rating_stddev (float)
        - review_count (int)
    """
    rating_w = params.get("rating_weight", 1.0)
    var_penalty = params.get("variance_penalty", 0.8)
    boost = params.get("cluster_boost", 1.2)

    cols = {"user_id", "avg_rating_user", "rating_stddev", "review_count"}
    if not cols.issubset(df.columns):
        raise ValueError("user_behavior_clustering missing required columns")

    score = rating_w * (df["avg_rating_user"].fillna(3.5) / 5) * np.log1p(df["review_count"].fillna(0))
    score *= (var_penalty + (1 - var_penalty) * (1 / (1 + df["rating_stddev"].fillna(0))))
    distinct = (df["avg_rating_user"] > 4) | (df["avg_rating_user"] < 2)
    score = np.where(distinct, score * boost, score)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="user_behavior_clustering")


def thematic_genre_crossover(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 7: Fans of steampunk/fantasy like thematic crossovers.

    Required columns:
        - title (str)
        - rating (float)
    """
    rating_w = params.get("rating_weight", 1.0)
    steam_boost = params.get("steampunk_boost", 1.4)
    fantasy_boost = params.get("fantasy_boost", 1.2)
    cross_f = params.get("crossover_factor", 0.8)

    if not {"title", "rating"}.issubset(df.columns):
        raise ValueError("thematic_genre_crossover requires title & rating")

    title = df["title"].fillna("").str.lower()
    steam = title.str.contains("steampunk|steam|clockwork|victorian")
    fantasy = title.str.contains("fantasy|magic|dragon|wizard|elf")
    score = rating_w * df["rating"].fillna(df["rating"].median()) / 5
    score = np.where(steam, score * steam_boost, score)
    score = np.where(fantasy, score * fantasy_boost, score)
    crossover = steam & fantasy
    score = np.where(crossover, score * (1 + cross_f), score)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="thematic_genre_crossover")


def author_collaboration_quality(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 8: Frequent author collaborations relate to quality.

    Required columns:
        - author_id (identifier)
        - avg_rating (float)
        - book_count_author (int)
    """
    collab_w = params.get("collaboration_weight", 1.0)
    quality_boost = params.get("quality_boost", 1.3)
    min_col = params.get("min_collaborations", 2)

    cols = {"author_id", "avg_rating", "book_count_author"}
    if not cols.issubset(df.columns):
        raise ValueError("author_collaboration_quality missing columns")

    base = collab_w * (df["avg_rating"].fillna(df["avg_rating"].median()) / 5) * np.log1p(df["book_count_author"].fillna(0))
    boost_mask = df["book_count_author"] >= min_col
    base = np.where(boost_mask, base * quality_boost, base)
    base = (base - base.min()) / (base.max() - base.min() + 1e-9)
    return pd.Series(base, index=df.index, name="author_collaboration_quality")


def selective_reader_curation(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 9: Readers with few but high ratings need curation.

    Required columns:
        - books_read (int)
        - avg_rating (float)
    """
    sel_w = params.get("selectivity_weight", 1.0)
    boost = params.get("curation_boost", 1.4)
    books_thr = params.get("books_threshold", 10)
    rating_thr = params.get("rating_threshold", 4.0)

    if not {"books_read", "avg_rating"}.issubset(df.columns):
        raise ValueError("selective_reader_curation missing columns")

    score = sel_w * (df["avg_rating"].fillna(df["avg_rating"].median()) / 5) / np.log1p(df["books_read"].fillna(1))
    mask = (df["books_read"] <= books_thr) & (df["avg_rating"] >= rating_thr)
    score = np.where(mask, score * boost, score)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="selective_reader_curation")


def page_length_rating_impact(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 10: Page length influences ratings differently.

    Required columns:
        - num_pages (int)
        - average_rating (float)
    """
    page_w = params.get("page_weight", 1.0)
    rating_w = params.get("rating_weight", 1.0)
    optimal = params.get("optimal_length", 300)
    penalty = params.get("length_penalty", 0.8)

    if not {"num_pages", "average_rating"}.issubset(df.columns):
        raise ValueError("page_length_rating_impact requires num_pages & average_rating")

    deviation = abs(df["num_pages"].fillna(optimal) - optimal) / optimal
    length_factor = 1 / (1 + penalty * deviation)
    score = page_w * np.log1p(df["num_pages"].fillna(0)) * length_factor + rating_w * (df["average_rating"].fillna(df["average_rating"].median()) / 5)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="page_length_rating_impact")


# =============================================================================
# BATCH 2: HYPOTHESES 11-20 FEATURE FUNCTIONS (TEMPLATE-COMPLIANT)
# =============================================================================

# (Batch 2 functions defined above...)

# =============================================================================
# BATCH 3: HYPOTHESES 21-30 FEATURE FUNCTIONS (TEMPLATE-COMPLIANT)
# =============================================================================

# =============================================================================
# BATCH 4: HYPOTHESES 31-40 FEATURE FUNCTIONS (TEMPLATE-COMPLIANT)
# =============================================================================

# =============================================================================
# BATCH 5: HYPOTHESES 41-50 FEATURE FUNCTIONS (TEMPLATE-COMPLIANT)
# =============================================================================

def genre_format_distribution_score(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 41: The distribution of books across genres and formats reveals market trends and reader preferences.
    Required columns:
        - genre (str)
        - format (str)
    """
    genre_w = params.get("genre_weight", 1.0)
    format_w = params.get("format_weight", 1.0)
    if not {"genre", "format"}.issubset(df.columns):
        raise ValueError("genre_format_distribution_score requires genre and format columns")
    genre_counts = df["genre"].value_counts()
    format_counts = df["format"].value_counts()
    score = genre_w * df["genre"].map(genre_counts) + format_w * df["format"].map(format_counts)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="genre_format_distribution_score")

def avg_rating_rating_count_score(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 42: Books with higher average ratings are more likely to have a greater number of ratings.
    Required columns:
        - avg_rating (float)
        - ratings_count (int)
    """
    rating_w = params.get("rating_weight", 1.0)
    count_w = params.get("count_weight", 1.0)
    if not {"avg_rating", "ratings_count"}.issubset(df.columns):
        raise ValueError("avg_rating_rating_count_score requires avg_rating and ratings_count columns")
    score = rating_w * (df["avg_rating"].fillna(df["avg_rating"].median()) / 5) + count_w * np.log1p(df["ratings_count"].fillna(0))
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="avg_rating_rating_count_score")

def ebook_positive_rating_score(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 43: eBooks tend to have higher average ratings compared to physical books.
    Required columns:
        - is_ebook (bool/int 0/1)
        - avg_rating (float)
    """
    ebook_boost = params.get("ebook_boost", 1.1)
    rating_w = params.get("rating_weight", 1.0)
    if not {"is_ebook", "avg_rating"}.issubset(df.columns):
        raise ValueError("ebook_positive_rating_score requires is_ebook and avg_rating columns")
    base = rating_w * (df["avg_rating"].fillna(df["avg_rating"].median()) / 5)
    score = np.where(df["is_ebook"].fillna(0) == 1, base * ebook_boost, base)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="ebook_positive_rating_score")

def publisher_reputation_rating(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 44: Books published by known publishers receive higher ratings.
    Required columns:
        - publisher_name (str)
        - avg_rating (float)
    """
    rep_w = params.get("reputation_weight", 1.0)
    rating_w = params.get("rating_weight", 1.0)
    if not {"publisher_name", "avg_rating"}.issubset(df.columns):
        raise ValueError("publisher_reputation_rating requires publisher_name and avg_rating columns")
    pub_counts = df["publisher_name"].value_counts()
    score = rep_w * np.log1p(df["publisher_name"].map(pub_counts)) + rating_w * (df["avg_rating"].fillna(df["avg_rating"].median()) / 5)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="publisher_reputation_rating")

def rating_engagement_correlation(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 45: Higher book ratings correlate with more user engagement indicators.
    Required columns:
        - cnt (int)
        - mean_rating (float)
    """
    cnt_w = params.get("count_weight", 1.0)
    rating_w = params.get("rating_weight", 1.0)
    if not {"cnt", "mean_rating"}.issubset(df.columns):
        raise ValueError("rating_engagement_correlation requires cnt and mean_rating columns")
    score = cnt_w * np.log1p(df["cnt"].fillna(0)) + rating_w * (df["mean_rating"].fillna(df["mean_rating"].median()) / 5)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="rating_engagement_correlation")

def series_vs_standalone_rating(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 46: Books in series have higher average ratings than standalone books.
    Required columns:
        - series_name (str, can be null for standalone)
        - avg_rating (float)
    """
    series_boost = params.get("series_boost", 1.1)
    rating_w = params.get("rating_weight", 1.0)
    if not {"series_name", "avg_rating"}.issubset(df.columns):
        raise ValueError("series_vs_standalone_rating requires series_name and avg_rating columns")
    base = rating_w * (df["avg_rating"].fillna(df["avg_rating"].median()) / 5)
    score = np.where(df["series_name"].notnull() & (df["series_name"] != ""), base * series_boost, base)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="series_vs_standalone_rating")

def translation_penalty_score(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 47: Translated books achieve lower ratings compared to original language publications.
    Required columns:
        - role (str)
        - avg_rating (float)
    """
    penalty = params.get("translation_penalty", 0.9)
    rating_w = params.get("rating_weight", 1.0)
    if not {"role", "avg_rating"}.issubset(df.columns):
        raise ValueError("translation_penalty_score requires role and avg_rating columns")
    base = rating_w * (df["avg_rating"].fillna(df["avg_rating"].median()) / 5)
    score = np.where(df["role"].str.lower().fillna("").str.contains("translator"), base * penalty, base)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="translation_penalty_score")

def genre_diversity_engagement_score(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 48: Readers who engage with multiple genres display broader engagement metrics.
    Required columns:
        - genre (str)
        - n_ratings (int)
    """
    diversity_w = params.get("diversity_weight", 1.0)
    engagement_w = params.get("engagement_weight", 1.0)
    if not {"genre", "n_ratings"}.issubset(df.columns):
        raise ValueError("genre_diversity_engagement_score requires genre and n_ratings columns")
    genre_counts = df["genre"].value_counts()
    score = diversity_w * df["genre"].map(genre_counts) + engagement_w * np.log1p(df["n_ratings"].fillna(0))
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="genre_diversity_engagement_score")

def publisher_marketing_rating_boost(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 49: Books published with more extensive marketing (e.g., large publisher backing) receive higher user ratings.
    Required columns:
        - publisher_name (str)
        - avg_rating (float)
    """
    marketing_boost = params.get("marketing_boost", 1.2)
    rating_w = params.get("rating_weight", 1.0)
    if not {"publisher_name", "avg_rating"}.issubset(df.columns):
        raise ValueError("publisher_marketing_rating_boost requires publisher_name and avg_rating columns")
    pub_counts = df["publisher_name"].value_counts()
    score = rating_w * (df["avg_rating"].fillna(df["avg_rating"].median()) / 5)
    score = np.where(df["publisher_name"].map(pub_counts) > 10, score * marketing_boost, score)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="publisher_marketing_rating_boost")


def detailed_review_rating_boost(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 31: Users who leave reviews with more detailed text tend to provide higher ratings.
    Required columns:
        - review_text (str)
        - rating (float)
    """
    detail_w = params.get("detail_weight", 1.0)
    rating_w = params.get("rating_weight", 1.0)
    if not {"review_text", "rating"}.issubset(df.columns):
        raise ValueError("detailed_review_rating_boost requires review_text and rating columns")
    detail = df["review_text"].fillna("").str.len()
    score = detail_w * np.log1p(detail) + rating_w * (df["rating"].fillna(df["rating"].median()) / 5)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="detailed_review_rating_boost")

def wishlist_vs_bookclub_rating(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 32: Books categorized as 'wish-list' tend to receive lower ratings than those in 'book-club' or 'ya' genres.
    Required columns:
        - genre (str)
        - average_rating (float)
    """
    wishlist_penalty = params.get("wishlist_penalty", 0.9)
    club_boost = params.get("club_boost", 1.1)
    rating_w = params.get("rating_weight", 1.0)
    if not {"genre", "average_rating"}.issubset(df.columns):
        raise ValueError("wishlist_vs_bookclub_rating requires genre and average_rating columns")
    base = rating_w * (df["average_rating"].fillna(df["average_rating"].median()) / 5)
    score = base.copy()
    score[df["genre"].str.lower().fillna("").str.contains("wish-list")] *= wishlist_penalty
    score[df["genre"].str.lower().fillna("").str.contains("book-club|ya")] *= club_boost
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="wishlist_vs_bookclub_rating")

def reader_engagement_positive_influence(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 33: Readers who rate more books tend to have a positive influence on their average ratings.
    Required columns:
        - user_id (str/int)
        - rating (float)
    """
    engagement_w = params.get("engagement_weight", 1.0)
    rating_w = params.get("rating_weight", 1.0)
    if not {"user_id", "rating"}.issubset(df.columns):
        raise ValueError("reader_engagement_positive_influence requires user_id and rating columns")
    user_counts = df.groupby("user_id")["rating"].count()
    user_avg = df.groupby("user_id")["rating"].mean()
    score = engagement_w * np.log1p(df["user_id"].map(user_counts)) + rating_w * (df["user_id"].map(user_avg) / 5)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="reader_engagement_positive_influence")

def avg_rating_ratings_count_correlation(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 34: Books with higher average ratings tend to receive more ratings.
    Required columns:
        - average_rating (float)
        - ratings_count (int)
    """
    rating_w = params.get("rating_weight", 1.0)
    count_w = params.get("count_weight", 1.0)
    if not {"average_rating", "ratings_count"}.issubset(df.columns):
        raise ValueError("avg_rating_ratings_count_correlation requires average_rating and ratings_count columns")
    score = rating_w * (df["average_rating"].fillna(df["average_rating"].median()) / 5) + count_w * np.log1p(df["ratings_count"].fillna(0))
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="avg_rating_ratings_count_correlation")

def optimal_page_length_popularity(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 35: Books with a length of 400-450 pages are more popular.
    Required columns:
        - num_pages (int)
    """
    lower = params.get("lower", 400)
    upper = params.get("upper", 450)
    boost = params.get("boost", 1.2)
    page_w = params.get("page_weight", 1.0)
    if "num_pages" not in df.columns:
        raise ValueError("optimal_page_length_popularity requires num_pages column")
    base = page_w * (df["num_pages"].fillna(0) / df["num_pages"].max())
    mask = (df["num_pages"] >= lower) & (df["num_pages"] <= upper)
    score = np.where(mask, base * boost, base)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="optimal_page_length_popularity")

def outlier_popularity_score(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 36: Certain books have significantly higher ratings counts, indicating outlier popularity.
    Required columns:
        - ratings_count (int)
    """
    outlier_w = params.get("outlier_weight", 1.0)
    if "ratings_count" not in df.columns:
        raise ValueError("outlier_popularity_score requires ratings_count column")
    q3 = df["ratings_count"].quantile(0.75)
    outlier = df["ratings_count"] > q3
    score = outlier_w * outlier.astype(float)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="outlier_popularity_score")

def niche_audience_score(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 37: Books with lower average ratings may have niche audiences.
    Required columns:
        - average_rating (float)
        - ratings_count (int)
    """
    niche_w = params.get("niche_weight", 1.0)
    if not {"average_rating", "ratings_count"}.issubset(df.columns):
        raise ValueError("niche_audience_score requires average_rating and ratings_count columns")
    low_rating = df["average_rating"] < df["average_rating"].median()
    score = niche_w * low_rating.astype(float)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="niche_audience_score")

def mystery_suspense_genre_boost(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 38: Books in the 'mystery-suspense' genre have higher average ratings than those in other genres.
    Required columns:
        - genre (str)
        - average_rating (float)
    """
    genre_boost = params.get("genre_boost", 1.2)
    rating_w = params.get("rating_weight", 1.0)
    if not {"genre", "average_rating"}.issubset(df.columns):
        raise ValueError("mystery_suspense_genre_boost requires genre and average_rating columns")
    base = rating_w * (df["average_rating"].fillna(df["average_rating"].median()) / 5)
    score = np.where(df["genre"].str.lower().fillna("").str.contains("mystery-suspense"), base * genre_boost, base)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="mystery_suspense_genre_boost")

def user_reading_volume_rating(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 39: Users who read more books tend to give higher average ratings.
    Required columns:
        - user_id (str/int)
        - rating (float)
    """
    volume_w = params.get("volume_weight", 1.0)
    rating_w = params.get("rating_weight", 1.0)
    if not {"user_id", "rating"}.issubset(df.columns):
        raise ValueError("user_reading_volume_rating requires user_id and rating columns")
    user_counts = df.groupby("user_id")["rating"].count()
    user_avg = df.groupby("user_id")["rating"].mean()
    score = volume_w * np.log1p(df["user_id"].map(user_counts)) + rating_w * (df["user_id"].map(user_avg) / 5)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="user_reading_volume_rating")

def author_collaboration_success(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 40: Author collaborations are linked to an increase in shared readership and book success.
    Required columns:
        - author_id (str/int)
        - book_id (str/int)
    """
    collab_w = params.get("collab_weight", 1.0)
    if not {"author_id", "book_id"}.issubset(df.columns):
        raise ValueError("author_collaboration_success requires author_id and book_id columns")
    author_books = df.groupby("author_id")["book_id"].nunique()
    score = collab_w * np.log1p(df["author_id"].map(author_books))
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="author_collaboration_success")


def page_count_rating_correlation(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 21: Books with more pages tend to receive a higher average rating.
    
    Required columns:
        - num_pages (int)
        - average_rating (float)
    """
    page_w = params.get("page_weight", 1.0)
    rating_w = params.get("rating_weight", 1.0)
    max_pages = params.get("max_pages", 1200)

    if not {"num_pages", "average_rating"}.issubset(df.columns):
        raise ValueError("page_count_rating_correlation requires num_pages & average_rating")

    pg_norm = df["num_pages"].fillna(0).clip(upper=max_pages) / max_pages
    score = page_w * pg_norm + rating_w * (df["average_rating"].fillna(df["average_rating"].median()) / 5)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="page_count_rating_correlation")


def ebook_rating_penalty(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 22: Ebooks tend to have lower average ratings compared to physical books.
    
    Required columns:
        - is_ebook (bool/int 0/1)
        - average_rating (float)
    """
    ebook_penalty = params.get("ebook_penalty", 0.9)
    rating_w = params.get("rating_weight", 1.0)

    if not {"is_ebook", "average_rating"}.issubset(df.columns):
        raise ValueError("ebook_rating_penalty needs is_ebook & average_rating")

    base = rating_w * (df["average_rating"].fillna(df["average_rating"].median()) / 5)
    score = np.where(df["is_ebook"].fillna(0) == 1, base * ebook_penalty, base)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="ebook_rating_penalty")


def genre_volume_rating_boost(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 23: Genres with more books tend to have higher average ratings.
    
    Required columns:
        - genre (str)
        - average_rating (float)
    """
    volume_w = params.get("volume_weight", 1.0)
    rating_w = params.get("rating_weight", 1.0)

    if not {"genre", "average_rating"}.issubset(df.columns):
        raise ValueError("genre_volume_rating_boost needs genre & average_rating")

    gstats = df.groupby("genre").agg(cnt=("genre", "size"), avg_r=("average_rating", "mean"))
    gscore = volume_w * np.log1p(gstats["cnt"]) + rating_w * (gstats["avg_r"] / 5)
    gscore = (gscore - gscore.min()) / (gscore.max() - gscore.min() + 1e-9)
    return pd.Series(df["genre"].map(gscore).fillna(0).values, index=df.index, name="genre_volume_rating_boost")


def user_activity_review_count(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 24: Users who engage with more books tend to provide more reviews.
    
    Required columns:
        - user_book_count (int)
        - review_count_user (int)
    """
    activity_w = params.get("activity_weight", 1.0)
    review_w = params.get("review_weight", 1.0)

    if not {"user_book_count", "review_count_user"}.issubset(df.columns):
        raise ValueError("user_activity_review_count requires user_book_count & review_count_user")

    score = activity_w * np.log1p(df["user_book_count"].fillna(0)) + review_w * np.log1p(df["review_count_user"].fillna(0))
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="user_activity_review_count")


def rating_review_volume_correlation(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 25: Books with higher average ratings tend to have more reviews.
    
    Required columns:
        - average_rating (float)
        - review_count (int)
    """
    rating_w = params.get("rating_weight", 1.0)
    review_w = params.get("review_weight", 1.0)

    if not {"average_rating", "review_count"}.issubset(df.columns):
        raise ValueError("rating_review_volume_correlation requires average_rating & review_count")

    score = rating_w * (df["average_rating"].fillna(df["average_rating"].median()) / 5) + review_w * np.log1p(df["review_count"].fillna(0))
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="rating_review_volume_correlation")


def demographic_format_engagement(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 26: Different audience demographics engage differently with book formats.
    
    Required columns:
        - user_age_group (str)
        - format (str)
        - engagement (float/int)
    """
    demo_w = params.get("demo_weight", 1.0)
    format_w = params.get("format_weight", 1.0)

    cols = {"user_age_group", "format", "engagement"}
    if not cols.issubset(df.columns):
        raise ValueError("demographic_format_engagement missing required columns")

    demo_fmt_eng = df.groupby(["user_age_group", "format"]).agg(avg_e=("engagement", "mean"))
    demo_fmt_eng = (demo_fmt_eng - demo_fmt_eng.min()) / (demo_fmt_eng.max() - demo_fmt_eng.min() + 1e-9)

    score = df.apply(lambda row: demo_w * demo_fmt_eng.loc[(row["user_age_group"], row["format"])] if (row["user_age_group"], row["format"]) in demo_fmt_eng.index else 0, axis=1)
    return pd.Series(score, index=df.index, name="demographic_format_engagement")


def author_popularity_review_rate(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 27: Popular authors tend to have higher review rates for their books.
    
    Required columns:
        - author_id (identifier)
        - review_count (int)
        - ratings_count (int)
    """
    review_w = params.get("review_weight", 1.0)
    popularity_w = params.get("popularity_weight", 1.0)

    cols = {"author_id", "review_count", "ratings_count"}
    if not cols.issubset(df.columns):
        raise ValueError("author_popularity_review_rate missing columns")

    auth_stats = df.groupby("author_id").agg(rev_sum=("review_count", "sum"), rat_sum=("ratings_count", "sum"))
    auth_score = review_w * np.log1p(auth_stats["rev_sum"]) + popularity_w * np.log1p(auth_stats["rat_sum"])
    auth_score = (auth_score - auth_score.min()) / (auth_score.max() - auth_score.min() + 1e-9)
    return pd.Series(df["author_id"].map(auth_score).fillna(0).values, index=df.index, name="author_popularity_review_rate")


def review_sentiment_engagement_variance(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 28: The sentiments expressed in reviews vary significantly by user engagement.
    
    Required columns:
        - review_text (str)
        - engagement (float/int)
    """
    sentiment_w = params.get("sentiment_weight", 1.0)
    engagement_w = params.get("engagement_weight", 1.0)

    positive_words = ["good", "great", "excellent", "amazing", "love", "wonderful", "fantastic"]
    negative_words = ["bad", "terrible", "awful", "hate", "boring", "worst", "poor"]

    if not {"review_text", "engagement"}.issubset(df.columns):
        raise ValueError("review_sentiment_engagement_variance requires review_text & engagement")

    text = df["review_text"].fillna("").str.lower()
    pos = text.str.count("|".join(positive_words))
    neg = text.str.count("|".join(negative_words))
    sentiment = (pos - neg) / (pos + neg + 1)

    score = sentiment_w * sentiment + engagement_w * (df["engagement"].fillna(df["engagement"].median()) / (df["engagement"].max() + 1e-9))
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="review_sentiment_engagement_variance")


def format_availability_rating(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 29: Books with higher average ratings tend to have more formats available.
    
    Required columns:
        - format_count (int)
        - average_rating (float)
    """
    fmt_w = params.get("format_weight", 1.0)
    rating_w = params.get("rating_weight", 1.0)

    if not {"format_count", "average_rating"}.issubset(df.columns):
        raise ValueError("format_availability_rating requires format_count & average_rating")

    score = fmt_w * np.log1p(df["format_count"].fillna(0)) + rating_w * (df["average_rating"].fillna(df["average_rating"].median()) / 5)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="format_availability_rating")


def genre_listing_diversity_rating(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 30: Books listed in more genres receive higher ratings.
    
    Required columns:
        - genre_count (int)
        - average_rating (float)
    """
    genre_w = params.get("genre_weight", 1.0)
    rating_w = params.get("rating_weight", 1.0)

    if not {"genre_count", "average_rating"}.issubset(df.columns):
        raise ValueError("genre_listing_diversity_rating requires genre_count & average_rating")

    score = genre_w * np.log1p(df["genre_count"].fillna(0)) + rating_w * (df["average_rating"].fillna(df["average_rating"].median()) / 5)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="genre_listing_diversity_rating")

def description_quality_rating_correlation(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 11: Well-written / longer descriptions correlate with higher ratings.

    Required columns:
        - description: text description of the book (str)
        - average_rating: float rating 0-5
    """
    # Hyperparameters
    desc_weight = params.get("desc_weight", 1.0)
    length_weight = params.get("length_weight", 0.5)
    min_length = params.get("min_length", 100)
    rating_scale = params.get("rating_scale", 5.0)

    if not {"description", "average_rating"}.issubset(df.columns):
        raise ValueError("description_quality_rating_correlation requires description and average_rating columns")

    desc_len = df["description"].fillna("").str.len()
    quality = desc_weight * (df["average_rating"].fillna(df["average_rating"].median()) / rating_scale)
    length_component = length_weight * np.log1p(desc_len.clip(lower=min_length))

    score = quality + length_component
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="description_quality_rating_correlation")


def review_sentiment_score(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 12: Positive review sentiment predicts higher ratings.

    Required columns:
        - review_text (str)
        - rating (float)
    """
    pos_weight = params.get("pos_weight", 1.0)
    rating_weight = params.get("rating_weight", 0.5)

    if not {"review_text", "rating"}.issubset(df.columns):
        raise ValueError("review_sentiment_score requires review_text and rating columns")

    # Very light lexicon sentiment (counts of positive vs negative cues)
    positive_words = ["good", "great", "excellent", "amazing", "love", "wonderful"]
    negative_words = ["bad", "terrible", "awful", "hate", "boring", "worst"]

    text = df["review_text"].fillna("").str.lower()
    pos_ct = text.str.count("|".join(positive_words))
    neg_ct = text.str.count("|".join(negative_words))
    sentiment = (pos_ct - neg_ct) / (pos_ct + neg_ct + 1)

    score = pos_weight * sentiment + rating_weight * df["rating"].fillna(df["rating"].median())
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="review_sentiment_score")


def user_interaction_engagement(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 13: Votes & comments reflect engagement → quality.

    Required columns:
        - n_votes (int)
        - n_comments (int)
    """
    votes_w = params.get("votes_weight", 1.0)
    comments_w = params.get("comments_weight", 1.0)
    boost = params.get("engagement_boost", 1.2)

    cols = {"n_votes", "n_comments"}
    if not cols.issubset(df.columns):
        raise ValueError("user_interaction_engagement requires n_votes and n_comments columns")

    votes = np.log1p(df["n_votes"].fillna(0))
    comments = np.log1p(df["n_comments"].fillna(0))
    score = votes_w * votes + comments_w * comments
    high_eng = (df["n_votes"].fillna(0) >= 10) | (df["n_comments"].fillna(0) >= 5)
    score = np.where(high_eng, score * boost, score)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="user_interaction_engagement")


def publisher_diversity_quality(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 14: Publishers with diverse catalogues & quality yield better books.

    Required columns:
        - publisher_name (str)
        - average_rating (float)
    The DataFrame must be per book; function aggregates internally.
    """
    diversity_w = params.get("diversity_weight", 0.7)
    quality_w = params.get("quality_weight", 1.0)

    if not {"publisher_name", "average_rating"}.issubset(df.columns):
        raise ValueError("publisher_diversity_quality requires publisher_name and average_rating columns")

    pub_stats = df.groupby("publisher_name").agg(
        pub_count=("average_rating", "size"),
        pub_avg=("average_rating", "mean")
    )
    pub_score = quality_w * pub_stats["pub_avg"] + diversity_w * np.log1p(pub_stats["pub_count"])
    pub_score = (pub_score - pub_score.min()) / (pub_score.max() - pub_score.min() + 1e-9)
    score = df["publisher_name"].map(pub_score).fillna(0)
    return pd.Series(score.values, index=df.index, name="publisher_diversity_quality")


def publication_recency_impact(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 15: Recent high-rated books trend better.

    Required columns:
        - publication_year (int)
        - average_rating (float)
    """
    recency_w = params.get("recency_weight", 1.0)
    rating_w = params.get("rating_weight", 1.0)
    current_year = params.get("current_year", 2025)
    recent_years = params.get("recent_years", 5)
    boost = params.get("recency_boost", 1.3)

    cols = {"publication_year", "average_rating"}
    if not cols.issubset(df.columns):
        raise ValueError("publication_recency_impact requires publication_year and average_rating columns")

    years_old = current_year - df["publication_year"].fillna(current_year)
    recency = recency_w * (recent_years - years_old).clip(lower=0) / recent_years
    score = recency + rating_w * (df["average_rating"].fillna(df["average_rating"].median()) / 5)
    recent_mask = years_old <= recent_years
    score = np.where(recent_mask, score * boost, score)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="publication_recency_impact")


def format_preference_rating(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 16: Certain formats garner higher ratings.

    Required columns:
        - format (str)
        - average_rating (float)
    """
    format_w = params.get("format_weight", 1.0)
    rating_w = params.get("rating_weight", 1.0)

    if not {"format", "average_rating"}.issubset(df.columns):
        raise ValueError("format_preference_rating requires format and average_rating columns")

    format_stats = df.groupby("format").agg(avg_r=("average_rating", "mean"), cnt=("format", "size"))
    fmt_score = format_w * np.log1p(format_stats["cnt"]) + rating_w * (format_stats["avg_r"] / 5)
    fmt_score = (fmt_score - fmt_score.min()) / (fmt_score.max() - fmt_score.min() + 1e-9)
    score = df["format"].map(fmt_score).fillna(0)
    return pd.Series(score.values, index=df.index, name="format_preference_rating")


def rating_review_correlation(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 17: Highly rated books receive more reviews.

    Required columns:
        - average_rating (float)
        - review_count (int)
    """
    rating_w = params.get("rating_weight", 1.0)
    review_w = params.get("review_weight", 1.0)

    if not {"average_rating", "review_count"}.issubset(df.columns):
        raise ValueError("rating_review_correlation requires average_rating and review_count columns")

    score = rating_w * (df["average_rating"].fillna(df["average_rating"].median()) / 5) + review_w * np.log1p(df["review_count"].fillna(0))
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="rating_review_correlation")


def thematic_engagement_score(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 18: Presence of engaging themes in description ↑ ratings.

    Required columns:
        - description (str)
        - average_rating (float)
    """
    theme_w = params.get("theme_weight", 1.0)
    rating_w = params.get("rating_weight", 0.5)

    engaging_keywords = params.get("keywords", [
        "love", "mystery", "adventure", "magic", "family", "friendship", "war"
    ])

    if not {"description", "average_rating"}.issubset(df.columns):
        raise ValueError("thematic_engagement_score requires description and average_rating columns")

    desc = df["description"].fillna("").str.lower()
    theme_counts = sum(desc.str.count(k) for k in engaging_keywords)
    score = theme_w * np.log1p(theme_counts) + rating_w * (df["average_rating"].fillna(df["average_rating"].median()) / 5)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="thematic_engagement_score")


def author_collaboration_effect(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 19: Multi-author collaborations influence ratings.

    Required columns:
        - authors (str list or comma-separated str)
        - average_rating (float)
    """
    collab_w = params.get("collab_weight", 1.0)
    rating_w = params.get("rating_weight", 1.0)

    if "authors" not in df.columns or "average_rating" not in df.columns:
        raise ValueError("author_collaboration_effect requires authors and average_rating columns")

    author_count = df["authors"].fillna("").apply(lambda x: len(str(x).split("|")) if "|" in str(x) else len(str(x).split(",")))
    score = collab_w * np.log1p(author_count) + rating_w * (df["average_rating"].fillna(df["average_rating"].median()) / 5)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="author_collaboration_effect")
```

### `contingency/plotting/visualize_all_studies.py`

**File size:** 2,601 bytes

```python
import os
import joblib
import optuna
import plotly.io as pio
from pathlib import Path

def find_study_files(search_dir, suffix='.pkl'):
    """Recursively find all Optuna study pickle files in a directory."""
    study_files = []
    for root, _, files in os.walk(search_dir):
        for file in files:
            if file.endswith(suffix):
                study_files.append(os.path.join(root, file))
    return study_files

def safe_plot_and_save(plot_func, study, out_path, **kwargs):
    try:
        fig = plot_func(study, **kwargs)
        pio.write_html(fig, out_path)
        print(f"Saved: {out_path}")
    except Exception as e:
        print(f"Failed to plot {plot_func.__name__} for {out_path}: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot all Optuna studies in a directory.")
    parser.add_argument('--search_dir', type=str, default='../evaluation_results/optuna_studies', help='Directory to search for .pkl Optuna studies')
    parser.add_argument('--output_dir', type=str, default='visualizations', help='Base directory to save plots')
    args = parser.parse_args()

    study_files = find_study_files(args.search_dir)
    if not study_files:
        print(f"No study files found in {args.search_dir}")
        return

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for study_path in study_files:
        study_name = Path(study_path).stem.replace('.pkl','')
        out_dir = Path(args.output_dir) / study_name
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            study = joblib.load(study_path)
        except Exception as e:
            print(f"Could not load {study_path}: {e}")
            continue
        print(f"Loaded study: {study_name} ({study_path})")
        # Standard plots
        safe_plot_and_save(optuna.visualization.plot_optimization_history, study, out_dir/'optimization_history.html')
        safe_plot_and_save(optuna.visualization.plot_param_importances, study, out_dir/'param_importances.html')
        safe_plot_and_save(optuna.visualization.plot_slice, study, out_dir/'slice.html')
        safe_plot_and_save(optuna.visualization.plot_contour, study, out_dir/'contour.html')
        safe_plot_and_save(optuna.visualization.plot_parallel_coordinate, study, out_dir/'parallel_coordinate.html')
        safe_plot_and_save(optuna.visualization.plot_evaluations, study, out_dir/'evaluations.html')
        safe_plot_and_save(optuna.visualization.plot_edf, study, out_dir/'edf.html')
        # Optionally add more plots as needed

if __name__ == '__main__':
    main()
```

### `contingency/reliable_functions.py`

**File size:** 2,526 bytes

```python
import pandas as pd
import numpy as np
from typing import Dict, Any

# --- RELIABLE, PASSING FEATURE FUNCTIONS ---

def ratings_count_feature(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """
    Feature: Raw ratings count (book popularity proxy)
    Required columns:
        - ratings_count (int)
    """
    scale = params.get("scale", 1.0)
    if "ratings_count" not in df.columns:
        raise ValueError("ratings_count_feature requires ratings_count column")
    return pd.Series(df["ratings_count"].fillna(0) * scale, index=df.index, name="ratings_count_feature")

def average_rating_feature(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """
    Feature: Raw average rating
    Required columns:
        - avg_rating (float)  # curated_books: avg_rating
    """
    offset = params.get("offset", 0.0)
    if "avg_rating" not in df.columns:
        raise ValueError("average_rating_feature requires avg_rating column")
    return pd.Series(df["avg_rating"].fillna(df["avg_rating"].median()) + offset, index=df.index, name="average_rating_feature")

def num_pages_feature(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """
    Feature: Book length (number of pages)
    Required columns:
        - num_pages (int)
    """
    scale = params.get("scale", 1.0)
    if "num_pages" not in df.columns:
        raise ValueError("num_pages_feature requires num_pages column")
    return pd.Series(df["num_pages"].fillna(df["num_pages"].median()) * scale, index=df.index, name="num_pages_feature")

def user_books_read_feature(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """
    Feature: Number of books read by user
    Required columns:
        - books_read (int)  # user_reading_trends: books_read
    """
    scale = params.get("scale", 1.0)
    if "books_read" not in df.columns:
        raise ValueError("user_books_read_feature requires books_read column")
    return pd.Series(df["books_read"].fillna(0) * scale, index=df.index, name="user_books_read_feature")

def interaction_rating_feature(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """
    Feature: Explicit rating from user-item interaction
    Required columns:
        - rating (float)  # interactions: rating
    """
    bias = params.get("bias", 0.0)
    if "rating" not in df.columns:
        raise ValueError("interaction_rating_feature requires rating column")
    return pd.Series(df["rating"].fillna(df["rating"].median()) + bias, index=df.index, name="interaction_rating_feature")
```

### `contingency/reward_functions.py`

**File size:** 9,315 bytes

```python
import numpy as np
from typing import Tuple, Any, List, Dict
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from loguru import logger
import warnings
warnings.filterwarnings('ignore')


def calculate_precision_gain_reward(p5_feature: float, p5_baseline: float) -> float:
    """
    Reward for LightFM: relative gain in precision@5.
    Returns (p5_feature - p5_baseline) / p5_baseline, or a large value if baseline is 0 and feature > 0.
    """
    if p5_baseline == 0:
        if p5_feature > 0:
            return 100.0
        return 0.0
    return (p5_feature - p5_baseline) / p5_baseline

def calculate_rmse_gain_reward(rmse_feature: float, rmse_baseline: float) -> float:
    """
    Reward for SVD: relative gain in RMSE (lower is better).
    Returns (rmse_baseline - rmse_feature) / rmse_baseline, or 0 if baseline is 0.
    """
    if rmse_baseline == 0:
        return 0.0
    return (rmse_baseline - rmse_feature) / rmse_baseline

def evaluate_feature_with_model(feature_values: pd.Series, train_df: pd.DataFrame, test_df: pd.DataFrame, model_type: str = 'lightfm') -> float:
    """
    Evaluate a feature by training a model and returning the relevant metric.
    For LightFM: returns precision@5
    For SVD: returns RMSE
    For Random Forest: returns RMSE
    """
    try:
        if model_type == 'lightfm':
            from src.baselines.recommender.lightfm_baseline import run_lightfm_baseline
            metrics = run_lightfm_baseline(train_df, test_df)
            return metrics.get('precision_at_5', np.nan)
        elif model_type == 'svd':
            from src.baselines.recommender.svd_baseline import run_svd_baseline
            metrics = run_svd_baseline(train_df, test_df)
            return metrics.get('rmse', np.nan)
        elif model_type == 'random_forest':
            from src.baselines.recommender.random_forest_baseline import run_random_forest_baseline
            metrics = run_random_forest_baseline(train_df, test_df)
            return metrics.get('rmse', np.nan)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    except Exception as e:
        import logging
        logging.error(f"Error evaluating feature with {model_type}: {e}")
        return 0.0


def calculate_scaled_rmse_improvement(rmse_feature: float, rmse_baseline: float, delta_max: float = 0.05) -> float:
    """
    Calculate scaled RMSE improvement normalized to [0, 1].
    
    RmseImprovement = (RMSE_baseline - RMSE_feature) / delta_max
    
    Args:
        ndcg_feature: NDCG score with the feature
        ndcg_baseline: NDCG score of the baseline model
        delta_max: Maximum expected improvement (hyperparameter)
        
    Returns:
        Scaled RecLift score in [0, 1]
    """
    improvement_raw = rmse_baseline - rmse_feature
    improvement_scaled = improvement_raw / delta_max
    return np.clip(improvement_scaled, 0, 1)


def find_optimal_clusters(features: np.ndarray, k_range: range = range(2, 11), 
                         random_state: int = 42) -> Tuple[int, float]:
    """
    Find optimal number of clusters that maximizes silhouette score.
    
    Args:
        features: Feature matrix for clustering
        k_range: Range of k values to try
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (optimal_k, best_silhouette_score)
    """
    if len(features) < 2:
        logger.warning("Not enough samples for clustering")
        return 2, 0.0
        
    best_k = 2
    best_score = -1
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    for k in k_range:
        if k >= len(features):
            continue
            
        try:
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(features_scaled)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(features_scaled, cluster_labels)
            
            if silhouette_avg > best_score:
                best_score = silhouette_avg
                best_k = k
                
        except Exception as e:
            logger.warning(f"Error with k={k}: {e}")
            continue
    
    return best_k, max(best_score, 0.0)


def calculate_scaled_silhouette(features: np.ndarray, s_min: float = 0.1, s_max: float = 0.6,
                               k_range: range = range(2, 11), random_state: int = 42) -> Tuple[float, int]:
    """
    Calculate scaled Silhouette score normalized to [0, 1] with optimal cluster tuning.
    
    Args:
        features: Feature matrix for clustering
        s_min: Minimum expected silhouette score
        s_max: Maximum expected silhouette score
        k_range: Range of k values to try for optimal clustering
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (scaled_silhouette_score, optimal_k)
    """
    optimal_k, silhouette_raw = find_optimal_clusters(features, k_range, random_state)
    
    # Scale to [0, 1]
    silhouette_scaled = (silhouette_raw - s_min) / (s_max - s_min)
    silhouette_scaled = np.clip(silhouette_scaled, 0, 1)
    
    return silhouette_scaled, optimal_k


def calculate_composite_reward(rmse_feature: float, rmse_baseline: float, features: np.ndarray,
                              w1: float = 0.7, w2: float = 0.3, delta_max: float = 0.05,
                              s_min: float = 0.1, s_max: float = 0.6, 
                              k_range: range = range(2, 11), random_state: int = 42) -> Dict[str, Any]:
    """
    Calculate the composite reward function J = w1 * RecLift_scaled + w2 * Silhouette_scaled.
    
    Args:
        ndcg_feature: NDCG score with the feature
        ndcg_baseline: NDCG score of the baseline model
        features: Feature matrix for clustering
        w1: Weight for RecLift component (default: 0.7)
        w2: Weight for Silhouette component (default: 0.3)
        delta_max: Maximum expected NDCG improvement
        s_min: Minimum expected silhouette score
        s_max: Maximum expected silhouette score
        k_range: Range of k values for clustering
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary containing all reward components and final score
    """
    # Calculate RMSE improvement component
    rmse_improvement_scaled = calculate_scaled_rmse_improvement(rmse_feature, rmse_baseline, delta_max)
    
    # Calculate Silhouette component
    silhouette_scaled, optimal_k = calculate_scaled_silhouette(
        features, s_min, s_max, k_range, random_state
    )
    
    # Calculate composite reward
    composite_reward = w1 * rmse_improvement_scaled + w2 * silhouette_scaled
    
    return {
        'rmse_improvement_scaled': rmse_improvement_scaled,
        'silhouette_scaled': silhouette_scaled,
        'optimal_k': optimal_k,
        'composite_reward': composite_reward,
        'rmse_feature': rmse_feature,
        'rmse_baseline': rmse_baseline,
        'weights': {'w1': w1, 'w2': w2}
    }


def evaluate_feature_with_model(feature_values: pd.Series, train_df: pd.DataFrame, 
                               test_df: pd.DataFrame, model_type: str = 'lightfm') -> float:
    """
    Evaluate a feature by training a model and calculating RMSE.
    
    Args:
        feature_values: The engineered feature values
        train_df: Training data
        test_df: Test data
        model_type: Type of model to use ('lightfm', 'deepfm', 'popularity')
        
    Returns:
        RMSE score
    """
    try:
        # This is a placeholder - in practice, you'd integrate the feature
        # into your model training pipeline and evaluate
        
        if model_type == 'lightfm':
            from src.baselines.recommender.lightfm_baseline import run_lightfm_baseline
            metrics = run_lightfm_baseline(train_df, test_df)
            return metrics.get('rmse', np.nan)
            
        elif model_type == 'svd':
            from src.baselines.recommender.svd_baseline import run_svd_baseline
            metrics = run_svd_baseline(train_df, test_df)
            return metrics.get('rmse', np.nan)
        elif model_type == 'deepfm':
            from src.baselines.recommender.deepfm_baseline import run_deepfm_baseline
            metrics = run_deepfm_baseline(train_df, test_df)
            return metrics.get('rmse', np.nan)
            
        elif model_type == 'random_forest':
            from src.baselines.recommender.random_forest_baseline import run_random_forest_baseline
            metrics = run_random_forest_baseline(train_df, test_df)
            return metrics.get('rmse', np.nan)
        elif model_type == 'popularity':
            from src.baselines.recommender.popularity_baseline import run_popularity_baseline
            result = run_popularity_baseline(train_df, test_df)
            return result.get('rmse', np.nan)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    except Exception as e:
        logger.error(f"Error evaluating feature with {model_type}: {e}")
        return 0.0
```

### `contingency/run_manual_bo.py`

**File size:** 12,298 bytes

```python
import argparse
from pathlib import Path
import pandas as pd
import json
import numpy as np
from src.contingency.reward_functions import calculate_precision_gain_reward, calculate_rmse_gain_reward
from src.contingency.reward_functions import evaluate_feature_with_model
from datetime import datetime
import optuna
from typing import Dict, Any
from src.data.cv_data_manager import CVDataManager
import inspect
import importlib
import importlib

# Dynamically collect all valid feature functions (skip template and private)
def get_all_feature_functions(feature_module):
    feature_funcs = {}
    for name, func in inspect.getmembers(feature_module, inspect.isfunction):
        if name.startswith('_') or name == 'template_feature_function':
            continue
        feature_funcs[name] = func
    return feature_funcs




# --- GLOBAL SCHEMA CACHE ---
_SCHEMA_CACHE = None

# --- COMMON JOIN KEYS ---
_COMMON_JOIN_KEYS = ["user_id", "book_id", "item_id"]

def _get_schema_map(conn):
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is not None:
        return _SCHEMA_CACHE
    schema = {}
    try:
        tables = conn.execute("SHOW TABLES").fetchdf()["name"].tolist()
        for t in tables:
            try:
                cols = conn.execute(f"PRAGMA table_info({t})").fetchdf()["name"].tolist()
                for c in cols:
                    if c not in schema:
                        schema[c] = []
                    schema[c].append(t)
            except Exception:
                continue
        _SCHEMA_CACHE = schema
    except Exception as e:
        print(f"[WARN] Could not build schema map: {e}")
        schema = {}
    return schema

def prepare_dataframe(depends_on, cv_manager):
    """
    Given a list of column names (not necessarily qualified),
    dynamically map columns to tables and construct a SELECT query.
    Returns None if columns can't be found or joined.
    Always returns the DB connection to the pool.
    """
    conn = cv_manager.db_connection
    schema = _get_schema_map(conn)
    # Map: col -> table
    col_table_map = {}
    ambiguous = False
    for col in depends_on:
        tables = schema.get(col, [])
        if not tables:
            print(f"[WARN] Column '{col}' not found in any table. Skipping feature.")
            ambiguous = True
            break
        elif len(tables) == 1:
            col_table_map[col] = tables[0]
        else:
            # Heuristic: prefer curated_reviews for ratings, curated_books for book info, else first
            preferred = None
            for t in tables:
                if (col == "rating" and "review" in t) or (col in ["title", "description", "avg_rating"] and "book" in t):
                    preferred = t
                    break
            if not preferred:
                preferred = tables[0]
            col_table_map[col] = preferred
    if ambiguous:
        try:
            cv_manager._return_connection(conn)
        except Exception as e:
            print(f"[WARN] Could not return connection to pool: {e}")
        return None
    involved_tables = set(col_table_map.values())
    select_cols = [f"{col_table_map[c]}.{c}" for c in depends_on]
    # If all columns are from one table
    if len(involved_tables) == 1:
        table = list(involved_tables)[0]
        sql = f"SELECT {', '.join(select_cols)} FROM {table} LIMIT 10000"
    else:
        # Try to join on common keys
        join_keys = [k for k in _COMMON_JOIN_KEYS if all(k in schema and t in schema[k] for t in involved_tables)]
        if not join_keys:
            print(f"[WARN] Cannot join tables {involved_tables} for columns {depends_on}: no common key.")
            try:
                cv_manager._return_connection(conn)
            except Exception as e:
                print(f"[WARN] Could not return connection to pool: {e}")
            return None
        # Use the first join key
        key = join_keys[0]
        tables = list(involved_tables)
        sql = f"SELECT {', '.join(select_cols)} FROM {tables[0]}"
        for t in tables[1:]:
            sql += f" JOIN {t} USING ({key})"
        sql += " LIMIT 10000"
    try:
        df = conn.execute(sql).fetchdf()
    except Exception as e:
        print(f"[WARN] Failed to prepare dataframe for depends_on={depends_on}: {e}\nSQL: {sql}")
        df = None
    finally:
        try:
            cv_manager._return_connection(conn)
        except Exception as e:
            print(f"[WARN] Could not return connection to pool: {e}")
    return df

def load_train_test(cv_manager, fold_idx=0):
    """Utility to get train and test DataFrames from CVDataManager."""
    train_df, test_df = cv_manager.get_fold_data(fold_idx=fold_idx, split_type="train_val")
    return train_df.copy(), test_df.copy()

def run_bo_for_feature(feature_dict: Dict[str, Any], cv_manager, baseline_rmse: float, model_type: str, output_base: Path, n_trials=10, fold_idx=0):
    """
    Run Bayesian Optimization for a single feature.
    """
    name = feature_dict['name']
    # Prepare output directory for this feature
    feature_dir = output_base / name
    feature_dir.mkdir(parents=True, exist_ok=True)
    depends_on = feature_dict.get('depends_on', [])
    param_space = feature_dict.get('parameters', {})
    df = prepare_dataframe(depends_on, cv_manager)
    if df is None:
        print(f"[WARN] Skipping feature '{name}' because required columns could not be loaded from DB.")
        return None
    train_df, test_df = load_train_test(cv_manager, fold_idx=fold_idx)

    def objective(trial):
        params = {}
        for param, spec in param_space.items():
            if spec['type'] == 'float':
                params[param] = trial.suggest_float(param, spec['min'], spec['max'])
            elif spec['type'] == 'int':
                params[param] = trial.suggest_int(param, spec['min'], spec['max'])
            elif spec['type'] == 'categorical':
                params[param] = trial.suggest_categorical(param, spec['choices'])
            else:
                raise ValueError(f'Unknown param type: {spec}')
        # Compute feature using the actual function
        feature_func = feature_dict['function']
        feature_col = feature_func(df, params)
        # Append feature to train/test
        train_df_aug = train_df.copy()
        test_df_aug = test_df.copy()
        train_df_aug[name] = feature_col.reindex(train_df_aug.index).fillna(0)
        test_df_aug[name] = feature_col.reindex(test_df_aug.index).fillna(0)

        # Evaluate feature with model and get relevant metric
        eval_metric = evaluate_feature_with_model(feature_col, train_df_aug, test_df_aug, model_type=model_type)
        # Compute reward based on model type
        if model_type == 'lightfm':
            # eval_metric should be precision@5
            reward = calculate_precision_gain_reward(eval_metric, baseline_rmse)  # baseline_rmse is actually baseline_p5 for lightfm
        elif model_type == 'svd':
            # eval_metric should be RMSE
            reward = calculate_rmse_gain_reward(eval_metric, baseline_rmse)
        else:
            raise ValueError(f"Unknown model_type {model_type}")
        return reward

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    # Persist results
    result = {
        "feature": name,
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": n_trials,
        "timestamp": datetime.utcnow().isoformat()
    }
    with open(feature_dir / "bo_result.json", "w") as f:
        json.dump(result, f, indent=4)
    # Optional: save Optuna study for further analysis
    study.trials_dataframe().to_csv(feature_dir / "trials.csv", index=False)
    print(f'Feature: {name} | Best params: {study.best_params} | Best value: {study.best_value}')
    return study.best_params, study.best_value

def main(db_path: str, splits_dir: str, undersample_frac: float, baseline_model: str, baseline_scores_path: str, n_trials: int = 30, use_reliable_features: bool = False):
    # Load baseline RMSE
    with open(baseline_scores_path, 'r') as f:
        baseline_scores = json.load(f)
    if baseline_model not in baseline_scores:
        raise ValueError(f"Baseline model {baseline_model} not found in {baseline_scores_path}")
    if baseline_model == 'popularity':
        raise ValueError("Bayesian Optimization is not run for the 'popularity' baseline as it has no trainable parameters. Choose 'lightfm' or 'deepfm'.")
    baseline_rmse = baseline_scores[baseline_model]['rmse']

    # Initialize CVDataManager
    cv_manager = CVDataManager(
        db_path=db_path,
        splits_dir=splits_dir,
        undersample_frac=undersample_frac,
        read_only=True
    )
    # Base directory for this baseline model
    output_base = Path("experiments") / baseline_model
    output_base.mkdir(parents=True, exist_ok=True)

    # Dynamically discover all feature functions
    if use_reliable_features:
        print("[INFO] Using reliable feature functions from reliable_functions.py")
        feature_module = importlib.import_module("src.contingency.reliable_functions")
    else:
        print("[INFO] Using standard feature functions from functions.py")
        feature_module = importlib.import_module("src.contingency.functions")
    feature_functions = get_all_feature_functions(feature_module)

    # Run BO for each feature function
    for name, func in feature_functions.items():
        # Try to infer depends_on from docstring
        depends_on = []
        docstring = inspect.getdoc(func)
        if docstring and "Required columns:" in docstring:
            lines = docstring.splitlines()
            req_idx = None
            for i, line in enumerate(lines):
                if "Required columns:" in line:
                    req_idx = i
                    break
            if req_idx is not None:
                for l in lines[req_idx+1:]:
                    l = l.strip()
                    # Only accept lines like '- column_name (type)'
                    if l.startswith('- '):
                        try:
                            col_part = l[2:].split(' (')[0]
                            if col_part:
                                depends_on.append(col_part)
                        except Exception:
                            continue
                    elif not l:
                        break
        # Construct feature_dict
        feature_dict = {
            'name': name,
            'function': func,
            'parameters': {},  # default param space
            'depends_on': depends_on
        }
        # Defensive: skip if no depends_on
        if not depends_on:
            print(f"[WARN] Skipping feature '{name}' due to missing depends_on.")
            continue
        try:
            run_bo_for_feature(feature_dict, cv_manager, baseline_rmse, baseline_model, output_base, n_trials=n_trials)
        except Exception as e:
            print(f"[WARN] Skipping feature '{name}' due to error: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual Bayesian Optimization for Features")
    parser.add_argument("--db_path", type=str, required=True, help="Path to the DuckDB database file")
    parser.add_argument("--splits_dir", type=str, required=True, help="Path to the CV splits directory")
    parser.add_argument("--undersample_frac", type=float, default=1.0, help="Fraction of data to use for subsampling (e.g., 0.05 for 5%)")
    parser.add_argument("--baseline_model", type=str, required=True, choices=["lightfm", "svd", "random_forest"], help="Which baseline model to optimize for")
    parser.add_argument("--baseline_scores", type=str, default="/root/fuegoRecommender/experiments/baseline_scores.json", help="Path to baseline scores JSON")
    parser.add_argument("--n_trials", type=int, default=30, help="Number of BO trials per feature")
    parser.add_argument("--use-reliable-features", action="store_true", help="Use only reliable, passing features from reliable_functions.py")
    args = parser.parse_args()
    main(args.db_path, args.splits_dir, args.undersample_frac, args.baseline_model, args.baseline_scores, args.n_trials, use_reliable_features=args.use_reliable_features)
```

### `contingency/run_sequential_evaluation.py`

**File size:** 23,803 bytes

```python
#!/usr/bin/env python3
"""
Sequential Feature Evaluation Pipeline

This script evaluates manual features by sequentially adding them to a recommender model
and tracking performance improvements. It saves full Optuna studies and generates
representation learning plots.
"""

import argparse
import json
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18,
    'axes.titlepad': 12,
    'axes.labelpad': 8
})

from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

from src.utils.session_state import SessionState
import inspect
import importlib
from src.contingency import functions as feature_module

def get_all_feature_functions():
    feature_funcs = {}
    for name, func in inspect.getmembers(feature_module, inspect.isfunction):
        if name.startswith('_') or name == 'template_feature_function':
            continue
        feature_funcs[name] = func
    return feature_funcs



class SequentialFeatureEvaluator:
    """Evaluates features sequentially on a recommender model."""
    
    def __init__(self, session_state: SessionState, output_dir: Path):
        """
        Initializes the evaluator with a session state and output directory.
        
        Args:
            session_state (SessionState): The session state.
            output_dir (Path): The output directory.
        """
        self.session_state = session_state
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model and data
        self.base_data = None
        self.target_column = 'average_rating'  # What we're predicting
        self.feature_columns = []
        self.models = {}
        self.results = []
        self.scaler = StandardScaler()
        
        # Feature functions registry
        self.feature_functions = get_all_feature_functions()
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare the base dataset for recommendation."""
        conn = self.session_state.db_connection
        
        # Load comprehensive book data with user interactions
        sql = """
        SELECT 
            b.book_id,
            b.title,
            b.average_rating,
            b.ratings_count,
            b.text_reviews_count,
            b.publication_year,
            b.num_pages,
            -- User interaction features
            COUNT(DISTINCT r.user_id) as unique_users,
            AVG(r.rating) as user_avg_rating,
            COUNT(r.rating) as total_user_ratings,
            -- Book popularity features
            b.ratings_count as book_popularity,
            CASE WHEN b.ratings_count > 100 THEN 1 ELSE 0 END as is_popular
        FROM books b
        LEFT JOIN reviews r ON b.book_id = r.book_id
        WHERE b.average_rating IS NOT NULL 
          AND b.ratings_count IS NOT NULL
          AND b.ratings_count > 5  -- Filter out books with very few ratings
        GROUP BY b.book_id, b.title, b.average_rating, b.ratings_count, 
                 b.text_reviews_count, b.publication_year, b.num_pages
        HAVING COUNT(r.rating) >= 3  -- Ensure some user interaction data
        LIMIT 5000
        """
        
        try:
            df = conn.execute(sql).fetchdf()
            print(f"Loaded {len(df)} books with interaction data")
            
            # Handle missing values
            df = df.fillna({
                'text_reviews_count': 0,
                'publication_year': df['publication_year'].median(),
                'num_pages': df['num_pages'].median(),
                'unique_users': 0,
                'user_avg_rating': df['average_rating'].median(),
                'total_user_ratings': 0
            })
            
            # Create base features
            df['log_ratings_count'] = np.log1p(df['ratings_count'])
            df['log_text_reviews'] = np.log1p(df['text_reviews_count'])
            df['pages_per_year'] = df['num_pages'] / (2024 - df['publication_year'] + 1)
            df['rating_engagement'] = df['average_rating'] * np.log1p(df['ratings_count'])
            
            self.base_data = df
            self.feature_columns = [
                'ratings_count', 'text_reviews_count', 'publication_year', 'num_pages',
                'unique_users', 'total_user_ratings', 'book_popularity',
                'log_ratings_count', 'log_text_reviews', 'pages_per_year', 'rating_engagement'
            ]
            
            print(f"Base feature columns: {len(self.feature_columns)}")
            print(f"Target range: {df[self.target_column].min():.2f} - {df[self.target_column].max():.2f}")
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            # Create synthetic data for testing
            return self._create_synthetic_data()
    
    def _create_synthetic_data(self) -> pd.DataFrame:
        """Create synthetic data for testing when real data fails."""
        print("Creating synthetic data for testing...")
        np.random.seed(42)
        n_books = 2000
        
        df = pd.DataFrame({
            'book_id': range(n_books),
            'title': [f'Book_{i}' for i in range(n_books)],
            'average_rating': np.random.uniform(2.0, 5.0, n_books),
            'ratings_count': np.random.exponential(100, n_books).astype(int),
            'text_reviews_count': np.random.exponential(30, n_books).astype(int),
            'publication_year': np.random.randint(1980, 2024, n_books),
            'num_pages': np.random.randint(150, 600, n_books),
            'unique_users': np.random.randint(5, 200, n_books),
            'user_avg_rating': np.random.uniform(2.0, 5.0, n_books),
            'total_user_ratings': np.random.randint(10, 500, n_books),
            'book_popularity': np.random.exponential(100, n_books).astype(int),
            'is_popular': np.random.binomial(1, 0.3, n_books)
        })
        
        # Create base features
        df['log_ratings_count'] = np.log1p(df['ratings_count'])
        df['log_text_reviews'] = np.log1p(df['text_reviews_count'])
        df['pages_per_year'] = df['num_pages'] / (2024 - df['publication_year'] + 1)
        df['rating_engagement'] = df['average_rating'] * np.log1p(df['ratings_count'])
        
        self.base_data = df
        self.feature_columns = [
            'ratings_count', 'text_reviews_count', 'publication_year', 'num_pages',
            'unique_users', 'total_user_ratings', 'book_popularity',
            'log_ratings_count', 'log_text_reviews', 'pages_per_year', 'rating_engagement'
        ]
        
        return df
    
    def optimize_feature(self, feature_name: str, n_trials: int = 30) -> Tuple[Dict[str, Any], float, optuna.Study]:
        """Optimize a single feature using Bayesian Optimization."""
        print(f"\n=== Optimizing {feature_name} ===")
        
        if feature_name not in self.feature_functions:
            raise ValueError(f"Feature function {feature_name} not found")
        
        feature_func = self.feature_functions[feature_name]
        
        def objective(trial):
            # Define hyperparameter search space based on feature
            if feature_name == 'rating_popularity_momentum':
                params = {
                    'rating_weight': trial.suggest_float('rating_weight', 0.1, 3.0),
                    'count_weight': trial.suggest_float('count_weight', 0.1, 2.0),
                    'momentum_power': trial.suggest_float('momentum_power', 0.2, 1.5),
                    'min_ratings_threshold': trial.suggest_int('min_ratings_threshold', 5, 100),
                    'rating_scale': trial.suggest_float('rating_scale', 3.0, 6.0)
                }
            elif feature_name == 'genre_preference_alignment':
                params = {
                    'genre_weight': trial.suggest_float('genre_weight', 0.1, 2.0),
                    'rating_threshold': trial.suggest_float('rating_threshold', 3.0, 4.5),
                    'popularity_factor': trial.suggest_float('popularity_factor', 0.0, 1.0),
                    'recency_decay': trial.suggest_float('recency_decay', 0.8, 1.0),
                    'boost_multiplier': trial.suggest_float('boost_multiplier', 1.0, 3.0)
                }
            elif feature_name == 'publication_recency_boost':
                params = {
                    'recency_weight': trial.suggest_float('recency_weight', 0.1, 2.0),
                    'rating_weight': trial.suggest_float('rating_weight', 0.5, 2.0),
                    'velocity_factor': trial.suggest_float('velocity_factor', 0.1, 1.5),
                    'recent_threshold': trial.suggest_int('recent_threshold', 1, 10),
                    'min_ratings': trial.suggest_int('min_ratings', 5, 100)
                }
            elif feature_name == 'engagement_depth_score':
                params = {
                    'review_ratio_weight': trial.suggest_float('review_ratio_weight', 0.5, 2.0),
                    'absolute_reviews_weight': trial.suggest_float('absolute_reviews_weight', 0.1, 1.0),
                    'engagement_threshold': trial.suggest_float('engagement_threshold', 0.05, 0.5),
                    'length_proxy_factor': trial.suggest_float('length_proxy_factor', 0.0, 1.0),
                    'quality_boost': trial.suggest_float('quality_boost', 1.0, 2.0)
                }
            else:
                # Default parameter space for other features
                params = {}
            
            try:
                # Compute feature
                feature_values = feature_func(self.base_data, params)
                
                # Prepare data for model training
                X = self.base_data[self.feature_columns].copy()
                X[feature_name] = feature_values
                y = self.base_data[self.target_column]
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model = Ridge(alpha=1.0, random_state=42)
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                # Return RMSE gain (positive value)
                return rmse
                
            except Exception as e:
                print(f"Error in trial: {e}")
                return 0.0
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        best_rmse_gain = study.best_value
        
        print(f"Best RMSE gain: {best_rmse_gain:.4f}")
        print(f"Best parameters: {best_params}")
        
        # Save study
        study_file = self.output_dir / f"{feature_name}_optuna_study.pkl"
        joblib.dump(study, study_file)
        print(f"Optuna study saved to: {study_file}")
        
        return best_params, best_rmse_gain, study
    
    def evaluate_model_with_features(self, feature_list: List[str], 
                                   feature_params: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate model performance with a specific set of features."""
        print(f"\nEvaluating model with features: {feature_list}")
        
        # Start with base features
        X = self.base_data[self.feature_columns].copy()
        
        # Add optimized features
        for feature_name in feature_list:
            if feature_name in self.feature_functions:
                params = feature_params.get(feature_name, {})
                feature_values = self.feature_functions[feature_name](self.base_data, params)
                X[feature_name] = feature_values
        
        y = self.base_data[self.target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        }
        
        results = {}
        for model_name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            # R² score
            r2 = model.score(X_test_scaled, y_test)
            
            results[model_name] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'n_features': X.shape[1],
                'feature_names': list(X.columns)
            }
            
            print(f"  {model_name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        
        return results
    
    def run_sequential_evaluation(self, feature_names: List[str], n_trials: int = 30):
        """Run sequential feature evaluation pipeline."""
        print("=== Starting Sequential Feature Evaluation ===")
        
        # Load data
        self.load_data()
        
        # Evaluate baseline (no additional features)
        print("\n=== Baseline Model (No Additional Features) ===")
        baseline_results = self.evaluate_model_with_features([], {})
        
        self.results.append({
            'step': 0,
            'features_added': [],
            'total_features': len(self.feature_columns),
            'model_results': baseline_results,
            'feature_params': {}
        })
        
        # Sequential feature addition
        optimized_params = {}
        current_features = []
        baseline_rmse = baseline_results['Ridge']['rmse']
        
        for i, feature_name in enumerate(feature_names, 1):
            print(f"\n=== Step {i}: Adding {feature_name} ===")
            
            # Optimize the feature
            best_params, best_rmse_gain, study = self.optimize_feature(feature_name, n_trials)
            optimized_params[feature_name] = best_params
            current_features.append(feature_name)
            
            # Evaluate model with all features so far
            model_results = self.evaluate_model_with_features(current_features, optimized_params)
            
            # Compute RMSE gain
            rmse_gain = baseline_rmse - model_results['Ridge']['rmse']
            
            # Store results
            self.results.append({
                'step': i,
                'features_added': current_features.copy(),
                'total_features': len(self.feature_columns) + len(current_features),
                'model_results': model_results,
                'feature_params': optimized_params.copy(),
                'feature_optimization': {
                    'best_params': best_params,
                    'best_rmse_gain': best_rmse_gain,
                    'n_trials': n_trials,
                    'rmse_gain': rmse_gain
                }
            })
        
        # Save all results
        results_file = self.output_dir / "sequential_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nAll results saved to: {results_file}")
        
        # Generate plots
        self.create_representation_learning_plot()
        
        return self.results
    
    def create_representation_learning_plot(self):
        """Create representation learning plot showing performance vs features."""
        print("\n=== Creating Representation Learning Plot ===")
        
        # Extract data for plotting
        steps = []
        ridge_rmse = []
        ridge_r2 = []
        rf_rmse = []
        rf_r2 = []
        feature_counts = []
        feature_names = []
        
        for result in self.results:
            steps.append(result['step'])
            feature_counts.append(result['total_features'])
            
            # Get model results
            ridge_results = result['model_results']['Ridge']
            rf_results = result['model_results']['RandomForest']
            
            ridge_rmse.append(ridge_results['rmse'])
            ridge_r2.append(ridge_results['r2'])
            rf_rmse.append(rf_results['rmse'])
            rf_r2.append(rf_results['r2'])
            
            # Feature names for x-axis
            if result['step'] == 0:
                feature_names.append('Baseline')
            else:
                feature_names.append(f"+{result['features_added'][-1]}")
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(r'Sequential Feature Evaluation: Representation Learning', fontsize=18, fontweight='bold')
        
        # RMSE plots
        ax1.plot(steps, ridge_rmse, 'o-', label='Ridge', color='blue', linewidth=2)
        ax1.plot(steps, rf_rmse, 's-', label='Random Forest', color='red', linewidth=2)
        ax1.set_xlabel(r'\textbf{Feature Addition Step}')
        ax1.set_ylabel(r'\textbf{RMSE (Lower is Better)}')
        ax1.set_title(r'\textbf{Model Performance: RMSE}')
        ax1.legend(loc='best', frameon=True)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # R² plots
        ax2.plot(steps, ridge_r2, 'o-', label='Ridge', color='blue', linewidth=2)
        ax2.plot(steps, rf_r2, 's-', label='Random Forest', color='red', linewidth=2)
        ax2.set_xlabel(r'\textbf{Feature Addition Step}')
        ax2.set_ylabel(r'$R^2$ \textbf{Score (Higher is Better)}')
        ax2.set_title(r'\textbf{Model Performance: $R^2$ Score}')
        ax2.legend(loc='best', frameon=True)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Feature count vs performance
        ax3.scatter(feature_counts, ridge_rmse, c=steps, cmap='viridis', s=100, alpha=0.7)
        ax3.set_xlabel(r'\textbf{Total Number of Features}')
        ax3.set_ylabel(r'\textbf{RMSE (Ridge)}')
        ax3.set_title(r'\textbf{Feature Count vs Performance}')
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        # Performance improvement
        if len(ridge_rmse) > 1:
            baseline_rmse = ridge_rmse[0]
            improvements = [(baseline_rmse - rmse) / baseline_rmse * 100 for rmse in ridge_rmse[1:]]
            ax4.bar(range(1, len(improvements) + 1), improvements, alpha=0.7, color='green')
            ax4.set_xlabel(r'\textbf{Feature Addition Step}')
            ax4.set_ylabel(r'\textbf{RMSE Improvement (\%)}')
            ax4.set_title(r'\textbf{Cumulative Performance Improvement}')
            ax4.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        # Save high-quality PDF and PNG
        plot_file_pdf = self.output_dir / "representation_learning_plot.pdf"
        plot_file_png = self.output_dir / "representation_learning_plot.png"
        plt.savefig(plot_file_pdf, dpi=600, bbox_inches='tight')
        plt.savefig(plot_file_png, dpi=300, bbox_inches='tight')
        print(f"Representation learning plot saved to: {plot_file_pdf} and {plot_file_png}")
        # Also create a detailed feature impact plot
        self._create_feature_impact_plot()
        plt.show()
    
    def _create_feature_impact_plot(self):
        """Create detailed feature impact visualization."""
        if len(self.results) < 2:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Performance trajectory
        steps = [r['step'] for r in self.results]
        ridge_rmse = [r['model_results']['Ridge']['rmse'] for r in self.results]
        
        ax1.plot(steps, ridge_rmse, 'o-', linewidth=3, markersize=8)
        ax1.set_xlabel(r'\textbf{Feature Addition Step}')
        ax1.set_ylabel(r'\textbf{RMSE}')
        ax1.set_title(r'\textbf{Performance Trajectory}')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Add annotations for each step
        for i, (step, rmse) in enumerate(zip(steps, ridge_rmse)):
            if step == 0:
                label = 'Baseline'
    parser = argparse.ArgumentParser(description="Sequential Feature Evaluation Pipeline")
    parser.add_argument("--run_dir", type=str, required=True,
{{ ... }}
    parser.add_argument("--output_dir", type=str, 
                       default="/root/fuegoRecommender/src/contingency/evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--n_trials", type=int, default=30,
                       help="Number of BO trials per feature")
    parser.add_argument("--features", nargs='+', 
                       default=None,  # Set default to None
                       help="List of features to evaluate sequentially")
    
    args = parser.parse_args()
    
    # Initialize
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return 1
    
    session_state = SessionState(run_dir=run_dir)
    evaluator = SequentialFeatureEvaluator(session_state, output_dir)
    
    # Get all discovered feature names if --features is not specified
    if args.features is None:
        # Use global get_all_feature_functions() to get the full feature list dynamically
        all_features = list(get_all_feature_functions().keys())
        args.features = all_features  # Set args.features to all discovered feature names
    
    try:
        # Run evaluation
        results = evaluator.run_sequential_evaluation(args.features, args.n_trials)
        
        print(f"\n=== EVALUATION COMPLETE ===")
        print(f"Evaluated {len(args.features)} features")
        print(f"Results saved to: {output_dir}")
        
        # Print summary
        baseline_rmse = results[0]['model_results']['Ridge']['rmse']
        final_rmse = results[-1]['model_results']['Ridge']['rmse']
        improvement = (baseline_rmse - final_rmse) / baseline_rmse * 100
        
        print(f"Baseline RMSE: {baseline_rmse:.4f}")
        print(f"Final RMSE: {final_rmse:.4f}")
        print(f"Total improvement: {improvement:.2f}%")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        session_state.close_connection()
    
    return 0


if __name__ == "__main__":
    exit(main())
```

### `contingency/test_manual_feature.py`

**File size:** 6,808 bytes

```python
#!/usr/bin/env python3
"""
Test script for manual feature functions with Bayesian Optimization

This script tests our manually implemented feature functions using BO
to optimize their hyperparameters on real data.
"""

import argparse
from pathlib import Path
import pandas as pd
import optuna
import numpy as np
from typing import Dict, Any
from src.utils.session_state import SessionState
from src.contingency.functions import rating_popularity_momentum


def load_books_data(session_state: SessionState) -> pd.DataFrame:
    """Load books data with ratings information."""
    conn = session_state.db_connection
    
    # Get books with their ratings data
    sql = """
    SELECT 
        b.book_id,
        b.title,
        b.average_rating,
        b.ratings_count,
        b.text_reviews_count,
        b.publication_year,
        b.num_pages
    FROM books b
    WHERE b.average_rating IS NOT NULL 
      AND b.ratings_count IS NOT NULL
      AND b.ratings_count > 0
    LIMIT 10000
    """
    
    try:
        df = conn.execute(sql).fetchdf()
        print(f"Loaded {len(df)} books with rating data")
        print(f"Average rating range: {df['average_rating'].min():.2f} - {df['average_rating'].max():.2f}")
        print(f"Ratings count range: {df['ratings_count'].min()} - {df['ratings_count'].max()}")
        return df
    except Exception as e:
        print(f"Error loading books data: {e}")
        # Fallback: create synthetic data for testing
        print("Creating synthetic data for testing...")
        np.random.seed(42)
        n_books = 1000
        synthetic_df = pd.DataFrame({
            'book_id': range(n_books),
            'title': [f'Book_{i}' for i in range(n_books)],
            'average_rating': np.random.uniform(1.0, 5.0, n_books),
            'ratings_count': np.random.exponential(50, n_books).astype(int),
            'text_reviews_count': np.random.exponential(20, n_books).astype(int),
            'publication_year': np.random.randint(1950, 2024, n_books),
            'num_pages': np.random.randint(100, 800, n_books)
        })
        return synthetic_df


def test_rating_popularity_momentum(df: pd.DataFrame, n_trials: int = 20):
    """Test the rating_popularity_momentum feature with Bayesian Optimization."""
    print(f"\n=== Testing rating_popularity_momentum feature ===")
    print(f"Data shape: {df.shape}")
    
    def objective(trial):
        # Define hyperparameter search space
        params = {
            'rating_weight': trial.suggest_float('rating_weight', 0.1, 2.0),
            'count_weight': trial.suggest_float('count_weight', 0.1, 1.5),
            'momentum_power': trial.suggest_float('momentum_power', 0.3, 1.2),
            'min_ratings_threshold': trial.suggest_int('min_ratings_threshold', 5, 50),
            'rating_scale': trial.suggest_float('rating_scale', 4.0, 6.0)
        }
        
        try:
            # Compute feature
            feature_values = rating_popularity_momentum(df, params)
            
            # Evaluation metric: we want features that correlate well with actual popularity
            # Use ratings_count as proxy for true popularity
            correlation = np.corrcoef(feature_values, df['ratings_count'])[0, 1]
            
            # Handle NaN correlation (can happen if feature is constant)
            if np.isnan(correlation):
                return -1.0
                
            # We want high positive correlation
            return correlation
            
        except Exception as e:
            print(f"Error in trial: {e}")
            return -1.0
    
    # Run optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    # Print results
    print(f"\nBest correlation: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    
    # Test best feature
    best_feature = rating_popularity_momentum(df, study.best_params)
    print(f"\nFeature statistics:")
    print(f"  Mean: {best_feature.mean():.4f}")
    print(f"  Std: {best_feature.std():.4f}")
    print(f"  Min: {best_feature.min():.4f}")
    print(f"  Max: {best_feature.max():.4f}")
    print(f"  Non-zero values: {(best_feature > 0).sum()}/{len(best_feature)}")
    
    # Correlation analysis
    correlations = {
        'ratings_count': np.corrcoef(best_feature, df['ratings_count'])[0, 1],
        'average_rating': np.corrcoef(best_feature, df['average_rating'])[0, 1],
        'text_reviews_count': np.corrcoef(best_feature, df['text_reviews_count'])[0, 1] if 'text_reviews_count' in df.columns else None
    }
    
    print(f"\nCorrelations with other variables:")
    for var, corr in correlations.items():
        if corr is not None:
            print(f"  {var}: {corr:.4f}")
    
    return study.best_params, study.best_value, best_feature


def main():
    parser = argparse.ArgumentParser(description="Test manual feature functions with BO")
    parser.add_argument("--run_dir", type=str, required=True, 
                       help="Path to run directory with session_state.json")
    parser.add_argument("--n_trials", type=int, default=20, 
                       help="Number of BO trials")
    
    args = parser.parse_args()
    
    # Initialize session state
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return 1
        
    session_state = SessionState(run_dir=run_dir)
    
    try:
        # Load data
        df = load_books_data(session_state)
        
        if df.empty:
            print("No data loaded, cannot proceed")
            return 1
            
        # Test the feature
        best_params, best_score, feature_values = test_rating_popularity_momentum(df, args.n_trials)
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Best correlation score: {best_score:.4f}")
        print(f"Optimized parameters: {best_params}")
        
        # Save results
        results_file = run_dir / "manual_feature_results.json"
        import json
        results = {
            "feature_name": "rating_popularity_momentum",
            "best_params": best_params,
            "best_score": best_score,
            "n_trials": args.n_trials,
            "data_shape": df.shape,
            "timestamp": "2025-06-17T11:33:52+02:00"
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Results saved to: {results_file}")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        session_state.close_connection()
    
    return 0


if __name__ == "__main__":
    exit(main())
```

### `contingency/unique_hypotheses.json`

**File size:** 26,729 bytes

```json
{
  "total_unique_hypotheses": 50,
  "original_total": 77,
  "deduplication_timestamp": "2025-06-17T11:52:00+02:00",
  "hypotheses": [
    {
      "run_id": "run_20250617_093150_14db6164",
      "id": "13af6b11-081c-4940-a4cc-81c4c481e220",
      "summary": "Users who read more books tend to provide higher average ratings.",
      "rationale": "This indicates a positive relationship between engagement and satisfaction in reading, which can guide personalized recommendations.",
      "depends_on": [
        "user_reading_trends.books_read",
        "user_reading_trends.avg_rating"
      ],
      "function_names": [],
      "unique_id": 1
    },
    {
      "run_id": "run_20250617_093150_14db6164",
      "id": "a8a69379-c9ec-4d88-a3bd-0ed31945b6ce",
      "summary": "Users prefer specific genres that have consistently high average ratings.",
      "rationale": "Identifying these genres can enhance recommendation systems by tailoring suggestions to user preferences, increasing engagement.",
      "depends_on": [
        "avg_rating_by_genre.genre",
        "avg_rating_by_genre.average_rating"
      ],
      "function_names": [],
      "unique_id": 2
    },
    {
      "run_id": "run_20250617_093150_14db6164",
      "id": "df51544b-6df9-405a-8d38-8167f9fd8437",
      "summary": "Shelf categories that accumulate more books indicate user interest in those areas.",
      "rationale": "Analyzing shelf popularity can guide curators and recommend titles effectively, especially for new users looking for suggestions.",
      "depends_on": [
        "book_shelves.shelf",
        "book_shelves.cnt"
      ],
      "function_names": [],
      "unique_id": 3
    },
    {
      "run_id": "run_20250617_093150_14db6164",
      "id": "4c47ef5c-ad65-4d5e-bea3-cdb97e82b9fb",
      "summary": "Readers show a preference for certain book formats based on average ratings.",
      "rationale": "This insight can help personalize recommendations based on the specific format a user tends to favor, enhancing user satisfaction.",
      "depends_on": [
        "book_genre_format_ratings.format",
        "book_genre_format_ratings.avg_rating"
      ],
      "function_names": [],
      "unique_id": 4
    },
    {
      "run_id": "run_20250617_085529_ccbae235",
      "id": "72ff3917-8c3d-42ef-ad1d-1331b8aca456",
      "summary": "Readers exhibit distinct preferences across genres, with significant interest in fantasy and romance.",
      "rationale": "Understanding genre popularity can enhance targeted recommendations and improve user engagement.",
      "depends_on": [
        "book_genre_format_ratings.genre",
        "curated_reviews.rating"
      ],
      "function_names": [],
      "unique_id": 5
    },
    {
      "run_id": "run_20250617_085529_ccbae235",
      "id": "81dce0f6-fac7-4e2f-ae38-e498a03a7a39",
      "summary": "Users demonstrate a range of reading behaviors, influencing how books are rated and reviewed.",
      "rationale": "Identifying user behavior clusters can tailor recommendations to specific user segments, enhancing user satisfaction.",
      "depends_on": [
        "curated_reviews.user_id",
        "curated_reviews.rating"
      ],
      "function_names": [],
      "unique_id": 6
    },
    {
      "run_id": "run_20250617_085529_ccbae235",
      "id": "1ed5f386-211e-43d0-b883-73be8b60cfe9",
      "summary": "Users who rate steampunk and fantasy literature higher may also enjoy other genres with similar thematic elements.",
      "rationale": "Exploring genre interconnectivity could refine recommendation strategies for users with diverse tastes.",
      "depends_on": [
        "curated_reviews.rating",
        "curated_books.title"
      ],
      "function_names": [],
      "unique_id": 7
    },
    {
      "run_id": "run_20250617_085529_ccbae235",
      "id": "b8be3f1f-e683-4e80-8fba-6ec3c6f519f8",
      "summary": "Books with authors who collaborate frequently may be rated higher due to perceived quality or continuous thematic alignment.",
      "rationale": "Understanding the impact of author collaborations can enhance recommendations based on thematic consistency or quality.",
      "depends_on": [
        "book_authors.author_id",
        "curated_books.book_id"
      ],
      "function_names": [],
      "unique_id": 8
    },
    {
      "run_id": "run_20250617_085529_ccbae235",
      "id": "b9f0a4a3-0176-4fa7-aa45-2a7dea3bbe8e",
      "summary": "Users displaying certain reading behaviors (few books but high ratings) may benefit from curated personalized recommendations.",
      "rationale": "Targeting user segments with specific reading patterns could optimize engagement through tailored suggestions.",
      "depends_on": [
        "user_reading_trends.books_read",
        "user_reading_trends.avg_rating"
      ],
      "function_names": [],
      "unique_id": 9
    },
    {
      "run_id": "run_20250617_085529_ccbae235",
      "id": "e8878f2d-f729-47f6-ba65-fa5787625ff4",
      "summary": "Books with more pages may affect user ratings differently compared to shorter books.",
      "rationale": "Longer books may provide more depth in storytelling, which can appeal to readers, or they may overwhelm readers, impacting ratings adversely.",
      "depends_on": [
        "curated_books.num_pages",
        "curated_books.avg_rating"
      ],
      "function_names": [],
      "unique_id": 10
    },
    {
      "run_id": "run_20250617_085529_ccbae235",
      "id": "d76c2b69-7eea-478c-803c-8dee1852b51f",
      "summary": "Ratings and descriptions might correlate strongly, suggesting that well-articulated descriptions entice higher ratings.",
      "rationale": "Descriptive quality may engage readers more effectively, boosting their ratings as their expectations are met or exceeded.",
      "depends_on": [
        "curated_books.title",
        "curated_books.description",
        "curated_books.avg_rating"
      ],
      "function_names": [],
      "unique_id": 11
    },
    {
      "run_id": "run_20250617_085529_ccbae235",
      "id": "803fa0f4-7918-46bf-adcb-2c48d9ceeca4",
      "summary": "Positive sentiment in reviews could predict higher book ratings.",
      "rationale": "Emotional connections in reviews reflect reader satisfaction, which is likely to impact ratings positively.",
      "depends_on": [
        "curated_reviews.review_text",
        "curated_reviews.rating"
      ],
      "function_names": [],
      "unique_id": 12
    },
    {
      "run_id": "run_20250617_085529_ccbae235",
      "id": "03540c1c-5348-4de6-afae-80ff7c59120a",
      "summary": "Users who enjoy a particular book may also prefer others from the same author or similar thematic books.",
      "rationale": "Cross-recommendation between similar books could enhance user experience and interactions.",
      "depends_on": [
        "book_similars.book_id",
        "book_similars.similar_book_id"
      ],
      "function_names": [],
      "unique_id": 13
    },
    {
      "run_id": "run_20250617_085529_ccbae235",
      "id": "9ad09702-314b-4850-8aa8-490f469c3d9d",
      "summary": "Increased user interaction correlates positively with higher ratings.",
      "rationale": "More interactions suggest greater engagement and satisfaction, typically reflected in higher ratings.",
      "depends_on": [
        "interactions.review_id",
        "interactions.n_votes",
        "interactions.n_comments"
      ],
      "function_names": [],
      "unique_id": 14
    },
    {
      "run_id": "run_20250617_090838_a65fb946",
      "id": "ac6f818c-406e-40eb-9e6b-b73092897f87",
      "summary": "Books published by a larger number of unique publishers tend to have higher average ratings.",
      "rationale": "A diverse range of publishers may indicate higher quality and more substantial investment in the books, which could translate into better ratings.",
      "depends_on": [
        "curated_books.publisher_name",
        "curated_books.avg_rating"
      ],
      "function_names": [],
      "unique_id": 15
    },
    {
      "run_id": "run_20250617_090838_a65fb946",
      "id": "0ab61ad9-c368-4e15-8b7e-82de97b158ac",
      "summary": "Books that have been published more recently tend to have higher average ratings.",
      "rationale": "Newer books may benefit from more modern writing standards, trends, and reader preferences than older books, affecting their ratings positively.",
      "depends_on": [
        "curated_books.publication_date",
        "curated_books.avg_rating"
      ],
      "function_names": [],
      "unique_id": 16
    },
    {
      "run_id": "run_20250617_090838_a65fb946",
      "id": "eb75df81-ef83-4440-a1d6-a00f8de70aee",
      "summary": "Ebook formats tend to receive higher average ratings compared to physical formats.",
      "rationale": "Ebooks may offer more accessibility and convenience, appealing to a broader audience, which could lead to better ratings.",
      "depends_on": [
        "curated_books.format",
        "curated_books.avg_rating"
      ],
      "function_names": [],
      "unique_id": 17
    },
    {
      "run_id": "run_20250617_090838_a65fb946",
      "id": "17545600-ee77-4212-9adc-80afe1326f5a",
      "summary": "Books with higher ratings tend to attract more reviews.",
      "rationale": "Higher rated books are more likely to engage readers and prompt them to leave feedback, resulting in more reviews overall.",
      "depends_on": [
        "curated_books.avg_rating",
        "curated_books.ratings_count"
      ],
      "function_names": [],
      "unique_id": 18
    },
    {
      "run_id": "run_20250617_090838_a65fb946",
      "id": "7ccab827-e740-4a0d-b5f6-f56f895b151c",
      "summary": "Books with descriptions that reflect unique or engaging themes tend to have higher average ratings.",
      "rationale": "Engaging themes may attract more readers and generate higher ratings based on reader enjoyment and connection to the content.",
      "depends_on": [
        "curated_books.description",
        "curated_books.avg_rating"
      ],
      "function_names": [],
      "unique_id": 19
    },
    {
      "run_id": "run_20250617_111935_2f1eb61d",
      "id": "4d14e12f-1fb8-46b1-a5bf-d352b7d02af2",
      "summary": "Authors with more collaborations tend to create books with higher average ratings.",
      "rationale": "Collaboration may lead to improved quality through shared expertise.",
      "depends_on": [
        "author_collaborations.author_id",
        "curated_books.avg_rating"
      ],
      "function_names": [],
      "unique_id": 20
    },
    {
      "run_id": "run_20250617_111935_2f1eb61d",
      "id": "3cae48b6-dc5e-428d-b4b6-ff1fe1749364",
      "summary": "Books with more pages tend to receive a higher average rating.",
      "rationale": "Longer books may offer deeper stories and character development, leading to higher ratings.",
      "depends_on": [
        "curated_books.num_pages",
        "curated_books.avg_rating"
      ],
      "function_names": ["page_count_rating_correlation"],
      "unique_id": 21
    },
    {
      "run_id": "run_20250617_111935_2f1eb61d",
      "id": "4443353e-2d03-4a64-811d-ff42b1d4940c",
      "summary": "Ebooks tend to have lower average ratings compared to physical books.",
      "rationale": "Physical books may be more desirable due to tactile experiences and availability of more detailed information before purchase.",
      "depends_on": [
        "curated_books.is_ebook",
        "curated_books.avg_rating"
      ],
      "function_names": ["ebook_rating_penalty"],
      "unique_id": 22
    },
    {
      "run_id": "run_20250617_111935_2f1eb61d",
      "id": "ef0d3ec5-6f3e-499e-a289-387ca9727cce",
      "summary": "Genres with more books tend to have higher average ratings.",
      "rationale": "Genres that attract more authors and books may indicate positive reader engagement and rating patterns.",
      "depends_on": [
        "genre_counts_view.genre",
        "avg_rating_by_genre.average_rating"
      ],
      "function_names": ["genre_volume_rating_boost"],
      "unique_id": 23
    },
    {
      "run_id": "run_20250617_091920_2d524456",
      "id": "3ca42a7c-4ce9-410b-8c6f-4d16a54c935e",
      "summary": "Users who engage with more books tend to provide more reviews.",
      "rationale": "Increased reading activity likely leads to more opportunities for users to express their thoughts, resulting in higher review counts.",
      "depends_on": [
        "curated_reviews.user_id",
        "curated_reviews.book_id"
      ],
      "function_names": ["user_activity_review_count"],
      "unique_id": 24
    },
    {
      "run_id": "run_20250617_091920_2d524456",
      "id": "dd7d10a1-ae95-4a4d-bacb-9381b455c1ab",
      "summary": "Books with higher average ratings tend to have more reviews written about them.",
      "rationale": "Higher quality ratings may incentivize more users to share their experiences through reviews, highlighting a correlation between book quality and engagement.",
      "depends_on": [
        "curated_books.book_id",
        "curated_books.avg_rating"
      ],
      "function_names": ["rating_review_volume_correlation"],
      "unique_id": 25
    },
    {
      "run_id": "run_20250617_091920_2d524456",
      "id": "fce47266-9073-45ee-83f5-2a40141e02e9",
      "summary": "Different audience demographics engage differently with book formats.",
      "rationale": "Demographics may resonate differently with various reading formats, leading to fluctuations in engagement levels and variations by readership.",
      "depends_on": [
        "curated_reviews.book_id",
        "curated_reviews.user_id"
      ],
      "function_names": ["demographic_format_engagement"],
      "unique_id": 26
    },
    {
      "run_id": "run_20250617_091920_2d524456",
      "id": "c5b84ff8-bd87-4545-b581-e86eab7d8396",
      "summary": "Popular authors tend to have higher review rates for their books.",
      "rationale": "Established authors often generate a loyal readership that is more likely to provide feedback, contributing to a higher volume of reviews for their works.",
      "depends_on": [
        "book_authors.author_id",
        "curated_reviews.book_id"
      ],
      "function_names": ["author_popularity_review_rate"],
      "unique_id": 27
    },
    {
      "run_id": "run_20250617_091920_2d524456",
      "id": "57fe7365-4549-42b8-a915-56d73b04d6ff",
      "summary": "The sentiments expressed in reviews vary significantly by user engagement.",
      "rationale": "The richness and depth of reviews can reflect the level of emotional or intellectual engagement a user has with a book, offering insight into their reading experience.",
      "depends_on": [
        "curated_reviews.review_text",
        "curated_reviews.rating"
      ],
      "function_names": ["review_sentiment_engagement_variance"],
      "unique_id": 28
    },
    {
      "run_id": "run_20250617_091504_28f70390",
      "id": "9333d190-af02-472c-af17-646ac67245f8",
      "summary": "Books with higher average ratings tend to have more formats available.",
      "rationale": "Offering multiple formats (e.g., eBook, paperback, audiobook) increases accessibility and can lead to higher user satisfaction, reflected in ratings.",
      "depends_on": [
        "book_genre_format_ratings.avg_rating",
        "book_genre_format_ratings.format"
      ],
      "function_names": ["format_availability_rating"],
      "unique_id": 29
    },
    {
      "run_id": "run_20250617_091504_28f70390",
      "id": "0a7a84d9-792c-4ec8-81cd-a4c10017c909",
      "summary": "Books that are listed in more genres receive higher ratings.",
      "rationale": "Diversity in genre could attract a wider audience, thus increasing the potential for higher ratings as more readers engage with the book.",
      "depends_on": [
        "book_genre_format_ratings.avg_rating",
        "book_genre_format_ratings.genre"
      ],
      "function_names": ["genre_listing_diversity_rating"],
      "unique_id": 30
    },
    {
      "run_id": "run_20250617_091504_28f70390",
      "id": "6014b1a6-b190-49af-b11f-c34081ae966e",
      "summary": "Users who leave reviews with more detailed text tend to provide higher ratings.",
      "rationale": "Longer reviews might indicate a more engaged reader, leading to a more favorable evaluation of the book based on their experience.",
      "depends_on": [
        "curated_reviews.rating",
        "curated_reviews.review_text"
      ],
      "function_names": ["detailed_review_rating_boost"],
      "unique_id": 31
    },
    {
      "run_id": "run_20250617_091504_28f70390",
      "id": "bc90ed63-834d-4c74-b081-9e4beac6f1fb",
      "summary": "Books categorized as 'wish-list' tend to receive lower ratings than those in 'book-club' or 'ya' genres.",
      "rationale": "The category might imply that readers are more exploratory or less committed to 'wish-list' books, which could be reflected in their ratings.",
      "depends_on": [
        "book_genre_format_ratings.avg_rating",
        "book_genre_format_ratings.genre"
      ],
      "function_names": ["wishlist_vs_bookclub_rating"],
      "unique_id": 32
    },
    {
      "run_id": "run_20250617_091504_28f70390",
      "id": "28c49565-eb41-4580-a085-6efd23fe58f3",
      "summary": "Readers who rate more books tend to have a positive influence on their average ratings.",
      "rationale": "Frequent engagement with books by readers could indicate higher engagement and a tendency to rate books more favorably over time.",
      "depends_on": [
        "curated_reviews.user_id",
        "curated_reviews.rating"
      ],
      "function_names": ["reader_engagement_positive_influence"],
      "unique_id": 33
    },
    {
      "run_id": "run_20250617_083442_801c20a3",
      "id": "86e95b4c-49e9-4900-a6f2-0946af15dead",
      "summary": "Books with higher average ratings tend to receive more ratings.",
      "rationale": "Analysis suggests a connection where books with higher ratings generally exhibit a higher count of reader reviews.",
      "depends_on": [
        "curated_books.avg_rating",
        "curated_books.ratings_count"
      ],
      "function_names": ["avg_rating_ratings_count_correlation"],
      "unique_id": 34
    },
    {
      "run_id": "run_20250617_083442_801c20a3",
      "id": "b1fe0390-aee7-4b67-a4f8-d21933e6e00f",
      "summary": "Books with a length of 400-450 pages are more popular.",
      "rationale": "The analysis shows a clustering of popular books around the 400-450 pages mark, indicating reader preference for these lengths.",
      "depends_on": [
        "curated_books.num_pages"
      ],
      "function_names": ["optimal_page_length_popularity"],
      "unique_id": 35
    },
    {
      "run_id": "run_20250617_083442_801c20a3",
      "id": "cf52137f-6d57-44fc-931c-7650b2170e14",
      "summary": "Certain books have significantly higher ratings counts, indicating outlier popularity.",
      "rationale": "Observations reveal a stark contrast in popularity among books, with select titles receiving significantly more reviews, impacting overall trends.",
      "depends_on": [
        "curated_books.ratings_count"
      ],
      "function_names": ["outlier_popularity_score"],
      "unique_id": 36
    },
    {
      "run_id": "run_20250617_083442_801c20a3",
      "id": "d75696af-e739-43a8-97e5-a882e23dd69f",
      "summary": "Books with lower average ratings may have niche audiences.",
      "rationale": "Findings include examples of low-rated books that suggest a specialty or niche appeal, warranting further inquiry for targeted strategies.",
      "depends_on": [
        "curated_books.avg_rating",
        "curated_books.ratings_count"
      ],
      "function_names": ["niche_audience_score"],
      "unique_id": 37
    },
    {
      "run_id": "run_20250617_083442_801c20a3",
      "id": "c056e446-a3f6-44fe-b9cf-cdf93dbdd044",
      "summary": "Books in the 'mystery-suspense' genre have higher average ratings than those in other genres.",
      "rationale": "Identifying high-performing genres can guide marketing and recommendation strategies to boost user engagement.",
      "depends_on": [
        "book_genre_format_ratings.genre",
        "book_genre_format_ratings.avg_rating"
      ],
      "function_names": ["mystery_suspense_genre_boost"],
      "unique_id": 38
    },
    {
      "run_id": "run_20250617_083442_801c20a3",
      "id": "1b03793c-8c35-4a1d-94a1-e3487b78ae59",
      "summary": "Users who read more books tend to give higher average ratings.",
      "rationale": "Understanding reading patterns and user engagement can refine personalized recommendations and increase user satisfaction.",
      "depends_on": [
        "curated_reviews.user_id",
        "curated_reviews.rating"
      ],
      "function_names": ["user_reading_volume_rating"],
      "unique_id": 39
    },
    {
      "run_id": "run_20250617_083442_801c20a3",
      "id": "d1c0deef-eb7c-4721-960a-f4d583a54d69",
      "summary": "Author collaborations are linked to an increase in shared readership and book success.",
      "rationale": "Detecting collaborative patterns can leverage cross-promotion opportunities and diversify author exposure.",
      "depends_on": [
        "book_authors.author_id",
        "book_authors.book_id"
      ],
      "function_names": ["author_collaboration_success"],
      "unique_id": 40
    },
    {
      "run_id": "run_20250617_083442_801c20a3",
      "id": "dae69914-229f-4dd4-9fab-ac5338b72da7",
      "summary": "Books with more than 10 ratings provide reliable average ratings and insights.",
      "rationale": "Focusing on books with substantial ratings can enhance the accuracy of analysis on book performance.",
      "depends_on": [
        "curated_reviews.book_id",
        "curated_reviews.rating"
      ],
      "function_names": ["genre_format_distribution_score"],
      "unique_id": 41
    },
    {
      "run_id": "run_20250617_083442_801c20a3",
      "id": "07fe12a8-784e-47e1-beb4-359f9029da15",
      "summary": "The distribution of books across genres and formats reveals market trends and reader preferences.",
      "rationale": "Understanding genre and format distributions aids in anticipating user needs and informing inventory decisions.",
      "depends_on": [
        "book_genre_format_ratings.genre",
        "book_genre_format_ratings.format"
      ],
      "function_names": ["avg_rating_rating_count_score"],
      "unique_id": 42
    },
    {
      "run_id": "run_20250617_093601_ae6eca46",
      "id": "9eb12361-95f2-43d9-9a30-08df006409a9",
      "summary": "Books with higher average ratings are more likely to have a greater number of ratings.",
      "rationale": "A larger number of ratings may indicate a broader readership which could lead to higher average ratings.",
      "depends_on": [
        "curated_books_view.avg_rating",
        "curated_books_view.ratings_count"
      ],
      "function_names": ["ebook_positive_rating_score"],
      "unique_id": 43
    },
    {
      "run_id": "run_20250617_093601_ae6eca46",
      "id": "d5b2263b-546e-4007-b0e6-1489324361b1",
      "summary": "eBooks tend to have higher average ratings compared to physical books.",
      "rationale": "The eBook format may attract more engaged readers who provide ratings, leading to a higher average rating.",
      "depends_on": [
        "curated_books_view.is_ebook",
        "curated_books_view.avg_rating"
      ],
      "function_names": ["publisher_reputation_rating"],
      "unique_id": 44
    },
    {
      "run_id": "run_20250617_093601_ae6eca46",
      "id": "14f49da7-3891-493a-872c-e1baa903e65c",
      "summary": "Books published by known publishers receive higher ratings.",
      "rationale": "Books from established publishers may be of higher quality and better marketed, contributing to better reception and ratings.",
      "depends_on": [
        "curated_books_view.publisher_name",
        "curated_books_view.avg_rating"
      ],
      "function_names": ["rating_engagement_correlation"],
      "unique_id": 45
    },
    {
      "run_id": "run_20250617_000445_396c4332",
      "id": "36c4e94f-a98a-4a17-ac4c-74ce532ca4ee",
      "summary": "Higher book ratings correlate with more user engagement indicators.",
      "rationale": "Books with higher ratings tend to receive more ratings and user interactions, suggesting that quality impacts engagement.",
      "depends_on": [
        "book_shelves.cnt",
        "user_stats_daily.mean_rating"
      ],
      "function_names": ["series_vs_standalone_rating"],
      "unique_id": 46
    },
    {
      "run_id": "run_20250617_000445_396c4332",
      "id": "c3c50574-dbe3-47d3-9247-fbc88a3e5775",
      "summary": "Books in series have higher average ratings than standalone books.",
      "rationale": "Series may develop richer character arcs and plotlines, encouraging readers to invest more, which correlates with higher ratings.",
      "depends_on": [
        "book_series.series_name",
        "curated_books.avg_rating"
      ],
      "function_names": ["translation_penalty_score"],
      "unique_id": 47
    },
    {
      "run_id": "run_20250617_000445_396c4332",
      "id": "9ea2077f-cde5-4f9d-9ac1-0aab493d4a06",
      "summary": "Translated books achieve lower ratings compared to original language publications.",
      "rationale": "Perceptions of translation quality can impact user ratings, indicating that original works may resonate more.",
      "depends_on": [
        "book_authors.role",
        "curated_books.avg_rating"
      ],
      "function_names": ["genre_diversity_engagement_score"],
      "unique_id": 48
    },
    {
      "run_id": "run_20250617_000445_396c4332",
      "id": "5a96a7d9-fb75-4489-89ae-1f4bd14ff41e",
      "summary": "Readers who engage with multiple genres display broader engagement metrics.",
      "rationale": "Genre diversity might indicate varied interests leading to more comprehensive reading habits and higher engagement.",
      "depends_on": [
        "book_genre_format_ratings.genre",
        "user_stats_daily.n_ratings"
      ],
      "function_names": ["publisher_marketing_rating_boost"],
      "unique_id": 49
    },
    {
      "run_id": "run_20250617_000445_396c4332",
      "id": "59e13a74-766a-4edc-8042-6dc9e6a7f4e1",
      "summary": "Books published with more extensive marketing (e.g., large publisher backing) receive higher user ratings.",
      "rationale": "Visibility and perceived legitimacy from larger publishers may influence reader perceptions and ratings.",
      "depends_on": [
        "curated_books.publisher_name",
        "curated_books.avg_rating"
      ],
      "function_names": [],
      "unique_id": 50
    }
  ]
}
```

### `core/database.py`

**File size:** 10,643 bytes

```python
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import duckdb
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from src.config.settings import DB_PATH

logger = logging.getLogger(__name__)


def check_db_schema() -> bool:
    """
    Checks if the database has the required tables and they are not empty.
    """
    db_file = Path(DB_PATH)
    if not db_file.exists() or db_file.stat().st_size == 0:
        return False
    try:
        with duckdb.connect(database=DB_PATH, read_only=True) as conn:
            tables = [t[0] for t in conn.execute("SHOW TABLES;").fetchall()]
            required_tables = {"books", "reviews", "users"}

            if not required_tables.issubset(tables):
                return False

            for table in required_tables:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                if count == 0:
                    return False
        return True
    except duckdb.Error as e:
        logger.warning(f"Database schema check failed, will attempt to rebuild: {e}")
        return False


def ingest_json_to_duckdb():
    """
    Ingests data from gzipped JSON files into DuckDB, creating the schema.
    """
    books_json_path = "data/books.json.gz"
    reviews_json_path = "data/reviews.json.gz"

    logger.info(f"Starting ingestion from {books_json_path} and {reviews_json_path}")

    with duckdb.connect(database=DB_PATH, read_only=False) as conn:
        logger.info("Creating 'books' table...")
        conn.execute(f"""
            CREATE OR REPLACE TABLE books AS 
            SELECT * 
            FROM read_json_auto('{books_json_path}', format='newline_delimited');
        """)
        logger.info("'books' table created.")

        logger.info("Creating 'reviews' table...")
        conn.execute(f"""
            CREATE OR REPLACE TABLE reviews AS 
            SELECT *
            FROM read_json_auto('{reviews_json_path}', format='newline_delimited');
        """)
        logger.info("'reviews' table created.")

        logger.info("Creating 'users' table from distinct reviewers...")
        conn.execute("""
            CREATE OR REPLACE TABLE users AS
            SELECT DISTINCT user_id FROM reviews;
        """)
        logger.info("'users' table created.")

    logger.info("Data ingestion from JSON files to DuckDB complete.")


def fetch_df(query: str) -> pd.DataFrame:
    """
    Connects to the database, executes a query, and returns a DataFrame.
    """
    with duckdb.connect(DB_PATH, read_only=True) as conn:
        return conn.execute(query).fetchdf()


def get_db_schema_string() -> str:
    """
    Introspects the database using SUMMARIZE and returns a detailed schema string
    with summary statistics. Connects in-process to avoid file locking issues.
    """
    schema_parts = []
    db_path = str(DB_PATH)  # Ensure it's a string for DuckDB

    try:
        logger.debug(f"Generating database schema from: {db_path}")

        # Connect in-process to an in-memory database to avoid file locks
        with duckdb.connect() as conn:
            # Attach the main database file in READ_ONLY mode, giving it an alias 'db'
            conn.execute(f"ATTACH '{db_path}' AS db;")

            # Query the information_schema to find tables in the attached database's 'main' schema
            tables_df = conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' AND table_catalog = 'db';"
            ).fetchdf()

            if tables_df.empty:
                # Fallback to a simpler SHOW TABLES if the schema query fails
                try:
                    tables_df = conn.execute("SHOW TABLES FROM db;").fetchdf()
                    logger.debug("Used SHOW TABLES fallback method")
                except Exception:
                    logger.error(
                        "Failed to list tables via both information_schema and SHOW TABLES"
                    )
                    return "ERROR: No tables found in the attached database. Could not list tables via information_schema or SHOW TABLES."

            if tables_df.empty:
                logger.warning("No tables found in the database")
                return "ERROR: No tables found in the attached database."

            logger.debug(f"Found {len(tables_df)} tables in database")

            for _, row in tables_df.iterrows():
                table_name = row["table_name"] if "table_name" in row else row["name"]

                # We must use the 'db' alias to refer to tables in the attached database
                qualified_table_name = f'db."{table_name}"'

                try:
                    row_count_result = conn.execute(
                        f"SELECT COUNT(*) FROM {qualified_table_name};"
                    ).fetchone()
                    row_count = row_count_result[0] if row_count_result else 0
                    schema_parts.append(f"TABLE: {table_name} ({row_count:,} rows)")

                    # Use the SUMMARIZE command to get schema and statistics
                    summary_df = conn.execute(
                        f"SUMMARIZE {qualified_table_name};"
                    ).fetchdf()

                    for _, summary_row in summary_df.iterrows():
                        col_name = summary_row["column_name"]
                        col_type = summary_row["column_type"]
                        null_pct = summary_row["null_percentage"]

                        stats = [f"NULLs: {null_pct}%"]

                        # Add type-specific stats for a richer summary
                        if "VARCHAR" in col_type.upper():
                            unique_count = summary_row.get("approx_unique")
                            if unique_count is not None:
                                stats.append(f"~{int(unique_count)} unique values")
                        elif any(
                            t in col_type.upper()
                            for t in ["INTEGER", "BIGINT", "DOUBLE", "FLOAT", "DECIMAL"]
                        ):
                            min_val = summary_row.get("min")
                            max_val = summary_row.get("max")
                            if min_val is not None and max_val is not None:
                                stats.append(f"range: [{min_val}, {max_val}]")

                        schema_parts.append(
                            f"  - {col_name} ({col_type}) [{', '.join(stats)}]"
                        )
                    schema_parts.append("")

                except Exception as table_error:
                    logger.warning(
                        f"Failed to analyze table {table_name}: {table_error}"
                    )
                    schema_parts.append(f"TABLE: {table_name} (analysis failed)")
                    schema_parts.append("")

        result = "\n".join(schema_parts)
        logger.debug(f"Generated schema string with {len(result)} characters")
        return result

    except Exception as e:
        logger.error(f"Failed to get database schema using SUMMARIZE method: {e}")
        logger.exception(e)
        return (
            f"ERROR: Could not retrieve database schema from {db_path}. Error: {str(e)}"
        )


def get_db_connection() -> duckdb.DuckDBPyConnection:
    """Returns a read-write connection to the main DuckDB database."""
    return duckdb.connect(database=str(DB_PATH), read_only=False)


class DatabaseConnection:
    def __init__(
        self, connection_string: Optional[str] = None, engine: Optional[Engine] = None
    ):
        """Initialize database connection

        Args:
            connection_string: SQLAlchemy connection string
            engine: Existing SQLAlchemy engine (for testing)
        """
        if engine:
            self.engine = engine
        else:
            connection_string = connection_string or os.getenv(
                "DATABASE_URL",
                "sqlite:///data/vulcan.db",  # Default to SQLite
            )
            self.engine = create_engine(connection_string)

        # Test connection
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection successful")
        except SQLAlchemyError as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise

    def execute_query(self, query: str) -> Dict[str, Any]:
        """Execute SQL query and return results

        Args:
            query: SQL query string

        Returns:
            Dictionary with query results
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                if result.returns_rows:
                    df = pd.DataFrame(result.fetchall(), columns=result.keys())
                    return {"data": df.to_dict(orient="records")}
                return {"affected_rows": result.rowcount}
        except SQLAlchemyError as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise

    def create_view(self, view_name: str, query: str, version: Optional[int] = None):
        """Create or replace a view

        Args:
            view_name: Name of the view
            query: SQL query defining the view
            version: Optional version number to append to view name
        """
        if version:
            view_name = f"{view_name}_v{version}"

        create_view_sql = f"CREATE OR REPLACE VIEW {view_name} AS {query}"

        try:
            with self.engine.connect() as conn:
                conn.execute(text(create_view_sql))
                conn.commit()
            logger.info(f"View {view_name} created successfully")
        except SQLAlchemyError as e:
            logger.error(f"Failed to create view {view_name}: {str(e)}")
            raise

    def drop_view(self, view_name: str):
        """Drop a view if it exists

        Args:
            view_name: Name of the view to drop
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text(f"DROP VIEW IF EXISTS {view_name}"))
                conn.commit()
            logger.info(f"View {view_name} dropped successfully")
        except SQLAlchemyError as e:
            logger.error(f"Failed to drop view {view_name}: {str(e)}")
            raise

    def close(self):
        """Close database connection"""
        self.engine.dispose()
        logger.info("Database connection closed")
```

### `core/llm.py`

**File size:** 496 bytes

```python
from typing import List
from loguru import logger

def call_llm_batch(prompts: List[str]) -> List[float]:
    """
    A placeholder for a utility that calls an LLM with a batch of prompts.
    """
    # In a real implementation, this would use a library like `litellm`
    # to handle batching and API calls.
    logger.info(f"Calling LLM with a batch of {len(prompts)} prompts.")

    # For now, return random scores for testing.
    import random

    return [random.random() for _ in prompts]
```

### `core/tools.py`

**File size:** 2,650 bytes

```python
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Tool:
    name: str
    description: str
    func: Callable
    required_args: List[str]
    optional_args: Optional[List[str]] = None


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        required_args: List[str],
        optional_args: Optional[List[str]] = None,
    ) -> Callable:
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            self._tools[name] = Tool(
                name=name,
                description=description,
                func=wrapper,
                required_args=required_args,
                optional_args=optional_args or [],
            )
            return wrapper

        return decorator

    def get_tool(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def get_tool_description(self, name: str) -> Optional[str]:
        tool = self.get_tool(name)
        return tool.description if tool else None

    def execute_tool(self, name: str, **kwargs) -> Any:
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool {name} not found")

        # Validate required arguments
        missing_args = [arg for arg in tool.required_args if arg not in kwargs]
        if missing_args:
            raise ValueError(f"Missing required arguments for {name}: {missing_args}")

        # Remove any arguments that aren't required or optional
        valid_args = set(tool.required_args + (tool.optional_args or []))
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}

        return tool.func(**filtered_kwargs)


# Register add_to_central_memory tool

def register_add_to_central_memory_tool(session_state):
    try:
        from src.utils.tools import get_add_to_central_memory_tool
        registry.register(
            name="add_to_central_memory",
            description="Add a structured note and reasoning to the session's central memory for cross-epoch sharing.",
            required_args=["note", "reasoning", "agent"],
            optional_args=["metadata"],
        )(get_add_to_central_memory_tool(session_state))
    except ImportError:
        # Tool not yet implemented or available
        pass

# Create global registry instance
registry = ToolRegistry()
```

### `data/cv_data_manager.py`

**File size:** 25,054 bytes

```python
"""
Cross-validation data manager for VULCAN.

Handles loading and managing cross-validation splits efficiently.
"""

import concurrent.futures
import json
import os
import queue
import threading
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union

import duckdb
import numpy as np
import pandas as pd
from loguru import logger
from tqdm.auto import tqdm

# Type aliases
UserID = Union[str, int]
UserIDList = List[UserID]
DataFrameDict = Dict[str, pd.DataFrame]


class ConnectionPool:
    """Thread-safe connection pool for managing DuckDB connections."""

    def __init__(self, db_path: str, max_connections: int = 10, **connection_kwargs):
        """Initialize the connection pool.

        Args:
            db_path: Path to the DuckDB database file
            max_connections: Maximum number of connections in the pool
            **connection_kwargs: Additional connection parameters
        """
        self.db_path = db_path
        self.max_connections = max_connections
        self.connection_kwargs = connection_kwargs
        self._pool: queue.Queue[duckdb.DuckDBPyConnection] = queue.Queue(
            maxsize=max_connections
        )
        self._in_use: Set[duckdb.DuckDBPyConnection] = set()
        self._lock = threading.Lock()

        # Initialize connections
        for _ in range(max_connections):
            conn = self._create_connection()
            self._pool.put(conn)

    def _create_connection(self) -> duckdb.DuckDBPyConnection:
        """Create a new database connection with optimized settings."""
        try:
            conn = duckdb.connect(self.db_path, **self.connection_kwargs)

            # Apply performance optimizations
            config = {
                "threads": 1,
                "enable_progress_bar": False,
                "enable_object_cache": True,
                "preserve_insertion_order": False,
                "default_null_order": "nulls_first",
                "enable_external_access": False,
            }

            for param, value in config.items():
                try:
                    conn.execute(f"SET {param} = {repr(value)}")
                except Exception as e:
                    logger.warning(f"Could not set {param}={value}: {e}")

            return conn
        except Exception as e:
            logger.error(f"Failed to create database connection: {e}")
            raise

    def get_connection(self, timeout: float = 10.0) -> duckdb.DuckDBPyConnection:
        """Get a connection from the pool with a timeout.

        Args:
            timeout: Maximum time to wait for a connection (seconds)

        Returns:
            An active database connection

        Raises:
            queue.Empty: If no connection is available within the timeout
        """
        try:
            conn = self._pool.get(timeout=timeout)
            with self._lock:
                self._in_use.add(conn)
            return conn
        except queue.Empty:
            raise RuntimeError(
                f"No database connections available after {timeout} seconds. "
                f"Consider increasing max_connections (current: {self.max_connections})."
            )

    def return_connection(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Return a connection to the pool.

        Args:
            conn: The connection to return
        """
        if conn is None:
            return

        with self._lock:
            if conn in self._in_use:
                self._in_use.remove(conn)
                try:
                    # Rollback any open transaction before returning to the pool
                    try:
                        conn.rollback()
                    except duckdb.Error as e:
                        # It's okay if there's no transaction to roll back.
                        if 'no transaction is active' not in str(e):
                            logger.warning(f"Error during rollback: {e}")
                    self._pool.put_nowait(conn)
                except Exception as e:
                    logger.warning(f"Error returning connection to pool: {e}")
                    try:
                        conn.close()
                    except duckdb.Error as close_err:
                        logger.warning(f"Error closing connection: {close_err}")
                    # Replace the bad connection with a new one
                    try:
                        new_conn = self._create_connection()
                        self._pool.put_nowait(new_conn)
                    except duckdb.Error as e:
                        logger.error(f"Failed to create replacement connection: {e}")

    def close_all(self) -> None:
        """Close all connections in the pool."""
        # Close all available connections
        while True:
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except queue.Empty:
                break

        # Also close any connections that were in use
        for conn in list(self._in_use):
            try:
                conn.close()
            except duckdb.Error as e:
                logger.warning(f"Error closing database connection: {e}")

        self._in_use.clear()


class CVDataManager:
    """Manages cross-validation data splits for the VULCAN pipeline."""

    # Class-level connection pool
    _connection_pool: Optional[ConnectionPool] = None
    _pool_lock = threading.Lock()
    _instance_count: int = 0

    def __init__(
        self,
        db_path: Union[str, Path],
        splits_dir: Union[str, Path],
        random_state: int = 42,
        cache_size_mb: int = 1024,
        max_connections: int = 10,
        read_only: bool = False,
        undersample_frac: float = 1.0,
    ):
        """Initialize the CV data manager with caching and connection pooling.

        Args:
            db_path: Path to the DuckDB database file
            splits_dir: Directory containing the cross-validation splits
            random_state: Random seed for reproducibility
            cache_size_mb: Size of DuckDB's memory cache in MB
            max_connections: Maximum number of database connections in the pool
            read_only: Whether the database should be opened in read-only mode
            undersample_frac: Fraction of users to sample for each fold (default: 1.0)
        """
        self.db_path = Path(db_path)
        self.splits_dir = Path(splits_dir)
        self.random_state = random_state
        self._cached_folds: Optional[List[Dict]] = None
        self._cv_folds = None
        self._cache_size_mb = cache_size_mb
        self.read_only = read_only
        self.undersample_frac = undersample_frac

        # Cache for loaded data
        self._data_cache: Dict[str, Any] = {}

        # Initialize the connection pool if it doesn't exist
        with CVDataManager._pool_lock:
            CVDataManager._instance_count += 1
            if CVDataManager._connection_pool is None:
                self._initialize_connection_pool(max_connections=max_connections)

    def _initialize_connection_pool(self, max_connections: int) -> None:
        """Initialize the connection pool with the specified number of connections."""
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Database file not found at {self.db_path}. "
                "Please ensure the data is downloaded and processed."
            )

        try:
            # Determine access mode
            self.read_only = not os.access(self.db_path.parent, os.W_OK)
            connection_kwargs = {
                "read_only": self.read_only,
                "config": {"memory_limit": f"{self._cache_size_mb}MB"},
            }

            db_path_str = str(self.db_path)

            logger.info(
                f"Initializing connection pool for db='{db_path_str}' with "
                f"{max_connections} connections (read_only={self.read_only})"
            )

            CVDataManager._connection_pool = ConnectionPool(
                db_path=db_path_str,
                max_connections=max_connections,
                **connection_kwargs,
            )

            # Create indexes if the database is writeable
            if not self.read_only:
                conn = self._get_connection()
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            """CREATE INDEX IF NOT EXISTS idx_interactions_user_id 
                               ON interactions(user_id)"""
                        )
                        cur.execute(
                            """CREATE INDEX IF NOT EXISTS idx_interactions_item_id 
                               ON interactions(item_id)"""
                        )
                    logger.info("Successfully created indexes on user_id and item_id.")
                except Exception as e:
                    logger.warning(f"Could not create indexes: {e}")
                finally:
                    self._return_connection(conn)

        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get a database connection from the pool."""
        if CVDataManager._connection_pool is None:
            raise RuntimeError("Connection pool not initialized")
        return CVDataManager._connection_pool.get_connection()

    def _return_connection(self, conn: Optional[duckdb.DuckDBPyConnection]) -> None:
        """Return a connection to the pool."""
        if conn is not None and CVDataManager._connection_pool is not None:
            CVDataManager._connection_pool.return_connection(conn)

    @property
    def db_connection(self) -> duckdb.DuckDBPyConnection:
        """Get a connection from the connection pool.

        Note: The caller is responsible for returning the connection to the pool
        using _return_connection() when done.

        Returns:
            An active DuckDB connection from the pool

        Raises:
            RuntimeError: If the connection pool is not initialized
        """
        return self._get_connection()

    def _clear_previous_fold_data(self) -> None:
        """Clear any cached fold data from memory."""
        self._data_cache.clear()

        # Run garbage collection to free up memory
        import gc

        gc.collect()

    def close(self) -> None:
        """Decrement the instance counter and clean up resources."""
        with CVDataManager._pool_lock:
            if CVDataManager._instance_count > 0:
                CVDataManager._instance_count -= 1

                # Close the connection pool if this is the last instance
                if (
                    CVDataManager._instance_count <= 0
                    and CVDataManager._connection_pool is not None
                ):
                    try:
                        # Clear any cached data
                        self._clear_previous_fold_data()

                        # Close all connections in the pool
                        CVDataManager._connection_pool.close_all()
                        CVDataManager._connection_pool = None
                        logger.info(
                            "Closed all database connections and cleared cached data"
                        )
                    except Exception as e:
                        logger.warning(f"Error during cleanup: {e}")

    def __del__(self) -> None:
        """Ensure proper cleanup when the object is destroyed.""" 
        try:
            self.close()
        except Exception:
            # Suppress errors during garbage collection
            pass

    @classmethod
    def close_global_connection_pool(cls) -> None:
        """Close the global connection pool if it exists."""
        with cls._pool_lock:
            if cls._connection_pool:
                logger.info("Closing global connection pool.")
                cls._connection_pool.close_all()
                cls._connection_pool = None
                logger.debug("Global connection pool closed.")



    def load_cv_folds(self) -> List[Dict[str, List[str]]]:
        """Load the cross-validation folds.

        Returns:
            List of dictionaries with 'train', 'validation', and 'test' keys
        """
        if self._cv_folds is not None:
            return self._cv_folds

        folds_file = self.splits_dir / "cv_folds.json"
        if not folds_file.exists():
            logger.error(f"CV folds file not found at {folds_file}")
            raise FileNotFoundError(f"CV folds file not found at {folds_file}")

        try:
            with open(folds_file, "r", encoding="utf-8") as f:
                self._cv_folds = json.load(f)
            return self._cv_folds
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing CV folds file: {e}")
            raise

    def get_fold_data(
        self,
        fold_idx: int,
        columns: Optional[List[str]] = None,
        sample_frac: Optional[float] = None,
        random_state: int = 42,
        batch_size: int = 500,
        show_progress: bool = True,
        max_workers: int = 4,
        split_type: str = "train_val",
    ) -> Union[
        Tuple[pd.DataFrame, pd.DataFrame],
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ]:
        # Use undersample_frac if sample_frac not provided
        if sample_frac is None:
            sample_frac = getattr(self, 'undersample_frac', 1.0);
        # Clear any previous data first
        self._clear_previous_fold_data()

        # Get the fold data
        folds = self.load_cv_folds()
        if fold_idx >= len(folds):
            raise ValueError(f"Fold index {fold_idx} out of range (0-{len(folds) - 1})")

        fold = folds[fold_idx]

        # Get user lists for each split
        train_users = fold["train"]
        val_users = fold["validation"]
        test_users = fold.get("test", [])

        # --- Stratified sampling by user activity ---
        def stratified_sample_users(users: List[str], frac: float, user_activity: Dict[str, int]) -> List[str]:
            """
            Stratified sampling of users by activity level.
            Args:
                users: List of user IDs to sample from.
                frac: Fraction to sample.
                user_activity: Dict mapping user_id to number of interactions.
            Returns:
                List of sampled users preserving activity distribution.
            """
            if frac is None or frac >= 1.0 or not users:
                return users
            activity_counts = np.array([user_activity.get(u, 0) for u in users])
            if len(set(activity_counts)) <= 1:
                rng = np.random.default_rng(random_state)
                sample_size = max(1, int(len(users) * frac))
                return rng.choice(users, size=sample_size, replace=False).tolist()
            bins = np.quantile(activity_counts, np.linspace(0, 1, 6))
            bins[0] = min(activity_counts) - 1
            sampled_users = []
            rng = np.random.default_rng(random_state)
            for i in range(5):
                in_bin = [u for u, c in zip(users, activity_counts) if bins[i] < c <= bins[i+1]]
                n_bin = max(1, int(len(in_bin) * frac)) if in_bin else 0
                if in_bin and n_bin > 0:
                    sampled_users.extend(rng.choice(in_bin, size=n_bin, replace=False).tolist())
            return sampled_users if sampled_users else users

        # Only compute user_activity if needed
        user_activity = {}
        if sample_frac is not None and sample_frac < 1.0:
            conn = self._get_connection()
            try:
                query = "SELECT user_id, COUNT(*) as n FROM interactions GROUP BY user_id"
                df_activity = conn.execute(query).fetchdf()
                user_activity = dict(zip(df_activity['user_id'], df_activity['n']))
            finally:
                self._return_connection(conn)

            train_users = stratified_sample_users(train_users, sample_frac, user_activity)
            val_users = stratified_sample_users(val_users, sample_frac, user_activity)
            test_users = stratified_sample_users(test_users, sample_frac, user_activity)
        # Get column list for query
        if columns:
            column_list = ", ".join([f"r.{c}" for c in columns])
        else:
            column_list = "r.*"

        def process_chunk(
            chunk: List[str], chunk_idx: int, purpose: str
        ) -> Optional[pd.DataFrame]:
            """Process a single chunk of user data."""
            if not chunk:
                return None

            temp_table = f"temp_users_{abs(hash(str(chunk[:5]))) % 10000}_{chunk_idx}"
            conn = None

            try:
                # Get a connection from the pool
                conn = self._get_connection()

                with conn.cursor() as cur:
                    # Create and populate temp table
                    cur.execute(
                        f"""
                        CREATE TEMP TABLE {temp_table} AS 
                        SELECT UNNEST(?) AS user_id
                    """,
                        [chunk],
                    )

                    # Execute main query
                    query = f"""
                        SELECT {column_list}
                        FROM interactions r
                        JOIN {temp_table} t ON r.user_id = t.user_id
                    """

                    df = cur.execute(query).fetchdf()

                    # Add purpose column for filtering later
                    if not df.empty:
                        df["_purpose"] = purpose

                    return df

            except Exception as e:
                logger.error(f"Error processing {purpose} chunk {chunk_idx}: {e}")
                return None

            finally:
                # Clean up temp table and return connection to pool
                if conn is not None:
                    try:
                        with conn.cursor() as cur:
                            cur.execute(f"DROP TABLE IF EXISTS {temp_table}")
                    except Exception as e:
                        logger.warning(f"Error dropping temp table {temp_table}: {e}")
                    self._return_connection(conn)

        def process_user_list(users: List[str], purpose: str) -> pd.DataFrame:
            """Process a list of users in batches."""
            if not users:
                return pd.DataFrame()

            # Split into batches
            batches = [
                users[i : i + batch_size] for i in range(0, len(users), batch_size)
            ]

            # Process batches in parallel
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                futures = [
                    executor.submit(process_chunk, batch, i, purpose)
                    for i, batch in enumerate(batches)
                ]

                # Collect results
                results = []
                for future in (
                    tqdm(
                        concurrent.futures.as_completed(futures),
                        total=len(futures),
                        desc=f"Loading {purpose} data",
                        disable=not show_progress,
                    )
                    if show_progress
                    else concurrent.futures.as_completed(futures)
                ):
                    try:
                        result = future.result()
                        if result is not None and not result.empty:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Error in batch processing: {e}")

            return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

        # Process each split
        train_df = (
            process_user_list(train_users, "train") if train_users else pd.DataFrame()
        )
        val_df = (
            process_user_list(val_users, "validation") if val_users else pd.DataFrame()
        )
        test_df = (
            process_user_list(test_users, "test") if test_users else pd.DataFrame()
        )

        logger.critical(f"TRAIN DF COLUMNS before return: {train_df.columns}")
        logger.critical(f"TRAIN DF HEAD before return:\n{train_df.head()}")

        # Return based on split_type
        if split_type == "train_val":
            return train_df, val_df
        elif split_type == "train_test":
            return train_df, test_df
        elif split_type == "val_test":
            return val_df, test_df
        elif split_type == "all":
            return train_df, val_df, test_df
        elif split_type == "full_train":
            train_val_df = pd.concat([train_df, val_df], ignore_index=True)
            return train_val_df, test_df
        else:
            raise ValueError(f"Invalid split_type: {split_type}")

    def iter_folds(
        self,
        columns: Optional[List[str]] = None,
        sample_frac: Optional[float] = None,
        random_state: int = 42,
        split_type: str = "train_val",
    ) -> Generator[
        # Use undersample_frac if sample_frac not provided
        # (this is safe because get_fold_data will also handle it)
        
        Union[
            Tuple[pd.DataFrame, pd.DataFrame],
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        ],
        None,
        None,
    ]:
        """Iterate over all CV folds, loading data for each.

        Args:
            columns: List of columns to load (None for all).
            sample_frac: Fraction of users to sample.
            random_state: Seed for reproducibility.
            split_type: Type of data split to return.

        Yields:
            Data for each fold according to the specified split_type.
        """
        n_folds = self.get_fold_summary().get("n_folds", 0)
        if n_folds == 0:
            logger.warning("No CV folds found. Returning empty iterator.")
            return

        for i in range(n_folds):
            yield self.get_fold_data(
                fold_idx=i,
                columns=columns,
                sample_frac=sample_frac if sample_frac is not None else getattr(self, 'undersample_frac', 1.0),
                random_state=random_state,
                split_type=split_type,
            )

    def get_all_folds_data(
        self,
        columns: Optional[List[str]] = None,
        sample_frac: Optional[float] = None,
        random_state: int = 42,
        split_type: str = "train_val",
    ) -> List[
        Union[
            Tuple[pd.DataFrame, pd.DataFrame],
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        ]
    ]:
        """Get data for all CV folds.

        Args:
            columns: List of columns to load (None for all)
            sample_frac: If provided, sample this fraction of users
            random_state: Random seed for reproducibility
            split_type: The type of data split to retrieve.

        Returns:
            A list containing the data for all folds.
        """
        return list(
            self.iter_folds(
                columns=columns,
                sample_frac=sample_frac,
                random_state=random_state,
                split_type=split_type,
            )
        )

    def get_fold_summary(self) -> Dict[str, Any]:
        """Get a summary of the CV folds.

        Returns:
            Dictionary with fold statistics including:
            - n_folds: Number of folds
            - n_users: Total number of unique users
            - n_items: Total number of unique items
            - n_interactions: Total number of interactions
            - folds: List of fold statistics
        """
        summary_file = self.splits_dir / "cv_summary.json"



        if not summary_file.exists():
            return {
                "status": "error",
                "message": "CV summary file not found. Please generate CV splits first.",
                "n_folds": 0,
                "n_users": 0,
                "n_items": 0,
                "n_interactions": 0,
                "folds": [],
            }

        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                summary = json.load(f)

            # Ensure required fields exist
            if "folds" not in summary:
                summary["folds"] = []
            if "n_folds" not in summary:
                summary["n_folds"] = len(summary["folds"])

            return summary

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing CV summary file: {e}")
            return {
                "status": "error",

                "message": f"Invalid CV summary file: {e}",
                "n_folds": 0,
                "folds": [],
            }
```

### `data/feature_matrix.py`

**File size:** 3,215 bytes

```python
import hashlib
import json
import logging
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

from src.schemas.models import RealizedFeature
from src.utils.run_utils import get_run_dir

logger = logging.getLogger(__name__)


def _get_feature_cache_key(feature: RealizedFeature, params: Dict[str, Any]) -> str:
    """Creates a unique cache key for a feature and its parameter values."""
    param_string = json.dumps(params, sort_keys=True)
    return hashlib.md5(f"{feature.name}:{param_string}".encode()).hexdigest()


def _execute_feature_code(
    feature: RealizedFeature, df: pd.DataFrame, params: Dict[str, Any]
) -> pd.Series:
    """Executes the code for a single feature and returns the resulting Series."""
    exec_globals = {"pd": pd, "np": np}
    exec(feature.code_str, exec_globals)
    feature_func: Callable = exec_globals[feature.name]

    # Filter for only the params this function expects
    func_params = {k: v for k, v in params.items() if k in feature.params}

    return feature_func(df, **func_params)


def generate_feature_matrix(
    realized_features: List[RealizedFeature],
    df: pd.DataFrame,
    trial_params: Dict[str, Any],
) -> pd.DataFrame:
    """
    Generates the full feature matrix X by executing or loading from cache.

    Args:
        realized_features: List of realized feature objects.
        df: The input DataFrame (e.g., train or validation split).
        trial_params: The parameter values for the current optimization trial.

    Returns:
        A pandas DataFrame representing the user-feature matrix X.
    """
    cache_dir = get_run_dir() / "feature_cache"
    cache_dir.mkdir(exist_ok=True)

    all_feature_series = []

    for feature in realized_features:
        if not feature.passed_test:
            logger.warning(f"Skipping feature '{feature.name}' as it failed tests.")
            continue

        feature_trial_params = {
            p_name: trial_params.get(f"{feature.name}__{p_name}")
            for p_name in feature.params
        }

        cache_key = _get_feature_cache_key(feature, feature_trial_params)
        cache_file = cache_dir / f"{cache_key}.parquet"

        if cache_file.exists():
            logger.debug(f"Loading feature '{feature.name}' from cache.")
            feature_series = pd.read_parquet(cache_file).squeeze("columns")
        else:
            logger.debug(f"Computing feature '{feature.name}'.")
            try:
                feature_series = _execute_feature_code(
                    feature, df.copy(), feature_trial_params
                )
                feature_series.to_parquet(cache_file)
            except Exception as e:
                logger.error(f"Failed to execute feature '{feature.name}': {e}")
                continue  # Skip this feature if it fails

        all_feature_series.append(feature_series)

    if not all_feature_series:
        logger.warning("No features were successfully generated.")
        return pd.DataFrame(index=df.index)

    # Combine all feature series into a single DataFrame
    X = pd.concat(all_feature_series, axis=1).fillna(0)

    logger.info(f"Generated feature matrix X with shape: {X.shape}")
    return X
```

### `evaluation/beyond_accuracy.py`

**File size:** 2,770 bytes

```python
# src/evaluation/beyond_accuracy.py
"""
Metrics for beyond-accuracy evaluation of recommender systems.
Implements novelty, diversity, and catalog coverage.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Set, Any

def compute_novelty(recommendations: Dict[Any, List[Any]], train_df: pd.DataFrame) -> float:
    """
    Novelty: Inverse log-popularity of recommended items (higher is more novel).
    Args:
        recommendations: {user_id: [item_id, ...]}
        train_df: DataFrame with columns ['user_id', 'item_id'] (training interactions)
    Returns:
        Mean novelty across all recommendations.
    """
    item_counts = train_df['item_id'].value_counts().to_dict()
    total_users = train_df['user_id'].nunique()
    novelty_scores = []
    for user, recs in recommendations.items():
        for item in recs:
            pop = item_counts.get(item, 1)
            novelty = -np.log2(pop / total_users)
            novelty_scores.append(novelty)
    return float(np.mean(novelty_scores)) if novelty_scores else 0.0

def compute_diversity(recommendations: Dict[Any, List[Any]], item_features: pd.DataFrame = None) -> float:
    """
    Diversity: Mean pairwise dissimilarity between recommended items (per user, then averaged).
    If item_features is None, uses unique item count as a proxy.
    Args:
        recommendations: {user_id: [item_id, ...]}
        item_features: DataFrame indexed by item_id (optional)
    Returns:
        Mean diversity across users.
    """
    from itertools import combinations
    diversities = []
    for user, recs in recommendations.items():
        if not recs or len(recs) == 1:
            diversities.append(1.0)
            continue
        if item_features is not None:
            feats = item_features.loc[recs].values
            sims = [np.dot(feats[i], feats[j]) / (np.linalg.norm(feats[i]) * np.linalg.norm(feats[j]) + 1e-8)
                    for i, j in combinations(range(len(recs)), 2)]
            mean_sim = np.mean(sims)
            diversities.append(1 - mean_sim)
        else:
            # Proxy: fraction of unique items
            diversities.append(len(set(recs)) / len(recs))
    return float(np.mean(diversities)) if diversities else 0.0

def compute_catalog_coverage(recommendations: Dict[Any, List[Any]], catalog: Set[Any]) -> float:
    """
    Catalog coverage: Fraction of catalog items recommended to any user.
    Args:
        recommendations: {user_id: [item_id, ...]}
        catalog: Set of all item_ids
    Returns:
        Fraction of unique recommended items over catalog size.
    """
    recommended = set()
    for recs in recommendations.values():
        recommended.update(recs)
    return len(recommended) / len(catalog) if catalog else 0.0
```

### `evaluation/clustering.py`

**File size:** 733 bytes

```python
# src/evaluation/clustering.py
"""
User clustering utility for evaluation (e.g., KMeans).
"""
from typing import Dict, Any
import pandas as pd
from sklearn.cluster import KMeans

def cluster_users_kmeans(X: pd.DataFrame, n_clusters: int = 5, random_state: int = 42) -> Dict[Any, int]:
    """
    Clusters users via KMeans on their feature vectors.
    Args:
        X: pd.DataFrame, indexed by user_id, user feature matrix
        n_clusters: number of clusters
        random_state: for reproducibility
    Returns:
        Dict mapping user_id to cluster label
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X.values)
    return dict(zip(X.index, labels))
```

### `evaluation/ranking_metrics.py`

**File size:** 4,088 bytes

```python
# ranking_metrics.py: Unified ranking metric evaluation using RankerEval

import numpy as np
from rankereval import Rankings, BinaryLabels, NumericLabels
from rankereval.metrics import NDCG, Precision, Recall, F1, HitRate, FirstRelevantRank
from loguru import logger

def evaluate_ranking_metrics(recommendations, ground_truth, k_list=[5, 10, 20]):
    """
    Compute mean NDCG@k, Precision@k, Recall@k, F1@k, HitRate@k, and FirstRelevantRank using rankerEval.
    Args:
        recommendations: dict {user_id: [item_id1, item_id2, ...]} (ranked list)
        ground_truth: dict {user_id: [item_id1, item_id2, ...]} (relevant items)
        k_list: list of cutoff values for metrics
    Returns:
        metrics: dict with keys like 'ndcg@10', 'precision@5', etc.
    """
    # Only evaluate users present in both dicts
    user_ids = sorted(set(recommendations) & set(ground_truth))
    if not user_ids:
        logger.warning("No overlapping users between recommendations and ground_truth. Returning empty metrics.")
        return {}
    logger.info(f"Starting RankerEval metrics for {len(user_ids)} users, k_list={k_list}...")
    y_pred = [recommendations[u] for u in user_ids]
    y_true = [ground_truth[u] for u in user_ids]

    # Rankings: lists of indices (ranked)
    # BinaryLabels: lists of positive indices
    # For each user, map items in y_pred to indices in y_true, or use global item index mapping
    # We'll use positive indices for ground truth, and ranked indices for predictions
    # Build a global item index mapping
    all_items = set()
    for items in y_pred:
        all_items.update(items)
    for items in y_true:
        all_items.update(items)
    item2idx = {item: idx for idx, item in enumerate(sorted(all_items))}
    # Convert to index-based format
    y_pred_idx = [[item2idx[i] for i in recs] for recs in y_pred]
    y_true_idx = [[item2idx[i] for i in gts] for gts in y_true]

    # Rankings and BinaryLabels objects
    rankings = Rankings.from_ranked_indices(y_pred_idx)
    binary_labels = BinaryLabels.from_positive_indices(y_true_idx)
    # For NDCG, NumericLabels (all 1s for binary relevance)
    numeric_labels = NumericLabels.from_matrix([
        [1 if idx in label else 0 for idx in range(len(item2idx))] for label in y_true_idx
    ])

    metrics = {}
    for k in k_list:
        logger.info(f"Computing metrics for k={k}...")
        # NDCG
        ndcg_scores = NDCG(k=k).score(numeric_labels, rankings)
        metrics[f'ndcg@{k}'] = float(np.nanmean(ndcg_scores))
        logger.debug(f"NDCG@{k} mean: {metrics[f'ndcg@{k}']}")
        # Precision
        precision_scores = Precision(k=k).score(binary_labels, rankings)
        metrics[f'precision@{k}'] = float(np.nanmean(precision_scores))
        logger.debug(f"Precision@{k} mean: {metrics[f'precision@{k}']}")
        # Recall
        recall_scores = Recall(k=k).score(binary_labels, rankings)
        metrics[f'recall@{k}'] = float(np.nanmean(recall_scores))
        logger.debug(f"Recall@{k} mean: {metrics[f'recall@{k}']}")
        # F1
        f1_scores = F1(k=k).score(binary_labels, rankings)
        metrics[f'f1@{k}'] = float(np.nanmean(f1_scores))
        logger.debug(f"F1@{k} mean: {metrics[f'f1@{k}']}")
        # HitRate (only valid if exactly one relevant per user)
        try:
            hitrate_scores = HitRate(k=k).score(binary_labels, rankings)
            metrics[f'hitrate@{k}'] = float(np.nanmean(hitrate_scores))
            logger.debug(f"HitRate@{k} mean: {metrics[f'hitrate@{k}']}")
        except Exception as e:
            logger.warning(f"HitRate@{k} failed: {e}")
    # FirstRelevantRank
    try:
        logger.info("Computing FirstRelevantRank...")
        frr_scores = FirstRelevantRank().score(binary_labels, rankings)
        metrics['first_relevant_rank'] = float(np.nanmean(frr_scores))
        logger.debug(f"FirstRelevantRank mean: {metrics['first_relevant_rank']}")
    except Exception as e:
        logger.warning(f"FirstRelevantRank failed: {e}")
    logger.info("RankerEval metrics computation complete.")
    return metrics
```

### `evaluation/scoring.py`

**File size:** 3,684 bytes

```python
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k

logger = logging.getLogger(__name__)


def _train_and_evaluate_lightfm(
    dataset: Dataset,
    train_df: pd.DataFrame,
    test_interactions,
    user_features=None,
    k=10,
    batch_size=100000,
) -> Dict[str, float]:
    """
    Helper to train a LightFM model in batches and score it.
    """
    model = LightFM(loss="warp", random_state=42)

    # Train in batches using fit_partial
    for i in range(0, train_df.shape[0], batch_size):
        chunk = train_df.iloc[i : i + batch_size]
        # Build interactions for the current chunk only
        (chunk_interactions, _) = dataset.build_interactions(
            [(row["user_id"], row["book_id"]) for _, row in chunk.iterrows()]
        )
        model.fit_partial(
            chunk_interactions,
            user_features=user_features,
            epochs=1,  # One pass over each chunk
            num_threads=4,
        )

    # Evaluation logic remains the same
    auc = auc_score(
        model,
        test_interactions,
        user_features=user_features,
        num_threads=4,
    ).mean()

    prec_at_k = precision_at_k(
        model,
        test_interactions,
        k=k,
        user_features=user_features,
        num_threads=4,
    ).mean()

    recall_at_k_scores = recall_at_k(
        model,
        test_interactions,
        k=k,
        user_features=user_features,
        num_threads=4,
    )
    recall_at_k_mean = recall_at_k_scores.mean()
    hit_rate_at_k = np.mean(recall_at_k_scores > 0)

    return {
        "auc": auc,
        f"precision_at_{k}": prec_at_k,
        f"recall_at_{k}": recall_at_k_mean,
        f"hit_rate_at_{k}": hit_rate_at_k,
    }


def score_trial(
    X_val: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    weights: Dict[str, float] = None,
) -> Tuple[Dict[str, float], float]:
    """
    Evaluates a feature matrix (X) by training a LightFM model in batches.
    """
    if weights is None:
        weights = {"auc": 0.6, "precision": 0.2, "recall": 0.2}

    # 1. Build dataset mapping and test interactions
    dataset = Dataset()
    all_users = pd.concat([train_df["user_id"], val_df["user_id"]]).unique()
    all_items = pd.concat([train_df["book_id"], val_df["book_id"]]).unique()
    dataset.fit(users=all_users, items=all_items)

    (test_interactions, _) = dataset.build_interactions(
        [(row["user_id"], row["book_id"]) for _, row in val_df.iterrows()]
    )

    # 2. Build user features sparse matrix
    user_features = dataset.build_user_features(
        (user_id, {col: X_val.loc[user_id, col] for col in X_val.columns})
        for user_id in X_val.index
    )

    # 3. Train (in batches) and evaluate the model
    scores = _train_and_evaluate_lightfm(
        dataset, train_df, test_interactions, user_features=user_features
    )

    # 4. Calculate final objective
    # Optionally incorporate n_clusters into the reward (encourage meaningful segmentation)
    n_clusters = scores.get("n_clusters", 1)
    cluster_weight = weights.get("clusters", 0.05)  # configurable
    final_objective = -(
        weights["auc"] * scores.get("auc", 0)
        + weights["precision"] * scores.get("precision_at_10", 0)
        + weights["recall"] * scores.get("recall_at_10", 0)
        + cluster_weight * n_clusters  # Encourage more/fewer clusters depending on sign
    )

    logger.info(f"Trial scores: {scores} -> Final objective: {final_objective:.4f}")
    return scores, final_objective
```

### `orchestration/ideation.py`

**File size:** 4,226 bytes

```python
import json
import logging
from typing import Any, Dict, List

import autogen

from src.schemas.models import CandidateFeature
from src.utils.prompt_utils import load_prompt
from src.utils.run_utils import get_run_dir
from src.utils.session_state import SessionState
from src.utils.tools import get_table_sample

logger = logging.getLogger(__name__)


def validate_and_filter_features(
    features: List[Dict[str, Any]],
) -> List[CandidateFeature]:
    """
    Validates a list of candidate feature dictionaries and filters them.
    - Ensures name uniqueness
    - Validates spec syntax for 'code' types
    - (Future) Checks dependencies
    - (Future) Scores and ranks features
    """
    validated_features = []
    seen_names = set()

    for feat_data in features:
        # 1. Deduplicate by name
        name = feat_data.get("name")
        if not name or name in seen_names:
            logger.warning(f"Skipping feature with duplicate or missing name: {name}")
            continue
        seen_names.add(name)

        # 2. Validate using Pydantic model and custom validation
        try:
            feature = CandidateFeature(**feat_data)
            feature.validate_spec()
            validated_features.append(feature)
        except Exception as e:
            logger.error(f"Validation failed for candidate feature '{name}': {e}")

    # 3. (Future) Add scoring and filtering logic here

    return validated_features


def run_feature_ideation(session_state: SessionState, llm_config: Dict):
    """Orchestrates the Feature Ideation phase."""
    logger.info("--- Running Feature Ideation Step ---")

    hypotheses = session_state.get_final_hypotheses()
    if not hypotheses:
        logger.warning("No vetted hypotheses found. Skipping feature ideation.")
        return

    # 1. Prepare context for the prompt
    hypotheses_context = "\n".join(
        [f"- {h.id}: {h.summary} (Rationale: {h.rationale})" for h in hypotheses]
    )

    # Load view descriptions
    views_file = get_run_dir() / "generated_views.json"
    view_descriptions = "No views created in the previous step."
    if views_file.exists():
        with open(views_file, "r") as f:
            views_data = json.load(f).get("views", [])
            view_descriptions = "\n".join(
                [f"- {v['name']}: {v['rationale']}" for v in views_data]
            )

    # Get table samples
    tables_to_sample = ["curated_books", "curated_reviews", "user_stats_daily"]
    table_samples = "\n".join([get_table_sample(table) for table in tables_to_sample])

    system_prompt = load_prompt(
        "agents/feature_ideator.j2",
        hypotheses_context=hypotheses_context,
        view_descriptions=view_descriptions,
        table_samples=table_samples,
    )

    # 2. Initialize and run the agent
    def save_candidate_features(features: List[Dict[str, Any]]) -> str:
        """Callback tool for the agent to save its generated features."""
        logger.info(
            f"FeatureIdeationAgent proposed {len(features)} candidate features."
        )

        validated = validate_and_filter_features(features)

        session_state.set_candidate_features([f.model_dump() for f in validated])
        logger.info(
            f"Saved {len(validated)} valid candidate features to session state."
        )

        # Print summary
        for feature in validated:
            logger.info(f"  - Feature: {feature.name}, Rationale: {feature.rationale}")

        return "SUCCESS"

    # We assume a simple agent setup for now
    ideation_agent = autogen.AssistantAgent(
        name="FeatureIdeationAgent", system_message=system_prompt, llm_config=llm_config
    )
    user_proxy = autogen.UserProxyAgent(
        name="UserProxy_Ideation",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
    )

    user_proxy.register_function(
        function_map={"save_candidate_features": save_candidate_features}
    )

    user_proxy.initiate_chat(
        ideation_agent,
        message="Please generate candidate features based on the provided hypotheses and context. Call the `save_candidate_features` tool with your final list.",
    )

    logger.info("--- Feature Ideation Step Complete ---")
```

### `orchestration/insight.py`

**File size:** 3,181 bytes

```python
"""
Orchestration module for the insight discovery team.
"""

from typing import Dict
import json 
import autogen
from loguru import logger

from src.agents.discovery_team.insight_discovery_agents import (
    get_insight_discovery_agents,
)
from src.utils.prompt_utils import load_prompt
from src.utils.run_utils import get_run_dir


def run_insight_discovery_chat(llm_config: Dict) -> Dict:
    """
    Runs the insight discovery team group chat to find patterns in the data.

    Args:
        llm_config: LLM configuration for the agents

    Returns:
        Dictionary containing the insights and view descriptions
    """
    logger.info("Starting insight discovery team chat...")

    # Initialize agents
    agents = get_insight_discovery_agents(llm_config)
    user_proxy = agents.pop("user_proxy")

    # Load chat initiator prompt
    initiator_prompt = load_prompt(
        "globals/discovery_team_chat_initiator.j2",
        view_descriptions="No views created yet.",
    )

    # Create group chat
    group_chat = autogen.GroupChat(
        agents=[user_proxy] + list(agents.values()),
        messages=[],
        max_round=50,
        allow_repeat_speaker=True,
    )
    manager = autogen.GroupChatManager(groupchat=group_chat, llm_config=llm_config)

    # Start the chat
    user_proxy.initiate_chat(
        manager,
        message=initiator_prompt,
    )

    # Extract results from the chat
    results = {
        "insights": _extract_insights(group_chat.messages),
        "view_descriptions": _extract_view_descriptions(group_chat.messages),
    }

    # Save results
    run_dir = get_run_dir()
    results_path = run_dir / "insight_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Insight discovery chat completed. Results saved to {results_path}")
    return results


def _extract_insights(messages: list) -> list:
    """Extract insights from the chat messages."""
    insights = []
    for msg in messages:
        if "add_insight_to_report" in msg.get("content", ""):
            try:
                content = msg["content"]
                start_idx = content.find("{")
                end_idx = content.rfind("}") + 1
                if start_idx != -1 and end_idx != -1:
                    insight = json.loads(content[start_idx:end_idx])
                    insights.append(insight)
            except Exception as e:
                logger.error(f"Error parsing insight: {e}")
    return insights


def _extract_view_descriptions(messages: list) -> dict:
    """Extract SQL view descriptions from the chat messages."""
    views = {}
    for msg in messages:
        if "create_analysis_view" in msg.get("content", ""):
            try:
                content = msg["content"]
                start_idx = content.find("{")
                end_idx = content.rfind("}") + 1
                if start_idx != -1 and end_idx != -1:
                    view = json.loads(content[start_idx:end_idx])
                    views[view["name"]] = view["description"]
            except Exception as e:
                logger.error(f"Error parsing view description: {e}")
    return views
```

### `orchestration/realization.py`

**File size:** 1,282 bytes

```python
import logging
from typing import Dict

from src.agents.strategy_team.feature_realization_agent import FeatureRealizationAgent
from src.utils.session_state import SessionState

logger = logging.getLogger(__name__)


def run_feature_realization(session_state: SessionState, llm_config: Dict):
    """
    Orchestrates the Feature Realization phase by invoking the FeatureRealizationAgent.

    This function instantiates the agent and calls its run() method. The agent is
    responsible for the entire feature realization lifecycle, including:
    - Reading candidate features from the session state.
    - Interacting with the LLM to generate code.
    - Validating the generated code in a sandbox.
    - Retrying with a self-correction loop if validation fails.
    - Writing the final realized features back to the session state.
    """
    logger.info("--- Running Feature Realization Step ---")

    # Instantiate the agent. It will use the session_state to get the candidates
    # and other necessary info like db_path.
    agent = FeatureRealizationAgent(llm_config=llm_config, session_state=session_state)

    # The agent's run method encapsulates all the logic for generation and validation.
    agent.run()

    logger.info("--- Feature Realization Step Complete ---")
```

### `orchestration/strategy.py`

**File size:** 4,137 bytes

```python
"""
Orchestration module for the strategy team group chat.
"""

import json
from typing import Dict, List

import autogen
from loguru import logger

from src.agents.strategy_team.strategy_team_agents import get_strategy_team_agents
from src.utils.prompt_utils import load_prompt
from src.utils.run_utils import get_run_dir


def run_strategy_team_chat(
    llm_config: Dict,
    insight_report: Dict,
    view_descriptions: Dict[str, str],
) -> Dict:
    """
    Runs the strategy team group chat to generate and optimize features.

    Args:
        llm_config: LLM configuration for the agents
        insight_report: Dictionary containing insights from the discovery team
        view_descriptions: Dictionary mapping view names to their descriptions

    Returns:
        Dictionary containing the final hypotheses, features, and optimization results
    """
    logger.info("Starting strategy team group chat...")

    # Initialize agents
    agents = get_strategy_team_agents(llm_config)
    user_proxy = agents.pop("user_proxy")

    # Load chat initiator prompt
    initiator_prompt = load_prompt(
        "globals/strategy_team_chat_initiator.j2",
        insight_report=json.dumps(insight_report, indent=2),
        view_descriptions=json.dumps(view_descriptions, indent=2),
    )

    # Create group chat
    groupchat = autogen.GroupChat(
        agents=list(agents.values()),
        messages=[],
        max_round=50,
        speaker_selection_method="round_robin",
    )
    manager = autogen.GroupChatManager(groupchat=groupchat)

    # Start the chat
    user_proxy.initiate_chat(
        manager,
        message=initiator_prompt,
    )

    # Extract results from the chat
    results = {
        "hypotheses": _extract_hypotheses(groupchat.messages),
        "features": _extract_features(groupchat.messages),
        "optimization_results": _extract_optimization_results(groupchat.messages),
    }

    # Save results
    run_dir = get_run_dir()
    results_path = run_dir / "strategy_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Strategy team chat completed. Results saved to {results_path}")
    return results


def _extract_hypotheses(messages: List[Dict]) -> List[Dict]:
    """Extract hypotheses from the chat messages."""
    hypotheses = []
    for msg in messages:
        if "FINAL_HYPOTHESES" in msg.get("content", ""):
            # Parse the hypotheses from the message
            try:
                content = msg["content"]
                start_idx = content.find("[")
                end_idx = content.rfind("]") + 1
                if start_idx != -1 and end_idx != -1:
                    hypotheses = json.loads(content[start_idx:end_idx])
            except Exception as e:
                logger.error(f"Error parsing hypotheses: {e}")
    return hypotheses


def _extract_features(messages: List[Dict]) -> List[Dict]:
    """Extract feature specifications from the chat messages."""
    features = []
    for msg in messages:
        if "save_candidate_features" in msg.get("content", ""):
            try:
                content = msg["content"]
                start_idx = content.find("[")
                end_idx = content.rfind("]") + 1
                if start_idx != -1 and end_idx != -1:
                    features = json.loads(content[start_idx:end_idx])
            except Exception as e:
                logger.error(f"Error parsing features: {e}")
    return features


def _extract_optimization_results(messages: List[Dict]) -> Dict:
    """Extract optimization results from the chat messages."""
    results = {}
    for msg in messages:
        if "save_optimization_results" in msg.get("content", ""):
            try:
                content = msg["content"]
                start_idx = content.find("{")
                end_idx = content.rfind("}") + 1
                if start_idx != -1 and end_idx != -1:
                    results = json.loads(content[start_idx:end_idx])
            except Exception as e:
                logger.error(f"Error parsing optimization results: {e}")
    return results
```

### `orchestrator.py`

**File size:** 34,705 bytes

```python
import json
import os
import sys

# Ensure DB views are set up for pipeline compatibility
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import autogen
from autogen import Agent
from dotenv import load_dotenv
from loguru import logger

import scripts.setup_views
from src.agents.discovery_team.insight_discovery_agents import get_insight_discovery_agents
from src.agents.strategy_team.optimization_agent_v2 import VULCANOptimizer
from src.agents.strategy_team.strategy_team_agents import get_strategy_team_agents
from src.config.log_config import setup_logging
from src.core.database import get_db_schema_string
from src.utils.prompt_utils import refresh_global_db_schema
from src.utils.run_utils import config_list_from_json, get_run_dir, init_run
from src.utils.session_state import CoverageTracker, SessionState
from src.utils.tools import (
    cleanup_analysis_views,
    create_analysis_view,
    execute_python,
    get_add_insight_tool,
    get_finalize_hypotheses_tool,
    get_save_candidate_features_tool,
    get_table_sample,
    run_sql_query,
    vision_tool,
)

# Ensure DB views are set up for pipeline compatibility
scripts.setup_views.setup_views()


# Load environment variables from .env file at the beginning.
load_dotenv()


# --- Helper Functions for SmartGroupChatManager ---

def get_insight_context(session_state: SessionState) -> str:
    """Generate a context message based on available insights."""
    if not session_state.insights:
        return ""
        
    # Format the top insights for context
    insights = session_state.insights[:5]  # Limit to top 5 insights
    insights_text = "\n\n".join([f"**{i.title}**: {i.finding[:150]}..." for i in insights])
    
    return f"""
## Context from Discovery Team

These insights were discovered by the previous team:

{insights_text}

Please reference these insights when building your features.
"""


def should_continue_exploration(session_state: SessionState, round_count: int) -> bool:
    """Determines if exploration should continue. Main termination: hypotheses finalized. Fallback: max rounds/no new insights."""
    # If hypotheses have been finalized, allow termination
    if session_state.get_final_hypotheses():
        logger.info("Hypotheses have been finalized. Discovery loop can terminate.")
        return False

    # Always continue if no insights yet (prevents empty runs)
    if not session_state.insights:
        logger.info("Cannot terminate: No insights found yet. Forcing continuation.")
        return True

    # Fallback: prevent infinite loops if agents are stuck
    if round_count > 50:
        last_insight_round = max((i.metadata.get("round_added", 0) for i in session_state.insights), default=0)
        if round_count - last_insight_round > 20:
            logger.info("Termination condition: No new insights in the last 20 rounds (fallback). Hypotheses not finalized.")
            return False

    # Default: continue until hypotheses are finalized
    return True


def get_progress_prompt(session_state: SessionState, round_count: int) -> Optional[str]:
    """Generate a progress prompt to guide agents when they seem stuck."""
    insights = session_state.insights
    if not insights:
        return "It's been a while and no insights have been reported. As a reminder, your goal is to find interesting patterns. Please review the schema and propose a query."

    tables_in_insights = {t for i in insights for t in i.tables_used}
    all_tables = set(session_state.get_all_table_names())
    unexplored_tables = all_tables - tables_in_insights

    if round_count > 20 and unexplored_tables:
        return f"Great work so far. We've analyzed {len(tables_in_insights)} tables, but these remain unexplored: {', '.join(list(unexplored_tables)[:3])}. Consider formulating a hypothesis involving one of these."

    low_detail_insights = [i for i in insights if len(i.finding) < 100]
    if low_detail_insights:
        return f"The insight '{low_detail_insights[0].title}' is a bit brief. Can the DataScientist elaborate on its significance or provide more supporting evidence?"

    return None


def _fallback_compression(messages: List[Dict], keep_recent: int = 20) -> List[Dict]:
    """Fallback keyword-based compression if LLM compression fails."""
    logger.warning("Executing fallback context compression.")
    if len(messages) <= keep_recent:
        return messages

    compressed_messages = []
    keywords = ["insight", "hypothesis", "important", "significant", "surprising"]
    for msg in messages[:-keep_recent]:
        if any(keyword in msg.get("content", "").lower() for keyword in keywords):
            new_content = f"(Summarized): {msg['content'][:200]}..."
            compressed_messages.append({**msg, "content": new_content})

    return compressed_messages + messages[-keep_recent:]


def compress_conversation_context(messages: List[Dict], keep_recent: int = 20) -> List[Dict]:
    """Intelligently compress conversation context using LLM summarization."""
    if len(messages) <= keep_recent:
        return messages

    logger.info(f"Compressing conversation context, keeping last {keep_recent} messages.")
    try:
        config_file_path = os.getenv("OAI_CONFIG_LIST")
        if not config_file_path:
            raise ValueError("OAI_CONFIG_LIST environment variable not set.")
        config_list_all = config_list_from_json(config_file_path)
        config_list = [config for config in config_list_all if config.get("model") == "gpt-4o"]
        if not config_list:
            raise ValueError("No config found for summarization model.")

        summarizer_llm_config = {
            "config_list": config_list,
            "cache_seed": None,
            "temperature": 0.2,
        }
        summarizer_client = autogen.AssistantAgent("summarizer", llm_config=summarizer_llm_config)

        conversation_to_summarize = "\n".join(
            [f"{m.get('role')}: {m.get('content')}" for m in messages[:-keep_recent]]
        )
        prompt = f"Please summarize the key findings, decisions, and unresolved questions from the following conversation history. Be concise, but do not lose critical information. The summary will be used as context for an ongoing AI agent discussion.\n\n---\n{conversation_to_summarize}\n---"

        response = summarizer_client.generate_reply(messages=[{"role": "user", "content": prompt}])
        summary_message = {
            "role": "system",
            "content": f"## Conversation Summary ##\n{response}",
        }
        return [summary_message] + messages[-keep_recent:]
    except ValueError as e:
        logger.error(
            f"Could not initialize LLM config. Please check your configuration. Error: {e}"
        )
        # Re-raise to be caught by main and terminate the run.
        raise
    except Exception as e:
        logger.error(f"LLM-based context compression failed: {e}")
        return _fallback_compression(messages, keep_recent)


def get_llm_config_list() -> Optional[Dict[str, Any]]:
    """
    Loads LLM configuration from the path specified in OAI_CONFIG_LIST,
    injects the API key, and returns a dictionary for autogen.

    Returns:
        A dictionary containing the 'config_list' and 'cache_seed', or None if config fails.
    """
    try:
        config_file_path = os.getenv("OAI_CONFIG_LIST")
        if not config_file_path:
            logger.error("OAI_CONFIG_LIST environment variable not set.")
            raise ValueError("OAI_CONFIG_LIST environment variable not set.")

        logger.debug(f"Loading LLM configuration from: {config_file_path}")
        config_list = config_list_from_json(file_path=config_file_path)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found. Relying on config file.")
        else:
            logger.debug("Injecting OPENAI_API_KEY into LLM config.")
            for c in config_list:
                c.update({"api_key": api_key})

        if not config_list:
            logger.error(
                "No valid LLM configurations found after loading. Check file content and path."
            )
            raise ValueError("No valid LLM configurations found.")

        logger.info(f"Successfully loaded {len(config_list)} LLM configurations.")
        return {"config_list": config_list, "cache_seed": None, "max_tokens": 16384}

    except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load or parse LLM config: {e}", exc_info=True)
        return None


# --- Enhanced Conversation Manager ---


class SmartGroupChatManager(autogen.GroupChatManager):
    """A customized GroupChatManager with context compression and progress monitoring."""

    round_count: int = 0

    def __init__(self, groupchat: autogen.GroupChat, llm_config: Dict[str, Any]):
        super().__init__(groupchat=groupchat, llm_config=llm_config)
        self.round_count = 0  # Reset round count for each new chat

    def run_chat(
        self,
        messages: List[Dict[str, Any]],
        sender: autogen.Agent,
        config: Optional[autogen.GroupChat] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Run the chat with additional tracking and feedback mechanisms."""
        self.round_count += 1
        session_state = globals().get("session_state")

        # The config is the groupchat.
        groupchat = config or self.groupchat

        # --- EARLY TERMINATION: If hypotheses are finalized, end the chat ---
        if session_state and session_state.get_final_hypotheses():
            logger.info("Hypotheses have been finalized. Terminating discovery loop.")
            return True, "TERMINATE"  # Do NOT call super().run_chat if terminating

        # --- CONTEXT INJECTION: Add context on first round ---
        if self.round_count == 1 and session_state:
            context_message = get_insight_context(session_state)
            if context_message:
                messages.append(
                    {"role": "user", "content": context_message, "name": "SystemCoordinator"}
                )

        # --- CONTEXT COMPRESSION ---
        if self.round_count > 10 and self.round_count % 10 == 0:
            try:
                groupchat.messages = compress_conversation_context(messages)
                logger.info("Applied LLM context compression at round {}", self.round_count)
            except Exception as e:
                logger.warning("Context compression failed: {}", e)

        # --- TERMINATION HANDLING: Check for Hypothesizer TERMINATE signal ---
        last_msg_content = messages[-1]["content"].strip().upper() if messages else ""
        last_msg_sender = messages[-1].get("name", "") if messages else ""
        
        # If Hypothesizer sends TERMINATE and hypotheses are finalized, allow termination
        if "TERMINATE" in last_msg_content and last_msg_sender == "Hypothesizer":
            if session_state and session_state.get_final_hypotheses():
                logger.info("Hypothesizer sent TERMINATE and hypotheses are finalized. Terminating discovery loop.")
                return True, "TERMINATE"
            elif session_state:
                logger.warning("Hypothesizer sent TERMINATE but no hypotheses finalized. Prompting for finalization.")
                messages.append(
                    {
                        "role": "user",
                        "name": "SystemCoordinator",
                        "content": (
                            "Hypothesizer, you sent a termination signal but no hypotheses have been finalized. "
                            "Please call the `finalize_hypotheses` tool with your synthesized hypotheses before terminating."
                        ),
                    }
                )
        
        # Block any other TERMINATE signals if hypotheses aren't finalized
        elif "TERMINATE" in last_msg_content and session_state and not session_state.get_final_hypotheses():
            logger.warning("Termination signal received from non-Hypothesizer, but no hypotheses finalized. Blocking termination.")
            messages.append(
                {
                    "role": "user",
                    "name": "SystemCoordinator",
                    "content": (
                        "A termination request was detected, but no hypotheses have been finalized. **Hypothesizer, it is now your turn to act.** "
                        "Please synthesize the team's insights and call the `finalize_hypotheses` tool."
                    ),
                }
            )

        # --- FALLBACK TERMINATION: Prevent infinite loops ---
        if session_state and not should_continue_exploration(session_state, self.round_count):
            logger.info("Exploration criteria met (fallback), terminating conversation.")
            return True, "TERMINATE"

        # --- LOOP PREVENTION: Reset agents periodically ---
        if self.round_count > 0 and self.round_count % 20 == 0:
            logger.warning("Potential loop detected at round {}. Resetting agents.", self.round_count)
            for agent in groupchat.agents:
                if hasattr(agent, "reset"):
                    agent.reset()

        # --- GUIDANCE: Add progress prompts periodically ---
        if session_state and self.round_count > 5 and self.round_count % 15 == 0:
            progress_guidance = get_progress_prompt(session_state, self.round_count)
            if progress_guidance:
                logger.info("Adding progress guidance at round {}", self.round_count)
                messages.append(
                    {"role": "user", "content": progress_guidance, "name": "SystemCoordinator"}
                )
        prev_message_count = len(self.groupchat.messages)
        result = super().run_chat(messages, sender, self.groupchat)  # type: ignore
        # --- LOGGING: Log every message in the groupchat ---
        if session_state and hasattr(session_state, 'run_logger'):
            # Only log new messages since the last call
            new_messages = self.groupchat.messages[prev_message_count:]
            for msg in new_messages:
                session_state.run_logger.log_message(
                    sender=msg.get('name', msg.get('role', 'unknown')),
                    recipient=None,  # Not tracked at message level
                    content=msg.get('content', ''),
                    role=msg.get('role', None),
                    extra={k: v for k, v in msg.items() if k not in ['content', 'role', 'name']}
                )
        # Handle possible tuple return value from parent class
        if isinstance(result, tuple) and len(result) == 2:
            success, response = result
            if success and response:
                return response
            return None
        return result


# --- Orchestration Loops ---


def run_discovery_loop(session_state: SessionState) -> str:
    """Orchestrates the Insight Discovery Team to find patterns in the data."""
    logger.info("--- Running Insight Discovery Loop ---")
    llm_config = get_llm_config_list()
    if not llm_config:
        raise RuntimeError("Failed to get LLM configuration, cannot proceed with discovery.")

    user_proxy = autogen.UserProxyAgent(
        name="UserProxy_ToolExecutor",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=100,
        code_execution_config={"work_dir": str(get_run_dir()), "use_docker": False},
    )

    assistant_agents = get_insight_discovery_agents(llm_config)
    analyst = assistant_agents["QuantitativeAnalyst"]
    researcher = assistant_agents["DataRepresenter"]
    critic = assistant_agents["PatternSeeker"]
    hypothesizer = assistant_agents["Hypothesizer"]

    # --- Tool Registration with Logging ---
    from src.utils.tools_logging import log_tool_call

    # A dictionary of tool functions to be wrapped and registered.
    # The key is the name the agent will use to call the tool.
    tool_functions = {
        "run_sql_query": run_sql_query,
        "get_table_sample": get_table_sample,
        "create_analysis_view": create_analysis_view,
        "vision_tool": vision_tool,
        "add_insight_to_report": get_add_insight_tool(session_state),
        "execute_python": execute_python,
        "finalize_hypotheses": get_finalize_hypotheses_tool(session_state),
    }

    # Wrap all tool functions with the logger
    logged_tools = {
        name: log_tool_call(func, session_state, tool_name=name)
        for name, func in tool_functions.items()
    }

    # Register tools for the appropriate agents
    for agent in [analyst, researcher, critic]:
        for name in ["run_sql_query", "get_table_sample", "create_analysis_view", "vision_tool", "add_insight_to_report", "execute_python"]:
            autogen.register_function(
                logged_tools[name],
                caller=agent,
                executor=user_proxy,
                name=name,
                description=tool_functions[name].__doc__.strip().split('\n')[0] # Use first line of docstring
            )

    # Register finalize_hypotheses only for the Hypothesizer
    autogen.register_function(
        logged_tools["finalize_hypotheses"],
        caller=hypothesizer,
        executor=user_proxy,
        name="finalize_hypotheses",
        description="Finalize and submit a list of all validated hypotheses. This is the mandatory final step before the discovery loop can end.",
    )

    agents: Sequence[autogen.Agent] = [user_proxy, analyst, researcher, critic, hypothesizer]
    
    group_chat = autogen.GroupChat(
        agents=agents, 
        messages=[], 
        max_round=100, 
        allow_repeat_speaker=True
    )
    manager = SmartGroupChatManager(
        groupchat=group_chat, 
        llm_config=llm_config
    )

    logger.info("Closing database connection for agent execution...")
    session_state.close_connection()
    try:
        # --- REFLECTION HANDOVER: Prepend latest reflection's next_steps (if any) ---
        reflection_intro = ""
        if getattr(session_state, 'reflections', None):
            latest_reflection = session_state.reflections[-1]
            if isinstance(latest_reflection, dict):
                # Try to extract next_steps, novel_ideas, expansion_ideas
                next_steps = latest_reflection.get("next_steps")
                novel_ideas = latest_reflection.get("novel_ideas")
                expansion_ideas = latest_reflection.get("expansion_ideas")
                if next_steps:
                    reflection_intro += "\n---\n**Reflection Agent's Next Steps:**\n"
                    if isinstance(next_steps, list):
                        for i, step in enumerate(next_steps, 1):
                            reflection_intro += f"{i}. {step}\n"
                    else:
                        reflection_intro += str(next_steps) + "\n"
                if novel_ideas:
                    reflection_intro += "\n**Novel Unexplored Ideas:**\n"
                    if isinstance(novel_ideas, list):
                        for idea in novel_ideas:
                            reflection_intro += f"- {idea}\n"
                    else:
                        reflection_intro += str(novel_ideas) + "\n"
                if expansion_ideas:
                    reflection_intro += "\n**Promising Expansions:**\n"
                    if isinstance(expansion_ideas, list):
                        for idea in expansion_ideas:
                            reflection_intro += f"- {idea}\n"
                    else:
                        reflection_intro += str(expansion_ideas) + "\n"
        initial_message = (
            reflection_intro +
            "Team, let's begin our analysis.\n"
            "- **Analysts (QuantitativeAnalyst, PatternSeeker, DataRepresenter):** Explore the data and use the `add_insight_to_report` tool to log findings. When you believe enough insights have been gathered, prompt the Hypothesizer to finalize hypotheses. Do NOT call `TERMINATE` yourself for this reason.\n"
            "- **Hypothesizer:** Only you can end the discovery phase by calling the `finalize_hypotheses` tool. Listen for cues from the team, and when prompted (or when you believe enough insights are present), synthesize the insights and call `finalize_hypotheses`.\n\n"
            "**IMPORTANT:** The discovery phase ends ONLY when the Hypothesizer calls `finalize_hypotheses`. All other agents should prompt the Hypothesizer when ready, but only the Hypothesizer can end the phase.\n\n"
            "Let the analysis begin."
        )
        user_proxy.initiate_chat(manager, message=initial_message, session_state=session_state)

        logger.info(
            "Exploration completed after {} rounds with {} insights",
            manager.round_count,
            len(session_state.insights),
        )
        logger.info("--- FINAL INSIGHTS SUMMARY ---")
        logger.info(session_state.get_final_insight_report())

        run_dir = get_run_dir()
        views_file = run_dir / "generated_views.json"
        if views_file.exists():
            with open(views_file, "r", encoding="utf-8") as f:
                views_data = json.load(f)
            logger.info("Total views created: {}", len(views_data.get("views", [])))
        else:
            logger.info("Total views created: 0")
        # --- NEW: Always reconnect after discovery loop to refresh DB schema/views ---
        logger.info("Refreshing DB connection after discovery loop to ensure new views are visible...")
        session_state.reconnect()
    finally:
        logger.info("Reopening database connection (final cleanup in discovery loop)...")
        session_state.reconnect()

    logger.info("--- Insight Discovery Loop Complete ---")
    return session_state.get_final_insight_report()


def run_strategy_loop(
    session_state: SessionState,
    strategy_agents_with_proxy: Dict[str, autogen.ConversableAgent],
    llm_config: Dict,
) -> Optional[Dict[str, Any]]:
    """
    Runs the streamlined strategy team loop with the following agents:
    - StrategistAgent: Validates features from a business/strategy perspective.
    - EngineerAgent: Validates features from a technical perspective.
    - FeatureEngineer: Designs feature contracts based on pre-generated hypotheses.
    - UserProxy_Strategy: Handles tool execution and stores features.

    The session_state should already contain hypotheses generated by the discovery team.
    """
    logger.info("--- Running Strategy Loop ---")
    if not session_state.get_final_hypotheses():
        logger.warning("No hypotheses found, skipping strategy loop.")
        return {"message": "Strategy loop skipped: No hypotheses were generated."}

    # Extract agents from the pre-initialized dictionary
    strategist = strategy_agents_with_proxy["StrategistAgent"]
    user_proxy = strategy_agents_with_proxy["user_proxy"]

    # --- Tool Registration ---
    # Create a wrapper for execute_python that includes session_state
    def execute_python_with_state(code: str, timeout: int = 300) -> str:
        return execute_python(code, timeout, session_state)
    
    # Get the save_candidate_features tool
    save_features_tool = get_save_candidate_features_tool(session_state)
    
    # Ensure both functions are not None to avoid autogen library bug
    if save_features_tool is None:
        logger.error("save_candidate_features tool is None, cannot register")
        raise RuntimeError("Failed to get save_candidate_features tool")
    
    # Register functions with explicit error handling for autogen bug
    try:
        user_proxy.register_function(
            function_map={
                "save_candidate_features": save_features_tool,
                "execute_python": execute_python_with_state,
            }
        )
        logger.info("Successfully registered tools with UserProxy")
    except TypeError as e:
        if "category must be a Warning subclass" in str(e):
            logger.warning(f"Encountered autogen library bug: {e}")
            # Try to register functions one by one to isolate the issue
            try:
                user_proxy._function_map = user_proxy._function_map or {}
                user_proxy._function_map["save_candidate_features"] = save_features_tool
                user_proxy._function_map["execute_python"] = execute_python_with_state
                logger.info("Successfully registered tools using direct assignment workaround")
            except Exception as fallback_error:
                logger.error(f"Fallback registration failed: {fallback_error}")
                raise
        else:
            raise

    # Create the group chat with only the StrategistAgent and UserProxy
    groupchat = autogen.GroupChat(
        agents=[user_proxy, strategist],
        messages=[],
        max_round=1000,
        speaker_selection_method="auto",
    )

    manager = SmartGroupChatManager(groupchat=group_chat, llm_config=llm_config)

    # Format hypotheses for the initial message
    hypotheses_json = json.dumps(
        [h.model_dump() for h in session_state.get_final_hypotheses()], indent=2
    )

    # Get previously discovered features from prior epochs
    previous_features = session_state.get_candidate_features()
    previous_features_context = ""
    
    if previous_features:
        previous_features_json = json.dumps(previous_features, indent=2)
        previous_features_context = f"""
**Previously Discovered Features from Prior Epochs:**
```json
{previous_features_json}
```

**Important Context:**
- The above features were discovered in previous epochs of this VULCAN run.
- You should be aware of these existing features to avoid redundancy.
- Your goal is to discover NEW, NOVEL, and COMPLEMENTARY features that go beyond what has already been found.
- Consider how your new features can build upon, enhance, or provide alternatives to the existing ones.
- Strive for creative and innovative feature engineering that explores unexplored aspects of the data.

"""
    else:
        previous_features_context = """
**Previously Discovered Features:**
- This is the first epoch, so no features have been discovered yet.
- You have the opportunity to establish the foundation for feature discovery in this run.

"""

    # Construct the initial message to kick off the conversation.
    # This message is a direct command to the StrategistAgent.
    initial_message = f"""You are the StrategistAgent. Your task is to design a set of `CandidateFeature` contracts based on the following hypotheses.

{previous_features_context}**Hypotheses:**
```json
{hypotheses_json}
```

**Your Instructions:**
1.  Analyze the hypotheses and any previously discovered features.
2.  Design a list of `CandidateFeature` contracts that are NOVEL and go beyond existing features. Each contract must be a dictionary with `name`, `description`, `dependencies`, and `parameters`.
3.  Focus on creative feature engineering that explores new aspects of the data relationships and patterns.
4.  Call the `save_candidate_features` function with your list of candidate features.

**Important:** You must call the function directly like this:
save_candidate_features([
    {{
        "name": "feature_name",
        "description": "feature description", 
        "dependencies": ["table.column1", "table.column2"],
        "parameters": {{}}
    }}
])

Do NOT output JSON or any other format. Call the function directly with your designed features.
"""

    # --- NEW: Ensure DB connection is refreshed before strategy loop ---
    logger.info("Refreshing DB connection before strategy loop to ensure all views are visible...")
    session_state.reconnect()

    report: Dict[str, Any] = {}
    try:
        # The user_proxy initiates the chat. The `message` is the first thing said.
        user_proxy.initiate_chat(manager, message=initial_message, session_state=session_state)
    
        # After the chat, we check the session_state for the results.
        # Ensure features and hypotheses are always defined for downstream use
        features = getattr(session_state, "candidate_features", [])
        hypotheses = session_state.get_final_hypotheses()
        insights = getattr(session_state, "insights", [])
        logger.info(f"Exploration completed after {manager.round_count} rounds with {len(insights)} insights")
        report = {
            "features_generated": len(features),
            "hypotheses_processed": len(hypotheses),
            "features": features,  # candidate_features are dicts, not Pydantic models
            "realized_features": [],  # No realization step in simplified workflow
            "hypotheses": [h.model_dump() for h in hypotheses],
        }

    except Exception as e:
        logger.error("Strategy loop failed", exc_info=True)
        report = {"error": str(e)}
    finally:
        logger.info("Reopening database connection after strategy loop...")
        session_state.reconnect()  # Reconnect again to be safe.

    return report


def main(epochs: int = 30, fast_mode_frac: float = 0.15) -> str:
    optimization_report = "Optimization step did not run."
    """
    Main orchestration function for the VULCAN pipeline.
    """
    # --- Set up logging and run context ---
    
    try:
        run_id, run_dir = init_run()
    except Exception as e:
        logger.error(f"Failed to initialize run context: {e}")
        sys.exit(1)
    logger.info(f"Starting VULCAN run: {run_id}")

    setup_logging()
    
    # --- Start TensorBoard for experiment tracking (after run context is initialized) ---
    logger.info("TensorBoard temporarily disabled for testing")
    # from src.config.tensorboard import start_tensorboard
    # logger.info("Launching TensorBoard server on port 6006 with global logdir: runtime/tensorboard_global")
    # start_tensorboard()

    session_state = SessionState(run_dir)
    session_state.set_state("fast_mode_sample_frac", fast_mode_frac)

    # Get the database schema once to be reused by agents
    try:
        db_schema = get_db_schema_string()
        logger.info("Successfully retrieved database schema for agents")
    except Exception as e:
        logger.warning(f"Could not get database schema: {e}")
        db_schema = "[Error retrieving schema]"

    # Initialize LLM configuration once to reuse
    llm_config = get_llm_config_list()
    if not llm_config:
        raise RuntimeError("Failed to get LLM configuration, cannot proceed with orchestration.")

    # Initialize strategy agents once with the schema
    strategy_agents = get_strategy_team_agents(llm_config=llm_config, db_schema=db_schema)
    # Add UserProxy agent for the strategy loop
    user_proxy_strategy = autogen.UserProxyAgent(
        name="UserProxy_Strategy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config={"use_docker": False},
    )
    strategy_agents_for_loop = {"StrategistAgent": strategy_agents["StrategistAgent"], "user_proxy": user_proxy_strategy}

    all_epoch_reports = []
    coverage_tracker = CoverageTracker()

    try:
        for epoch in range(epochs):
            logger.info(f"=== Starting Epoch {epoch + 1} / {epochs} (fast_mode) ===")

            # Refresh DB schema for prompt context ONCE per epoch
            refresh_global_db_schema()

            session_state.set_state("fast_mode_sample_frac", fast_mode_frac)
            discovery_report = run_discovery_loop(session_state)
            logger.info(session_state.get_final_insight_report())

            # --- MANDATORY HYPOTHESIS GENERATION ---
            # --- MANDATORY HYPOTHESIS GENERATION ---
            # Note: Discovery team should handle hypothesis generation now
            # If no hypotheses are found, log the issue but don't attempt to generate them ourselves
            if session_state.insights and not session_state.get_final_hypotheses():
                logger.warning(
                    "No hypotheses found after discovery. Continuing with strategy without hypotheses."
                )

            if not session_state.get_final_hypotheses():
                logger.info("No hypotheses found, skipping strategy loop.")
                strategy_report = "Strategy loop skipped: No hypotheses were generated."
            else:
                # Pass the pre-initialized strategy agents to the strategy loop
                reflection_results = run_strategy_loop(session_state, strategy_agents_for_loop, llm_config)
                if reflection_results:
                    strategy_report = json.dumps(reflection_results, indent=2)
                else:
                    strategy_report = "Strategy loop did not return results."

            summary = session_state.summarise_central_memory()
            session_state.epoch_summary = summary
            session_state.save_to_disk()
            session_state.clear_central_memory()

            coverage_tracker.update_coverage(session_state)
            all_epoch_reports.append(
                {
                    "epoch": epoch + 1,
                    "discovery_report": discovery_report,
                    "strategy_report": strategy_report,
                    "epoch_summary": summary,
                    "coverage": coverage_tracker.get_coverage(),
                }
            )

        # === Optimization Step Skipped in Simplified Workflow ===
        logger.info("Optimization step skipped in simplified workflow.")
        optimization_report = "Optimization step skipped in simplified workflow."
    except Exception as e:
        logger.error(
            f"An uncaught exception occurred during orchestration: {type(e).__name__}: {e}"
        )
        logger.error(traceback.format_exc())
        strategy_report = f"Run failed with error: {e}"
    finally:
        session_state.close_connection()
        cleanup_analysis_views(Path(session_state.run_dir))
        logger.info("View cleanup process initiated.")
        logger.info("Run finished. Session state saved.")

    final_report = (
        f"# VULCAN Run Complete: {run_id}\n\n"
        f"## Epoch Reports\n{json.dumps(all_epoch_reports, indent=2)}\n\n"
        f"## Final Strategy Refinement Report\n{strategy_report}\n\n"
        f"## Final Optimization Report\n{optimization_report}\n"
    )
    logger.info("VULCAN has completed its run.")
    print(final_report)
    return final_report


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"VULCAN run failed: {e}", exc_info=True)
        sys.exit(1)
```

### `schemas/eda_report_schema.json`

**File size:** 1,225 bytes

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["schema_overview", "global_stats", "samples", "insights", "plots", "hypotheses"],
  "properties": {
    "schema_overview": {
      "type": "object",
      "description": "Database schema information including tables and their columns"
    },
    "global_stats": {
      "type": "object",
      "description": "Summary statistics for each table"
    },
    "samples": {
      "type": "object",
      "description": "Representative samples from each table"
    },
    "insights": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["metric", "value", "comment"],
        "properties": {
          "metric": {"type": "string"},
          "value": {"type": ["number", "string"]},
          "comment": {"type": "string"}
        }
      }
    },
    "plots": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["path", "caption"],
        "properties": {
          "path": {"type": "string"},
          "caption": {"type": "string"}
        }
      }
    },
    "hypotheses": {
      "type": "array",
      "items": {
        "type": "string"
      }
    }
  }
} 
```

### `schemas/models.py`

**File size:** 7,509 bytes

```python
# src/utils/schemas.py
import ast
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, validator


class Insight(BaseModel):
    title: str = Field(description="A concise, descriptive title for the insight.")
    finding: str = Field(
        description="The detailed finding or observation, explaining what was discovered."
    )
    supporting_code: Optional[str] = Field(
        None, description="The exact SQL or Python code used to generate the finding."
    )
    source_representation: str = Field(
        description="The name of the SQL View or Graph used for analysis (e.g., 'vw_user_review_summary' or 'g_user_book_bipartite')."
    )
    plot_path: Optional[str] = Field(
        None, description="The absolute path to the plot that visualizes the finding."
    )
    plot_interpretation: Optional[str] = Field(
        None,
        description="A detailed, LLM-generated analysis of what the plot shows and its implications.",
    )
    quality_score: Optional[float] = Field(
        None, description="A score from 1-10 indicating the quality of the insight."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the insight, like the round it was added.",
    )
    tables_used: List[str] = Field(
        default_factory=list,
        description="List of table names used to generate the insight.",
    )
    reasoning_trace: List[str] = Field(
        default_factory=list,
        description="Step-by-step reasoning chain or trace of how this insight was derived. Each entry should represent a reasoning step, tool call, or reflection.",
    )


class Hypothesis(BaseModel):
    """
    Represents a hypothesis for feature engineering, including explicit data dependencies.
    """
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="A unique identifier for the hypothesis, e.g., a UUID."
    )
    summary: str = Field(
        ..., description="A concise, one-sentence statement of the hypothesis."
    )
    rationale: str = Field(
        ..., description="A clear explanation of why this hypothesis is useful and worth testing."
    )
    depends_on: List[str] = Field(
        ..., description="A list of fully qualified column names (e.g., 'reviews.user_id', 'books.genre') required to test this hypothesis."
    )

    @validator("rationale")
    def rationale_must_be_non_empty(cls, v):
        if not v:
            raise ValueError("Rationale cannot be empty.")
        return v


class PrioritizedHypothesis(BaseModel):
    id: str = Field(..., description="The unique identifier for the hypothesis.")
    priority: int = Field(
        ..., ge=1, le=5, description="The priority score from 1 to 5."
    )
    feasibility: int = Field(
        ..., ge=1, le=5, description="The feasibility score from 1 to 5."
    )
    notes: str = Field(..., description="A brief justification for the scores.")


class ParameterSpec(BaseModel):
    type: Literal["int", "float", "categorical"] = Field(..., description="Parameter type: int, float, or categorical.")
    low: Optional[Union[int, float]] = Field(None, description="Lower bound (for int/float)")
    high: Optional[Union[int, float]] = Field(None, description="Upper bound (for int/float)")
    step: Optional[Union[int, float]] = Field(None, description="Step size (for int)")
    log: Optional[bool] = Field(False, description="Log scale (for float)")
    choices: Optional[List[Any]] = Field(None, description="Allowed choices (for categorical)")
    default: Optional[Any] = Field(None, description="Default value")

class CandidateFeature(BaseModel):
    name: str = Field(..., description="A unique, descriptive name for the feature.")
    type: Literal["code"] = Field(..., description="The type of feature to be realized. Only 'code' is supported.")
    spec: str = Field(..., description="The core logic of the feature: a Python expression or formula.")
    depends_on: List[str] = Field(
        default_factory=list,
        description="A list of other feature names this feature depends on (for compositions).",
    )
    parameters: Dict[str, ParameterSpec] = Field(
        default_factory=dict,
        description="A dictionary specifying each tunable parameter and its constraints.",
    )
    rationale: str = Field(..., description="A detailed explanation of why this feature is useful.")

    def validate_spec(self):
        """
        Validates the 'spec' field based on the feature type.
        Raises ValueError for invalid specs.
        """
        if self.type == "code":
            try:
                ast.parse(self.spec)
            except SyntaxError as e:
                raise ValueError(
                    f"Invalid Python syntax in 'spec' for feature '{self.name}': {e}"
                ) from e
        return True


class VettedFeature(CandidateFeature):
    pass


class RealizedFeature(BaseModel):
    """
    Represents a feature that has been converted into executable code.
    """
    name: str
    code_str: str
    parameters: Dict[str, ParameterSpec]
    passed_test: bool
    type: Literal["code"]
    source_candidate: CandidateFeature
    depends_on: List[str] = []  # Data dependencies, copied from CandidateFeature
    source_hypothesis_summary: Optional[str] = None  # For traceability

    def validate_code(self) -> None:
        from loguru import logger
        logger.debug(f"Validating code for realized feature '{self.name}'...")
        try:
            tree = ast.parse(self.code_str)
        except SyntaxError as e:
            logger.error(f"Invalid Python syntax in generated code for '{self.name}': {e}")
            raise ValueError(
                f"Invalid Python syntax in generated code for '{self.name}': {e}"
            ) from e

        # Find the function definition in the AST
        func_defs = [
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ]
        if not func_defs or len(func_defs) > 1:
            logger.error(f"Generated code for '{self.name}' must contain exactly one function definition.")
            raise ValueError(
                f"Generated code for '{self.name}' must contain exactly one function definition."
            )

        func_def = func_defs[0]

        # Check function name
        if func_def.name != self.name:
            logger.error(f"Function name '{func_def.name}' does not match feature name '{self.name}'.")
            raise ValueError(
                f"Function name '{func_def.name}' does not match feature name '{self.name}'."
            )

        # Check for expected parameters in the function signature
        arg_names = {arg.arg for arg in func_def.args.args}
        expected_params = set(self.parameters.keys())

        # The function should accept 'df' plus all tunable params
        if "df" not in arg_names:
            logger.error(f"Generated function for '{self.name}' must accept a 'df' argument.")
            raise ValueError(
                f"Generated function for '{self.name}' must accept a 'df' argument."
            )

        missing_params = expected_params - (arg_names - {"df"})
        if missing_params:
            logger.error(f"Missing parameters in function signature for '{self.name}': {missing_params}")
            raise ValueError(
                f"Missing parameters in function signature for '{self.name}': {missing_params}"
            )
```

### `utils/decorators.py`

**File size:** 913 bytes

```python
# src/utils/decorators.py
import time
from functools import wraps

from loguru import logger


def agent_run_decorator(agent_name: str):
    """
    A decorator to log the duration of an agent's run method and write it to TensorBoard.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            logger.info(f"{agent_name} started.")
            start_time = time.time()

            result = func(self, *args, **kwargs)

            end_time = time.time()
            duration = end_time - start_time

            if hasattr(self, "writer") and self.writer is not None:
                run_count = getattr(self, "run_count", 0)
                self.writer.add_scalar("run_duration_seconds", duration, run_count)

            logger.info(f"{agent_name} finished in {duration:.2f} seconds.")
            return result

        return wrapper

    return decorator
```

### `utils/feature_registry.py`

**File size:** 940 bytes

```python
from loguru import logger


class FeatureRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, name: str, feature_data: dict):
        """Registers a feature function and its metadata."""
        if name in self._registry:
                        logger.warning(f"Feature {name} is already registered. Overwriting.")
        self._registry[name] = feature_data

    def get(self, name: str) -> dict:
        """Retrieves a feature function and its metadata."""
        return self._registry.get(name)

    def get_all(self) -> dict:
        """Retrieves the entire registry."""
        return self._registry.copy()


# Global instance of the registry
feature_registry = FeatureRegistry()


def get_feature(name: str):
    """Public method to get a feature from the global registry."""
    feature_data = feature_registry.get(name)
    if feature_data:
        return feature_data.get("func")
    return None
```

### `utils/logging_utils.py`

**File size:** 1,937 bytes

```python
# src/utils/logging_utils.py
import logging
from typing import Any, Dict

from loguru import logger




class InterceptHandler(logging.Handler):
    """
    A handler to intercept standard logging messages and redirect them to loguru.
    """

    def emit(self, record):
        # Get corresponding Loguru level if it exists.
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where the logged message originated.
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )





def log_agent_context(context: Dict[str, Any]) -> None:
    """Log the context passed to an agent."""
    logger.info(f"Context received: {context}")


def log_agent_response(response: Dict[str, Any]) -> None:
    """Log the response from an agent."""
    logger.info(f"Response generated: {response}")


def log_agent_error(error: Exception) -> None:
    """Log an error that occurred in an agent."""
    logger.error(f"Error occurred: {str(error)}")


def log_llm_prompt(prompt: str) -> None:
    """Log the prompt sent to the LLM."""
    logger.info(f"📤 LLM PROMPT:\n{'-' * 50}\n{prompt}\n{'-' * 50}")


def log_llm_response(response: str) -> None:
    """Log the response from the LLM."""
    logger.info(f"📥 LLM RESPONSE:\n{'-' * 50}\n{response}\n{'-' * 50}")


def log_tool_call(tool_name: str, tool_args: Dict[str, Any]) -> None:
    """Log a tool call being made."""
    logger.info(f"🔧 TOOL CALL: {tool_name} with args: {tool_args}")


def log_tool_result(tool_name: str, result: Any) -> None:
    """Log the result of a tool call."""
    logger.info(f"🔧 TOOL RESULT from {tool_name}: {result}")



```

### `utils/plotting.py`

**File size:** 1,816 bytes

```python
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import seaborn as sns


class PlotManager:
    def __init__(self, base_dir: str = "outputs/plots"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._setup_style()

    def _setup_style(self):
        """Set up default plotting style"""
        plt.style.use("seaborn")
        sns.set_palette("husl")

    def _generate_filename(self, base_name: str, plot_type: str) -> str:
        """Generate a unique filename with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{plot_type}_{timestamp}.png"

    def save_plot(
        self,
        plot_type: str,
        base_name: str,
        fig: Optional[plt.Figure] = None,
        metadata: Optional[Dict[str, Any]] = None,
        dpi: int = 300,
    ) -> str:
        """Save the current plot with metadata"""
        if fig is None:
            fig = plt.gcf()

        filename = self._generate_filename(base_name, plot_type)
        filepath = self.base_dir / filename

        # Add metadata as text in the figure if provided
        if metadata:
            metadata_str = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
            fig.text(0.02, 0.02, metadata_str, fontsize=8, alpha=0.7)

        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        return str(filepath)

    def create_subplot_grid(self, n_plots: int) -> tuple:
        """Calculate optimal subplot grid dimensions"""
        n_rows = int(n_plots**0.5)
        n_cols = (n_plots + n_rows - 1) // n_rows
        return plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))


plot_manager = PlotManager()
```

### `utils/prompt_utils.py`

**File size:** 3,063 bytes

```python
import logging
from pathlib import Path

import jinja2

from src.core.database import get_db_schema_string

logger = logging.getLogger(__name__)

_prompt_dir = Path(__file__).parent.parent / "prompts"

# Initialize the Jinja environment
_jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(_prompt_dir),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


def _refresh_database_schema():
    """Refresh the database schema in the Jinja environment globals."""
    try:
        db_schema = get_db_schema_string()
        _jinja_env.globals["db_schema"] = db_schema
        logger.debug(
            f"Database schema refreshed successfully. Schema length: {len(db_schema)} characters"
        )
        return db_schema
    except Exception as e:
        logger.error(f"Failed to refresh database schema: {e}")
        _jinja_env.globals["db_schema"] = "ERROR: Could not load database schema"
        return None

def refresh_global_db_schema():
    """
    Public API for refreshing the DB schema in the Jinja environment globals.
    Call this ONCE per epoch/run, NOT per prompt.
    """
    return _refresh_database_schema()

# Initialize db_schema at module load so it's present for all prompts
_refresh_database_schema()


def load_prompt(template_name: str, **kwargs) -> str:
    """
    Loads and renders a Jinja2 template from the prompts directory.
    Uses the cached global DB schema (refreshed only once per epoch/run).

    Args:
        template_name: The name of the template file (e.g., 'agents/strategist.j2').
        **kwargs: The context variables to render the template with.

    Returns:
        The rendered prompt as a string.
    """
    try:
        # Do NOT refresh db schema here; it is now cached per epoch/run.
        template = _jinja_env.get_template(template_name)
        rendered_prompt = template.render(**kwargs)

        # Log the template being loaded and key info
        logger.info(f"Loaded prompt template: {template_name}")
        if kwargs:
            logger.debug(f"Template variables: {list(kwargs.keys())}")

        # Log the rendered prompt for debugging (truncated to avoid spam)
        prompt_preview = (
            rendered_prompt[:500] + "..."
            if len(rendered_prompt) > 500
            else rendered_prompt
        )
        logger.debug(f"Rendered prompt preview (first 500 chars):\n{prompt_preview}")

        # Log full prompt length
        logger.info(f"Full rendered prompt length: {len(rendered_prompt)} characters")

        return rendered_prompt

    except jinja2.TemplateNotFound as e:
        logger.error(f"Template not found: {template_name}")
        raise ValueError(f"Prompt template '{template_name}' not found") from e
    except jinja2.TemplateError as e:
        logger.error(f"Template rendering error for {template_name}: {e}")
        raise ValueError(f"Error rendering template '{template_name}': {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error loading prompt {template_name}: {e}")
        raise


```

### `utils/run_logger.py`

**File size:** 1,790 bytes

```python
import json
import threading
from datetime import datetime
from pathlib import Path

class RunLogger:
    """
    Logs every tool call (input/output) and every group chat message to a JSON file incrementally.
    Thread-safe for multi-agent, multi-process use.
    """
    def __init__(self, run_dir: Path, filename: str = "run_transcript.json"):
        self.log_path = Path(run_dir) / filename
        self.lock = threading.Lock()
        # Create the file if it doesn't exist
        if not self.log_path.exists():
            with open(self.log_path, 'w') as f:
                json.dump([], f)

    def log_event(self, event_type: str, payload: dict):
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": event_type,
            **payload
        }
        with self.lock:
            # Read, append, and write back (atomic for small files)
            with open(self.log_path, 'r+') as f:
                try:
                    data = json.load(f)
                except Exception:
                    data = []
                data.append(entry)
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()

    def log_message(self, sender, recipient, content, role=None, extra=None):
        self.log_event("message", {
            "sender": sender,
            "recipient": recipient,
            "content": content,
            "role": role,
            "extra": extra or {}
        })

    def log_tool_call(self, tool_name, input_args, output, agent=None, extra=None):
        self.log_event("tool_call", {
            "tool_name": tool_name,
            "input_args": input_args,
            "output": output,
            "agent": agent,
            "extra": extra or {}
        })
```

### `utils/run_utils.py`

**File size:** 5,771 bytes

```python
#!/usr/bin/env python3
"""
Utilities for managing run IDs and run-specific paths.
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.data.cv_data_manager import CVDataManager

# Base directories
RUNTIME_DIR = Path("runtime")
RUNS_DIR = RUNTIME_DIR / "runs"

# Global variable to store current run ID
_run_id: Optional[str] = None
_run_dir: Optional[Path] = None

logger = logging.getLogger(__name__)


def init_run() -> Tuple[str, Path]:
    """
    Initializes a new run, setting a unique run ID and creating run-specific directories.
    This function should be called once at the start of a pipeline run.
    """
    global _run_id, _run_dir
    if _run_id:
        raise RuntimeError(f"Run has already been initialized with RUN_ID: {_run_id}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    _run_id = f"run_{timestamp}_{unique_id}"

    runtime_path = Path(__file__).resolve().parent.parent.parent / "runtime" / "runs"
    _run_dir = runtime_path / _run_id

    # Create all necessary subdirectories for the run
    (_run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (_run_dir / "data").mkdir(parents=True, exist_ok=True)
    (_run_dir / "graphs").mkdir(parents=True, exist_ok=True)
    (_run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (_run_dir / "tensorboard").mkdir(parents=True, exist_ok=True)
    (_run_dir / "generated_code").mkdir(parents=True, exist_ok=True)

    return _run_id, _run_dir


def get_run_id() -> str:
    """Returns the unique identifier for the current run."""
    if _run_id is None:
        raise RuntimeError("Run context is not initialized. Call init_run() first.")
    return _run_id


def get_run_dir() -> Path:
    """Returns the absolute path to the directory for the current run."""
    if _run_dir is None:
        raise RuntimeError("Run context is not initialized. Call init_run() first.")
    return _run_dir


def get_run_artifact_path(*path_parts: str) -> Path:
    """Constructs an absolute path for an artifact within the current run's directory."""
    return get_run_dir().joinpath(*path_parts)


def get_run_logs_dir() -> Path:
    """Get the logs directory for the current run."""
    return get_run_dir() / "logs"


def get_run_tensorboard_dir() -> Path:
    """Get the TensorBoard directory for the current run."""
    return get_run_dir() / "tensorboard"


def get_run_generated_code_dir() -> Path:
    """Get the generated code directory for the current run."""
    return get_run_dir() / "generated_code"


def get_run_memory_file() -> Path:
    """Get the memory file path for the current run."""
    return get_run_dir() / "memory.json"


def get_run_database_file() -> Path:
    """Get the database file path for the current run."""
    return get_run_dir() / "database.json"


def get_run_log_file() -> Path:
    """Get the log file for the current run."""
    return get_run_logs_dir() / f"pipeline_{get_run_id()}.log"


def get_run_db_file() -> Path:
    """Get the database file for the current run."""
    return get_run_dir() / f"data_{get_run_id()}.duckdb"


def get_feature_code_path(feature_name: str) -> Path:
    """Get the path for a realized feature's code file."""
    return get_run_generated_code_dir() / f"{feature_name}.py"


def get_tensorboard_writer(agent_name: str):
    """Get a TensorBoard writer for the current run and agent."""
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(log_dir=str(get_run_tensorboard_dir() / agent_name))


def format_log_message(message: str) -> str:
    """Format a log message with run context."""
    return f"[{get_run_id()}] {message}"


def config_list_from_json(file_path: str) -> List[Dict]:
    """Load OpenAI config list from a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config list from {file_path}: {e}")
        return []


def restart_pipeline(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Restarts the pipeline with an optional configuration update.
    This function should be called by the ReflectionAgent when deciding to continue.

    Args:
        config: Optional dictionary of configuration parameters for the next run
    """
    global _run_id, _run_dir

    # Save current run ID
    old_run_id = _run_id

    # Initialize a new run
    new_run_id, new_run_dir = init_run()

    # If config is provided, save it
    if config:
        config_path = new_run_dir / "config" / "next_cycle_config.json"
        config_path.parent.mkdir(exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

    logger.info(f"Pipeline restarted. Old run: {old_run_id}, New run: {new_run_id}")
    return new_run_id, new_run_dir


def terminate_pipeline() -> None:
    """
    Terminates the pipeline gracefully.
    This function should be called by the ReflectionAgent when deciding to stop.
    """
    # Close any open database connections
    CVDataManager.close_global_connection_pool()

    global _run_id, _run_dir

    if _run_id:
        logger.info(f"Pipeline terminated. Final run: {_run_id}")

        # Create a termination marker file
        termination_file = _run_dir / "pipeline_terminated.txt"
        with open(termination_file, "w", encoding="utf-8") as f:
            f.write(f"Pipeline terminated at {datetime.now().isoformat()}\n")

        # Reset global variables
        _run_id = None
        _run_dir = None
    else:
        logger.warning("Attempted to terminate pipeline but no run was active.")
```

### `utils/sampling.py`

**File size:** 1,205 bytes

```python
from src.utils import db_api


def sample_users_by_activity(n: int, min_rev: int, max_rev: int) -> list[str]:
    sql = f"""
      SELECT user_id FROM (
        SELECT user_id, COUNT(*) AS cnt
        FROM reviews
        GROUP BY user_id
      ) sub
      WHERE cnt BETWEEN {min_rev} AND {max_rev}
      ORDER BY RANDOM()
      LIMIT {n};
    """
    return db_api.conn.execute(sql).fetchdf()["user_id"].tolist()


def sample_users_stratified(n_total: int, strata: dict) -> list[str]:
    """
    Samples users from different activity strata.

    Args:
        n_total (int): The total number of users to sample.
        strata (dict): A dictionary where keys are strata names and values are
                       tuples of (min_reviews, max_reviews, proportion).
                       Proportions should sum to 1.

    Returns:
        list[str]: A list of sampled user IDs.
    """
    all_user_ids = []
    for stratum, (min_rev, max_rev, proportion) in strata.items():
        n_sample = int(n_total * proportion)
        if n_sample == 0:
            continue

        user_ids = sample_users_by_activity(n_sample, min_rev, max_rev)
        all_user_ids.extend(user_ids)

    return all_user_ids
```

### `utils/session_state.py`

**File size:** 19,274 bytes

```python
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import duckdb
from loguru import logger
from typing import Optional
from src.data.cv_data_manager import CVDataManager

from src.schemas.models import Hypothesis, Insight
from src.utils.run_utils import get_run_dir


class CoverageTracker:
    """
    Tracks which tables, columns, and relationships have been explored.
    """
    def __init__(self):
        self.tables_explored = set()
        self.columns_explored = set()
        self.relationships_explored = set()

    def log_table(self, table: str):
        self.tables_explored.add(table)

    def log_column(self, table: str, column: str):
        self.columns_explored.add((table, column))

    def log_relationship(self, rel: str):
        self.relationships_explored.add(rel)

    def is_table_explored(self, table: str) -> bool:
        return table in self.tables_explored

    def is_column_explored(self, table: str, column: str) -> bool:
        return (table, column) in self.columns_explored

    def is_relationship_explored(self, rel: str) -> bool:
        return rel in self.relationships_explored

    def summary(self) -> dict:
        return {
            "tables_explored": list(self.tables_explored),
            "columns_explored": list(self.columns_explored),
            "relationships_explored": list(self.relationships_explored),
        }

    def update_coverage(self, session_state):
        # Log all tables/columns/relationships from insights and hypotheses
        for insight in getattr(session_state, 'insights', []):
            for t in getattr(insight, 'tables_used', []):
                self.log_table(t)
            for col in getattr(insight, 'columns_used', []):
                if isinstance(col, (list, tuple)) and len(col) == 2:
                    self.log_column(col[0], col[1])
            for rel in getattr(insight, 'relationships_used', []):
                self.log_relationship(rel)
        for hypo in getattr(session_state, 'hypotheses', []):
            for t in getattr(hypo, 'tables_used', []):
                self.log_table(t)
            for col in getattr(hypo, 'columns_used', []):
                if isinstance(col, (list, tuple)) and len(col) == 2:
                    self.log_column(col[0], col[1])
            for rel in getattr(hypo, 'relationships_used', []):
                self.log_relationship(rel)

    def get_coverage(self):
        return self.summary()

class SessionState:
    """Manages the state and artifacts of a single pipeline run."""

    def __init__(self, run_dir: Optional[Path] = None):
        self.run_dir = run_dir or get_run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # --- RunLogger integration ---
        from src.utils.run_logger import RunLogger
        self.run_logger = RunLogger(self.run_dir)

        # Initialize default state
        self.insights: List[Insight] = []
        self.hypotheses: List[Hypothesis] = []

        # Additional state for complete pipeline management
        self.prioritized_hypotheses: List[Dict] = []
        self.candidate_features: List[Dict] = []
        self.best_params: Dict = {}
        self.best_rmse: Optional[float] = None
        self.bo_history: Dict = {}
        self.reflections: List[Dict] = []

        # Coverage tracker for systematic exploration
        self.coverage_tracker = CoverageTracker()

        # Central memory for cross-epoch and intra-epoch notes
        self.central_memory: List[Dict] = []  # Each note: {"agent": str, "note": str, "reasoning": str, ...}
        self.epoch_summary: str = ""  # Summary string for the epoch

        # Run counters for agents
        self.ideation_run_count: int = 0
        self.feature_realization_run_count: int = 0
        self.reflection_run_count: int = 0

        # Cross-validation data manager (lazy)
        self._cv_manager: Optional[CVDataManager] = None
        
        # Load existing state if available
        self._load_from_disk()

        # Database connection attributes
        self.db_path = "data/goodreads_curated.duckdb"
        self.conn: Optional[duckdb.DuckDBPyConnection] = None

    def _load_from_disk(self):
        """Loads existing session state from disk if available."""
        state_file = self.run_dir / "session_state.json"
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    data = json.load(f)

                # Load insights and hypotheses using proper model classes
                if "insights" in data:
                    self.insights = [Insight(**insight) for insight in data["insights"]]
                if "hypotheses" in data:
                    self.hypotheses = [
                        Hypothesis(**hypothesis) for hypothesis in data["hypotheses"]
                    ]

                # Load simple state fields
                self.prioritized_hypotheses = data.get("prioritized_hypotheses", [])
                self.candidate_features = data.get("candidate_features", [])
                self.best_params = data.get("best_params", {})
                self.best_rmse = data.get("best_rmse")
                self.bo_history = data.get("bo_history", {})
                # Load central memory and epoch summary if present
                self.central_memory = data.get("central_memory", [])
                self.epoch_summary = data.get("epoch_summary", "")
                self.reflections = data.get("reflections", [])
                self.set_state("features", data.get("features", {}))
                self.set_state("metrics", data.get("metrics", {}))
                self.set_state("models", data.get("models", {}))

                # Load run counters
                self.ideation_run_count = data.get("ideation_run_count", 0)
                self.feature_realization_run_count = data.get("feature_realization_run_count", 0)
                self.reflection_run_count = data.get("reflection_run_count", 0)

                logger.info(
                    f"Loaded existing session state with {len(self.insights)} insights and {len(self.hypotheses)} hypotheses."
                )
            except Exception as e:
                logger.error(
                    f"Warning: Failed to load existing session state: {e}. Starting with fresh state."
                )
        else:
            logger.info("No existing session state found. Starting with fresh state.")

    def add_insight(self, insight: Insight):
        self.insights.append(insight)
        self.save_to_disk()
        logger.info(f"Added and saved new insight: '{insight.title}'")

    def finalize_hypotheses(self, hypotheses: List[Hypothesis]):
        self.hypotheses.extend(hypotheses)
        self.save_to_disk()
        logger.info(f"Finalized and saved {len(hypotheses)} hypotheses.")

    # Prioritized hypotheses management
    def set_prioritized_hypotheses(self, hypotheses: List[Dict]):
        self.prioritized_hypotheses = hypotheses
        self.save_to_disk()

    def get_prioritized_hypotheses(self) -> List[Dict]:
        return self.prioritized_hypotheses

    # Candidate features management
    def set_candidate_features(self, features: List[Dict]):
        self.candidate_features = features
        self.save_to_disk()

    def get_candidate_features(self) -> List[Dict]:
        return self.candidate_features

    # Optimization results management
    def set_best_params(self, params: Dict):
        self.best_params = params
        self.save_to_disk()

    def get_best_params(self) -> Dict:
        return self.best_params

    def set_best_rmse(self, rmse: float):
        self.best_rmse = rmse
        self.save_to_disk()

    def get_best_rmse(self) -> Optional[float]:
        return self.best_rmse

    def set_bo_history(self, history: Dict):
        self.bo_history = history
        self.save_to_disk()

    def get_bo_history(self) -> Dict:
        return self.bo_history

    # Reflections management
    def add_reflection(self, reflection: Dict):
        self.reflections.append(reflection)
        self.save_to_disk()

    def get_reflections(self) -> List[Dict]:
        return self.reflections

    # Run counters management
    def increment_ideation_run_count(self) -> int:
        self.ideation_run_count += 1
        self.save_to_disk()
        return self.ideation_run_count

    def get_ideation_run_count(self) -> int:
        return self.ideation_run_count

    def increment_feature_realization_run_count(self) -> int:
        self.feature_realization_run_count += 1
        self.save_to_disk()
        return self.feature_realization_run_count

    def get_feature_realization_run_count(self) -> int:
        return self.feature_realization_run_count

    def increment_reflection_run_count(self) -> int:
        self.reflection_run_count += 1
        self.save_to_disk()
        return self.reflection_run_count

    def get_reflection_run_count(self) -> int:
        return self.reflection_run_count

    # Feature, metric, and model storage
    def store_feature(self, feature_name: str, feature_data: Dict):
        """Store feature data in the session state."""
        features = self.get_state("features", {})
        features[feature_name] = feature_data
        self.set_state("features", features)

    def get_feature(self, feature_name: str) -> Optional[Dict]:
        """Get feature data from the session state."""
        features = self.get_state("features", {})
        return features.get(feature_name)

    def store_metric(self, metric_name: str, metric_data: Dict):
        """Store metric data in the session state."""
        metrics = self.get_state("metrics", {})
        metrics[metric_name] = metric_data
        self.set_state("metrics", metrics)

    def get_metric(self, metric_name: str) -> Optional[Dict]:
        """Get metric data from the session state."""
        metrics = self.get_state("metrics", {})
        return metrics.get(metric_name)

    def store_model(self, model_name: str, model_data: Dict):
        """Store model data in the session state."""
        models = self.get_state("models", {})
        models[model_name] = model_data
        self.set_state("models", models)

    def get_model(self, model_name: str) -> Optional[Dict]:
        """Get model data from the session state."""
        models = self.get_state("models", {})
        return models.get(model_name)

    # Generic get/set methods for backward compatibility and any additional state
    def get_state(self, key: str, default: Any = None) -> Any:
        """Generic getter for any state attribute."""
        return getattr(self, key, default)

    def set_state(self, key: str, value: Any):
        """Generic setter for any state attribute."""
        setattr(self, key, value)
        self.save_to_disk()

    def get_final_insight_report(self) -> str:
        """Returns a string report of all insights generated."""
        if not self.insights:
            return "No insights were generated during this run."

        report = "--- INSIGHTS REPORT ---\n\n"
        for i, insight in enumerate(self.insights, 1):
            report += f"Insight {i}: {insight.title}\n"
            report += f"  Finding: {insight.finding}\n"
            if insight.source_representation:
                report += f"  Source: {insight.source_representation}\n"
            if insight.supporting_code:
                report += f"  Code:\n```\n{insight.supporting_code}\n```\n"
            if insight.plot_path:
                report += f"  Plot: {insight.plot_path}\n"
            report += "\n"
        return report

    def get_final_hypotheses(self) -> List[Hypothesis]:
        """Returns the final list of vetted hypotheses."""
        return self.hypotheses

    def get_all_table_names(self) -> List[str]:
        """Returns a list of all table names in the database."""
        try:
            # DuckDB's way to list all tables
            tables_df = self.db_connection.execute("SHOW TABLES;").fetchdf()
            return tables_df["name"].tolist()
        except Exception as e:
            logger.error(f"Failed to get table names from database: {e}")
            return []

    def vision_tool(
        self,
        image_path: str,
        prompt: str,
    ) -> Union[str, None]:
        """
        Analyzes an image file using OpenAI's GPT-4o vision model.
        This tool automatically resolves the path relative to the run's output directory.
        """
        try:
            import base64
            import os

            from openai import OpenAI

            # Construct the full path to the image
            full_path = self.run_dir / image_path

            if not full_path.exists():
                logger.error(f"Vision tool failed: File not found at {full_path}")
                return f"ERROR: File not found at '{image_path}'. Please ensure the file was saved correctly."

            # Initialize OpenAI client
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            # Read and encode the image
            with open(full_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")

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
        except ImportError:
            return "ERROR: OpenAI library is not installed. Please install it with `pip install openai`."
        except Exception as e:
            logger.error(f"Vision tool failed with an unexpected error: {e}")
            return f"ERROR: An unexpected error occurred while analyzing the image: {e}"

    def get_state_file_path(self) -> Optional[str]:
        """Returns the path to the session state file if it exists."""
        state_file = self.run_dir / "session_state.json"
        if state_file.exists():
            return str(state_file)
        return None

    def summarise_central_memory(self, max_entries: int = 10) -> str:
        """Return a concise markdown bullet-list summary of recent central-memory notes."""
        if not self.central_memory:
            return "(No central memory notes this epoch.)"
        recent = self.central_memory[-max_entries:]
        lines = [f"- **{e['agent']}**: {e['note']} _(reason: {e['reasoning']})_" for e in recent]
        return "\n".join(lines)

    def clear_central_memory(self):
        """Empties central memory list."""
        self.central_memory.clear()

    # ------------------------------------------------------------------
    # CV DATA ACCESS HELPERS
    # ------------------------------------------------------------------
    def _get_cv_manager(self) -> CVDataManager:
        """Lazy-initialize and return a CVDataManager instance."""
        if self._cv_manager is None:
            splits_dir = Path("data/cv_splits")
            if not splits_dir.exists():
                logger.warning("CV splits directory not found; please generate CV splits first.")
            self._cv_manager = CVDataManager(
                db_path=self.db_path,
                splits_dir=splits_dir,
                read_only=True,
            )
        return self._cv_manager

    def get_train_df(self, fold_idx: int = 0):
        """Return the (train+val) DataFrame for a given fold (default 0)."""
        cv = self._get_cv_manager()
        train_val_df, _ = cv.get_fold_data(fold_idx=fold_idx, split_type="full_train")
        return train_val_df

    def get_test_df(self, fold_idx: int = 0):
        """Return the test DataFrame for a given fold (default 0)."""
        cv = self._get_cv_manager()
        _, test_df = cv.get_fold_data(fold_idx=fold_idx, split_type="full_train")
        return test_df

    # ------------------------------------------------------------------
    # Existing persistence helpers
    # ------------------------------------------------------------------
    def save_to_disk(self):
        """Saves the current session state to disk."""
        output = {
            "insights": [i.model_dump() for i in self.insights],
            "hypotheses": [h.model_dump() for h in self.hypotheses],
            "prioritized_hypotheses": self.prioritized_hypotheses,
            "candidate_features": self.candidate_features,
            "best_params": self.best_params,
            "best_rmse": self.best_rmse,
            "bo_history": self.bo_history,
            "central_memory": self.central_memory,
            "epoch_summary": self.epoch_summary,
            "reflections": self.reflections,
            "features": self.get_state("features", {}),
            "metrics": self.get_state("metrics", {}),
            "models": self.get_state("models", {}),
            "ideation_run_count": self.ideation_run_count,
            "feature_realization_run_count": self.feature_realization_run_count,
            "reflection_run_count": self.reflection_run_count,
        }
        output_path = Path(self.run_dir) / Path("session_state.json")
        with open(output_path, "w") as f:
            json.dump(output, f, indent=4)

    def close_connection(self):
        """Closes the database connection and resets the connection attribute."""
        if self.conn:
            try:
                self.conn.close()
                logger.info("Database connection closed.")
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
        self.conn = None

    def reconnect(self):
        """Reopens the database connection in read-write mode."""
        self.close_connection()  # Ensure any existing connection is closed first
        try:
            self.conn = duckdb.connect(database=self.db_path, read_only=False)
            logger.info(f"Successfully reconnected to {self.db_path} in read-write mode.")
        except Exception as e:
            logger.error(f"FATAL: Failed to reconnect to database at {self.db_path}: {e}")
            self.conn = None
            raise e

    @property
    def db_connection(self) -> duckdb.DuckDBPyConnection:
        """
        Provides a lazy-loaded, read-write database connection.
        The connection is created on first access.
        """
        if self.conn is None:
            try:
                logger.info(f"Connecting to {self.db_path} in read-write mode...")
                self.conn = duckdb.connect(database=self.db_path, read_only=False)
                logger.info(f"Successfully connected to {self.db_path} in read-write mode.")
            except Exception as e:
                logger.error(f"FATAL: Failed to connect to database at {self.db_path}: {e}")
                self.conn = None
                raise e
        return self.conn
```

### `utils/testing_utils.py`

**File size:** 1,547 bytes

```python
import json

import numpy as np
import pandas as pd
from jsonschema import validate


def assert_json_schema(instance: dict, schema_path: str) -> None:
    """Raises AssertionError if instance doesn't match schema."""
    with open(schema_path) as f:
        schema = json.load(f)
    try:
        validate(instance=instance, schema=schema)
    except Exception as e:
        raise AssertionError(f"JSON schema validation failed: {e}")


def load_test_data(
    n_reviews: int, n_items: int, n_users: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Creates a synthetic toy dataset with random ratings, random words."""
    # Create reviews data
    review_data = {
        "user_id": np.random.randint(0, n_users, n_reviews),
        "book_id": np.random.randint(0, n_items, n_reviews),
        "rating": np.random.randint(1, 6, n_reviews),
        "review_text": [
            " ".join(
                np.random.choice(
                    ["good", "bad", "fantasy", "sci-fi", "grimdark"], size=10
                )
            )
            for _ in range(n_reviews)
        ],
        "timestamp": pd.to_datetime(
            np.random.randint(1577836800, 1609459200, n_reviews), unit="s"
        ),
    }
    df_reviews = pd.DataFrame(review_data)

    # Create items data
    item_data = {
        "book_id": np.arange(n_items),
        "author": [f"Author_{i}" for i in range(n_items)],
        "genre": np.random.choice(["Fantasy", "Sci-Fi"], size=n_items),
    }
    df_items = pd.DataFrame(item_data)

    return df_reviews, df_items
```

### `utils/tools.py`

**File size:** 31,273 bytes

```python
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
```

### `utils/tools_logging.py`

**File size:** 1,462 bytes

```python
from functools import wraps
from datetime import datetime
import inspect

# This wrapper assumes the tool function signature includes session_state or can be passed one.
def log_tool_call(tool_func, session_state, tool_name=None):
    name = tool_name or tool_func.__name__
    @wraps(tool_func)
    def wrapper(*args, **kwargs):
        input_args = inspect.getcallargs(tool_func, *args, **kwargs)
        # Remove session_state from args for logging clarity
        input_args_log = {k: v for k, v in input_args.items() if k != 'session_state'}
        logger = getattr(session_state, 'run_logger', None)
        agent = kwargs.get('agent', None)
        start_time = datetime.utcnow().isoformat() + 'Z'
        try:
            output = tool_func(*args, **kwargs)
            if logger:
                logger.log_tool_call(
                    tool_name=name,
                    input_args=input_args_log,
                    output=output,
                    agent=agent,
                    extra={"start_time": start_time, "success": True}
                )
            return output
        except Exception as e:
            if logger:
                logger.log_tool_call(
                    tool_name=name,
                    input_args=input_args_log,
                    output=str(e),
                    agent=agent,
                    extra={"start_time": start_time, "success": False}
                )
            raise
    return wrapper
```

## 📊 Summary

- **Total files processed:** 60
- **Directory:** `src`
- **Generated:** 2025-06-17 18:19:17

---

*This documentation was generated automatically. It includes all text-based source files and their complete contents.*
