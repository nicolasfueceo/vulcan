# Source Code Documentation

Generated on: 2025-06-16 16:37:04

This document contains the complete source code structure and contents of the `src` directory.

## 📁 Full Directory Structure

```
├── .gitignore
├── .ruff_cache/
│   ├── .gitignore
│   ├── 0.11.13/
│   │   ├── 10035001097010849136
│   │   ├── 10209213205140466974
│   │   ├── 11256956069432148776
│   │   ├── 1231301740235175813
│   │   ├── 12404207031273964348
│   │   ├── 12986171790077239247
│   │   ├── 13043826984468259098
│   │   ├── 13450541447674714635
│   │   ├── 14130817377459621375
│   │   ├── 14195369555485797494
│   │   ├── 14303290313017948516
│   │   ├── 15715095191921689679
│   │   ├── 1639093997070980232
│   │   ├── 16701771181186239481
│   │   ├── 16789539735357863146
│   │   ├── 17033856446843569390
│   │   ├── 18047837004551977747
│   │   ├── 18076770827875221276
│   │   ├── 2172303405870675659
│   │   ├── 3096963027186623868
│   │   ├── 3102837809023614707
│   │   ├── 3427590981331677085
│   │   ├── 3495669085465520408
│   │   ├── 450221922813433944
│   │   ├── 5320217578922298852
│   │   ├── 644445844811209309
│   │   ├── 711651472226615739
│   │   ├── 8332078308367618719
│   │   ├── 8475453963848076840
│   │   ├── 8584604848142013301
│   │   ├── 901234538733284022
│   │   ├── 9394952767601150040
│   │   └── 9713079930055746817
│   └── CACHEDIR.TAG
├── .windsurf/
│   └── rules/
│       ├── thesis-audience.md
│       └── writing-thesis.md
├── README.md
├── config/
│   └── OAI_CONFIG_LIST.json
├── data/
│   ├── .gitkeep
│   ├── __init__.py
│   ├── cache_metadata.py
│   ├── curated_reviews_partitioned/
│   │   └── part-0.parquet
│   ├── generate_cv_splits.py
│   ├── processed/
│   │   └── cv_splits/
│   │       ├── cv_folds.json
│   │       └── cv_summary.json
│   ├── root/
│   │   └── fuegoRecommender/
│   ├── splits/
│   │   ├── cold_start_users.json
│   │   ├── cv_folds.json
│   │   ├── cv_summary.json
│   │   └── sample_users.json
│   └── sql_views/
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
│   └── code_mindmap.mermaid
├── generate_src_docs.py
├── generated_prompts/
│   ├── CandidateFeature.schema.json
│   ├── DataRepresenter.txt
│   ├── Hypothesis.schema.json
│   ├── Hypothesizer.txt
│   ├── PatternSeeker.txt
│   ├── QuantitativeAnalyst.txt
│   ├── base_analyst.txt
│   ├── data_representer.txt
│   ├── engineer_agent.txt
│   ├── feature_engineer.txt
│   ├── feature_realization.txt
│   ├── feature_realization_agent.txt
│   ├── optimization_agent.txt
│   ├── pattern_seeker.txt
│   ├── quantitative_analyst.txt
│   ├── reflection_agent.txt
│   └── strategist_agent.txt
├── pipeline_test_output.txt
├── pyproject.toml
├── report/
│   ├── .DS_Store
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
├── requirements.txt
├── scripts/
│   ├── __pycache__/
│   ├── check_lightfm_openmp.py
│   ├── create_interactions_view.sql
│   ├── dump_agent_prompts.py
│   ├── dump_json_schemas.py
│   ├── inspect_cv_splits.py
│   ├── setup_views.py
│   ├── test_feature_optimization_pipeline.py
│   ├── test_feature_realization.py
│   ├── test_finalize_hypotheses.py
│   ├── test_hypothesizer_agent.py
│   ├── test_optimization_end_to_end.py
│   ├── test_schema_validation.py
│   ├── test_strategy_team.py
│   └── test_view_persistence.py
├── src/
│   ├── __pycache__/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── __pycache__/
│   │   ├── discovery_team/
│   │   │   ├── __pycache__/
│   │   │   └── insight_discovery_agents.py
│   │   └── strategy_team/
│   │       ├── __pycache__/
│   │       ├── evaluation_agent.py
│   │       ├── feature_auditor_agent.py
│   │       ├── feature_realization_agent.py
│   │       ├── optimization_agent_v2.py
│   │       ├── reflection_agent.py
│   │       └── strategy_team_agents.py
│   ├── baselines/
│   │   ├── __pycache__/
│   │   ├── feature_engineer/
│   │   │   └── featuretools_baseline.py
│   │   ├── recommender/
│   │   │   ├── deepfm_baseline.py
│   │   │   ├── popularity_baseline.py
│   │   │   ├── ranking_utils.py
│   │   │   └── svd_baseline.py
│   │   └── run_all_baselines.py
│   ├── config/
│   │   ├── __pycache__/
│   │   ├── log_config.py
│   │   ├── settings.py
│   │   └── tensorboard.py
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
│   │   └── scoring.py
│   ├── orchestration/
│   │   ├── __pycache__/
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
│   ├── report/
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
│       ├── run_utils.py
│       ├── sampling.py
│       ├── session_state.py
│       ├── testing_utils.py
│       └── tools.py
├── src_documentation.md
├── test_run_dir/
│   └── session_state.json
└── tests/
    ├── __init__.py
    ├── __pycache__/
    ├── agents/
    │   └── strategy_team/
    │       ├── __pycache__/
    │       └── test_evaluation_agent.py
    ├── conftest.py
    ├── debug_db_connection.py
    ├── evaluation/
    │   ├── __pycache__/
    │   ├── test_beyond_accuracy.py
    │   └── test_clustering.py
    ├── test_central_memory.py
    ├── test_insight_discovery.py
    ├── test_optimization_agent.py
    ├── test_optimization_end_to_end.py
    └── test_orchestrator_e2e.py
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

**File size:** 8,519 bytes

```python
# src/agents/evaluation_agent.py
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
        # --- 1. Load hold-out data ---
        data_manager = CVDataManager(
            db_path=session_state.db_path,
            splits_dir="data/processed/cv_splits",
        )
        n_folds = data_manager.get_fold_summary().get("n_folds", 1)
        full_train_df, test_df = data_manager.get_fold_data(fold_idx=n_folds-1, split_type="full_train")
        # --- 2. Generate feature matrices ---
        X_train = VULCANOptimizer._generate_feature_matrix(full_train_df, realized_features, best_params)
        X_test = VULCANOptimizer._generate_feature_matrix(test_df, realized_features, best_params)
        # --- 3. Global LightFM model ---
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
        # --- 4. Clustering and Intra-Cluster Models ---
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
        # Log the number of clusters to TensorBoard and metrics
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
        # --- 5. Beyond-Accuracy Metrics ---
        def get_recommendations(model, dataset, user_ids, k):
            # Recommend top-k for each user (returns a sparse matrix)
            recs = {}
            for i, user_id in enumerate(user_ids):
                scores = model.predict(i, np.arange(len(all_items)), user_features=None)
                top_items = np.argsort(-scores)[:k]
                rec_items = [all_items[j] for j in top_items]
                recs[user_id] = rec_items
            return recs
        # Global recommendations for beyond-accuracy
        # (Assume last trained model is global)
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
        # Cluster beyond-accuracy
        for label, user_ids in clusters.items():
            recs = get_recommendations(model, dataset, user_ids, k=10)
            cluster_metrics[label]["novelty"] = compute_novelty(recs, full_train_df)
            cluster_metrics[label]["diversity"] = compute_diversity(recs)
            cluster_metrics[label]["catalog_coverage"] = compute_catalog_coverage(recs, catalog)
        # --- 6. Logging and Artifact Saving ---
        self.writer.add_hparams(best_params, global_metrics)
        session_state.set_state("final_evaluation_metrics", {
            "global": global_metrics,
            "clusters": cluster_metrics
        })
        # Save final report
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

### `agents/strategy_team/feature_auditor_agent.py`

**File size:** 2,424 bytes

```python
import logging
from src.utils.tools import compute_summary_stats, create_plot

from src.utils.run_utils import get_run_dir

logger = logging.getLogger(__name__)

class FeatureAuditorAgent:
    """
    Audits realized features for informativeness using comprehensive statistics, plots, and vision analysis.
    """
    def __init__(self, db_path, vision_tool):
        self.db_path = db_path
        self.vision_tool = vision_tool  # Callable: vision_tool(plot_path) -> str
        self.plots_dir = get_run_dir() / "plots"
        self.plots_dir.mkdir(exist_ok=True)

    def audit_feature(self, feature_name: str) -> dict:
        """
        For a given feature (column in a realized features view/table):
        - Compute summary stats
        - Generate and save plot
        - Use vision tool to interpret plot
        - Log structured insight
        Returns a dict with stats, plot_path, vision_summary, and a boolean 'informative'.
        """
        try:
            stats_md = compute_summary_stats(feature_name)
            # Generate histogram plot for the feature
            plot_path = create_plot(f'SELECT "{feature_name}" FROM realized_features', plot_type="hist", x=feature_name, file_name=f"{feature_name}_hist.png")
            vision_summary = self.vision_tool(plot_path) if not plot_path.startswith("ERROR") else "Plot could not be generated."
            # Simple informativeness filter: feature is informative if not constant and not mostly missing
            informative = ("No data" not in stats_md and "ERROR" not in stats_md and "Missing: 0" not in stats_md)
            insight = {
                "feature": feature_name,
                "stats": stats_md,
                "plot_path": plot_path,
                "vision_summary": vision_summary,
                "informative": informative
            }
            logger.info(f"Audited feature {feature_name}: informative={informative}")
            return insight
        except Exception as e:
            logger.error(f"Failed to audit feature {feature_name}: {e}")
            return {"feature": feature_name, "error": str(e), "informative": False}

    def audit_features(self, feature_names: list) -> list:
        """
        Audits a list of features and returns a list of insight dicts.
        """
        results = []
        for feat in feature_names:
            results.append(self.audit_feature(feat))
        return results
```

### `agents/strategy_team/feature_realization_agent.py`

**File size:** 15,423 bytes

```python
# src/agents/strategy_team/feature_realization_agent.py
import json
from typing import Dict, List, Tuple

import autogen
from loguru import logger

from src.schemas.models import CandidateFeature, RealizedFeature
from src.utils.feature_registry import feature_registry
from src.utils.prompt_utils import load_prompt
from src.utils.session_state import SessionState
from src.utils.tools import execute_python
from src.utils.decorators import agent_run_decorator


class FeatureRealizationAgent:
    def __init__(self, llm_config: Dict, session_state: SessionState):
        """Initialize the feature realization agent."""
        logger.info("Initializing FeatureRealizationAgent")
        self.llm_config = llm_config
        self.session_state = session_state
        self.db_path = session_state.get_state("db_path")
        if not self.db_path:
            raise ValueError(
                "db_path not found in SessionState. It must be initialized by the Orchestrator."
            )

        self.llm_agent = autogen.AssistantAgent(
            name="FeatureRealizationAssistant",
            llm_config=self.llm_config,
            system_message=(
                "You are an expert Python programmer. You must ONLY output the Python function code matching the provided template. "
                "Do NOT include any tool directives (e.g., @UserProxy_Strategy please run ...), object/class instantiations, or extra markdown/code blocks. "
                "Fill in ONLY the logic section marked in the template. Do NOT alter the function signature or imports. "
                "Your code must be clean, efficient, robust, and use only standard libraries like pandas and numpy."
            ),
        )

    @agent_run_decorator("FeatureRealizationAgent")
    def run(self) -> None:
        """
        Main method to realize candidate features from the session state, enforcing the contract-based template.
        Each candidate is realized using a strict function template, validated, and registered. Redundant/legacy code is removed.
        """
        logger.info("Starting feature realization...")
        candidate_features_data = self.session_state.get_candidate_features()
        if not candidate_features_data:
            logger.warning("No candidate features found to realize.")
            self.session_state.set_state("realized_features", [])
            self.session_state.set_state("features", {})
            return

        candidate_features = [CandidateFeature(**f) for f in candidate_features_data]
        realized_features: List[RealizedFeature] = []
        MAX_RETRIES = 2

        user_proxy = autogen.UserProxyAgent(
            name="TempProxy",
            human_input_mode="NEVER",
            code_execution_config=False,
        )

        fast_mode_sample_frac = self.session_state.get_state("fast_mode_sample_frac")
        logger.info(f"[FeatureRealizationAgent] fast_mode_sample_frac before set: {fast_mode_sample_frac}")
        self.session_state.set_state("optimizer_sample_frac", fast_mode_sample_frac)
        optimizer_sample_frac = self.session_state.get_state("optimizer_sample_frac")
        logger.info(f"[FeatureRealizationAgent] optimizer_sample_frac after set: {optimizer_sample_frac}")

        for candidate in candidate_features:
            logger.info(f"Attempting to realize feature: {candidate.name}")
            is_realized = False
            last_error = ""
            code_str = ""
            for attempt in range(MAX_RETRIES + 1):
                if attempt == 0:
                    template_kwargs = dict(
                        feature_name=candidate.name,
                        rationale=candidate.rationale,
                        depends_on=candidate.depends_on,
                        parameters=list(candidate.parameters.keys()),
                    )
                    message = load_prompt("agents/strategy_team/feature_realization_agent.j2", **template_kwargs)
                    prompt = (
                        "Your only job is to fill in the Python function template for this feature. "
                        "Do NOT add any extra markdown or explanations. Output ONLY the function code block."
                    )
                    message = f"{message}\n\n{prompt}"
                    # === DEBUG: Log everything being sent to LLM ===
                    logger.info("=== LLM CALL PAYLOAD ===")
                    logger.info(f"System message: {self.llm_agent.system_message}")
                    logger.info(f"Prompt message: {message}")
                    # Print chat history if present (should be empty for new agent)
                    if hasattr(self.llm_agent, 'chat_messages'):
                        import pprint
                        logger.info("Current chat history for llm_agent:")
                        pprint.pprint(self.llm_agent.chat_messages)
                else:
                    message = (
                        f"The previous code you wrote for the feature '{candidate.name}' failed validation with the following error:\n---\nERROR:\n{last_error}\n---\n"
                        "Please provide a corrected version of the LOGIC BLOCK ONLY to be inserted into the function."
                    )
                user_proxy.initiate_chat(self.llm_agent, message=message, max_turns=1, silent=True)
                last_message = user_proxy.last_message(self.llm_agent)
                if not last_message or "content" not in last_message:
                    last_error = "LLM response was empty or invalid."
                    code_str = ""
                    continue
                response_msg = last_message["content"]
                try:
                    code_str = response_msg.split("```python")[1].split("```", 1)[0].strip()
                except IndexError:
                    code_str = response_msg.strip()
                passed, last_error = self._validate_feature(candidate.name, code_str, candidate.parameters)
                if passed:
                    logger.success(f"Successfully validated feature '{candidate.name}' on attempt {attempt + 1}.")
                    is_realized = True
                    break
            realized = RealizedFeature(
                name=candidate.name,
                code_str=code_str,
                params=candidate.parameters,
                passed_test=is_realized,
                type=candidate.type,
                source_candidate=candidate,
            )
            realized_features.append(realized)
            if is_realized:
                self._register_feature(realized)
        # --- Correlation-based feature pruning ---
        import pandas as pd
        import numpy as np
        feature_series = {}
        for r in realized_features:
            if r.passed_test:
                try:
                    temp_namespace = {}
                    exec(r.code_str, globals(), temp_namespace)
                    func = temp_namespace[r.name]
                    dummy_df = pd.DataFrame({"user_id": [1, 2, 3], "book_id": [10, 20, 30], "rating": [4, 5, 3]})
                    s = func(dummy_df, **r.params)
                    if isinstance(s, pd.Series):
                        feature_series[r.name] = s.reset_index(drop=True)
                except Exception as e:
                    logger.warning(f"Could not compute feature '{r.name}' for correlation analysis: {e}")
        if len(feature_series) > 1:
            feature_matrix = pd.DataFrame(feature_series)
            corr = feature_matrix.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            to_drop = set()
            for col in upper.columns:
                for row in upper.index:
                    if upper.loc[row, col] > 0.95:
                        var_row = feature_matrix[row].var()
                        var_col = feature_matrix[col].var()
                        drop = row if var_row < var_col else col
                        to_drop.add(drop)
            pruned_realized = [r for r in realized_features if r.name not in to_drop]
            if to_drop:
                logger.info(f"Pruned highly correlated features: {sorted(list(to_drop))}")
        else:
            pruned_realized = realized_features
        # Save to both realized_features and features for downstream use
        self.session_state.set_state("realized_features", [r.model_dump() for r in pruned_realized])
        self.session_state.set_state("features", {r.name: r.model_dump() for r in pruned_realized if r.passed_test})
        successful_count = len([r for r in pruned_realized if r.passed_test])
        logger.info(f"Finished feature realization. Successfully realized and validated {successful_count} features after correlation pruning.")

    def _register_feature(self, feature: RealizedFeature):
        """Registers a validated feature in the feature registry."""
        logger.info(f"Registering feature '{feature.name}' in the feature registry.")
        try:
            # Compile the code string into a function object
            temp_namespace = {}
            exec(feature.code_str, globals(), temp_namespace)
            feature_func = temp_namespace[feature.name]

            if not callable(feature_func):
                raise ValueError(f"The compiled code for feature '{feature.name}' is not a callable function.")

            feature_data = {
                "type": feature.type,
                "func": feature_func,  # Store the compiled function
                "params": feature.params,
                "source_candidate": feature.source_candidate,
            }
        except Exception as e:
            logger.error(f"Failed to compile and register feature '{feature.name}': {e}")
            return  # Do not register a broken feature
        feature_registry.register(name=feature.name, feature_data=feature_data)

    def _realize_code_feature(self, candidate: CandidateFeature) -> RealizedFeature:
        """Realizes a feature based on a code spec by wrapping it in a function."""
        logger.info(f"Realizing code feature: {candidate.name}")
        param_string = ", ".join(candidate.parameters.keys())
        code_str = f"""
import pandas as pd
import numpy as np

def {candidate.name}(df: pd.DataFrame, {param_string}):
    # This feature was generated based on the spec:
    # {candidate.spec}
    try:
        return {candidate.spec}
    except Exception as e:
        # Add context to the error. The ERROR: prefix is for the validator.
        print(f"ERROR: Error executing feature '{candidate.name}': {{e}}")
        return None
"""
        return RealizedFeature(
            name=candidate.name,
            code_str=code_str,
            parameters=candidate.parameters,
            passed_test=False,  # Will be set after validation
            type=candidate.type,
            source_candidate=candidate,
        )

    def _realize_llm_feature(self, candidate: CandidateFeature) -> RealizedFeature:
        """Realizes a feature using an LLM call to generate the code."""
        logger.info(f"Realizing LLM feature: {candidate.name}")
        prompt = load_prompt(
            "realize_feature_from_spec",
            feature_name=candidate.name,
            feature_rationale=candidate.rationale,
            feature_spec=candidate.spec,
            feature_params=json.dumps(candidate.parameters, indent=2),
        )

        response = self.llm_agent.generate_reply(
            messages=[{"role": "user", "content": prompt}]
        )

        if not isinstance(response, str):
            logger.error(
                f"LLM did not return a valid string response. Got: {type(response)}"
            )
            code_str = (
                f"# LLM FAILED: Response was not a string for feature {candidate.name}"
            )
        elif "```python" in response:
            code_str = response.split("```python")[1].split("```")[0].strip()
        else:
            code_str = response  # Assume the whole response is code

        return RealizedFeature(
            name=candidate.name,
            code_str=code_str,
            params=candidate.parameters,
            passed_test=False,
            type=candidate.type or "code",
            source_candidate=candidate,
        )

    def _realize_composition_feature(
        self, candidate: CandidateFeature
    ) -> RealizedFeature:
        """Realizes a feature by composing existing features. (Simplified)"""
        logger.info(f"Realizing composition feature: {candidate.name}")
        # This is a placeholder for a much more complex logic.
        # A robust implementation would require a dependency graph and careful parameter mapping.
        logger.warning(
            f"Composition feature '{candidate.name}' uses simplified realization logic."
        )

        dep_calls = []
        all_params = {}
        for dep_name in candidate.depends_on:
            dep_feature_data = feature_registry.get(dep_name)
            if not dep_feature_data:
                raise ValueError(f"Dependency '{dep_name}' not found in registry.")

            dep_params = dep_feature_data.get("params", {})
            all_params.update({f"{dep_name}__{k}": v for k, v in dep_params.items()})
            param_arg_str = ", ".join(dep_params.keys())
            dep_calls.append(
                f"    values['{dep_name}'] = {dep_name}(df, {param_arg_str})"
            )

        dep_calls_str = "\n".join(dep_calls)

        code_str = f"""
import pandas as pd
import numpy as np

# This is a simplified composition. A real implementation would need to handle imports.

def {candidate.name}(df: pd.DataFrame, {", ".join(all_params.keys())}):
    values = {{}}
{dep_calls_str}

    # The spec is a formula like 'feat_a * feat_b'
    result = eval('{candidate.spec}', {{"np": np}}, values)
    return result
"""

        return RealizedFeature(
            name=candidate.name,
            code_str=code_str,
            params=all_params,
            passed_test=False,
            type=candidate.type,
            source_candidate=candidate,
        )

    def _validate_feature(self, name: str, code_str: str, params: Dict) -> Tuple[bool, str]:
        """
        Validates the feature's code by executing it in a sandbox.
        Returns a tuple of (bool, str) for (pass/fail, error_message).
        """
        logger.info(f"Validating feature: {name}")
        try:
            # The new prompt template ensures the generated function can handle an empty dataframe.
            param_args = ", ".join(
                [f"{key}={repr(value)}" for key, value in params.items()]
            )
            validation_code = (
                code_str
                + "\nimport pandas as pd\nimport numpy as np\n# Validation Call\n"
                + f"print({name}(pd.DataFrame(), {param_args}))"
            )

            output = execute_python(validation_code)

            if "ERROR:" in output:
                logger.warning(f"Validation failed for {name}. Error:\n{output}")
                return False, output

            logger.success(f"Validation successful for {name}")
            return True, ""
        except Exception as e:  # pylint: disable=broad-except
            error_message = f"Exception during validation for {name}: {e}"
            logger.error(error_message, exc_info=True)
            return False, str(e)
```

### `agents/strategy_team/optimization_agent_v2.py`

**File size:** 23,251 bytes

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

**File size:** 4,162 bytes

```python
# src/agents/reflection_agent.py
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

        # Gather current state
        insights = session_state.get_final_insight_report()
        hypotheses = session_state.get_final_hypotheses()
        views = session_state.get_available_views()

        # Load reflection prompt
        reflection_prompt = load_prompt(
            "agents/reflection_agent.j2",
            insights=insights,
            hypotheses=json.dumps(hypotheses, indent=2),
            views=json.dumps(views, indent=2),
        )

        # Run reflection chat
        self.user_proxy.initiate_chat(
            self.assistant,
            message=reflection_prompt,
        )

        # Get the last message from the reflection agent
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
            # Parse the response from the reflection agent
            response = json.loads(last_message_content)
            should_continue = response.get("should_continue", False)
            reasoning = response.get("reasoning", "")
            next_steps = response.get("next_steps", "")
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing reflection agent response: {e}")
            logger.error(f"Raw response: {last_message_content}")
            # Provide a default response
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

**File size:** 4,404 bytes

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
    agent_prompts = {
        "StrategistAgent": load_prompt("agents/strategy_team/strategist_agent.j2", db_schema=db_schema),
        "EngineerAgent": load_prompt("agents/strategy_team/engineer_agent.j2", db_schema=db_schema),
        "FeatureEngineer": load_prompt("agents/feature_realization.j2"),
    }

    # Define the schema for the save_candidate_features tool
    save_candidate_features_tool_schema = {
        "type": "function",
        "function": {
            "name": "save_candidate_features",
            "description": "Saves a list of candidate feature specifications. Each feature spec should be a dictionary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "candidate_features_data": {
                        "type": "array",
                        "description": "A list of candidate features, where each feature is a dictionary defining its 'name', 'description', 'dependencies', and 'parameters'.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Unique, snake_case name."},
                                "description": {"type": "string", "description": "Explanation of the feature."},
                                "dependencies": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of source column names."
                                },
                                "parameters": {
                                    "type": "object",
                                    "description": "Dictionary of tunable parameters (name: {type, description}). Empty if no params."
                                }
                            },
                            "required": ["name", "description", "dependencies", "parameters"]
                        }
                    }
                },
                "required": ["candidate_features_data"]
            }
        }
    }

    # Create agents with loaded prompts
    agents = {}
    for name, prompt in agent_prompts.items():
        current_llm_config = llm_config.copy()
        if name == "FeatureEngineer":
            current_llm_config["tools"] = [save_candidate_features_tool_schema]
        
        agents[name] = autogen.AssistantAgent(
            name=name,
            system_message=prompt,
            llm_config=current_llm_config,
        )


    # Add user proxy for code execution with faster termination condition
    user_proxy = autogen.UserProxyAgent(
        name="UserProxy_Strategy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=15,  # Increased to allow more iterations within a single chat
        is_termination_msg=lambda x: "FINAL_FEATURES" in x.get("content", ""),  # Updated termination message
        code_execution_config={"work_dir": str(get_run_dir()), "use_docker": False},
    )

    # Register save_candidate_features tool for user_proxy
    # Assume session_state will be passed in or made available at runtime
    # This is a placeholder; actual registration should occur where session_state is available
    # user_proxy.register_tool(get_save_candidate_features_tool(session_state))

    agents["user_proxy"] = user_proxy
    return agents
```

### `baselines/feature_engineer/featuretools_baseline.py`

**File size:** 6,015 bytes

```python
import featuretools as ft
import pandas as pd


def run_featuretools_baseline(
    train_df: pd.DataFrame, books_df: pd.DataFrame, users_df: pd.DataFrame, test_df: pd.DataFrame = None
) -> dict:
    # Featuretools requires nanosecond precision for datetime columns.
    # Convert all relevant columns to ensure compatibility.
    import logging
    logger = logging.getLogger("featuretools_baseline")
    def clean_datetime_columns(df):
        for col in ["date_added", "date_updated", "read_at", "started_at"]:
            if col in df.columns and str(df[col].dtype).startswith("datetime64"):
                if hasattr(df[col].dt, "tz") and df[col].dt.tz is not None:
                    df[col] = df[col].dt.tz_localize(None)
                df[col] = df[col].astype("datetime64[ns]")
        return df

    train_df = clean_datetime_columns(train_df)
    books_df = clean_datetime_columns(books_df)
    users_df = clean_datetime_columns(users_df)
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
        for k in [5, 10, 20]:
            scores = _train_and_evaluate_lightfm(
                dataset, train_df, test_interactions, user_features=user_features_train, k=k
            )
            metrics[f"precision_at_{k}"] = scores.get(f"precision_at_{k}", 0)
            metrics[f"recall_at_{k}"] = scores.get(f"recall_at_{k}", 0)
            metrics[f"hit_rate_at_{k}"] = scores.get(f"hit_rate_at_{k}", 0)
        # Beyond-accuracy metrics
        from lightfm import LightFM
        model = LightFM(loss="warp", random_state=42)
        (train_interactions, _) = dataset.build_interactions(
            [(row["user_id"], row["book_id"]) for _, row in train_df.iterrows()]
        )
        model.fit(train_interactions, user_features=user_features_train, epochs=5, num_threads=4)
        def get_recommendations(model, dataset, user_ids, k):
            recs = {}
            for i, user_id in enumerate(user_ids):
                scores = model.predict(i, np.arange(len(all_items)), user_features=None)
                top_items = np.argsort(-scores)[:k]
                rec_items = [all_items[j] for j in top_items]
                recs[user_id] = rec_items
            return recs
        global_recs = get_recommendations(model, dataset, list(feature_matrix.index), k=10)
        novelty = compute_novelty(global_recs, train_df)
        diversity = compute_diversity(global_recs)
        catalog = set(all_items)
        coverage = compute_catalog_coverage(global_recs, catalog)
        metrics.update({"novelty": novelty, "diversity": diversity, "catalog_coverage": coverage})
        logger.success(f"Featuretools+LightFM metrics: {metrics}")
        return metrics
    logger.success("Featuretools baseline finished successfully.")
    return feature_matrix
```

### `baselines/recommender/deepfm_baseline.py`

**File size:** 4,196 bytes

```python
import itertools

import pandas as pd
import torch
from deepctr_torch.inputs import SparseFeat, get_feature_names
from deepctr_torch.models import DeepFM
from loguru import logger
from sklearn.preprocessing import LabelEncoder

from .ranking_utils import calculate_ndcg, get_top_n_recommendations


def run_deepfm_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
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

    # 1. Data Preprocessing
    logger.info("Preprocessing data for DeepFM...")
    data = pd.concat([train_df, test_df], ignore_index=True)
    sparse_features = ["user_id", "book_id"]
    target = "rating"

    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 2. Define Feature Columns
    logger.info("Defining feature columns for DeepCTR...")
    feat_voc_size = {feat: data[feat].nunique() for feat in sparse_features}
    fixlen_feature_columns = [
        SparseFeat(feat, vocabulary_size=feat_voc_size[feat], embedding_dim=4)
        for feat in sparse_features
    ]
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3. Split data for training and testing
    train = data.iloc[: len(train_df)]
    test = data.iloc[len(train_df) :]
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}
    train_labels = train[target].values
    test_labels = test[target].values

    # 4. Instantiate and Train Model
    logger.info("Instantiating and training DeepFM model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeepFM(
        linear_feature_columns=linear_feature_columns,
        dnn_feature_columns=dnn_feature_columns,
        task="regression",
        device=device,
    )
    model.compile("adam", "mse", metrics=["mse"])
    model.fit(
        train_model_input,
        train_labels,
        batch_size=256,
        epochs=10,
        verbose=1,
        validation_data=(test_model_input, test_labels),
    )

    # 5. Evaluate for Ranking (NDCG@10)
    logger.info("Evaluating model for ranking (NDCG@10)...")
    all_users = data["user_id"].unique()
    all_items = data["book_id"].unique()
    all_pairs = pd.DataFrame(
        list(itertools.product(all_users, all_items)), columns=["user_id", "book_id"]
    )
    train_pairs = train[["user_id", "book_id"]].drop_duplicates()
    anti_test_df = pd.merge(
        all_pairs, train_pairs, on=["user_id", "book_id"], how="left", indicator=True
    )
    anti_test_df = anti_test_df[anti_test_df["_merge"] == "left_only"].drop(
        columns=["_merge"]
    )
    anti_test_model_input = {name: anti_test_df[name] for name in feature_names}
    anti_test_predictions = model.predict(anti_test_model_input, batch_size=256)
    anti_test_df["rating"] = anti_test_predictions
    top_n = get_top_n_recommendations(anti_test_df, n=10)
    ndcg_score = calculate_ndcg(top_n, test, k=10, batch_size=1000)
    logger.info(f"DeepFM baseline NDCG@10: {ndcg_score:.4f}")

    # 6. Evaluate for Accuracy (MSE)
    logger.info("Evaluating model on the test set...")
    predictions = model.predict(test_model_input, batch_size=256)
    import numpy as np
    mse = np.mean((test_labels - predictions) ** 2)
    rmse = np.sqrt(mse)
    logger.info(f"DeepFM baseline RMSE: {rmse:.4f}")
    metrics = {
        "mse": mse,
        "rmse": rmse,
        "ndcg@10": ndcg_score,
    }
    logger.info(f"DeepFM metrics: {metrics}")
    logger.success("DeepFM baseline finished successfully.")
    return metrics
```

### `baselines/recommender/popularity_baseline.py`

**File size:** 1,015 bytes

```python
import pandas as pd
from .ranking_utils import calculate_ndcg

def run_popularity_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, top_n: int = 10) -> dict:
    """
    Recommend the most popular items (books) in the training set to all users in the test set.
    Returns NDCG@10 and the list of most popular books.
    """
    # Compute most popular books by count of ratings in train set
    pop_books = (
        train_df.groupby('book_id')['rating'].count()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )
    # For each user in test set, recommend the same top-N popular books
    user_ids = test_df['user_id'].unique()
    recommendations = {user_id: pop_books for user_id in user_ids}

    # Prepare ground truth for NDCG
    ground_truth = (
        test_df.groupby('user_id')['book_id'].apply(list).to_dict()
    )
    ndcg = calculate_ndcg(recommendations, ground_truth, k=top_n)
    return {
        'ndcg@10': ndcg,
        'top_n_books': pop_books
    }
```

### `baselines/recommender/ranking_utils.py`

**File size:** 2,169 bytes

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
    Calculate mean NDCG@k for a set of recommendations and ground truth, processing users in batches.
    recommendations: {user_id: [rec1, rec2, ...]}
    ground_truth: {user_id: [item1, item2, ...]}
    batch_size: Number of users to process at once (to avoid OOM)
    """
    user_ids = list(recommendations.keys())
    ndcgs = []
    for i in range(0, len(user_ids), batch_size):
        batch_users = user_ids[i:i+batch_size]
        for user_id in batch_users:
            recs = recommendations[user_id]
            gt = ground_truth.get(user_id, [])
            if not gt:
                continue
            ideal_dcg = sum([1.0 / np.log2(j + 2) for j in range(min(len(gt), k))])
            dcg = 0.0
            for j, rec in enumerate(recs[:k]):
                if rec in gt:
                    dcg += 1.0 / np.log2(j + 2)
            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
            ndcgs.append(ndcg)
    return float(np.mean(ndcgs)) if ndcgs else 0.0
```

### `baselines/recommender/svd_baseline.py`

**File size:** 2,351 bytes

```python
import pandas as pd
from loguru import logger
from surprise import SVD, Dataset, Reader
from surprise.accuracy import mae, rmse

from .ranking_utils import (
    calculate_ndcg,
    get_top_n_recommendations,
)


def run_svd_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Runs the SVD baseline, evaluating with RMSE, MAE, and NDCG@10.
    """
    logger.info("Starting SVD baseline...")

    # 1. Load Data
    reader = Reader(rating_scale=(1, 5))
    train_data = Dataset.load_from_df(train_df[["user_id", "book_id", "rating"]], reader)
    trainset = train_data.build_full_trainset()
    testset = list(test_df[['user_id', 'book_id', 'rating']].itertuples(index=False, name=None))

    # Build an anti-test set for generating predictions for items not in the training set
    anti_testset = trainset.build_anti_testset()

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

    # 4. Evaluate for Ranking (NDCG)
    logger.info("Evaluating model for ranking (NDCG@10)...")
    ranking_predictions = model.test(anti_testset)

    # Convert predictions to a DataFrame
    predictions_df = pd.DataFrame(
        ranking_predictions,
        columns=["user_id", "book_id", "true_rating", "rating", "details"],
    )

    top_n = get_top_n_recommendations(predictions_df, n=10)
    ndcg_score = calculate_ndcg(top_n, test_df, n=10)
    logger.info(f"SVD baseline NDCG@10: {ndcg_score:.4f}")

    # 5. Return Metrics
    # TODO: Replace the following placeholders with actual computed values
    test_rmse = None  # Replace with actual RMSE computation
    test_mae = None   # Replace with actual MAE computation
    metrics = {
        "rmse": test_rmse,
        "mae": test_mae,
        "ndcg@10": ndcg_score,
    }
    logger.info(f"SVD metrics: {metrics}")
    logger.success("SVD baseline finished successfully.")
    return metrics
```

### `baselines/run_all_baselines.py`

**File size:** 5,428 bytes

```python
import json
from pathlib import Path

from loguru import logger

from src.baselines.recommender.deepfm_baseline import run_deepfm_baseline
from src.baselines.feature_engineer.featuretools_baseline import run_featuretools_baseline
from src.baselines.recommender.svd_baseline import run_svd_baseline
from torch.utils.tensorboard import SummaryWriter

# ... (other imports remain unchanged)

from src.data.cv_data_manager import CVDataManager


def main():
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

    logger.info("Loading train/test data for fold 0...")
    # Use 'full_train' to get combined training and validation data against the test set.
    train_df, test_df = data_manager.get_fold_data(fold_idx=0, split_type="full_train")

    # Dictionary to store results from all baselines
    all_results = {}

    # 2. Run Featuretools baseline (now uses LightFM for evaluation)
    logger.info("--- Running Featuretools Baseline (LightFM evaluation) ---")
    writer = SummaryWriter("reports/tensorboard_baselines")
    try:
        featuretools_metrics = run_featuretools_baseline(train_df, books_df, users_df, test_df)
        all_results["featuretools_lightfm"] = {
            "status": "success",
            "metrics": featuretools_metrics,
        }
        logger.success(f"Featuretools+LightFM baseline completed. Metrics: {featuretools_metrics}")
        if "precision_at_10" in featuretools_metrics:
            writer.add_scalar("featuretools_lightfm/precision_at_10", featuretools_metrics["precision_at_10"])
        if "n_clusters" in featuretools_metrics:
            writer.add_scalar("featuretools_lightfm/n_clusters", featuretools_metrics["n_clusters"])
    except Exception as e:
        logger.error(f"Featuretools+LightFM baseline failed: {e}")
        all_results["featuretools_lightfm"] = {"status": "failure", "error": str(e)}

    # 3. Run SVD baseline (full dataset)
    logger.info("--- Running SVD Baseline ---")
    try:
        svd_results = run_svd_baseline(train_df, test_df)
        all_results["svd"] = {"status": "success", "metrics": svd_results}
        logger.success(f"SVD baseline completed. Metrics: {svd_results}")
        if "rmse" in svd_results:
            writer.add_scalar("svd/RMSE", svd_results["rmse"])
    except Exception as e:
        logger.error(f"SVD baseline failed: {e}")
        all_results["svd"] = {"status": "failure", "error": str(e)}

    # 5. Run Popularity baseline (to be implemented)
    logger.info("--- Running Popularity Baseline ---")
    try:
        from src.baselines.recommender.popularity_baseline import run_popularity_baseline
        popularity_results = run_popularity_baseline(train_df, test_df)
        all_results["popularity"] = {"status": "success", "metrics": popularity_results}
        logger.success(f"Popularity baseline completed. Metrics: {popularity_results}")
    except Exception as e:
        logger.error(f"Popularity baseline failed: {e}")
        all_results["popularity"] = {"status": "failure", "error": str(e)}

    # 4. Run DeepFM baseline
    logger.info("--- Running DeepFM Baseline ---")
    try:
        deepfm_results = run_deepfm_baseline(train_df, test_df)
        all_results["deepfm"] = {"status": "success", "metrics": deepfm_results}
        logger.success(f"DeepFM baseline completed. Metrics: {deepfm_results}")
    except Exception as e:
        logger.error(f"DeepFM baseline failed: {e}")
        all_results["deepfm"] = {"status": "failure", "error": str(e)}

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
    main()
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

**File size:** 1,390 bytes

```python
import subprocess
from loguru import logger
from typing import Optional

from torch.utils.tensorboard import SummaryWriter

from src.utils.run_utils import get_run_tensorboard_dir


def start_tensorboard() -> None:
    """Start TensorBoard in the background."""
    log_dir = get_run_tensorboard_dir()
    try:
        # Start TensorBoard in the background
        subprocess.Popen(
            ["tensorboard", "--logdir", str(log_dir), "--port", "6006"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        logger.warning(f"Could not start TensorBoard: {e}")


def get_tensorboard_writer() -> SummaryWriter:
    """Get a TensorBoard writer for the current run."""
    log_dir = get_run_tensorboard_dir()
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

### `core/database.py`

**File size:** 10,655 bytes

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
            conn.execute(f"ATTACH '{db_path}' AS db (READ_ONLY);")

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

**File size:** 24,516 bytes

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
    ):
        """Initialize the CV data manager with caching and connection pooling.

        Args:
            db_path: Path to the DuckDB database file
            splits_dir: Directory containing the cross-validation splits
            random_state: Random seed for reproducibility
            cache_size_mb: Size of DuckDB's memory cache in MB
            max_connections: Maximum number of database connections in the pool
            read_only: Whether the database should be opened in read-only mode
        """
        self.db_path = Path(db_path)
        self.splits_dir = Path(splits_dir)
        self.random_state = random_state
        self._cached_folds: Optional[List[Dict]] = None
        self._cv_folds = None
        self._cache_size_mb = cache_size_mb
        self.read_only = read_only

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
                sample_frac=sample_frac,
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

**File size:** 31,602 bytes

```python
import json
import os
import sys

# Ensure DB views are set up for pipeline compatibility
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import autogen
from autogen import Agent
from dotenv import load_dotenv
from loguru import logger

import scripts.setup_views
from src.agents.discovery_team.insight_discovery_agents import get_insight_discovery_agents
from src.agents.strategy_team.strategy_team_agents import get_strategy_team_agents
from src.agents.strategy_team.feature_realization_agent import FeatureRealizationAgent
from src.agents.strategy_team.optimization_agent_v2 import VULCANOptimizer
from src.config.log_config import setup_logging
from src.core.database import get_db_schema_string
from src.utils.prompt_utils import refresh_global_db_schema
from src.utils.run_utils import config_list_from_json, get_run_dir, init_run
from src.utils.prompt_utils import load_prompt
from src.utils.session_state import CoverageTracker, SessionState
from src.utils.tools import (
    cleanup_analysis_views,
    create_analysis_view,
    execute_python,
    get_add_insight_tool,
    get_finalize_hypotheses_tool,
    get_table_sample,
    run_sql_query,
    vision_tool,
    get_save_features_tool,
    get_save_candidate_features_tool,
    get_add_to_central_memory_tool,
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
        self, messages: List[Dict[str, Any]], sender: autogen.Agent, config: Optional[Dict[str, Any]] = None
    ) -> Union[str, Dict[str, Any], None]:
        """Run the chat with additional tracking and feedback mechanisms."""
        self.round_count += 1
        session_state = globals().get("session_state")

        # --- EARLY TERMINATION: If hypotheses are finalized, end the chat ---
        if session_state and hasattr(session_state, "hypotheses") and len(session_state.hypotheses) > 0:
            logger.info("Hypotheses have been finalized. Injecting TERMINATE and ending chat.")
            self.groupchat.messages.append(
                {
                    "role": "assistant",
                    "content": "TERMINATE",
                    "name": "SystemCoordinator",
                }
            )
            # Call the parent class to process the termination message and return
            return super().run_chat(self.groupchat.messages, sender, self.groupchat)

        # If we're at round 1, attach insights/discovery context
        if self.round_count == 1:
            if session_state and hasattr(session_state, "insights"):
                context_message = get_insight_context(session_state)
                if context_message:
                    self.groupchat.messages.append(
                        {
                            "role": "user",
                            "content": context_message,
                            "name": "SystemCoordinator",
                        }
                    )

        # Try to compress context if it's getting too long
        if self.round_count > 10 and self.round_count % 10 == 0:
            try:
                self.groupchat.messages = compress_conversation_context(self.groupchat.messages)
                logger.info("Applied LLM context compression at round {}", self.round_count)
            except Exception as e:
                logger.warning("Context compression failed: {}", e)

        # --- ENFORCE: Do not allow termination until finalize_hypotheses has been called ---
        # If a termination signal is detected, but session_state.hypotheses is empty, inject a reminder and prevent termination
        if session_state:
            # Detect attempted termination in the last message
            last_msg = self.groupchat.messages[-1]["content"].strip() if self.groupchat.messages else ""
            attempted_termination = any(term in last_msg for term in ["FINAL_INSIGHTS", "TERMINATE"])
            hypotheses_finalized = hasattr(session_state, "hypotheses") and len(session_state.hypotheses) > 0
            if attempted_termination and not hypotheses_finalized:
                logger.warning("Termination signal received, but no hypotheses have been finalized. Blocking termination.")
                # Inject a message that forces the Hypothesizer to act.
                self.groupchat.messages.append(
                    {
                        "role": "user",
                        "name": "SystemCoordinator",
                        "content": (
                            "A termination request was detected, but no hypotheses have been finalized. **Hypothesizer, it is now your turn to act.** "
                            "Please synthesize the team's insights and call the `finalize_hypotheses` tool."
                        ),
                    }
                )
                # Prevent actual termination this round
                return super().run_chat(self.groupchat.messages, sender, self.groupchat)
        # --- END ENFORCE ---
        # Check if we should terminate based on discovery criteria
        if session_state and self.round_count > 15 and not should_continue_exploration(session_state, self.round_count):
            if len(session_state.insights) > 0:
                logger.info(
                    "Exploration criteria met and insights found, terminating conversation"
                )
                self.groupchat.messages.append(
                    {
                        "role": "assistant",
                        "content": "TERMINATE",
                        "name": "SystemCoordinator",
                    }
                )
            else:
                logger.info(
                    "Termination criteria met, but no insights found. Forcing continuation."
                )

        # Reset agents if potential loop detected
        if self.round_count > 0 and self.round_count % 20 == 0:
            logger.warning("Potential loop detected. Resetting agents.")
            # Reset all agents to clear their memory
            for agent in self.groupchat.agents:
                # Use getattr for safer access to reset method
                reset_method = getattr(agent, "reset", None)
                if reset_method and callable(reset_method):
                    reset_method()

        # Add progress prompts to guide agents periodically
        if session_state and self.round_count > 5 and self.round_count % 15 == 0:
            progress_guidance = get_progress_prompt(session_state, self.round_count)
            if progress_guidance:
                logger.info("Adding progress guidance at round {}", self.round_count)
                self.groupchat.messages.append(
                    {
                        "role": "user",
                        "content": progress_guidance,
                        "name": "SystemCoordinator",
                    }
                )

        # --- VERBOSE LOGGING: Trace agent selection ---
        try:
            next_agent = self.groupchat.select_speaker(last_speaker=self.last_speaker, selector=self.selector)
            logger.info(f"[DEBUG] Next agent selected by groupchat.select_speaker(): {getattr(next_agent, 'name', next_agent)}")
        except Exception as e:
            logger.error(f"[DEBUG] Exception during agent selection: {e}")
        # Let the parent class handle the actual chat execution
        # Pass the GroupChat object as config for correct typing
        result = super().run_chat(messages, sender, self.groupchat)  # type: ignore
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
        is_termination_msg=lambda x: "TERMINATE" in x.get("content", "").strip(),
        code_execution_config={"work_dir": str(get_run_dir()), "use_docker": False},
    )

    assistant_agents = get_insight_discovery_agents(llm_config)
    analyst = assistant_agents["QuantitativeAnalyst"]
    researcher = assistant_agents["DataRepresenter"]
    critic = assistant_agents["PatternSeeker"]
    hypothesizer = assistant_agents["Hypothesizer"]

    # Register tools for analysis agents only
    for agent in [analyst, researcher, critic]:
        autogen.register_function(
            run_sql_query,
            caller=agent,
            executor=user_proxy,
            name="run_sql_query",
            description="Run a SQL query.",
        )
        autogen.register_function(
            get_table_sample,
            caller=agent,
            executor=user_proxy,
            name="get_table_sample",
            description="Get a sample of rows from a table.",
        )
        autogen.register_function(
            create_analysis_view,
            caller=agent,
            executor=user_proxy,
            name="create_analysis_view",
            description="Create a temporary SQL view.",
        )
        autogen.register_function(
            vision_tool,
            caller=agent,
            executor=user_proxy,
            name="vision_tool",
            description="Analyze an image.",
        )
        autogen.register_function(
            get_add_insight_tool(session_state),
            caller=agent,
            executor=user_proxy,
            name="add_insight_to_report",
            description="Saves insights to the report.",
        )
        autogen.register_function(
            execute_python,
            caller=agent,
            executor=user_proxy,
            name="execute_python",
            description="Execute arbitrary Python code for analysis, stats, or plotting.",
        )

    # Register finalize_hypotheses only for the Hypothesizer
    autogen.register_function(
        get_finalize_hypotheses_tool(session_state),
        caller=hypothesizer,
        executor=user_proxy,
        name="finalize_hypotheses",
        description="Finalize and submit a list of all validated hypotheses. This is the mandatory final step before the discovery loop can end.",
    )

    agents: Sequence[Agent] = [user_proxy, analyst, researcher, critic, hypothesizer]
    group_chat = autogen.GroupChat(
        agents=agents, messages=[], max_round=100, allow_repeat_speaker=True
    )
    manager = SmartGroupChatManager(groupchat=group_chat, llm_config=llm_config)

    logger.info("Closing database connection for agent execution...")
    session_state.close_connection()
    try:
        initial_message = (
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
    engineer = strategy_agents_with_proxy["EngineerAgent"]
    feature_engineer = strategy_agents_with_proxy["FeatureEngineer"]
    user_proxy = strategy_agents_with_proxy["user_proxy"]

    # --- Tool Registration for this specific loop ---
    # The user proxy needs access to the session_state to save features.
    # CRITICAL: Only register tools relevant to this team. `finalize_hypotheses`
    # belongs to the discovery team and was causing confusion.
    # Register the tools the agents in this group chat can use.
    # The user proxy needs access to the session_state to save features.
    user_proxy.register_function(
        function_map={
            "save_candidate_features": get_save_candidate_features_tool(session_state),
            "execute_python": execute_python,
        }
    )

    # Create the group chat with the necessary agents
    groupchat = autogen.GroupChat(
        agents=[user_proxy, strategist, engineer, feature_engineer],
        messages=[],
        max_round=1000,
        speaker_selection_method="auto",
    )

    manager = SmartGroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # Format hypotheses for the initial message
    hypotheses_json = json.dumps(
        [h.model_dump() for h in session_state.get_final_hypotheses()], indent=2
    )

    # Construct the initial message to kick off the conversation.
    # This message is a direct command to the FeatureEngineer to ensure it acts first.
    initial_message = f"""You are the FeatureEngineer. Your task is to design a set of `CandidateFeature` contracts based on the following hypotheses.

**Hypotheses:**
```json
{hypotheses_json}
```

**Your Instructions:**
1.  Analyze the hypotheses.
2.  Design a list of `CandidateFeature` contracts. Each contract must be a dictionary with `name`, `description`, `dependencies`, and `parameters`.
3.  Use the `save_candidate_features` tool to submit your designs. Your response MUST be a call to this tool.

The StrategistAgent and EngineerAgent will then review your work. Begin now.
"""

    logger.info("Reopening database connection before strategy loop...")
    session_state.reconnect()

    # --- NEW: Ensure DB connection is refreshed before strategy loop ---
    logger.info("Refreshing DB connection before strategy loop to ensure all views are visible...")
    session_state.reconnect()

    report: Dict[str, Any] = {}
    try:
        # The user_proxy initiates the chat. The `message` is the first thing said.
        user_proxy.initiate_chat(manager, message=initial_message)

        # After the chat, we check the session_state for the results.
        features = getattr(session_state, "candidate_features", [])
        hypotheses = session_state.get_final_hypotheses()

        # --- Feature Realization Step ---
        if features:
            feature_realization_agent = FeatureRealizationAgent(llm_config=llm_config, session_state=session_state)
            feature_realization_agent.run()
            realized_features = getattr(session_state, "features", {})
            realized_features_list = list(realized_features.values()) if isinstance(realized_features, dict) else realized_features
        else:
            realized_features_list = []

        report = {
            "features_generated": len(features),
            "hypotheses_processed": len(hypotheses),
            "features": features,  # candidate_features are dicts, not Pydantic models
            "realized_features": [f["name"] if isinstance(f, dict) and "name" in f else getattr(f, "name", None) for f in realized_features_list],
            "hypotheses": [h.model_dump() for h in hypotheses],
        }

    except Exception as e:
        logger.error("Strategy loop failed", exc_info=True)
        report = {"error": str(e)}
    finally:
        logger.info("Reopening database connection after strategy loop...")
        session_state.reconnect()  # Reconnect again to be safe.

    return report


def main(epochs: int = 1, fast_mode_frac: float = 0.15) -> str:
    """
    Main function to run the VULCAN agent orchestration.
    Now supports epoch-based execution. Each epoch runs the full pipeline in fast_mode (subsampled data).
    After all epochs, a final full-data optimization and evaluation is performed.
    """

    run_id, run_dir = init_run()
    logger.info(f"Starting VULCAN Run ID: {run_id}")
    setup_logging()
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
                reflection_results = run_strategy_loop(session_state, strategy_agents, llm_config)
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

        # === Optimization Step ===
        logger.info("Starting optimization step with realized features...")
        realized_features = list(session_state.features.values()) if hasattr(session_state, 'features') and session_state.features else []
        if not realized_features:
            logger.warning("No realized features found for optimization. Skipping optimization step.")
            optimization_report = "No realized features found. Optimization skipped."
        else:
            optimizer = VULCANOptimizer(session=session_state)
            try:
                optimization_result = optimizer.optimize(features=realized_features, n_trials=10, use_fast_mode=True)
                optimization_report = optimization_result.json(indent=2)
                logger.info(f"Optimization completed. Best score: {optimization_result.best_score}")
            except Exception as opt_e:
                logger.error(f"Optimization failed: {opt_e}")
                optimization_report = f"Optimization failed: {opt_e}"

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
        f"## Final Strategy Refinement Report\n{strategy_report}\n"
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

**File size:** 7,019 bytes

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

    def validate_code(self) -> None:
        """
        Validates the generated code string for correctness.
        - Parses the code to ensure it's valid Python.
        - Checks that the function name matches the feature name.
        - Verifies that all specified params are in the function signature.
        """
        try:
            tree = ast.parse(self.code_str)
        except SyntaxError as e:
            raise ValueError(
                f"Invalid Python syntax in generated code for '{self.name}': {e}"
            ) from e

        # Find the function definition in the AST
        func_defs = [
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ]
        if not func_defs or len(func_defs) > 1:
            raise ValueError(
                f"Generated code for '{self.name}' must contain exactly one function definition."
            )

        func_def = func_defs[0]

        # Check function name
        if func_def.name != self.name:
            raise ValueError(
                f"Function name '{func_def.name}' does not match feature name '{self.name}'."
            )

        # Check for expected parameters in the function signature
        arg_names = {arg.arg for arg in func_def.args.args}
        expected_params = set(self.parameters.keys())

        # The function should accept 'df' plus all tunable params
        if "df" not in arg_names:
            raise ValueError(
                f"Generated function for '{self.name}' must accept a 'df' argument."
            )

        missing_params = expected_params - (arg_names - {"df"})
        if missing_params:
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

**File size:** 17,499 bytes

```python
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import duckdb
from loguru import logger

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
        output_path = self.run_dir / "session_state.json"
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

**File size:** 29,634 bytes

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

            # Create the view
            full_sql = f"CREATE OR REPLACE VIEW {actual_name} AS ({sql_query})"
            write_conn.execute(full_sql)
            logger.info(f"[TOOL SUCCESS] Created view {actual_name} with query: {sql_query}")
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


def _execute_python_run_code(pipe, code, run_dir):
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

   

    # Provide a real DuckDB connection for the code
    conn = duckdb.connect(database=str(DB_PATH), read_only=False)
    # If in future you want to expose CV folds or other context, load and inject here.
    local_ns = {
        "save_plot": save_plot,
        "get_table_sample": get_table_sample,
        "conn": conn,
        "__builtins__": __builtins__,
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


def execute_python(code: str, timeout: int = 300) -> str:
    """
    NOTE: A pre-configured DuckDB connection object named `conn` is already provided in the execution environment. DO NOT create your own connection using duckdb.connect(). Use the provided `conn` for all SQL operations (e.g., conn.sql(...)).

    NOTE: After every major code block or SQL result, you should print the result using `print('!!!', result)` so outputs are clearly visible in logs and debugging is easier.

    Executes a string of Python code in a controlled, headless, and time-limited environment with injected helper functions.
    Args:
        code: Python code to execute
        timeout: Maximum time (seconds) to allow execution (default: 300)
    Returns:
        The stdout of the executed code, or an error message if it fails.
    """
    import multiprocessing

    run_dir = str(get_run_dir())
    parent_conn, child_conn = multiprocessing.Pipe()
    p = multiprocessing.Process(target=_execute_python_run_code, args=(child_conn, code, run_dir))
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
        with duckdb.connect(database=str(db_path), read_only=True) as conn:
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

    return save_candidate_features
```

## 📊 Summary

- **Total files processed:** 42
- **Directory:** `src`
- **Generated:** 2025-06-16 16:37:04

---

*This documentation was generated automatically. It includes all text-based source files and their complete contents.*
