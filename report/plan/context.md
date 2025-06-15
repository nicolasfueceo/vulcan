Purpose: This document provides the complete context for the VULCAN project. It serves as the single source of truth for understanding the project's motivation, its final architecture, the underlying theoretical framework, and the rigorous experimental plan designed to evaluate its contributions.

1. Project Goal & Core Philosophy
1.1. The Problem: The Feature Engineering Bottleneck
Modern recommender systems, despite their sophisticated models, are fundamentally limited by the quality of their input features. Manual Feature Engineering (FE) is slow, expensive, and dependent on human intuition, while traditional Automated Feature Engineering (AutoFE) tools often act as "black boxes," combinatorially generating thousands of features without strategic reasoning or domain understanding [cite: Planning_Report.pdf].

1.2. The VULCAN Hypothesis
Our core thesis is that a collaborative multi-agent system, designed to simulate the hypothesis-driven workflow of an expert data science team, is a superior paradigm for automated feature engineering. We hypothesize that this approach can discover higher-quality, more novel, and more interpretable features than both manual methods and monolithic automated tools.

1.3. Core Philosophy
VULCAN is built on the principle of moving from "black-box" automation to transparent, reasoning-based automation. The system's intelligence emerges from the structured, adversarial, and collaborative interactions between specialized agents, each with a distinct role in the scientific discovery process.

2. The VULCAN System Architecture (As Implemented)
The system is a procedurally-driven pipeline managed by a central orchestrator (src/orchestrator.py). It operates on a SessionState object (src/utils/session_state.py) which acts as the single source of truth for a given experimental run [cite: src_documentation.md].

2.1. Phase 1: Insight & Strategy Formation
This phase uses two autogen.GroupChat sessions to simulate a research and strategy team.

Insight Discovery Team (src/orchestration/insight.py):

Agents: A team of DataRepresenter, QuantitativeAnalyst, and PatternSeeker agents [cite: src/agents/discovery_team/insight_discovery_agents.py].

Workflow: This team performs a comprehensive EDA on the data. It uses tools like create_analysis_view to simplify data, run_sql_query for analysis, and the multi-modal vision_tool to interpret plots. Its findings are saved as structured Insight objects to the SessionState [cite: src/utils/tools.py].

Orchestration: This chat is managed by a SmartGroupChatManager that uses an LLM for context compression and provides progress-nudging prompts to keep the agents on track [cite: src/orchestrator.py].

Strategy Team (src/orchestration/strategy.py):

Agents: A team of HypothesisAgent, StrategistAgent, and EngineerAgent [cite: src/agents/strategy_team/strategy_team_agents.py].

Workflow: This team receives the Insight Report from the SessionState and engages in a structured debate to critique and refine the raw insights into a final, vetted list of Hypothesis objects, which are then saved back to the SessionState.

2.2. Phase 2: Feature Generation & Optimization

Feature Ideation (src/orchestration/ideation.py): The FeatureIdeationAgent takes the vetted hypotheses and translates them into detailed CandidateFeature specifications, including definitions for tunable hyperparameters (parameter_spec) [cite: src/schemas/models.py].

Feature Realization (src/orchestration/realization.py): The FeatureRealizationAgent is a critical component that implements a self-correction loop.

It uses a "Coder" LLM agent to write a parameterized Python function based on the candidate feature's spec and a strict Jinja2 template (src/prompts/agents/feature_realization.j2).

It validates the generated code in a secure sandbox.

If validation fails, it re-prompts the Coder agent with the failed code and the specific error message, instructing it to generate a fix. This loop continues until validation passes or retries are exhausted [cite: src/agents/strategy_team/feature_realization_agent.py].

Optimization (src/agents/strategy_team/optimization_agent_v2.py): This is a non-agentic module, the VULCANOptimizer, that implements the core bilevel optimization.

It uses the optuna library for Bayesian Optimization.

It performs robust, cross-validated evaluation using the CVDataManager [cite: src/data/cv_data_manager.py].

Its goal is to find the optimal set of feature hyperparameters that maximizes our formal objective function.

2.3. Phase 3: Evaluation & Reflection

Evaluation (src/agents/strategy_team/evaluation_agent.py): Takes the best hyperparameters from the optimization phase, retrains a final model, and evaluates it on a held-out test set to produce the final performance metrics [cite: src_documentation.md].

Reflection (src/agents/strategy_team/reflection_agent.py): This agent analyzes the results of the entire run (from the SessionState) and controls the main while loop in the orchestrator, deciding whether to terminate or initiate a new "meta-learning" cycle [cite: src_documentation.md].

3. Formal Problem Definition: Bilevel Optimization
The theoretical core of VULCAN is a bilevel optimization problem.

The Outer Loop: A search for the optimal vector of feature hyperparameters, θ 
∗
 . This is performed by the VULCANOptimizer using Bayesian Optimization.
θ 
∗
 =argmin 
θ
​
 J(θ)

The Inner Loop: For any given set of features generated with parameters θ, a recommendation model (LightFM) is trained and evaluated.

The Objective Function (J(θ)): The function to be minimized. It is a weighted sum designed to balance multiple objectives:
J(θ)=−w 
1
​
 ⋅LiftGain−w 
2
​
 ⋅ClusterQuality+w 
3
​
 ⋅FeatureComplexity

LiftGain: Measures the improvement of intra-cluster recommendation accuracy (NDCG@10) over a global baseline model.

ClusterQuality: Measures the quality of user clusters using the Silhouette Score.

FeatureComplexity: A penalty term based on the number or computational cost of the features.

4. Thesis Evaluation Plan: Baselines & Experiments
To prove the efficacy of VULCAN, we will conduct a rigorous experimental comparison.

4.1. Baseline Models (src/baselines/)

Feature Engineering Baselines:

Featuretools: Represents the state-of-the-art in non-agentic, combinatorial AutoFE.

Recommender System Baselines:

SVD (surprise): A classic, feature-agnostic collaborative filtering model to establish a performance floor.

DeepFM (deepctr-torch): A state-of-the-art deep learning model that learns feature interactions implicitly, representing an alternative paradigm to explicit FE.

Controlled Arena:

LightFM: This feature-aware model will be used as the "arena" to provide a fair, head-to-head comparison of the feature sets produced by VULCAN, Featuretools, and Manual FE.

4.2. Planned Ablation Studies

No Collaboration: Run a version of VULCAN as a linear agent pipeline to quantify the benefit of the GroupChat architecture.

Simplified Objective: Run the optimizer with a simplified objective function (e.g., only LiftGain) to prove the value of the multi-objective approach.

4.3. Planned Evaluation Metrics

Accuracy: Precision@k, Recall@k, NDCG@k.

Beyond-Accuracy: Novelty, Diversity, Catalog Coverage.

System-Level: Feature Complexity, Computational Cost, and Cluster Quality (Silhouette Score).

5. Logging & Artifacts
The pipeline is instrumented to produce a standardized set of artifacts for each run, ensuring reproducibility and providing the data needed for analysis and plotting. Key artifacts saved in runtime/runs/<run_id>/artifacts/ include:

optimization_study.pkl: The complete optuna study object.

final_report.json: A comprehensive summary of all metrics, results, and run statistics.

realized_features.json: A list of all features generated, including their code and validation status.