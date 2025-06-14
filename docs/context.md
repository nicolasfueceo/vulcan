Of course. Creating a clear, up-to-date context document is essential for maintaining project alignment, especially for an LLM that needs to understand the current state of a complex system.

Based on the latest `src_documentation.md` you provided, here is a detailed script outlining what the VULCAN system is and what remains to be built.

---
### **Project VULCAN: System Architecture & Implementation Roadmap**

**Version:** 1.5 (as of 2025-06-13)
**Audience:** Project LLM / New Collaborator
**Purpose:** To provide a comprehensive overview of the current system architecture and the remaining implementation tasks.

### **1. High-Level Project Goal**

We are building VULCAN, a self-improving, multi-agent feature engineering system designed to autonomously discover, create, validate, and optimize features for a book recommendation engine. The system's core objective is to find feature transformations that improve the quality of user clusters, which in turn enhances the accuracy of intra-cluster recommendations.

---
### **2. Current Architecture ("What We Are Building")**

The system is architected as a multi-phase, procedurally-driven pipeline managed by a central orchestrator. It leverages `autogen` for multi-agent conversations and has a strong emphasis on modularity, structured prompting, and robust state management.

**2.1. Orchestration (`src/orchestrator.py`)**

* The main entry point is `orchestrator.py`, which initializes a run-specific context and a `SessionState` object that acts as the single source of truth for the entire pipeline run.
* The orchestrator manages a primary `while True:` loop, allowing for continuous, multi-run operation, with the `ReflectionAgent` controlling the decision to loop or terminate.
* It sequentially calls modular scripts for each phase of the pipeline (`run_discovery_loop`, `run_strategy_loop`, `run_feature_ideation`, etc.).

**2.2. Phase 1: Insight & Strategy**

This phase uses two collaborative `GroupChat` sessions to simulate a research and strategy team.

* **Insight Discovery Team (`run_discovery_loop`):**
    * **Purpose:** To perform a comprehensive Exploratory Data Analysis (EDA).
    * **Agents:** A `GroupChat` consisting of `DataRepresenter`, `QuantitativeAnalyst`, and `PatternSeeker`.
    * **Workflow:** The team uses a set of granular tools (`run_sql_query`, `create_analysis_view`, `vision_tool`) to explore the database. Findings are saved as structured `Insight` objects to the `SessionState` via the `add_insight_to_report` tool.
    * **Advanced Features:** This loop is managed by a `SmartGroupChatManager` that handles advanced context compression and progress guidance to keep the conversation on track.

* **Strategy Team (`run_strategy_loop`):**
    * **Purpose:** To refine raw insights into a vetted list of testable hypotheses.
    * **Agents:** A `GroupChat` of `HypothesisAgent`, `StrategistAgent`, and `EngineerAgent`.
    * **Workflow:** The team engages in a structured debate to critique proposed hypotheses on their strategic value and technical feasibility. The final list is saved to `SessionState` via the `finalize_hypotheses` tool.

**2.3. Phase 2: Feature Generation & Optimization**

The orchestrator calls these steps sequentially after the strategy loop.

* **Feature Ideation (`orchestration/ideation.py`):**
    * Takes the vetted hypotheses from `SessionState` and uses the `FeatureIdeationAgent` to brainstorm `CandidateFeature` specifications, including tunable parameters.
* **Feature Realization (`orchestration/realization.py`):**
    * Takes the candidate features and uses the `FeatureRealizationAgent` to generate and perform a sandboxed test on the feature's Python code.
* **Optimization (`agents/strategy_team/optimization_agent_v2.py`):**
    * The `VULCANOptimizer` class uses `optuna` to perform Bayesian Optimization on the realized features' hyperparameters, evaluating them with cross-validation.

**2.4. Core Utilities**

* **Prompt Management (`src/utils/prompt_utils.py`):** The system uses a sophisticated Jinja2-based templating engine to dynamically build and load all agent prompts, separating logic from prompt content.
* **State Management (`src/utils/session_state.py`):** The `SessionState` class is the central, authoritative object for managing all artifacts within a single run.
* **Schemas (`src/schemas/models.py`):** A single file defines all Pydantic models (`Insight`, `Hypothesis`, `CandidateFeature`, etc.) for structured data exchange.

---
### **3. Remaining Tasks ("What Is Remaining to Build")**

While the high-level architecture is in place and connected, several downstream components are still using legacy patterns and need to be refactored to fully align with the modern V2 architecture.

**3.1. High Priority: Core Refactoring & Integration**

* **Unify State Management:**
    * **Task:** Refactor the `EvaluationAgent` and `ReflectionAgent`. They currently import and use `get_mem` from the legacy `src/utils/memory.py` module.
    * **Action:** Modify their `run()` methods to accept `session_state: SessionState` as an argument. All data they need must be read from this object, and their results must be written back to it. Once no agents use `memory.py`, it can be deleted.

* **Unify Orchestration Model:**
    * **Task:** The `EvaluationAgent` and `ReflectionAgent` are still designed around a `pubsub` event-driven model.
    * **Action:** Refactor them to be simple classes whose `run()` methods are called procedurally by the main `orchestrator.py`, just like the ideation and realization steps. Once complete, `src/utils/pubsub.py` can be deleted.

* **Consolidate Optimization Agents:**
    * **Task:** The project contains both `optimization_agent.py` (using `skopt`) and the more advanced `optimization_agent_v2.py` (using `optuna`).
    * **Action:** Formally deprecate and delete the older `optimization_agent.py`. Ensure the orchestrator exclusively uses the `VULCANOptimizer` from `optimization_agent_v2.py`.

* **Cleanup Redundant Code:**
    * **Task:** Delete agents and utilities that have been fully superseded.
    * **Action:** Delete `src/agents/strategy_team/reasoning_agent.py` (role is now part of the `Strategy Team` chat). Delete `src/utils/sql_views.py` as its logic is now contained within `src/utils/tools.py`.

**3.2. Future Work: Research & Meta-Learning Capabilities**

Once the core V1.4 pipeline is fully refactored and stable, the following advanced features (from our previous discussions) can be implemented.

* **Implement the `DomainExpertAgent`:**
    * **Task:** Introduce a new agent into the `Insight Discovery Team` that uses the LLM itself as a tool for qualitative analysis and injection of external domain knowledge.

* **Implement Rich Observability:**
    * **Task:** Enhance the `OptimizationAgent` and `EvaluationAgent` to log detailed metrics, hyperparameters, and artifacts to TensorBoard and a structured `final_report.md` for each run.

* **Implement the Meta-Learning Cycle:**
    * **Task:** Build the persistent `feature_store` and add the logic to the orchestrator to trigger the "meta-analysis cycle," where the `Insight Discovery Team` is re-invoked to perform EDA on the results of previous successful runs.