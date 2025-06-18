# FuegoRecommender Refactoring Thoughts

## 1. Analysis Progress & Key Findings

- **Database Module (`src/core/database.py`):**
  - Contains both DuckDB and SQLAlchemy logic.
  - Clear candidate for splitting into atomic modules: DuckDB utilities, SQLAlchemy connection manager, and schema/ingestion helpers.

- **Orchestration Modules (`src/orchestration/ideation.py`, `insight.py`, `strategy.py`, `realization.py`):**
  - All follow a similar pattern: agent setup, prompt/context loading, chat initiation, tool registration, and result extraction.
  - Significant code duplication in chat setup and result parsing.
  - Opportunity to abstract common orchestration logic into shared utilities or base classes.

- **General Pattern:**
  - Business logic (feature engineering, evaluation, DB ops) is often mixed with orchestration logic (agent management, prompt handling, chat loops).
  - This makes it harder to test, extend, and debug individual components.

## 2. Early Refactor Ideas

- **Atomicity & Modularity:**
  - Break large files into focused modules by responsibility (e.g., feature functions, DB utilities, agent orchestration, prompt utilities).
  - Group related modules into logical folders (e.g., `db/`, `features/`, `agents/`, `orchestration/`).

- **Duplication Reduction:**
  - Abstract repeated patterns in orchestration (agent setup, chat management, result extraction) into reusable functions or classes.
  - Standardize tool registration and callback mechanisms for agents.

- **Dependency Minimization:**
  - Limit cross-imports between business logic and orchestration layers.
  - Use interfaces or dependency injection where possible to decouple modules.

## 3. Next Steps
yes co
- Continue analysis of `src/agents/` and `src/baselines/`.
- Map dependencies and identify duplicate code across these and previously analyzed modules.
- Begin drafting a concrete plan for separating business logic from orchestration/agent layers.

---

## 4. Logic Simplification Opportunities

### 1. Tool Functions (`get_save_candidate_features_tool`, `get_finalize_hypotheses_tool`)
- **Observation:** Both define inner functions with similar validation, error handling, and session state updates. Repetitive try/except and logging.
- **Recommendation:**
  - Abstract common validation/session update logic into a shared utility.
  - Use decorators for error handling/logging to reduce repetition.
  - Consider a generic tool factory with schema/session targets as parameters.

### 2. `_execute_python_run_code`
- **Observation:** Deeply nested logic for plotting, code execution, and error handling. Manual namespace construction.
- **Recommendation:**
  - Factor out plot setup and namespace construction into helpers.
  - Use context managers for resource cleanup.
  - Reduce inline imports and definitions.

### 3. Error Handling Patterns
- **Observation:** Many broad try/except blocks with generic error messages.
- **Recommendation:**
  - Use specific exceptions where possible.
  - Centralize error logging.
  - Prefer context-specific error messages for easier debugging.

### 4. Control Flow & Redundancy
- **Observation:** Verbose validation loops and deduplication could use comprehensions or built-ins.
- **Recommendation:**
  - Use list comprehensions, set operations, and built-in validators.
  - Leverage Pydantic’s built-in validation more extensively.

### 5. Configuration and Constants
- **Observation:** Paths and config values are scattered/hardcoded.
- **Recommendation:**
  - Centralize configuration in a settings module or via environment variables.

### 8. Baseline Evaluation Patterns in `src/baselines/`
- **Observation:**
  - All baseline scripts follow similar structures: data prep, model training, metric calculation, and logging.
  - Repeated patterns for metric aggregation, error handling, and fold iteration.
  - Data loading and preprocessing logic is duplicated.
- **Recommendations:**
  - Abstract evaluation logic (data splitting, metric aggregation, logging) into shared utilities.
  - Use decorators/context managers for error handling and logging.
  - Centralize metric computation and reporting.
  - Factor out data preprocessing and fold handling into helpers.

---

## 5. Inter-File and Inter-Folder Dependency Map

### 1. Orchestration Modules (`src/orchestration/` & `src/orchestrator.py`)
- Depend on: `src/agents/`, `src/utils/`, `src/schemas/models`, external (`autogen`, `loguru`)
- Pattern: Each orchestration phase loads agents/prompts, manages group chat, and updates session state. Main orchestrator coordinates all phases and handles logging/cleanup.

### 2. Agent Modules (`src/agents/`)
- Depend on: `src/utils/`, external (`autogen`, `tensorboardX`, `loguru`)
- Pattern: Agent classes/factories load prompts, instantiate agents, and log results. Strategy team agents are orchestrated by orchestration modules.

### 3. Utilities (`src/utils/`)
- Depend on: Each other, external (`loguru`, `jinja2`, `pydantic`)
- Pattern: Provide logging, state management, prompt rendering, error handling, tool registration.

### 4. Data Management (`src/data/`)
- Depend on: `src/schemas/models`, `src/utils/run_utils`, external (`pandas`, `numpy`, `duckdb`)
- Pattern: `CVDataManager` is used across baselines, evaluation, and feature engineering for fold management.

### 5. Feature Engineering & Contingency (`src/contingency/`)
- Depend on: `src/utils/`, `src/baselines/`, `src/data/cv_data_manager`
- Pattern: Feature and reward functions are called by orchestration and agents, sometimes directly, sometimes via tools.

### 6. Baselines (`src/baselines/`)
- Depend on: `src/data/cv_data_manager`, `src/utils/`, external (`pandas`, `numpy`, `sklearn`, `lightfm`, `deepctr_torch`, `surprise`)
- Pattern: Baseline scripts are callable independently or via main orchestrator.

#### Cross-Cutting Utilities:
- **Session State:** Used throughout for run-tracking, feature/candidate management, persistence.
- **Prompt Loading:** Central to all agent orchestration and creation.
- **Tool Registration:** Orchestrator and agents register tools with user proxy agents for group chat.
- **Database Access:** `database.py` and `CVDataManager` used for data access, with configs in `settings.py`.

---

## [TO DO] Business Logic Separation Strategy
- After completing the analysis, outline how to:
  - Identify pure business logic (feature engineering, evaluation, DB ops)
  - Move business logic into dedicated modules
  - Keep orchestration/agent code focused on coordination and communication
  - Define clear interfaces between orchestration and business logic

---

*This file will be updated continuously throughout the refactor planning and implementation process.*

---

## Next Steps
- Continue scanning for duplication in feature engineering, orchestration, and agent coordination logic
- Prepare a detailed refactor plan based on these findings
- Present the plan for review before implementation

---

# Additional Duplicate/Near-Duplicate Patterns (Feature Engineering & Orchestration)

## 7. Feature Function Boilerplate & Patterns
**Files:** `src/contingency/functions.py`
- Nearly all feature functions:
  - Accept `df: pd.DataFrame` and `params: Dict[str, Any]`
  - Validate required columns (often with similar code)
  - Handle missing values (median fill, zeros, etc.)
  - Normalize or scale outputs (min-max, z-score, etc.)
  - Return a `pd.Series` with the same index as `df`
- **Recommendation:**
  - Create decorators/utilities for column validation, missing value handling, and output normalization/scaling.
  - Provide a feature function base template or mixin for new features.

## 8. Orchestration Agent Setup/Prompt Loading
**Files:** `src/orchestration/strategy.py`, `src/orchestration/insight.py`, `src/agents/strategy_team/strategy_team_agents.py`, `src/agents/discovery_team/insight_discovery_agents.py`
- Loading agent prompts via Jinja2 templates, passing db schema/context.
- Instantiating agents with similar code blocks.
- Registering tools/functions with user proxies in a repeated way.
- **Recommendation:** Abstract agent/prompt setup into a shared factory or helper. Centralize tool registration logic.

## 9. Validation and Filtering of Features/Hypotheses
**Files:** `src/orchestration/ideation.py`, `src/utils/tools.py`
- Functions for validating candidate features/hypotheses using Pydantic models, deduplication, and filtering.
- Repeated error handling and logging.
- **Recommendation:** Generalize validation/filtering into a single utility with configurable schema and deduplication logic.

## 10. Clustered Try/Except and Logging Blocks
**Files:** Feature functions, plotting scripts, agent run methods.
- Try/except for error resilience, with logging and fallback values.
- Sometimes nearly copy-pasted between scripts and feature functions.
- **Recommendation:** Use a decorator or context manager for error-resilient execution with logging and fallback.

## 11. Parameter Extraction and Defaulting
**Files:** Feature functions, agent configs, baseline configs.
- Extraction of hyperparameters from `params` dict with default values in many places.
- **Recommendation:** Utility/helper for parameter extraction with defaults, possibly with type enforcement.

---

# Next Actions
- Begin drafting the comprehensive refactor plan (atomic modules, folder structure, dependency minimization, logic simplification) as the next milestone.
- If specific format or example code for abstractions is needed, request before implementation.

---

# DRAFT: Comprehensive Refactor Plan (2025-06-17)

## 1. High-Level Principles
- **Atomic Modules:** Each file should serve a single responsibility (feature, utility, agent, orchestration, etc.).
- **Minimal Dependencies:** Only import what is needed; avoid cross-imports between business logic and orchestration.
- **Centralized Utilities:** Shared logic (validation, logging, error handling, parameter extraction) in `src/utils/atomic/` submodules.
- **Clear Separation:** Distinguish between business logic (feature engineering, evaluation, DB ops) and orchestration/agent coordination.
- **Testability:** Each atomic module should be independently testable.

## 2. Proposed Folder Structure

```
src/
  agents/
    base.py                 # Shared agent logic, logging, mixins
    strategy_team/
      evaluation_agent.py
      optimization_agent.py
      reflection_agent.py
      strategy_team_agents.py
    discovery_team/
      insight_discovery_agents.py
  baselines/
    recommender/
      lightfm_baseline.py
      random_forest_baseline.py
      svd_baseline.py
    run_all_baselines.py
  contingency/
    functions/
      __init__.py
      batch_01.py            # Features 1-10
      batch_02.py            # Features 11-20
      ...
      shared_utils.py        # Decorators, validation, normalization
    aggregate_hypotheses.py
    run_manual_bo.py
    run_sequential_evaluation.py
    reward_functions.py
  core/
    database.py
    tools.py
  orchestration/
    ideation.py
    insight.py
    realization.py
    strategy.py
    shared.py                # Agent/prompt factories, tool registration
  utils/
    atomic/
      validation.py
      error_handling.py
      parameter.py
      logging.py
    session_state.py
    run_logger.py
    feature_registry.py
    plotting.py
    prompt_utils.py
    run_utils.py
  schemas/
    models.py
  config/
    settings.py
```

## 3. Atomic Module Breakdown
- **Feature Functions:** Move each feature to its own file or group by batch; import shared validation/missing-value decorators.
- **Utilities:** Split large utility files into atomic units (validation, error handling, parameter extraction, logging).
- **Agent Logic:** Move shared agent setup/logging to base class; orchestration modules only coordinate, do not implement business logic.
- **Tool Factories:** Generic tool function factories for schema validation/session state update.
- **Prompt/Agent Factories:** Centralized in orchestration/shared.py for all agent instantiation/prompt loading.

## 4. Dependency Minimization
- **No cross-imports** between feature logic and orchestration/agents.
- **Orchestration** only imports agent factories, tool registration, and session state.
- **Feature engineering** only imports atomic utilities and schemas.
- **Baselines** only import atomic metric aggregation, data management, and logging.

## 5. Logic Simplification Opportunities
- Use decorators/utilities for repetitive validation, error handling, parameter extraction.
- Centralize all error logging and fallback logic.
- Provide template/mixin for new feature functions.
- Standardize agent setup and tool registration.

## 6. Folder Additions/Removals
- **Add:** `src/utils/atomic/`, `src/contingency/functions/`, `src/orchestration/shared.py`
- **Remove:** Monolithic/legacy files after migration (e.g., old `functions.py`, `tools.py`)

## 7. Next Steps
- Review and confirm/refine this plan.
- Begin phased refactor: create folders, move/split files, update imports, test atomic modules.
- Maintain continuous documentation and update `thoughts.md` after each major step.
