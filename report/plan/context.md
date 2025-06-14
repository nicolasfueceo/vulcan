# context.md

## Project Purpose and Motivation

**VULCAN** is an agentic, modular, and extensible framework for automated data analysis, feature engineering, and recommendation system prototyping. It is designed to orchestrate a team of AI agents—each with specialized roles—to autonomously explore, analyze, and engineer features from large-scale tabular datasets (notably, Goodreads book data). The system aims to accelerate insight discovery, automate the end-to-end ML workflow, and provide robust baselines and evaluation for recommendation and ranking tasks.

VULCAN’s core motivation is to:
- Enable autonomous, multi-agent exploration and reasoning over complex data.
- Automate the feature ideation, realization, and evaluation pipeline.
- Provide a reproducible, extensible platform for ML experimentation and benchmarking.

---

## High-Level Architecture

The project is organized around the following architectural pillars:

### 1. **Agent-Based Orchestration**
- **Orchestrator (`orchestrator.py`)**: The central entry point, managing the flow of the pipeline and coordinating agent interactions.
- **Agents (`agents/`)**: Modular agent definitions, grouped by team (e.g., `discovery_team`, `strategy_team`). Each agent is instantiated with its own prompt, LLM config, and tool access.

### 2. **Pipeline Stages**
- **Insight Discovery**: Agents collaborate to find patterns and generate hypotheses from the curated data.
- **Feature Ideation**: Specialized agents brainstorm and validate candidate features based on discovered insights.
- **Feature Realization**: Candidate features are engineered, validated, and materialized into feature matrices.
- **Strategy & Evaluation**: Baseline models (e.g., SVD, DeepFM, Featuretools) are run and compared, with results scored and reported.

### 3. **Data Handling and Utilities**
- **Data Curation**: External scripts curate and normalize raw Goodreads data into a DuckDB database.
- **Feature Matrix Construction (`data/feature_matrix.py`)**: Handles the transformation of engineered features into model-ready matrices.
- **Session State (`utils/session_state.py`)**: Central state management for orchestrator runs.
- **Schema Management (`schemas/`)**: Pydantic models for all major data artifacts.

### 4. **Configuration and Prompts**
- **Config (`config/`)**: Centralized settings, logging, and LLM configuration management.
- **Prompts (`prompts/`)**: Jinja2 templates for agent system messages, ensuring modular and reusable agent instructions.

### 5. **Evaluation and Reporting**
- **Baselines (`baselines/`)**: Scripts and utilities for running and evaluating ML baselines.
- **Evaluation (`evaluation/`)**: Scoring and metrics calculation.
- **Reporting (`report/`)**: Output artifacts, experiment summaries, and documentation.

---

## Key Components and Their Roles

### `orchestrator.py`
- The main pipeline driver. Loads environment/config, initializes agents, manages the group chat (via `SmartGroupChatManager`), and coordinates all pipeline stages from insight discovery to evaluation.

### `agents/`
- **discovery_team/**: Defines agents for quantitative analysis, pattern seeking, and data representation.
- **strategy_team/**: Agents focused on selecting, combining, and evaluating features and strategies.

### `orchestration/`
- Contains legacy orchestration modules for ideation, insight, realization, and strategy. These are now largely superseded by the unified orchestrator but provide useful reference logic.

### `utils/`
- **session_state.py**: Manages run state, caching, and database connections.
- **tools.py**: Implements tool functions callable by agents (e.g., SQL execution, view creation).
- **prompt_utils.py**: Loads and renders prompt templates for agents.
- **run_utils.py**: Utilities for run management and reproducibility.

### `data/`
- **cv_data_manager.py**: Handles cross-validation splits and data partitioning.
- **feature_matrix.py**: Builds feature matrices for ML models.

### `baselines/`
- Implements and runs baseline models (SVD, DeepFM, Featuretools) for benchmarking against the agent-generated pipeline.

### `schemas/`
- **models.py**: Pydantic models for features, insights, and other pipeline artifacts.
- **eda_report_schema.json**: JSON schema for EDA reports.

### `core/`
- **database.py**: Core database interaction logic.
- **llm.py**: Abstractions for LLM interaction.
- **tools.py**: Core tool definitions.

### `config/`
- **settings.py**: Centralized configuration (paths, environment, etc.).
- **log_config.py**: Logging setup.
- **tensorboard.py**: Tensorboard integration for experiment tracking.

### `prompts/`
- Modular Jinja2 templates for all agent roles and pipeline stages, enabling flexible prompt engineering.

---

## Workflow Overview

1. **Data Curation**: Raw Goodreads data is curated into a normalized DuckDB database.
2. **Orchestration Start**: The orchestrator loads configs, sets up logging, and initializes the session state.
3. **Agent Team Formation**: Specialized agents are instantiated with their own prompts and LLM configs.
4. **Insight Discovery Loop**: Agents collaborate to discover insights, generate SQL views, and propose hypotheses.
5. **Feature Ideation & Realization**: Candidate features are brainstormed, validated, and engineered.
6. **Feature Matrix Construction**: Features are materialized into matrices ready for ML modeling.
7. **Baseline and Strategy Evaluation**: Baseline models are run and scored; agent strategies are compared.
8. **Reporting**: Results, insights, and artifacts are saved for review and further iteration.

---

## Extensibility and Design Principles

- **Modularity**: Each agent, tool, and pipeline stage is decoupled and easily swappable.
- **Reproducibility**: Session state, configs, and data partitions are centrally managed for reproducible experiments.
- **Scalability**: The agent framework and tool interface are designed for extension to new domains and tasks.
- **Transparency**: Logging, reporting, and schema validation ensure traceability and debuggability.

---

## Summary

VULCAN is a sophisticated, agent-driven platform for automated data science and recommendation system prototyping. Its architecture balances automation, modularity, and extensibility, making it suitable for both research and production-grade ML pipelines. The project’s design empowers teams to rapidly iterate on feature engineering, insight discovery, and model evaluation with minimal manual intervention.
