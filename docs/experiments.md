# VULCAN Experimentation Protocol for Goodreads Dataset

## 1. Introduction

This document outlines the experimental protocol for evaluating the VULCAN system on the **Goodreads dataset**. The primary focus of this experiment is to assess the system's ability to autonomously generate valuable features from rich, unstructured text data, specifically user reviews.

VULCAN employs a **Progressive Feature Evolution** strategy, guided by a UCB1 (Upper Confidence Bound) algorithm. This algorithm manages a portfolio of feature engineering "agents," each representing a distinct strategy for feature creation or refinement. The goal is to determine if this agentic, adaptive approach can discover features that improve the performance of a downstream recommendation model.

## 2. Research Questions

-   **RQ1: Text-based Feature Efficacy:** Can VULCAN autonomously generate impactful features from raw review text that improve recommendation quality? How does the UCB-guided approach compare to a simpler, non-adaptive baseline?
-   **RQ2: Agent Dynamics on Textual Data:** What is the selection pattern of agents when the primary source of new information is text? Do agents like `LLMRowAgent` become more valuable?
-   **RQ3: Value of Iterative Refinement:** Do the `RefineTopAgent` and `ReflectAndRefineAgent` contribute significantly to improving text-based features over multiple generations?
-   **RQ4: Cost-Benefit Analysis:** What is the computational cost (LLM tokens, time) versus the performance gain (e.g., nDCG lift) of the features discovered by VULCAN?

## 3. Experimental Setup

### 3.1. Dataset

-   **Source:** The **Goodreads dataset**, containing user-book interactions, book metadata, and a large corpus of user-written reviews.
-   **Key Challenge:** The core of the feature engineering task is to leverage the `review_text` column to create features that capture nuanced user preferences and item characteristics.
-   **Data Splitting:** The data is structured with **outer and inner cross-validation folds**.
    -   **Outer Folds:** Used for creating distinct train/test splits to ensure that our final evaluation is robust and not specific to a single data partition. The experiment will run across all available outer folds.
    -   **Inner Folds:** Used within a given outer fold for the VULCAN system's internal validation. The `validation` set for feature scoring is drawn from the inner fold, while the `training` set is used for the evaluator model.
    -   **Hold-out Test Set:** For each outer fold, a hold-out test set is used **only** for the final evaluation of the feature set discovered by VULCAN. It is never seen during the evolutionary process.

### 3.2. Models

-   **Feature Evaluator Model:** A lightweight model (e.g., LightFM) will be used internally by VULCAN to rapidly score candidate features based on their performance on the inner-fold validation set.
-   **Final Recommendation Model:** A more powerful model (e.g., XGBoost or a neural network) will be trained on the full outer-fold training set plus the discovered features. Its performance will be measured on the hold-out test set.

### 3.3. Experimental Conditions

1.  **Baseline (`IdeateOnly`):** Only the `IdeateNewAgent` and `LLMRowAgent` are used, without UCB selection. This simulates a non-adaptive generation process.
2.  **VULCAN (`UCB_AllAgents`):** The full system with the UCB1 algorithm managing all agents, allowing it to adapt its strategy based on agent performance.

### 3.4. Hyperparameters

The experiment will be governed by a dedicated configuration file (e.g., `goodreads_large_experiment.yaml`). Key parameters will include:
-   **LLM Model:** `gpt-4-turbo` (for its strong text comprehension).
-   **Max Generations:** `100`
-   **Population Size:** `50`
-   **UCB Exploration Constant (c):** `2.0`

## 4. Evaluation Metrics

### 4.1. Primary Metrics (Recommendation Performance on Test Set)

-   **nDCG@10**
-   **MAP@10**
-   **Recall@10**

### 4.2. Secondary Metrics (System Behavior)

-   **Agent Selection Frequency:** To understand which strategies the UCB algorithm favors.
-   **Reward Curves:** To track the performance of generated features over generations.
-   **Feature Type Distribution:** The proportion of `code-based` vs. `llm_based` features in the final population.
-   **LLM Costs:** Token usage and monetary cost per experiment run.

## 5. Methodology for Unbiased Results

-   **Strict Data Separation:** The rigorous use of outer and inner folds ensures the test data remains completely unseen during feature generation and selection.
-   **Averaging Across Folds:** Final performance metrics will be averaged across all outer folds to provide a more generalizable result.
-   **Multiple Random Seeds:** Each experimental condition will be run with multiple random seeds to account for stochasticity in the LLM and selection processes. Results will be reported with mean and standard deviation.

This protocol provides a clear framework for a robust and insightful evaluation of VULCAN's capabilities on a challenging, text-rich dataset. 