# Project Overview: LLM-Driven Feature Engineering for Recommender Systems

## 1. Project Objectives

1. **Automated Feature Discovery via LLMs**  
   - Use a Large Language Model (LLM) to autonomously generate, mutate, and refine user/item features from Goodreads review and rating data.  
   - Evolve a population of candidate features in a progressive, generation-based loop.  
   - Leverage a reward signal that balances:
     - **Cluster separability** (e.g. silhouette score, number of clusters)  
     - **Recommendation quality** (e.g. intra-cluster RMSE or Precision@K improvements)  

2. **Warm-Start Recommendation (Intra-Cluster Models)**  
   - For each candidate feature, cluster users based on that feature alone.  
   - Train simple recommender models (e.g. LightFM, SVD, KNN) on each cluster’s user-item interactions.  
   - Evaluate resulting Precision@K, NDCG, and clustering metrics—combine them into a scalar “feature reward.”  
   - Retain top-scoring features, so that the final set produces highly separated clusters and strong within-cluster recommendations.

3. **RL-Guided Search Strategy**  
   - Benchmark a basic **ε-greedy** policy (exploration vs exploitation) for choosing **“Generate New Feature”** vs **“Mutate Existing Feature.”**  
   - Integrate more advanced bandit or tree-search strategies (e.g. **UCB**, **Thompson Sampling**, or **MCTS with priors**) to improve how quickly the search converges on high-reward features.  
   - Maintain ε-greedy as a baseline for comparison.  

4. **Cold-Start User Assignment via Bayesian Q&A (Future Phase)**  
   - Once clusters and their archetypal feature profiles are finalized, design a short conversational questionnaire to assign a brand-new user to the most likely cluster.  
   - Use a Bayesian updating framework: ask questions that maximize expected information gain about which cluster the user belongs to.  
   - Quickly identify the user’s segment, then serve recommendations from that cluster’s model.

5. **Rigorous Evaluation & Benchmarking**  
   - During the **iterative search**, perform a **fast-mode evaluation** on a small subsample (e.g. 10% of users, 20% of items) to score candidate features without full-scale training.  
   - At the **final generation**, run a **full evaluation** on the entire train/test split—train multiple recommenders (LightFM with more epochs, Surprise SVD, KNN) and compute Precision@5/10, Recall@K, NDCG@K, and clustering metrics.  
   - Compare advanced RL strategies (UCB, Thompson) against ε-greedy on:
     - Cumulative reward curve over generations  
     - Convergence speed (generations to reach X% of max reward)  
     - Final cluster separation and recommendation lift  
   - Save all results, tree structures, and metrics for post-hoc analysis and frontend visualization.

6. **Logging & Visualization**  
   - Use **TensorBoard** (or the internal logging framework) to track:
     - **Reward vs. Generation**  
     - **Optimal Number of Clusters vs. Generation**  
     - **Silhouette Score vs. Generation**  
     - **Recommendation Metrics (Precision@K, NDCG@K) vs. Generation**  
   - Export the evolving **feature tree** as JSON (nodes/edges with IDs, parent IDs, scores, action types) so the frontend can render it.  
   - Produce metrics JSON or CSV logs for easy plotting of historical performance in the custom frontend.

## 2. Data Flow & Pipeline Components

1. **Data Preparation**  
   - Input: Goodreads reviews and ratings stored in SQLite databases (`train.db`, `val.db`, `test.db`).  
   - Split Strategy:
     - **Design Set** (e.g. 80% of users) for feature engineering and clustering.
       - Nested CV:  
         - Outer K-fold splits: Partition users into “TrainFE” vs “ValClusters.”  
         - Inner M-fold splits (within TrainFE) for training vs validating candidate features.  
     - **Hold-Out Set** (20% of users) reserved for final cold-start and end-to-end evaluation.

2. **LLM-Based Feature Generation (FeatureAgent)**  
   - **Prompt Construction:**  
     - Provide the LLM with:  
       - Task description (recommender objective)  
       - Metadata for existing features (names, descriptions)  
       - A small sample of user-item interactions/aggregates (serialized).  
       - Action context: either “GENERATE_NEW” or “MUTATE_EXISTING” with parent feature details.  
     - Instruct the LLM to output a Python snippet (or structured JSON) that computes a new feature as a pandas Series (indexed by user).  
   - **Candidate Mutation:**  
     - When “MUTATE_EXISTING” is chosen, the agent receives the code or description of the worst-performing feature and is told to produce a variant (e.g., apply a transformation or combine it with another simple statistic).  
   - **Code Repair Loop:**  
     - If the LLM’s generated code fails to execute, automatically invoke the LLM again (with the original prompt + error traceback) to fix syntax or logic, up to a configured number of attempts.

3. **Feature Execution & Fast-Mode Evaluation**  
   - **Fast-Mode Sampling:**  
     - Randomly subsample a fraction of users and items per feature (fractions configurable).  
     - Execute the candidate’s code on that mini-dataset to produce feature values.  
   - **Mini-Cluster & Mini-Recommendation:**  
     - Cluster sampled users using the candidate feature (e.g. KMeans on 1D feature values).  
     - Train a quick LightFM model (5–10 epochs) on the sampled training interactions.  
     - Evaluate Precision@5 (or NDCG@5) on held-out sample interactions.  
   - **Fast Reward Computation:**  
     - Compute normalized silhouette on the clustered subsample.  
     - Combine silhouette + mini-Precision@K (weighted) into a scalar “fast reward.”  
     - Return this score to the orchestrator for ranking in that generation.

4. **Progressive Evolution Orchestrator (RL Loop)**  
   - **Initialize:**  
     - Start with a **seed population** of newly generated features (N features).  
     - Evaluate them in fast mode; keep top M features (population size).  
   - **Each Generation:**  
     1. **Action Selection:** Use an RL strategy (ε-greedy, UCB, or Thompson Sampling) to decide:
        - **Generate_New:** ask the LLM for a completely new feature (prompt with existing population context).  
        - **Mutate_Existing:** select one parent feature (e.g. worst performer or via tournament) and prompt the LLM to mutate it.  
     2. **Candidate Creation:** Receive 1–B candidate features from the LLM.  
     3. **Fast-Mode Scoring:** For each candidate, run fast-mode evaluation and compute fast reward.  
     4. **Selection:** Add candidates to the population, sort by reward, discard lowest until population size is M.  
     5. **Log & Visualize:** Record generation metrics (best_reward, avg_reward, cluster_count, silhouette) to TensorBoard and JSON logs. Append any new tree nodes to `tree.json`.  
   - **Terminate:** After `max_generations` or convergence, switch to **full evaluation** for the **final population**.  

5. **Full Evaluation (Final Phase)**  
   - Use **entire dataset** (train/test splits) with no subsampling.  
   - Train multiple recommenders per cluster:
     - **LightFM** (e.g. 50 epochs, grid-tuned hyperparameters).  
     - **Surprise SVD** (e.g. 50 factors, 20 epochs).  
     - **KNN-Basic** (e.g. user–user or item–item).  
   - Evaluate on test set:  
     - Precision@5/10, Recall@K, NDCG@5/10.  
     - Compute final silhouette and cluster count.  
     - Derive “full reward” or “overall score” for each feature.  
   - Identify the **best feature** (or top K features) from the final population according to full reward.

6. **Cold-Start Questionnaire (Future Work)**  
   - For each of the final clusters, identify “archetypal” features (small set of features that best distinguish that cluster).  
   - Build a **Bayesian Q&A engine** that:
     - Maintains a prior distribution over cluster membership for a new user.  
     - At each question turn, selects the question expected to yield the highest information gain (given archetypal feature values).  
     - Updates posterior belief as user answers (mapping answer to a feature value estimate).  
     - Stops when confidence in a cluster assignment exceeds a threshold, then assigns the user to that cluster’s recommender.

## 3. Configuration & Extensibility

- **Configurable RL Strategy** (`config.rl.strategy`):  
  - `"epsilon"` (baseline)  
  - `"ucb"` (Upper Confidence Bound)  
  - `"thompson"` (Thompson Sampling)  
  - Optionally: `"mcts_prior"` (MCTS with heuristic or learned priors)  

- **Reward Weights** (`config.reward.weights`):  
  - `silhouette_weight` (e.g. 0.3)  
  - `cluster_count_weight` (e.g. 0.3)  
  - `rmse_weight` (e.g. 0.4)  

- **Fast-Mode Sampling Rates** (`config.evaluation.fast_user_fraction`, `fast_item_fraction`):  
  - Default: 0.10 (users), 0.20 (items).  

- **Logging Flags** (`config.logging`):  
  - `use_tensorboard: true/false`  
  - `save_tree_json: true/false`  
  - `log_dir: "experiments/logs"`  

- **Ablation Modes** (`config.ablation`):  
  - Toggle individual components for ablations (e.g. override `rl.strategy`, adjust `reward.weights`, swap clustering method, etc.).  

- **Output Structure**  
