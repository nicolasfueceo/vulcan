Below is a comprehensive, first‐principles specification for a truly self‐improving, multi‐agent LLM‐driven feature‐engineering pipeline. The design treats each “agent” (an LLM‐backed module or a simple Python function) as a specialist in a research workflow—continuously cycling through data exploration, hypothesis generation, feature ideation, realization, evaluation, and reflection. This pipeline can be left running overnight (or longer), with the EDA agent “always on,” constantly updating its understanding and surfacing new directions. We also emphasize clean, maintainable code, strictly structured JSON exchanges, and end‐to‐end (E2E) testing points at every stage.

---

# Table of Contents

1. Objectives & High‐Level Architecture
2. Project Memory & Context Sharing
3. Agent Roles & Responsibilities
   3.1 Autogen vs. LangChain vs. Custom Orchestration
   3.2 DataThinkerAgent (Continuous EDA)
   3.3 HypothesisAgent (Prioritization & New Directions)
   3.4 FeatureIdeationAgent (Iterative Multi‐Pass Ideation)
   3.5 FeatureRealizationAgent (Code & LLM Wrappers)
   3.6 OptimizationAgent (BO / Search)
   3.7 ReflectionAgent (Critique & Next‐Step Suggestion)
4. Infinite Loop Workflow (Putting It All Together)
5. Structured JSON Schemas (All Outputs)
6. Testing & End-to-End Validation Points
7. Critical Evaluation: Why the Pipeline Might Fail & Mitigations
8. Recommended Libraries & Abstractions
9. Appendix: Example Pseudocode Skeleton

---

## 1 Objectives & High-Level Architecture

### 1.1 Core Objectives

1. **Continuously discover new, high-value features**—both code-engineered (e.g. new aggregations, embeddings) and pure LLM-inferred (e.g. “sentiment drift scores”)—from Goodreads Fantasy data, without hard-coding all templates in advance.
2. **Leverage domain knowledge** (books, genres, review language) and NLP capabilities (LLM embeddings, sentiment) to enrich both user and item representations.
3. **Iteratively refine hypotheses**: let the system “think” like a small research team, cycling between Data Exploration, Hypothesis Generation, Feature Ideation, Realization, Evaluation, and Reflection. Each agent has read-only access to a shared memory and appends its own structured outputs.
4. **Self-improving loop**: The EDA (DataThinker) agent never stops. Even as features are being tested downstream, it continues exploring updated data (e.g. new features’ outputs) to propose fresh insights.
5. **Maintain clean, modular, maintainable code** with strict JSON schemas. Every agent writes explicit JSON (no free-form text).
6. **Monitor & log** everything—metrics, intermediate results, decisions—so that at any point you can reproduce the entire pipeline, generate plots, or inspect exactly why a feature was chosen.

### 1.2 High-Level Architecture Diagram

```text
                                           ┌─────────────────────┐
                                           │ Project Memory.json │
                                           │ – eda_reports       │
                                           │ – hypotheses        │
                                           │ – feature_proposals │
                                           │ – realized_funcs    │
                                           │ – bo_history        │
                                           │ – reflections       │
                                           │ – final_eval        │
                                           └─────────────────────┘
                                                      ▲ ▲ ▲ ▲ ▲ ▲ ▲
                                                      │ │ │ │ │ │ │
  (∞) DataThinkerAgent  —>   (∗) HypothesisAgent   —>  …  —>  ReflectionAgent
   “Always running”           “New hypotheses”              “Next-steps”
    (EDA + code)                 (rank & filter)
         │                                                                         
         ▼                                                                         
  FeatureIdeationAgent  —>  FeatureRealizationAgent  —>  OptimizationAgent 
  (ideates code & llm)       (parse DSL, wrap LLM)     (BO + FM training)
         │                            │                              │
         └────────────────────────────┴──────────────────────────────┘
       (shared context: new features, new data summaries, new results)
```

* **Note (∞)**: DataThinkerAgent never stops: as new “derived features” appear in memory, it reevaluates and outputs updated EDA.
* **Note (∗)**: HypothesisAgent may be triggered whenever DataThinkerAgent appends a new “insight” that crosses a threshold (e.g. a new unigram repeats in >5% of reviews, or a novel cluster emerges).

---

## 2 Project Memory & Context Sharing

Everything hinges on a single, shared, append-only JSON file (or lightweight local database)—we’ll call it **`project_memory.json`**. Each agent:

* **Reads** from a known “namespace” inside `project_memory.json` (e.g. `"eda_reports"`, `"hypotheses"`, `"feature_proposals"`, etc.).
* **Appends** its own validated output under that namespace in a strictly typed JSON structure.

### 2.1 Suggested Folder Structure

```
project_root/
│
├── src/
│   ├── agents/
│   │   ├── data_thinker_agent.py
│   │   ├── hypothesis_agent.py
│   │   ├── feature_ideation_agent.py
│   │   ├── feature_realization_agent.py
│   │   ├── optimization_agent.py
│   │   └── reflection_agent.py
│  
│   ├── orchestrator.py
│   ├── utils/
│   │   ├── logging.py
│   │   ├── memory.py
│   │   ├── testing_utils.py
│   │   └── llm_api.py
│  
│   └── scripts/
│       └── run_pipeline.py
│
├── data/
│   ├── goodreads_reviews.csv
│   ├── goodreads_items.csv
│   └── … (any preprocessed subsets)
│
├── project_memory.json
│
├── prompts/
│   ├── data_thinker_prompt.txt
│   ├── hypothesis_prompt.txt
│   ├── ideation_pass1_prompt.txt
│   ├── ideation_pass2_prompt.txt
│   ├── reflection_prompt.txt
│   └── … (others as needed)
│
├── tests/
│   ├── test_data_thinker_e2e.py
│   ├── test_hypothesis_agent_e2e.py
│   ├── test_feature_realization.py
│   ├── test_optimization_flow.py
│   └── …
│
└── README.md
```

### 2.2 `project_memory.json` Schema (Top Level)

```json5
{
  // Stage 1 – continuous EDA outputs (list of timestamped reports)
  "eda_reports": [
    {
      "timestamp": "2025-06-10T22:15:00Z",
      "numeric_summaries": { /* see JSON schema below */ },
      "textual_insights": [ "…", "…" ],
      "cross_tab_insights": [ "…", "…" ],
      "metadata": { "rows_sampled": 50000, "bigram_counts": 1000 }
    },
    { /* next run of EDA */, … }
  ],

  // Stage 2 – aggregated hypotheses, with prioritization
  "hypotheses": [
    {
      "timestamp": "2025-06-10T22:16:10Z",
      "hypothesis": "Users who mention 'grimdark' more than 5 times in recent 50 reviews have preference for darker fantasy subgenres.",
      "priority": 5,
      "notes": "30% of sampled reviews mention 'grimdark'; novel and feasible via LLM sentiment."
    },
    { /* more */ }
  ],

  // Stage 3 – feature proposals (multi-pass). Each entry is appended as soon as it’s generated
  "feature_proposals": [
    {
      "timestamp": "2025-06-10T22:17:00Z",
      "pass": 1,
      "proposals": [
        {
          "name": "GrimdarkSentimentCount",
          "type": "code",
          "dsl": "ParamA * COUNT( review_text contains 'grimdark' in last ParamB days )",
          "chain_of_thought": "Step 1: We see 'grimdark' appears frequently... Step 2: Counting occurrences yields a measure of darkness preference...",
          "rationale": "Captures intensity of 'grimdark' mentions over recent window."
        },
        {
          "name": "DarknessAffinityScore",
          "type": "llm",
          "prompt": "Given the last 10 reviews: <USER_REVIEWS>, rate from 0 to 1 how much user prefers dark fantasy.",
          "chain_of_thought": "Step 1: 'Dark fantasy' is mentioned often in items of subgenre X... Step 2: Asking LLM yields richer nuance..."
        },
        /* 4 more proposals for pass=1 */
      ]
    },
    {
      "timestamp": "2025-06-10T22:20:30Z",
      "pass": 2,
      "proposals": [
        {
          "name": "GrimdarkSentimentCount",
          "type": "code",
          "dsl": "ParamA * COUNT( lower(review_text) contains 'grimdark' in last ParamB days )",
          "chain_of_thought": "...", 
          "rationale": "Fixed case sensitivity by using lower(...). Expected effort=2, impact=4, notes='easy to run; high signal per EDA.'"
        },
        {
          "name": "DarknessAffinityScore",
          "type": "llm",
          "prompt": "You are an expert in fantasy tone. Given these 10 reviews: <USER_REVIEWS>, output a decimal between 0 and 1 (no text) indicating how strongly the user prefers dark fantasy themes.",
          "chain_of_thought": "...",
          "rationale": "Refined prompt to direct tone toward dark fantasy; includes system instruction."
        },
        /* possibly refinements or discards */
      ]
    },
    /* further passes if any */
  ],

  // Stage 4 – realized functions summary
  "realized_functions": {
    "code": [
      {
        "name": "GrimdarkSentimentCount",
        "param_names": ["ParamA","ParamB"],
        "status": "valid",  // or "error"
        "error": ""         // error message if any
      },
      { /*…*/ }
    ],
    "llm": [
      {
        "name": "DarknessAffinityScore",
        "dependencies": ["scale"],
        "status": "valid"
      },
      { /*…*/ }
    ]
  },

  // Stage 5 – optimization & BO history
  "bo_history": {
    "trials": [
      {
        "timestamp": "2025-06-10T23:00:00Z",
        "params": {
          "GrimdarkSentimentCount_ParamA": 2.3,
          "GrimdarkSentimentCount_ParamB": 45,
          "DarknessAffinityScore_scale": 3.1,
          "K": 4,
          "fm_dim": 16,
          "lambda_w": 0.0001,
          "lambda_v": 0.00001
        },
        "rmse": 0.9123
      },
      { /* … */ }
    ],
    "best_params": { /* same structure as above */ },
    "best_rmse": 0.8942
  },

  // Stage 5 – reflections
  "reflections": [
    {
      "timestamp": "2025-06-10T23:30:00Z",
      "top_features": [
        { "feature": "GrimdarkSentimentCount", "weight": 2.1 },
        { "feature": "SomeOtherFeature", "weight": 1.5 }
      ],
      "low_value_features": [
        { "feature": "DarknessAffinityScore", "weight": 0.02 }
      ],
      "feature_interactions": [
        { "pair": ["GrimdarkSentimentCount", "AnotherFeat"], "interaction_weight": 0.12 }
      ],
      "bo_insights": [
        "ParamB for GrimdarkSentimentCount always at 180 (max) → suggests sentiment window should be entire history.",
        "DarknessAffinityScore_scale near 0 in top 5 trials → LLM prompt not adding value compared to code feature."
      ],
      "next_steps": [
        "Drop DarknessAffinityScore (its scale is near 0).",
        "Refine GrimdarkSentimentCount to use entire history or exponential time decay instead of cutoff.",
        "Propose a new template combining sentiment with subgenre transitions."
      ]
    }
  ],

  // Stage 6 – final evaluations
  "final_eval": {
    "warm": { 
      "rmse": 0.883, 
      "ndcg@10": 0.412, 
      "precision@10": 0.274 
    },
    "cold": {
      "ndcg@10": 0.275,
      "precision@10": 0.158
    },
    "cold_questions": [
      { "question": "On a scale 0–1, given these 10 reviews: <USER_REVIEWS>, how much do you prefer dark fantasy?", "type": "llm" },
      { "question": "How many books with 'grimdark' in the title or review have you read in the last year? (Answer an integer)", "type": "code" }
    ]
  }
}
```

> Note: Each section is appended over time. The pipeline never overwrites earlier entries; this preserves a full audit trail.

---

## 3 Agent Roles & Responsibilities

Each agent is a Python module (e.g. `src/agents/data_thinker_agent.py`) that:

* **Loads** the pieces of `project_memory.json` relevant to its task.
* **Runs code** or **calls an LLM** to generate structured JSON (never free-form text).
* **Validates** that its output conforms exactly to the JSON schema.
* **Appends** the new JSON blob (with a timestamp) to `project_memory.json`.

We assume a shared utility `utils/memory.py` provides functions:

```python
def load_memory() -> dict:
    # loads and returns project_memory.json

def append_to_memory(key: str, value: dict) -> None:
    # appends `value` to project_memory[key] (must be a list)
    # then writes project_memory.json back to disk
```

### 3.1 Autogen vs. LangChain vs. Custom Orchestration

* **Autogen (Microsoft)** and **LangChain** both offer agent frameworks that handle multi‐agent conversation, tool invocation, and memory. If you foresee adding many “tools” (e.g. pandas operations, plotting, caching) and want built-in retry/looping, Autogen may save boilerplate.
* **LangChain** is widely used, has built-in LLM wrappers, memory classes, and can easily connect to TensorBoard or W\&B.
* **Custom orchestration** (just a loop in `run_pipeline.py` calling each agent as a function) is simpler to debug and keeps dependencies minimal.

**Recommendation**:

* Use a **lightweight** Autogen or LangChain setup to manage LLM calls and context, but code each “Agent” as a simple Python function that returns validated JSON. This gives you the flexibility to swap back to direct API calls if needed.
* Structure each agent around a shared “Memory” object so that cross‐agent context is explicit.

### 3.2 DataThinkerAgent (Continuous EDA)

#### 3.2.1 Role & Responsibilities

1. **Run code** to compute numeric summaries (counts, distributions, correlations).
2. **Sample text** (e.g. 50–100 random reviews) and compute basic NLP insights: top unigrams/bigrams/trigrams, simple sentiment distribution (using a rule‐based or small LLM/embedding model).
3. **Ingest new data**: whenever new features (from FeatureRealization) are stored, recompute joint summaries (e.g. correlation between a newly generated “darkness count” feature and rating).
4. **Continuously think**: it never stops. Each time it’s invoked (e.g. once per hour, or triggered whenever `project_memory["realized_functions"]` changes), it reads the latest data + features and writes a new `eda_reports` entry.
5. **Produce** the following JSON schema:

```json5
{
  "timestamp": "ISO8601",
  "numeric_summaries": {
    "num_users": 120345,
    "num_items": 54321,
    "reviews_per_user": { "mean": 9.2, "median": 6, "max": 235, "min": 1 },
    "reviews_per_item": { "mean": 45.8, "median": 12, "max": 1050, "min": 1 },
    "rating_histogram": { "1": 0.05, "2": 0.10, "3": 0.15, "4": 0.35, "5": 0.35 },
    "timestamp_range": { "earliest": "2010-01-05", "latest": "2025-05-30" }
  },
  "textual_insights": [
    { "bigram": "grim dark", "frequency": 0.12 },
    { "bigram": "epic world", "frequency": 0.08 },
    { "sentiment_mean": 0.76, "sentiment_std": 0.12 }
  ],
  "cross_tab_insights": [
    "Pearson correlation between user rating variance and review count: 0.32",
    "Users who mention 'grimdark' more than 3 times have average rating 4.1 vs. 3.7 (p<0.01)",
    "Top‐10 authors by review count: ['Sanderson', 'Martin', 'Rowling', …]"
  ],
  "new_feature_correlations": [
    // computed if new features exist in memory
    { "feature": "GrimdarkSentimentCount", "corr_with_rating": 0.21 },
    { "feature": "DarknessAffinityScore", "corr_with_rating": 0.05 }
  ],
  "metadata": {
    "rows_sampled": 50000,
    "bigram_vocab_size": 1000
  }
}
```

#### 3.2.2 How to Invoke / Loop

1. **Trigger**:

   * At pipeline start.
   * Whenever `project_memory["realized_functions"]` is updated (new code or LLM feature realized).
   * (Optionally) on a fixed interval (e.g. every hour) if the pipeline runs continuously.

2. **Implementation Sketch** (`data_thinker_agent.py`):

```python
import pandas as pd
import numpy as np
from datetime import datetime
from utils.memory import load_memory, append_to_memory
from utils.llm_api import call_llm_for_text  # for sentiment or trigram frequency if needed

def compute_numeric_summaries(df_reviews, df_items):
    # Standard pandas operations
    num_users = df_reviews["user_id"].nunique()
    num_items = df_reviews["book_id"].nunique()
    reviews_per_user = df_reviews.groupby("user_id").size()
    reviews_per_item = df_reviews.groupby("book_id").size()
    rating_counts = df_reviews["rating"].value_counts(normalize=True).to_dict()
    earliest = df_reviews["timestamp"].min().strftime("%Y-%m-%d")
    latest = df_reviews["timestamp"].max().strftime("%Y-%m-%d")
    return {
        "num_users": num_users,
        "num_items": num_items,
        "reviews_per_user": {
            "mean": float(reviews_per_user.mean()),
            "median": float(reviews_per_user.median()),
            "max": int(reviews_per_user.max()),
            "min": int(reviews_per_user.min())
        },
        "reviews_per_item": {
            "mean": float(reviews_per_item.mean()),
            "median": float(reviews_per_item.median()),
            "max": int(reviews_per_item.max()),
            "min": int(reviews_per_item.min())
        },
        "rating_histogram": { str(k): float(v) for k,v in rating_counts.items() },
        "timestamp_range": { "earliest": earliest, "latest": latest }
    }

def compute_textual_insights(df_reviews, sample_size=100):
    # Sample randomly up to sample_size reviews
    sample = df_reviews.sample(n=min(len(df_reviews), sample_size), random_state=42)
    texts = sample["review_text"].tolist()
    # Basic trigram counts via direct LLM call (or small model)
    # To keep CODE simple, demonstrate a call to LLM for bigrams:
    prompt = f"Extract top 3 bigrams from these reviews: {texts[:20]}. Return JSON: [{'{'}\"bigram\":...,\"frequency\":...{'}'},...]."
    resp = call_llm_for_text(model="gpt-4", messages=[{"role":"system","content":"You are a text analyst."},
                                                       {"role":"user","content":prompt}])
    # Expect resp is a JSON text; parse it safely:
    try:
        bigrams = json.loads(resp)
    except:
        bigrams = []
    # Simple sentiment: ask LLM for average sentiment
    prompt_s = f"Given these reviews: {texts[:20]}, output a JSON {{\"sentiment_mean\":<float>, \"sentiment_std\":<float>}}."
    resp_s = call_llm_for_text(model="gpt-4", messages=[{"role":"system","content":"You are a sentiment analyzer."},
                                                        {"role":"user","content":prompt_s}])
    try:
        sent_stats = json.loads(resp_s)
    except:
        sent_stats = {"sentiment_mean": 0.0, "sentiment_std":0.0}
    # Combine
    insights = []
    for bg in bigrams:
        insights.append({ "bigram": bg["bigram"], "frequency": bg["frequency"] })
    insights.append({ "sentiment_mean": sent_stats.get("sentiment_mean",0.0),
                      "sentiment_std": sent_stats.get("sentiment_std",0.0) })
    return insights

def compute_cross_tab_insights(df_reviews, df_items):
    # Using pandas correl & simple grouping
    reviews_per_user = df_reviews.groupby("user_id").size().reset_index(name="count")
    user_rating_var = df_reviews.groupby("user_id")["rating"].var().reset_index(name="rating_var")
    merged = reviews_per_user.merge(user_rating_var, on="user_id")
    corr = float(merged["count"].corr(merged["rating_var"]))
    # Example: correlation between 'grimdark' mentions and rating
    df_reviews["mentions_grimdark"] = df_reviews["review_text"].str.lower().str.contains("grimdark")
    grimdark_counts = df_reviews.groupby("user_id")["mentions_grimdark"].sum().reset_index(name="grimdark_count")
    df_avg_rating = df_reviews.groupby("user_id")["rating"].mean().reset_index(name="avg_rating")
    merged2 = grimdark_counts.merge(df_avg_rating, on="user_id")
    corr2 = float(merged2["grimdark_count"].corr(merged2["avg_rating"]))
    # Top authors
    # Join reviews->items to see author per review
    df_merged = df_reviews.merge(df_items[["book_id","author"]], on="book_id", how="left")
    top_authors = df_merged["author"].value_counts().head(5).index.tolist()
    return [
        f"Correlation between review count and rating variance: {corr:.3f}",
        f"Users with more 'grimdark' mentions have correlation {corr2:.3f} with avg rating.",
        f"Top-5 authors by review count: {top_authors}"
    ]

def run_data_thinker(df_reviews, df_items):
    memory = load_memory()
    last_eda = memory.get("eda_reports", [])[-1] if memory.get("eda_reports") else None

    # Recompute only if new functions or hourly—simplify: always recompute
    numeric = compute_numeric_summaries(df_reviews, df_items)
    textual = compute_textual_insights(df_reviews, df_items)
    cross_tab = compute_cross_tab_insights(df_reviews, df_items)
    # If realized_functions exist, compute correlations:
    realized = memory.get("realized_functions", {}).get("code", []) + \
               memory.get("realized_functions", {}).get("llm", [])
    new_feature_correlations = []
    if realized:
        # For each realized function (just test on training set)
        for feat_entry in realized:
            feat_name = feat_entry["name"]
            # Dynamically load fn via a registry? Simplify: assume imported
            fn = globals().get(feat_name)
            if fn:
                try:
                    series = fn(df_reviews, df_items, **{p:1.0 for p in feat_entry.get("param_names", [])})
                    corr = float(series.fillna(0).corr(df_reviews.groupby("user_id")["rating"].mean()))
                    new_feature_correlations.append({ "feature": feat_name, "corr_with_rating": corr })
                except:
                    continue

    eda_report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "numeric_summaries": numeric,
        "textual_insights": textual,
        "cross_tab_insights": cross_tab,
        "new_feature_correlations": new_feature_correlations,
        "metadata": {
            "rows_sampled": min(len(df_reviews), 100),
            "bigram_vocab_size": len(textual)
        }
    }
    append_to_memory("eda_reports", eda_report)

# Entry point for the agent
if __name__ == "__main__":
    df_reviews = pd.read_csv("data/goodreads_reviews.csv", parse_dates=["timestamp"])
    df_items = pd.read_csv("data/goodreads_items.csv")
    run_data_thinker(df_reviews, df_items)
```

#### 3.2.3 Testing & E2E Goals

* **Test #1**: Run `DataThinkerAgent` on a small sample (e.g. 10k reviews, 2k items).

  * **Check**:

  1. `project_memory.json` gains one new entry under `"eda_reports"`.
  2. That entry’s `numeric_summaries` keys exist and are numerically plausible (e.g. `"num_users"` > 0).
  3. `textual_insights` is a list of ≤ 5 objects, each with `"bigram"` and `"frequency"` or sentiment stats.
  4. `cross_tab_insights` is a list of 3 strings.
  5. If no realized functions exist, `"new_feature_correlations"` is empty list.

* **Test #2**: After adding a dummy code feature (e.g. `DarkMentionsCount` realized by hand), re-run `DataThinkerAgent`:

  * **Check**:

  1. `"new_feature_correlations"` contains an entry for that feature, and the `corr_with_rating` is a float ∈ \[−1,1].
  2. `eda_reports` list length increments.

---

### 3.3 HypothesisAgent (Prioritization & New Directions)

#### 3.3.1 Role & Responsibilities

1. **Monitor** the EDA stream (`"eda_reports"`)—specifically, look for signs that warrant a new hypothesis:

   * New bigrams/bigrams above a threshold frequency.
   * Numeric correlations above a threshold (e.g. |corr| ≥ 0.2 with rating).
   * Emergence of low correlation for an existing feature (suggest dropping).
2. **Continuously refine** the list of hypotheses in memory:

   * Append new hypotheses derived from fresh EDA.
   * Assign or update priority scores.
   * Store rationale.
3. **Output** strictly‐typed JSON:

```json5
{
  "timestamp": "ISO8601",
  "new_hypotheses": [
    {
      "hypothesis": "Users whose 'GrimdarkSentimentCount' > 10 have avg rating 4.3 vs 3.8.",
      "priority": 4,
      "notes": "Corr=0.21; high signal; code implementation is straightforward; novelty moderate."
    },
    {
      "hypothesis": "Frequent use of character names (e.g. 'Drizzt') suggests subgenre affinity; propose counting named characters per user.",
      "priority": 3,
      "notes": "Textual insight: 'Drizzt' appears in 5% of sample; pure‐LLM extraction needed; effort=3."
    }
  ]
}
```

#### 3.3.2 Triggering & Loop

* **Trigger** whenever a new EDA report is appended.
* **Workflow**:

  1. Load last two EDA entries. Compare top bigrams/correlations with previous run.
  2. If any new bigram frequency > threshold (say 0.05), propose a hypothesis.
  3. Always re-evaluate existing hypotheses: if their underlying numeric correlation has drifted (e.g. fallen < 0.05), mark priority down or drop.
  4. Append JSON under `project_memory["hypotheses"]`.

#### 3.3.3 Prompt to HypothesisAgent

```text
SYSTEM:
You are the “HypothesisAgent.” Each time a new EDA report arrives, analyze the differences from previous EDA. Identify:
1. Any new bigrams/trigrams whose frequency > 0.05 and were not seen before → propose a hypothesis that counting those might predict user preference.
2. Any numeric correlations (“new_feature_correlations”) > 0.20 (abs) → propose a hypothesis that this feature is predictive.
3. Reevaluate existing hypotheses in memory: if their underlying statistic (e.g. correlation) has dropped below 0.05, lower their priority or mark “deprecated.”

For each new or updated hypothesis, create an object:
{
  "hypothesis": "...",
  "priority": <1-5>,
  "notes": "..."
}

Return a JSON: { "new_hypotheses": [ ... ] }  
```

#### 3.3.4 Testing & E2E Goals

* **Test #1**: Populate `eda_reports` with two entries:

  1. First entry has no bigrams/correlations.
  2. Second entry has a new bigram “grimdark magic” with frequency 0.07.
     – **Expected**: HypothesisAgent writes one new hypothesis about “grimdark magic count.”
* **Test #2**: Add a dummy feature with correlation 0.3 under `new_feature_correlations` in EDA.
  – **Expected**: HypothesisAgent writes a hypothesis: “Feature X correlates with rating (0.3).”

---

### 3.4 FeatureIdeationAgent (Iterative Multi-Pass Ideation)

#### 3.4.1 Role & Responsibilities

1. **Consume** all hypotheses with priority ≥ 3 from `project_memory["hypotheses"]`.
2. **First Pass (Creativity)**: For each hypothesis, generate 2–3 raw “feature ideas”: one *code-template* and one *LLM-feature*. Record “chain of thought” for each.
3. **Second Pass (Filter & Rank)**: Validate each idea against a restricted DSL (for code) or structured LLM prompt guidelines. Assign “expected\_effort” (1–5) and “expected\_impact” (1–5).
4. **Optional Third Pass**: Merge overlapping suggestions, drop low‐value ones.
5. **Always append** strictly‐typed JSON to `project_memory["feature_proposals"]`, with fields:

```json5
{
  "timestamp": "ISO8601",
  "pass": <1 or 2>,
  "proposals": [
    {
      "name": "<AlphanumericName>",
      "type": "code" | "llm",
      "dsl"?: "<DSL expression>",       // only for code
      "prompt"?: "<LLM prompt>",       // only for llm
      "chain_of_thought": "<long text>",
      "rationale": "<1-sentence>",
      "expected_effort"?: <1-5>,        // only in pass 2
      "expected_impact"?: <1-5>,        // only in pass 2
      "notes"?: "<why accepted or rejected>"  // only pass 2
    },
    /* … more … */
  ]
}
```

#### 3.4.2 Triggering & Loop

* **Trigger** whenever **HypothesisAgent** appends new hypotheses with priority ≥ 3.
* **Workflow**:

  1. Fetch top N hypotheses (priority ≥ 3).
  2. **Pass 1** (creativity): send a chain-of-thought prompt to LLM that includes context:
     – Top N hypotheses (text),
     – Recent EDA summaries (concise),
     – Any previously realized features (names only).
  3. **Pass 2** (filter): for each “raw” idea, ask LLM (or run a code snippet) to rewrite into valid DSL/LLM prompt; then assign effort/impact.
  4. Append both passes’ outputs to memory.
  5. If any “accepted” proposals remain (effort ≤ 4, impact ≥ 3), break; else wait for new hypotheses.

#### 3.4.3 Prompt Templates

##### Pass 1 Prompt (`ideation_pass1_prompt.txt`):

```text
SYSTEM:
You are the “FeatureIdeationAgent”—an expert in engineering user/item features for recommendation. 
Below are:
  1. Top {N} hypotheses (with their priority and notes):
        {list of hypothesis objects}
  2. A brief EDA summary: 
        {concise numeric + textual EDA}
  3. Already realized features: {list of feature names}.

TASK (Pass 1):
For each hypothesis, generate *two* feature proposals:

  A. A **code-template** in this restricted DSL:

     TemplateName(x) = ParamA * COUNT(condition in last ParamB days)
                    | ParamA * VARIANCE(rating if condition)
                    | ParamA * SentimentEmbeddingMean(review_text, condition)
     Condition can reference item metadata or review_text.

  B. A **pure LLM-feature** prompt: replace `<USER_REVIEWS>` or `<ITEM_DESC>` placeholders accordingly. Prepend: “Answer only with a single decimal [0–1].”

For each proposal, provide these fields as JSON:
  {
    "name": "<uniqueName>",
    "type": "code" or "llm",
    "dsl": "<DSL expression>"      // only if type=="code"
    "prompt": "<LLM prompt>"       // only if type=="llm"
    "chain_of_thought": "<long, step-by-step reasoning text>",
    "rationale": "<one-sentence summary>"
  }
Return exactly 2 * N JSON objects in an array under key `"proposals"`. Do not output anything else.

END SYSTEM
```

##### Pass 2 Prompt (`ideation_pass2_prompt.txt`):

```text
SYSTEM:
You are still the “FeatureIdeationAgent,” now doing Pass 2 (Filter & Rank). You have:
  – Raw proposals from Pass 1: {list of proposal objects}.
  – DSL Grammar (exactly these forms allowed):
      TemplateName(x) = ParamA * COUNT(condition in last ParamB days)
                     | ParamA * VARIANCE(rating if condition)
                     | ParamA * SentimentEmbeddingMean(review_text, condition)
    Condition must be one of: “genre == 'Fantasy'”, “author == 'X'”, or “timestamp >= Now - ParamB days”.

  – LLM Prompt Guidelines:
      * Must include “<USER_REVIEWS>” or “<ITEM_DESCRIPTION>”.
      * Must start with “Answer only with a single decimal between 0 and 1.”

TASK (Pass 2):
For each raw proposal where `type == "code"`:
  1. Check if `dsl` strictly follows the allowed grammar. 
     – If it uses unsupported operations (“median”, “entropy”), **rewrite** it into an equivalent allowed form or **mark** `"notes": "discarded: unsupported operation"`.
  2. After rewriting (or if valid), assign:
     – `"expected_effort"`: 1–5 (ease of implementation; 1=trivial pandas code, 5=complex LLM required).
     – `"expected_impact"`: 1–5 (expected signal based on EDA; 1=low, 5=high).

For each raw proposal where `type == "llm"`:
  1. Check if `prompt` contains exactly one of the placeholders `<USER_REVIEWS>` or `<ITEM_DESCRIPTION>`. 
     – If missing, **rewrite** to include it. 
     – Prepend “Answer only with a single decimal between 0 and 1.”
  2. Assign `expected_effort` (1=prompt tweak, 5=complex multi‐turn chain) and `expected_impact`.

Output an array under `"proposals"` where each object is:
  {
    "name": "...",
    "type": "code" or "llm",
    "dsl": "...",       // only for code
    "prompt": "...",    // only for llm
    "chain_of_thought": "...",  // copy from Pass 1
    "rationale": "...",
    "expected_effort": <1-5>,
    "expected_impact": <1-5>,
    "notes": "..."     // explain any rewriting or discarding
  }
Do not output anything else.

END SYSTEM
```

#### 3.4.4 Testing & E2E Goals

* **Test #1**: Provide 3 hypotheses with artificially easy/obvious DSL forms.
  – **Pass 1**: Expect 6 proposals (3 code, 3 llm). Each must include `chain_of_thought` and `rationale`.
  – **Pass 2**: Provide one raw proposal using an unsupported DSL (e.g. “MEDIAN”). Expect it to be rewritten to a supported form (“VARIANCE” or “MEAN”), or flagged discarded.
  – **Check**: All proposals in Pass 2 have exactly the required fields; `expected_effort` ∈ \[1,5], `expected_impact` ∈ \[1,5].

* **Test #2**: Feed a proposal missing `<USER_REVIEWS>` placeholder.
  – **Pass 2**: Expect the `prompt` field to be rewritten to include the placeholder and the “Answer only…” preamble.

---

### 3.5 FeatureRealizationAgent (Code & LLM Wrappers)

#### 3.5.1 Role & Responsibilities

1. **Consume** the final `project_memory["feature_proposals"]` entries where `"pass" == 2` and `"notes"` does not indicate “discarded.”
2. **Separate** them into:

   * **Code functions** (`type == "code"` → parse `dsl`).
   * **LLM wrappers** (`type == "llm"` → wrap `prompt`).
3. **For code**:
   a. Parse DSL into a safe, vectorized Python function.
   b. Validate by running on a small sample; if error → record in `project_memory["realization_errors"]`.
   c. Store mapping `(feature_name → function_obj, param_names)`.
4. **For LLM**:
   a. Create a wrapper function that, given `user_id` or `item_id`, assembles the prompt, calls LLM, parses float, enforces \[0,1], caches.
   b. Store mapping `(feature_name → function_obj, dependencies=["scale"])`.
5. **Append** to `project_memory["realized_functions"]` under separate `"code"` and `"llm"` keys.

#### 3.5.2 JSON Schema for Each Realized Function

```json
// For code
{
  "name": "GrimdarkSentimentCount",
  "param_names": ["ParamA", "ParamB"],
  "status": "valid",          // or "error"
  "error": ""                 // non‐empty if "error"
}

// For LLM
{
  "name": "DarknessAffinityScore",
  "dependencies": ["scale"],  // always include scale if prompt needs tuning
  "status": "valid",          
  "error": ""                 
}
```

#### 3.5.3 Testing & E2E Goals

* **Test #1**: Provide a known valid DSL:
  – Expect creation of a Python function that runs on sample data without exception.
  – `project_memory["realized_functions"]["code"]` contains an entry with `"status": "valid"`.
* **Test #2**: Provide an invalid DSL (e.g. uses “MEDIAN”):
  – Expect `"status": "error"`, along with a descriptive `"error"` message.
* **Test #3**: Provide an LLM prompt missing “Answer only…”.
  – Expect the wrapper to still create a function, but the prompt is **automatically** prepended with the “Answer only…” instruction.

---

### 3.6 OptimizationAgent (BO / Search)

#### 3.6.1 Role & Responsibilities

1. **Build a search space** over all realized functions’ parameters plus hyperparameters:

   * For each code feature: each `ParamX` → `[0,10]` or a more data-informed bound (extracted from EDA).
   * For each LLM feature: `scale` → `[0,5]`.
   * Clustering: `K ∈ [2, 10]`.
   * FM dims: `fm_dim ∈ [4, 64]`.
   * FM regularizations: `lambda_w`, `lambda_v ∈ [1e-6, 1e-2]`.
2. **Define objective(θ)**:
   a. For warm‐start:
   i. Materialize feature matrix on train+val (calls code + LLM wrappers).
   ii. Fit clustering (KMeans or GMM) with `K`.
   iii. Train FM on cluster‐assigned data with dims/regs.
   iv. Compute validation RMSE (and log `tensorboard.add_scalar("Val_RMSE", rmse, step)`).
   v. Return RMSE.
   b. For speed, run on a stratified subset (e.g. 30% of users, 50% of items), as long as it remains representative.
3. **Run Bayesian optimization** (e.g. `skopt.gp_minimize`) for a fixed budget (e.g. 50 calls).
4. **Log** every trial into `project_memory["bo_history"]["trials"]` with full `θ` and `rmse`.
5. **Append** best parameters to `project_memory["bo_history"]["best_params"]` and `["best_rmse"]`.

#### 3.6.2 Testing & E2E Goals

* **Test #1**: Provide a trivial “dummy feature” that returns zeros for all users.
  – Expect RMSE to match a baseline (e.g. global average).
  – `bo_history` should record at least one trial.
* **Test #2**: Provide both a trivial dummy feature and a known strong code feature (e.g. a perfect predictor for a synthetic toy dataset).
  – Expect BO to discover that code feature’s high weight (RMSE drops significantly).
  – Check that `best_params` sets that feature’s scale/ParamX to a high value.

---

### 3.7 ReflectionAgent (Critique & Next-Step Suggestion)

#### 3.7.1 Role & Responsibilities

1. **Consume**:

   * `project_memory["bo_history"]` (all trials, best\_params, best\_rmse).
   * The final FM model from the latest BO run (passed as an argument or reloaded from disk).
2. **Extract**:

   * Linear weights `w_j` and, if available, interaction weights `v_{j,k}`.
   * Look across the top n trials (e.g. best 5) to see if certain `ParamX` repeatedly at boundary → suggests rethinking that dimension.
3. **Generate** a “Reflection Note” with:

   * `"top_features"`: list up to 5 `(feature, weight)`.
   * `"low_value_features"`: features with |weight| < 0.01 across top n trials.
   * `"feature_interactions"`: list `(pair, interaction_weight)` for |v| > threshold (e.g. 0.1).
   * `"bo_insights"`: list of strings like “Parameter <name> always at 0 or max.”
   * `"next_steps"`: at least 3 bullet points, e.g. “drop Feature X,” “refine Feature Y DSL to exponential decay,” “Propose new template combining features A & B.”
4. **Output** JSON under `project_memory["reflections"]`.

#### 3.7.2 Prompt Template (`reflection_prompt.txt`)

```text
SYSTEM:
You are the “ReflectionAgent,” an expert at interpreting Bayesian‐optimization results and factorization‐machine models. You have:

1. BO history: under `bo_history.trials`, a list of { "params": {...}, "rmse": <float> }.
2. Best_params: under `bo_history.best_params`: the winning parameter set.
3. A final FM model object with attributes:
     - linear_weights: { "feature_name": <weight>, ... }
     - interaction_weights: { ("FeatA","FeatB"): <weight>, ... }

TASK:
Produce a JSON object with key `"reflection_note"` containing:

{
  "top_features": [  // up to 5 features with highest |linear_weights|
    { "feature": "<name>", "weight": <float> }, …
  ],
  "low_value_features": [
    { "feature": "<name>", "weight": <float> }, …
  ],
  "feature_interactions": [
    { "pair": ["<FeatA>", "<FeatB>"], "interaction_weight": <float> }, …
  ],
  "bo_insights": [
    "<string describing param boundary behavior>",
    …
  ],
  "next_steps": [
    "<bullet suggestion: drop/refine/add new templates>",
    …
  ]
}

Guidance:
– If any `ParamX` in the top 5 BO trials is always at the same boundary (e.g. 0 or 10), add an insight “ParamX always at <value> → consider broadening/ removing this dimension.”  
– Include exactly 3–5 next steps.  

END SYSTEM
```

#### 3.7.3 Testing & E2E Goals

* **Test #1**: Supply a synthetic FM with known weights:
  – `linear_weights`: { “A”: 5, “B”: 0.001, “C”: −3 }
  – `interaction_weights`: { (“A,” “C”): 0.2 }
  – `bo_history`: contains a trial with `ParamA` always at 10.
  – **Check**: `reflection_note` lists “A” as top, “B” as low\_value, interaction (“A”,“C”) with weight 0.2, “ParamA always at 10” in bo\_insights, and at least 3 next\_steps.

---

## 4 Infinite Loop Workflow (Putting It All Together)

Below is a step-by-step “master” orchestrator that ties all agents into a continuous loop. In practice, you might run this as a long‐running process (e.g. a supervised service or cron job) or trigger each agent on events. We show pseudo-code for `orchestrator.py`.

```python
import time
from datetime import datetime
from agents.data_thinker_agent import run_data_thinker
from agents.hypothesis_agent import run_hypothesis_agent
from agents.feature_ideation_agent import run_feature_ideation
from agents.feature_realization_agent import run_feature_realization
from agents.optimization_agent import run_optimization
from agents.reflection_agent import run_reflection
from utils.memory import load_memory, append_to_memory

# Configurable parameters
EDA_INTERVAL = 3600       # seconds between EDA runs (if no changes)
LOOP_SLEEP = 300          # main loop sleep interval
LAST_EDA_TIME = None
LAST_REALIZATION_COUNT = 0
LAST_HYPOTHESIS_COUNT = 0
LAST_REFLECTION_COUNT = 0
MAX_ITERS = None          # or `` for truly endless

def orchestrator():
    global LAST_EDA_TIME, LAST_REALIZATION_COUNT, LAST_HYPOTHESIS_COUNT, LAST_REFLECTION_COUNT
    iteration = 0
    while True:
        iteration += 1

        memory = load_memory()
        # ---------- Stage 1: DataThinkerAgent (∞) ----------
        now = time.time()
        # Trigger if never run, if realized_functions changed, or interval passed
        realized_count = len(memory.get("realized_functions", {}).get("code", [])) + \
                         len(memory.get("realized_functions", {}).get("llm", []))
        if (LAST_EDA_TIME is None or 
            realized_count != LAST_REALIZATION_COUNT or 
            now - (LAST_EDA_TIME or 0) >= EDA_INTERVAL):
            df_reviews = pd.read_csv("data/goodreads_reviews.csv", parse_dates=["timestamp"])
            df_items = pd.read_csv("data/goodreads_items.csv")
            run_data_thinker(df_reviews, df_items)
            LAST_REALIZATION_COUNT = realized_count
            LAST_EDA_TIME = time.time()

        # Refresh memory
        memory = load_memory()

        # ---------- Stage 2: HypothesisAgent ----------
        hypothesis_count = len(memory.get("hypotheses", []))
        if hypothesis_count != LAST_HYPOTHESIS_COUNT:
            run_hypothesis_agent()
            LAST_HYPOTHESIS_COUNT = len(load_memory().get("hypotheses", []))

        memory = load_memory()

        # ---------- Stage 3: FeatureIdeationAgent (triggered by new hypotheses) ----------
        # Simplest: run whenever new high-priority hypotheses appear
        top_hyp_count = len([h for h in memory.get("hypotheses", []) if h["priority"] >= 3])
        # We only run one pass at a time; if proposals exist not yet realized, skip
        existing_proposals = len(memory.get("feature_proposals", []))
        if top_hyp_count > 0 and existing_proposals == 0:
            run_feature_ideation() 
            # After pass 1, feature_proposals pass==1 will exist; 
            # then run again for pass==2 in a subsequent iteration

        memory = load_memory()

        # Check if pass 2 proposals are fully present before realization
        pass2_proposals = [p for p in memory.get("feature_proposals", []) if p["pass"] == 2]
        if pass2_proposals and not memory.get("realized_functions"):
            run_feature_realization()

        memory = load_memory()

        # ---------- Stage 4: OptimizationAgent ----------
        if memory.get("realized_functions"):
            # Only run BO if we don’t already have best_params, or new functions appeared since last BO
            bo_done = "bo_history" in memory
            new_funcs_count = len(memory["realized_functions"]["code"]) + len(memory["realized_functions"]["llm"])
            if (not bo_done) or (new_funcs_count > memory.get("bo_history", {}).get("functions_count", 0)):
                run_optimization()
                # After optimization, record how many functions were used
                memory2 = load_memory()
                memory2["bo_history"]["functions_count"] = new_funcs_count
                append_to_memory("bo_history", memory2["bo_history"])  # update
                # After BO, trigger reflection next iteration

        memory = load_memory()

        # ---------- Stage 5: ReflectionAgent ----------
        reflection_done = len(memory.get("reflections", []))
        if reflection_done < len(memory.get("bo_history", {}).get("trials", [])):
            # We only want one reflection per BO run
            run_reflection()

        memory = load_memory()

        # ---------- Termination Condition (Optional) ----------
        if MAX_ITERS and iteration >= MAX_ITERS:
            break

        # ---------- Sleep until next cycle ----------
        time.sleep(LOOP_SLEEP)

if __name__ == "__main__":
    orchestrator()
```

> **Key Points**:
> – The **DataThinkerAgent** is triggered on three conditions: first run, new realized functions count changes, or periodic interval.
> – **HypothesisAgent** runs whenever new EDA has appended more entries.
> – **FeatureIdeationAgent** only runs if there are top‐priority hypotheses and no existing proposals; it will produce pass 1 proposals, and on next iteration, pass 2 proposals.
> – **FeatureRealizationAgent** runs only when pass 2 proposals exist and nothing has been realized yet.
> – **OptimizationAgent** runs when realized functions exist, and either no BO has run before or new functions have been introduced.
> – **ReflectionAgent** runs once per BO round.
> – The loop never exits (unless you set `MAX_ITERS`).

---

## 5 Structured JSON Schemas (All Outputs)

Below are the precise JSON schemas each agent must adhere to; any deviation should raise a validation error.

### 5.1 `eda_reports[*]`

```json5
{
  "timestamp": "string (ISO8601)",
  "numeric_summaries": {
    "num_users": "integer ≥0",
    "num_items": "integer ≥0",
    "reviews_per_user": {
      "mean": "float ≥0",
      "median": "float ≥0",
      "max": "integer ≥0",
      "min": "integer ≥0"
    },
    "reviews_per_item": {
      "mean": "float ≥0",
      "median": "float ≥0",
      "max": "integer ≥0",
      "min": "integer ≥0"
    },
    "rating_histogram": {
      "1": "float between 0 and 1",
      "2": "float",
      "3": "float",
      "4": "float",
      "5": "float"
    },
    "timestamp_range": {
      "earliest": "YYYY-MM-DD",
      "latest": "YYYY-MM-DD"
    }
  },
  "textual_insights": [
    {
      // Either a bigram insight or sentiment stats
      "bigram": "string",        // optional if not sentiment entry
      "frequency": "float [0,1]" // optional
    },
    {
      "sentiment_mean": "float [-1,1]",
      "sentiment_std": "float ≥0"
    }
  ],
  "cross_tab_insights": [
    "string description",  // describing correlation metrics or top-n lists
    "string"
  ],
  "new_feature_correlations": [
    {
      "feature": "string",
      "corr_with_rating": "float [-1,1]"
    }
  ],
  "metadata": {
    "rows_sampled": "integer >0",
    "bigram_vocab_size": "integer >0"
  }
}
```

### 5.2 `hypotheses[*]`

```json5
{
  "timestamp": "string (ISO8601)",
  "hypothesis": "string",
  "priority": "integer in [1,5]",
  "notes": "string"
}
```

### 5.3 `feature_proposals[*]`

```json5
{
  "timestamp": "string (ISO8601)",
  "pass": "integer ==1 or 2",
  "proposals": [
    {
      "name": "string (alphanumeric & underscores only)",
      "type": "code" | "llm",
      // if type == "code":
      "dsl": "string (DSL expression)",
      // if type == "llm":
      "prompt": "string (LLM prompt with placeholders)",
      "chain_of_thought": "string ( ≥2 sentences, may be multi-line )",
      "rationale": "string (1 sentence)",
      // Pass 2 only:
      "expected_effort": "integer in [1,5]",
      "expected_impact": "integer in [1,5]",
      "notes": "string (why accepted/rejected/refined)" 
    },
    // repeated for each proposal
  ]
}
```

### 5.4 `realized_functions.code[*]`

```json5
{
  "name": "string",
  "param_names": ["ParamA","ParamB",…],
  "status": "valid" | "error",
  "error": "string (empty if valid)"
}
```

### 5.5 `realized_functions.llm[*]`

```json5
{
  "name": "string",
  "dependencies": ["scale"] | [],   // always include “scale” if the prompt needs it
  "status": "valid" | "error",
  "error": "string"
}
```

### 5.6 `bo_history.trials[*]`

```json5
{
  "timestamp": "string (ISO8601)",
  "params": {
    // e.g. "GrimdarkSentimentCount_ParamA": 2.3,
    //       "DarknessAffinityScore_scale": 3.1,
    //       "K": 4, "fm_dim": 16, "lambda_w": 0.0001, "lambda_v": 0.00001
  },
  "rmse": "float ≥ 0"
}
```

### 5.7 `bo_history.best_params` & `best_rmse`

```json5
"best_params": { /* same schema as a single trial’s params */ },
"best_rmse": "float ≥0"
```

### 5.8 `reflections[*]`

```json5
{
  "timestamp": "string (ISO8601)",
  "top_features": [
    { "feature": "string", "weight": "float" }, …
  ],
  "low_value_features": [
    { "feature": "string", "weight": "float" }, …
  ],
  "feature_interactions": [
    { "pair": ["string","string"], "interaction_weight": "float" }, …
  ],
  "bo_insights": [
    "string", …
  ],
  "next_steps": [
    "string", …
  ]
}
```

### 5.9 `final_eval`

```json5
{
  "warm": {
    "rmse": "float ≥ 0",
    "ndcg@10": "float [0,1]",
    "precision@10": "float [0,1]"
  },
  "cold": {
    "ndcg@10": "float [0,1]",
    "precision@10": "float [0,1]"
  },
  "cold_questions": [
    { "question": "string", "type": "code" | "llm" }, …
  ]
}
```

---

## 6 Testing & End-to-End Validation Points

At every major agent boundary, we define a minimal E2E test that runs the agent on a toy subset and verifies:

1. **Memory growth**: The correct key in `project_memory.json` is appended.
2. **JSON schema**: The appended entry exactly matches the prescribed schema (no missing fields, correct types).
3. **Logical plausibility**: Certain numerical checks (e.g. correlation ∈ \[−1,1]).

Below is a summary table of tests you should implement under `tests/`:

| Agent            | Test Name               | Goal                         | Check |
| :--------------- | :---------------------- | :--------------------------- | :---- |
| DataThinkerAgent | `test_data_thinker_e2e` | Run on 10k reviews/2k items. |       |

1. `project_memory["eda_reports"]` length increments.
2. Numeric fields non-negative, hist sums≈1.0.

\| HypothesisAgent | `test_hypothesis_agent_e2e` | Use an EDA with a bigram freq0.07. |

1. `project_memory["hypotheses"]` gains new entry.
2. Priority ≥3 if corr>0.2.

\| FeatureIdeationAgent | `test_feature_ideation_e2e` | Provide 2 hypotheses. Run pass1 and pass2. |

1. After pass1: `feature_proposals` contains entries with pass==1 (size 4).
2. After pass2: `feature_proposals` contains pass==2, each with valid DSL/prompt and effort/impact.

\| FeatureRealizationAgent | `test_feature_realization` | Supply 2 code proposals, 1 llm proposal. |

1. `realized_functions` created with correct “status” fields.
2. Invalid DSL triggers `"status":"error"`.

\| OptimizationAgent | `test_optimization_flow` | Supply a synthetic dataset where one code feature fully predicts rating. |

1. `bo_history["trials"]` length≥1.
2. `best_params` sets that feature’s scale large.

\| ReflectionAgent | `test_reflection_agent` | Give dummy BO history & FM weights. |

1. `project_memory["reflections"]` appended.
2. `"top_features"` includes correct top‐weight features.

\| Orchestrator | `test_orchestrator_smoke` | Run for 3 iterations on a toy dataset. |

1. All keys in `project_memory.json` appear.
2. No JSON–decode errors, every schema validated.

Use a test utility `utils/testing_utils.py` containing functions like:

```python
def assert_json_schema(instance: dict, schema: dict) -> None:
    # Raises AssertionError if instance doesn’t match schema

def load_test_data(n_reviews: int, n_items: int) -> (pd.DataFrame,pd.DataFrame):
    # Creates a synthetic toy dataset with random ratings, random words
```

---

## 7 Critical Evaluation: Why the Pipeline Might Fail & Mitigations

Below are potential pitfalls (“failure modes”) at each stage, with suggested mitigations:

1. **DataThinkerAgent stagnation**: Over time, new features dominate, and EDA may produce repetitive insights.
   – *Mitigation*: Compare newest EDA to last two runs; only append if there’s significant change (new feature correlations > 0.05 or new textual n-grams). Otherwise, skip.

2. **HypothesisAgent churn**: Too many trivial hypotheses (e.g. “count occurrences of ‘fantasy’”), leading to wasted cycles.
   – *Mitigation*: Only consider hypotheses with priority ≥ 3. Increase correlation threshold to 0.25 if too many.

3. **FeatureIdeationAgent creativity vs. feasibility gap**: LLM produces “fantasy index” requiring complicated text parsing not in grammar.
   – *Mitigation*: In Pass 2, automatically discard DSLs outside grammar. If LLM keeps producing invalid ones, refine prompt to reduce degrees of freedom.

4. **FeatureRealization errors**: DSL parsing brittle.
   – *Mitigation*: Build a robust DSL parser (e.g. use a library like `lark-parser` or `parsimonious`) rather than naive regex.

5. **OptimizationAgent divergence or extreme cost**: BO over too many parameters (10+ dimensions) can be slow.
   – *Mitigation*: Limit initial search to a stratified data subset (e.g. 30% of users). Use lower BO calls (20) in early iterations, then refine in final round (50). Use `skopt` with early stopping if RMSE stagnates.

6. **ReflectionAgent inaction**: If FM weights are near zero for all features, ReflectionAgent has no clear next steps.
   – *Mitigation*: If no feature has weight > 0.1, ask ReflectionAgent to propose “completely new” hypothesis (fallback to asking LLM to propose random DSL templates).

7. **Infinite Loop runaway**: Agents keep iterating without improving.
   – *Mitigation*: Keep track of “best\_rmse” over time. If it hasn’t improved by > 1% over last 2 iterations, break or reduce exploration.

8. **LLM cost explosion**: Every pass uses GPT-4 on every row (LLM feature).
   – *Mitigation*:

   * Cache all LLM calls in LLMInferenceAgent.
   * Batch LLM inference (ask per 10 users at once if possible) using LangChain’s batch prompts.
   * Fall back to a smaller model (e.g. an on‐device sentiment model) if cost becomes too high.

---

## 8 Recommended Libraries & Abstractions

### 8.1 LLM Wrappers & Agent Frameworks

* **LangChain**:

  * Use `LLMChain` for structured prompts.
  * Use `ConversationAgent` or `ZeroShotAgent` to build multi-step pipelines.
  * Built-in memory modules can stage small context windows (\~4k tokens), but since we need longer, integrate with `langchainplus-sdk` or chunk large memory.

* **Autogen (ms/autogen)**:

  * Provides “roles” (assistant, user, memory) and helps orchestrate multi-agent tasks.
  * Can automatically manage context window for chain-of-thought; can emit JSON responses.
  * Suggestion: Use Autogen for HypothesisAgent and FeatureIdeationAgent to leverage its “task templates” and “multi-round dialogues.”

* **Parsing DSL**:

  * `lark-parser` or `parsimonious` to define a small grammar for DSL. Avoid brittle regex.
  * This grammar could be:

```ebnf
?start: TEMPLATE
TEMPLATE: NAME "(" X ")" "=" EXPRESSION
EXPRESSION: TERM (("+"|"-") TERM)*
TERM: PARAM "*" FUNC
FUNC: "COUNT(" CONDITION " in last " PARAM " days)" 
    | "VARIANCE(rating if " CONDITION ")" 
    | "SentimentEmbeddingMean(review_text, " CONDITION ")"
CONDITION: "genre == '" NAME "'" | "author == '" NAME "'" | "timestamp >= Now - " PARAM " days"
PARAM: /Param[A-Z][A-Za-z0-9_]*/
NAME: /[A-Za-z_][A-Za-z0-9_]*/
X: NAME
%ignore " "
```

* **Bayesian Optimization**:

  * `scikit-optimize (skopt)` or `optuna` (for more advanced pruning & multi-objective).

* **Factorization Machines**:

  * `tffm` or `fastFM` or implement a simple FM with PyTorch. Ensure you can extract weights for reflection.

* **Clustering**:

  * `scikit-learn`’s `KMeans` or `GaussianMixture`.

* **Logging & Tracking**:

  * Use **TensorBoard** for scalar metrics (`rmse`, `ndcg`, etc.).
  * Use a file‐based logger (e.g. `structlog` or Python’s `logging`) to write structured JSON logs to `logs/`.
  * Optionally integrate **Weights & Biases** for remote tracking if desired.

* **Testing**:

  * Use **pytest** for E2E tests.
  * Use `jsonschema` to validate JSON outputs against the schemas defined above.
  * Define test fixtures in `tests/fixtures/` with synthetic data.

---

## 9 Appendix: Example Pseudocode Skeleton

Below is a very high-level pseudocode summary, referring to the code files above, showing how each agent ties into the pipeline. This is not runnable code but a skeleton illustrating function calls.

```python
# orchestrator.py

from utils.memory import load_memory
from agents.data_thinker_agent import run_data_thinker
from agents.hypothesis_agent import run_hypothesis_agent
from agents.feature_ideation_agent import run_feature_ideation
from agents.feature_realization_agent import run_feature_realization
from agents.optimization_agent import run_optimization
from agents.reflection_agent import run_reflection
import time

def main_loop():
    while True:
        memory = load_memory()
        # Stage 1: EDA
        if should_run_data_thinker(memory):
            run_data_thinker(...)
        # Stage 2: Hypotheses
        if should_run_hypothesis(memory):
            run_hypothesis_agent()
        # Stage 3: Ideation
        if should_run_ideation(memory):
            run_feature_ideation()
        # Stage 4: Realization
        if should_run_realization(memory):
            run_feature_realization()
        # Stage 5: Optimization
        if should_run_optimization(memory):
            run_optimization()
        # Stage 6: Reflection
        if should_run_reflection(memory):
            run_reflection()
        # Sleep briefly
        time.sleep(300)

if __name__ == "__main__":
    main_loop()
```

Each `should_run_xxx(memory)` checks differences in memory counts or timestamps to decide whether to invoke that agent. Each `run_xxx(...)` function:

```python
# e.g. in data_thinker_agent.py

def run_data_thinker(df_reviews, df_items):
    # Compute numeric/textual/cross-tab
    eda_json = compute_eda(...)
    append_to_memory("eda_reports", eda_json)
```

```python
# in hypothesis_agent.py

def run_hypothesis_agent():
    eda = load_memory()["eda_reports"][-2:]  # last two runs
    hyp_json = create_hypotheses_from_eda(eda)
    append_to_memory("hypotheses", hyp_json["new_hypotheses"])
```

```python
# in feature_ideation_agent.py

def run_feature_ideation():
    hyps = [h for h in load_memory()["hypotheses"] if h["priority"]>=3]
    # Pass 1
    pass1_json = call_llm_with_prompt("ideation_pass1_prompt.txt", context={"hypotheses":hyps, ...})
    append_to_memory("feature_proposals", {"timestamp":now, "pass":1, "proposals":pass1_json})
    # next iteration, Pass 2
    pass2_json = call_llm_with_prompt("ideation_pass2_prompt.txt", context={"raw_proposals":pass1_json, ...})
    append_to_memory("feature_proposals", {"timestamp":now, "pass":2, "proposals":pass2_json})
```

```python
# in feature_realization_agent.py

def run_feature_realization():
    prop_list = [p for p in load_memory()["feature_proposals"] if p["pass"]==2]
    for entry in prop_list:
        for proposal in entry["proposals"]:
            if proposal["type"]=="code":
                result = realize_code_template(proposal["dsl"], proposal["name"])
                append_to_memory("realized_functions.code", result)
            else:  # llm
                result = wrap_llm_feature(proposal["prompt"], proposal["name"])
                append_to_memory("realized_functions.llm", result)
```

```python
# in optimization_agent.py

def run_optimization():
    realized = load_memory()["realized_functions"]
    dims = build_search_space_from_realized(realized)
    bo_res = perform_bo(dims, eval_function)   # eval_function uses realized functions to build features, FM, etc.
    append_to_memory("bo_history.trials", bo_res.trials)
    append_to_memory("bo_history.best_params", bo_res.best_params)
    append_to_memory("bo_history.best_rmse", bo_res.best_rmse)
```

```python
# in reflection_agent.py

def run_reflection():
    bo_history = load_memory()["bo_history"]
    # load FM model from disk or pass as argument
    fm_model = load_final_fm_model()
    reflection_json = create_reflection(bo_history, fm_model)
    append_to_memory("reflections", reflection_json)
```

---

# Final Words

This specification lays out an **infinite‐loop, fully automated, multi-agent pipeline** that mimics a human data science workflow—Data Exploration, Hypothesis Generation, Feature Ideation, Feature Realization, Optimization, and Reflection—in a closed loop. Each agent:

* Runs autonomously, triggered by changes in memory or time intervals,
* Reads and writes strictly structured JSON to a shared `project_memory.json`,
* Is testable end-to-end on toy data, and
* Can be debugged in isolation because each step has a clear input/output contract.

You can start by implementing each agent one at a time, writing E2E tests to verify its behavior, then integrate them into the orchestrator loop. Over a night of running on your Goodreads Fantasy subset, the pipeline should discover new code and LLM features, tune them, reflect on their utility, and propose even better next steps—ultimately yielding publishable, rigorously validated results.
