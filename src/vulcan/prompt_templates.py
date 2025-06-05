from langchain_core.output_parsers import PydanticOutputParser

from vulcan.schemas.agent_schemas import LLMFeatureOutput

# Create a parser to generate JSON formatting instructions for the LLM
parser = PydanticOutputParser(pydantic_object=LLMFeatureOutput)
JSON_FORMAT_INSTRUCTIONS = parser.get_format_instructions()


SYSTEM_PROMPT = """You are VULCAN, an autonomous AI data scientist and feature engineer.
Your primary goal is to create novel and predictive features for a machine learning model.
You operate on a pandas DataFrame named `df`.

**Feature Types**
You can generate features of three types:
1.  `code_based`: You write Python code to create the feature. This is for standard transformations (e.g., aggregations, polynomial features, binning).
2.  `llm_based`: You write a prompt that will be sent to another LLM to perform inference on each row of the data. This is for complex NLP tasks like sentiment analysis, entity extraction, or text classification. **DO NOT** write code that uses libraries like NLTK, spaCy, or TextBlob. Instead, create a prompt for the `LLMRowAgent`.
3.  `hybrid`: A combination of both, where you might preprocess data with code, use an LLM for inference, and then post-process the result with more code.

**Your Task**
- When asked to generate, refine, or mutate a feature, you must analyze the request and the provided context.
- You must provide your reasoning in a clear, step-by-step chain of thought.
- You must output your final feature definition in a valid JSON format, following the provided instructions.
"""

# --- Feature Ideation Prompt ---
CREATE_FEATURE_PROMPT = """
**Objective:** Create a new, innovative feature to improve a recommendation model.

**Context:**
- You are provided with the data schema and a sample of the first 10 rows of the training data.
- Context: `{context}`
- Existing Features: `{existing_features}`

**Task:**
1.  **Analyze Context:** Review the available data columns, the data sample, and existing features.
2.  **Brainstorm Strategies:** Think of several creative strategies for new features (e.g., user-item interactions, content-based features, temporal features, graph-based features).
3.  **Select & Justify:** Choose the most promising strategy and justify your choice.
4.  **Implement Feature:** Write a Python function `create_feature(df)` that implements your chosen strategy. The function must be self-contained and not rely on any external variables other than the input DataFrame. All necessary imports must be inside the function.
5.  **Output:** Format your response as a JSON object. The `code` field must contain the full Python code for your `create_feature` function as a single string.

**Response Format:**
`{format_instructions}`
"""


# --- Feature Mutation Prompt ---
MUTATE_FEATURE_PROMPT = """
**Objective:** Mutate an existing feature to create a new, potentially better version.

**Context:**
- Feature to Mutate:
  - Name: `{feature_name}`
  - Description: `{feature_description}`
  - Code/Prompt: `{feature_implementation}`
- You are also provided with the data schema and a sample of the first 10 rows of the training data.
- Context: `{context}`

**Task:**
1.  **Analyze Feature:** Understand the purpose and implementation of the existing feature in light of the data sample.
2.  **Brainstorm Mutations:** Think of ways to alter the feature. Examples:
    - Change a parameter (e.g., window size in a moving average).
    - Apply a different aggregation function (e.g., `mean` to `std`).
    - Use a different text processing technique (e.g., TF-IDF to embeddings).
    - Add a new interaction term.
3.  **Select & Justify:** Choose the most promising mutation and justify why it might improve the feature.
4.  **Implement Mutation:** Write a new Python function `create_feature(df)` that implements the mutated feature. The function must be self-contained.
5.  **Output:** Format your response as a JSON object. The `code` field must contain the full Python code for your `create_feature` function as a single string.

**Response Format:**
`{format_instructions}`
"""


# --- Feature Refinement Prompt (from top-performing) ---
REFINE_TOP_FEATURE_PROMPT = """
**Objective:** Refine a high-performing feature to make it even better.

**Context:**
- High-Performing Feature to Refine:
  - Name: `{feature_name}`
  - Description: `{feature_description}`
  - Code/Prompt: `{feature_implementation}`
- You are also provided with the data schema and a sample of the first 10 rows of the training data.
- Context: `{context}`

**Task:**
1.  **Analyze Feature:** Understand what makes this feature effective, using the data sample as a guide.
2.  **Brainstorm Refinements:** Think of subtle but powerful ways to improve it. Examples:
    - Add normalization or scaling.
    - Combine it with another simple feature.
    - Adjust its complexity (e.g., simplify a complex calculation, or add a non-linear transformation).
3.  **Select & Justify:** Choose the best refinement and explain your reasoning.
4.  **Implement Refinement:** Write a new Python function `create_feature(df)` that implements the refined feature. The function must be self-contained.
5.  **Output:** Format your response as a JSON object. The `code` field must contain the full Python code for your `create_feature` function as a single string.

**Response Format:**
`{format_instructions}`
"""

# --- Row-Based LLM Feature Prompt ---
ROW_INFERENCE_PROMPT = """
**Objective:** Create a feature that requires LLM inference for each row of data by generating a numerical score on a well-defined, objective scale.

**Context:**
- Data Schema: `{context}`
- Text Columns available for processing: `{text_columns}`
- Sample Data Rows: `{data_rows}`

**Task:**
1.  **Analyze Context:** Review the data schema and sample rows to understand the data.
2.  **Propose LLM Task:** Define a task for an LLM that can be applied to each row to extract a meaningful feature. The task must result in a numerical score.
    - Example Task: "On a scale of 1 to 10, how likely is this user to enjoy a book with a similar plot?"
    - Example Task: "On a scale of 1 to 5, what is the level of agreement between the review text and the book's average rating?"
3.  **Define a Scoring Rubric:** You MUST define a clear, objective rubric for the numerical scale. This rubric will be included in the prompt to the LLM to ensure consistent, unbiased scoring.
    - Example Rubric:
        - 1: Very Unlikely / Strong Disagreement
        - 2: Unlikely / Some Disagreement
        - 3: Neutral / Mixed
        - 4: Likely / Some Agreement
        - 5: Very Likely / Strong Agreement
4.  **Create Prompt:** Write a clear, concise prompt that will be given to the LLM for each row. This prompt must include the task and the scoring rubric.
5.  **Output:** Format your response as a JSON object according to the instructions below. The `llm_prompt` field should contain the template you created.

**Response Format:**
`{format_instructions}`
"""


# --- Reflection and Refinement Prompt ---
REFLECT_AND_REFINE_PROMPT = """
**Objective:** Analyze a feature's performance and propose a refined version.

**Context:**
- Evaluated Feature:
  - Name: `{feature_name}`
  - Description: `{feature_description}`
  - Implementation: `{feature_implementation}`
  - Performance Score: `{feature_score}`
  - Evaluation Metrics: `{feature_metrics}`

**Task:**
1.  **Analyze Performance:** Review the feature's implementation and its evaluation metrics. What are its strengths and weaknesses? Why do you think it achieved the score it did?
2.  **Hypothesize Improvement:** Based on your analysis, form a hypothesis for how the feature could be improved.
3.  **Propose Refinement:** Describe the specific changes you would make to the feature's implementation.
4.  **Implement Refinement:** Write a new Python function `create_feature(df)` that implements the refined feature. The function must be self-contained.
5.  **Output:** Format your response as a JSON object. The `code` field must contain the full Python code for your `create_feature` function as a single string.

**Response Format:**
`{format_instructions}`
"""

# --- Repair Feature Prompt ---
REPAIR_FEATURE_PROMPT = """
**Objective:** Repair a non-functional Python feature code based on the provided error message.

**Context:**
- The feature execution failed, and you have the faulty code and the resulting error.
- Your goal is to fix the code so it executes successfully.
- The corrected feature should still be a valid `FeatureDefinition`.

**Instructions:**
1.  **Analyze:** Carefully examine the `faulty_code` and the `error_message`.
2.  **Identify the Root Cause:** Determine why the code is failing. Common issues include incorrect pandas API usage, missing imports, logical errors, or data type mismatches.
3.  **Correct the Code:** Rewrite the `code` to fix the error. Ensure the corrected code is robust and handles potential edge cases (e.g., empty DataFrames, missing columns).
4.  **Maintain the original intent of the feature.** Do not create a completely new feature, but rather fix the existing one.
5.  **Explain the fix:** In your chain-of-thought, clearly explain what the error was and how your new code fixes it.

**Input:**
- `faulty_code`: The original Python code that failed.
- `error_message`: The error message produced during execution.

**Output:**
- You must follow the `format_instructions` to produce a valid JSON object containing:
    - Your chain of thought reasoning for the repair.
    - The corrected feature definition (the `name` and `description` should be the same, but the `code` will be new).
"""

# --- Mathematical Feature Prompt ---
MATH_FEATURE_PROMPT = """
**Objective:** Create a new feature using only mathematical and statistical operations.

**Context:**
- You are provided with the data schema and a sample of the first 10 rows of the training data.
- Context: `{context}`
- Existing Features: `{existing_features}`

**Task:**
1.  **Analyze Context:** Review the numerical columns in the data schema and sample.
2.  **Brainstorm Strategies:** Think of several creative strategies for new features using only mathematical or statistical methods. Examples:
    - Polynomial combinations of existing numerical features.
    - Binning or discretization of a continuous variable.
    - Group-by aggregations (e.g., average rating per user, standard deviation of review length per book).
    - Ratios or interaction terms between numerical columns.
3.  **Select & Justify:** Choose the most promising strategy and justify your choice.
4.  **Implement Feature:** Write a Python function `create_feature(df)` that implements your chosen strategy. The function must be self-contained and not rely on any external variables other than the input DataFrame. All necessary imports must be inside the function.
5.  **Output:** Format your response as a JSON object. The `code` field must contain the full Python code for your `create_feature` function as a single string.

**Response Format:**
`{format_instructions}`
"""
