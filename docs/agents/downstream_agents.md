---
team_name: "Downstream Agents"
objective: "Implement, validate, and deploy features into the recommender pipeline."
agents:
  - name: "FeatureRealizationAgent"
    role: "Implements candidate features as Python functions, using LLMs to generate code."
    primary_tools: ["generate_function_code", "validate_feature"]
  - name: "EvaluationAgent"
    role: "Evaluates feature impact on model performance."
    primary_tools: ["run_evaluation", "generate_report"]
primary_output: "A set of realized, validated features and evaluation results."
---

### Workflow Overview
Downstream Agents receive optimized feature definitions from the Strategy Team. The `FeatureRealizationAgent` implements features in code, and the `EvaluationAgent` benchmarks their impact, producing results for both human review and feedback to upstream agents.
