---
team_name: "Insight Discovery Team"
objective: "Perform a comprehensive Exploratory Data Analysis (EDA) on the curated database to find novel patterns and generate a structured Insight Report."
agents:
  - name: "DataRepresenter"
    role: "Creates and documents SQL views to simplify data access for the team."
    primary_tools: ["create_analysis_view"]
  - name: "QuantitativeAnalyst"
    role: "Runs statistical tests, generates quantitative plots, and uses vision models to find quantitative patterns."
    primary_tools: ["run_sql_query", "vision_tool", "add_insight_to_report"]
  - name: "PatternSeeker"
    role: "Finds non-obvious correlations, anomalies, and qualitative insights."
    primary_tools: ["run_sql_query", "add_insight_to_report"]
primary_output: "A list of structured Insight objects saved to the SessionState."
---

### Workflow Overview
This team operates as a collaborative GroupChat managed by the `SmartGroupChatManager`. The process begins with the `DataRepresenter` creating initial views. The `QuantitativeAnalyst` and `PatternSeeker` then work in parallel to analyze the data from different perspectives, surfacing both quantitative and qualitative insights. All findings are added to the Insight Report for downstream agents.
