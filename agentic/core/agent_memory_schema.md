# AgentMemory Serialization Schema

## Purpose
Defines what is stored in AgentMemory, how it is serialized, and guidelines for extending memory for new agentic workflows.

## Top-Level Structure
AgentMemory is serialized as a JSON object (dict) with the following recommended keys:

- `insights`: List of discovered insights (each a dict, e.g., `{ "title": ..., "finding": ..., ... }`)
- `features`: List of engineered features (dicts or code snippets)
- `scratchpad`: Dict for intermediate agent state, e.g., `{ "step": 2, "notes": "..." }`
- `run_metadata`: Dict with run-specific info (timestamps, run_id, etc)
- `artifacts`: List of paths or references to generated artifacts (plots, files)
- `logs`: Optional, list of log entries or events

## Example
```json
{
  "insights": [
    { "title": "Correlation found", "finding": "...", "rationale": "..." }
  ],
  "features": [
    { "name": "rating_popularity_momentum", "code": "def ..." }
  ],
  "scratchpad": {
    "step": 2,
    "notes": "Feature engineered."
  },
  "run_metadata": {
    "run_id": "20240617_123456",
    "started": "2024-06-17T12:34:56Z"
  },
  "artifacts": ["plots/plot1.png"],
  "logs": [
    { "event": "start", "timestamp": "..." },
    { "event": "feature_saved", "timestamp": "..." }
  ]
}
```

## Guidelines
- All keys are optional but recommended for robust workflows.
- Use `scratchpad` for any agent-specific intermediate state.
- Extend with new keys as needed for your workflow, but document them here.
- Keep all values JSON-serializable.
