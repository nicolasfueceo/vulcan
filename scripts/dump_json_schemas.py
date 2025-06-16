# scripts/dump_json_schemas.py
"""
Dumps the JSON schema for Hypothesis and CandidateFeature models to files for prompt inclusion/documentation.
"""
from src.schemas.models import Hypothesis, CandidateFeature
import json

hypothesis_schema = Hypothesis.model_json_schema()
candidate_feature_schema = CandidateFeature.model_json_schema()

with open("generated_prompts/Hypothesis.schema.json", "w") as f:
    json.dump(hypothesis_schema, f, indent=2)

with open("generated_prompts/CandidateFeature.schema.json", "w") as f:
    json.dump(candidate_feature_schema, f, indent=2)

print("Schemas dumped to generated_prompts/.")
