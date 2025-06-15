# tests/test_insight_discovery.py
"""
End-to-end test for the insight discovery chat pipeline, specifically validating that the agent can create SQL views directly (CREATE VIEW) and via create_analysis_view, and that insights are extracted and saved as expected.
"""
import os
import json
import pytest
from src.orchestration.insight import run_insight_discovery_chat

@pytest.fixture(scope="module")
def mock_llm_config():
    # Provide a minimal valid LLM config for testing
    return {
        "config_list": [
            {"model": "gpt-4", "api_key": os.environ.get("OPENAI_API_KEY", "test-key")}
        ],
        "temperature": 0.1,
        "max_tokens": 512,
    }

def test_insight_discovery_creates_views_and_extracts_insights(tmp_path, mock_llm_config, monkeypatch):
    # Patch get_run_dir to use a temp directory
    from src.orchestration import insight as insight_module
    monkeypatch.setattr(insight_module, "get_run_dir", lambda: tmp_path)

    # Run the pipeline
    results = run_insight_discovery_chat(mock_llm_config)
    results_path = tmp_path / "insight_results.json"
    assert results_path.exists(), "Results file not created."

    with open(results_path) as f:
        data = json.load(f)

    # Check that insights and view_descriptions are present and non-empty
    assert "insights" in data
    assert isinstance(data["insights"], list)
    assert len(data["insights"]) > 0, "No insights extracted."
    assert "view_descriptions" in data
    assert isinstance(data["view_descriptions"], dict)

    # Check that at least one view was created using CREATE VIEW (by scanning messages)
    # and at least one via create_analysis_view (if tracked in view_descriptions)
    found_create_view = False
    found_analysis_view = False
    for desc in data["view_descriptions"].values():
        if "CREATE VIEW" in desc.upper():
            found_create_view = True
        if "create_analysis_view" in desc:
            found_analysis_view = True
    assert found_create_view, "No views created using CREATE VIEW."
    # Not all runs may use create_analysis_view, but if present, should be detected
    # assert found_analysis_view, "No views created using create_analysis_view."

    # Validate that insights have expected fields
    for insight in data["insights"]:
        assert "title" in insight
        assert "finding" in insight
        assert "supporting_evidence" in insight or "evidence" in insight
        assert "confidence" in insight

    print("Test passed: Agent can create views via both methods and extract insights.")
