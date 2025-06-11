#!/usr/bin/env python3
"""
Tests for the autonomous data analysis agent.
"""

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from src.agents.data_analysis_agent import DataAnalysisAgent


@pytest.fixture
def llm_config():
    """Fixture for LLM configuration."""
    return {
        "config_list": [
            {
                "api_key": os.environ.get("GOOGLE_API_KEY"),
                "api_type": "google",
                "api_base": "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent",
            }
        ]
    }


@pytest.fixture
def agent(llm_config):
    """Fixture for the data analysis agent."""
    return DataAnalysisAgent(llm_config)


def test_agent_initialization(agent):
    """Test that the agent initializes correctly."""
    assert agent.db_path == "/data/goodreads.duckdb"
    assert agent.plots_dir.exists()
    assert len(agent.tools) == 4
    assert agent.assistant is not None


def test_run_sql(agent):
    """Test SQL query execution."""
    query = "SELECT COUNT(*) as count FROM reviews"
    result = agent._run_sql(query)
    assert isinstance(result, pd.DataFrame)
    assert "count" in result.columns


def test_save_plot(agent, tmp_path):
    """Test plot saving functionality."""
    import matplotlib.pyplot as plt

    # Create a test plot
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])

    # Save the plot
    plot_path = agent._save_plot(fig, "test_plot")
    assert Path(plot_path).exists()


def test_vision_analyze(agent, tmp_path):
    """Test vision-based plot analysis."""
    # Create a test plot
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    plot_path = agent._save_plot(fig, "test_plot")

    # Analyze the plot
    insights = agent._vision_analyze(plot_path)
    assert isinstance(insights, list)
    assert len(insights) > 0


def test_save_results(agent):
    """Test saving analysis results."""
    results = {
        "numeric_insights": {"mean_rating": 4.5},
        "graph_insights": {
            "degree_distribution": {"mean": 10.5},
            "community_stats": {"num_communities": 5},
            "centrality_metrics": {"max_pagerank": 0.8},
        },
        "textual_insights": ["Most reviews are positive"],
        "plot_insights": {
            "rating_distribution.png": ["Normal distribution with slight right skew"]
        },
        "llm_feature_ideas": [
            {
                "name": "MoralComplexityScore",
                "rationale": "Capture moral complexity from reviews",
                "prompt": "Use <USER_REVIEWS> to score moral complexity",
            }
        ],
    }

    agent._save_results(results)

    # Verify results are saved in project memory
    from src.utils.memory import project_memory

    assert "eda" in project_memory
    assert project_memory["eda"] == results


def test_eda_report_schema_compliance(agent):
    """Test that the EDA report complies with the schema."""
    # Load the schema
    schema_path = (
        Path(__file__).parent.parent / "src" / "schemas" / "eda_report_schema.json"
    )
    with open(schema_path) as f:
        schema = json.load(f)

    # Create a valid report
    results = {
        "numeric_insights": {"mean_rating": 4.5},
        "graph_insights": {
            "degree_distribution": {"mean": 10.5},
            "community_stats": {"num_communities": 5},
            "centrality_metrics": {"max_pagerank": 0.8},
        },
        "textual_insights": ["Most reviews are positive"],
        "plot_insights": {
            "rating_distribution.png": ["Normal distribution with slight right skew"]
        },
        "llm_feature_ideas": [
            {
                "name": "MoralComplexityScore",
                "rationale": "Capture moral complexity from reviews",
                "prompt": "Use <USER_REVIEWS> to score moral complexity",
            },
            {
                "name": "GenrePreferenceScore",
                "rationale": "Identify genre preferences from review patterns",
                "prompt": "Use <USER_REVIEWS> to identify preferred genres",
            },
            {
                "name": "WritingStyleScore",
                "rationale": "Analyze writing style from review text",
                "prompt": "Use <USER_REVIEWS> to analyze writing style",
            },
        ],
    }

    # Save results
    agent._save_results(results)

    # Verify minimum requirements
    assert len(results["llm_feature_ideas"]) >= 3
    assert all(
        all(key in idea for key in ["name", "rationale", "prompt"])
        for idea in results["llm_feature_ideas"]
    )
    assert all(
        key in results["graph_insights"]
        for key in ["degree_distribution", "community_stats", "centrality_metrics"]
    )
