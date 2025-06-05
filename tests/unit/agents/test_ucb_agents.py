from unittest.mock import patch

import pytest

from vulcan.agents.ucb_agents import (
    IdeateNewAgent,
    LLMRowAgent,
    MutateExistingAgent,
    RefineTopAgent,
    ReflectAndRefineAgent,
)
from vulcan.evolution.progressive_orchestrator import FeatureCandidate
from vulcan.schemas import FeatureDefinition, FeatureType


@pytest.fixture
def mock_llm():
    with patch("vulcan.agents.ucb_agents.llm") as mock_llm_gen:
        mock_llm_gen.generate.return_value = "mocked llm response"
        yield mock_llm_gen


class TestBaseUCBAgent:
    def test_ucb1_no_pulls(self):
        agent = IdeateNewAgent("test")
        assert agent.get_ucb1(total_rounds=10) == float("inf")

    def test_ucb1_calculation(self):
        agent = IdeateNewAgent("test")
        agent.count = 5
        agent.reward_sum = 2.5
        # avg_reward = 0.5
        # bonus = sqrt(2 * log(10) / 5) approx 0.959
        # ucb approx 1.459
        assert agent.get_ucb1(total_rounds=10) == pytest.approx(1.459, 0.001)


class TestUCBAgents:
    def test_ideate_new_agent(self, mock_llm):
        agent = IdeateNewAgent("ideate_new")
        mock_llm.generate.side_effect = [
            "Step 1: reason",
            "result = df.mean()",
        ]
        feature = agent.select(context="test_context")
        assert isinstance(feature, FeatureDefinition)
        assert feature.feature_type == FeatureType.CODE_BASED
        assert "feat_new_" in feature.name
        assert "reason" in feature.llm_chain_of_thought_reasoning
        assert "df.mean" in feature.code
        assert mock_llm.generate.call_count == 2

    def test_refine_top_agent(self, mock_llm):
        agent = RefineTopAgent("refine_top")
        base_feature_def = FeatureDefinition(
            name="base_feat",
            description="base desc",
            feature_type=FeatureType.CODE_BASED,
            code="result = df.sum()",
        )
        base_candidate = FeatureCandidate(
            feature=base_feature_def, score=0.8, generation=1
        )
        mock_llm.generate.side_effect = ["reflection reasoning", "refined code"]

        feature = agent.select(existing_features=[base_candidate])

        assert isinstance(feature, FeatureDefinition)
        assert feature.name == "base_feat_refined"
        assert "reflection reasoning" in feature.llm_chain_of_thought_reasoning
        assert "refined code" in feature.code
        assert mock_llm.generate.call_count == 2

    def test_mutate_existing_agent(self, mock_llm):
        agent = MutateExistingAgent("mutate_existing")
        base_feature_def = FeatureDefinition(
            name="base_feat",
            description="base desc",
            feature_type=FeatureType.CODE_BASED,
            code="result = df.sum()",
        )
        base_candidate = FeatureCandidate(
            feature=base_feature_def, score=0.8, generation=1
        )
        mock_llm.generate.return_value = "mutated code"

        feature = agent.select(existing_features=[base_candidate])

        assert isinstance(feature, FeatureDefinition)
        assert feature.name == "base_feat_mutated"
        assert "mutated code" in feature.code
        assert mock_llm.generate.call_count == 1

    def test_llm_row_agent(self, mock_llm):
        agent = LLMRowAgent("llm_row")
        feature = agent.select(
            data_rows=[{"text": "a"}, {"text": "b"}], text_columns=["text"]
        )

        assert isinstance(feature, FeatureDefinition)
        assert feature.feature_type == FeatureType.LLM_BASED
        assert feature.llm_prompt is not None
        assert feature.text_columns == ["text"]

    def test_reflect_and_refine_agent(self, mock_llm):
        agent = ReflectAndRefineAgent("reflect_refine")
        base_feature_def = FeatureDefinition(
            name="base_feat",
            description="base desc",
            feature_type=FeatureType.CODE_BASED,
            code="result = df.sum()",
        )
        base_candidate = FeatureCandidate(
            feature=base_feature_def, score=0.8, generation=1
        )
        mock_llm.generate.side_effect = ["reflection reasoning", "refined code"]

        feature = agent.select(evaluated_feature=base_candidate)
        assert isinstance(feature, FeatureDefinition)
        assert "reflected_refined" in feature.name
        assert "reflection reasoning" in feature.llm_chain_of_thought_reasoning
        assert "refined code" in feature.code
        assert mock_llm.generate.call_count == 2
