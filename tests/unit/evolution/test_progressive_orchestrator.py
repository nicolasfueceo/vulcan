from unittest.mock import MagicMock, patch

import pytest

from vulcan.evolution.progressive_orchestrator import (
    FeatureCandidate,
    ProgressiveEvolutionOrchestrator,
)
from vulcan.schemas import (
    ExperimentConfig,
    FeatureDefinition,
    FeatureType,
    VulcanConfig,
)


@pytest.fixture
def mock_config():
    config = MagicMock(spec=VulcanConfig)
    config.experiment = MagicMock(spec=ExperimentConfig)
    config.experiment.population_size = 10
    config.experiment.generation_size = 5
    config.experiment.max_repair_attempts = 1
    config.experiment.output_dir = "test_experiments"
    config.experiment.mutation_rate = 0.5

    # Attach a mock for the 'get' method
    config.experiment.get = MagicMock()

    # Configure the 'get' method on the mocked experiment object
    def get_side_effect(key, default=None):
        return {
            "early_stopping_patience": 3,
            "reflection_threshold": 0.6,
        }.get(key, default)

    config.experiment.get.side_effect = get_side_effect

    return config


@pytest.fixture
def orchestrator(mock_config):
    with (
        patch("vulcan.evolution.progressive_orchestrator.SummaryWriter"),
        patch("vulcan.evolution.progressive_orchestrator.FeatureExecutor"),
        patch("vulcan.evolution.progressive_orchestrator.RecommendationEvaluator"),
    ):
        orchestrator = ProgressiveEvolutionOrchestrator(
            config=mock_config,
            results_manager=MagicMock(),
        )
        # Mock the agents inside the orchestrator
        orchestrator.agents = {
            "IdeateNew": MagicMock(),
            "RefineTop": MagicMock(),
        }
        orchestrator.reflection_agent = MagicMock()
        orchestrator.logger = MagicMock()
    return orchestrator


@pytest.mark.asyncio
async def test_choose_action_ucb(orchestrator):
    # Mock get_ucb1 scores for agents
    orchestrator.agents["IdeateNew"].get_ucb1.return_value = 1.0
    orchestrator.agents["RefineTop"].get_ucb1.return_value = 2.0

    orchestrator.total_ucb_pulls = 10

    name, agent = orchestrator._choose_action_ucb()

    assert name == "RefineTop"
    assert agent == orchestrator.agents["RefineTop"]
    orchestrator.agents["IdeateNew"].get_ucb1.assert_called_with(10)
    orchestrator.agents["RefineTop"].get_ucb1.assert_called_with(10)


@pytest.mark.asyncio
async def test_generate_candidates(orchestrator):
    mock_agent = MagicMock()
    mock_agent.name = "IdeateNew"
    mock_agent.select.return_value = FeatureDefinition(
        name="test_feat",
        description="test_desc",
        feature_type=FeatureType.CODE_BASED,
        code="pass",
    )

    mock_data_context = MagicMock()
    mock_data_context.data_schema = {}
    mock_data_context.n_users = 100
    mock_data_context.n_items = 50

    orchestrator.generation_size = 3
    candidates = await orchestrator._generate_candidates(mock_agent, mock_data_context)

    assert len(candidates) == 3
    assert mock_agent.select.call_count == 3
    assert isinstance(candidates[0], FeatureCandidate)
    assert candidates[0].feature.name == "test_feat"


def test_log_feature_to_disk(orchestrator, tmp_path):
    orchestrator.run_dir = tmp_path
    feature = FeatureDefinition(
        name="log_test_feat",
        description="log test desc",
        feature_type=FeatureType.CODE_BASED,
        code="pass",
        llm_chain_of_thought_reasoning="some reasoning",
    )
    metrics = {"reward": 0.9}
    orchestrator.config.experiment.get.return_value = 0.5  # acceptance_threshold

    orchestrator._log_feature_to_disk(feature, metrics)

    log_file = tmp_path / "features" / "log_test_feat.json"
    assert log_file.exists()
    import json

    with open(log_file) as f:
        data = json.load(f)
    assert data["name"] == "log_test_feat"
    assert data["metrics"]["reward"] == 0.9
    assert data["accepted"] is True
