"""
Test the Hypothesizer agent in isolation by mocking prior agent interactions.
Ensures that the Hypothesizer reads insights and calls finalize_hypotheses.
"""
import sys
from pathlib import Path
import pytest
from src.agents.discovery_team.insight_discovery_agents import get_insight_discovery_agents
from src.utils.tools import get_finalize_hypotheses_tool
from src.utils.run_utils import init_run
from src.schemas.models import Insight
from src.utils.session_state import SessionState
from src.orchestrator import get_llm_config_list

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class DummyLLM:
    def chat(self, *args, **kwargs):
        # Always return a dummy finalize_hypotheses tool call
        return {
            "role": "tool",
            "name": "finalize_hypotheses",
            "content": '[{"summary": "Books with high rating variance are more likely to be popular.", "rationale": "Analysis of user ratings shows a strong correlation between variance and popularity."}]',
        }

@pytest.fixture
def session_state():
    init_run()  # Ensure run context is initialized
    ss = SessionState()
    ss.set_state("insights", [
        Insight(
            title="High variance in ratings for popular books.",
            finding="Variance=2.1, Popularity=high",
            supporting_code=None,
            source_representation="vw_user_review_summary",
            plot_path=None,
            plot_interpretation=None,
            quality_score=None,
            metadata={},
            tables_used=["books"],
            reasoning_trace=["Variance calculated from ratings"]
        ),
        Insight(
            title="Shorter books have more reviews.",
            finding="Pages<200, Reviews>1000",
            supporting_code=None,
            source_representation="vw_book_stats",
            plot_path=None,
            plot_interpretation=None,
            quality_score=None,
            metadata={},
            tables_used=["books"],
            reasoning_trace=["Counted reviews for books under 200 pages"]
        ),
    ])
    ss.set_state("hypotheses", [])
    return ss

def test_hypothesizer_finalizes_hypotheses(session_state):
    llm_config = get_llm_config_list()
    agents = get_insight_discovery_agents(llm_config)
    hypothesizer = agents["Hypothesizer"]
    hypothesizer.register_tool(get_finalize_hypotheses_tool(session_state))

    # Simulate prior messages (insights added by other agents)
    prior_messages = [
        {"role": "assistant", "name": "QuantitativeAnalyst", "content": "add_insight_to_report: High variance in ratings for popular books."},
        {"role": "assistant", "name": "DataRepresenter", "content": "add_insight_to_report: Shorter books have more reviews."},
        {"role": "user", "name": "SystemCoordinator", "content": "Hypothesizer, please synthesize the insights and finalize hypotheses."},
    ]

    # Simulate chat turn with real LLM
    response = hypothesizer.generate_reply(prior_messages)
    assert response["name"] == "finalize_hypotheses"
    assert "summary" in response["content"]
    # Check that session_state.hypotheses is updated
    assert session_state.hypotheses and len(session_state.hypotheses) > 0, "Hypotheses should be populated after finalize_hypotheses."

if __name__ == "__main__":
    pytest.main([__file__])
