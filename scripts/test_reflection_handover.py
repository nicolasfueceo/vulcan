import unittest
from unittest.mock import patch, MagicMock
from src.orchestrator import run_discovery_loop
from src.utils.session_state import SessionState

def make_fake_reflection():
    return {
        "next_steps": [
            "Explore the author-book relationships for underrepresented genres.",
            "Investigate books with high rating variance across genres."
        ],
        "novel_ideas": [
            "Analyze sentiment in book descriptions.",
            "Cluster authors by genre diversity."
        ],
        "expansion_ideas": [
            "Deep dive into top-rated authors in the 'Fantasy' genre.",
            "Expand on the relationship between book length and average rating."
        ],
        "summary": "Reflection summary here."
    }

class TestReflectionHandover(unittest.TestCase):
    @patch('autogen.UserProxyAgent.initiate_chat')
    def test_reflection_handover_to_discovery(self, mock_initiate_chat):
        from src.utils.run_utils import init_run
        init_run()  # Ensure run context is initialized
        session_state = SessionState()
        session_state.reflections.append(make_fake_reflection())
        # Patch agents to avoid real LLM calls
        with patch('src.orchestrator.get_insight_discovery_agents') as mock_agents:
            dummy_agent = MagicMock()
            agents_dict = {k: dummy_agent for k in ["QuantitativeAnalyst", "DataRepresenter", "PatternSeeker", "Hypothesizer"]}
            mock_agents.return_value = agents_dict
            with patch('src.orchestrator.autogen.UserProxyAgent'):
                with patch('src.orchestrator.SmartGroupChatManager'):
                    run_discovery_loop(session_state)
        # Check that the initial message contains reflection handover content
        called_args = mock_initiate_chat.call_args[1]  # kwargs
        msg = called_args['message']
        self.assertIn("Reflection Agent's Next Steps", msg)
        self.assertIn("Explore the author-book relationships", msg)
        self.assertIn("Novel Unexplored Ideas", msg)
        self.assertIn("Analyze sentiment in book descriptions", msg)
        self.assertIn("Promising Expansions", msg)
        self.assertIn("Deep dive into top-rated authors", msg)

if __name__ == "__main__":
    unittest.main()
