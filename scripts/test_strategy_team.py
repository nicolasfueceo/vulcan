# scripts/test_strategy_team.py
"""
Integration test for the streamlined Strategy Team pipeline.
- Sets up a fake session state with mock insights AND mock hypotheses
- Runs the streamlined strategy loop with the FeatureEngineer agent
- Asserts that features are generated and saved
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.orchestrator import run_strategy_loop
from src.utils.run_utils import init_run
from src.utils.session_state import Insight, SessionState
from src.schemas.models import Hypothesis, RealizedFeature, CandidateFeature


def make_mock_data():
    # Create both mock insights and mock hypotheses
    insights = [
        Insight(
            title="Users who read more than 10 books per year have higher average ratings.",
            finding="Frequent readers may be more positive in their ratings.",
            supporting_code="""
                SELECT user_id, AVG(rating) as avg_rating, COUNT(*) as num_books
                FROM ratings
                GROUP BY user_id
                HAVING COUNT(*) > 10
                ORDER BY avg_rating DESC
            """,
            source_representation="vw_frequent_readers",
            plot_path="plots/user_reading_frequency_vs_rating.png",
            plot_interpretation="Scatter plot showing positive correlation between number of books read and average rating.",
            quality_score=8.5,
            tables_used=["users", "ratings"],
            metadata={
                "analysis_round": 1, 
                "source": "initial_analysis",
                "rationale": "Users who read more books tend to rate more positively, possibly because they are more engaged with the platform."
            },
            reasoning_trace=[
                "Hypothesized that frequent readers might rate differently",
                "Verified with SQL query on ratings data",
                "Created visualization to confirm the trend",
                "Analyzed correlation between reading frequency and rating"
            ]
        ),
        Insight(
            title="Books with multiple authors receive higher ratings.",
            finding="Collaboration may improve book quality.",
            supporting_code="""
                SELECT b.book_id, COUNT(DISTINCT ba.author_id) as num_authors, AVG(r.rating) as avg_rating
                FROM books b
                JOIN book_authors ba ON b.book_id = ba.book_id
                JOIN ratings r ON b.book_id = r.book_id
                GROUP BY b.book_id
                HAVING COUNT(DISTINCT ba.author_id) > 1
                ORDER BY avg_rating DESC
            """,
            source_representation="vw_books_multi_author",
            plot_path="plots/author_count_vs_rating.png",
            plot_interpretation="Box plot showing higher median ratings for books with multiple authors.",
            quality_score=7.8,
            tables_used=["books", "book_authors", "ratings"],
            metadata={
                "analysis_round": 1, 
                "source": "initial_analysis",
                "rationale": "Books with multiple authors may benefit from diverse perspectives and complementary skills."
            },
            reasoning_trace=[
                "Noticed potential pattern in multi-author books",
                "Analyzed average ratings by author count",
                "Verified statistical significance of the finding"
            ]
        )
    ]
    
    # Create pre-generated hypotheses that would normally come from the discovery team
    hypotheses = [
        Hypothesis(
            summary="More engaged readers give more positive ratings",
            rationale="Frequent readers may have more context for comparison, are likely enthusiasts, and may appreciate a wider range of content."
        ),
        Hypothesis(
            summary="Collaborative authorship improves book quality",
            rationale="Multiple authors may bring diverse perspectives, complementary skills, and more rigorous editing to a book project."
        )
    ]
    
    return insights, hypotheses


from src.orchestrator import get_llm_config_list
from src.core.database import get_db_schema_string
from src.agents.strategy_team.strategy_team_agents import get_strategy_team_agents

def main():
    init_run()
    session_state = SessionState()
    
    # Set up both insights and pre-generated hypotheses in the session state
    insights, hypotheses = make_mock_data()
    session_state.insights = insights
    session_state.hypotheses = hypotheses  # Pre-populated hypotheses
    
    db_schema = get_db_schema_string()
    llm_config = get_llm_config_list() or {}
    strategy_agents = get_strategy_team_agents(llm_config=llm_config, db_schema=db_schema)
    
    print("[TEST] Running streamlined strategy loop with mock insights and hypotheses...")
    result = run_strategy_loop(session_state, strategy_agents, llm_config)
    print("[TEST] Strategy loop result:", result)
    
    # Check features were generated by our new FeatureEngineer agent
    features = getattr(session_state, "features", [])
    print(f"[TEST] Features generated: {len(features)}")
    for feature in features:
        print(feature)
        
    assert len(features) > 0, "No features were generated by the streamlined strategy loop!"
    for feature in features:
        assert getattr(feature, "name", None), "Feature missing name"
        assert getattr(feature, "code_str", None), "Feature missing implementation code"
        assert getattr(feature, "type", None), "Feature missing type"
        # If feature is a RealizedFeature, check for proper code implementation
        if isinstance(feature, RealizedFeature):
            assert feature.code_str.strip(), "Feature code is empty!"
        # If feature is a CandidateFeature, check for proper spec
        elif isinstance(feature, CandidateFeature):
            assert getattr(feature, "spec", None), "CandidateFeature missing spec"
            assert getattr(feature, "rationale", None), "CandidateFeature missing rationale"
    
    print("[TEST] Streamlined strategy team integration test PASSED.")


if __name__ == "__main__":
    main()
