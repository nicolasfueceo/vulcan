# scripts/test_finalize_hypotheses.py
"""
Script to test the finalize_hypotheses tool with a fake hypothesis input.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.tools import get_finalize_hypotheses_tool
from src.utils.session_state import SessionState
from src.utils.run_utils import init_run

def main():
    init_run()
    session_state = SessionState()
    finalize_hypotheses = get_finalize_hypotheses_tool(session_state)
    fake_hypotheses = [
        {
            "summary": "Users who read more than 10 books per year have higher average ratings.",
            "rationale": "Frequent readers may be more positive in their ratings, which could inform segmentation."
        },
        {
            "summary": "Books with multiple authors receive higher ratings.",
            "rationale": "Collaboration may improve book quality, leading to higher ratings."
        }
    ]
    result = finalize_hypotheses(fake_hypotheses)
    print("Finalize Hypotheses Result:", result)

if __name__ == "__main__":
    main()
