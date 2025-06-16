from src.utils.run_utils import init_run
from src.utils.session_state import SessionState
from src.orchestrator import run_discovery_loop

def main():
    init_run()
    session_state = SessionState()
    session_state.reflections.append({
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
    })

    print("=== Running discovery loop with reflection handover ===")
    run_discovery_loop(session_state)
    print("=== Finished ===")

if __name__ == "__main__":
    main()
