# scripts/test_prompt.py
import sys
from pathlib import Path

# Add project root to Python path for proper imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.prompt_utils import load_prompt

if __name__ == "__main__":
    print("--- Testing Prompt Rendering with Dynamic DB Schema ---")

    # We can use any agent prompt, as they all inherit the base template
    # that includes the database schema.
    test_template = "agents/discovery_team/data_representer.j2"
    print(f"Loading template: {test_template}\n")

    try:
        rendered_prompt = load_prompt(test_template)
        print("--- RENDERED PROMPT ---")
        print(rendered_prompt)
        print("\n--- END OF PROMPT ---")

        if "ERROR: Could not retrieve database schema." in rendered_prompt:
            print("\n\n❌ TEST FAILED: The database schema could not be retrieved.")
        elif "TABLE:" in rendered_prompt:
            print(
                "\n\n✅ TEST PASSED: The database schema was successfully injected into the prompt."
            )
        else:
            print(
                "\n\n❌ TEST FAILED: The database schema was not found in the rendered prompt."
            )
            print(
                "Please ensure the database file exists at 'data/goodreads_curated.duckdb' and is valid."
            )

    except Exception as e:
        print(f"\n\n❌ TEST FAILED: An exception occurred during prompt rendering: {e}")
