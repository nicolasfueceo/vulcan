import os
from pathlib import Path

def main():
    prompt_root = Path("src/prompts/agents")
    output_dir = Path("generated_prompts")
    output_dir.mkdir(exist_ok=True)

    for j2_file in prompt_root.rglob("*.j2"):
        role_name = j2_file.stem  # e.g., hypothesizer
        with open(j2_file, "r", encoding="utf-8") as src_f:
            prompt = src_f.read()
        with open(output_dir / f"{role_name}.txt", "w", encoding="utf-8") as out_f:
            out_f.write(prompt)
        print(f"Dumped {role_name} prompt to {output_dir / f'{role_name}.txt'}")

    print(f"All prompts dumped to {output_dir.resolve()}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
