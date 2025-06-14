# scripts/safe_delete.py

import argparse
import ast
import os
from pathlib import Path
from typing import List


# Define some colors for terminal output for better readability
class BColors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def path_to_module(path: Path, project_root: Path) -> str:
    """Converts a file path to a Python module import string."""
    # Get the path relative to the project's source directory (e.g., 'src')
    try:
        # Assuming your source code is in 'src'
        src_root = project_root / "src"
        relative_path = path.relative_to(src_root)
    except ValueError:
        # Fallback to project root if not in 'src'
        relative_path = path.relative_to(project_root)

    # Remove the .py extension and replace path separators with dots
    module_path = str(relative_path.with_suffix("")).replace(os.path.sep, ".")
    return module_path


def find_importers(target_file: Path, project_root: Path) -> List[Path]:
    """
    Scans the entire project for Python files that import the target file.

    Args:
        target_file: The Path object of the file to be checked.
        project_root: The root directory of the project to scan.

    Returns:
        A list of paths to files that import the target file.
    """
    importers = []
    target_module_str = path_to_module(target_file, project_root)

    # Use rglob to recursively find all Python files
    for file_path in project_root.rglob("*.py"):
        # Skip the file itself and any files in cache directories
        if file_path.samefile(target_file) or "__pycache__" in str(file_path):
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Parse the file content into an Abstract Syntax Tree
                tree = ast.parse(content, filename=str(file_path))

                # Walk through the tree to find all import nodes
                for node in ast.walk(tree):
                    is_importer = False
                    if isinstance(node, ast.Import):
                        # Handles cases like `import src.utils.memory`
                        for alias in node.names:
                            if alias.name == target_module_str:
                                is_importer = True
                                break
                    elif isinstance(node, ast.ImportFrom):
                        # Handles cases like `from src.utils.memory import get_mem`
                        # This also handles relative imports by resolving the path
                        if node.module:
                            # Resolve relative imports (e.g., from . import utils)
                            if node.level > 0:
                                parent_dir = file_path.parent
                                resolved_path = parent_dir
                                for _ in range(node.level - 1):
                                    resolved_path = resolved_path.parent
                                imported_module_path = path_to_module(
                                    resolved_path / f"{node.module}.py", project_root
                                )
                            else:
                                imported_module_path = node.module

                            if imported_module_path == target_module_str:
                                is_importer = True

                    if is_importer:
                        importers.append(file_path)
                        # Once found, no need to check other nodes in this file
                        break
        except (SyntaxError, UnicodeDecodeError) as e:
            # Silently ignore files that can't be parsed
            # print(f"{BColors.WARNING}Warning: Could not parse {file_path}: {e}{BColors.ENDC}")
            pass
        except Exception as e:
            print(
                f"{BColors.FAIL}An unexpected error occurred while processing {file_path}: {e}{BColors.ENDC}"
            )

    return list(set(importers))  # Return unique list


def safe_delete(target_file_path: str):
    """
    Checks if a file is safe to delete and prompts the user for confirmation.
    """
    project_root = Path(__file__).parent.parent.resolve()
    target_file = Path(target_file_path).resolve()

    if not target_file.exists():
        print(
            f"{BColors.FAIL}Error: File not found at '{target_file_path}'.{BColors.ENDC}"
        )
        return

    print(
        f"\n{BColors.HEADER}Scanning project for imports of '{target_file.name}'...{BColors.ENDC}"
    )

    importers = find_importers(target_file, project_root)

    if importers:
        print(f"{BColors.FAIL}{BColors.BOLD}Deletion Unsafe!{BColors.ENDC}")
        print(
            f"The file '{target_file.name}' is imported by the following {len(importers)} file(s):"
        )
        for importer in importers:
            print(f"  - {importer.relative_to(project_root)}")
        print("\nPlease remove these imports before attempting deletion again.")
    else:
        print(f"{BColors.OKGREEN}{BColors.BOLD}Deletion Safe.{BColors.ENDC}")
        print(f"No active imports of '{target_file.name}' were found in the project.")

        try:
            choice = input(
                f"Are you sure you want to permanently delete this file? [y/N]: "
            )
            if choice.lower() == "y":
                os.remove(target_file)
                print(
                    f"{BColors.OKGREEN}Successfully deleted '{target_file.name}'.{BColors.ENDC}"
                )
            else:
                print("Deletion cancelled.")
        except KeyboardInterrupt:
            print("\nDeletion cancelled.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to safely check for dependencies before deleting a Python file.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "files",
        metavar="FILE",
        type=str,
        nargs="+",
        help="One or more paths to Python files you want to delete.",
    )
    args = parser.parse_args()

    for file_to_delete in args.files:
        safe_delete(file_to_delete)
