#!/usr/bin/env python3
"""
Script to generate a comprehensive markdown document of the src directory
for LLM analysis. Includes file tree and complete file contents.
"""

import fnmatch
import mimetypes
import os
from datetime import datetime
from pathlib import Path


def load_gitignore_patterns():
    """Load and parse .gitignore patterns."""
    patterns = []
    if os.path.exists(".gitignore"):
        with open(".gitignore", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line)
    return patterns


def should_ignore(path, gitignore_patterns):
    """Check if a path should be ignored based on gitignore patterns."""
    # Convert path to relative path
    rel_path = os.path.relpath(path)

    # Always ignore .git directory and its contents
    if rel_path.startswith(".git/"):
        return True

    # Check each pattern
    for pattern in gitignore_patterns:
        # Handle directory patterns
        if pattern.endswith("/"):
            if fnmatch.fnmatch(rel_path, pattern[:-1]) or fnmatch.fnmatch(
                rel_path, pattern + "*"
            ):
                return True
        # Handle file patterns
        elif fnmatch.fnmatch(rel_path, pattern):
            return True
    return False


def is_text_file(file_path):
    """Check if a file is a text file that should be included."""
    # Skip common binary and cache file extensions
    skip_extensions = {
        ".pyc",
        ".pyo",
        ".pyd",
        ".so",
        ".dll",
        ".dylib",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".ico",
        ".pdf",
        ".zip",
        ".tar",
        ".gz",
        ".rar",
        ".7z",
        ".mp3",
        ".mp4",
        ".avi",
        ".mov",
        ".wav",
        ".db",
        ".sqlite",
        ".sqlite3",
        ".duckdb",
    }

    file_ext = Path(file_path).suffix.lower()
    if file_ext in skip_extensions:
        return False

    # Skip __pycache__ directories
    if "__pycache__" in str(file_path):
        return False

    # Check MIME type
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type and mime_type.startswith("text/"):
        return True

    # Common code file extensions
    code_extensions = {
        ".py",
        ".js",
        ".ts",
        ".html",
        ".css",
        ".json",
        ".yml",
        ".yaml",
        ".md",
        ".txt",
        ".cfg",
        ".conf",
        ".ini",
        ".toml",
        ".xml",
        ".sql",
        ".sh",
        ".bat",
        ".dockerfile",
        ".gitignore",
        ".env",
    }

    return file_ext in code_extensions or file_ext == ""


def generate_file_tree(
    directory, prefix="", max_depth=10, current_depth=0, gitignore_patterns=None
):
    """Generate a visual file tree representation."""
    if current_depth > max_depth:
        return ""

    if gitignore_patterns is None:
        gitignore_patterns = []

    items = []
    try:
        entries = sorted(os.listdir(directory))
        for entry in entries:
            path = os.path.join(directory, entry)
            if should_ignore(path, gitignore_patterns):
                continue
            if os.path.isdir(path):
                items.append((entry, path, True))
            else:
                items.append((entry, path, False))
    except PermissionError:
        return f"{prefix}[Permission Denied]\n"

    tree_output = ""
    for i, (name, path, is_dir) in enumerate(items):
        is_last = i == len(items) - 1
        current_prefix = "└── " if is_last else "├── "
        next_prefix = "    " if is_last else "│   "

        if is_dir:
            tree_output += f"{prefix}{current_prefix}{name}/\n"
            tree_output += generate_file_tree(
                path,
                prefix + next_prefix,
                max_depth,
                current_depth + 1,
                gitignore_patterns,
            )
        else:
            tree_output += f"{prefix}{current_prefix}{name}\n"

    return tree_output


def get_language_from_extension(file_path):
    """Get the programming language for syntax highlighting based on file extension."""
    ext = Path(file_path).suffix.lower()
    language_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".html": "html",
        ".css": "css",
        ".json": "json",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".sql": "sql",
        ".sh": "bash",
        ".md": "markdown",
        ".xml": "xml",
        ".toml": "toml",
        ".ini": "ini",
        ".cfg": "ini",
        ".conf": "ini",
    }
    return language_map.get(ext, "text")


def read_file_content(file_path):
    """Read file content safely with encoding detection."""
    encodings = ["utf-8", "utf-16", "latin-1", "cp1252"]

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            return f"[Error reading file: {e}]"

    return "[Unable to decode file content]"


def generate_src_documentation(src_dir="src", output_file="src_documentation.md"):
    """Generate comprehensive markdown documentation of the src directory."""

    if not os.path.exists(src_dir):
        print(f"Error: Directory '{src_dir}' does not exist")
        return

    # Load gitignore patterns
    gitignore_patterns = load_gitignore_patterns()

    with open(output_file, "w", encoding="utf-8") as f:
        # Header
        f.write("# Source Code Documentation\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(
            "This document contains the complete source code structure and contents "
        )
        f.write(f"of the `{src_dir}` directory.\n\n")

        # Full Directory Tree
        f.write("## 📁 Full Directory Structure\n\n")
        f.write("```\n")
        f.write(
            generate_file_tree(".", gitignore_patterns=gitignore_patterns)
        )  # Generate tree from root directory
        f.write("```\n\n")

        # File Contents (only for src directory)
        f.write("## 📄 File Contents (src directory only)\n\n")

        # Collect all files from src directory
        all_files = []
        for root, dirs, files in os.walk(src_dir):
            # Skip __pycache__ directories
            dirs[:] = [d for d in dirs if d != "__pycache__"]

            for file in files:
                file_path = os.path.join(root, file)
                if should_ignore(file_path, gitignore_patterns):
                    continue
                if is_text_file(file_path):
                    rel_path = os.path.relpath(file_path, src_dir)
                    all_files.append((file_path, rel_path))

        # Sort files for consistent output
        all_files.sort(key=lambda x: x[1])

        # Process each file
        for file_path, rel_path in all_files:
            try:
                file_size = os.path.getsize(file_path)

                f.write(f"### `{rel_path}`\n\n")
                f.write(f"**File size:** {file_size:,} bytes\n\n")

                # Skip very large files
                if file_size > 100_000:  # 100KB limit
                    f.write("*[File too large to display - over 100KB]*\n\n")
                    continue

                content = read_file_content(file_path)

                if content.strip():
                    language = get_language_from_extension(file_path)
                    f.write(f"```{language}\n")
                    f.write(content)
                    if not content.endswith("\n"):
                        f.write("\n")
                    f.write("```\n\n")
                else:
                    f.write("*[Empty file]*\n\n")

            except Exception as e:
                f.write(f"*[Error processing file: {e}]*\n\n")

        # Summary
        f.write("## 📊 Summary\n\n")
        f.write(f"- **Total files processed:** {len(all_files)}\n")
        f.write(f"- **Directory:** `{src_dir}`\n")
        f.write(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        f.write("*This documentation was generated automatically. ")
        f.write(
            "It includes all text-based source files and their complete contents.*\n"
        )

    print(f"✅ Documentation generated: {output_file}")
    print(f"📄 Processed {len(all_files)} files from {src_dir}")

    # Show file size
    doc_size = os.path.getsize(output_file)
    print(f"📦 Output file size: {doc_size:,} bytes ({doc_size / 1024:.1f} KB)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate markdown documentation of source directory"
    )
    parser.add_argument(
        "--src", default="src", help="Source directory to document (default: src)"
    )
    parser.add_argument(
        "--output",
        default="src_documentation.md",
        help="Output markdown file (default: src_documentation.md)",
    )
    parser.add_argument(
        "--include-tests", action="store_true", help="Also include tests directory"
    )

    args = parser.parse_args()

    # Generate main src documentation
    generate_src_documentation(args.src, args.output)

    # Optionally include tests
    if args.include_tests and os.path.exists("tests"):
        tests_output = args.output.replace(".md", "_with_tests.md")

        with open(tests_output, "w", encoding="utf-8") as f:
            # Combine src and tests
            f.write("# Complete Source Code Documentation (Including Tests)\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Read src documentation
            with open(args.output, "r", encoding="utf-8") as src_file:
                src_content = src_file.read()
                # Skip the header and add src content
                src_lines = src_content.split("\n")
                f.write("\n".join(src_lines[4:]))  # Skip first 4 lines (header)

            f.write("\n\n# 🧪 Tests Directory\n\n")

        # Generate tests documentation and append
        print("\n🧪 Also processing tests directory...")
        generate_src_documentation("tests", "temp_tests.md")

        # Append tests content
        with open(tests_output, "a", encoding="utf-8") as f:
            with open("temp_tests.md", "r", encoding="utf-8") as tests_file:
                tests_content = tests_file.read()
                # Skip header and add tests content
                tests_lines = tests_content.split("\n")
                f.write("\n".join(tests_lines[7:]))  # Skip header lines

        # Cleanup
        os.remove("temp_tests.md")
        print(f"✅ Combined documentation generated: {tests_output}")
