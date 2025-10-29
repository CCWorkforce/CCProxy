#!/usr/bin/env python3
"""Fix all commented out closing parentheses/brackets that are causing syntax errors."""

import re
from pathlib import Path
from typing import Any


def fix_commented_syntax(filepath: Path) -> bool:
    """Fix commented out closing parentheses and brackets."""
    try:
        content = filepath.read_text()
        original = content

        # Patterns to fix - commented out closing parens/brackets with specific indentation
        patterns = [
            (r"^#(\s+\))", r"\1"),  # Commented closing paren with leading spaces
            (r"^#(\s+\])", r"\1"),  # Commented closing bracket with leading spaces
            (r"^#(\s+\})", r"\1"),  # Commented closing brace with leading spaces
        ]

        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

        if content != original:
            filepath.write_text(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def main() -> Any:
    """Fix all Python files with commented syntax issues."""
    # Only fix our project files, not .venv
    directories = [
        "/Volumes/Data/OpenSources/AI4Org/CCProxy/benchmarks",
        "/Volumes/Data/OpenSources/AI4Org/CCProxy/tests",
        "/Volumes/Data/OpenSources/AI4Org/CCProxy/ccproxy",
    ]

    fixed_files = []
    for dir_path in directories:
        directory = Path(dir_path)
        if directory.exists():
            for py_file in directory.rglob("*.py"):
                if fix_commented_syntax(py_file):
                    fixed_files.append(py_file)
                    print(f"✓ Fixed: {py_file.name}")

    print(f"\nFixed {len(fixed_files)} files")
    return len(fixed_files)


if __name__ == "__main__":
    main()
