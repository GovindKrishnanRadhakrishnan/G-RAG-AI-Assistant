"""
conftest.py — pytest configuration for the Research Assistant project.

Ensures the project root is on sys.path so that `src.*` imports resolve
correctly regardless of the working directory pytest is invoked from.
"""
import sys
from pathlib import Path

# Insert project root as first entry so it always wins over stale sys.path entries
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
