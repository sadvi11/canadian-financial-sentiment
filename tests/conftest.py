"""Put the Flask app on the import path for the test suite.

The API lives in `05_api/`, which cannot be imported as a package because the
directory name starts with a digit. Tests therefore import `app` directly and
this file is what makes that resolve — without it, collection fails with
ModuleNotFoundError before a single test runs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "05_api"))
