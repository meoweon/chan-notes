"""Script bootstrap: ensure the repository root is importable.

Usage (first line of a script):

    import _path  # noqa: F401
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
