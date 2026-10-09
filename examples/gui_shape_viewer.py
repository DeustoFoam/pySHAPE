"""Launcher for the pySHAPE GUI.

The implementation lives in ``pyshape.gui`` so it can be imported as
``python -m pyshape.gui`` as well. This script is kept for convenience and
backward compatibility with the path mentioned in the README.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pyshape.gui import main

if __name__ == "__main__":
    main()
