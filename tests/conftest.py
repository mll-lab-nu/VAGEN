import os
import sys

# tests/ is not a package, so its own helpers have to be importable by plain name.
sys.path.insert(0, os.path.dirname(__file__))
