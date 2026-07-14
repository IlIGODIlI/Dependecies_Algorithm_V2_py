"""
repl.py — Root-level backwards-compatibility wrapper.

Allows running `python repl.py` from the project root,
just like the original flat layout.

Delegates to src/vela_causal/repl.py.
"""
import sys
import os

# Add src/ to the module search path so the vela_causal package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from vela_causal.repl import main

if __name__ == "__main__":
    main()
