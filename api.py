"""
api.py — Root-level backwards-compatibility wrapper.

Allows running `uvicorn api:app --reload --port 8000` from the project root,
just like the original flat layout.

Delegates to src/vela_causal/api.py.
"""
import sys
import os

# Add src/ to the module search path so the vela_causal package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from vela_causal.api import app  # noqa: F401 — uvicorn looks for `api:app`

if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
    except ImportError:
        print("FastAPI not available — install with: pip install fastapi uvicorn")
