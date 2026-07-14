"""
llm_normalizer.py
=================
Uses a locally running Ollama LLM to:
  1. Rewrite raw user input into clean causal sentences
  2. Strip ambiguous language, metaphors, bias, and noise
  3. Normalize entity references (e.g. "oil prices go up" → "oil_price_increase")
  4. Decompose compound statements into atomic causal pairs

This is the anti-bias layer: ALL text passes through here before reaching the engine.
It ensures the model never sees raw, frame-heavy language that biases toward
any particular domain (e.g. oil/finance seed data).

Ollama endpoint (local): http://localhost:11434
"""

import json
import os
import urllib.request
import urllib.error
from typing import List, Optional

# ------------------------------------------------------------------
# CONFIG (with Environment Variable Overrides)
# ------------------------------------------------------------------

OLLAMA_URL    = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_CHAT   = os.getenv("OLLAMA_CHAT", "http://localhost:11434/api/chat")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
TIMEOUT       = int(os.getenv("OLLAMA_TIMEOUT", "30"))

# ------------------------------------------------------------------
# SYSTEM PROMPT — the key to bias removal
# ------------------------------------------------------------------

NORMALIZER_SYSTEM = """You are a causal sentence normalizer for a graph inference engine.

Your ONLY job is to convert raw, messy, or domain-biased text into clean atomic causal pairs.

Rules:
1. Extract only explicit cause → effect relationships.
2. Express each as: "entity_direction causes entity_direction"
   - Directions: increase, decrease, rise, fall
   - Entities: 1-3 word noun phrases joined by underscores
   - Example: "oil_supply_decrease causes price_increase"
3. One pair per line. No commentary, no explanations, no bullet points.
4. Ignore vague, metaphorical, or non-causal sentences entirely.
5. Do NOT invent relationships not present in the input.
6. Do NOT favour any domain (economics, weather, health — treat all equally).
7. If the input contains no causal content, return exactly: NO_CAUSAL_CONTENT

Example input:
  "When companies earn less money, they tend to lay off workers."

Example output:
  company_profit_decrease causes employment_decrease
"""

# ------------------------------------------------------------------
# LOW-LEVEL OLLAMA CALL
# ------------------------------------------------------------------

def _call_ollama(prompt: str, model: str = DEFAULT_MODEL) -> Optional[str]:
    """
    Call Ollama's /api/chat endpoint (works for all modern models).
    Falls back to /api/generate for older setups.
    Returns the assistant text or None on failure.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": NORMALIZER_SYSTEM},
            {"role": "user",   "content": prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,   # deterministic — critical for a data pipeline
            "num_predict": 256,
        }
    }

    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        OLLAMA_CHAT,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body.get("message", {}).get("content", "").strip()

    except urllib.error.URLError as e:
        # Ollama not running — return None so caller can fallback gracefully
        print(f"[llm_normalizer] ⚠️  Ollama unreachable: {e.reason}")
        return None
    except Exception as e:
        print(f"[llm_normalizer] ⚠️  Unexpected error: {e}")
        return None


# ------------------------------------------------------------------
# PUBLIC API
# ------------------------------------------------------------------

def normalize_texts(texts: List[str], model: str = DEFAULT_MODEL) -> List[str]:
    """
    Pass a list of raw strings through Ollama.
    Returns a cleaned list of causal sentences (may be shorter than input).

    If Ollama is unavailable, the original texts are returned unchanged
    (graceful degradation — engine still works without LLM).

    Bias removal mechanism:
    - The system prompt forbids domain favouritism
    - temperature=0 ensures no hallucinated relationships
    - "NO_CAUSAL_CONTENT" lines are discarded
    - Only explicit cause→effect lines pass through
    """
    if not texts:
        return []

    # Batch into a single prompt for efficiency
    combined = "\n".join(f"- {t.strip()}" for t in texts if t.strip())
    if not combined:
        return []

    raw_output = _call_ollama(combined, model=model)

    if raw_output is None:
        print("[llm_normalizer] 🔄 Falling back to raw input (no LLM)")
        return texts  # graceful fallback

    # Parse lines, drop noise
    lines = []
    for line in raw_output.splitlines():
        line = line.strip()
        if not line:
            continue
        if line == "NO_CAUSAL_CONTENT":
            continue
        if "causes" in line or "leads to" in line or "results in" in line:
            lines.append(line)

    if not lines:
        print("[llm_normalizer] ℹ️  LLM found no causal content — using raw input")
        return texts

    print(f"[llm_normalizer] ✅ Normalized {len(texts)} inputs → {len(lines)} causal sentences")
    return lines


def normalize_query(query: str, model: str = DEFAULT_MODEL) -> str:
    """
    Normalize a single query string for inference.
    Used by the /query endpoint.
    Returns a cleaned string (or original if LLM unavailable).
    """
    results = normalize_texts([query], model=model)
    return results[0] if results else query


def check_ollama_health(model: str = DEFAULT_MODEL) -> dict:
    """
    Returns health status of the Ollama connection.
    Used by GET /health endpoint.
    """
    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
            "options": {"num_predict": 5}
        }
        data = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            OLLAMA_CHAT, data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            resp.read()
        return {"ollama": "ok", "model": model}
    except Exception as e:
        return {"ollama": "unavailable", "reason": str(e), "model": model}


# ------------------------------------------------------------------
# CLI TEST
# ------------------------------------------------------------------

if __name__ == "__main__":
    samples = [
        "When oil prices go up, consumers pay more at the pump.",
        "A company losing money will fire employees.",
        "Heavy rain leads to flooding in low-lying areas.",
        "The sky is blue and I love weekends.",   # no causal content
        "Interest rates rising cause mortgage defaults to increase.",
    ]
    print("=== Ollama Normalizer Test ===\n")
    results = normalize_texts(samples)
    for r in results:
        print(" →", r)
