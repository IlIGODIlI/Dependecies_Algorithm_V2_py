"""
engine.py
=========
Directed causal graph engine.
  - Stores ONLY cause → effect edges (no bidirectional pairs)
  - Per-model isolation (each model_id has its own graph)
  - Persistent storage via JSON (human-readable, debuggable)
  - No pre-trained / bootstrap data — models start completely empty
  - Passes ALL input through llm_normalizer before processing
    (this is the core of the de-biasing system)
"""

import os
import json
import math
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

from tokenizer import get_polarity
from extractor import extract_from_texts
from llm_normalizer import normalize_texts, normalize_query

# ------------------------------------------------------------------
# PERSISTENCE
# ------------------------------------------------------------------

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def _model_path(model_id: str) -> str:
    safe_id = "".join(c for c in model_id if c.isalnum() or c in "_-")
    return os.path.join(MODELS_DIR, f"{safe_id}.json")


def _empty_model() -> dict:
    return {
        "edges":      {},   # cause → {effect: {"strength": float, "polarity": int, "count": int}}
        "word_freq":  {},   # token → occurrence count
        "doc_count":  0,
        "model_id":   "",
        "description": "",
    }


def save_model(state: dict):
    path = _model_path(state["model_id"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def load_model(model_id: str) -> dict:
    path = _model_path(model_id)
    if not os.path.exists(path):
        state = _empty_model()
        state["model_id"] = model_id
        return state
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_models() -> List[dict]:
    models = []
    for fname in os.listdir(MODELS_DIR):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(MODELS_DIR, fname)) as f:
                    data = json.load(f)
                    models.append({
                        "model_id":    data.get("model_id", fname[:-5]),
                        "description": data.get("description", ""),
                        "doc_count":   data.get("doc_count", 0),
                        "edge_count":  sum(len(v) for v in data.get("edges", {}).values()),
                        "vocab_size":  len(data.get("word_freq", {})),
                    })
            except Exception:
                pass
    return models


def delete_model(model_id: str) -> bool:
    path = _model_path(model_id)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


# ------------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------------

# Minimum strength threshold — edges below this are pruned on query
MIN_STRENGTH = 0.05

def train(
    model_id: str,
    texts: List[str],
    use_llm: bool = True,
    llm_model: str = "llama3",
    description: str = "",
) -> dict:
    """
    Train a model on a list of text strings.

    Pipeline:
      1. [Optional] Normalize through Ollama LLM — removes domain bias,
         rewrites into clean causal sentences, discards non-causal content.
      2. Extract explicit cause→effect pairs.
      3. Update directed edge graph with strength and polarity.
      4. Persist to disk.

    Returns summary statistics.

    Args:
        model_id:    Identifier for this model (e.g. "flood_model")
        texts:       Raw input sentences
        use_llm:     Whether to pass through Ollama normalizer (default: True)
                     Set to False if Ollama is unavailable or for testing.
        llm_model:   Ollama model name (default: "llama3")
        description: Optional model description
    """
    state = load_model(model_id)
    if description:
        state["description"] = description

    # --- Step 1: LLM normalization (bias removal) ---
    if use_llm:
        print(f"[engine] 🤖 Sending {len(texts)} texts through Ollama ({llm_model}) for normalization …")
        normalized = normalize_texts(texts, model=llm_model)
    else:
        print(f"[engine] ⏭️  Skipping LLM normalization (use_llm=False)")
        normalized = texts

    if not normalized:
        return {"error": "No processable text after normalization", "pairs_added": 0}

    # --- Step 2: Extract causal pairs ---
    pairs = extract_from_texts(normalized)

    if not pairs:
        return {
            "model_id":        model_id,
            "pairs_added":     0,
            "total_edges":     sum(len(v) for v in state["edges"].values()),
            "warning":         "No causal pairs could be extracted. Ensure sentences use explicit causal language: 'X causes Y', 'X leads to Y', etc.",
        }

    # --- Step 3: Update graph ---
    edges     = state["edges"]
    word_freq = state["word_freq"]
    added     = 0

    for (cause, c_pol, effect, e_pol) in pairs:
        # Word frequency
        word_freq[cause]  = word_freq.get(cause, 0) + 1
        word_freq[effect] = word_freq.get(effect, 0) + 1

        # Directed edge: cause → effect only
        if cause not in edges:
            edges[cause] = {}

        if effect not in edges[cause]:
            edges[cause][effect] = {"strength": 0.0, "polarity": 0, "count": 0}

        entry = edges[cause][effect]
        entry["count"]    += 1
        # Polarity: same-direction = reinforcing (+1), opposite = conflicting (-1)
        edge_polarity      = c_pol * e_pol if (c_pol != 0 and e_pol != 0) else 1
        entry["polarity"]  = edge_polarity
        # Strength: normalized count (will be re-normalized on read)
        entry["strength"] += 1.0
        added += 1

    state["doc_count"] += len(normalized)

    # --- Step 4: Normalize strengths (probability-like, per cause node) ---
    for cause, effects in edges.items():
        total = sum(e["count"] for e in effects.values())
        for effect in effects:
            edges[cause][effect]["strength"] = edges[cause][effect]["count"] / max(total, 1)

    save_model(state)

    return {
        "model_id":    model_id,
        "pairs_added": added,
        "total_edges": sum(len(v) for v in edges.values()),
        "doc_count":   state["doc_count"],
        "vocab_size":  len(word_freq),
    }


def train_from_file(
    model_id: str,
    path: str,
    use_llm: bool = True,
    llm_model: str = "llama3",
) -> dict:
    """Load lines from a text file and train."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        return train(model_id, lines, use_llm=use_llm, llm_model=llm_model)
    except Exception as e:
        return {"error": str(e)}


# ------------------------------------------------------------------
# GRAPH ACCESS
# ------------------------------------------------------------------

def get_graph(model_id: str, min_strength: float = MIN_STRENGTH) -> Dict:
    """
    Return pruned adjacency dict: cause → [(effect, strength, polarity), ...]
    Sorted by strength descending. Weak edges filtered out.
    """
    state = load_model(model_id)
    graph = {}
    for cause, effects in state["edges"].items():
        neighbors = [
            (effect, data["strength"], data["polarity"])
            for effect, data in effects.items()
            if data["strength"] >= min_strength
        ]
        if neighbors:
            graph[cause] = sorted(neighbors, key=lambda x: -x[1])
    return graph


def get_stats(model_id: str) -> dict:
    state = load_model(model_id)
    return {
        "model_id":   model_id,
        "doc_count":  state["doc_count"],
        "vocab_size": len(state["word_freq"]),
        "edge_count": sum(len(v) for v in state["edges"].values()),
        "top_causes": sorted(
            state["word_freq"].items(), key=lambda x: -x[1]
        )[:10],
    }


def reset_model(model_id: str):
    """Wipe all data for a model."""
    path = _model_path(model_id)
    if os.path.exists(path):
        os.remove(path)
    state = _empty_model()
    state["model_id"] = model_id
    save_model(state)
    return {"reset": True, "model_id": model_id}
