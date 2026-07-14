"""
inference.py
============
Beam-search causal inference with:
  - Log-space scoring (prevents underflow on long chains)
  - Depth penalty (keeps results relevant to input)
  - Direction consistency filtering (no illogical causal jumps)
  - Polarity alignment scoring
  - First-hop boost (direct neighbours rank higher)
  - Path tracking for explainability
  - explain_path() for frontend "why" panel
"""

import math
from typing import List, Dict

from .tokenizer import get_polarity, tokenize
from .llm_normalizer import normalize_query

# ------------------------------------------------------------------
# BEAM SEARCH INFERENCE
# ------------------------------------------------------------------

def infer(
    graph: Dict,
    inputs: List[str],
    beam_width: int = 5,
    max_depth: int = 4,
    min_confidence: float = 0.05,
    top_n: int = 10,
) -> List[dict]:
    """
    Beam-search traversal of the directed causal graph.

    Args:
        graph:          Adjacency dict from engine.get_graph()
        inputs:         List of structured tokens (entity_direction)
        beam_width:     Max candidates kept at each step
        max_depth:      Max hops from input nodes
        min_confidence: Filter results below this threshold
        top_n:          Max results returned

    Returns:
        List of dicts: {token, confidence, polarity, path, depth}
    """
    # Track best score and path per node
    best_score: Dict[str, float] = {}
    best_path:  Dict[str, List[str]] = {}

    # Queue: (node, log_score, direction, depth, path)
    queue = []

    for token in inputs:
        pol = get_polarity(token)
        direction = pol if pol != 0 else 1
        score = 0.0
        queue.append((token, score, direction, 0, [token]))
        best_score[token] = score
        best_path[token]  = [token]

    input_set      = set(inputs)
    input_polarity = sum(get_polarity(t) for t in inputs)

    visited_edges = set()  # (node, neighbour) — prevent cycles

    while queue:
        # Beam: keep top beam_width by score
        queue = sorted(queue, key=lambda x: -x[1])[:beam_width]
        current, score, direction, depth, path = queue.pop(0)

        if depth >= max_depth:
            continue

        for (neighbor, strength, edge_polarity) in graph.get(current, []):
            edge_key = (current, neighbor)
            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)

            if strength <= 0:
                continue

            # --- Direction consistency ---
            new_direction   = direction * edge_polarity
            neighbor_pol    = get_polarity(neighbor)
            if neighbor_pol != 0 and new_direction != 0:
                if neighbor_pol != new_direction:
                    continue  # illogical jump — skip

            # --- Log-space score with depth penalty ---
            new_score = score + math.log(strength + 1e-9) - (0.25 * depth)

            # --- First-hop boost ---
            if depth == 0:
                new_score += 0.6

            # --- Global polarity alignment ---
            if input_polarity != 0 and neighbor_pol != 0:
                if input_polarity * neighbor_pol > 0:
                    new_score += 0.2   # consistent direction
                else:
                    new_score -= 0.4   # opposing direction — penalize

            new_path = path + [neighbor]

            if new_score > best_score.get(neighbor, -float("inf")):
                best_score[neighbor] = new_score
                best_path[neighbor]  = new_path
                queue.append((neighbor, new_score, new_direction, depth + 1, new_path))

    # --- Softmax to get probabilities ---
    results = finalize(
        best_score, best_path, input_set, top_n, min_confidence
    )
    return results


def finalize(
    scores:    Dict[str, float],
    paths:     Dict[str, List[str]],
    input_set: set,
    top_n:     int,
    min_conf:  float,
) -> List[dict]:
    """Convert raw log-scores to softmax probabilities and rank."""
    if not scores:
        return []

    max_score = max(scores.values())
    exp_scores = {k: math.exp(v - max_score) for k, v in scores.items()}
    total      = sum(exp_scores.values())

    results = []
    for token, raw_prob in exp_scores.items():
        prob = raw_prob / total
        if prob < min_conf:
            continue
        path  = paths.get(token, [token])
        depth = len(path) - 1
        results.append({
            "token":      token,
            "confidence": round(prob, 4),
            "polarity":   get_polarity(token),
            "path":       path,
            "depth":      depth,
            "is_input":   token in input_set,
        })

    results.sort(key=lambda x: -x["confidence"])
    return results[:top_n]


# ------------------------------------------------------------------
# QUERY PIPELINE
# ------------------------------------------------------------------

def query(
    model_id: str,
    query_text: str,
    graph: Dict,
    use_llm: bool = True,
    llm_model: str = "llama3",
    **kwargs,
) -> dict:
    """
    Full query pipeline:
      1. Normalize query through Ollama (removes domain bias in query language)
      2. Tokenize into structured tokens
      3. Run beam-search inference
      4. Return ranked predictions

    Returns dict with predictions list and metadata.
    """
    # Step 1: LLM normalization of query
    if use_llm:
        normalized_query = normalize_query(query_text, model=llm_model)
        print(f"[inference] Query normalized: '{query_text}' → '{normalized_query}'")
    else:
        normalized_query = query_text

    # Step 2: Tokenize
    tokens = tokenize(normalized_query)

    # Fallback: also try tokenizing original query if normalization gave nothing
    if not tokens:
        tokens = tokenize(query_text)

    if not tokens:
        return {
            "model_id":   model_id,
            "query":      query_text,
            "tokens":     [],
            "predictions": [],
            "warning":    "No structured tokens found. Query must reference entity + direction (e.g. 'price increase', 'supply decrease').",
        }

    # Step 3: Inference
    results = infer(graph, tokens, **kwargs)

    return {
        "model_id":    model_id,
        "query":       query_text,
        "normalized":  normalized_query,
        "tokens":      tokens,
        "predictions": results,
    }


# ------------------------------------------------------------------
# EXPLAINABILITY
# ------------------------------------------------------------------

def explain_path(
    graph: Dict,
    input_token: str,
    output_token: str,
    max_depth: int = 6,
) -> dict:
    """
    Find the shortest causal path from input_token to output_token.
    Uses BFS for shortest-path guarantee.

    Returns:
        {
            "found": bool,
            "path": ["tok1", "tok2", ...],
            "steps": [{"from": ..., "to": ..., "strength": ..., "polarity": ...}],
            "explanation": "plain English sentence"
        }
    """
    if input_token not in graph:
        return {
            "found": False,
            "path":  [],
            "steps": [],
            "explanation": f"'{input_token}' has no outgoing causal edges in this model.",
        }

    # BFS
    from collections import deque
    queue   = deque([(input_token, [input_token])])
    visited = {input_token}

    while queue:
        current, path = queue.popleft()

        if len(path) > max_depth + 1:
            continue

        for (neighbor, strength, polarity) in graph.get(current, []):
            if neighbor in visited:
                continue
            new_path = path + [neighbor]

            if neighbor == output_token:
                steps = []
                for i in range(len(new_path) - 1):
                    fr = new_path[i]
                    to = new_path[i + 1]
                    edge_data = None
                    for (n, s, p) in graph.get(fr, []):
                        if n == to:
                            edge_data = (s, p)
                            break
                    steps.append({
                        "from":     fr,
                        "to":       to,
                        "strength": round(edge_data[0], 4) if edge_data else 0,
                        "polarity": edge_data[1] if edge_data else 0,
                    })

                explanation = _build_explanation(new_path)
                return {
                    "found":       True,
                    "path":        new_path,
                    "steps":       steps,
                    "explanation": explanation,
                }

            visited.add(neighbor)
            queue.append((neighbor, new_path))

    return {
        "found":       False,
        "path":        [],
        "steps":       [],
        "explanation": f"No causal path found from '{input_token}' to '{output_token}' within {max_depth} hops.",
    }


def _build_explanation(path: List[str]) -> str:
    """Convert a token path to a readable English sentence."""
    def humanize(token: str) -> str:
        parts = token.rsplit("_", 1)
        if len(parts) == 2:
            entity, direction = parts
            entity = entity.replace("_", " ")
            if direction == "increase":
                return f"{entity} increases"
            elif direction == "decrease":
                return f"{entity} decreases"
        return token.replace("_", " ")

    if len(path) == 1:
        return humanize(path[0])

    parts = [humanize(path[0])]
    for tok in path[1:]:
        parts.append(f"which causes {humanize(tok)}")

    return ", ".join(parts) + "."
