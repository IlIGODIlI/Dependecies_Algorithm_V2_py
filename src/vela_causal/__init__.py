"""
Vela Causal Reasoning Engine v2.0
==================================
Ollama-normalized · Bias-free · Directed Graph · Multi-model
"""

from .llm_normalizer import normalize_texts, normalize_query, check_ollama_health
from .tokenizer import tokenize, tokenize_with_polarity, get_polarity
from .extractor import extract_causal_pairs, extract_from_texts
from .engine import train, train_from_file, get_graph, get_stats, reset_model, delete_model, list_models
from .inference import infer, query, explain_path

__version__ = "2.0.0"
__all__ = [
    "normalize_texts",
    "normalize_query",
    "check_ollama_health",
    "tokenize",
    "tokenize_with_polarity",
    "get_polarity",
    "extract_causal_pairs",
    "extract_from_texts",
    "train",
    "train_from_file",
    "get_graph",
    "get_stats",
    "reset_model",
    "delete_model",
    "list_models",
    "infer",
    "query",
    "explain_path",
]
