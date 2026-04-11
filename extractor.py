"""
extractor.py
============
Extracts EXPLICIT causal pairs from text.
Uses only syntactic causal markers — never co-occurrence.
This is the core of the de-biasing: relationships must be stated,
not inferred from word proximity.
"""

import re
from typing import List, Tuple, Optional
from tokenizer import tokenize_with_polarity, get_polarity

# ------------------------------------------------------------------
# CAUSAL CONNECTORS (order matters — check longer ones first)
# ------------------------------------------------------------------

CAUSAL_PATTERNS = [
    # 3-word patterns first (most specific)
    r"(?P<cause>.+?)\s+results?\s+in\s+(?P<effect>.+)",
    r"(?P<cause>.+?)\s+leads?\s+to\s+(?P<effect>.+)",
    r"(?P<cause>.+?)\s+gives?\s+rise\s+to\s+(?P<effect>.+)",
    r"(?P<cause>.+?)\s+brings?\s+about\s+(?P<effect>.+)",
    r"(?P<cause>.+?)\s+is\s+followed\s+by\s+(?P<effect>.+)",
    # 2-word / simple patterns
    r"(?P<cause>.+?)\s+causes?\s+(?P<effect>.+)",
    r"(?P<cause>.+?)\s+triggers?\s+(?P<effect>.+)",
    r"(?P<cause>.+?)\s+produces?\s+(?P<effect>.+)",
    r"(?P<cause>.+?)\s+creates?\s+(?P<effect>.+)",
    r"(?P<cause>.+?)\s+drives?\s+(?P<effect>.+)",
    # Arrow notation
    r"(?P<cause>.+?)\s*->\s*(?P<effect>.+)",
    r"(?P<cause>.+?)\s*→\s*(?P<effect>.+)",
]

# Pre-compile for speed
_COMPILED = [re.compile(p, re.IGNORECASE) for p in CAUSAL_PATTERNS]


# ------------------------------------------------------------------
# PAIR EXTRACTION
# ------------------------------------------------------------------

def extract_causal_pairs(text: str) -> List[Tuple[str, int, str, int]]:
    """
    Returns a list of (cause_token, cause_polarity, effect_token, effect_polarity).

    A single sentence can produce multiple pairs if multiple structured
    tokens exist on either side of the causal marker.

    ONLY tokens matching entity_direction format are included.
    Bare nouns are silently discarded.
    """
    text = text.strip()
    pairs = []

    for pattern in _COMPILED:
        match = pattern.search(text)
        if not match:
            continue

        cause_text  = match.group("cause").strip()
        effect_text = match.group("effect").strip()

        cause_tokens  = tokenize_with_polarity(cause_text)
        effect_tokens = tokenize_with_polarity(effect_text)

        if not cause_tokens or not effect_tokens:
            break  # pattern matched but no structured tokens → try next

        # Cross-product: each cause token → each effect token
        for (c_tok, c_pol) in cause_tokens:
            for (e_tok, e_pol) in effect_tokens:
                if c_tok != e_tok:
                    pairs.append((c_tok, c_pol, e_tok, e_pol))

        break  # use first matching pattern only

    return pairs


def extract_from_texts(texts: List[str]) -> List[Tuple[str, int, str, int]]:
    """Extract causal pairs from a list of sentences."""
    all_pairs = []
    for text in texts:
        # Handle multi-sentence inputs (split on period or newline)
        sentences = re.split(r"[.\n]+", text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 5:
                pairs = extract_causal_pairs(sentence)
                all_pairs.extend(pairs)
    return all_pairs


# ------------------------------------------------------------------
# CLI TEST
# ------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        "oil supply decrease causes price increase",
        "company profit decrease leads to employment levels decrease",
        "interest rate rise results in investment decrease",
        "hospital load increase triggers staff stress increase",
        "crop yield decrease → food price increase",
        "the sky is blue",                           # no causal marker
        "unemployment",                              # no structure at all
    ]
    print("=== Extractor Test ===\n")
    for t in tests:
        pairs = extract_causal_pairs(t)
        print(f"  Input: {t}")
        if pairs:
            for c, cp, e, ep in pairs:
                print(f"    {c}({'+' if cp > 0 else '-'}) → {e}({'+' if ep > 0 else '-'})")
        else:
            print(f"    (no causal pairs extracted)")
        print()
