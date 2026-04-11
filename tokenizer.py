"""
tokenizer.py
============
Produces ONLY structured tokens of the form: entity_direction
Tokens without a direction are discarded — this prevents the model
from building co-occurrence edges on bare nouns (which caused the
oil/price domain bias in the original engine).
"""

import re
from typing import List, Optional, Tuple

# ------------------------------------------------------------------
# DIRECTION VOCABULARY
# ------------------------------------------------------------------

DIRECTION_WORDS = {
    "increase", "increases", "increased", "increasing",
    "decrease", "decreases", "decreased", "decreasing",
    "rise",     "rises",     "rose",      "rising",
    "fall",     "falls",     "fell",      "falling",
    "grow",     "grows",     "grew",      "growing",
    "drop",     "drops",     "dropped",   "dropping",
    "surge",    "surges",    "surged",    "surging",
    "collapse", "collapses", "collapsed", "collapsing",
    "expand",   "expands",   "expanded",  "expanding",
    "shrink",   "shrinks",   "shrank",    "shrinking",
}

# Canonical direction → polarity
CANONICAL = {
    "increase": ("increase", +1),
    "increases": ("increase", +1),
    "increased": ("increase", +1),
    "increasing": ("increase", +1),
    "rise": ("increase", +1),
    "rises": ("increase", +1),
    "rose": ("increase", +1),
    "rising": ("increase", +1),
    "grow": ("increase", +1),
    "grows": ("increase", +1),
    "grew": ("increase", +1),
    "growing": ("increase", +1),
    "surge": ("increase", +1),
    "surges": ("increase", +1),
    "surged": ("increase", +1),
    "surging": ("increase", +1),
    "expand": ("increase", +1),
    "expands": ("increase", +1),
    "expanded": ("increase", +1),
    "expanding": ("increase", +1),

    "decrease": ("decrease", -1),
    "decreases": ("decrease", -1),
    "decreased": ("decrease", -1),
    "decreasing": ("decrease", -1),
    "fall": ("decrease", -1),
    "falls": ("decrease", -1),
    "fell": ("decrease", -1),
    "falling": ("decrease", -1),
    "drop": ("decrease", -1),
    "drops": ("decrease", -1),
    "dropped": ("decrease", -1),
    "dropping": ("decrease", -1),
    "collapse": ("decrease", -1),
    "collapses": ("decrease", -1),
    "collapsed": ("decrease", -1),
    "collapsing": ("decrease", -1),
    "shrink": ("decrease", -1),
    "shrinks": ("decrease", -1),
    "shrank": ("decrease", -1),
    "shrinking": ("decrease", -1),
}

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "to", "of", "in", "on", "at", "by", "for", "with",
    "and", "or", "but", "if", "when", "then", "that", "this",
    "it", "its", "will", "can", "may", "might", "should", "would",
    "have", "has", "had", "do", "does", "did", "which", "from",
    "as", "so", "also", "often", "usually", "typically", "generally",
    "cause", "causes", "caused", "causing",
    "lead", "leads", "led", "leading",
    "result", "results", "resulted", "resulting",
    "due",
}

# Max entity words before the direction word (e.g. "oil supply" = 2 words)
MAX_ENTITY_WORDS = 4


# ------------------------------------------------------------------
# CORE EXTRACTOR
# ------------------------------------------------------------------

def extract_structured_tokens(text: str) -> List[Tuple[str, int]]:
    """
    Scan text for patterns: [entity words] [direction word]
    Returns a list of (token_string, polarity) tuples.

    Example:
        "oil supply decrease" → [("oil_supply_decrease", -1)]
        "price increase"      → [("price_increase", +1)]
        "unemployment"        → []   ← no direction → discarded
    """
    # Clean text: lower, strip symbols except underscores and hyphens
    text = re.sub(r"[^a-zA-Z\s_\-]", " ", text.lower())
    text = re.sub(r"[-_]+", " ", text)
    words = text.split()

    tokens = []
    i = 0
    while i < len(words):
        word = words[i]
        if word in CANONICAL:
            dir_label, polarity = CANONICAL[word]
            # Collect entity words immediately preceding this direction word
            entity_words = []
            j = i - 1
            while j >= 0 and (i - j) <= MAX_ENTITY_WORDS:
                candidate = words[j]
                if candidate in STOPWORDS or candidate in DIRECTION_WORDS:
                    break
                entity_words.insert(0, candidate)
                j -= 1

            if entity_words:
                entity = "_".join(entity_words)
                token  = f"{entity}_{dir_label}"
                tokens.append((token, polarity))

        i += 1

    return tokens


def tokenize(text: str) -> List[str]:
    """
    Public interface: returns list of structured token strings only.
    No bare nouns, no unstructured words — ONLY entity_direction format.
    """
    return [tok for tok, _ in extract_structured_tokens(text)]


def tokenize_with_polarity(text: str) -> List[Tuple[str, int]]:
    """Returns (token, polarity) pairs."""
    return extract_structured_tokens(text)


def get_polarity(token: str) -> int:
    """Infer polarity from a token's direction suffix."""
    if token.endswith("_increase"):
        return +1
    if token.endswith("_decrease"):
        return -1
    # Check for other direction words
    for part in token.split("_"):
        if part in CANONICAL:
            return CANONICAL[part][1]
    return 0


# ------------------------------------------------------------------
# CLI TEST
# ------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        "oil supply decrease causes price increase",
        "company profit decrease leads to employment levels decrease",
        "heavy rain causes flooding",                          # no direction → empty
        "hospital load increase causes staff stress increase",
        "interest rate rise leads to investment decrease",
        "the economy",                                         # bare noun → empty
        "crop damage leads to food shortage",                  # no direction words
    ]
    print("=== Tokenizer Test ===\n")
    for t in tests:
        result = tokenize(t)
        print(f"  Input : {t}")
        print(f"  Tokens: {result}")
        print()
