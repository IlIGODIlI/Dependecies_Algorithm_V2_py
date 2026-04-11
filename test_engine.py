"""
test_engine.py
==============
Full integration test — runs WITHOUT Ollama (use_llm=False).
Validates that the engine is bias-free and correctly handles
diverse domains equally.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import engine
import inference as inf
from tokenizer import tokenize, tokenize_with_polarity, get_polarity
from extractor import extract_causal_pairs, extract_from_texts

PASS = "✅"
FAIL = "❌"
SEP  = "─" * 55

def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print('='*55)

def check(label, condition, extra=""):
    status = PASS if condition else FAIL
    suffix = f"  ({extra})" if extra else ""
    print(f"  {status} {label}{suffix}")
    return condition

# ------------------------------------------------------------------
# 1. TOKENIZER
# ------------------------------------------------------------------
section("1. Tokenizer — structured tokens only")

tests = [
    ("oil supply decrease",   ["oil_supply_decrease"]),
    ("price increase",        ["price_increase"]),
    ("employment levels decrease", ["employment_levels_decrease"]),
    ("heavy rain",            []),   # no direction → empty
    ("unemployment",          []),   # bare noun → empty
    ("interest rate rise",    ["interest_rate_increase"]),
]

all_pass = True
for text, expected in tests:
    result = tokenize(text)
    ok     = result == expected
    check(f"tokenize('{text}')", ok, f"got {result}")
    if not ok:
        all_pass = False

# ------------------------------------------------------------------
# 2. POLARITY
# ------------------------------------------------------------------
section("2. Polarity detection")

pol_tests = [
    ("price_increase",    +1),
    ("cost_decrease",     -1),
    ("employment_levels_decrease", -1),
    ("demand_increase",   +1),
]
for tok, expected in pol_tests:
    pol = get_polarity(tok)
    check(f"polarity('{tok}') == {expected}", pol == expected, f"got {pol}")

# ------------------------------------------------------------------
# 3. CAUSAL EXTRACTOR
# ------------------------------------------------------------------
section("3. Causal pair extraction")

extract_tests = [
    (
        "oil supply decrease causes price increase",
        [("oil_supply_decrease", -1, "price_increase", +1)]
    ),
    (
        "company profit decrease leads to employment levels decrease",
        [("company_profit_decrease", -1, "employment_levels_decrease", -1)]
    ),
    (
        "interest rate rise results in investment decrease",
        [("interest_rate_increase", +1, "investment_decrease", -1)]
    ),
    (
        "the sky is blue",       # no causal marker
        []
    ),
    (
        "unemployment",          # no structure
        []
    ),
]

for text, expected_pairs in extract_tests:
    pairs = extract_causal_pairs(text)
    if expected_pairs:
        ok = len(pairs) > 0 and pairs[0][0] == expected_pairs[0][0]
        check(f"extract('{text[:45]}...')", ok, f"got {pairs}")
    else:
        check(f"extract returns empty for '{text}'", len(pairs) == 0, f"got {pairs}")

# ------------------------------------------------------------------
# 4. ENGINE TRAINING (no LLM)
# ------------------------------------------------------------------
section("4. Engine — train and persist (use_llm=False)")

TEST_MODEL = "_test_bias_free"
engine.reset_model(TEST_MODEL)

# Mix of diverse domains — NOT just oil/finance
training_data = [
    # Economics
    "company profit decrease causes employment levels decrease",
    "interest rate rise leads to investment decrease",
    # Weather / environment
    "rainfall increase causes flood risk increase",
    "temperature increase results in drought risk increase",
    # Healthcare
    "hospital capacity decrease leads to patient wait time increase",
    "vaccination rate increase causes infection rate decrease",
    # Supply chain
    "supply chain disruption increase causes delivery delay increase",
    "raw material cost increase leads to product price increase",
]

result = engine.train(TEST_MODEL, training_data, use_llm=False)
check("Training returns pairs_added > 0", result.get("pairs_added", 0) > 0, str(result))
check("Edge count > 0", result.get("total_edges", 0) > 0, f"edges={result.get('total_edges')}")
check("No error field", "error" not in result)

# ------------------------------------------------------------------
# 5. GRAPH STRUCTURE
# ------------------------------------------------------------------
section("5. Graph — directed only, no bidirectional edges")

graph = engine.get_graph(TEST_MODEL)
check("Graph is not empty", len(graph) > 0)

# Spot check: company_profit_decrease should cause employment_levels_decrease
if "company_profit_decrease" in graph:
    neighbors = [n for n, s, p in graph["company_profit_decrease"]]
    check(
        "company_profit_decrease → employment_levels_decrease",
        "employment_levels_decrease" in neighbors,
        f"neighbors: {neighbors}"
    )
else:
    check("company_profit_decrease in graph", False, "key missing from graph")

# Verify no unexpected reversal edges (bidirectional check)
# If A→B was trained, B should NOT auto-appear as B→A
if "employment_levels_decrease" in graph:
    reverse_neighbors = [n for n, s, p in graph["employment_levels_decrease"]]
    check(
        "No auto-reverse edge employment → company_profit",
        "company_profit_decrease" not in reverse_neighbors,
        f"reverse neighbors: {reverse_neighbors}"
    )

# ------------------------------------------------------------------
# 6. INFERENCE
# ------------------------------------------------------------------
section("6. Inference — query and rank")

results = inf.query(
    model_id   = TEST_MODEL,
    query_text = "company profit decrease",
    graph      = graph,
    use_llm    = False,
)
check("Query returns predictions", len(results["predictions"]) > 0)
check("Tokens extracted", len(results["tokens"]) > 0, str(results["tokens"]))

# Top prediction should involve employment
if results["predictions"]:
    top = results["predictions"][0]
    check(
        "Top prediction has confidence > 0",
        top["confidence"] > 0,
        f"top={top['token']} conf={top['confidence']}"
    )
    check("Prediction has path", len(top["path"]) > 0)
    check("Prediction has polarity", "polarity" in top)

# ------------------------------------------------------------------
# 7. EXPLAINABILITY
# ------------------------------------------------------------------
section("7. Explain path")

exp = inf.explain_path(graph, "company_profit_decrease", "employment_levels_decrease")
check("Path found", exp["found"], str(exp))
if exp["found"]:
    check("Path contains both tokens",
        "company_profit_decrease" in exp["path"] and "employment_levels_decrease" in exp["path"])
    check("Explanation is a string", isinstance(exp["explanation"], str) and len(exp["explanation"]) > 5)
    print(f"    Explanation: {exp['explanation']}")

exp_fail = inf.explain_path(graph, "nonexistent_token", "other_token")
check("Returns found=False for unknown token", not exp_fail["found"])

# ------------------------------------------------------------------
# 8. DOMAIN BIAS CHECK
# ------------------------------------------------------------------
section("8. Domain balance — bias detection")

stats = engine.get_stats(TEST_MODEL)
top_tokens = [tok for tok, freq in stats["top_causes"][:5]]
print(f"  Top 5 tokens: {top_tokens}")

# Oil/finance should NOT dominate if we didn't train on it
oil_dominated = any("oil" in t or "crude" in t for t in top_tokens)
check("Oil/finance tokens do NOT dominate top-5", not oil_dominated,
    "Model was trained without oil data; top tokens should reflect training")

# Should have diverse domains represented
all_tokens = [t for t, _ in stats["top_causes"]]
domains_found = {
    "economics":   any("profit" in t or "employment" in t or "interest" in t
                       for t in all_tokens),
    "weather":     any("flood" in t or "rainfall" in t or "drought" in t
                       for t in all_tokens),
    "healthcare":  any("hospital" in t or "vaccination" in t or "infection" in t
                       for t in all_tokens),
}
for domain, found in domains_found.items():
    check(f"Domain '{domain}' represented in vocab", found)

# ------------------------------------------------------------------
# 9. MULTI-MODEL ISOLATION
# ------------------------------------------------------------------
section("9. Multi-model isolation")

MODEL_A = "_test_model_a"
MODEL_B = "_test_model_b"

engine.reset_model(MODEL_A)
engine.reset_model(MODEL_B)

engine.train(MODEL_A, ["rainfall increase causes flood risk increase"], use_llm=False)
engine.train(MODEL_B, ["vaccination rate increase causes infection rate decrease"], use_llm=False)

graph_a = engine.get_graph(MODEL_A)
graph_b = engine.get_graph(MODEL_B)

check("Model A has rainfall edge", "rainfall_increase" in graph_a)
check("Model A does NOT have vaccination edge", "vaccination_rate_increase" not in graph_a)
check("Model B has vaccination edge", "vaccination_rate_increase" in graph_b)
check("Model B does NOT have rainfall edge", "rainfall_increase" not in graph_b)

# Cleanup test models
engine.delete_model(TEST_MODEL)
engine.delete_model(MODEL_A)
engine.delete_model(MODEL_B)

# ------------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------------
section("✅ Test complete")
print("  Run with Ollama enabled: python repl.py")
print("  Start API server: uvicorn api:app --reload --port 8000")
print()
