"""
repl.py
=======
Interactive CLI for the Vela causal engine.
All input passes through Ollama normalization (unless --no-llm flag used).

Usage:
    python repl.py                    # default model, with LLM
    python repl.py --model flood      # use 'flood' model
    python repl.py --no-llm           # skip Ollama normalization
    python repl.py --llm-model mistral
"""

import sys
from . import engine
from . import inference as inf
from .llm_normalizer import check_ollama_health

# ------------------------------------------------------------------
# DISPLAY HELPERS
# ------------------------------------------------------------------

def print_banner():
    print("""
╔══════════════════════════════════════════════════════╗
║       VELA  —  Causal Reasoning Engine  v2.0         ║
║       Ollama-normalized  |  Bias-free  |  Directed    ║
╚══════════════════════════════════════════════════════╝
""")


def print_help():
    print("""
Commands:
  train: <sentence>           Learn from a single sentence
  file: <path>                Learn from a text file
  query: <text>               Run causal inference
  explain: <tok1> -> <tok2>   Show causal path between two tokens
  graph                       Show all edges in current model
  stats                       Show model statistics
  models                      List all saved models
  switch: <model_id>          Switch to a different model
  reset                       Wipe current model's memory
  llm on / llm off            Toggle Ollama normalization
  health                      Check Ollama connection
  help                        Show this message
  exit
""")


def print_predictions(results: list, query: str):
    print(f"\n🔮 Predictions for: '{query}'\n")
    if not results["predictions"]:
        print("  ⚠️  No predictions. Try training more causal sentences.")
        if results.get("warning"):
            print(f"  ℹ️  {results['warning']}")
        return

    for i, r in enumerate(results["predictions"], 1):
        arrow   = "↑" if r["polarity"] > 0 else "↓" if r["polarity"] < 0 else "→"
        label   = r["token"].replace("_", " ")
        path_str = " → ".join(r["path"])
        print(f"  {i}. {arrow} {label:<35} confidence: {r['confidence']:.3f}  hops: {r['depth']}")
        print(f"     path: {path_str}")

    if results.get("normalized") and results["normalized"] != query:
        print(f"\n  [LLM normalized query to: '{results['normalized']}']")


def print_graph(graph: dict):
    if not graph:
        print("  (graph is empty — train the model first)")
        return
    print(f"\n{'─'*60}")
    for cause, neighbors in sorted(graph.items()):
        print(f"  {cause}")
        for effect, strength, polarity in neighbors:
            pol_sym = "+" if polarity > 0 else "-"
            print(f"    └─[{pol_sym}]→ {effect}  (strength: {strength:.3f})")
    print(f"{'─'*60}")


# ------------------------------------------------------------------
# REPL
# ------------------------------------------------------------------

def run_repl(model_id: str = "default", use_llm: bool = True, llm_model: str = "llama3"):
    print_banner()

    # Health check
    status = check_ollama_health(llm_model)
    if status["ollama"] == "ok":
        print(f"✅ Ollama connected  (model: {llm_model})")
    else:
        print(f"⚠️  Ollama not available — LLM normalization will be SKIPPED")
        print(f"   Reason: {status.get('reason', 'unknown')}")
        print(f"   Start Ollama with: ollama serve")
        use_llm = False

    print(f"📦 Active model: {model_id}")
    stats = engine.get_stats(model_id)
    print(f"   {stats['doc_count']} docs | {stats['edge_count']} edges | {stats['vocab_size']} tokens")
    print_help()

    while True:
        try:
            user_input = input(f"\n[{model_id}]>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        # EXIT
        if cmd == "exit":
            break

        # HELP
        if cmd == "help":
            print_help()
            continue

        # HEALTH
        if cmd == "health":
            status = check_ollama_health(llm_model)
            print(f"\n  Ollama: {status['ollama']}")
            if "reason" in status:
                print(f"  Reason: {status['reason']}")
            continue

        # LLM TOGGLE
        if cmd == "llm on":
            use_llm = True
            print("  ✅ LLM normalization ON")
            continue
        if cmd == "llm off":
            use_llm = False
            print("  ⏭️  LLM normalization OFF")
            continue

        # MODELS LIST
        if cmd == "models":
            models = engine.list_models()
            if not models:
                print("  (no saved models)")
            else:
                for m in models:
                    active = " ◀ active" if m["model_id"] == model_id else ""
                    print(f"  {m['model_id']:<20} docs:{m['doc_count']}  edges:{m['edge_count']}{active}")
            continue

        # SWITCH MODEL
        if user_input.startswith("switch:"):
            model_id = user_input.replace("switch:", "").strip()
            stats = engine.get_stats(model_id)
            print(f"  Switched to model '{model_id}'")
            print(f"  {stats['doc_count']} docs | {stats['edge_count']} edges | {stats['vocab_size']} tokens")
            continue

        # STATS
        if cmd == "stats":
            stats = engine.get_stats(model_id)
            print(f"\n📊 Model: {model_id}")
            print(f"   Docs trained : {stats['doc_count']}")
            print(f"   Vocab size   : {stats['vocab_size']}")
            print(f"   Edge count   : {stats['edge_count']}")
            if stats["top_causes"]:
                print(f"   Top tokens   :")
                for tok, freq in stats["top_causes"][:8]:
                    print(f"     {tok:<35} freq={freq}")
            continue

        # GRAPH
        if cmd == "graph":
            graph = engine.get_graph(model_id)
            print_graph(graph)
            continue

        # TRAIN FROM TEXT
        if user_input.startswith("train:"):
            text = user_input.replace("train:", "").strip()
            result = engine.train(
                model_id=model_id, texts=[text],
                use_llm=use_llm, llm_model=llm_model
            )
            if "error" in result:
                print(f"  ❌ {result['error']}")
            elif result.get("warning"):
                print(f"  ⚠️  {result['warning']}")
            else:
                print(f"  ✅ +{result['pairs_added']} pairs | total edges: {result['total_edges']}")
            continue

        # TRAIN FROM FILE
        if user_input.startswith("file:"):
            path = user_input.replace("file:", "").strip()
            result = engine.train_from_file(
                model_id=model_id, path=path,
                use_llm=use_llm, llm_model=llm_model
            )
            if "error" in result:
                print(f"  ❌ {result['error']}")
            else:
                print(f"  ✅ +{result.get('pairs_added', 0)} pairs | total edges: {result.get('total_edges', 0)}")
            continue

        # EXPLAIN PATH
        if user_input.startswith("explain:"):
            parts = user_input.replace("explain:", "").strip().split("->")
            if len(parts) != 2:
                print("  Usage: explain: token_one -> token_two")
                continue
            tok_in  = parts[0].strip().replace(" ", "_")
            tok_out = parts[1].strip().replace(" ", "_")
            graph   = engine.get_graph(model_id)
            result  = inf.explain_path(graph, tok_in, tok_out)
            print(f"\n{'─'*50}")
            if result["found"]:
                print(f"  Path: {' → '.join(result['path'])}")
                print(f"\n  {result['explanation']}")
                for step in result["steps"]:
                    pol = "+" if step["polarity"] > 0 else "-"
                    print(f"    [{pol}] {step['from']} → {step['to']}  strength={step['strength']}")
            else:
                print(f"  ❌ {result['explanation']}")
            print(f"{'─'*50}")
            continue

        # RESET
        if cmd == "reset":
            confirm = input(f"  ⚠️  Wipe model '{model_id}'? Type 'yes' to confirm: ").strip()
            if confirm.lower() == "yes":
                engine.reset_model(model_id)
                print("  🗑️  Model cleared.")
            else:
                print("  Cancelled.")
            continue

        # QUERY (default — any other input)
        query_text = user_input
        if user_input.startswith("query:"):
            query_text = user_input.replace("query:", "").strip()

        graph   = engine.get_graph(model_id)
        results = inf.query(
            model_id   = model_id,
            query_text = query_text,
            graph      = graph,
            use_llm    = use_llm,
            llm_model  = llm_model,
        )
        print_predictions(results, query_text)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Vela Causal REPL")
    parser.add_argument("--model",     default="default", help="Model ID to use")
    parser.add_argument("--no-llm",   action="store_true", help="Disable Ollama normalization")
    parser.add_argument("--llm-model", default="llama3", help="Ollama model name")
    args = parser.parse_args()

    run_repl(
        model_id  = args.model,
        use_llm   = not args.no_llm,
        llm_model = args.llm_model,
    )


if __name__ == "__main__":
    main()
