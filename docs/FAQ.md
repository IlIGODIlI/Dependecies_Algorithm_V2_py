# Frequently Asked Questions

---

### Do I need Ollama to use Vela?

**No.** Ollama is optional. Without it, the engine processes raw input directly — you just lose the LLM normalization step that removes domain bias and cleans up informal language. All core functionality (training, querying, inference, explainability) works without Ollama.

To run without Ollama:

```bash
# CLI
python repl.py --no-llm

# API
curl -X POST http://localhost:8000/train \
  -d '{"data": ["..."], "use_llm": false}'
```

---

### What format should training data be in?

Training sentences should contain **explicit causal language** with direction words:

**Good:**
- `"company profit decrease causes employment decrease"`
- `"rainfall increase leads to flood risk increase"`
- `"interest rate rise results in investment decrease"`

**Won't work:**
- `"unemployment"` (bare noun, no direction)
- `"the sky is blue"` (no causal relationship)
- `"oil and prices"` (co-occurrence, no explicit causality)

---

### Why are some tokens discarded?

The tokenizer only keeps tokens in `entity_direction` format. This is intentional — bare nouns like `"unemployment"` or `"oil"` create vague co-occurrence edges that lead to domain bias. Only tokens like `"unemployment_increase"` or `"oil_supply_decrease"` are kept.

---

### Can I use models other than llama3?

Yes! Any model available via `ollama list` works. Set it via:

- **CLI**: `python repl.py --llm-model mistral`
- **API**: Include `"llm_model": "mistral"` in the request body
- **Environment variable**: `OLLAMA_MODEL=mistral`

---

### Where are model files stored?

By default, in a `models/` directory in the working directory. Each model is a single JSON file. You can change the path via the `VELA_MODELS_DIR` environment variable.

---

### Can I export/import models?

Models are plain JSON files. To export, copy the file from `models/`. To import, place it in the `models/` directory. The filename should match the model ID (e.g. `my_model.json` for model ID `my_model`).

---

### How do I reset a model?

**CLI:**
```
[my_model]>>> reset
```

**API:**
```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"model_id": "my_model"}'
```

This wipes all edges, vocabulary, and training history for that model.

---

### What happens if Ollama goes down during use?

The engine degrades gracefully. If Ollama is unreachable during a training or query call, the raw input is used instead. A warning is printed to the console, but the operation completes successfully.

---

### How is edge strength calculated?

Edge strength is the normalised count per cause node:

```
strength(A → B) = count(A → B) / sum(count(A → all_effects))
```

This means strengths within a cause node sum to approximately 1.0, representing relative likelihood.

---

### Why beam search instead of simple BFS?

Beam search provides:
- **Ranked results** with confidence scores (not just reachability)
- **Depth penalty** to prefer closer, more relevant predictions
- **Direction consistency** filtering to avoid illogical causal jumps
- **Scalability** — beam width limits exploration of very dense graphs

---

### Can I train with data from multiple domains simultaneously?

Yes, and this is encouraged. The engine treats all domains equally. Training with diverse data (economics, weather, healthcare, etc.) produces a more balanced and useful model.
