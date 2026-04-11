# Vela Causal Reasoning Engine v2.0
**Ollama-normalized · Bias-free · Directed Graph · Multi-model**

## File Structure
```
vela_causal/
├── llm_normalizer.py   ← Ollama integration (bias removal layer)
├── tokenizer.py        ← Strict entity_direction tokens only
├── extractor.py        ← True cause→effect extraction (no co-occurrence)
├── engine.py           ← Directed graph + persistence + multi-model
├── inference.py        ← Beam search + explainability
├── api.py              ← FastAPI backend
├── repl.py             ← Interactive CLI
└── test_engine.py      ← Full test suite (28 passing tests)
```

## Setup

```bash
pip install fastapi uvicorn

# Install and start Ollama (for bias removal)
# https://ollama.ai
ollama pull llama3
ollama serve
```

## Run

```bash
# REPL (interactive CLI)
python repl.py

# With a specific model
python repl.py --model flood_model

# Without LLM normalization (faster, no Ollama needed)
python repl.py --no-llm

# API server
uvicorn api:app --reload --port 8000

# Tests
python test_engine.py
```

## How Bias Removal Works

The original engine had bias toward oil/price data because:
1. It used **co-occurrence** — any words near each other became "related"
2. It had **bootstrap training data** seeded with oil/finance sentences
3. No filtering of domain-heavy language

This engine fixes all three:

### 1. Ollama Normalizer (`llm_normalizer.py`)
Every piece of text — training data AND queries — passes through an LLM with `temperature=0` and a strict system prompt that:
- Rewrites raw input into clean causal pairs
- **Explicitly forbids domain favouritism**
- Discards non-causal content
- Normalizes informal language (e.g. "oil gets expensive" → `oil_price_increase`)

### 2. No Co-occurrence
`extractor.py` uses only **explicit causal markers**: "causes", "leads to", "results in", etc.
No `combinations()`. Relationships must be stated, not implied by proximity.

### 3. Structured Tokens Only
`tokenizer.py` rejects bare nouns. Every token MUST be `entity_direction` format.
`"unemployment"` → discarded. `"unemployment_increase"` → kept.
This prevents the model from building vague associations between nouns.

### 4. Empty Start
No bootstrap, no pre-trained data. Every model starts completely empty.

## API Usage

```bash
# Train
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"model_id": "my_model", "data": ["company profit decrease causes employment decrease"]}'

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"model_id": "my_model", "query": "profit decrease"}'

# Explain path
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"model_id": "my_model", "input_token": "company_profit_decrease", "output_token": "employment_decrease"}'

# Skip LLM (faster)
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"model_id": "my_model", "data": [...], "use_llm": false}'
```

## Ollama Models

The default model is `llama3`. Change with `llm_model` parameter:
```json
{"model_id": "x", "data": [...], "llm_model": "mistral"}
```

Any model available via `ollama list` works. Recommended:
- `llama3` — best quality
- `phi3` — lightweight, fast
- `mistral` — good balance

## Graceful Degradation

If Ollama is not running, the engine falls back to processing raw input directly.
All core functionality works without Ollama — you just lose the normalization/bias-removal step.
