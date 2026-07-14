<div align="center">

# Vela — Causal Reasoning Engine v2.0

**Bias-free · Ollama-normalized · Directed Graph · Multi-model**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=llama&logoColor=white)](https://ollama.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

A production-grade causal inference engine that builds directed graphs from natural language, with an integrated Ollama LLM layer for automatic bias removal and entity normalization.

[Quick Start](#-quick-start) · [API Docs](docs/API.md) · [Architecture](docs/ARCHITECTURE.md) · [FAQ](docs/FAQ.md)

</div>

---

## Overview

Vela learns explicit **cause → effect** relationships from text, stores them in a directed graph, and answers causal queries using beam-search inference with explainable paths. Unlike co-occurrence-based engines, Vela only recognises relationships that are explicitly stated — eliminating domain bias from training data.

Every piece of text — both training data and queries — can optionally pass through a local **Ollama LLM** that normalises phrasing, strips metaphorical language, and enforces structured `entity_direction` token formats.

### Key Features

| Feature | Description |
|---|---|
| **Bias-Free Design** | No co-occurrence, no bootstrap data, no domain favouritism |
| **Ollama Integration** | LLM normalisation layer rewrites messy input into clean causal pairs |
| **Directed Graphs** | Cause → effect edges only — no bidirectional noise |
| **Beam Search Inference** | Log-space scoring, depth penalty, polarity alignment |
| **Explainability** | Every prediction includes a full causal path with "why" explanations |
| **Multi-Model** | Isolated per-model graphs with independent training and querying |
| **Graceful Degradation** | Works without Ollama — LLM normalisation is optional |
| **FastAPI + CLI** | REST API with interactive Swagger docs, plus a rich REPL |

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                        User Input (text)                          │
└───────────────────────────┬────────────────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │   Ollama LLM Normalizer     │  ← bias removal layer
              │   (llm_normalizer.py)       │     temperature=0
              └─────────────┬───────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │   Causal Pair Extractor     │  ← explicit markers only
              │   (extractor.py)            │     "causes", "leads to"…
              └─────────────┬───────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │   Structured Tokenizer      │  ← entity_direction only
              │   (tokenizer.py)            │     bare nouns discarded
              └─────────────┬───────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │   Directed Graph Engine     │  ← cause → effect edges
              │   (engine.py)               │     JSON persistence
              └─────────────┬───────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │   Beam Search Inference     │  ← ranked predictions
              │   (inference.py)            │     with explainability
              └─────────────────────────────┘
```

> For a deep dive, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Quick Start

### Prerequisites

- **Python 3.9+**
- **Ollama** (optional, for LLM normalization): [ollama.ai](https://ollama.ai)

### 1. Clone & Install

```bash
git clone https://github.com/aryan/vela-causal.git
cd vela-causal

# Install dependencies
pip install -r requirements.txt

# (Optional) Pull an Ollama model for bias removal
ollama pull llama3
ollama serve
```

### 2. Interactive CLI (REPL)

```bash
python repl.py

# With a specific model
python repl.py --model flood_model

# Without LLM normalization (faster, no Ollama needed)
python repl.py --no-llm
```

### 3. REST API

```bash
uvicorn api:app --reload --port 8000
```

Then open [http://localhost:8000/docs](http://localhost:8000/docs) for interactive Swagger documentation.

---

## Usage Examples

### Train a Model

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "my_model",
    "data": ["company profit decrease causes employment decrease"]
  }'
```

### Query for Predictions

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"model_id": "my_model", "query": "profit decrease"}'
```

### Explain a Causal Path

```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "my_model",
    "input_token": "company_profit_decrease",
    "output_token": "employment_decrease"
  }'
```

### Skip LLM Normalization

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"model_id": "my_model", "data": [...], "use_llm": false}'
```

> See [docs/API.md](docs/API.md) for complete endpoint reference.

---

## Configuration

All settings can be overridden via environment variables. Copy `.env.example` to `.env` and customize:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama generate endpoint |
| `OLLAMA_CHAT` | `http://localhost:11434/api/chat` | Ollama chat endpoint |
| `OLLAMA_MODEL` | `llama3` | Default Ollama model |
| `OLLAMA_TIMEOUT` | `30` | Request timeout (seconds) |
| `VELA_MODELS_DIR` | `models` | Directory for persisted model graphs |
| `VELA_MIN_STRENGTH` | `0.05` | Minimum edge strength threshold |
| `VELA_MAX_ENTITY_WORDS` | `4` | Max words in an entity token |

---

## Project Structure

```
vela-causal/
├── src/vela_causal/          # Core library
│   ├── __init__.py           #   Package exports & version
│   ├── llm_normalizer.py     #   Ollama LLM integration (bias removal)
│   ├── tokenizer.py          #   Structured entity_direction tokenizer
│   ├── extractor.py          #   Explicit causal pair extraction
│   ├── engine.py             #   Directed graph engine & persistence
│   ├── inference.py          #   Beam-search inference & explainability
│   ├── api.py                #   FastAPI application
│   └── repl.py               #   Interactive CLI
├── tests/
│   └── test_engine.py        # Integration test suite
├── docs/                     # Detailed documentation
│   ├── ARCHITECTURE.md
│   ├── INSTALLATION.md
│   ├── API.md
│   └── FAQ.md
├── assets/                   # Presentations, diagrams
├── .github/                  # CI workflows, issue templates
├── repl.py                   # CLI entry point (wrapper)
├── api.py                    # API entry point (wrapper)
├── pyproject.toml            # PEP 621 package config
├── requirements.txt          # pip dependencies
├── .env.example              # Environment variable template
├── LICENSE                   # MIT License
└── README.md                 # This file
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| API Framework | FastAPI + Uvicorn |
| LLM Integration | Ollama (local, via stdlib `urllib`) |
| Data Storage | JSON files (human-readable, debuggable) |
| Graph Traversal | Custom beam-search with log-space scoring |
| Packaging | PEP 621 (`pyproject.toml`) |

---

## Ollama Models

The default model is `llama3`. Change via the `llm_model` parameter or `OLLAMA_MODEL` env var.

Any model available via `ollama list` works. Recommended:

| Model | Notes |
|---|---|
| `llama3` | Best quality |
| `phi3` | Lightweight, fast |
| `mistral` | Good balance |

---

## Testing

```bash
# Run the full integration test suite
python tests/test_engine.py
```

The test suite validates:
1. Tokenizer correctness (structured tokens only)
2. Polarity detection
3. Causal pair extraction
4. Engine training & persistence
5. Directed graph structure (no bidirectional edges)
6. Inference & ranking
7. Explainability paths
8. Domain bias detection
9. Multi-model isolation

---

## How Bias Removal Works

The original engine had bias toward oil/price data because:
1. It used **co-occurrence** — any words near each other became "related"
2. It had **bootstrap training data** seeded with oil/finance sentences
3. No filtering of domain-heavy language

This engine fixes all three:

1. **Ollama Normalizer** — Every input passes through an LLM with `temperature=0` that rewrites text into clean causal pairs and explicitly forbids domain favouritism.
2. **No Co-occurrence** — Only explicit causal markers (`"causes"`, `"leads to"`, `"results in"`) are used.
3. **Structured Tokens Only** — Bare nouns like `"unemployment"` are discarded. Only `entity_direction` format tokens (`"unemployment_increase"`) are kept.
4. **Empty Start** — No bootstrap, no pre-trained data. Every model starts completely empty.

---

## Roadmap

- [ ] Graph visualization dashboard (D3.js / Cytoscape)
- [ ] Batch training from CSV/JSON
- [ ] Confidence calibration & scoring improvements
- [ ] Plugin system for custom extractors
- [ ] Docker image for one-command deployment
- [ ] WebSocket streaming for real-time inference
- [ ] Model export/import (portable graph snapshots)

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Security

For reporting vulnerabilities, please see [SECURITY.md](SECURITY.md).

---

## License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

**Built with purpose. Designed for clarity.**

</div>
