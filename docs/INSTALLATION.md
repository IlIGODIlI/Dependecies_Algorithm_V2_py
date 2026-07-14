# Installation Guide

This guide covers all the ways to get Vela up and running.

---

## Prerequisites

| Requirement | Version | Required |
|---|---|---|
| Python | 3.9+ | Yes |
| Ollama | Latest | Optional (for LLM normalization) |
| pip | Latest | Yes |

---

## Quick Install

```bash
# Clone the repository
git clone https://github.com/aryan/vela-causal.git
cd vela-causal

# Install Python dependencies
pip install -r requirements.txt
```

That's it — the core engine works out of the box.

---

## Installing with pip (editable mode)

For development, install the package in editable mode:

```bash
pip install -e .

# With dev dependencies (pytest, ruff)
pip install -e ".[dev]"
```

This registers the `vela-repl` CLI command:

```bash
vela-repl --model my_model --no-llm
```

---

## Setting Up Ollama (Optional)

Ollama provides the LLM normalization layer that removes domain bias from training data and queries. The engine works without it, but normalization significantly improves quality.

### 1. Install Ollama

Download from [ollama.ai](https://ollama.ai) or:

```bash
# macOS / Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download installer from https://ollama.ai/download
```

### 2. Pull a Model

```bash
ollama pull llama3
```

Recommended models:

| Model | Size | Speed | Quality |
|---|---|---|---|
| `llama3` | ~4.7 GB | Medium | Best |
| `phi3` | ~2.3 GB | Fast | Good |
| `mistral` | ~4.1 GB | Medium | Good |

### 3. Start Ollama

```bash
ollama serve
```

Ollama runs on `http://localhost:11434` by default.

---

## Environment Variables

Copy the template and adjust:

```bash
cp .env.example .env
```

Available settings:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama generate endpoint |
| `OLLAMA_CHAT` | `http://localhost:11434/api/chat` | Ollama chat endpoint |
| `OLLAMA_MODEL` | `llama3` | Default Ollama model |
| `OLLAMA_TIMEOUT` | `30` | Request timeout (seconds) |
| `VELA_MODELS_DIR` | `models` | Where model JSON files are stored |
| `VELA_MIN_STRENGTH` | `0.05` | Minimum edge strength for query results |
| `VELA_MAX_ENTITY_WORDS` | `4` | Max words in an entity token |

---

## Running

### Interactive CLI

```bash
python repl.py                     # Default model, with LLM
python repl.py --model flood       # Specific model
python repl.py --no-llm            # Skip Ollama (faster)
python repl.py --llm-model mistral # Use Mistral model
```

### REST API Server

```bash
uvicorn api:app --reload --port 8000
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for Swagger UI.

### Run Tests

```bash
python tests/test_engine.py
```

---

## Verifying Installation

Run the health check:

```bash
# Via API
curl http://localhost:8000/health

# Via REPL
python repl.py
# Then type: health
```

Expected output with Ollama running:

```json
{
  "api": "ok",
  "ollama": {"ollama": "ok", "model": "llama3"},
  "models": []
}
```

Without Ollama:

```json
{
  "api": "ok",
  "ollama": {"ollama": "unavailable", "reason": "..."},
  "models": []
}
```

Both are valid — the engine degrades gracefully.

---

## Troubleshooting

### Ollama connection refused

Ensure Ollama is running:

```bash
ollama serve
```

Check the URL matches your environment:

```bash
curl http://localhost:11434/api/tags
```

### UnicodeEncodeError on Windows

Set the encoding before running:

```powershell
$env:PYTHONIOENCODING="utf-8"
python repl.py
```

### FastAPI not found

Install the API dependencies:

```bash
pip install fastapi uvicorn
```
