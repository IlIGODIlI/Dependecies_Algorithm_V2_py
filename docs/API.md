# API Reference

Base URL: `http://localhost:8000`

Interactive Swagger docs available at [`/docs`](http://localhost:8000/docs).

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check (API + Ollama) |
| `GET` | `/models` | List all trained models |
| `GET` | `/models/{model_id}/stats` | Model statistics |
| `GET` | `/models/{model_id}/graph` | Full causal graph |
| `DELETE` | `/models/{model_id}` | Delete a model |
| `POST` | `/train` | Train model on text data |
| `POST` | `/train/file` | Train model from a file |
| `POST` | `/query` | Query for causal predictions |
| `POST` | `/explain` | Explain a causal path |
| `POST` | `/reset` | Reset (wipe) a model |

---

## `GET /health`

Check API and Ollama status.

**Parameters:**

| Name | Type | Default | Description |
|---|---|---|---|
| `llm_model` | query string | `llama3` | Ollama model to check |

**Response:**

```json
{
  "api": "ok",
  "ollama": {
    "ollama": "ok",
    "model": "llama3"
  },
  "models": [
    {
      "model_id": "my_model",
      "description": "",
      "doc_count": 8,
      "edge_count": 8,
      "vocab_size": 16
    }
  ]
}
```

---

## `GET /models`

List all trained models.

**Response:**

```json
{
  "models": [
    {
      "model_id": "my_model",
      "description": "Supply chain model",
      "doc_count": 42,
      "edge_count": 15,
      "vocab_size": 30
    }
  ]
}
```

---

## `GET /models/{model_id}/stats`

Get statistics for a specific model.

**Response:**

```json
{
  "model_id": "my_model",
  "doc_count": 8,
  "vocab_size": 16,
  "edge_count": 8,
  "top_causes": [
    ["company_profit_decrease", 2],
    ["interest_rate_increase", 1]
  ]
}
```

---

## `GET /models/{model_id}/graph`

Return the full causal graph for a model.

**Parameters:**

| Name | Type | Default | Description |
|---|---|---|---|
| `min_strength` | query float | `0.05` | Minimum edge strength to include |

**Response:**

```json
{
  "model_id": "my_model",
  "graph": {
    "company_profit_decrease": [
      {
        "effect": "employment_levels_decrease",
        "strength": 1.0,
        "polarity": 1
      }
    ]
  }
}
```

---

## `DELETE /models/{model_id}`

Delete a model permanently.

**Response (200):**

```json
{"deleted": true, "model_id": "my_model"}
```

**Response (404):**

```json
{"detail": "Model 'my_model' not found"}
```

---

## `POST /train`

Train a model on provided text data.

**Request Body:**

```json
{
  "model_id": "my_model",
  "data": [
    "company profit decrease causes employment decrease",
    "interest rate rise leads to investment decrease"
  ],
  "use_llm": true,
  "llm_model": "llama3",
  "description": "Economic causal model"
}
```

| Field | Type | Default | Required | Description |
|---|---|---|---|---|
| `model_id` | string | `"default"` | No | Model identifier |
| `data` | string[] | — | **Yes** | List of training sentences |
| `use_llm` | bool | `true` | No | Pass through Ollama normalizer |
| `llm_model` | string | `"llama3"` | No | Ollama model name |
| `description` | string | `""` | No | Model description |

**Response (200):**

```json
{
  "model_id": "my_model",
  "pairs_added": 2,
  "total_edges": 2,
  "doc_count": 2,
  "vocab_size": 4
}
```

---

## `POST /train/file`

Train from a text file on the server filesystem.

**Parameters:**

| Name | Type | Default | Description |
|---|---|---|---|
| `path` | query string | — | Path to text file |
| `model_id` | query string | `"default"` | Model identifier |
| `use_llm` | query bool | `true` | Use Ollama normalizer |
| `llm_model` | query string | `"llama3"` | Ollama model name |

---

## `POST /query`

Query the causal model for predictions.

**Request Body:**

```json
{
  "model_id": "my_model",
  "query": "profit decrease",
  "use_llm": true,
  "llm_model": "llama3",
  "top_n": 8
}
```

| Field | Type | Default | Required | Description |
|---|---|---|---|---|
| `model_id` | string | `"default"` | No | Model identifier |
| `query` | string | — | **Yes** | Natural language query |
| `use_llm` | bool | `true` | No | Normalize query through Ollama |
| `llm_model` | string | `"llama3"` | No | Ollama model name |
| `top_n` | int | `8` | No | Max predictions to return |

**Response (200):**

```json
{
  "model_id": "my_model",
  "query": "profit decrease",
  "normalized": "company_profit_decrease causes employment_decrease",
  "tokens": ["company_profit_decrease"],
  "predictions": [
    {
      "token": "employment_levels_decrease",
      "confidence": 0.69,
      "polarity": -1,
      "direction": "decrease",
      "path": ["company_profit_decrease", "employment_levels_decrease"],
      "hops": 1
    }
  ]
}
```

---

## `POST /explain`

Explain the causal path between two tokens.

**Request Body:**

```json
{
  "model_id": "my_model",
  "input_token": "company_profit_decrease",
  "output_token": "employment_levels_decrease"
}
```

**Response (found):**

```json
{
  "model_id": "my_model",
  "found": true,
  "path": ["company_profit_decrease", "employment_levels_decrease"],
  "steps": [
    {
      "from": "company_profit_decrease",
      "to": "employment_levels_decrease",
      "strength": 1.0,
      "polarity": 1
    }
  ],
  "explanation": "company profit decreases, which causes employment levels decreases."
}
```

**Response (not found):**

```json
{
  "model_id": "my_model",
  "found": false,
  "path": [],
  "explanation": "No causal path found from 'token_a' to 'token_b' within 6 hops."
}
```

---

## `POST /reset`

Wipe all data for a model (edges, vocabulary, doc count).

**Request Body:**

```json
{
  "model_id": "my_model"
}
```

**Response:**

```json
{"reset": true, "model_id": "my_model"}
```

---

## Error Responses

All errors follow FastAPI's standard format:

```json
{
  "detail": "Human-readable error message"
}
```

| Status | Meaning |
|---|---|
| `400` | Bad request (missing fields, invalid file path) |
| `404` | Model not found or has no data |
| `422` | Unprocessable entity (no text after normalization) |
