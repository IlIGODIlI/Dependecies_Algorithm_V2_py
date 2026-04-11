"""
api.py
======
FastAPI backend for the Vela causal reasoning platform.

All endpoints pass data through the Ollama LLM normalizer
before it reaches the engine — this is the primary mechanism
for removing domain bias (e.g. oil/price over-representation
from seed training data in the old engine).

Run:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000

Or without uvicorn:
    python api.py
"""

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    print("⚠️  FastAPI not installed. Run: pip install fastapi uvicorn")

from typing import List, Optional
import engine
import inference as inf
from llm_normalizer import check_ollama_health

# ------------------------------------------------------------------
# APP
# ------------------------------------------------------------------

if HAS_FASTAPI:
    app = FastAPI(
        title="Vela Causal Reasoning API",
        description="Production causal inference engine with Ollama LLM normalization",
        version="2.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # tighten in production
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # REQUEST / RESPONSE MODELS
    # ------------------------------------------------------------------

    class TrainRequest(BaseModel):
        model_id:    str = "default"
        data:        List[str]
        use_llm:     bool = True          # set False to bypass Ollama
        llm_model:   str  = "llama3"
        description: Optional[str] = ""

    class QueryRequest(BaseModel):
        model_id:  str = "default"
        query:     str
        use_llm:   bool = True
        llm_model: str  = "llama3"
        top_n:     int  = 8

    class ExplainRequest(BaseModel):
        model_id:     str = "default"
        input_token:  str
        output_token: str

    class ResetRequest(BaseModel):
        model_id: str

    class DeleteRequest(BaseModel):
        model_id: str

    # ------------------------------------------------------------------
    # HEALTH
    # ------------------------------------------------------------------

    @app.get("/health")
    def health(llm_model: str = "llama3"):
        """Check API and Ollama status."""
        ollama_status = check_ollama_health(llm_model)
        return {
            "api":    "ok",
            "ollama": ollama_status,
            "models": engine.list_models(),
        }

    # ------------------------------------------------------------------
    # MODELS
    # ------------------------------------------------------------------

    @app.get("/models")
    def list_models():
        """List all trained models."""
        return {"models": engine.list_models()}

    @app.delete("/models/{model_id}")
    def delete_model(model_id: str):
        """Delete a model."""
        deleted = engine.delete_model(model_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        return {"deleted": True, "model_id": model_id}

    @app.get("/models/{model_id}/stats")
    def model_stats(model_id: str):
        """Get statistics for a model."""
        return engine.get_stats(model_id)

    @app.get("/models/{model_id}/graph")
    def model_graph(model_id: str, min_strength: float = 0.05):
        """Return the full causal graph for a model."""
        graph = engine.get_graph(model_id, min_strength=min_strength)
        # Convert to serializable format
        serialized = {
            cause: [
                {"effect": effect, "strength": round(s, 4), "polarity": p}
                for effect, s, p in neighbors
            ]
            for cause, neighbors in graph.items()
        }
        return {"model_id": model_id, "graph": serialized}

    # ------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------

    @app.post("/train")
    def train(req: TrainRequest):
        """
        Train a model on provided text data.

        The `use_llm` flag (default: True) routes all text through the
        Ollama normalizer before training. This:
          - Rewrites raw input into clean causal sentences
          - Removes domain-specific framing (e.g. finance/weather bias)
          - Discards non-causal content
          - Ensures only explicit cause→effect relationships are learned

        Set `use_llm: false` to skip normalization (faster, but no bias removal).
        """
        if not req.data:
            raise HTTPException(status_code=400, detail="'data' field is required and must not be empty")

        result = engine.train(
            model_id    = req.model_id,
            texts       = req.data,
            use_llm     = req.use_llm,
            llm_model   = req.llm_model,
            description = req.description or "",
        )

        if "error" in result:
            raise HTTPException(status_code=422, detail=result["error"])

        return result

    @app.post("/train/file")
    def train_from_file(path: str, model_id: str = "default", use_llm: bool = True, llm_model: str = "llama3"):
        """Train from a file path on the server."""
        result = engine.train_from_file(
            model_id  = model_id,
            path      = path,
            use_llm   = use_llm,
            llm_model = llm_model,
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result

    # ------------------------------------------------------------------
    # QUERY
    # ------------------------------------------------------------------

    @app.post("/query")
    def query(req: QueryRequest):
        """
        Query the causal model.

        The query text is normalized through Ollama before tokenization,
        ensuring that informal or biased query language is cleaned up
        (e.g. "what happens when oil gets expensive" → "oil_price_increase").

        Returns ranked predictions with confidence scores and causal paths.
        """
        graph = engine.get_graph(req.model_id)

        if not graph:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{req.model_id}' has no trained data. POST to /train first."
            )

        result = inf.query(
            model_id   = req.model_id,
            query_text = req.query,
            graph      = graph,
            use_llm    = req.use_llm,
            llm_model  = req.llm_model,
            top_n      = req.top_n,
        )

        # Flatten predictions for clean API response
        predictions = [
            {
                "token":      r["token"],
                "confidence": r["confidence"],
                "polarity":   r["polarity"],
                "direction":  "increase" if r["polarity"] > 0 else "decrease" if r["polarity"] < 0 else "neutral",
                "path":       r["path"],
                "hops":       r["depth"],
            }
            for r in result["predictions"]
            if not r["is_input"]   # don't return the input tokens as predictions
        ]

        return {
            "model_id":    req.model_id,
            "query":       req.query,
            "normalized":  result.get("normalized", req.query),
            "tokens":      result.get("tokens", []),
            "predictions": predictions,
        }

    # ------------------------------------------------------------------
    # EXPLAIN
    # ------------------------------------------------------------------

    @app.post("/explain")
    def explain(req: ExplainRequest):
        """
        Explain the causal path from input_token to output_token.
        Used by the frontend when a user clicks on a prediction.
        """
        graph  = engine.get_graph(req.model_id)
        result = inf.explain_path(graph, req.input_token, req.output_token)

        if not result["found"]:
            return {
                "model_id":    req.model_id,
                "found":       False,
                "path":        [],
                "explanation": result["explanation"],
            }

        return {
            "model_id":    req.model_id,
            "found":       True,
            "path":        result["path"],
            "steps":       result["steps"],
            "explanation": result["explanation"],
        }

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------

    @app.post("/reset")
    def reset(req: ResetRequest):
        """Reset (wipe) all data for a model."""
        return engine.reset_model(req.model_id)

    # ------------------------------------------------------------------
    # ENTRY POINT
    # ------------------------------------------------------------------

    if __name__ == "__main__":
        uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

else:
    print("FastAPI not available — install with: pip install fastapi uvicorn")
