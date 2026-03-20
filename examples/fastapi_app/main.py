"""Example: FastAPI app with async EvalPulse tracking.

Demonstrates @atrack decorator for async LLM endpoints.
Run: uvicorn examples.fastapi_app.main:app --reload
"""

from evalpulse import atrack, init, shutdown

try:
    from fastapi import FastAPI
except ImportError:
    print("Install fastapi: pip install fastapi uvicorn")
    raise

app = FastAPI(title="EvalPulse FastAPI Demo")


@app.on_event("startup")
async def startup():
    init()


@app.on_event("shutdown")
async def shutdown_event():
    shutdown()


@atrack(app="fastapi-demo", model="echo-model")
async def generate_response(query: str) -> str:
    """Simulated async LLM call."""
    return f"Async response to: {query}"


@app.get("/chat")
async def chat(query: str = "Hello"):
    response = await generate_response(query)
    return {"query": query, "response": response}


@app.get("/health")
async def health():
    return {"status": "ok"}
