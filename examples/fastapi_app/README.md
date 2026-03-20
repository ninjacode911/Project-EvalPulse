# FastAPI App Example

An async FastAPI application with EvalPulse tracking.

## Setup

```bash
pip install evalpulse fastapi uvicorn
uvicorn examples.fastapi_app.main:app --reload
```

## Endpoints

- `GET /chat?query=Hello` — Chat endpoint with `@atrack`
- `GET /health` — Health check

## What it demonstrates

- `@atrack` async decorator for FastAPI endpoints
- Non-blocking evaluation in async context
