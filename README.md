<div align="center">

# EvalPulse

**Real-Time LLM Health Monitoring — Hallucination, Drift, RAG Quality & Prompt Regression**

*Evaluate your LLMs the way production systems demand.*

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-Source%20Available-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-150%20passing-brightgreen)](tests/)
[![HF Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-FFD21E)](https://huggingface.co/spaces/NinjainPJs/EvalPulse)

[**Live Demo →**](https://huggingface.co/spaces/NinjainPJs/EvalPulse)

</div>

---

## Overview

EvalPulse is an open-source LLM evaluation platform that monitors the health of LLM applications in real-time. Drop a single decorator onto any function and EvalPulse automatically scores hallucinations, tracks semantic drift, evaluates RAG quality, measures response toxicity, and runs prompt regression tests against golden datasets.

The live Gradio dashboard shows health trends, alert history, and per-module breakdowns — all backed by SQLite with zero cloud dependencies required and a $0/month operating cost.

**What makes this different from typical evaluation libraries:**
- **5 evaluation modules** — hallucination, semantic drift, RAG quality, response quality, and prompt regression all running in a single background worker.
- **3-line integration** — `@track` decorator or `async with EvalContext(...)` — no refactoring needed.
- **Health Score formula** — a single 0-100 composite metric combining all 5 modules into one actionable number.
- **Alert system** — configurable thresholds trigger Slack or email notifications automatically.
- **Security-first** — prompt injection protection via XML delimiters, parameterized SQL, SSRF protection.

---

## Architecture

```
Your LLM Function
      |  @track(input_key="prompt", output_key="response")
      v
+-------------------------------------------------------------+
|                    EvalPulse Worker                          |
|  (background thread — non-blocking, zero latency overhead)  |
|                                                             |
|  +------------------+  +-------------------+               |
|  | Hallucination    |  | Semantic Drift     |               |
|  | Scorer           |  | Detector           |               |
|  | SelfCheckGPT +   |  | Cosine distance    |               |
|  | LLM-as-judge     |  | over rolling window|               |
|  +------------------+  +-------------------+               |
|                                                             |
|  +------------------+  +-------------------+               |
|  | RAG Quality      |  | Response Quality   |               |
|  | Evaluator        |  | Scorer             |               |
|  | faithfulness +   |  | toxicity + lang +  |               |
|  | relevance +      |  | sentiment +        |               |
|  | groundedness     |  | denial detection   |               |
|  +------------------+  +-------------------+               |
|                                                             |
|  +-------------------------------------------+             |
|  | Prompt Regression Tester                   |             |
|  | golden dataset + GitHub Actions CI         |             |
|  +-------------------------------------------+             |
|                                                             |
|              Health Score Formula (0-100)                   |
+----------------------------+--------------------------------+
                             |
             +---------------v--------------+
             |  SQLite Storage (WAL mode)    |
             |  EvalRecord per call          |
             +---------------+--------------+
                             |
             +---------------v--------------+
             |    Gradio Dashboard (4 tabs)  |
             |  Health trends, alert log,    |
             |  per-module breakdown, config |
             +------------------------------+
```

---

## Features

| Feature | Detail |
|---------|--------|
| **Hallucination Scorer** | Embedding consistency + optional LLM-as-judge (Groq Llama-3.1-70B) |
| **Semantic Drift Detector** | Cosine distance shift in response embedding distribution over time |
| **RAG Quality Evaluator** | Faithfulness, context relevance, answer relevancy, and groundedness |
| **Response Quality Scorer** | Sentiment, toxicity (detoxify), language detection, denial detection |
| **Prompt Regression Tester** | Golden dataset CI testing — fails the GitHub Actions build on regression |
| **Health Score** | Single 0-100 composite metric across all 5 modules |
| **Alert System** | Configurable thresholds trigger Slack or Email notifications |
| **3-line integration** | `@track` decorator or `async with EvalContext(...)` — no refactoring |
| **Async support** | `@atrack` for FastAPI and async codebases |
| **$0/month** | Runs entirely on free-tier infrastructure |

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Backend** | Python 3.11+, FastAPI, Uvicorn | REST API and async evaluation worker |
| **Dashboard** | Gradio 4.0+, Plotly | Live health dashboard with 4 tabs |
| **Storage** | SQLite (WAL mode) | Evaluation records and alert log |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | Semantic similarity and drift detection |
| **Vector Store** | ChromaDB (in-process, on-disk) | Embedding storage for drift baseline |
| **Hallucination** | SelfCheckGPT + Groq Llama-3.1-70B | Multi-sample consistency + LLM-as-judge |
| **Toxicity** | detoxify (local model) | Response toxicity scoring |
| **Config** | Pydantic-settings + YAML | Centralized, env-override config |
| **Testing** | pytest (150 tests), pytest-asyncio | Unit and integration test coverage |
| **Linting** | ruff | Fast Python linting and formatting |
| **Deployment** | HuggingFace Spaces (free tier) | Live demo hosting |

---

## Project Structure

```
evalpulse/
├── evalpulse/
│   ├── sdk.py                # @track decorator + EvalContext + @atrack
│   ├── worker.py             # Background evaluation worker thread
│   ├── models.py             # EvalEvent + EvalRecord (Pydantic v2)
│   ├── config.py             # YAML config with env var overrides
│   ├── health_score.py       # Composite 0-100 Health Score formula
│   ├── alerts.py             # Threshold checking and alert dispatch
│   ├── notifications.py      # Slack and Email notification handlers
│   ├── cli.py                # CLI: dashboard, status, config commands
│   └── modules/
│       ├── hallucination.py  # Module 1: SelfCheckGPT + LLM-as-judge
│       ├── drift.py          # Module 2: cosine-based semantic drift
│       ├── rag_eval.py       # Module 3: faithfulness + relevance + groundedness
│       ├── quality.py        # Module 4: toxicity + sentiment + denial
│       ├── regression.py     # Module 5: golden dataset regression testing
│       └── embeddings.py     # Shared sentence-transformers wrapper
├── dashboard/
│   └── app.py                # Gradio dashboard (health, drift, alerts, config)
├── tests/                    # 150 unit + integration tests
└── examples/
    ├── groq_chatbot.py       # Example: Groq chatbot with @track
    ├── rag_pipeline.py       # Example: RAG pipeline evaluation
    └── fastapi_integration.py # Example: async FastAPI with @atrack
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Optionally a Groq API key ([free at console.groq.com](https://console.groq.com)) for LLM-as-judge hallucination scoring

### 1. Clone and install

```bash
git clone https://github.com/ninjacode911/Project-EvalPulse.git
cd Project-EvalPulse
pip install -e .
```

### 2. Add the decorator

```python
from evalpulse import track

@track(input_key="prompt", output_key="response")
def my_llm_function(prompt: str) -> str:
    return call_llm(prompt)

# Async FastAPI support
from evalpulse import atrack

@atrack(input_key="prompt", output_key="response")
async def my_async_llm(prompt: str) -> str:
    return await call_llm_async(prompt)
```

### 3. Launch the dashboard

```bash
evalpulse dashboard   # http://localhost:7860
evalpulse status      # check evaluation worker status
```

---

## Running Tests

```bash
pytest tests/ -v
# Expected: 150 passed
```

---

## Configuration

All configuration is driven by `evalpulse.yaml` (or environment variables). Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | Optional. Enables LLM-as-judge hallucination scoring |
| `SLACK_WEBHOOK_URL` | — | Optional. Enables Slack alert notifications |
| `ALERT_HEALTH_THRESHOLD` | `70` | Health Score below this triggers an alert |
| `DRIFT_WINDOW` | `100` | Rolling window size for drift baseline |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model for embeddings |

---

## License

**Source Available — All Rights Reserved.** See [LICENSE](LICENSE) for full terms.

The source code is publicly visible for viewing and educational purposes. Any use in personal, commercial, or academic projects requires explicit written permission from the author.

To request permission: navnitamrutharaj1234@gmail.com

**Author:** Navnit Amrutharaj
