<div align="center">

# EvalPulse

**Real-time LLM health monitoring: hallucination detection, semantic drift, RAG quality, and prompt regression testing**

[![Live Demo](https://img.shields.io/badge/Live_Demo-HuggingFace_Spaces-8b5cf6?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/NinjainPJs/EvalPulse)
[![Tests](https://img.shields.io/badge/Tests-150_passing-22c55e?style=for-the-badge&logo=pytest)](tests/)
[![License](https://img.shields.io/badge/License-Source_Available-f59e0b?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-3776ab?style=for-the-badge&logo=python)](evalpulse/)

5 evaluation modules. $0/month. 3-line integration.

</div>

---

## Overview

EvalPulse is an open-source LLM evaluation platform that monitors the health of LLM applications in real-time. Drop a decorator on any function and EvalPulse automatically scores hallucinations, tracks semantic drift, evaluates RAG quality, measures response toxicity, and runs prompt regression tests against golden datasets.

The live Gradio dashboard shows health trends, alert history, and per-module breakdowns — all backed by SQLite with zero cloud dependencies required.

---

## Pipeline

```
Your LLM Function
        |  @track decorator or EvalContext
        v
+----------------------------------------------------------+
|                    EvalPulse Worker                       |
|                                                           |
|  +---------------+  +---------------+  +-------------+   |
|  | Hallucination |  |   Semantic    |  | RAG Quality |   |
|  |    Scorer     |  |    Drift      |  |  Evaluator  |   |
|  +---------------+  +---------------+  +-------------+   |
|  +---------------+  +------------------------------+     |
|  |   Response    |  |  Prompt Regression Tester    |     |
|  |   Quality     |  |  (golden dataset + CI)       |     |
|  +---------------+  +------------------------------+     |
|                                                           |
|              Health Score Formula (0-100)                 |
+----------------------------+------------------------------+
                             |
             +---------------v--------------+
             |      SQLite Storage (WAL)    |
             +---------------+--------------+
                             |
             +---------------v--------------+
             |      Gradio Dashboard         |
             |   Health trends + Alerts      |
             +------------------------------+
```

---

## Features

| Module | What It Measures |
|--------|-----------------|
| **Hallucination Scorer** | Embedding consistency + LLM-as-judge (Groq Llama-3.1-70B optional) |
| **Semantic Drift Detector** | Cosine distance shift in response embedding distribution |
| **RAG Quality Evaluator** | Faithfulness, context relevance, answer relevancy, groundedness |
| **Response Quality Scorer** | Sentiment, toxicity (detoxify), language detection, denial detection |
| **Prompt Regression Tester** | Golden dataset CI testing via GitHub Actions |
| **Health Score** | Composite 0-100 metric across all 5 modules |
| **Alert System** | Configurable thresholds with Slack/Email notifications |
| **3-line integration** | `@track` decorator or `async with EvalContext(...)` |
| **Security** | Prompt injection protection, SQL injection prevention, SSRF protection |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11+, FastAPI, Uvicorn |
| Dashboard | Gradio 4.0+, Plotly |
| Storage | SQLite (WAL mode), optional PostgreSQL |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB (in-process, on-disk) |
| Hallucination | SelfCheckGPT + Groq Llama-3.1-70B (optional) |
| Toxicity | detoxify (local model) |
| Testing | pytest (150 tests), pytest-asyncio |
| Linting | ruff |
| Deployment | HuggingFace Spaces (free tier) |

---

## Project Structure

```
evalpulse/
├── evalpulse/
│   ├── sdk.py                # @track decorator + EvalContext
│   ├── worker.py             # Background evaluation worker
│   ├── models.py             # EvalEvent + EvalRecord (Pydantic v2)
│   ├── config.py             # YAML config with env var overrides
│   ├── health_score.py       # Composite health score formula
│   ├── alerts.py             # Threshold checking
│   ├── notifications.py      # Slack/Email dispatch
│   ├── cli.py                # CLI commands
│   └── modules/
│       ├── hallucination.py  # Module 1: hallucination scoring
│       ├── drift.py          # Module 2: semantic drift detection
│       ├── rag_eval.py       # Module 3: RAG quality evaluation
│       ├── quality.py        # Module 4: response quality scoring
│       ├── regression.py     # Module 5: prompt regression testing
│       └── embeddings.py     # Shared sentence-transformers
├── dashboard/app.py          # Gradio dashboard (4 tabs)
├── tests/                    # 150 unit + integration tests
└── examples/                 # Groq chatbot, RAG pipeline, FastAPI
```

---

## Quick Start

```bash
git clone https://github.com/ninjacode911/Project-EvalPulse.git
cd Project-EvalPulse
pip install -e .
```

```python
from evalpulse import track

@track(input_key="prompt", output_key="response")
def my_llm_function(prompt: str) -> str:
    return call_llm(prompt)

# Async support
from evalpulse import atrack

@atrack(input_key="prompt", output_key="response")
async def my_async_llm(prompt: str) -> str:
    return await call_llm_async(prompt)
```

```bash
evalpulse dashboard   # Launch Gradio dashboard at http://localhost:7860
evalpulse status      # Check evaluation worker status
pytest tests/         # Run all 150 tests
```

---

## License

Source Available — All Rights Reserved. See [LICENSE](LICENSE) for details.
