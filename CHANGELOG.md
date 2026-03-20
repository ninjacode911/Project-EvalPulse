# Changelog

## v0.1.0 (March 2026)

### Features

- **SDK**: `@track` decorator, `@atrack` async decorator, `EvalContext` context manager
- **Module 1 — Hallucination Scorer**: Embedding-based consistency + optional Groq LLM-as-judge
- **Module 2 — Semantic Drift Detector**: sentence-transformers + ChromaDB cosine drift
- **Module 3 — RAG Quality Evaluator**: Faithfulness, context relevance, answer relevancy, groundedness
- **Module 4 — Response Quality Scorer**: Sentiment (VADER), toxicity (detoxify), language detection, denial detection
- **Module 5 — Prompt Regression Tester**: Golden dataset testing with configurable thresholds
- **Health Score**: Composite 0-100 score with weighted formula and None redistribution
- **Alert Engine**: Configurable thresholds with cooldown deduplication
- **Notifications**: Email (SMTP) and Slack webhook support
- **Storage**: SQLite (default) with WAL mode for concurrent access
- **Dashboard**: 4-tab Gradio interface (Overview, Hallucination, Drift, RAG & Quality)
- **CLI**: `evalpulse init`, `evalpulse regression run`, `evalpulse dashboard`
- **CI/CD**: GitHub Actions for linting, testing, and regression checks

### Known Limitations

- SelfCheckGPT integration planned but not yet implemented (using embedding consistency instead)
- UMAP visualization in dashboard is placeholder (data available but not yet wired)
- Postgres backend structure exists but not fully tested
- Sliding window for drift baseline not yet implemented (uses full history)

### Dependencies

- Python >= 3.11
- Evidently AI, sentence-transformers, ChromaDB, Gradio, Groq, detoxify, langdetect
