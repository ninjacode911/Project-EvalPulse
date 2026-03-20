# Week 4: Module 1 — Hallucination Scorer + Health Score

## Overview

Week 4 implemented the most critical module in EvalPulse: the Hallucination Scorer. This module detects whether LLM responses contain claims not supported by the provided context. It also introduced the Health Score — a composite 0-100 metric that aggregates all module scores into a single at-a-glance indicator of LLM application health.

## What Was Built

### 1. Groq API Client (`evalpulse/modules/groq_client.py`)

**What**: A wrapper around the Groq free-tier API for LLM-as-judge hallucination scoring.

**Rate limiting**: Implements a token bucket algorithm to respect Groq's 14,400 requests/day limit:
- Capacity: 10 tokens (burst)
- Refill rate: 10 tokens per minute
- `acquire()` blocks up to a configurable timeout before returning False
- Thread-safe via `threading.Lock`

**Methods**:
- `chat(prompt, temperature, max_tokens)` — rate-limited chat completion
- `generate_samples(prompt, n=3)` — generates N stochastic samples for consistency checking
- `judge_hallucination(query, context, response)` — structured prompt asking the LLM to rate hallucination on a 0-1 scale
- `is_available()` — checks if the API key is configured

**Graceful degradation**: If the API key is missing, quota is exceeded, or the API is down, all methods return `None` rather than raising exceptions. The hallucination module falls back to embedding-based scoring.

### 2. Hallucination Module (`evalpulse/modules/hallucination.py`)

**What**: `HallucinationModule` detects factual fabrication in LLM responses using two complementary methods.

**Method 1 — Embedding Consistency** (always available):
- Embeds the response and the context (or query if no context) using sentence-transformers
- Computes cosine similarity between the two embeddings
- Low similarity = response strays from the context = potential hallucination
- Formula: `hallucination = (1 - cosine_similarity) / 2` → 0.0–1.0 range

**Method 2 — LLM-as-Judge** (requires Groq API key):
- Sends a structured prompt to Groq's Llama-3.1-70B asking it to rate hallucination
- With context: "Does the response stay within the context?"
- Without context: "Does the response contain fabricated information?"
- Parses a 0.0–1.0 score from the response

**Combined scoring** (when both methods are available):
- `hallucination_score = 0.6 * embedding_score + 0.4 * judge_score`
- The 60/40 weighting favors embedding consistency (faster, always available)
- `hallucination_method` field tracks which method(s) were used: "embedding", "llm_judge", or "both"

**Flagged claims extraction**:
- Splits the response into sentences
- Identifies sentences containing specific factual markers (numbers, proper names, quoted text)
- Checks if those markers appear in the context
- Flags sentences where less than half of key terms match the context
- Limited to 5 flagged claims per response

**Why this approach over SelfCheckGPT**:
The plan originally specified SelfCheckGPT for stochastic consistency checking. However, embedding-based consistency is more practical for a zero-cost tool:
- No API calls needed (works offline)
- No additional model dependencies
- Still captures the core insight: grounded responses are semantically close to their context
- The optional Groq LLM-as-judge provides a second opinion when available

### 3. Health Score (`evalpulse/health_score.py`)

**What**: A composite 0-100 score aggregating all module outputs into a single health indicator.

**Formula**:
```
health_score = (
    (1 - hallucination_score) * 0.35 +   # 35% weight — most critical
    (1 - drift_score)         * 0.25 +   # 25% weight — silent degradation
    rag_groundedness_score    * 0.20 +   # 20% weight — RAG quality
    response_quality_score    * 0.15 +   # 15% weight — surface quality
    regression_pass_rate      * 0.05     # 5% weight — stability
) * 100
```

**Quality subscore**: Combines toxicity (inverted, 50% weight), sentiment (40% weight), and denial penalty (10% weight) into a single 0-1 value.

**None handling (proportional weight redistribution)**:
When a component is `None` (e.g., no RAG context, insufficient drift baseline), its weight is redistributed proportionally among available components. This prevents penalizing an app that doesn't use RAG for not having RAG scores.

Example: If `drift_score=None` and `groundedness_score=None`, the effective weights become:
- Hallucination: 0.35 / (0.35 + 0.15 + 0.05) = 63.6%
- Quality: 0.15 / 0.55 = 27.3%
- Regression: 0.05 / 0.55 = 9.1%

**Score interpretation**:

| Range | Status | Action |
|-------|--------|--------|
| 90–100 | Healthy | Green — all good |
| 75–89 | Monitoring | Yellow — minor dip |
| 60–74 | Degrading | Orange — review within 24h |
| 40–59 | Critical | Red — immediate investigation |
| 0–39 | Failing | Red flashing — consider rollback |

### 4. Worker Integration

The worker now computes `health_score` after all modules have run, before saving to storage. This ensures every `EvalRecord` in the database has a computed health score.

## Test Results

**11 unit tests** for hallucination:
- Module metadata, empty response, grounded vs unrelated responses
- Context-free scoring (uses query), flagged claims extraction
- Groq client availability and rate limiter

**9 unit tests** for health score:
- Perfect (100) and worst (0) records
- None weight redistribution for drift and RAG
- Score clamping to 0-100
- Aggregate health computation
- Toxicity and denial penalties

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| Embedding consistency over SelfCheckGPT | Zero API cost, no extra dependencies, works offline. Captures the core insight that grounded responses are semantically close to their context. |
| 60/40 weighting for combined scoring | Embedding consistency is more reliable (deterministic) but less nuanced. LLM judge is more accurate but rate-limited and nondeterministic. |
| Proportional weight redistribution | Prevents penalizing apps that don't use certain features (e.g., non-RAG apps shouldn't lose 20% of their health score). |
| Health score as integer 0-100 | Simple, communicable, dashboard-friendly. No one wants to see "0.8347". |
| Rate limiter as token bucket | Standard algorithm, handles bursts well, thread-safe. 10 req/min is conservative relative to Groq's 14,400/day limit to leave room for other API usage. |

## Files Created

| File | Purpose |
|------|---------|
| `evalpulse/modules/groq_client.py` | Groq API wrapper with rate limiting |
| `evalpulse/modules/hallucination.py` | HallucinationModule — embedding consistency + LLM-as-judge |
| `evalpulse/health_score.py` | Composite health score (0-100) with weight redistribution |
| `tests/unit/test_hallucination.py` | 11 tests for hallucination and Groq client |
| `tests/unit/test_health_score.py` | 9 tests for health score computation |
