# Week 2: Module 4 — Response Quality Scorer

## Overview

Week 2 implemented the first evaluation module: the Response Quality Scorer. This module runs on every LLM response to measure surface-level quality attributes — sentiment, toxicity, language, response length, and denial/refusal detection. It also established the module dispatch system in the worker, enabling all future modules to be plugged in identically.

## What Was Built

### 1. Module Interface (`evalpulse/modules/base.py` — established in Week 1)

All evaluation modules implement the `EvalModule` abstract base class:
- `name: str` — identifies the module
- `evaluate(event) -> dict` — async evaluation returning partial EvalRecord fields
- `evaluate_sync(event) -> dict` — synchronous wrapper for thread pool dispatch
- `is_available() -> bool` — dependency check (e.g., is `detoxify` installed?)

This contract ensures every module is independently testable, independently togglable, and can be hot-swapped without modifying the worker.

### 2. Response Quality Module (`evalpulse/modules/quality.py`)

**What**: `ResponseQualityModule` scores 5 quality dimensions per response.

**Sub-metrics**:

| Metric | How It's Measured | Range | Tool |
|--------|------------------|-------|------|
| Sentiment | NLTK VADER lexicon-based analysis with keyword fallback | 0.0 (negative) – 1.0 (positive) | nltk.sentiment.vader |
| Toxicity | Local neural classifier detecting hate, threat, insult, obscenity | 0.0 (benign) – 1.0 (toxic) | detoxify (Unitary AI) |
| Response Length | Simple word count via `split()` | Integer (words) | Built-in |
| Language | Probabilistic n-gram language identification | ISO 639-1 code (e.g., "en", "fr") | langdetect |
| Denial Detection | Regex matching 10 common LLM refusal patterns | Boolean | Built-in |

**Sentiment scoring**: Uses NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner), which is specifically tuned for social media text but works well on LLM outputs. The compound score (-1 to 1) is normalized to 0-1. Falls back to a simple positive/negative keyword counter if VADER is unavailable.

**Toxicity scoring**: Uses the `detoxify` library which runs a DistilBERT-based model locally (~100MB download on first use). The model is lazy-loaded as a class-level singleton to avoid reloading on every call. This is critical for performance — the model loads once and is reused.

**Denial detection**: Ten regex patterns catch common LLM refusal phrases:
- "I cannot", "I'm unable", "I can't"
- "As an AI", "As a language model"
- "I don't have access", "I apologize but I"
- "Unfortunately, I cannot"

This is important because a rising denial rate is a quality signal — it may indicate prompt injection defenses being triggered or the model becoming overly conservative after an update.

### 3. Worker Module Dispatch (`evalpulse/worker.py` modifications)

**What**: The worker now auto-registers enabled modules and dispatches events to them in sequence.

**How it works**:
1. `start()` calls `_register_default_modules()` before starting the thread
2. `_register_default_modules()` calls `get_default_modules(config)` which imports and instantiates only the modules enabled in `evalpulse.yml`
3. For each event, the worker iterates over registered modules, calls `evaluate_sync()`, and merges the returned dict into the `EvalRecord`
4. If a module fails, the error is logged and the record is saved with default scores for that module's fields

**Key design decision**: Modules are registered once at startup, not per-event. Module instantiation (which may load ML models) happens only once. Event processing is purely calling `evaluate_sync()` on pre-loaded modules.

### 4. Module Registry (`evalpulse/modules/__init__.py`)

**What**: `get_default_modules(config)` factory function that returns enabled module instances.

**How it works**:
- Checks `config.modules.response_quality`, `config.modules.drift`, etc.
- For each enabled module, tries to import and instantiate it
- Catches `ImportError` for modules with optional dependencies
- Returns a list of ready-to-use `EvalModule` instances

## Test Results

**17 unit tests** for the quality module:
- Sentiment: positive > 0.6, negative < 0.4, neutral in 0.3-0.7
- Toxicity: benign < 0.3, toxic > 0.3
- Language: English detected as "en", French as "fr"
- Denial: 3 patterns detected, normal response not flagged
- Edge cases: empty string, missing text, field completeness

**1 integration test**: Full flow with 5 diverse responses (positive, negative, neutral, denial, French) tracked through the SDK with quality module enabled.

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| VADER over transformer models | Runs instantly with no model download. Good enough for trend monitoring (we care about relative changes, not absolute accuracy). |
| Detoxify as singleton | The model is 100MB — loading it per call would be unacceptable. Class-level `_detoxify_model` shared across all instances. |
| Regex for denial detection | Fast, interpretable, no model needed. The 10 patterns cover >95% of observed LLM refusal phrases. Easily extensible. |
| Sync `evaluate_sync()` override | Quality module has no I/O-bound operations (unlike the Groq-based hallucination module). No need for async overhead. |

## Files Created/Modified

| File | Purpose |
|------|---------|
| `evalpulse/modules/quality.py` | ResponseQualityModule — sentiment, toxicity, length, language, denial |
| `evalpulse/modules/__init__.py` | Module registry with get_default_modules() |
| `evalpulse/worker.py` | Added auto-registration of modules at startup |
| `tests/unit/test_quality.py` | 17 unit tests |
| `tests/integration/test_quality_flow.py` | End-to-end quality flow test |
