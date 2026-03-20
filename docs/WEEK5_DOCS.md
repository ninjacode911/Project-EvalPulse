# Week 5: Module 3 — RAG Quality Evaluator

## Overview

Week 5 built the RAG Quality Evaluator — the module that measures retrieval-augmented generation pipeline quality. It scores how relevant retrieved context is, whether the LLM response stays faithful to that context, and whether the response actually answers the user's query.

## What Was Built

### RAG Quality Module (`evalpulse/modules/rag_eval.py`)

**What**: `RAGQualityModule` evaluates 4 RAG-specific metrics using embedding similarity.

**Four metrics**:

1. **Context Relevance** (query vs context): Is the retrieved context actually relevant to the query? Computed as the cosine similarity between the query embedding and the context embedding. A high score means the retrieval system is pulling in relevant documents; a low score means the retriever is returning noise.

2. **Faithfulness** (response vs context): Does the response stay within the boundaries of the retrieved context? This metric blends two signals:
   - **Full-text embedding similarity (60% weight)**: Cosine similarity between the entire response and the entire context. Captures whether the response is broadly about the same topic as the context.
   - **Sentence-level analysis (40% weight)**: Splits the response into individual sentences, embeds each one separately, and computes cosine similarity of each sentence against the full context embedding. The final sentence-level score is the mean of all sentence similarities. This catches cases where the overall response is topically similar but individual sentences contain fabricated details not present in the context.

3. **Answer Relevancy** (query vs response): Does the response address what the user actually asked? Cosine similarity between the query embedding and the response embedding. A high score means the LLM stayed on-topic; a low score means the response wandered or answered a different question.

4. **Groundedness**: A weighted composite score that combines the other three metrics into a single quality indicator:
   - 40% faithfulness
   - 30% context relevance
   - 30% answer relevancy

   Faithfulness gets the highest weight because unfaithful responses (hallucinations grounded in the wrong information) are the most dangerous failure mode in RAG systems.

### Why These Metrics

RAG pipelines have three distinct failure points, and each metric targets one:

- **Retriever failure** -> Context Relevance catches this. If the retriever returns irrelevant documents, context relevance drops even if the LLM does its best with bad input.
- **Generator hallucination** -> Faithfulness catches this. The LLM might generate plausible-sounding text that goes beyond what the context actually says.
- **Off-topic response** -> Answer Relevancy catches this. The LLM might faithfully summarize the context but fail to answer the actual question.

Groundedness provides a single number for dashboards and alerts when you need one metric to represent overall RAG quality.

### Sentence-Level Faithfulness Refinement

The sentence-level analysis exists because full-text embedding similarity can miss localized hallucinations. Consider a response that is 90% faithful but contains one fabricated statistic. The full-text embedding similarity will still be high because the overall semantic content matches. But when you embed that one fabricated sentence individually and compare it against the context, its similarity drops noticeably. The 60/40 blend lets the sentence-level signal pull the score down without over-penalizing long responses where one sentence is slightly off-topic.

### Non-RAG Handling

When `context` is None (meaning the LLM call was not a RAG call), all 4 metric fields return None. This is a deliberate design choice: non-RAG applications should not be penalized with low RAG scores. The downstream dashboard and alert engine check for None and skip RAG metrics for non-RAG calls.

### Why Embedding Similarity Over Evidently RAGEvals

Evidently's RAG evaluation presets (`ColumnMapping` + `Report` with RAG presets) require DataFrame-based batch processing. They are designed for offline evaluation of datasets, not real-time per-call scoring. For EvalPulse's use case — scoring every LLM call as it happens — direct embedding comparison is:

- **Faster**: A cosine similarity computation on pre-computed embeddings is a single dot product, taking microseconds. No DataFrame construction or report generation overhead.
- **Simpler**: No dependency on Evidently's RAG column mapping conventions or report parsing.
- **Sufficient**: Cosine similarity on good embeddings (sentence-transformers) captures semantic relationships accurately enough for monitoring and alerting. We are not doing academic-grade evaluation — we are detecting regressions and anomalies.

## Test Results

11 unit tests covering:
- Module metadata (name, description, version)
- Non-RAG handling (all metrics None when context is None)
- Relevant context scoring (high scores for semantically related query/context)
- Irrelevant context scoring (low scores for unrelated query/context)
- Unfaithful response detection (low faithfulness when response contradicts context)
- Score range validation (all scores between 0.0 and 1.0)
- Groundedness composite correctness (verifies 40/30/30 weighting)
- Edge cases (empty strings, very short inputs, single-word queries)

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| Cosine similarity for all metrics | Fast (dot product on normalized vectors), interpretable (0-1 range), no external API calls needed, works offline |
| 60/40 blend for faithfulness | Full-text similarity captures overall topic match; sentence-level catches localized fabricated details that full-text misses |
| 40/30/30 groundedness weighting | Faithfulness is the most critical RAG quality signal (hallucinated content is worse than off-topic or irrelevant retrieval); relevance and answer quality share the remaining weight equally |
| None for non-RAG calls | Prevents penalizing applications that do not use retrieval; downstream components check for None |
| Sentence splitting via simple regex | Avoids NLP library dependency (spaCy/NLTK) for a simple task; handles period/question/exclamation splits well enough for monitoring purposes |

## Files Created

| File | Purpose |
|------|---------|
| `evalpulse/modules/rag_eval.py` | RAGQualityModule with 4 metrics (context relevance, faithfulness, answer relevancy, groundedness) |
| `tests/unit/test_rag_eval.py` | 11 unit tests covering all metrics, edge cases, and non-RAG handling |

## Interview Talking Points

- **Why not use an LLM-as-judge for faithfulness?** LLM-as-judge (e.g., GPT-4 grading whether a response is faithful) adds latency, cost, and a dependency on an external API for every evaluation. For real-time monitoring, embedding similarity provides a fast, deterministic signal. LLM-as-judge can be added as an optional "deep eval" mode later.

- **How would you improve faithfulness detection?** The current approach misses logical contradictions (e.g., context says "X happened in 2020" and response says "X happened in 2019"). NLI (Natural Language Inference) models could detect entailment vs contradiction at the sentence level. This would replace the sentence-level cosine similarity with an entailment classifier.

- **Why not TF-IDF or BM25 for context relevance?** TF-IDF/BM25 measure lexical overlap, not semantic similarity. A query about "car" and context about "automobile" would score low with lexical methods but high with embeddings. Since modern RAG systems use semantic retrieval, the evaluation should match.
