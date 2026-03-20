# Week 6: Module 5 — Prompt Regression Tester + CLI

## Overview

Week 6 built the Prompt Regression Tester and the CLI. The regression module runs automated tests against golden datasets to catch quality regressions when prompts change or models are updated. The CLI provides `evalpulse init`, `evalpulse regression run`, and `evalpulse dashboard` commands, enabling CI/CD integration and developer workflow.

## What Was Built

### 1. Golden Dataset System (`evalpulse/modules/golden_dataset.py`)

**What**: Pydantic models for defining test expectations with JSON serialization.

**Models**:

- `GoldenExample`: Represents a single test case.
  - `query` (str): The input question or prompt.
  - `expected_response` (str): The ideal or reference response (used for comparison, not exact matching).
  - `context` (Optional[str]): Retrieved context for RAG test cases; None for non-RAG cases.
  - `max_hallucination` (Optional[float]): Per-example hallucination threshold. If the hallucination score exceeds this, the example fails.
  - `max_toxicity` (Optional[float]): Per-example toxicity threshold.
  - `min_faithfulness` (Optional[float]): Per-example faithfulness floor. Only checked when context is provided.
  - `expected_language` (Optional[str]): Expected response language code (e.g., "en"). Fails if the detected language does not match.

- `GoldenDataset`: Named collection of examples.
  - `name` (str): Dataset identifier (e.g., "customer-support-v2").
  - `version` (str): Semantic version for tracking dataset evolution.
  - `examples` (List[GoldenExample]): The test cases.

- `load_golden_dataset(path) -> GoldenDataset`: Reads JSON file, validates with Pydantic.
- `save_golden_dataset(dataset, path)`: Serializes to formatted JSON.

**Why per-example thresholds**: Different queries have fundamentally different quality expectations. A factual question ("What is the capital of France?") needs strict hallucination limits because there is one correct answer. A creative writing prompt ("Write a poem about autumn") can tolerate higher hallucination scores because creativity inherently diverges from training data patterns. Per-example thresholds let teams encode these expectations directly in the test data rather than using a one-size-fits-all global threshold.

**Why JSON format**: Golden datasets are version-controlled alongside code. JSON is human-readable (teams can review diffs in PRs), machine-parseable (CI systems can generate them programmatically), and requires no additional dependencies to load. YAML was considered but adds a PyYAML dependency for minimal readability improvement over formatted JSON.

### 2. Regression Test Runner (`evalpulse/modules/regression.py`)

**What**: `PromptRegressionModule` runs batch regression tests against a golden dataset.

**How it works**:

1. Takes an `llm_function` (callable that accepts a query string and returns a response string) and a `GoldenDataset`.
2. For each `GoldenExample` in the dataset:
   a. Calls `llm_function(example.query)` to get the actual response.
   b. Optionally runs configured eval modules (hallucination, toxicity, RAG) on the response.
   c. Checks for violations against per-example thresholds.
3. Returns a `RegressionResult` containing:
   - `total`: Number of examples tested.
   - `passed`: Number of examples with zero violations.
   - `failed`: Number of examples with at least one violation.
   - `pass_rate`: `passed / total` as a float.
   - `failures`: List of `FailureDetail` objects with the query, violations, actual scores, and expected thresholds.

**Violation checks** (in order):
1. **Empty response**: Response is None or empty string.
2. **Hallucination threshold**: `hallucination_score > example.max_hallucination` (if threshold is set).
3. **Toxicity threshold**: `toxicity_score > example.max_toxicity` (if threshold is set).
4. **Faithfulness minimum**: `faithfulness_score < example.min_faithfulness` (if threshold is set and context is present).
5. **Language match**: Detected language does not match `example.expected_language` (if set).

**Batch mode vs real-time**: Unlike other EvalPulse modules that run per-event (intercepting LLM calls via the `@track` decorator), the regression module runs as a batch test suite. It calls the LLM function directly, making it suitable for CI pipelines where you want to test a prompt template or model change against a fixed set of expectations before deploying.

### 3. CLI (`evalpulse/cli.py`)

**What**: Command-line interface with three subcommands, built with argparse.

**Commands**:

#### `evalpulse init [--force]`
- Creates an `evalpulse.yml` configuration file in the current directory from a built-in template.
- The template includes all configurable options with comments explaining each one.
- `--force` overwrites an existing config file.
- Exits with code 0 on success, code 1 if the file already exists (without `--force`).

#### `evalpulse regression run --dataset PATH [--output DIR] [--threshold FLOAT]`
- Loads the golden dataset from the specified JSON file.
- Runs the regression test suite.
- Prints a summary table (total, passed, failed, pass rate).
- If `--output` is specified, writes detailed results to a JSON file in that directory.
- If the pass rate is below `--threshold` (default: 1.0, meaning all must pass), exits with code 1.
- If the pass rate meets or exceeds the threshold, exits with code 0.

#### `evalpulse dashboard [--host HOST] [--port PORT] [--share]`
- Launches the Gradio monitoring dashboard.
- `--host` defaults to `127.0.0.1` (localhost only).
- `--port` defaults to `7860` (Gradio default).
- `--share` creates a public Gradio share link for remote access.

**CI/CD integration**: The `--threshold` flag and non-zero exit code are specifically designed for CI pipelines. A GitHub Actions workflow can run `evalpulse regression run --dataset tests/golden.json --threshold 0.95` and the step will fail if more than 5% of test cases regress. No wrapper scripts needed.

**Why argparse over click/typer**: EvalPulse has exactly three subcommands. Click and Typer add dependencies and decorator-based APIs that are overkill for this scope. Argparse ships with Python, produces adequate help text, and handles subcommand routing cleanly with `add_subparsers()`.

### 4. Sample Golden Dataset (`examples/golden_datasets/sample_golden.json`)

5 examples covering diverse test scenarios:
1. **Python basics**: General knowledge question, no context.
2. **Machine learning**: Definition question with strict hallucination threshold (0.2).
3. **Neural networks**: Technical explanation with moderate thresholds.
4. **RAG scenario**: Includes context field, tests faithfulness with min_faithfulness threshold.
5. **Deep learning**: Open-ended question with relaxed thresholds.

## Test Results

**Regression tests** (12 unit tests):
- Golden dataset CRUD (create, load, save, roundtrip)
- Perfect LLM behavior (all examples pass)
- Empty response LLM (all examples fail with "empty response" violation)
- Failing LLM with threshold violations (hallucination, toxicity, language)
- Pass rate calculation accuracy
- JSON serialization roundtrip (save then load preserves all fields)
- Empty dataset handling (0 examples -> 0 pass rate without crashing)
- Per-example threshold isolation (one example's thresholds do not affect others)

**CLI tests** (3 unit tests):
- Help output contains all three subcommand names
- `init` creates a config file with expected structure
- `regression run` with missing dataset file exits with error message

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| JSON for golden datasets | Human-readable, version-controllable in Git, easy to generate programmatically from logs or spreadsheets, no extra dependencies |
| Per-example thresholds | Different queries need different quality bars; global thresholds are too coarse for mixed workloads |
| Exit code 1 on failure | Unix convention for CI/CD integration; GitHub Actions, Jenkins, and all CI systems interpret non-zero exit as failure |
| argparse over click/typer | Zero additional dependencies; three subcommands do not justify a framework |
| Batch mode for regression | Regression testing is inherently a batch operation (run N tests, report results); real-time per-call evaluation is handled by other modules |
| LLM function as callable parameter | Decouples the regression runner from any specific LLM provider; works with OpenAI, Anthropic, local models, or mock functions |

## Files Created

| File | Purpose |
|------|---------|
| `evalpulse/modules/golden_dataset.py` | Golden dataset Pydantic schema + JSON I/O functions |
| `evalpulse/modules/regression.py` | PromptRegressionModule batch test runner |
| `evalpulse/cli.py` | CLI with init, regression run, and dashboard subcommands |
| `examples/golden_datasets/sample_golden.json` | Sample golden dataset with 5 diverse examples |
| `tests/unit/test_regression.py` | 12 unit tests for regression module |
| `tests/unit/test_cli.py` | 3 unit tests for CLI |

## Interview Talking Points

- **How does this compare to tools like promptfoo or DeepEval?** Promptfoo and DeepEval are standalone evaluation frameworks. EvalPulse's regression module is integrated into a broader monitoring platform — the same metrics used for regression testing (hallucination, faithfulness, toxicity) are the same metrics used for real-time monitoring and alerting. This means regression test thresholds stay consistent with production alert thresholds.

- **How would you scale regression testing to thousands of examples?** The current implementation runs sequentially. For large datasets, you would add async execution with `asyncio.gather()` to parallelize LLM calls (the bottleneck), add progress reporting, and support dataset sharding for distributed CI runners.

- **Why not compare against expected_response directly?** Exact match or even BLEU/ROUGE comparison is too brittle for LLM outputs. The same correct answer can be phrased many different ways. Instead, we check behavioral properties (hallucination, toxicity, faithfulness) that should hold regardless of phrasing.
