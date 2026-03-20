# Groq Chatbot Example

A simple chatbot powered by Groq's free API, instrumented with EvalPulse.

## Setup

```bash
export GROQ_API_KEY=gsk_your_key_here
pip install evalpulse
python main.py
```

## What it demonstrates

- `@track` decorator on a real LLM function
- Hallucination scoring
- Quality monitoring (sentiment, toxicity)
- Health score computation
