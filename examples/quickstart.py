"""EvalPulse Quick Start Example.

Demonstrates minimal integration with the @track decorator.
Run: python examples/quickstart.py
"""

from evalpulse import init, shutdown, track

# Initialize EvalPulse
init()


@track(app="quickstart-demo", model="echo-model")
def my_llm(query: str) -> str:
    """Simulated LLM function."""
    return f"Here is the answer to: {query}"


def main():
    print("EvalPulse Quick Start Demo")
    print("=" * 40)

    queries = [
        "What is Python?",
        "Explain machine learning",
        "How do neural networks work?",
        "What is RAG?",
        "Describe transformers",
    ]

    for q in queries:
        response = my_llm(q)
        print(f"Q: {q}")
        print(f"A: {response}")
        print()

    # Wait for background worker to process all events
    # (first run downloads models which takes a few seconds)
    import time

    print("Waiting for evaluation to complete...")
    time.sleep(10)

    print("Shutting down EvalPulse...")
    shutdown(timeout=15.0)
    print("Done! Check evalpulse.db for evaluation results.")


if __name__ == "__main__":
    main()
