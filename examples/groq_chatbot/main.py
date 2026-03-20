"""Example: Groq-powered chatbot instrumented with EvalPulse.

Demonstrates hallucination and quality monitoring on a real LLM.
Requires: GROQ_API_KEY environment variable.
"""

import os

from evalpulse import init, shutdown, track


@track(app="groq-chatbot", model="llama-3.1-70b")
def ask_groq(query: str) -> str:
    """Ask a question using the Groq API."""
    try:
        from groq import Groq

        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[{"role": "user", "content": query}],
            max_tokens=200,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"


def main():
    init()
    print("Groq Chatbot with EvalPulse Monitoring")
    print("=" * 45)

    queries = [
        "What is Python?",
        "Who created Linux?",
        "What is the capital of France?",
    ]

    for q in queries:
        print(f"\nQ: {q}")
        answer = ask_groq(q)
        print(f"A: {answer[:200]}")

    shutdown()
    print("\nDone! Run 'evalpulse dashboard' to see results.")


if __name__ == "__main__":
    main()
