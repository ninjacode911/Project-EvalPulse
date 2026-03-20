"""Example: RAG pipeline instrumented with EvalPulse.

Demonstrates RAG quality monitoring with EvalContext.
"""

from evalpulse import EvalContext, init, shutdown

# Simple in-memory knowledge base
KNOWLEDGE_BASE = {
    "python": (
        "Python is a high-level programming language created "
        "by Guido van Rossum in 1991. It emphasizes code "
        "readability and supports multiple paradigms."
    ),
    "machine learning": (
        "Machine learning is a subset of AI that enables "
        "systems to learn from data without being explicitly "
        "programmed. It includes supervised, unsupervised, "
        "and reinforcement learning."
    ),
    "rag": (
        "RAG (Retrieval-Augmented Generation) combines "
        "information retrieval with text generation. It "
        "retrieves relevant documents and uses them as "
        "context for generating responses."
    ),
}


def retrieve(query: str) -> str:
    """Simple keyword-based retriever."""
    query_lower = query.lower()
    for key, doc in KNOWLEDGE_BASE.items():
        if key in query_lower:
            return doc
    return "No relevant documents found."


def generate(query: str, context: str) -> str:
    """Simple response generator (simulated)."""
    return f"Based on the retrieved information: {context[:100]}..."


def rag_answer(query: str) -> str:
    """Full RAG pipeline with EvalPulse monitoring."""
    context = retrieve(query)

    with EvalContext(app="rag-demo", query=query, context=context) as ctx:
        response = generate(query, context)
        ctx.log(response)

    return response


def main():
    init()
    print("RAG Pipeline with EvalPulse Monitoring")
    print("=" * 45)

    queries = [
        "What is Python?",
        "Explain machine learning",
        "How does RAG work?",
        "What is quantum computing?",  # Not in KB
    ]

    for q in queries:
        print(f"\nQ: {q}")
        answer = rag_answer(q)
        print(f"A: {answer}")

    shutdown()
    print("\nDone! Run 'evalpulse dashboard' to see RAG metrics.")


if __name__ == "__main__":
    main()
