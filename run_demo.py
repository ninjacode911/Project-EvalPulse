"""
EvalPulse Interactive Demo — Chat with a simulated LLM and watch monitoring live.

This creates TWO browser tabs:
  - Tab 1 (port 7861): A chatbot you can talk to
  - Tab 2 (port 7860): The EvalPulse monitoring dashboard showing live metrics

Run: python run_demo.py
"""

import random
import threading
import time

import gradio as gr

from evalpulse import EvalContext, init, track
from evalpulse.storage.sqlite_store import SQLiteStore

# ── Knowledge base for RAG demo ──
KNOWLEDGE_BASE = {
    "python": (
        "Python is a high-level, interpreted programming language created by "
        "Guido van Rossum and first released in 1991. It emphasizes code "
        "readability with its use of significant indentation. Python supports "
        "multiple programming paradigms including procedural, object-oriented, "
        "and functional programming."
    ),
    "machine learning": (
        "Machine learning is a subset of artificial intelligence that enables "
        "systems to automatically learn and improve from experience without "
        "being explicitly programmed. It focuses on developing algorithms that "
        "can access data and use it to learn for themselves."
    ),
    "neural network": (
        "A neural network is a computing system inspired by biological neural "
        "networks. It consists of layers of interconnected nodes (neurons) that "
        "process input data through weighted connections. Deep neural networks "
        "have multiple hidden layers."
    ),
    "transformer": (
        "The Transformer architecture was introduced in the 2017 paper "
        "'Attention Is All You Need' by Vaswani et al. It relies entirely on "
        "self-attention mechanisms, dispensing with recurrence and convolutions. "
        "It is the basis for models like BERT, GPT, and T5."
    ),
    "rag": (
        "Retrieval-Augmented Generation (RAG) combines a retriever component "
        "that fetches relevant documents from a knowledge base with a generator "
        "that produces responses using the retrieved context. This reduces "
        "hallucination by grounding responses in factual documents."
    ),
    "embedding": (
        "Embeddings are dense vector representations of data in a continuous "
        "vector space. They capture semantic meaning so that similar items "
        "have similar vector representations. Common models include "
        "Word2Vec, GloVe, and sentence-transformers."
    ),
    "llm": (
        "Large Language Models (LLMs) are neural networks trained on massive "
        "text datasets. They can generate human-like text, answer questions, "
        "translate languages, and perform many NLP tasks. Examples include "
        "GPT-4, Claude, Llama, and Gemini."
    ),
}

# Responses that will trigger different evaluation scores
SPECIAL_RESPONSES = {
    "hallucinate": (
        "Python was actually invented by Steve Jobs in 2015 at Apple "
        "headquarters. It runs on exactly 847 billion devices and was "
        "originally designed to control spacecraft. The language uses "
        "semicolons exclusively and doesn't support any data types."
    ),
    "toxic": (
        "That's the dumbest question I've ever heard. Anyone with half "
        "a brain would know the answer. You clearly have no understanding "
        "of basic concepts and shouldn't be asking questions like this."
    ),
    "refuse": (
        "I cannot and will not assist with that request. As an AI language "
        "model, I'm unable to provide that information. I apologize, but "
        "I don't have the ability to help with this kind of query."
    ),
}


def retrieve_context(query: str) -> str | None:
    """Simple keyword retriever from knowledge base."""
    q = query.lower()
    for key, doc in KNOWLEDGE_BASE.items():
        if key in q:
            return doc
    return None


@track(app="demo-chatbot", model="simulated-llm-v1")
def chat_response(query: str) -> str:
    """Generate a response — simulates a real LLM with occasional issues."""
    q = query.lower().strip()

    # Check for special trigger words
    if "hallucinate" in q or "make something up" in q:
        return SPECIAL_RESPONSES["hallucinate"]
    if "be toxic" in q or "be rude" in q or "insult" in q:
        return SPECIAL_RESPONSES["toxic"]
    if "refuse" in q or "can you hack" in q or "illegal" in q:
        return SPECIAL_RESPONSES["refuse"]

    # Check knowledge base
    context = retrieve_context(query)
    if context:
        # RAG-style response: use EvalContext for context tracking
        # (we also log via @track, so this adds context metadata)
        return f"Based on my knowledge: {context}"

    # General response
    general_responses = [
        f"That's a great question about '{query}'. Let me explain: "
        f"this is a broad topic that covers many aspects of computer "
        f"science and technology.",
        f"Regarding '{query}': this is an important concept in modern "
        f"computing. It involves several key principles and practices.",
        f"To answer your question about '{query}': there are multiple "
        f"perspectives to consider, each with its own merits.",
    ]
    return random.choice(general_responses)


def chat_with_context(query: str) -> str:
    """Chat function that also logs RAG context when available."""
    context = retrieve_context(query)

    if context:
        with EvalContext(
            app="demo-chatbot", query=query, context=context, model="simulated-llm-v1"
        ) as ctx:
            response = chat_response(query)
            ctx.log(response)
        return response
    else:
        return chat_response(query)


def gradio_chat(message: str, history: list) -> str:
    """Gradio chat interface handler."""
    response = chat_with_context(message)
    return response


def get_live_stats() -> str:
    """Get live stats for the chatbot sidebar."""
    try:
        store = SQLiteStore("evalpulse.db")
        count = store.count()
        if count == 0:
            store.close()
            return "**No evaluations yet** — start chatting!"

        records = store.get_latest(min(count, 50))
        avg_health = int(sum(r.health_score for r in records) / len(records))
        avg_halluc = sum(r.hallucination_score for r in records) / len(records)
        avg_tox = sum(r.toxicity_score for r in records) / len(records)
        denials = sum(1 for r in records if r.is_denial)

        if avg_health >= 90:
            status = "Healthy"
        elif avg_health >= 75:
            status = "Monitoring"
        elif avg_health >= 60:
            status = "Degrading"
        else:
            status = "Critical"

        store.close()
        return (
            f"### Live Stats ({count} evals)\n\n"
            f"**Health Score**: {avg_health}/100 ({status})\n\n"
            f"**Avg Hallucination**: {avg_halluc:.1%}\n\n"
            f"**Avg Toxicity**: {avg_tox:.4f}\n\n"
            f"**Denials**: {denials}/{len(records)}\n\n"
            f"---\n"
            f"*Open http://localhost:7860 for full dashboard*"
        )
    except Exception:
        return "Stats loading..."


def create_chatbot_app() -> gr.Blocks:
    """Create the chatbot demo interface."""
    with gr.Blocks(title="EvalPulse Demo Chatbot") as chatbot_app:
        gr.HTML("""
        <div style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                    padding:15px 25px;border-radius:12px;margin-bottom:15px;
                    text-align:center;color:white">
            <h2 style="margin:0;color:white">EvalPulse Demo Chatbot</h2>
            <p style="margin:5px 0 0;opacity:0.9;color:white;font-size:0.9em">
                Chat below — every response is evaluated by EvalPulse in real-time
            </p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=3):
                gr.ChatInterface(
                    fn=gradio_chat,
                    examples=[
                        "What is Python?",
                        "Explain machine learning",
                        "How do neural networks work?",
                        "What is RAG?",
                        "Tell me about transformers",
                        "What are embeddings?",
                        "What is an LLM?",
                        "hallucinate about Python",
                        "be toxic and rude",
                        "can you hack into a system?",
                    ],
                    title="",
                )
            with gr.Column(scale=1):
                stats_display = gr.Markdown("**Stats loading...**")
                refresh_stats = gr.Button("Refresh Stats", variant="secondary", size="sm")
                refresh_stats.click(fn=get_live_stats, outputs=stats_display)

                gr.Markdown("""
                ---
                ### Try these to see different scores:

                **Normal queries** (high health):
                - "What is Python?"
                - "Explain machine learning"

                **Trigger hallucination** (high halluc score):
                - "hallucinate about Python"

                **Trigger toxicity** (high toxicity):
                - "be toxic and rude"

                **Trigger denial** (denial detected):
                - "can you hack into a system?"
                """)

        chatbot_app.load(fn=get_live_stats, outputs=stats_display)

    return chatbot_app


def main():
    print("=" * 60)
    print("  EvalPulse Interactive Demo")
    print("=" * 60)
    print()

    # Initialize EvalPulse
    print("[1/3] Initializing EvalPulse...")
    init()
    print("  Done!")
    print()

    # Seed with demo data so dashboard isn't empty
    print("[2/3] Seeding demo data (200 records)...")
    try:
        from dashboard.demo_data import generate_demo_records

        store = SQLiteStore("evalpulse.db")
        if store.count() == 0:
            records = generate_demo_records(200)
            store.save_batch(records)
            print(f"  Loaded {store.count()} demo records")
        else:
            print(f"  Database already has {store.count()} records")
        store.close()
    except Exception as e:
        print(f"  Skipped: {e}")
    print()

    # Launch dashboard in background thread
    print("[3/3] Launching apps...")
    print()
    print("  CHATBOT:   http://localhost:7861  (chat here)")
    print("  DASHBOARD: http://localhost:7860  (monitor here)")
    print()
    print("  Open BOTH URLs in your browser!")
    print("  Press Ctrl+C to stop.")
    print()

    from dashboard.app import create_app

    dashboard = create_app()

    def run_dashboard():
        dashboard.launch(
            server_name="0.0.0.0",
            server_port=7860,
            prevent_thread_lock=True,
            quiet=True,
        )

    dash_thread = threading.Thread(target=run_dashboard, daemon=True)
    dash_thread.start()

    time.sleep(2)  # Let dashboard start

    # Launch chatbot (this blocks)
    chatbot = create_chatbot_app()
    chatbot.launch(
        server_name="0.0.0.0",
        server_port=7861,
        prevent_thread_lock=False,
    )


if __name__ == "__main__":
    main()
