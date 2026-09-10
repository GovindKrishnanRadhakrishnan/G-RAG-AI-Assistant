"""
app.py — Advanced RAG Research Assistant (Streamlit Chat UI)
------------------------------------------------------------
Run with:
    streamlit run app.py

Features:
  • Left sidebar: Ollama connection status, PDF upload + ingest
  • Chat area: user/assistant bubbles with avatars
  • Below each answer: collapsible Sources + Evaluation badges
  • Conversation memory across turns (last 5 exchanges)
  • Clear conversation button
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

# Make src importable when running from project root
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.rag.retriever import AdvancedRetriever
from src.rag.document_processor import DocumentProcessor
from src.config import get_settings

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Advanced RAG Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — sleek dark-mode chat UI
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---------- global ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #0f1117; color: #e2e8f0; }

    /* ---------- sidebar ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #131720 100%);
        border-right: 1px solid #2d3748;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #a78bfa; }

    /* ---------- chat bubbles ---------- */
    .chat-user {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: #fff;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 16px;
        margin: 6px 0;
        max-width: 75%;
        float: right;
        clear: both;
        box-shadow: 0 4px 15px rgba(79,70,229,0.3);
        font-size: 0.95rem;
        line-height: 1.55;
    }
    .chat-assistant {
        background: linear-gradient(135deg, #1e2533 0%, #252d3d 100%);
        color: #e2e8f0;
        border-radius: 18px 18px 18px 4px;
        padding: 12px 16px;
        margin: 6px 0;
        max-width: 85%;
        float: left;
        clear: both;
        border: 1px solid #2d3748;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .chat-clearfix { clear: both; margin: 4px 0; }

    /* ---------- label chips ---------- */
    .label-user {
        font-size: 0.7rem; color: #a78bfa; float: right;
        margin-bottom: 2px; font-weight: 600;
    }
    .label-assistant {
        font-size: 0.7rem; color: #60a5fa; float: left;
        margin-bottom: 2px; font-weight: 600;
    }

    /* ---------- score badges ---------- */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 99px;
        font-size: 0.72rem;
        font-weight: 600;
        margin: 2px 4px 2px 0;
    }
    .badge-green  { background: #064e3b; color: #6ee7b7; border: 1px solid #059669; }
    .badge-yellow { background: #78350f; color: #fcd34d; border: 1px solid #d97706; }
    .badge-red    { background: #7f1d1d; color: #fca5a5; border: 1px solid #dc2626; }

    /* ---------- source card ---------- */
    .source-card {
        background: #1a2035;
        border-left: 3px solid #4f46e5;
        border-radius: 6px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 0.82rem;
        color: #94a3b8;
    }
    .source-title { color: #818cf8; font-weight: 600; }

    /* ---------- header ---------- */
    .main-header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 60%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header { color: #64748b; font-size: 0.9rem; margin-bottom: 1.5rem; }

    /* ---------- status indicator ---------- */
    .status-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .dot-green  { background: #22c55e; box-shadow: 0 0 6px #22c55e; }
    .dot-red    { background: #ef4444; box-shadow: 0 0 6px #ef4444; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
settings = get_settings()

def _init_retriever() -> AdvancedRetriever:
    return AdvancedRetriever(
        persist_directory=settings.VECTOR_DB_PATH,
        ollama_host=settings.OLLAMA_HOST,
        ollama_model=settings.OLLAMA_MODEL,
        top_k_retrieval=settings.TOP_K_RETRIEVAL,
        top_k_final=settings.TOP_K_FINAL,
        compression_threshold=settings.COMPRESSION_THRESHOLD,
        memory_turns=settings.MEMORY_TURNS,
        cross_encoder_model=settings.CROSS_ENCODER_MODEL,
        embed_model_name=settings.EMBED_MODEL,
        run_evaluation=True,
    )

if "retriever" not in st.session_state:
    with st.spinner("⚙️ Loading AI models (first run may take a minute)…"):
        st.session_state.retriever = _init_retriever()

if "messages" not in st.session_state:
    st.session_state.messages = []   # [{role, content, sources, evaluation, rewritten}]

if "docs_ingested" not in st.session_state:
    st.session_state.docs_ingested = 0

retriever: AdvancedRetriever = st.session_state.retriever

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("# 🔬 Research Assistant")
    st.markdown("*Advanced RAG · Powered by Ollama*")
    st.divider()

    # --- Ollama status ---
    st.markdown("### 🌐 Ollama Status")
    try:
        import requests as _req
        r = _req.get(f"{settings.OLLAMA_HOST}/api/tags", timeout=3)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        model_list = ", ".join(models[:4]) or "—"
        st.markdown(
            f'<span class="status-dot dot-green"></span>'
            f'<span style="color:#22c55e;font-size:0.85rem;">Connected</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"Model in use: **{settings.OLLAMA_MODEL}**")
        st.caption(f"Available: {model_list}")
    except Exception:
        st.markdown(
            f'<span class="status-dot dot-red"></span>'
            f'<span style="color:#ef4444;font-size:0.85rem;">Ollama not reachable</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"Expected at `{settings.OLLAMA_HOST}`")

    st.divider()

    # --- Document ingestion ---
    st.markdown("### 📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more research papers to add to the knowledge base.",
    )

    if st.button("🔄 Ingest Documents", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("No files selected.")
        else:
            processor = DocumentProcessor()
            total_chunks = 0
            progress = st.progress(0, text="Processing…")

            for idx, f in enumerate(uploaded_files):
                # Write to a temp file so DocumentProcessor can open it
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name

                try:
                    chunks = processor.process_pdf(tmp_path)
                    # Attach the original filename as metadata
                    for c in chunks:
                        c["metadata"]["source_file"] = f.name
                    retriever.ingest(chunks)
                    total_chunks += len(chunks)
                except Exception as exc:
                    st.error(f"Error processing {f.name}: {exc}")
                finally:
                    os.unlink(tmp_path)

                progress.progress(
                    (idx + 1) / len(uploaded_files),
                    text=f"Processed {idx + 1}/{len(uploaded_files)} files…",
                )

            st.session_state.docs_ingested += total_chunks
            st.success(f"✅ Ingested {total_chunks} chunks from {len(uploaded_files)} file(s).")

    if st.session_state.docs_ingested:
        st.metric("Total Chunks", st.session_state.docs_ingested)

    st.divider()

    # --- Settings display ---
    st.markdown("### ⚙️ Pipeline Settings")
    st.caption(f"Top-K retrieval: **{settings.TOP_K_RETRIEVAL}**")
    st.caption(f"Top-K final (after re-rank): **{settings.TOP_K_FINAL}**")
    st.caption(f"Compression threshold: **{settings.COMPRESSION_THRESHOLD}**")
    st.caption(f"Memory turns: **{settings.MEMORY_TURNS}**")

    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        retriever.clear_memory()
        st.rerun()

# ---------------------------------------------------------------------------
# Main area — Header
# ---------------------------------------------------------------------------
st.markdown('<p class="main-header">Advanced RAG Research Assistant</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Hybrid Search · Cross-Encoder Re-ranking · Context Compression · Source Attribution</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helper: score → badge HTML
# ---------------------------------------------------------------------------

def _score_badge(label: str, score: float) -> str:
    if score >= 0.7:
        cls = "badge-green"
    elif score >= 0.4:
        cls = "badge-yellow"
    else:
        cls = "badge-red"
    return f'<span class="badge {cls}">{label}: {score:.0%}</span>'


# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------

def render_messages() -> None:
    for msg in st.session_state.messages:
        role = msg["role"]

        if role == "user":
            st.markdown(f'<div class="label-user">You</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="chat-clearfix"></div>', unsafe_allow_html=True)

        else:  # assistant
            st.markdown('<div class="label-assistant">🔬 Assistant</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="chat-assistant">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="chat-clearfix"></div>', unsafe_allow_html=True)

            # Sources + Evaluation in expanders (below the bubble)
            col_src, col_eval = st.columns([3, 1])

            with col_src:
                sources = msg.get("sources", [])
                rewritten = msg.get("rewritten_query", "")
                if sources:
                    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
                        if rewritten and rewritten != msg.get("original_question", rewritten):
                            st.caption(f"🔄 Rewritten query: *{rewritten}*")
                        for i, src in enumerate(sources, 1):
                            fname = Path(src.get("source_file", "Unknown")).name
                            page  = src.get("page", "?")
                            snippet = src.get("snippet", "")[:250]
                            st.markdown(
                                f'<div class="source-card">'
                                f'<span class="source-title">[{i}] {fname}</span>'
                                f' · Page {page}<br>'
                                f'<span style="font-size:0.78rem;color:#64748b;">'
                                f'{snippet}…</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

            with col_eval:
                evaluation = msg.get("evaluation", {})
                if evaluation:
                    with st.expander("📊 Scores", expanded=False):
                        faith = evaluation.get("faithfulness", 0.5)
                        rel   = evaluation.get("relevance", 0.5)
                        st.markdown(
                            _score_badge("Faithfulness", faith) + "<br>" +
                            _score_badge("Relevance", rel),
                            unsafe_allow_html=True,
                        )


render_messages()

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Ask a question about your documents…"):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("🧠 Thinking…"):
        result = retriever.query(prompt)

    answer    = result["answer"]
    sources   = result["sources"]
    evals     = result["evaluation"]
    rewritten = result["rewritten_query"]

    st.session_state.messages.append({
        "role"             : "assistant",
        "content"          : answer,
        "sources"          : sources,
        "evaluation"       : evals,
        "rewritten_query"  : rewritten,
        "original_question": prompt,
    })

    st.rerun()
