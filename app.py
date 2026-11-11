import json
import os
import socket
import uuid
from pathlib import Path

import streamlit as st

from rag.agent import Agent, transcribe_audio
from rag.ingest import load_bytes
from rag.store import VectorStore

# Force offline flags off so models can download on cloud servers like Streamlit Cloud
for _var in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]:
    if _var in os.environ:
        del os.environ[_var]
    os.environ[_var] = "0"

try:
    import huggingface_hub.constants

    huggingface_hub.constants.HF_HUB_OFFLINE = False
except Exception:
    pass

# Prioritize IPv4 over stalled IPv6 network routes
_orig_gai = socket.getaddrinfo


def _ipv4_first_gai(*args, **kwargs):
    results = _orig_gai(*args, **kwargs)
    ipv4 = [r for r in results if r[0] == socket.AF_INET]
    return ipv4 if ipv4 else results


socket.getaddrinfo = _ipv4_first_gai

st.set_page_config(
    page_title="Nexus AI — Document & Web Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom Modern Design & CSS Styling ----------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Header & Brand glow */
.brand-title {
    font-size: 1.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    padding-bottom: 2px;
}

.brand-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #c7d2fe;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 3px 10px;
    border-radius: 9999px;
    margin-top: 4px;
}

/* Status Widget */
.status-card {
    background: linear-gradient(180deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.7) 100%);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 12px;
    padding: 12px 14px;
    margin: 12px 0;
}

.status-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.78rem;
    color: #94a3b8;
    margin: 4px 0;
}

.status-value {
    color: #f1f5f9;
    font-weight: 600;
}

/* Metric KPI Cards in Evaluation Dashboard */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 14px;
    padding: 16px 20px;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.25);
}

[data-testid="stMetricLabel"] {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #f8fafc 0%, #cbd5e1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Buttons Styling */
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.88rem;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    border: 1px solid rgba(148, 163, 184, 0.2);
}

.stButton > button:hover {
    transform: translateY(-1px);
    border-color: #6366f1;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.25);
}

/* Chat Messages */
[data-testid="stChatMessage"] {
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 14px;
    border: 1px solid rgba(148, 163, 184, 0.12);
    background: rgba(30, 41, 59, 0.4);
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
}

/* Hero Section */
.hero-container {
    text-align: center;
    padding: 40px 20px 24px 20px;
    max-width: 800px;
    margin: 0 auto;
}

.hero-title {
    font-size: 2.3rem;
    font-weight: 800;
    line-height: 1.2;
    background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 12px;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    margin-bottom: 28px;
    line-height: 1.5;
}

.hero-badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.18) 0%, rgba(56, 189, 248, 0.18) 100%);
    border: 1px solid rgba(99, 102, 241, 0.35);
    color: #c7d2fe;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 9999px;
    margin-bottom: 16px;
}

/* Prompt Card Containers */
.prompt-card {
    background: linear-gradient(145deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 8px;
    transition: all 0.2s ease;
}

.prompt-card-title {
    font-weight: 700;
    font-size: 0.9rem;
    color: #e2e8f0;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.prompt-card-desc {
    font-size: 0.8rem;
    color: #94a3b8;
    line-height: 1.4;
}

/* File badge item */
.file-item {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 10px;
    padding: 8px 12px;
    margin: 6px 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* Clean tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent;
    border-bottom: 1px solid rgba(148, 163, 184, 0.15);
    padding-bottom: 6px;
}

.stTabs [data-baseweb="tab"] {
    height: 42px;
    border-radius: 8px;
    padding: 0 18px;
    font-weight: 600;
    font-size: 0.92rem;
    color: #94a3b8;
    background-color: transparent;
}

.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    background: rgba(99, 102, 241, 0.2) !important;
    border: 1px solid rgba(99, 102, 241, 0.4) !important;
}

/* Hide Streamlit default chrome if desired */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_store() -> VectorStore:
    return VectorStore()


store = get_store()

# ---------- Session State Management ----------
if "sessions" not in st.session_state:
    st.session_state.sessions = {
        "default": {"title": "General Chat", "messages": []}
    }
    st.session_state.current_session_id = "default"

if st.session_state.current_session_id not in st.session_state.sessions:
    st.session_state.current_session_id = next(iter(st.session_state.sessions))

current_session = st.session_state.sessions[st.session_state.current_session_id]

# ---------- Sidebar: Navigation, Sessions & Files ----------
with st.sidebar:
    st.markdown(
        """
        <div style="margin-bottom: 14px;">
            <div class="brand-title">⚡ Nexus RAG</div>
            <div class="brand-badge">● HYBRID SEARCH & AGENT</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # System Status Card
    doc_count = store.count()
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-row">
                <span>AI Engine</span>
                <span class="status-value">Gemini Flash Lite</span>
            </div>
            <div class="status-row">
                <span>Retrieval Mode</span>
                <span class="status-value">Hybrid BM25 + Vector</span>
            </div>
            <div class="status-row">
                <span>Knowledge Chunks</span>
                <span class="status-value">{doc_count} chunks indexed</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Multi-Chat Conversation Manager
    st.markdown("##### 💬 Conversations")
    col_new, col_del = st.columns([3, 1])
    if col_new.button("➕ New Chat", use_container_width=True):
        new_id = uuid.uuid4().hex[:8]
        st.session_state.sessions[new_id] = {"title": "New Chat", "messages": []}
        st.session_state.current_session_id = new_id
        st.rerun()

    if len(st.session_state.sessions) > 1 and col_del.button("🗑️", help="Delete active chat", use_container_width=True):
        del st.session_state.sessions[st.session_state.current_session_id]
        st.session_state.current_session_id = next(iter(st.session_state.sessions))
        st.rerun()

    session_ids = list(st.session_state.sessions.keys())
    active_index = session_ids.index(st.session_state.current_session_id)

    selected_id = st.selectbox(
        "Switch Chat",
        options=session_ids,
        index=active_index,
        format_func=lambda sid: st.session_state.sessions[sid]["title"],
        label_visibility="collapsed",
    )
    if selected_id != st.session_state.current_session_id:
        st.session_state.current_session_id = selected_id
        st.rerun()

    st.divider()

    # Knowledge Base Ingestion
    st.markdown("##### 📁 Document Ingestion")
    files = st.file_uploader(
        "Upload PDF, Word, Excel, CSV, JSON, TXT",
        type=["pdf", "docx", "xlsx", "csv", "json", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if st.button("⚡ Index Documents", disabled=not files, use_container_width=True):
        for f in (files or []):
            with st.spinner(f"Indexing {f.name}…"):
                n = store.add_document(f.name, load_bytes(f.name, f.getvalue()))
            st.success(f"✓ {f.name}: {n} chunks")

    # Indexed Sources & Quick Summarizer
    sources = store.sources()
    if sources:
        st.markdown(f"**Indexed Files ({len(sources)}):**")
        for src in sources:
            c1, c2, c3 = st.columns([3, 1, 1])
            icon = "📄"
            if src.endswith((".xlsx", ".csv")):
                icon = "📊"
            elif src.endswith(".docx"):
                icon = "📝"
            elif src.endswith(".json"):
                icon = "⚙️"

            c1.caption(f"{icon} {src}")
            if c2.button("📝", key=f"sum-{src}", help=f"Quick summary of {src}"):
                st.session_state.pending_prompt = (
                    f"Please give me a concise executive summary and key takeaways of the document: {src}"
                )
                st.rerun()
            if c3.button("✕", key=f"del-{src}", help=f"Delete {src} from index"):
                store.delete_source(src)
                st.rerun()

    st.divider()
    k = st.slider("Chunks to retrieve (k)", 1, 10, 4)

    # Actions: Clear & Export Active Chat
    c_clear, c_export = st.columns(2)
    if c_clear.button("Clear Chat", use_container_width=True):
        current_session["messages"] = []
        st.rerun()

    if current_session["messages"]:
        transcript_lines = []
        for m in current_session["messages"]:
            transcript_lines.append(f"**{m['role'].upper()}:**\n{m['content']}\n")
        c_export.download_button(
            "📥 Export",
            data="\n".join(transcript_lines),
            file_name=f"{current_session['title'].replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

# ---------- Top Tabs: Chat, Document Inspector & Evaluation Dashboard ----------
tab_chat, tab_docs, tab_eval = st.tabs(
    ["💬 Intelligent Assistant", "📄 Document Inspector & Viewer", "📊 Benchmark Dashboard"]
)


def show_details(msg: dict):
    if msg.get("tool_calls"):
        with st.expander("🛠️ Agent Execution Trace"):
            for t in msg["tool_calls"]:
                tool_name = t.get("tool", "tool")
                tool_inp = t.get("input", "")
                st.caption(f"⚡ **Tool invoked**: `{tool_name}`")
                st.code(f"{tool_name}({tool_inp})", language="python")

    if msg.get("sources"):
        with st.expander(f"📎 Retrieved Passages ({len(msg['sources'])})"):
            for i, s in enumerate(msg["sources"], 1):
                src_name = s.get("source", "unknown")
                pg = s.get("page", 1)
                score = s.get("score")
                score_badge = f" · Match score: `{int(score * 100)}%`" if score else ""
                st.markdown(f"**[{i}] 📄 {src_name} — Page {pg}**{score_badge}")
                st.info(s.get("text", ""))


# ==========================================
# TAB 1: CHAT ASSISTANT
# ==========================================
with tab_chat:
    # If conversation is empty, display high-impact welcome hero & prompt suggestion chips
    if not current_session["messages"]:
        st.markdown(
            """
            <div class="hero-container">
                <div class="hero-badge-pill">✨ Next-Generation RAG & Agent Intelligence</div>
                <h1 class="hero-title">Ask anything across your documents, code & the live web</h1>
                <p class="hero-subtitle">
                    Equipped with Reciprocal Rank Fusion Hybrid Search, Web Navigation, Python REPL, and Real-Time Streaming.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<p style='text-align:center; color:#94a3b8; font-size:0.85rem; font-weight:600; margin-bottom:12px;'>⚡ CLICK A SUGGESTION TO GET STARTED:</p>", unsafe_allow_html=True)
        col_s1, col_s2 = st.columns(2)

        with col_s1:
            st.markdown(
                """
                <div class="prompt-card">
                    <div class="prompt-card-title">📄 Document Deep Dive</div>
                    <div class="prompt-card-desc">Query knowledge, policies, and contracts from your uploaded files with citations.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("🔍 'Summarize the key points in my uploaded files'", use_container_width=True):
                st.session_state.pending_prompt = "Summarize the key points and findings in my uploaded files."
                st.rerun()

            st.markdown(
                """
                <div class="prompt-card" style="margin-top: 12px;">
                    <div class="prompt-card-title">🧮 Python Calculator & Math</div>
                    <div class="prompt-card-desc">Execute exact code, compound formulas, statistics, and aggregations.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("⚡ 'Calculate compound interest on $10,000 at 7% for 5 years'", use_container_width=True):
                st.session_state.pending_prompt = "Calculate compound interest on $10,000 at 7% for 5 years using Python and show the breakdown."
                st.rerun()

        with col_s2:
            st.markdown(
                """
                <div class="prompt-card">
                    <div class="prompt-card-title">🌐 Live Web Search</div>
                    <div class="prompt-card-desc">Search the live internet for recent world events, technical specs, and news.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("🌍 'What are the newest breakthrough AI models announced recently?'", use_container_width=True):
                st.session_state.pending_prompt = "What are the newest breakthrough AI models announced recently?"
                st.rerun()

            st.markdown(
                """
                <div class="prompt-card" style="margin-top: 12px;">
                    <div class="prompt-card-title">💡 General Brainstorming & Coding</div>
                    <div class="prompt-card-desc">Design architectures, write clean code, or get creative problem solutions.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("💡 'Give me 3 innovative full-stack project ideas with AI'", use_container_width=True):
                st.session_state.pending_prompt = "Give me 3 innovative full-stack project ideas combining AI and modern web frameworks."
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

    # Render conversation history
    for m in current_session["messages"]:
        avatar = "👤" if m["role"] == "user" else "⚡"
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                show_details(m)

    # Voice Input Recorder
    with st.expander("🎙️ Voice Input (Record question with microphone)", expanded=False):
        col_rec, col_info = st.columns([3, 2])
        with col_rec:
            recorded_audio = st.audio_input("Record audio", label_visibility="collapsed")
        with col_info:
            st.caption("Click to record your voice question. Gemini will transcribe it into text automatically.")
        if recorded_audio:
            with st.spinner("Transcribing your voice query with Gemini…"):
                transcribed_text = transcribe_audio(
                    recorded_audio.getvalue(), mime_type=recorded_audio.type or "audio/wav"
                )
            if transcribed_text and not transcribed_text.startswith("[Audio transcription error"):
                st.session_state.pending_prompt = transcribed_text
                st.rerun()
            elif transcribed_text:
                st.warning(transcribed_text)

    # Always render chat input so it stays permanently mounted
    chat_val = st.chat_input("Ask a question, paste a link to read, or query your documents…")
    prompt = chat_val
    if "pending_prompt" in st.session_state and st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        del st.session_state.pending_prompt

    if prompt:
        # Update chat title if still default
        if current_session["title"] in {"New Chat", "General Chat"}:
            clean_title = prompt.strip()[:24]
            current_session["title"] = clean_title + ("…" if len(prompt) > 24 else "")

        current_session["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        history = [
            {"role": m["role"], "content": m["content"]}
            for m in current_session["messages"][:-1]
        ][-8:]

        with st.chat_message("assistant", avatar="⚡"):
            agent = Agent(store, k=k)
            response_placeholder = st.empty()
            full_response = ""

            try:
                with st.spinner("Analyzing & selecting optimal tools…"):
                    stream = agent.run_stream(prompt, history)
                    for chunk in stream:
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                        break

                for chunk in stream:
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
            except Exception as e:
                full_response = f"⚠️ An error occurred while generating the response: {e}"

            if not full_response.strip():
                full_response = "I couldn't find an answer for that query."

            response_placeholder.markdown(full_response)
            answer = full_response

            msg = {
                "role": "assistant",
                "content": answer,
                "sources": agent.last_sources,
                "tool_calls": agent.last_tool_calls,
            }
            show_details(msg)

        current_session["messages"].append(msg)


# ==========================================
# TAB 2: DOCUMENT INSPECTOR & VIEWER
# ==========================================
with tab_docs:
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h2 style="margin: 0; font-weight: 800;">📄 Document Inspector & Knowledge Base Viewer</h2>
            <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 4px;">
                Inspect indexed documents, explore individual passages and page numbers, or search within any file.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_sources = store.sources()
    if not all_sources:
        st.info("No documents indexed yet. Upload documents using the sidebar to inspect them here.")
    else:
        col_sel, col_action1, col_action2 = st.columns([3, 1, 1])
        with col_sel:
            selected_doc = st.selectbox("Select document to inspect:", options=all_sources)
        with col_action1:
            st.write("")
            st.write("")
            if selected_doc and st.button("📝 Quick Summary", key="view_summary_btn", use_container_width=True):
                st.session_state.pending_prompt = (
                    f"Please give me a concise executive summary and key takeaways of the document: {selected_doc}"
                )
                st.rerun()
        with col_action2:
            st.write("")
            st.write("")
            if selected_doc and st.button("🗑️ Delete File", key="view_delete_btn", use_container_width=True):
                store.delete_source(selected_doc)
                st.rerun()

        if selected_doc:
            chunks = store.get_document_chunks(selected_doc)
            pages = sorted({c["page"] for c in chunks})

            # Metrics
            d1, d2, d3 = st.columns(3)
            d1.metric("Total Indexed Chunks", len(chunks))
            d2.metric("Total Pages / Sections", len(pages))
            doc_ext = Path(selected_doc).suffix.upper() or "FILE"
            d3.metric("Document Type", doc_ext)

            st.divider()

            # Search inside document
            doc_search = st.text_input(
                "🔍 Filter chunks in this document by keyword:",
                placeholder="Type to filter passages…",
            )
            filtered_chunks = [
                c for c in chunks if not doc_search or doc_search.lower() in c["text"].lower()
            ]

            st.caption(f"Showing **{len(filtered_chunks)}** of **{len(chunks)}** chunks")

            for i, chunk in enumerate(filtered_chunks, 1):
                with st.expander(f"Chunk #{i} · Page {chunk['page']}", expanded=(i <= 3)):
                    st.markdown(f"**Source**: `{selected_doc}` · **Page**: `{chunk['page']}`")
                    st.text_area(
                        "Passage Text",
                        value=chunk["text"],
                        height=120,
                        disabled=True,
                        key=f"chunk_txt_{selected_doc}_{i}",
                    )


# ==========================================
# TAB 3: BENCHMARK & EVALUATION DASHBOARD
# ==========================================
with tab_eval:
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h2 style="margin: 0; font-weight: 800;">📊 Benchmark & Observability Dashboard</h2>
            <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 4px;">
                Continuous automated evaluation metrics across tool routing, retrieval accuracy, and factual faithfulness.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    results_file = Path("evaluation/results.json")
    if results_file.exists():
        try:
            data = json.loads(results_file.read_text(encoding="utf-8"))
            summary = data.get("summary", {})
            rows = data.get("rows", [])

            # High-Level Metric KPI Cards
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Benchmark Cases", summary.get("n", 0))
            tool_acc = summary.get("tool_accuracy", 0.0)
            m2.metric("Tool Selection Accuracy", f"{int(tool_acc * 100)}%")
            retrieval_rate = summary.get("retrieval_hit_rate", 0.0)
            m3.metric("Retrieval Hit Rate", f"{int(retrieval_rate * 100)}%")
            faith = summary.get("faithfulness_avg_of_5", 0.0)
            m4.metric("Faithfulness Score", f"{faith} / 5.0")

            st.divider()
            st.markdown("### 📋 Test Case Evaluations")

            for idx, r in enumerate(rows, 1):
                is_passed = r.get("tool_ok")
                status_icon = "🟢 PASSED" if is_passed else "🔴 FAILED"
                with st.expander(f"{status_icon} | Test #{idx}: {r.get('question')}"):
                    c_a, c_b = st.columns(2)
                    c_a.write(f"**Tools Triggered:** `{', '.join(r.get('tools', []))}`")
                    c_b.write(f"**Faithfulness Rating:** `{r.get('faithful', 'N/A')}/5.0`")
                    st.markdown(f"**Agent Response:**\n> {r.get('answer')}")
                    if r.get("judge_reason"):
                        st.info(f"**Judge Assessment:** {r['judge_reason']}")
        except Exception as err:
            st.error(f"Error loading evaluation results: {err}")
    else:
        st.info("No evaluation results found yet. Run `python -m evaluation.run_eval` to generate benchmark data.")
