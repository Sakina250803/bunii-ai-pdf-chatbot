"""
Kimetsu no Bunseki (鬼滅の分析) — "Slaying Analysis"
An AI PDF Chatbot themed around Slayer / Kimetsu no Yaiba.

Run locally:
    streamlit run app.py

Requires a Gemini API key (get one free at https://aistudio.google.com/apikey)
"""
import os
import time
import traceback

import streamlit as st
from dotenv import load_dotenv

from rag_engine import (
    GeminiRAGClient,
    build_document_index,
    answer_question,
    PDFExtractionError,
    EmptyDocumentError,
    EmbeddingError,
    GenerationError,
)

load_dotenv()

# --------------------------------------------------------------------------
# Page config + theme
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Bunii AI PDF Chatbot",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()

BREATHING_STYLES = [
    "Water Breathing", "Flame Breathing", "Thunder Breathing",
    "Wind Breathing", "Stone Breathing", "Insect Breathing",
    "Sound Breathing", "Love Breathing", "Mist Breathing",
]

# --------------------------------------------------------------------------
# Session state initialization
# --------------------------------------------------------------------------

def init_state():
    defaults = {
        "vectorstore": None,
        "client": None,
        "chunks": None,
        "full_text": None,
        "doc_name": None,
        "chat_history": [],
        "api_key": os.getenv("GOOGLE_API_KEY", ""),
        "processing": False,
        "num_chunks": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_state()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown("###  bunii.ai PDF Chatbot")
    st.markdown("Configure your API Key to begin.")

    api_key_input = st.text_input(
        "Gemini API Key",
        value=st.session_state.api_key,
        type="password",
        help="Get a free key at https://aistudio.google.com/apikey",
    )
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
        st.session_state.client = None  # force re-init on change

    st.markdown("---")
    st.markdown("###  Settings")
    chunk_size = st.slider("Chunk size", 500, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", 50, 400, 200, step=50)
    top_k = st.slider("Chunks to retrieve (k)", 2, 8, 4)

    st.markdown("---")
    if st.session_state.doc_name:
        st.success(f" Scroll loaded: **{st.session_state.doc_name}**")
        st.caption(f"{st.session_state.num_chunks} fragments indexed")
        if st.button(" Discard scroll & start over"):
            for key in ["vectorstore", "chunks", "full_text", "doc_name", "chat_history", "num_chunks"]:
                st.session_state[key] = None if key != "chat_history" else []
            st.session_state.num_chunks = 0
            st.rerun()
    else:
        st.info("No document loaded yet.")

    st.markdown("---")
    st.caption(
        "This app is a demo of the bunii.ai PDF Chatbot, built with Streamlit and the Gemini API. "
    )

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.markdown(
    """
    <div class="ds-banner">
        <h1>bunii.ai PDF Chatbot</h1>
        <div class="subtitle">The Slaying PDF Chatbot &nbsp;|&nbsp; Total Concentration: Document Breathing</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Guard: need an API key before anything else works
# --------------------------------------------------------------------------

if not st.session_state.api_key:
    st.warning(
        "🗡️ **A api key is required to slay this document.**\n\n"
        "Enter your free Gemini API key in the sidebar to begin. "
        "Get one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)."
    )
    st.stop()

# Lazily construct the client, with error handling
if st.session_state.client is None:
    try:
        st.session_state.client = GeminiRAGClient(api_key=st.session_state.api_key)
    except Exception as e:
        st.error(f"Could not initialize Gemini client: {e}")
        st.stop()

client = st.session_state.client

# --------------------------------------------------------------------------
# Upload + ingest
# --------------------------------------------------------------------------

tab_chat, tab_summary, tab_keypoints = st.tabs(
    [" Interrogate the Scroll", " Full Summary", " Key Points"]
)

uploaded_file = st.file_uploader(
    "Upload a PDF to begin your slaying analysis",
    type="pdf",
    disabled=st.session_state.processing,
)

if uploaded_file is not None and uploaded_file.name != st.session_state.doc_name:
    st.session_state.processing = True
    progress = st.progress(0, text="Sharpening knowledge...")

    try:
        file_bytes = uploaded_file.read()
        if len(file_bytes) == 0:
            raise PDFExtractionError("The uploaded file is empty.")
        if len(file_bytes) > 25 * 1024 * 1024:
            raise PDFExtractionError("File too large (max 25MB). Try a smaller PDF.")

        progress.progress(20, text="Extracting text from pages...")

        store, chunks, full_text = build_document_index(
            file_bytes,
            client,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            source=uploaded_file.name,
        )

        progress.progress(90, text="Finalizing vector index...")

        st.session_state.vectorstore = store
        st.session_state.chunks = chunks
        st.session_state.full_text = full_text
        st.session_state.doc_name = uploaded_file.name
        st.session_state.num_chunks = len(chunks)
        st.session_state.chat_history = []

        progress.progress(100, text="Ready!")
        time.sleep(0.3)
        progress.empty()
        st.success(f"**{uploaded_file.name}** has been slain into {len(chunks)} fragments. Ask away!")

    except PDFExtractionError as e:
        progress.empty()
        st.error(f" **PDF problem:** {e}")
    except EmptyDocumentError as e:
        progress.empty()
        st.error(f" **No readable text:** {e}")
    except EmbeddingError as e:
        progress.empty()
        st.error(f" **Embedding failed** (check your API key / quota): {e}")
    except Exception as e:
        progress.empty()
        st.error(f" **Unexpected error:** {e}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())
    finally:
        st.session_state.processing = False

# --------------------------------------------------------------------------
# Tab 1: Chat
# --------------------------------------------------------------------------

with tab_chat:
    if st.session_state.vectorstore is None:
        st.info(" Upload a PDF to start your slaying analysis.")
    else:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"], avatar="🗡️" if msg["role"] == "user" else "🗡️"):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander(f" Sources ({len(msg['sources'])})"):
                        for rc in msg["sources"]:
                            st.markdown(
                                f'<span class="page-badge">Page {rc.chunk.page}</span> '
                                f'relevance: {rc.score:.2f}',
                                unsafe_allow_html=True,
                            )
                            st.caption(rc.chunk.text[:300] + ("..." if len(rc.chunk.text) > 300 else ""))

        question = st.chat_input("Ask your question,  friend...")

        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user", avatar="🗡️"):
                st.markdown(question)

            with st.chat_message("assistant", avatar="🗡️"):
                with st.spinner("Using Total Concentration Breathing..."):
                    result = answer_question(
                        question,
                        st.session_state.vectorstore,
                        client,
                        k=top_k,
                        chat_history=st.session_state.chat_history[:-1],
                    )

                if result.error:
                    st.error(f" {result.error}")
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": f" Error: {result.error}"}
                    )
                else:
                    st.markdown(result.answer)
                    if result.sources:
                        with st.expander(f" Sources ({len(result.sources)})"):
                            for rc in result.sources:
                                st.markdown(
                                    f'<span class="page-badge">Page {rc.chunk.page}</span> '
                                    f'relevance: {rc.score:.2f}',
                                    unsafe_allow_html=True,
                                )
                                st.caption(rc.chunk.text[:300] + ("..." if len(rc.chunk.text) > 300 else ""))
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": result.answer, "sources": result.sources}
                    )

# --------------------------------------------------------------------------
# Tab 2: Summary
# --------------------------------------------------------------------------

with tab_summary:
    if st.session_state.vectorstore is None:
        st.info(" Upload a PDF above first.")
    else:
        if st.button(" Generate Full Summary", key="gen_summary"):
            with st.spinner("Performing Water Breathing: Full Document Style..."):
                try:
                    summary = client.summarize(st.session_state.full_text, mode="summary")
                    st.session_state["_summary_cache"] = summary
                except GenerationError as e:
                    st.error(f" Summary generation failed: {e}")

        if st.session_state.get("_summary_cache"):
            st.markdown(st.session_state["_summary_cache"])

# --------------------------------------------------------------------------
# Tab 3: Key Points
# --------------------------------------------------------------------------

with tab_keypoints:
    if st.session_state.vectorstore is None:
        st.info(" Upload a PDF above first.")
    else:
        if st.button(" Extract Key Points", key="gen_keypoints"):
            with st.spinner("Performing Thunder Breathing: First Form..."):
                try:
                    points = client.summarize(st.session_state.full_text, mode="key_points")
                    st.session_state["_keypoints_cache"] = points
                except GenerationError as e:
                    st.error(f" Key point extraction failed: {e}")

        if st.session_state.get("_keypoints_cache"):
            st.markdown(st.session_state["_keypoints_cache"])
