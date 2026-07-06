"""
Unit tests for rag_engine.py.
Gemini API calls are mocked so these run offline, deterministically, in CI.
"""
import io
import sys
import os
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_engine import (
    Chunk,
    RetrievedChunk,
    VectorStore,
    GeminiRAGClient,
    extract_pages,
    clean_text,
    chunk_text,
    build_document_index,
    answer_question,
    PDFExtractionError,
    EmptyDocumentError,
    EmbeddingError,
    GenerationError,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def make_pdf_bytes(pages_text):
    """Build a real minimal PDF with given page texts using pypdf/reportlab-free approach."""
    from pypdf import PdfWriter
    import subprocess

    # Use fpdf-free simple approach: build via pypdf's blank pages + text is hard
    # without a rendering library, so instead we use reportlab if available,
    # else fall back to a fixed pre-built single-page PDF fixture.
    try:
        from reportlab.pdfgen import canvas
        buf = io.BytesIO()
        c = canvas.Canvas(buf)
        for text in pages_text:
            c.drawString(72, 720, text)
            c.showPage()
        c.save()
        return buf.getvalue()
    except ImportError:
        return None


# --------------------------------------------------------------------------
# clean_text
# --------------------------------------------------------------------------

def test_clean_text_collapses_whitespace():
    assert clean_text("hello   \n\n  world\t\t!") == "hello world !"


def test_clean_text_strips_edges():
    assert clean_text("   padded text   ") == "padded text"


def test_clean_text_empty():
    assert clean_text("") == ""
    assert clean_text("   ") == ""


# --------------------------------------------------------------------------
# chunk_text
# --------------------------------------------------------------------------

def test_chunk_text_basic_splitting():
    pages = [(1, "word " * 500)]  # 2500 chars
    chunks = chunk_text(pages, chunk_size=1000, chunk_overlap=200)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 1000
        assert c.page == 1


def test_chunk_text_preserves_page_numbers():
    pages = [(1, "short text page one"), (2, "short text page two")]
    chunks = chunk_text(pages, chunk_size=1000, chunk_overlap=200)
    pages_seen = {c.page for c in chunks}
    assert pages_seen == {1, 2}


def test_chunk_text_empty_pages_raises():
    pages = [(1, ""), (2, "   ")]
    with pytest.raises(EmptyDocumentError):
        chunk_text(pages)


def test_chunk_text_no_pages_raises():
    with pytest.raises(EmptyDocumentError):
        chunk_text([])


def test_chunk_text_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text([(1, "some text")], chunk_size=100, chunk_overlap=200)


def test_chunk_text_unique_ids_increasing():
    pages = [(1, "word " * 1000)]
    chunks = chunk_text(pages, chunk_size=500, chunk_overlap=50)
    ids = [c.chunk_id for c in chunks]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)


def test_chunk_text_progress_guaranteed_no_infinite_loop():
    # pathological case: chunk_overlap close to chunk_size shouldn't hang
    pages = [(1, "x" * 5000)]
    chunks = chunk_text(pages, chunk_size=100, chunk_overlap=99)
    assert len(chunks) > 0  # completes without hanging


# --------------------------------------------------------------------------
# extract_pages
# --------------------------------------------------------------------------

def test_extract_pages_invalid_bytes_raises():
    with pytest.raises(PDFExtractionError):
        extract_pages(b"not a real pdf")


def test_extract_pages_with_real_pdf():
    pdf_bytes = make_pdf_bytes(["Hello World Page One", "Second Page Content"])
    if pdf_bytes is None:
        pytest.skip("reportlab not installed, skipping real-PDF test")
    pages = extract_pages(pdf_bytes)
    assert len(pages) == 2
    assert "Hello World" in pages[0][1]


def test_extract_pages_password_protected_pdf_raises():
    """Regression test: a real (non-empty) password must be rejected,
    not silently treated as readable."""
    try:
        from reportlab.pdfgen import canvas
        from pypdf import PdfWriter, PdfReader
    except ImportError:
        pytest.skip("reportlab not installed")

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 720, "secret content")
    c.showPage()
    c.save()

    writer = PdfWriter()
    reader = PdfReader(io.BytesIO(buf.getvalue()))
    for page in reader.pages:
        writer.add_page(page)
    writer.encrypt("realpassword123")
    enc_buf = io.BytesIO()
    writer.write(enc_buf)

    with pytest.raises(PDFExtractionError):
        extract_pages(enc_buf.getvalue())


def test_extract_pages_empty_password_encrypted_pdf_still_readable():
    """A PDF encrypted with an empty password should still be extractable."""
    try:
        from reportlab.pdfgen import canvas
        from pypdf import PdfWriter, PdfReader
    except ImportError:
        pytest.skip("reportlab not installed")

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 720, "open content")
    c.showPage()
    c.save()

    writer = PdfWriter()
    reader = PdfReader(io.BytesIO(buf.getvalue()))
    for page in reader.pages:
        writer.add_page(page)
    writer.encrypt("")
    enc_buf = io.BytesIO()
    writer.write(enc_buf)

    pages = extract_pages(enc_buf.getvalue())
    assert len(pages) == 1
    assert "open content" in pages[0][1]


# --------------------------------------------------------------------------
# VectorStore
# --------------------------------------------------------------------------

def test_vectorstore_add_and_search():
    store = VectorStore(dim=4)
    chunks = [
        Chunk(text="apple fruit", page=1, chunk_id=0),
        Chunk(text="banana fruit", page=1, chunk_id=1),
        Chunk(text="car vehicle", page=2, chunk_id=2),
    ]
    vectors = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    store.add(vectors, chunks)
    assert len(store) == 3

    query = np.array([1.0, 0.0, 0.0, 0.0])
    results = store.search(query, k=2)
    assert len(results) == 2
    # closest match should be "apple fruit" (identical vector)
    assert results[0].chunk.text == "apple fruit"
    assert results[0].score > results[1].score


def test_vectorstore_search_empty_returns_empty():
    store = VectorStore(dim=4)
    results = store.search(np.array([1.0, 0.0, 0.0, 0.0]), k=3)
    assert results == []


def test_vectorstore_mismatched_lengths_raises():
    store = VectorStore(dim=4)
    vectors = np.zeros((2, 4))
    chunks = [Chunk(text="a", page=1, chunk_id=0)]
    with pytest.raises(ValueError):
        store.add(vectors, chunks)


def test_vectorstore_k_larger_than_index_size():
    store = VectorStore(dim=4)
    chunks = [Chunk(text="only one", page=1, chunk_id=0)]
    store.add(np.array([[1.0, 0.0, 0.0, 0.0]]), chunks)
    results = store.search(np.array([1.0, 0.0, 0.0, 0.0]), k=10)
    assert len(results) == 1  # doesn't crash, caps at actual size


def test_vectorstore_zero_vector_no_divide_by_zero():
    store = VectorStore(dim=4)
    chunks = [Chunk(text="zero vec", page=1, chunk_id=0)]
    store.add(np.array([[0.0, 0.0, 0.0, 0.0]]), chunks)
    results = store.search(np.array([0.0, 0.0, 0.0, 0.0]), k=1)
    assert len(results) == 1  # doesn't raise/NaN-crash


# --------------------------------------------------------------------------
# GeminiRAGClient - mocked
# --------------------------------------------------------------------------

def test_client_requires_api_key():
    with pytest.raises(ValueError):
        GeminiRAGClient(api_key="")
    with pytest.raises(ValueError):
        GeminiRAGClient(api_key="   ")


@patch("google.genai.Client")
def test_embed_texts_success(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_embedding = MagicMock()
    mock_embedding.values = [0.1] * 768
    mock_result = MagicMock()
    mock_result.embeddings = [mock_embedding, mock_embedding]
    mock_client.models.embed_content.return_value = mock_result

    client = GeminiRAGClient(api_key="fake-key")
    vectors = client.embed_texts(["text one", "text two"])
    assert vectors.shape == (2, 768)


@patch("google.genai.Client")
def test_embed_texts_empty_list(mock_client_cls):
    client = GeminiRAGClient(api_key="fake-key")
    vectors = client.embed_texts([])
    assert vectors.shape[0] == 0


@patch("google.genai.Client")
def test_embed_texts_api_failure_raises_embedding_error(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.models.embed_content.side_effect = Exception("network error")

    client = GeminiRAGClient(api_key="fake-key")
    with pytest.raises(EmbeddingError):
        client.embed_texts(["some text"])


@patch("google.genai.Client")
def test_embed_texts_requests_matching_output_dimensionality(mock_client_cls):
    """Regression test: real bug where Gemini defaulted to 3072-dim embeddings
    while FAISS index was built expecting 768-dim, causing an AssertionError
    deep inside faiss. Fix: explicitly request output_dimensionality=EMBED_DIM."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    captured = {}

    def fake_embed(model, contents, config):
        captured["config"] = config
        emb_list = []
        for _ in contents:
            emb = MagicMock()
            emb.values = [0.1] * 768
            emb_list.append(emb)
        result = MagicMock()
        result.embeddings = emb_list
        return result

    mock_client.models.embed_content.side_effect = fake_embed

    client = GeminiRAGClient(api_key="fake-key")
    vectors = client.embed_texts(["some text"])

    assert captured["config"].output_dimensionality == client.EMBED_DIM
    assert vectors.shape[1] == client.EMBED_DIM


@patch("google.genai.Client")
def test_embed_texts_dimension_mismatch_raises_clear_error(mock_client_cls):
    """If the API ever returns a different dimension than requested/expected,
    fail with a clear EmbeddingError instead of crashing later inside FAISS."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    def fake_embed_wrong_size(model, contents, config):
        emb_list = []
        for _ in contents:
            emb = MagicMock()
            emb.values = [0.1] * 3072  # wrong size
            emb_list.append(emb)
        result = MagicMock()
        result.embeddings = emb_list
        return result

    mock_client.models.embed_content.side_effect = fake_embed_wrong_size

    client = GeminiRAGClient(api_key="fake-key")
    with pytest.raises(EmbeddingError, match="dim embeddings"):
        client.embed_texts(["some text"])


@patch("google.genai.Client")
def test_generate_answer_no_context_returns_fallback(mock_client_cls):
    client = GeminiRAGClient(api_key="fake-key")
    result = client.generate_answer("what is this?", context_chunks=[])
    assert "couldn't find" in result.lower()


@patch("google.genai.Client")
def test_generate_answer_success(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_response = MagicMock()
    mock_response.text = "This is the answer (p. 1)."
    mock_client.models.generate_content.return_value = mock_response

    client = GeminiRAGClient(api_key="fake-key")
    chunk = Chunk(text="relevant content", page=1, chunk_id=0)
    result = client.generate_answer("question?", [RetrievedChunk(chunk=chunk, score=0.9)])
    assert result == "This is the answer (p. 1)."


@patch("google.genai.Client")
def test_generate_answer_empty_response_raises(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_response = MagicMock()
    mock_response.text = None
    mock_client.models.generate_content.return_value = mock_response

    client = GeminiRAGClient(api_key="fake-key")
    chunk = Chunk(text="content", page=1, chunk_id=0)
    with pytest.raises(GenerationError):
        client.generate_answer("question?", [RetrievedChunk(chunk=chunk, score=0.9)])


@patch("google.genai.Client")
def test_generate_answer_api_failure_raises(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.models.generate_content.side_effect = Exception("timeout")

    client = GeminiRAGClient(api_key="fake-key")
    chunk = Chunk(text="content", page=1, chunk_id=0)
    with pytest.raises(GenerationError):
        client.generate_answer("question?", [RetrievedChunk(chunk=chunk, score=0.9)])


@patch("google.genai.Client")
def test_summarize_invalid_mode_raises(mock_client_cls):
    client = GeminiRAGClient(api_key="fake-key")
    with pytest.raises(ValueError):
        client.summarize("some text", mode="not_a_real_mode")


@patch("google.genai.Client")
def test_summarize_success(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_response = MagicMock()
    mock_response.text = "Summary text."
    mock_client.models.generate_content.return_value = mock_response

    client = GeminiRAGClient(api_key="fake-key")
    result = client.summarize("long document text " * 100, mode="summary")
    assert result == "Summary text."


# --------------------------------------------------------------------------
# answer_question orchestration
# --------------------------------------------------------------------------

@patch("google.genai.Client")
def test_answer_question_empty_input(mock_client_cls):
    client = GeminiRAGClient(api_key="fake-key")
    store = VectorStore(dim=3)
    result = answer_question("   ", store, client)
    assert result.error == "empty_question"


@patch("google.genai.Client")
def test_answer_question_full_flow(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_embedding = MagicMock()
    mock_embedding.values = [1.0, 0.0, 0.0]
    mock_embed_result = MagicMock()
    mock_embed_result.embeddings = [mock_embedding]
    mock_client.models.embed_content.return_value = mock_embed_result

    mock_gen_response = MagicMock()
    mock_gen_response.text = "Final answer."
    mock_client.models.generate_content.return_value = mock_gen_response

    client = GeminiRAGClient(api_key="fake-key")
    client.EMBED_DIM = 3
    store = VectorStore(dim=3)
    chunk = Chunk(text="doc content", page=1, chunk_id=0)
    store.add(np.array([[1.0, 0.0, 0.0]]), [chunk])

    result = answer_question("what does it say?", store, client, k=1)
    assert result.error is None
    assert result.answer == "Final answer."
    assert len(result.sources) == 1


@patch("google.genai.Client")
def test_answer_question_embedding_failure_returns_error_not_exception(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.models.embed_content.side_effect = Exception("api down")

    client = GeminiRAGClient(api_key="fake-key")
    store = VectorStore(dim=3)
    result = answer_question("question", store, client)
    assert result.error is not None
    assert result.answer == ""


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
