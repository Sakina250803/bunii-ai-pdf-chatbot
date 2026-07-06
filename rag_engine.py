"""
Core RAG engine for the PDF chatbot.
Kept separate from the Streamlit UI so it can be unit tested without
spinning up a UI or hitting the network.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import faiss
from pypdf import PdfReader


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------

@dataclass
class Chunk:
    text: str
    page: int
    chunk_id: int
    source: str = "document"


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass
class AnswerResult:
    answer: str
    sources: List[RetrievedChunk] = field(default_factory=list)
    error: Optional[str] = None


# --------------------------------------------------------------------------
# Custom exceptions (so the UI can show friendly messages)
# --------------------------------------------------------------------------

class PDFExtractionError(Exception):
    pass


class EmptyDocumentError(Exception):
    pass


class EmbeddingError(Exception):
    pass


class GenerationError(Exception):
    pass


# --------------------------------------------------------------------------
# PDF loading + chunking
# --------------------------------------------------------------------------

def extract_pages(file_bytes: bytes) -> List[tuple[int, str]]:
    """Extract text per page from PDF bytes. Returns list of (page_num, text)."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as e:
        raise PDFExtractionError(f"Could not read PDF: {e}") from e

    if reader.is_encrypted:
        try:
            result = reader.decrypt("")
            # pypdf returns PasswordType: 0 = not decrypted, 1 = user password, 2 = owner password
            if not result:
                raise PDFExtractionError(
                    "This PDF is password-protected. Please upload an unlocked PDF."
                )
        except PDFExtractionError:
            raise
        except Exception:
            raise PDFExtractionError(
                "This PDF is password-protected. Please upload an unlocked PDF."
            )

    pages = []
    try:
        num_pages = len(reader.pages)
    except Exception:
        raise PDFExtractionError(
            "This PDF is password-protected or corrupted and could not be read. "
            "Please upload an unlocked, valid PDF."
        )

    for i in range(num_pages):
        try:
            text = reader.pages[i].extract_text() or ""
        except Exception:
            text = ""
        pages.append((i + 1, text))

    if not pages:
        raise PDFExtractionError("PDF has no pages.")

    return pages


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def chunk_text(
    pages: List[tuple[int, str]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    source: str = "document",
) -> List[Chunk]:
    """Split page text into overlapping chunks, tracking source page per chunk."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: List[Chunk] = []
    chunk_id = 0

    for page_num, raw_text in pages:
        text = clean_text(raw_text)
        if not text:
            continue

        start = 0
        n = len(text)
        while start < n:
            end = min(start + chunk_size, n)
            piece = text[start:end]

            # try not to cut mid-word at the boundary (skip on final piece)
            if end < n:
                last_space = piece.rfind(" ")
                if last_space > chunk_size * 0.5:
                    piece = piece[:last_space]
                    end = start + last_space

            piece = piece.strip()
            if piece:
                chunks.append(Chunk(text=piece, page=page_num, chunk_id=chunk_id, source=source))
                chunk_id += 1

            if end >= n:
                break
            start = max(end - chunk_overlap, start + 1)  # guarantee progress

    if not chunks:
        raise EmptyDocumentError(
            "No extractable text found in this PDF. It may be a scanned "
            "image PDF without OCR text."
        )

    return chunks


# --------------------------------------------------------------------------
# Embeddings + vector store (Gemini embeddings + FAISS)
# --------------------------------------------------------------------------

class VectorStore:
    """Thin wrapper around a FAISS index mapping chunks <-> vectors."""

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine sim via normalized vectors
        self.chunks: List[Chunk] = []

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return vectors / norms

    def add(self, vectors: np.ndarray, chunks: List[Chunk]):
        if vectors.shape[0] != len(chunks):
            raise ValueError("vectors/chunks length mismatch")
        vectors = self._normalize(vectors.astype("float32"))
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int = 4) -> List[RetrievedChunk]:
        if self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        q = self._normalize(query_vector.astype("float32").reshape(1, -1))
        scores, indices = self.index.search(q, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(RetrievedChunk(chunk=self.chunks[idx], score=float(score)))
        return results

    def __len__(self):
        return len(self.chunks)


# --------------------------------------------------------------------------
# Gemini client wrapper
# --------------------------------------------------------------------------

class GeminiRAGClient:
    """
    Wraps Gemini embedding + generation calls.
    Isolated so it can be mocked/faked in tests without a real API key.
    """

    EMBED_MODEL = "gemini-embedding-001"
    GEN_MODEL = "gemini-2.5-flash"
    EMBED_DIM = 768

    def __init__(self, api_key: str):
        if not api_key or not api_key.strip():
            raise ValueError("A Gemini API key is required.")
        from google import genai  # imported lazily so tests can run without it installed if needed
        self._genai = genai
        self.client = genai.Client(api_key=api_key)

    def embed_texts(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        if not texts:
            return np.zeros((0, self.EMBED_DIM), dtype="float32")
        try:
            from google.genai import types
            result = self.client.models.embed_content(
                model=self.EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.EMBED_DIM,
                ),
            )
            vectors = [e.values for e in result.embeddings]
            arr = np.array(vectors, dtype="float32")
            if arr.shape[1] != self.EMBED_DIM:
                raise EmbeddingError(
                    f"Gemini returned {arr.shape[1]}-dim embeddings but "
                    f"EMBED_DIM is set to {self.EMBED_DIM}. This usually means "
                    f"the embedding model changed its default output size — "
                    f"update EMBED_DIM in rag_engine.py to match."
                )
            return arr
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Embedding request failed: {e}") from e

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text], task_type="RETRIEVAL_QUERY")[0]

    def generate_answer(
        self,
        question: str,
        context_chunks: List[RetrievedChunk],
        chat_history: Optional[List[dict]] = None,
    ) -> str:
        if not context_chunks:
            return (
                "I couldn't find anything relevant to that question in the "
                "document. Try rephrasing, or ask something closer to the "
                "document's actual content."
            )

        context_block = "\n\n".join(
            f"[Page {rc.chunk.page}] {rc.chunk.text}" for rc in context_chunks
        )

        history_block = ""
        if chat_history:
            history_block = "\n".join(
                f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                for m in chat_history[-6:]
            )

        prompt = f"""You are a helpful assistant answering questions about a PDF document.
Answer ONLY using the provided context below. If the answer isn't in the
context, say you don't know based on the document. Cite page numbers
inline like (p. 3) when you use a specific fact.

{f"Recent conversation:{chr(10)}{history_block}{chr(10)}" if history_block else ""}
Context from the document:
{context_block}

Question: {question}

Answer:"""

        try:
            response = self.client.models.generate_content(
                model=self.GEN_MODEL,
                contents=prompt,
            )
            if not response or not getattr(response, "text", None):
                raise GenerationError("Empty response from model.")
            return response.text
        except GenerationError:
            raise
        except Exception as e:
            raise GenerationError(f"Answer generation failed: {e}") from e

    def summarize(self, full_text: str, mode: str = "summary") -> str:
        max_chars = 60000  # keep prompt within safe bounds
        text = full_text[:max_chars]

        if mode == "summary":
            instruction = "Write a clear, well-organized summary of this document in 4-6 paragraphs."
        elif mode == "key_points":
            instruction = "Extract the 8-10 most important key points from this document as a bulleted list."
        else:
            raise ValueError(f"Unknown mode: {mode}")

        prompt = f"{instruction}\n\nDocument:\n{text}"

        try:
            response = self.client.models.generate_content(
                model=self.GEN_MODEL,
                contents=prompt,
            )
            if not response or not getattr(response, "text", None):
                raise GenerationError("Empty response from model.")
            return response.text
        except GenerationError:
            raise
        except Exception as e:
            raise GenerationError(f"Summarization failed: {e}") from e


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def build_document_index(
    file_bytes: bytes,
    client: GeminiRAGClient,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    source: str = "document",
    embed_batch_size: int = 50,
) -> tuple[VectorStore, List[Chunk], str]:
    """
    Full ingestion pipeline: extract -> chunk -> embed -> index.
    Returns (vectorstore, chunks, full_text).
    """
    pages = extract_pages(file_bytes)
    full_text = " ".join(clean_text(t) for _, t in pages)

    chunks = chunk_text(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap, source=source)

    store = VectorStore(dim=client.EMBED_DIM)

    for i in range(0, len(chunks), embed_batch_size):
        batch = chunks[i:i + embed_batch_size]
        vectors = client.embed_texts([c.text for c in batch])
        if vectors.shape[0] != len(batch):
            raise EmbeddingError("Mismatch between embedded vectors and chunk batch.")
        store.add(vectors, batch)

    return store, chunks, full_text


def answer_question(
    question: str,
    store: VectorStore,
    client: GeminiRAGClient,
    k: int = 4,
    chat_history: Optional[List[dict]] = None,
) -> AnswerResult:
    if not question or not question.strip():
        return AnswerResult(answer="Please enter a question.", error="empty_question")

    try:
        query_vec = client.embed_query(question)
        retrieved = store.search(query_vec, k=k)
        answer = client.generate_answer(question, retrieved, chat_history=chat_history)
        return AnswerResult(answer=answer, sources=retrieved)
    except (EmbeddingError, GenerationError) as e:
        return AnswerResult(answer="", error=str(e))
