# ⚔️ Kimetsu no Bunseki — AI PDF Chatbot (Demon Slayer theme)

An AI-powered PDF chatbot: upload any PDF, ask questions, get AI summaries,
key-point extraction, and answers with page-number citations. Built with
Streamlit + Google Gemini + FAISS, themed around Demon Slayer / Kimetsu no Yaiba.

## Features

- 📄 Upload any PDF and chat with it
- 🗡️ Ask questions, get answers grounded in the document with page citations
- 🌊 One-click full document summary
- ⚡ One-click key-point extraction
- 🎨 Custom Demon Slayer visual theme (haori checkerboard, flame gradients, forest-green sidebar)
- ✅ Fully tested core logic (30 unit tests, offline/mocked)

## Tech Stack

- **UI:** Streamlit
- **LLM + Embeddings:** Google Gemini (`gemini-2.5-flash` + `gemini-embedding-001`)
- **Vector store:** FAISS (in-memory, per-session)
- **PDF parsing:** pypdf

## Project Structure

```
pdf-chatbot/
├── app.py                  # Streamlit UI
├── rag_engine.py           # Core RAG logic (PDF parsing, chunking, embeddings, retrieval, generation)
├── requirements.txt
├── .env.example
├── .streamlit/
│   └── config.toml         # Theme + server config
├── assets/
│   └── style.css           # Demon Slayer custom CSS
└── tests/
    └── test_rag_engine.py  # 30 unit tests, mocked API calls
```

## 1. Local Setup

```bash
git clone <your-repo-url>
cd pdf-chatbot
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Get a **free** Gemini API key: https://aistudio.google.com/apikey

Copy `.env.example` to `.env` and fill in your key:
```bash
cp .env.example .env
# edit .env and paste your key
```

Or skip `.env` entirely and paste your key directly into the sidebar when the app runs.

## 2. Run Locally

```bash
streamlit run app.py
```

Open http://localhost:8501

## 3. Run Tests

```bash
pip install pytest reportlab   # reportlab only needed for test PDF generation
pytest tests/ -v
```

All 30 tests should pass. They're fully mocked, so no API key or network access is needed to run them.

## 4. Deploy — Streamlit Community Cloud (free, recommended)

1. Push this project to a **public or private GitHub repo**.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **"New app"**, select your repo, branch, and set the main file to `app.py`.
4. Under **Advanced settings → Secrets**, add:
   ```toml
   GOOGLE_API_KEY = "your_gemini_api_key_here"
   ```
5. Click **Deploy**. Your app will be live at `https://<your-app-name>.streamlit.app`.

> Note: with secrets configured this way, users won't need to paste their own
> API key — but the app also lets each visitor paste their own key in the
> sidebar, which is safer if you're sharing this publicly (avoids you paying
> for everyone's usage).

### Alternative: Deploy on Render / Railway / Fly.io

These platforms work too since this is a standard Streamlit app. General steps:
1. Add a `Procfile` (see below) or use their Python buildpack.
2. Set the start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
3. Set `GOOGLE_API_KEY` as an environment variable in their dashboard.

Example `Procfile` for Render/Railway/Heroku-style platforms:
```
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

### Alternative: Docker

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t kimetsu-pdf-chatbot .
docker run -p 8501:8501 -e GOOGLE_API_KEY=your_key kimetsu-pdf-chatbot
```

## Known Limitations

- Scanned/image-only PDFs (no embedded text layer) won't work — there's no OCR step. The app detects this and shows a clear error rather than failing silently.
- Vector index is in-memory per session — restarting the app clears uploaded documents. For persistence across restarts, you'd add disk or database-backed storage.
- Max upload size is 25MB by default (configurable in `.streamlit/config.toml` and enforced in `app.py`).
- Each user session holds its own document index; this is fine for personal/small-team use. For high-traffic multi-user deployment, consider adding per-user rate limiting on Gemini API calls.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Embedding request failed" | Check your API key is valid and has quota at aistudio.google.com |
| "No extractable text found" | Your PDF is likely scanned images — try a text-based PDF, or add an OCR preprocessing step |
| App is slow on large PDFs | Increase chunk size in the sidebar to reduce the number of embedding calls |
| Deployed app can't find API key | Make sure you added `GOOGLE_API_KEY` under Secrets (Streamlit Cloud) or environment variables (other platforms) |
