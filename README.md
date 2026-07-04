# Multilingual Document RAG MVP

This workspace contains a Streamlit app for uploading documents, building a local FAISS knowledge base, and asking grounded questions over the uploaded files.

What it does:

- Upload PDF, DOCX, XLSX, CSV, TXT, or MD files
- Extract text from the files, with OCR fallback for scanned PDFs
- Supports English plus Devanagari OCR when Tesseract Marathi/Hindi language data is available
- Split the text into chunks and build a local FAISS knowledge base
- Ask questions in a chat UI and get answers grounded in the uploaded files
- Show the source file and page for the retrieved chunks

How to run:

1. Install the packages in `requirements.txt`.
2. Set `CEREBRAS_API_KEY` in your environment.
3. Start Streamlit with `streamlit run app.py`.

## Deployment recommendation

Deploy this app on **Railway**, not Vercel.

Why Railway is the better fit:

- This is a long-running Streamlit server, not a serverless API function.
- The app uses large Python dependencies such as `sentence-transformers`, `faiss-cpu`, `pymupdf`, and OCR tooling.
- The app writes uploaded files and FAISS indexes to disk.
- OCR needs the `tesseract` system package plus Marathi/Hindi language data.

Vercel is excellent for Next.js/frontends and serverless functions, but this app is a Python Streamlit process with local state and heavier native dependencies. Railway can run it as a Docker container.

## Deploy on Railway

1. Push this project to GitHub.
2. Create a new Railway project from that GitHub repo.
3. Railway should detect the `Dockerfile`.
4. Add these Railway variables:

```text
CEREBRAS_API_KEY=your_active_key
CEREBRAS_MODEL=gpt-oss-120b
CEREBRAS_MAX_TOKENS=1024
CEREBRAS_TEMPERATURE=0.2
CEREBRAS_TOP_P=1
CEREBRAS_REASONING_EFFORT=medium
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
APP_DATA_DIR=/app/runtime_data
APP_TESSDATA_DIR=/usr/share/tesseract-ocr/5/tessdata
```

5. Add a Railway volume if you want uploads and FAISS indexes to survive redeploys:

```text
Mount path: /app/runtime_data
```

Without a volume, uploads and indexes can disappear when Railway rebuilds or restarts the container.

## Local Docker run

```powershell
docker build -t maharashtra-rag .
docker run --rm -p 8501:8501 --env-file .env maharashtra-rag
```

Then open:

```text
http://localhost:8501
```

Notes:

- OCR for scanned PDFs needs Tesseract installed on the machine in addition to the Python package. The Docker image installs Tesseract with English, Hindi, and Marathi language packs.
- The chat model uses Cerebras `gpt-oss-120b` through the SDK example you provided.
- The index is saved in `data/faiss_index` and the uploaded files are copied into `data/uploads`.
- In deployment, set `APP_DATA_DIR=/app/runtime_data`; the index is then saved in `/app/runtime_data/faiss_index` and uploads in `/app/runtime_data/uploads`.
