# RAG starter for DeployKar

Upload PDFs or text files, then ask questions about them. Answers come only from your documents, with the
source file and page. One small backend (FastAPI) and one Postgres database with vector search.

## Deploy on DeployKar

1. Fork this repo (or click **New → RAG app** in DeployKar).
2. On the canvas, add a **Postgres** card and turn on **Vectors**, and an **AI** card.
3. Add a **Backend** card from your fork and draw a wire from each card to it. The Postgres wire sets
   `DATABASE_URL`; the AI wire sets `OPENAI_BASE_URL` and `OPENAI_API_KEY`. No other AI account needed.
4. In the backend's environment variables, set `ADMIN_TOKEN`: any long random text. You need it to upload or
   delete documents.
5. Deploy. Open the backend's address and upload a file.

No AI card? Set `OPENAI_API_KEY` to your own key for any OpenAI-compatible API (and `OPENAI_BASE_URL` if it
isn't OpenAI's).

The tables are created on first start. If something is missing, the page tells you what to set up.

## Settings

| Variable | Default | What it does |
| --- | --- | --- |
| `DATABASE_URL` | (from the wire) | Postgres with Vectors on |
| `OPENAI_API_KEY` | none | Key for the AI API |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Any OpenAI-compatible API |
| `CHAT_MODEL` | `default` with an AI card (the card's choice), else `gpt-4o-mini` | Writes the answers |
| `CHAT_MAX_TOKENS` / `REASONING_EFFORT` | not sent | For reasoning models (gpt-oss, o-series): e.g. `2000` / `low` |
| `EMBEDDING_MODEL` | `default` with an AI card, else `text-embedding-3-small` | Turns text into vectors |
| `EMBEDDING_DIM` | worked out | Numbers per vector: from the existing table, else one test embedding. Changing the embedding model later means re-uploading |
| `ADMIN_TOKEN` | none | Needed to upload and delete. Uploads are off without it |
| `MAX_UPLOAD_MB` | `10` | Largest file |
| `CHUNK_CHARS` / `CHUNK_OVERLAP` | `1200` / `200` | How documents are cut |
| `TOP_K` | `5` | Chunks sent with each question |

## API

```bash
# upload (answers at once; the file is read in the background)
curl -X POST "$URL/api/documents" -H "Authorization: Bearer $ADMIN_TOKEN" -F file=@handbook.pdf

# ask
curl -X POST "$URL/api/ask" -H 'Content-Type: application/json' -d '{"question": "How many leave days do I get?"}'
# {"answer": "You get 24 days a year [1].", "sources": [{"n": 1, "document": "handbook.pdf", "page": 12, ...}]}

curl "$URL/api/documents"                        # list and state (processing / ready / failed)
curl -X DELETE "$URL/api/documents/3" -H "Authorization: Bearer $ADMIN_TOKEN"
```

Anyone with the address can ask questions, and every question uses your AI key. Put the app behind your own
login, or set a spending limit with your AI provider, before you share it widely.

## How it works

1. **Upload:** the file's text is cut into overlapping chunks, each chunk becomes a vector
   (`EMBEDDING_MODEL`), and both are stored in the `chunks` table.
2. **Ask:** the question becomes a vector, Postgres finds the closest chunks (`ORDER BY embedding <=> …`,
   HNSW index), and those chunks plus the question go to `CHAT_MODEL`.

Scanned PDFs (images only) have no text to read: run OCR on them first.

## Growing it

- **Lots of files:** reading happens inside the web app. For big batches, add a DeployKar **Queue** and a
  second backend that does the reading, so uploads never slow down answers.
- **Files kept as files:** add a **Storage** bucket and save the originals there.

## Run locally

```bash
pip install -r requirements-dev.txt
pytest                                                    # database tests are skipped
RAG_TEST_DATABASE_URL=postgresql://... pytest             # also runs them (throwaway database with pgvector)
DATABASE_URL=postgresql://... OPENAI_API_KEY=... ADMIN_TOKEN=dev uvicorn app.main:app --reload
```

The local database needs the pgvector extension.
