"""
RAG starter for DeployKar: upload documents, ask questions, get answers with their sources.

  GET  /                     a small page to try it
  GET  /health               for DeployKar's health check
  GET  /api/status           what's set up and what isn't
  POST /api/documents        upload a PDF or text file (needs ADMIN_TOKEN); it's read in the background
  GET  /api/documents        uploaded documents and their state
  DELETE /api/documents/{id} remove one (needs ADMIN_TOKEN)
  POST /api/ask              {"question": "..."} -> {"answer", "sources"}
"""
import hmac
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import ai, config, db, rag

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rag")

PAGE = Path(__file__).parent / "static" / "index.html"


@asynccontextmanager
async def lifespan(_app):
    try:
        db.ensure_schema()
        log.info("Database ready")
    except Exception as e:  # keep serving: /api/status explains what to set up
        log.warning("Not ready yet: %s", e)
    yield


app = FastAPI(title="RAG starter", lifespan=lifespan)


def _ready() -> None:
    try:
        db.ensure_schema()
    except db.SetupNeeded as e:
        raise HTTPException(503, str(e))


def _admin(authorization: str | None) -> None:
    if not config.ADMIN_TOKEN:
        raise HTTPException(403, "Uploads are off: set ADMIN_TOKEN in this backend's environment variables.")
    given = (authorization or "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(given.encode(), config.ADMIN_TOKEN.encode()):
        raise HTTPException(401, "Wrong or missing admin token.")


@app.get("/", include_in_schema=False)
def page():
    return FileResponse(PAGE)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/api/status")
def status():
    steps = {
        "database": bool(config.DATABASE_URL),
        "vectors": db.vectors_on() if config.DATABASE_URL else False,
        "ai_key": bool(config.AI_API_KEY),
        "admin_token": bool(config.ADMIN_TOKEN),
    }
    return {"ready": all(steps.values()), "steps": steps,
            "models": {"chat": config.CHAT_MODEL, "embedding": config.EMBEDDING_MODEL}}


def _ingest(doc_id: int, filename: str, data: bytes) -> None:
    """Read, chunk, embed and store one document (runs after the upload has answered)."""
    try:
        pages = rag.extract_pages(filename, data)
        chunks = rag.chunk_pages(pages, config.CHUNK_CHARS, config.CHUNK_OVERLAP)
        if not chunks:
            raise rag.UnsupportedFile("No text found (a scanned PDF needs OCR first).")
        embeddings = ai.embed([c["content"] for c in chunks])
        db.finish_document(doc_id, chunks, embeddings)
        log.info("Document %s ready: %d chunks", doc_id, len(chunks))
    except (rag.UnsupportedFile, ai.AIError) as e:
        db.fail_document(doc_id, str(e))
    except Exception as e:
        log.exception("Document %s failed", doc_id)
        db.fail_document(doc_id, f"Couldn't read this file ({type(e).__name__}).")


@app.post("/api/documents", status_code=202)
async def upload(background: BackgroundTasks, file: UploadFile = File(...),
                 authorization: str | None = Header(default=None)):
    _admin(authorization)
    _ready()
    data = await file.read(config.MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Files up to {config.MAX_UPLOAD_MB} MB.")
    name = Path(file.filename or "upload").name[:200]
    if not rag.supported(name):
        raise HTTPException(400, "Upload a PDF or a text file (.txt, .md, .csv, .json, .html).")
    doc = db.add_document(name)
    background.add_task(_ingest, doc["id"], name, data)
    return doc


@app.get("/api/documents")
def documents():
    _ready()
    return db.list_documents()


@app.delete("/api/documents/{doc_id}")
def remove(doc_id: int, authorization: str | None = Header(default=None)):
    _admin(authorization)
    _ready()
    if not db.delete_document(doc_id):
        raise HTTPException(404, "No such document.")
    return {"deleted": doc_id}


class Question(BaseModel):
    question: str = Field(..., min_length=1, max_length=config.MAX_QUESTION_CHARS)


@app.post("/api/ask")
def ask(body: Question):
    _ready()
    try:
        [embedding] = ai.embed([body.question])
        hits = db.search(embedding, config.TOP_K)
        if not hits:
            return {"answer": "No documents yet: upload some first.", "sources": []}
        answer = ai.chat(rag.build_messages(body.question, hits))
    except ai.AIError as e:
        raise HTTPException(502, str(e))
    return {"answer": answer, "sources": [
        {"n": i, "document": h["document"], "page": h["page"], "score": h["score"], "text": h["content"][:300]}
        for i, h in enumerate(hits, 1)]}
