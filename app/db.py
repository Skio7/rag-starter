"""Postgres with pgvector: documents and their chunks, and the nearest-chunk search."""
import psycopg

from app import config
from app.rag import vector_literal


class SetupNeeded(RuntimeError):
    """Something on the DeployKar canvas isn't set up yet; the message says what."""


def connect() -> psycopg.Connection:
    if not config.DATABASE_URL:
        raise SetupNeeded("No database: wire a Postgres card to this backend (it sets DATABASE_URL).")
    # prepare_threshold=None: DeployKar's databases sit behind PgBouncer in transaction mode,
    # where server-side prepared statements don't work.
    return psycopg.connect(config.DATABASE_URL, autocommit=True, prepare_threshold=None, connect_timeout=10)


_ready = False


def ensure_schema() -> None:
    """Create the tables once. Needs the Postgres card's Vectors switch on."""
    global _ready
    if _ready:
        return
    with connect() as conn:
        if not conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone():
            raise SetupNeeded("Vectors are off: turn on Vectors on your Postgres card, then try again.")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id bigserial PRIMARY KEY,
                name text NOT NULL,
                status text NOT NULL DEFAULT 'processing',
                error text,
                chunks integer NOT NULL DEFAULT 0,
                created_at timestamptz NOT NULL DEFAULT now()
            )""")
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id bigserial PRIMARY KEY,
                document_id bigint NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                page integer NOT NULL,
                position integer NOT NULL,
                content text NOT NULL,
                embedding vector({config.EMBEDDING_DIM}) NOT NULL
            )""")
        # an existing table made for another embedding size can't take these vectors
        row = conn.execute("SELECT atttypmod FROM pg_attribute WHERE attrelid = 'chunks'::regclass "
                           "AND attname = 'embedding'").fetchone()
        if row and row[0] != config.EMBEDDING_DIM:
            raise SetupNeeded(f"The chunks table holds {row[0]}-number embeddings but EMBEDDING_DIM is "
                              f"{config.EMBEDDING_DIM}. Set EMBEDDING_DIM back to {row[0]}.")
        conn.execute("CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks "
                     "USING hnsw (embedding vector_cosine_ops)")
        conn.execute("CREATE INDEX IF NOT EXISTS chunks_document ON chunks (document_id)")
    _ready = True


def vectors_on() -> bool:
    try:
        with connect() as conn:
            return conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone() is not None
    except Exception:
        return False


def add_document(name: str) -> dict:
    with connect() as conn:
        row = conn.execute("INSERT INTO documents (name) VALUES (%s) RETURNING id, name, status, chunks, created_at",
                           (name,)).fetchone()
    return _doc(row)


def finish_document(doc_id: int, chunks: list[dict], embeddings: list[list[float]]) -> None:
    with connect() as conn, conn.transaction():
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (document_id, page, position, content, embedding) VALUES (%s, %s, %s, %s, %s::vector)",
                [(doc_id, c["page"], i, c["content"], vector_literal(e))
                 for i, (c, e) in enumerate(zip(chunks, embeddings))])
        conn.execute("UPDATE documents SET status = 'ready', chunks = %s, error = NULL WHERE id = %s",
                     (len(chunks), doc_id))


def fail_document(doc_id: int, error: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE documents SET status = 'failed', error = %s WHERE id = %s", (error[:500], doc_id))


def list_documents() -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT id, name, status, chunks, created_at, error FROM documents "
                            "ORDER BY created_at DESC LIMIT 500").fetchall()
    return [_doc(r) for r in rows]


def delete_document(doc_id: int) -> bool:
    with connect() as conn:
        return conn.execute("DELETE FROM documents WHERE id = %s", (doc_id,)).rowcount > 0


def search(embedding: list[float], k: int) -> list[dict]:
    """The k chunks closest in meaning (cosine distance), with a 0-1 similarity score."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT d.name, c.page, c.content, 1 - (c.embedding <=> %s::vector) AS score "
            "FROM chunks c JOIN documents d ON d.id = c.document_id WHERE d.status = 'ready' "
            "ORDER BY c.embedding <=> %s::vector LIMIT %s",
            (vector_literal(embedding), vector_literal(embedding), k)).fetchall()
    return [{"document": r[0], "page": r[1], "content": r[2], "score": round(float(r[3]), 4)} for r in rows]


def _doc(row) -> dict:
    doc = {"id": row[0], "name": row[1], "status": row[2], "chunks": row[3], "created_at": row[4].isoformat()}
    if len(row) > 5 and row[5]:
        doc["error"] = row[5]
    return doc
