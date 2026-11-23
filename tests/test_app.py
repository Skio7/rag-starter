"""Chunking, prompts and the API, with the database and the AI service replaced by fakes."""
import pytest
from fastapi.testclient import TestClient

from app import ai, config, db, main, rag


# ---------------------------------------------------------------- chunks and prompts

def test_chunks_cover_the_text_with_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    pieces = rag.chunk_text(text, size=300, overlap=60)
    assert len(pieces) > 5
    assert all(len(p) <= 300 for p in pieces)
    assert pieces[0].startswith("Sentence number 0")
    assert "Sentence number 199 is here." in pieces[-1]
    # consecutive pieces share text, so a fact on the boundary is whole in one of them
    assert pieces[1][:20] in pieces[0]


def test_chunks_prefer_paragraph_ends():
    text = ("A" * 250) + ".\n\n" + ("B" * 250)
    pieces = rag.chunk_text(text, size=400, overlap=0)
    assert pieces[0].endswith(".") and set(pieces[0][:-1]) == {"A"}


def test_empty_and_tiny_text():
    assert rag.chunk_text("   \n\n ") == []
    assert rag.chunk_text("hello") == ["hello"]


def test_text_files_and_refusals():
    assert rag.extract_pages("notes.md", "# Hi\r\nthere".encode()) == [(1, "# Hi\r\nthere")]
    assert rag.supported("Report.PDF") and rag.supported("a.txt") and not rag.supported("a.exe")
    with pytest.raises(rag.UnsupportedFile):
        rag.extract_pages("photo.png", b"...")


def test_pages_keep_their_numbers():
    chunks = rag.chunk_pages([(1, "first page text"), (2, "second page text")], 1200, 200)
    assert [c["page"] for c in chunks] == [1, 2]


def test_vector_literal():
    assert rag.vector_literal([1, 0.5, -0.25]) == "[1,0.5,-0.25]"


def test_prompt_numbers_sources_and_keeps_them_out_of_the_rules():
    hits = [{"document": "a.pdf", "page": 3, "content": "Refunds within 7 days."}]
    system, user = rag.build_messages("Refund window?", hits)
    assert system["role"] == "system" and "never invent" in system["content"]
    assert "[1] a.pdf (page 3):\nRefunds within 7 days." in user["content"]
    assert user["content"].endswith("Question: Refund window?")


# ---------------------------------------------------------------- the API

class FakeDB:
    def __init__(self):
        self.docs, self.chunks, self.next_id = {}, [], 1

    def ensure_schema(self):
        pass

    def vectors_on(self):
        return True

    def add_document(self, name):
        doc = {"id": self.next_id, "name": name, "status": "processing", "chunks": 0, "created_at": "now"}
        self.docs[doc["id"]] = doc
        self.next_id += 1
        return dict(doc)

    def finish_document(self, doc_id, chunks, embeddings):
        assert len(chunks) == len(embeddings)
        self.chunks += [(doc_id, c) for c in chunks]
        self.docs[doc_id].update(status="ready", chunks=len(chunks))

    def fail_document(self, doc_id, error):
        self.docs[doc_id].update(status="failed", error=error)

    def list_documents(self):
        return list(self.docs.values())

    def delete_document(self, doc_id):
        return self.docs.pop(doc_id, None) is not None

    def search(self, embedding, k):
        return [{"document": self.docs[d]["name"], "page": c["page"], "content": c["content"], "score": 0.9}
                for d, c in self.chunks][:k]


@pytest.fixture
def client(monkeypatch):
    fake = FakeDB()
    for name in ("ensure_schema", "vectors_on", "add_document", "finish_document", "fail_document",
                 "list_documents", "delete_document", "search"):
        monkeypatch.setattr(db, name, getattr(fake, name))
    monkeypatch.setattr(config, "DATABASE_URL", "postgresql://x")
    monkeypatch.setattr(config, "AI_API_KEY", "sk-test")
    monkeypatch.setattr(config, "ADMIN_TOKEN", "admin-secret")
    asked = []
    monkeypatch.setattr(ai, "embed", lambda texts: [[0.0] * config.EMBEDDING_DIM for _ in texts])
    monkeypatch.setattr(ai, "chat", lambda messages: asked.append(messages) or "Within 7 days [1].")
    with TestClient(main.app) as c:
        c.fake, c.asked = fake, asked
        yield c


def test_status_and_page(client):
    assert client.get("/health").json() == {"ok": True}
    assert client.get("/api/status").json()["ready"] is True
    assert "Ask your documents" in client.get("/").text


def test_upload_needs_the_admin_token(client):
    files = {"file": ("policy.txt", b"Refunds within 7 days.", "text/plain")}
    assert client.post("/api/documents", files=files).status_code == 401
    assert client.post("/api/documents", files=files, headers={"Authorization": "Bearer nope"}).status_code == 401


def test_upload_read_ask_delete(client):
    auth = {"Authorization": "Bearer admin-secret"}
    r = client.post("/api/documents", headers=auth,
                    files={"file": ("policy.txt", b"Refunds are accepted within 7 days of purchase.", "text/plain")})
    assert r.status_code == 202
    doc_id = r.json()["id"]
    [doc] = client.get("/api/documents").json()
    assert doc["status"] == "ready" and doc["chunks"] == 1  # the background task ran

    answer = client.post("/api/ask", json={"question": "What's the refund window?"}).json()
    assert answer["answer"] == "Within 7 days [1]."
    assert answer["sources"][0]["document"] == "policy.txt"
    assert "Refunds are accepted" in client.asked[0][1]["content"]

    assert client.delete(f"/api/documents/{doc_id}", headers=auth).status_code == 200
    assert client.get("/api/documents").json() == []


def test_refusals(client):
    auth = {"Authorization": "Bearer admin-secret"}
    assert client.post("/api/documents", headers=auth,
                       files={"file": ("virus.exe", b"MZ", "application/octet-stream")}).status_code == 400
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
    assert client.post("/api/ask", json={"question": "x" * 3000}).status_code == 422
    assert client.post("/api/ask", json={"question": "anything?"}).json()["sources"] == []


def test_a_file_without_text_fails_clearly(client):
    auth = {"Authorization": "Bearer admin-secret"}
    client.post("/api/documents", headers=auth, files={"file": ("empty.txt", b"   ", "text/plain")})
    [doc] = client.get("/api/documents").json()
    assert doc["status"] == "failed" and "No text" in doc["error"]


def test_uploads_off_without_a_token(client, monkeypatch):
    monkeypatch.setattr(config, "ADMIN_TOKEN", "")
    r = client.post("/api/documents", files={"file": ("a.txt", b"hi", "text/plain")})
    assert r.status_code == 403 and "ADMIN_TOKEN" in r.json()["detail"]


def test_setup_messages_reach_the_caller(client, monkeypatch):
    def vectors_off():
        raise db.SetupNeeded("Vectors are off: turn on Vectors on your Postgres card, then try again.")
    monkeypatch.setattr(db, "ensure_schema", vectors_off)
    r = client.post("/api/ask", json={"question": "hi"})
    assert r.status_code == 503 and "turn on Vectors" in r.json()["detail"]


def test_ai_errors_are_plain(client, monkeypatch):
    def down(texts):
        raise ai.AIError("The AI service refused the request (401). Incorrect API key.")
    monkeypatch.setattr(ai, "embed", down)
    r = client.post("/api/ask", json={"question": "hi"})
    assert r.status_code == 502 and "401" in r.json()["detail"]


def test_embedding_size_is_checked(monkeypatch):
    monkeypatch.setattr(config, "AI_API_KEY", "k")
    monkeypatch.setattr(ai, "_post", lambda path, body: {"data": [{"index": 0, "embedding": [0.1] * 8}]})
    with pytest.raises(ai.AIError, match="EMBEDDING_DIM"):
        ai.embed(["hi"])
