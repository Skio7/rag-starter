"""The real SQL against a real Postgres with pgvector. Skipped unless RAG_TEST_DATABASE_URL is set; the
database's owner must be able to create the extension (a throwaway local database is fine)."""
import os

import psycopg
import pytest

from app import config, db

URL = os.environ.get("RAG_TEST_DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not URL, reason="RAG_TEST_DATABASE_URL not set")


@pytest.fixture
def database(monkeypatch):
    monkeypatch.setattr(config, "DATABASE_URL", URL)
    monkeypatch.setattr(config, "EMBEDDING_DIM", 3)
    monkeypatch.setattr(db, "_ready", False)
    with psycopg.connect(URL, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS chunks, documents")
        conn.execute("DROP EXTENSION IF EXISTS vector")
    yield URL
    with psycopg.connect(URL, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS chunks, documents")


def _vectors_on():
    with psycopg.connect(URL, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")


def test_says_when_vectors_are_off(database):
    assert db.vectors_on() is False
    with pytest.raises(db.SetupNeeded, match="turn on Vectors"):
        db.ensure_schema()


def test_store_search_delete(database):
    _vectors_on()
    assert db.vectors_on() is True
    db.ensure_schema()
    db.ensure_schema()  # repeat is harmless

    doc = db.add_document("policy.pdf")
    assert doc["status"] == "processing"
    chunks = [{"page": 1, "content": "refunds"}, {"page": 2, "content": "shipping"}, {"page": 3, "content": "hours"}]
    db.finish_document(doc["id"], chunks, [[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]])
    [listed] = db.list_documents()
    assert listed["status"] == "ready" and listed["chunks"] == 3

    hits = db.search([1, 0, 0], 2)
    assert [h["content"] for h in hits] == ["refunds", "hours"]
    assert hits[0]["document"] == "policy.pdf" and hits[0]["page"] == 1 and hits[0]["score"] == pytest.approx(1.0)

    failed = db.add_document("scan.pdf")
    db.fail_document(failed["id"], "No text found")
    assert db.search([1, 0, 0], 10)[-1]["document"] == "policy.pdf"  # failed documents are never searched

    assert db.delete_document(doc["id"]) is True
    assert db.search([1, 0, 0], 5) == []
    assert db.delete_document(doc["id"]) is False


def test_size_comes_from_one_test_embedding_then_from_the_table(database, monkeypatch):
    from app import ai
    _vectors_on()
    calls = []
    monkeypatch.setattr(config, "EMBEDDING_DIM", 0)
    monkeypatch.setattr(ai, "embed", lambda texts: calls.append(texts) or [[0.1] * 5])
    db.ensure_schema()
    assert config.EMBEDDING_DIM == 5 and len(calls) == 1
    monkeypatch.setattr(db, "_ready", False)
    monkeypatch.setattr(config, "EMBEDDING_DIM", 0)
    db.ensure_schema()  # the table now says 5: no second test embedding
    assert config.EMBEDDING_DIM == 5 and len(calls) == 1


def test_refuses_a_different_embedding_size(database, monkeypatch):
    _vectors_on()
    db.ensure_schema()
    monkeypatch.setattr(db, "_ready", False)
    monkeypatch.setattr(config, "EMBEDDING_DIM", 8)
    with pytest.raises(db.SetupNeeded, match="EMBEDDING_DIM back to 3"):
        db.ensure_schema()
