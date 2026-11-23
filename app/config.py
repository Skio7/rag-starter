"""Settings, all from environment variables (DeployKar's canvas wires set most of them)."""
import os


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


# The Postgres card (with Vectors on) wired to this backend sets DATABASE_URL (and PGSSLMODE).
DATABASE_URL = os.environ.get("DATABASE_URL", "")

# Any OpenAI-compatible API: OpenAI itself today, DeployKar AI once its card is wired (it sets both).
AI_BASE_URL = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
AI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
# Sent only when set (some providers reject them). Reasoning models (gpt-oss, o-series) need room to think
# before they answer: e.g. CHAT_MAX_TOKENS=2000 and REASONING_EFFORT=low.
CHAT_MAX_TOKENS = _int("CHAT_MAX_TOKENS", 0)
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "").strip()
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
# Must match the embedding model (text-embedding-3-small: 1536). Changing it later means re-uploading.
EMBEDDING_DIM = _int("EMBEDDING_DIM", 1536)

# Uploading and deleting documents needs this token; asking questions doesn't.
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")

MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 10)
CHUNK_CHARS = _int("CHUNK_CHARS", 1200)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 200)
TOP_K = _int("TOP_K", 5)
MAX_QUESTION_CHARS = 2000
