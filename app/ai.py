"""Calls to an OpenAI-compatible API: embeddings and chat. Plain HTTP, no SDK, so the app stays small."""
import httpx

from app import config

TIMEOUT = httpx.Timeout(60.0, connect=5.0)
BATCH = 64


class AIError(RuntimeError):
    pass


def _post(path: str, body: dict) -> dict:
    if not config.AI_API_KEY:
        raise AIError("No AI key: set OPENAI_API_KEY (or wire a DeployKar AI card to this backend).")
    try:
        r = httpx.post(f"{config.AI_BASE_URL}{path}", json=body, timeout=TIMEOUT,
                       headers={"Authorization": f"Bearer {config.AI_API_KEY}"})
    except httpx.HTTPError as e:
        raise AIError(f"The AI service didn't answer ({type(e).__name__}). Try again.") from e
    if r.status_code >= 400:
        try:
            detail = r.json().get("error", {}).get("message", "")
        except Exception:
            detail = ""
        raise AIError(f"The AI service refused the request ({r.status_code}). {detail}".strip())
    return r.json()


def embed(texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        data = _post("/embeddings", {"model": config.EMBEDDING_MODEL, "input": texts[i:i + BATCH]})["data"]
        out.extend(item["embedding"] for item in sorted(data, key=lambda d: d["index"]))
    for vector in out:
        if config.EMBEDDING_DIM and len(vector) != config.EMBEDDING_DIM:
            raise AIError(f"The embedding model returned {len(vector)} numbers, but EMBEDDING_DIM is "
                          f"{config.EMBEDDING_DIM}. Set EMBEDDING_DIM to {len(vector)} before uploading anything.")
    return out


def chat(messages: list[dict]) -> str:
    body = {"model": config.CHAT_MODEL, "messages": messages}
    if config.CHAT_MAX_TOKENS:
        body["max_tokens"] = config.CHAT_MAX_TOKENS
    if config.REASONING_EFFORT:
        body["reasoning_effort"] = config.REASONING_EFFORT
    choice = _post("/chat/completions", body)["choices"][0]
    answer = (choice.get("message", {}).get("content") or "").strip()
    if not answer:
        # reasoning models can spend the whole output allowance thinking and say nothing
        hint = (" It ran out of room: raise CHAT_MAX_TOKENS (e.g. 2000) or set REASONING_EFFORT=low."
                if choice.get("finish_reason") == "length" else " Try again.")
        raise AIError("The AI model returned no answer." + hint)
    return answer
