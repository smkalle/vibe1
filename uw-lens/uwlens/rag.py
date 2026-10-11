"""Optional research summary from a local LLM (Ollama on localhost). Off by default.

The prompt confines the model to the retrieved evidence, requires [n] citations, and forbids
issuing an underwriting decision: the output is a research aid for a human underwriter.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

from .search import Hit

SYSTEM = (
    "You are a research assistant for a life and health insurance underwriter. "
    "Answer ONLY from the numbered evidence. Cite every claim with [n]. "
    "If the evidence does not answer the question, say so. "
    "Do not make or recommend an underwriting decision, premium rating, or diagnosis; "
    "summarize what the evidence says and list open questions the underwriter should verify. "
    "Answer in the language of the question."
)

Transport = Callable[[str, dict, float], dict]


def _http_post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - localhost by design
        return json.loads(resp.read().decode())


@dataclass
class RagAnswer:
    text: str
    citations: list[int]
    model: str
    error: str | None = None


def build_prompt(question: str, hits: list[Hit], max_chars: int = 1500) -> str:
    blocks = []
    for n, h in enumerate(hits, 1):
        r = h.record
        loc = f", {r['locator']}" if r.get("locator") else ""
        blocks.append(f"[{n}] {r['title']} ({r['source']}, {r['modality']}{loc})\n{r['text'][:max_chars]}")
    return "Evidence:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\nAnswer with citations:"


def is_local(url: str) -> bool:
    return bool(re.match(r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?(/|$)", url))


def answer(question: str, hits: list[Hit], *, url: str, model: str, transport: Transport | None = None,
           timeout: float = 120.0, allow_remote: bool = False) -> RagAnswer:
    if not hits:
        return RagAnswer("No evidence retrieved; nothing to summarize.", [], model)
    if not allow_remote and not is_local(url):
        return RagAnswer("", [], model, error=f"Refusing non-local LLM endpoint {url} (PHI must stay on this machine).")
    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": 0.1},
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(question, hits)}],
    }
    try:
        data = (transport or _http_post)(url.rstrip("/") + "/api/chat", payload, timeout)
        text = (data.get("message") or {}).get("content", "").strip()
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return RagAnswer("", [], model, error=f"Local LLM unavailable at {url}: {exc}. Start it with `ollama serve`.")
    cites = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(hits)})
    return RagAnswer(text, cites, model)
