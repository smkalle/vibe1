"""Generic OpenAI-compatible chat scorer (default: GLM-5.3) for the same typed questions Jev answers.

Jev's Decisions API takes {model, state, questions} natively; plain chat models need
the schema rendered into a prompt and the JSON answer parsed back into the same wire
shape, so simulate/evaluate/contract code works unchanged.

LLM_MODE  auto (default) | live | mock | record | replay
          auto = live when OPENROUTER_API_KEY is set, else mock (mock_jev, offline)
LLM_MODEL default z-ai/glm-5.3 (OpenRouter slug)
LLM_URL   default https://openrouter.ai/api/v1/chat/completions
LLM_EFFORT reasoning effort, default low (GLM-5.3 always thinks; max is pricey)
LLM_FIXTURES record/replay directory, default ./fixtures (files prefixed glm_)

The OpenRouter key is shared with Jev: env OPENROUTER_API_KEY or per-session memory.
Same hygiene rule: never written to disk, fixtures, results or logs.
"""
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

import mock_jev
from jev_client import JevError, latency_of, validate_response

DEFAULT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "z-ai/glm-5.3"
DEFAULT_EFFORT = "low"
PRICE_IN_PER_TOKEN = 1.40 / 1_000_000
PRICE_OUT_PER_TOKEN = 4.40 / 1_000_000
RETRY_STATUSES = {408, 429, 500, 502, 503, 504}
MAX_RETRIES = 3
MAX_FIXUPS = 2
BACKOFF_S = 1.0
FIXTURE_PREFIX = "glm_"

# Completion-size guardrails for cost estimates, calibrated on live z-ai/glm-5.3
# (measured: forecast mean 37 / p95 53, review mean 242 / max 282 tokens).
EST_COMPLETION_TOKENS_FORECAST = 150
EST_COMPLETION_TOKENS_REVIEW = 600
# Expected sizes (live means above), for "what will this probably cost" rather than the cap.
EXP_COMPLETION_TOKENS_FORECAST = 37
EXP_COMPLETION_TOKENS_REVIEW = 242


def mode() -> str:
    m = os.environ.get("LLM_MODE", "auto").lower()
    if m == "auto":
        return "live" if os.environ.get("OPENROUTER_API_KEY") else "mock"
    if m not in {"live", "mock", "record", "replay"}:
        raise JevError(f"unknown LLM_MODE {m!r}")
    return m


def model() -> str:
    return os.environ.get("LLM_MODEL", DEFAULT_MODEL)


def fixtures_dir() -> Path:
    return Path(os.environ.get("LLM_FIXTURES", Path(__file__).parent / "fixtures"))


@dataclass(frozen=True)
class Config:
    """Settings for one run. The key is never printed (repr=False) and never written to disk."""

    mode: str = "mock"
    api_key: str | None = field(default=None, repr=False)
    url: str = DEFAULT_URL
    model: str = DEFAULT_MODEL
    effort: str = DEFAULT_EFFORT
    fixtures: Path = Path(__file__).parent / "fixtures"

    def __post_init__(self):
        if self.mode not in {"live", "mock", "record", "replay"}:
            raise JevError(f"unknown mode {self.mode!r}")


def config_from_env() -> Config:
    return Config(
        mode=mode(),
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        url=os.environ.get("LLM_URL", DEFAULT_URL),
        model=model(),
        effort=os.environ.get("LLM_EFFORT", DEFAULT_EFFORT),
        fixtures=fixtures_dir(),
    )


def fixture_key(model_name: str, state: Any, questions: dict) -> str:
    canon = json.dumps({"scorer": "glm", "model": model_name, "state": state, "questions": questions},
                       sort_keys=True, separators=(",", ":"))
    return FIXTURE_PREFIX + hashlib.sha256(canon.encode()).hexdigest()


# prompts ---------------------------------------------------------------------
TYPE_SPEC = {
    "noul": '{"type": "noul", "noul": <probability 0..1>}',
    "choice": '{"type": "choice", "choice": <one criteria key>, "confidence": <0..1>, '
              '"probabilities": {<every criteria key>: <0..1, sums to 1>}}',
    "score": '{"type": "score", "score": <number, 0-based index into criteria>, "confidence": <0..1>, '
              '"legend": {<index string>: <criterion text>}, "probabilities": {<index string>: <0..1, sums to 1>}}',
}


def build_messages(state: Any, questions: dict) -> list:
    lines = ["You judge AI-receptionist calls from STRUCTURE ONLY (actors, stages, tool results, timing).",
             "There is no transcript. Answer every question with exactly the JSON shape for its type.",
             "Return one JSON object: {\"answers\": {<question name>: <typed answer>}}. No other text."]
    for name, q in questions.items():
        lines.append(f"- {name} ({q['type']}): {q.get('instructions', '')}")
        if q.get("criteria"):
            lines.append(f"  criteria: {json.dumps(q['criteria'], separators=(',', ':'))}")
        lines.append(f"  shape: {TYPE_SPEC[q['type']]}")
    return [
        {"role": "system", "content": "You are a precise JSON-only judge. You always reply with a single valid JSON object."},
        {"role": "user", "content": "\n".join(lines) + "\n\nstate:\n"
         + json.dumps(state, separators=(",", ":"))},
    ]


def extract_json(text: str) -> dict:
    """Pull a JSON object out of chat output (tolerates fences and prose)."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    try:
        obj = json.loads(text)
    except ValueError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise JevError(f"no JSON object in reply: {text[:200]!r}")
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise JevError(f"reply JSON is not an object: {text[:200]!r}")
    return obj


def _norm_usage(usage: dict) -> dict:
    """Map chat usage onto the Jev usage shape the contract validator expects."""
    usage = dict(usage or {})
    usage.setdefault("input_tokens", usage.get("prompt_tokens", 0))
    usage.setdefault("output_tokens", usage.get("completion_tokens", 0))
    return usage


def to_wire(parsed: dict, questions: dict, model_name: str, usage: dict) -> dict:
    """Normalize chat JSON into the Jev wire shape ({model, answers, usage})."""
    if not isinstance(parsed, dict):
        raise JevError(f"reply JSON is not an object: {str(parsed)[:200]!r}")
    answers = parsed.get("answers", parsed)
    if not isinstance(answers, dict):
        raise JevError(f"reply has no answers object: {str(parsed)[:200]!r}")
    for name, q in questions.items():
        a = answers.get(name)
        if isinstance(a, dict) and a.get("type") is None:
            a = dict(a, type=q["type"])  # chat models often drop the discriminator; it is implied
            answers[name] = a
    return {"model": model_name, "answers": answers, "usage": _norm_usage(usage)}


def project_cost(calls, review=True):
    """Rough pre-run USD estimate: chars/4 input tokens + assumed completion tokens."""
    import json as _j

    n_req = tokens_in = tokens_out = 0
    from questions import REVIEW as _REVIEW, WILL_BOOK as _WB
    from simulate import forecast_points as _fp

    for c in calls:
        for i in _fp(c["events"]):
            tokens_in += len(_j.dumps(build_messages({"s": 1}, _WB), separators=(",", ":"))) // 4
            tokens_in += len(_j.dumps(c["events"][: i + 1], separators=(",", ":"))) // 4
            tokens_out += EST_COMPLETION_TOKENS_FORECAST
            n_req += 1
        if review:
            tokens_in += len(_j.dumps(build_messages({"s": 1}, _REVIEW), separators=(",", ":"))) // 4
            tokens_in += len(_j.dumps(c["events"], separators=(",", ":"))) // 4
            tokens_out += EST_COMPLETION_TOKENS_REVIEW
            n_req += 1
    usd = tokens_in * PRICE_IN_PER_TOKEN + tokens_out * PRICE_OUT_PER_TOKEN
    exp_out = tokens_out * EXP_COMPLETION_TOKENS_FORECAST / EST_COMPLETION_TOKENS_FORECAST if not review else \
        sum(EXP_COMPLETION_TOKENS_FORECAST * len(_fp(c["events"])) + EXP_COMPLETION_TOKENS_REVIEW for c in calls)
    return {"requests": n_req, "input_tokens": tokens_in, "est_completion_tokens": tokens_out,
            "usd": round(usd, 4), "usd_expected": round(tokens_in * PRICE_IN_PER_TOKEN + exp_out * PRICE_OUT_PER_TOKEN, 4)}


# transport ---------------------------------------------------------------------
def _post_chat(payload: dict, timeout: float, cfg: Config) -> dict:
    key = cfg.api_key
    if not key:
        raise JevError("Set OPENROUTER_API_KEY (or LLM_MODE=mock)")
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost/jev-call-lab",
        "X-Title": "jev-call-lab-glm",
    }
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = requests.post(cfg.url, json=payload, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                raise JevError(f"request failed: {type(e).__name__}: {e}") from e
        else:
            if r.status_code < 400:
                return r.json()
            if r.status_code not in RETRY_STATUSES or attempt == MAX_RETRIES:
                raise JevError(f"{r.status_code}: {r.text[:500]}")
        time.sleep(BACKOFF_S * 2**attempt)
    raise AssertionError("unreachable")


def _chat_once(state, questions, timeout, cfg) -> dict:
    payload = {"model": cfg.model, "messages": build_messages(state, questions),
               "temperature": 0, "response_format": {"type": "json_object"},
               "reasoning": {"effort": cfg.effort}}
    raw = _post_chat(payload, timeout, cfg)
    try:
        content = raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise JevError(f"unexpected chat reply shape: {str(raw)[:300]!r}")
    return content, raw.get("usage") or {}


def _attempt(content: str, questions: dict, model_name: str, usage: dict):
    """Parse + validate one reply. Returns (body or None, error list)."""
    try:
        body = to_wire(extract_json(content), questions, model_name, usage)
    except JevError as e:
        return None, [str(e)[:200]]
    return body, validate_response(questions, body)


def _usage_cost(usage: dict) -> tuple:
    if isinstance(usage.get("cost"), (int, float)):
        return float(usage["cost"]), "usage.cost"
    cost = (usage.get("prompt_tokens") or 0) * PRICE_IN_PER_TOKEN + \
        (usage.get("completion_tokens") or 0) * PRICE_OUT_PER_TOKEN
    return cost, "glm_tokens_x_price"


def decide(state: Any, questions: dict, timeout: float = 90.0, cfg: Config | None = None) -> dict:
    """One request: every question here is about the same state. Returns the Jev wire shape + _latency/_cost/_mode."""
    cfg = cfg or config_from_env()
    m = cfg.mode
    fixture = cfg.fixtures / f"{fixture_key(cfg.model, state, questions)}.json"

    t0 = time.perf_counter()
    recorded_ms = None
    if m == "mock":
        body = mock_jev.answer({"model": cfg.model, "state": state, "questions": questions})
    elif m == "replay":
        if not fixture.exists():
            raise JevError(f"replay miss: no fixture {fixture.name} (record it first with LLM_MODE=record)")
        saved = json.loads(fixture.read_text())
        recorded_ms = saved.get("latency_ms")
        body, usage = saved["response"], _norm_usage(saved.get("usage", {}))
        body = {**body, "usage": usage or body.get("usage", {})}
    else:
        messages = build_messages(state, questions)
        content, usage = _chat_once(state, questions, timeout, cfg)
        body, errors = _attempt(content, questions, cfg.model, usage)
        for _ in range(MAX_FIXUPS):
            if not errors:
                break
            messages = messages + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": "Your last reply was invalid: " + "; ".join(errors)
                 + ". Reply again with ONLY the corrected JSON object."},
            ]
            payload = {"model": cfg.model, "messages": messages,
                       "temperature": 0, "response_format": {"type": "json_object"},
                       "reasoning": {"effort": cfg.effort}}
            raw = _post_chat(payload, timeout, cfg)
            content = raw["choices"][0]["message"]["content"]
            usage = raw.get("usage") or usage
            body, errors = _attempt(content, questions, cfg.model, usage)
        if errors or body is None:
            raise JevError(f"GLM reply failed contract after {MAX_FIXUPS} fixups: {'; '.join(errors)}")
        if m == "record":
            fixture.parent.mkdir(parents=True, exist_ok=True)
            fixture.write_text(json.dumps({"request": {"model": cfg.model, "state": state, "questions": questions},
                                           "usage": usage, "response": body,
                                           "latency_ms": (time.perf_counter() - t0) * 1000}, indent=1))
    latency_ms, latency_source = latency_of(m, (time.perf_counter() - t0) * 1000, recorded_ms)

    usage = body.get("usage") or {}
    if m == "mock":
        from jev_client import PRICE_PER_INPUT_TOKEN
        cost = (usage.get("input_tokens") or 0) * PRICE_PER_INPUT_TOKEN
        source = "tokens_x_price"
    else:
        cost, source = _usage_cost(usage)
    return {**body, "_latency_ms": latency_ms, "_latency_source": latency_source,
            "_cost_usd": cost, "_cost_source": source, "_mode": m}


def _smoke():
    from questions import WILL_BOOK

    print(f"mode={mode()} model={model()} url={os.environ.get('LLM_URL', DEFAULT_URL)}")
    r = decide({"current_event": "identity_check_failed", "current_stage": "identity"}, WILL_BOOK)
    errs = validate_response(WILL_BOOK, r)
    print(json.dumps({k: (v if k != "answers" else "...") for k, v in r.items()}, indent=2))
    print("contract:", "OK" if not errs else errs)
    raise SystemExit(1 if errs else 0)


if __name__ == "__main__":
    import sys

    if "--smoke" in sys.argv:
        _smoke()
    print(__doc__)
