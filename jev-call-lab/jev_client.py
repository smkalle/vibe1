"""Jev client for OpenRouter's Decisions API, with mock and record/replay modes.

JEV_MODE   auto (default) | live | mock | record | replay
           auto = live when OPENROUTER_API_KEY is set, else mock
JEV_URL    default https://openrouter.ai/api/alpha/decisions
           (TypeSafe SDK path: https://openrouter.ai/api/v1/systemone)
JEV_MODEL  default typesafe/jev-1.13
JEV_FIXTURES  record/replay directory, default ./fixtures

    python jev_client.py --smoke     # one request: checks URL, model and answer shape
"""
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

import mock_jev

DEFAULT_URL = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_MODEL = "typesafe/jev-1.13"
PRICE_PER_INPUT_TOKEN = 0.042 / 1_000_000  # output tokens are free
RETRY_STATUSES = {408, 429, 500, 502, 503, 504}
MAX_RETRIES = 3
BACKOFF_S = 1.0


class JevError(RuntimeError):
    pass


def mode() -> str:
    m = os.environ.get("JEV_MODE", "auto").lower()
    if m == "auto":
        return "live" if os.environ.get("OPENROUTER_API_KEY") else "mock"
    if m not in {"live", "mock", "record", "replay"}:
        raise JevError(f"unknown JEV_MODE {m!r}")
    return m


def model() -> str:
    return os.environ.get("JEV_MODEL", DEFAULT_MODEL)


def fixtures_dir() -> Path:
    return Path(os.environ.get("JEV_FIXTURES", Path(__file__).parent / "fixtures"))


@dataclass(frozen=True)
class Config:
    """Settings for one run. The key is never printed (repr=False) and never written to disk."""

    mode: str = "mock"
    api_key: str | None = field(default=None, repr=False)
    url: str = DEFAULT_URL
    model: str = DEFAULT_MODEL
    fixtures: Path = Path(__file__).parent / "fixtures"

    def __post_init__(self):
        if self.mode not in {"live", "mock", "record", "replay"}:
            raise JevError(f"unknown mode {self.mode!r}")


def config_from_env() -> Config:
    return Config(
        mode=mode(),
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        url=os.environ.get("JEV_URL", DEFAULT_URL),
        model=model(),
        fixtures=fixtures_dir(),
    )


def cache_key(model_name: str, state: Any, questions: dict) -> str:
    canon = json.dumps({"model": model_name, "state": state, "questions": questions}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


def _post(payload: dict, timeout: float, cfg: Config) -> dict:
    key = cfg.api_key
    if not key:
        raise JevError("Set OPENROUTER_API_KEY (or JEV_MODE=mock)")
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost/jev-call-lab",
        "X-Title": "jev-call-lab",
    }
    url = cfg.url
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                # requests errors carry the URL and reason, never headers, so the key cannot leak here.
                raise JevError(f"request failed: {type(e).__name__}: {e}") from e
        else:
            if r.status_code < 400:
                return r.json()
            if r.status_code not in RETRY_STATUSES or attempt == MAX_RETRIES:
                raise JevError(f"{r.status_code}: {r.text[:500]}")
        time.sleep(BACKOFF_S * 2**attempt)
    raise AssertionError("unreachable")


REAL_LATENCY = {"measured", "recorded"}


def latency_of(mode_name: str, elapsed_ms: float, recorded_ms) -> tuple:
    """Latency to report, and where it came from.

    Only 'measured' (a live call) and 'recorded' (a live call's time saved in its fixture) are
    network latency. Mock answers and replays of older fixtures without a saved time only
    measure a local function call or file read, and must never be compared with real calls.
    """
    if mode_name in {"live", "record"}:
        return elapsed_ms, "measured"
    if mode_name == "replay" and isinstance(recorded_ms, (int, float)):
        return float(recorded_ms), "recorded"
    return elapsed_ms, "replay_local" if mode_name == "replay" else "mock"


def decide(state: Any, questions: dict, timeout: float = 30.0, cfg: Config | None = None) -> dict:
    """One request: every question here is about the same state. cfg defaults to the environment."""
    cfg = cfg or config_from_env()
    m = cfg.mode
    payload = {"model": cfg.model, "state": state, "questions": questions}
    fixture = cfg.fixtures / f"{cache_key(payload['model'], state, questions)}.json"

    t0 = time.perf_counter()
    recorded_ms = None
    if m == "mock":
        body = mock_jev.answer(payload)
    elif m == "replay":
        if not fixture.exists():
            raise JevError(f"replay miss: no fixture {fixture.name} (record it first with JEV_MODE=record)")
        saved = json.loads(fixture.read_text())
        body, recorded_ms = saved["response"], saved.get("latency_ms")
    else:
        body = _post(payload, timeout, cfg)
        if m == "record":
            fixture.parent.mkdir(parents=True, exist_ok=True)
            fixture.write_text(json.dumps({"request": payload, "response": body,
                                           "latency_ms": (time.perf_counter() - t0) * 1000}, indent=1))
    latency_ms, latency_source = latency_of(m, (time.perf_counter() - t0) * 1000, recorded_ms)

    usage = body.get("usage") or {}
    if isinstance(usage.get("cost"), (int, float)):
        cost, source = float(usage["cost"]), "usage.cost"
    else:
        cost, source = (usage.get("input_tokens") or 0) * PRICE_PER_INPUT_TOKEN, "tokens_x_price"
    return {**body, "_latency_ms": latency_ms, "_latency_source": latency_source,
            "_cost_usd": cost, "_cost_source": source, "_mode": m}


def _prob(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and 0.0 <= x <= 1.0 and math.isfinite(x)


def validate_response(questions: dict, body: dict) -> list[str]:
    """Check every answer against its question (eval E1). Empty list = valid."""
    errors = []
    answers = body.get("answers")
    if not isinstance(answers, dict):
        return ["answers missing"]
    for name, q in questions.items():
        a = answers.get(name)
        if not isinstance(a, dict):
            errors.append(f"{name}: no answer")
            continue
        if a.get("type") != q["type"]:
            errors.append(f"{name}: type {a.get('type')!r} != {q['type']!r}")
            continue
        if q["type"] == "noul":
            if not _prob(a.get("noul")):
                errors.append(f"{name}: noul {a.get('noul')!r} not in [0,1]")
        elif q["type"] == "choice":
            labels = set(q["criteria"])
            probs = a.get("probabilities") or {}
            if a.get("choice") not in labels:
                errors.append(f"{name}: choice {a.get('choice')!r} not in criteria")
            if set(probs) != labels or not all(_prob(p) for p in probs.values()):
                errors.append(f"{name}: probabilities keys/values invalid")
            elif abs(sum(probs.values()) - 1) > 0.02:
                errors.append(f"{name}: probabilities sum {sum(probs.values()):.3f}")
            if not _prob(a.get("confidence")):
                errors.append(f"{name}: confidence invalid")
        elif q["type"] == "score":
            n = len(q["criteria"])
            levels = {str(i) for i in range(n)}
            s = a.get("score")
            if not isinstance(s, (int, float)) or not 0 <= s <= n - 1:
                errors.append(f"{name}: score {s!r} outside [0, {n - 1}]")
            probs = a.get("probabilities") or {}
            if set(probs) != levels or not all(_prob(p) for p in probs.values()):
                errors.append(f"{name}: probabilities keys/values invalid")
            elif abs(sum(probs.values()) - 1) > 0.02:
                errors.append(f"{name}: probabilities sum {sum(probs.values()):.3f}")
            if set(a.get("legend") or {}) != levels:
                errors.append(f"{name}: legend keys invalid")
            if not _prob(a.get("confidence")):
                errors.append(f"{name}: confidence invalid")
    if not isinstance((body.get("usage") or {}).get("input_tokens"), int):
        errors.append("usage.input_tokens missing")
    return errors


def _smoke():
    from questions import WILL_BOOK

    state = {"current_event": "identity_check_failed", "current_stage": "identity"}
    print(f"mode={mode()} model={model()} url={os.environ.get('JEV_URL', DEFAULT_URL)}")
    r = decide(state, WILL_BOOK)
    print(json.dumps({k: v for k, v in r.items()}, indent=2))
    errs = validate_response(WILL_BOOK, r)
    print("contract:", "OK" if not errs else errs)
    raise SystemExit(1 if errs else 0)


if __name__ == "__main__":
    import sys

    if "--smoke" in sys.argv:
        _smoke()
    print(__doc__)
