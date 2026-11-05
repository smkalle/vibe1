"""Tiny OpenAI-compatible server around any Omni Pocket engine.

Run it on a PC/laptop with a GPU (or on UserLAnd if you have the RAM), then
point the phone's ``remote`` engine at ``http://<host>:8088/v1``.

    python -m omnipocket serve --engine transformers --host 0.0.0.0 --port 8088

Endpoints: ``GET /health``, ``GET /v1/models``, ``POST /v1/chat/completions``
(``stream`` true/false, multimodal ``image_url`` data URLs and
``input_audio`` WAV parts, optional ``modalities: ["text","audio"]``).
Set ``OMNI_SERVER_API_KEY`` to require ``Authorization: Bearer <key>``.
"""

from __future__ import annotations

import base64
import hmac
import json
import os
import threading
import time
import uuid

from .engines import OmniRequest
from .media import decode_data_url, normalize_speech, prepare_image


def parse_messages(messages: list[dict]) -> OmniRequest:
    """OpenAI chat messages -> OmniRequest. The last user message carries media."""
    if not messages:
        raise ValueError("messages must not be empty")
    req = OmniRequest()
    systems = [m for m in messages if m.get("role") == "system"]
    if systems:
        req.system = _text_of(systems[-1].get("content"))
    convo = [m for m in messages if m.get("role") in ("user", "assistant")]
    if not convo or convo[-1].get("role") != "user":
        raise ValueError("last non-system message must be from the user")
    for m in convo[:-1]:
        req.history.append({"role": m["role"], "content": _text_of(m.get("content"))})
    content = convo[-1].get("content")
    if isinstance(content, str):
        req.text = content
        return req
    texts = []
    for part in content or []:
        kind = part.get("type")
        if kind == "text":
            texts.append(part.get("text", ""))
        elif kind == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            _, data = decode_data_url(url)
            req.image = prepare_image(data)
        elif kind == "input_audio":
            audio = part.get("input_audio") or {}
            if audio.get("format", "wav") != "wav":
                raise ValueError("only wav input_audio is supported")
            req.audio = normalize_speech(base64.b64decode(audio.get("data", "")))
        else:
            raise ValueError(f"unsupported content part: {kind}")
    req.text = "\n".join(t for t in texts if t)
    return req


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(p.get("text", "") for p in content or [] if p.get("type") == "text")


def create_app(engine):
    from fastapi import FastAPI, Header, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI(title="omni-pocket server")
    api_key = os.environ.get("OMNI_SERVER_API_KEY", "")
    gpu_lock = threading.Lock()  # one generation at a time: a 2B model saturates a consumer GPU

    def check(auth: str | None):
        if api_key and not hmac.compare_digest(auth or "", f"Bearer {api_key}"):
            raise HTTPException(status_code=401, detail="bad or missing API key")

    @app.get("/health")
    def health():
        return {"ok": True, "engine": engine.name, "model": engine.model}

    @app.get("/v1/models")
    def models(authorization: str | None = Header(default=None)):
        check(authorization)
        return {"object": "list", "data": [{"id": engine.model, "object": "model", "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    def chat(body: dict, authorization: str | None = Header(default=None)):
        check(authorization)
        try:
            req = parse_messages(body.get("messages") or [])
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        req.max_new_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 256)
        req.temperature = float(body.get("temperature", 0.3))
        req.want_speech = "audio" in (body.get("modalities") or [])
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if body.get("stream") and not req.want_speech:
            def events():
                with gpu_lock:
                    for piece in engine.stream(req):
                        chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                                 "model": engine.model,
                                 "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
                        yield f"data: {json.dumps(chunk)}\n\n"
                done = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": engine.model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                yield f"data: {json.dumps(done)}\n\ndata: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")

        with gpu_lock:
            res = engine.generate(req)
        message: dict = {"role": "assistant", "content": res.text}
        if res.audio_wav:
            message["audio"] = {"id": cid, "data": base64.b64encode(res.audio_wav).decode(), "transcript": res.text}
        return JSONResponse({
            "id": cid, "object": "chat.completion", "created": created, "model": engine.model,
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": res.prompt_tokens or 0, "completion_tokens": res.output_tokens or 0,
                      "total_tokens": (res.prompt_tokens or 0) + (res.output_tokens or 0)},
        })

    return app


def run(engine, host: str = "127.0.0.1", port: int = 8088) -> None:
    import uvicorn

    if hasattr(engine, "load"):
        print(f"loading {engine.model} ...", flush=True)
        engine.load()
    uvicorn.run(create_app(engine), host=host, port=port)
