"""OpenAI-compatible client. Talks to ``python -m omnipocket serve`` on a PC,
or to any server that accepts OpenAI multimodal chat messages
(``image_url`` data URLs + ``input_audio`` WAV parts).

Also the client for llama.cpp's ``llama-server`` (engine ``llamacpp``), which
accepts the same ``image_url`` / ``input_audio`` parts. Speech output there
comes from its ``/tts`` endpoint.

Only ``requests`` is needed, so this is the lightest way to use a real model
from Termux."""

from __future__ import annotations

import json
import time
from typing import Iterator

import requests

from ..media import b64, data_url, image_to_jpeg, to_wav_bytes
from .base import OmniRequest, OmniResult, StreamTimer


def build_messages(req: OmniRequest) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": req.system}]
    for turn in req.history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    content: list[dict] = []
    if req.image is not None:
        content.append({"type": "image_url", "image_url": {"url": data_url(image_to_jpeg(req.image), "image/jpeg")}})
    if req.audio is not None:
        content.append({"type": "input_audio", "input_audio": {"data": b64(to_wav_bytes(req.audio)), "format": "wav"}})
    if req.text.strip():
        content.append({"type": "text", "text": req.text.strip()})
    messages.append({"role": "user", "content": content})
    return messages


class RemoteEngine:
    name = "remote"

    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 300.0,
                 session: requests.Session | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self.timeout = timeout
        self.http = session or requests.Session()

    def __repr__(self) -> str:  # never show the key
        return f"RemoteEngine(base_url={self.base_url!r}, model={self.model!r})"

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def _payload(self, req: OmniRequest, stream: bool) -> dict:
        body = {
            "model": self.model,
            "messages": build_messages(req),
            "max_tokens": req.max_new_tokens,
            "temperature": req.temperature,
            "stream": stream,
        }
        if req.want_speech:
            body["modalities"] = ["text", "audio"]
            body["audio"] = {"format": "wav", "voice": "default"}
        return body

    @property
    def root(self) -> str:
        return self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url

    def health(self) -> dict:
        r = self.http.get(f"{self.root}/health", timeout=5)
        r.raise_for_status()
        return r.json()

    def stream(self, req: OmniRequest) -> Iterator[str]:
        r = self.http.post(f"{self.base_url}/chat/completions", headers=self._headers(),
                           json=self._payload(req, stream=True), timeout=self.timeout, stream=True)
        if r.status_code >= 400:
            raise RuntimeError(f"remote error {r.status_code}: {r.text[:300]}")
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            data = raw[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            for choice in chunk.get("choices", []):
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    yield piece

    def generate(self, req: OmniRequest) -> OmniResult:
        if not req.want_speech:
            timer = StreamTimer(self.stream(req))
            for _ in timer:
                pass
            return OmniResult(text=timer.text, engine=self.name, model=self.model,
                              output_tokens=timer.chunks, ttft_s=timer.ttft, total_s=timer.elapsed)
        t0 = time.perf_counter()
        r = self.http.post(f"{self.base_url}/chat/completions", headers=self._headers(),
                           json=self._payload(req, stream=False), timeout=self.timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"remote error {r.status_code}: {r.text[:300]}")
        body = r.json()
        msg = body["choices"][0]["message"]
        audio = None
        if isinstance(msg.get("audio"), dict) and msg["audio"].get("data"):
            import base64

            audio = base64.b64decode(msg["audio"]["data"])
        usage = body.get("usage") or {}
        text = msg.get("content") or (msg.get("audio") or {}).get("transcript", "")
        res = OmniResult(text=text, engine=self.name, model=self.model, audio_wav=audio,
                         prompt_tokens=usage.get("prompt_tokens"),
                         output_tokens=usage.get("completion_tokens"),
                         total_s=time.perf_counter() - t0)
        if res.audio_wav is None and text:
            res.audio_wav, note = self.tts(text)
            if note:
                res.notes.append(note)
        return res

    def tts(self, text: str, voice: str | None = None) -> tuple[bytes | None, str | None]:
        """llama-server ``POST /tts`` -> WAV bytes (LFM3 speech decoder). Returns
        (None, note) when the server has no speech output."""
        body = {"input": text[:2000]}
        if voice:
            body["voice"] = voice
        try:
            r = self.http.post(f"{self.root}/tts", headers=self._headers(), json=body, timeout=self.timeout)
        except requests.RequestException as exc:
            return None, f"speech unavailable ({exc.__class__.__name__}) - using phone TTS"
        if r.status_code >= 400 or not r.content[:4] == b"RIFF":
            return None, f"speech unavailable (/tts {r.status_code}) - using phone TTS"
        return r.content, None
