import base64
import socket
import threading
import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from omnipocket.engines import MockEngine, OmniRequest, RemoteEngine  # noqa: E402
from omnipocket.engines.remote import build_messages  # noqa: E402
from omnipocket.media import normalize_speech, prepare_image  # noqa: E402
from omnipocket.serve import create_app, parse_messages  # noqa: E402
from tests.conftest import make_jpeg, make_wav  # noqa: E402


def media_request():
    return OmniRequest(text="describe", image=prepare_image(make_jpeg(100, 50)),
                       audio=normalize_speech(make_wav(seconds=0.4)))


def test_parse_messages_roundtrip():
    req = parse_messages(build_messages(media_request()))
    assert req.text == "describe" and req.image.size == (100, 50) and req.audio.sample_rate == 16_000


@pytest.mark.parametrize("bad", [[], [{"role": "assistant", "content": "x"}],
                                 [{"role": "user", "content": [{"type": "video"}]}]])
def test_parse_messages_rejects(bad):
    with pytest.raises(ValueError):
        parse_messages(bad)


def test_chat_non_stream_and_stream():
    client = TestClient(create_app(MockEngine()))
    assert client.get("/health").json()["engine"] == "mock"
    body = {"model": "x", "messages": build_messages(media_request()), "max_tokens": 50}
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 200
    assert "100x50" in r.json()["choices"][0]["message"]["content"]
    r = client.post("/v1/chat/completions", json=body | {"stream": True})
    assert r.status_code == 200 and r.text.rstrip().endswith("data: [DONE]")


def test_bad_request_is_400():
    client = TestClient(create_app(MockEngine()))
    r = client.post("/v1/chat/completions", json={"messages": []})
    assert r.status_code == 400


def test_api_key_required(monkeypatch):
    monkeypatch.setenv("OMNI_SERVER_API_KEY", "k1")
    client = TestClient(create_app(MockEngine()))
    msg = {"messages": [{"role": "user", "content": "hi"}]}
    assert client.post("/v1/chat/completions", json=msg).status_code == 401
    ok = client.post("/v1/chat/completions", json=msg, headers={"Authorization": "Bearer k1"})
    assert ok.status_code == 200


class SpeakingMock(MockEngine):
    def generate(self, req):
        res = super().generate(req)
        res.audio_wav = b"RIFF0000WAVE"
        return res


def test_speech_modality_returns_audio():
    client = TestClient(create_app(SpeakingMock()))
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}],
                                                   "modalities": ["text", "audio"], "stream": True})
    msg = r.json()["choices"][0]["message"]
    assert base64.b64decode(msg["audio"]["data"]) == b"RIFF0000WAVE"


def test_remote_engine_against_live_server():
    """Real HTTP round-trip: phone client -> uvicorn server -> mock engine."""
    uvicorn = pytest.importorskip("uvicorn")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(MockEngine()), host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        eng = RemoteEngine(f"http://127.0.0.1:{port}/v1", "mock-omni")
        assert eng.health()["ok"] is True
        from omnipocket.config import Settings, server_alive

        assert server_alive(f"http://127.0.0.1:{port}/v1")
        s = Settings(engine="auto", llama_url=f"http://127.0.0.1:{port}/v1")
        assert s.resolve_engine() == "llamacpp"
        res = eng.generate(media_request())
        assert "100x50" in res.text and res.ttft_s is not None and res.output_tokens > 3
    finally:
        server.should_exit = True
        t.join(timeout=5)
