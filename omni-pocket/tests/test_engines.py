import json

import pytest

from omnipocket.config import Settings
from omnipocket.engines import MockEngine, OmniRequest, RemoteEngine, make_engine
from omnipocket.engines.remote import build_messages
from omnipocket.media import normalize_speech, prepare_image
from tests.conftest import make_jpeg, make_wav


def full_request(**kw):
    return OmniRequest(text="what is this?", image=prepare_image(make_jpeg(320, 240)),
                       audio=normalize_speech(make_wav(seconds=0.5)), **kw)


def test_mock_engine_mentions_every_modality():
    res = MockEngine().generate(full_request())
    assert "320x240" in res.text and "audio" in res.text and "what is this?" in res.text
    assert res.output_tokens and res.ttft_s is not None and res.engine == "mock"


def test_request_modalities_and_empty():
    assert OmniRequest().is_empty()
    assert full_request().modalities() == ["text", "image", "audio"]


def test_auto_engine_resolution(monkeypatch):
    s = Settings.from_env()
    assert s.resolve_engine() == "mock"  # conftest pins OMNI_ENGINE=mock
    s.engine = "auto"
    monkeypatch.setattr("omnipocket.platform.has_module", lambda m: False)
    assert s.resolve_engine() == "mock"
    monkeypatch.setenv("OMNI_REMOTE_URL", "http://pc:8088/v1")
    assert s.resolve_engine() == "remote"


def test_make_engine_rejects_unknown():
    with pytest.raises(ValueError):
        make_engine("nope", Settings())


def test_build_messages_openai_shape():
    req = full_request(history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
    msgs = build_messages(req)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    parts = msgs[-1]["content"]
    assert [p["type"] for p in parts] == ["image_url", "input_audio", "text"]
    assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert parts[1]["input_audio"]["format"] == "wav"


class FakeResponse:
    def __init__(self, lines=None, body=None, status=200):
        self.lines, self.body, self.status_code, self.text = lines or [], body, status, "err"

    def iter_lines(self, decode_unicode=True):
        yield from self.lines

    def json(self):
        return self.body


class FakeSession:
    def __init__(self, resp):
        self.resp, self.calls = resp, []

    def post(self, url, headers=None, json=None, timeout=None, stream=False):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return self.resp


def sse(*pieces):
    out = []
    for p in pieces:
        out.append("data: " + json.dumps({"choices": [{"delta": {"content": p}}]}))
        out.append("")
    return out + ["data: [DONE]"]


def test_remote_stream_and_auth_header():
    sess = FakeSession(FakeResponse(lines=sse("Hel", "lo")))
    eng = RemoteEngine("http://pc:8088/v1/", "m", api_key="sekret", session=sess)
    res = eng.generate(OmniRequest(text="hi"))
    assert res.text == "Hello" and res.output_tokens == 2
    call = sess.calls[0]
    assert call["url"] == "http://pc:8088/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer sekret"
    assert call["json"]["stream"] is True
    assert "sekret" not in repr(eng)


def test_remote_speech_mode_decodes_audio():
    import base64

    body = {"choices": [{"message": {"content": "hi", "audio": {"data": base64.b64encode(b"RIFFxx").decode()}}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1}}
    sess = FakeSession(FakeResponse(body=body))
    res = RemoteEngine("http://pc/v1", "m", session=sess).generate(OmniRequest(text="x", want_speech=True))
    assert res.audio_wav == b"RIFFxx" and res.prompt_tokens == 5
    assert sess.calls[0]["json"]["modalities"] == ["text", "audio"]


def test_remote_error_surfaces():
    sess = FakeSession(FakeResponse(status=500))
    with pytest.raises(RuntimeError, match="500"):
        RemoteEngine("http://pc/v1", "m", session=sess).generate(OmniRequest(text="x"))


def test_transformers_engine_conversation_without_loading(tmp_path):
    from omnipocket.engines.transformers_engine import TransformersEngine

    eng = TransformersEngine("LiquidAI/LFM3-Omni-2B")
    req = full_request(history=[{"role": "user", "content": "earlier"}])
    conv = eng._conversation(req, tmp_path)
    assert conv[0] == {"role": "system", "content": req.system}
    assert conv[1]["content"] == [{"type": "text", "text": "earlier"}]
    kinds = [p["type"] for p in conv[-1]["content"]]
    assert kinds == ["image", "audio", "text"]
    assert (tmp_path / "image.jpg").exists() and (tmp_path / "audio.wav").exists()
    assert eng._gen_kwargs(OmniRequest(temperature=0))["generation_mode"] == "text"
