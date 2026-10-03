import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import llm_client
from jev_client import JevError, validate_response
from questions import REVIEW, WILL_BOOK


def _reply(noul=0.7, usage=None):
    return {"choices": [{"message": {"content": json.dumps(
        {"answers": {"will_book": {"type": "noul", "noul": noul}}})}}],
        "usage": usage or {"prompt_tokens": 100, "completion_tokens": 50}}


def _server(handler_cls):
    srv = HTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _cfg(srv, tmp_path, **kw):
    d = dict(mode="live", api_key="test-key",
             url=f"http://127.0.0.1:{srv.server_address[1]}/x", fixtures=tmp_path)
    d.update(kw)
    return llm_client.Config(**d)


# prompts + parsing ---------------------------------------------------------------
def test_prompt_covers_every_question():
    msgs = llm_client.build_messages({"current_event": "x"}, REVIEW)
    blob = json.dumps(msgs)
    for name in REVIEW:
        assert name in blob
    assert "trace_so_far" not in blob  # state is passed separately, never embedded


def test_extract_json_tolerates_fences_and_prose():
    assert llm_client.extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm_client.extract_json('here you go {"a": 1} done') == {"a": 1}
    with pytest.raises(JevError):
        llm_client.extract_json("no json here")


def test_to_wire_rejects_non_dict_answers():
    with pytest.raises(JevError):
        llm_client.to_wire(["not", "a", "dict"], WILL_BOOK, "m", {})


def test_to_wire_backfills_missing_type():
    body = llm_client.to_wire({"answers": {"will_book": {"noul": 0.4}}}, WILL_BOOK, "m", {"prompt_tokens": 5})
    assert body["answers"]["will_book"]["type"] == "noul"
    assert validate_response(WILL_BOOK, body) == []


# mock ------------------------------------------------------------------------------
def test_mock_reports_jev_style_cost():
    r = llm_client.decide({"current_event": "call_started"}, WILL_BOOK,
                           cfg=llm_client.Config(mode="mock"))
    assert r["_cost_source"] == "tokens_x_price" and r["_mode"] == "mock"
    assert validate_response(WILL_BOOK, r) == []


def test_fixture_prefix_keeps_glm_apart_from_jev():
    assert llm_client.fixture_key("m", {"a": 1}, WILL_BOOK).startswith("glm_")


# live -------------------------------------------------------------------------------
def test_live_roundtrip_and_token_cost(tmp_path):
    seen = {}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.update(body)
            assert self.headers["Authorization"] == "Bearer test-key"
            out = json.dumps(_reply()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

    srv = _server(H)
    try:
        r = llm_client.decide({"s": 1}, WILL_BOOK, cfg=_cfg(srv, tmp_path))
    finally:
        srv.shutdown()
    assert seen["model"] == "z-ai/glm-5.3" and seen["reasoning"] == {"effort": "low"}
    assert r["answers"]["will_book"]["noul"] == 0.7
    assert r["_cost_usd"] == pytest.approx(100 * llm_client.PRICE_IN_PER_TOKEN + 50 * llm_client.PRICE_OUT_PER_TOKEN)
    assert r["_cost_source"] == "glm_tokens_x_price"


def test_live_prefers_usage_cost(tmp_path):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            out = json.dumps(_reply(usage={"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.5})).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

    srv = _server(H)
    try:
        r = llm_client.decide({"s": 1}, WILL_BOOK, cfg=_cfg(srv, tmp_path))
    finally:
        srv.shutdown()
    assert (r["_cost_usd"], r["_cost_source"]) == (0.5, "usage.cost")


def test_fixup_repairs_bad_json(tmp_path):
    calls = {"n": 0}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            calls["n"] += 1
            self.rfile.read(int(self.headers["Content-Length"]))
            content = "oops not json" if calls["n"] == 1 else json.dumps(
                {"answers": {"will_book": {"type": "noul", "noul": 0.2}}})
            out = json.dumps({"choices": [{"message": {"content": content}}],
                              "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

    srv = _server(H)
    try:
        r = llm_client.decide({"s": 1}, WILL_BOOK, cfg=_cfg(srv, tmp_path))
    finally:
        srv.shutdown()
    assert calls["n"] == 2 and r["answers"]["will_book"]["noul"] == 0.2


def test_record_and_replay(tmp_path):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            out = json.dumps(_reply(0.9)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

    srv = _server(H)
    try:
        rec = llm_client.decide({"s": 1}, WILL_BOOK, cfg=_cfg(srv, tmp_path, mode="record"))
        files = list(tmp_path.glob("glm_*.json"))
        assert len(files) == 1
        rep = llm_client.decide({"s": 1}, WILL_BOOK,
                                cfg=llm_client.Config(mode="replay", fixtures=tmp_path))
    finally:
        srv.shutdown()
    assert rep["answers"] == rec["answers"]
    with pytest.raises(JevError, match="replay miss"):
        llm_client.decide({"never": "seen"}, WILL_BOOK, cfg=llm_client.Config(mode="replay", fixtures=tmp_path))


def test_config_rejects_unknown_mode():
    with pytest.raises(JevError):
        llm_client.Config(mode="bogus")
