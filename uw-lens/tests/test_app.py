"""Headless UI tests with Streamlit AppTest: generate -> index -> search -> pin -> export, plus RAG
against a real local HTTP server that mimics Ollama's /api/chat."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest
APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _results(at):
    return [m.value for m in at.markdown if m.value.startswith("**[")]


def _indexed_app():
    at = AppTest.from_file(APP, default_timeout=180).run()
    _ok(at)
    at.button(key="make_sample").click().run()
    _ok(at)
    at.button(key="index_go").click().run()
    _ok(at)
    return at


def test_empty_index_prompts_user():
    at = AppTest.from_file(APP, default_timeout=60).run()
    _ok(at)
    assert any("index is empty" in i.value for i in at.info)
    assert any("hashing" in m.value.lower() for m in at.markdown)  # active embedder is visible


def test_full_workflow():
    at = _indexed_app()
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Indexed files"] == "19" and metrics["Files"] == "19"  # sidebar refreshed after indexing

    at.text_input(key="query").set_value("CPAP compliance moderate sleep apnea").run()
    _ok(at)
    res = _results(at)
    assert len(res) == 8 and any("sleep apnea" in r.lower() for r in res[:3])

    # filter to Japanese sources, Japanese example query
    at.button(key="ex_3").click().run()
    _ok(at)
    assert at.session_state["query"] == "高額療養費 自己負担限度額"
    assert "高額療養費制度の概要" in _results(at)[0]
    at.multiselect(key="f_src").set_value(["japan"]).run()
    _ok(at)
    assert all("Japan public insurance" in c.value for c in at.caption if "· `" in c.value)

    # pin, annotate and export
    at.multiselect(key="f_src").set_value([]).run()
    pin = next(b for b in at.button if b.key and b.key.startswith("pin_"))
    pin.click().run()
    _ok(at)
    pack = at.session_state["pack"]
    assert len(pack.items) == 1
    at.text_input(key="case_ref").set_value("UWR-0417").run()
    note_key = f"note_{pack.items[0].id}"
    at.text_input(key=note_key).set_value("Limit applies to monthly cost").run()
    _ok(at)
    md = at.session_state["pack"].to_markdown()
    assert "UWR-0417" in md and "Limit applies to monthly cost" in md

    # incremental re-index reports everything unchanged
    at.button(key="index_go").click().run()
    _ok(at)
    assert {m.label: m.value for m in at.metric}["Unchanged"] == "19"


def test_phi_not_shown_in_results():
    at = _indexed_app()
    at.text_input(key="query").set_value("attending physician statement applicant phone email").run()
    _ok(at)
    page = " ".join(m.value for m in at.markdown)
    assert "Attending Physician Statement" in page  # the APS itself is retrieved...
    for secret in ("090-1234-5678", "1234-5678", "taro.example@example.jp", "Taro Example", "UW-2026-00417",
                   "1971-03-14"):
        assert secret not in page, secret  # ...but none of its identifiers are displayed
    assert "PHONE" in page and "EMAIL" in page


class _FakeOllama(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.seen.append(body)
        out = json.dumps({"message": {"content": "CPAP use meets the 70% threshold [1]. Verify download dates [2]."}})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out.encode())

    def log_message(self, *a):
        pass


@pytest.fixture()
def fake_ollama():
    srv = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    srv.seen = []
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv
    srv.shutdown()


def test_rag_is_off_by_default_and_cites_when_on(fake_ollama):
    at = _indexed_app()
    at.text_input(key="query").set_value("CPAP compliance").run()
    assert not [b for b in at.button if b.key == "rag_go"]  # A6: off by default
    at.toggle(key="use_rag").set_value(True).run()
    at.text_input(key="ollama_url").set_value(f"http://127.0.0.1:{fake_ollama.server_port}").run()
    at.button(key="rag_go").click().run()
    _ok(at)
    assert fake_ollama.seen and fake_ollama.seen[0]["stream"] is False
    page = " ".join(m.value for m in at.markdown)
    assert "Research summary" in page and "70% threshold" in page
