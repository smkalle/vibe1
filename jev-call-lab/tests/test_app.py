"""W12: the workbench's full mock-mode happy path, driven headlessly with Streamlit AppTest."""
from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest
APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _button(at, key=None, label=None):
    for b in at.button:
        if (key and b.key == key) or (label and b.label == label):
            return b
    raise AssertionError(f"no button {key or label}: {[(b.key, b.label) for b in at.button]}")


def _step(at, i):
    at.radio(key="step").set_value(at.radio(key="step").options[i]).run()
    _ok(at)


def test_w12_full_workflow_in_mock_mode():
    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state["jev_mode"] = "mock"
    at.session_state["glm_mode"] = "mock"
    at.run()
    _ok(at)

    # ① Connect: test both approaches; the key survives navigation (widget-state persistence)
    _button(at, key="test_jev").click().run()
    _ok(at)
    assert any("contract OK" in s.value for s in at.success)
    at.text_input(key="api_key").set_value("sk-or-v1-" + "c" * 64).run()

    # ② Corpus
    _step(at, 1)
    at.radio(key="corpus_id").set_value("synthetic").run()
    _ok(at)
    assert any("ranking accuracy before the outcome" in m.value for m in at.markdown)

    # ③ Plan: freeze a 40-call sample
    _step(at, 2)
    at.slider(key="plan_n").set_value(40).run()
    _button(at, label="Freeze sample").click().run()
    _ok(at)
    sid = at.session_state["sample_id"]
    assert "-n40-" in sid
    assert at.session_state["api_key"].startswith("sk-or-v1-")  # still set after leaving step ①

    # ④ Run both LLM approaches (mock) and a probe
    _step(at, 3)
    _button(at, key="run_jev").click().run()
    _button(at, key="run_glm").click().run()
    _ok(at)
    import workspace
    for a in ("jev", "glm"):
        m = workspace.Project("default").latest_run(sid, a)
        assert (m["status"], m["n_done"], m["mode"]) == ("done", 40, "mock")
    _button(at, key="probe_jev").click().run()
    _ok(at)
    assert any("No network timing" in i.value for i in at.info)  # mock never counts as latency (W8)

    # ⑤ Compare, with technical details
    at.session_state["tech"] = True
    _step(at, 4)
    assert any(m.value.startswith("Use ") for m in at.markdown)

    # ⑥ Decide: gate passes (no key on disk), decision saved, report exported
    _step(at, 5)
    assert any("Gate: PASS" in s.value for s in at.success)
    _button(at, key="save_in_call").click().run()
    _ok(at)
    assert workspace.Project("default").meta()["decisions"][0]["use_case"] == "in_call"

    # Summary view for business readers
    at.radio(key="view").set_value("Summary").run()
    _ok(at)
    assert at.title[0].value == "Which approach should we use?"

    # Tools
    at.radio(key="view").set_value("Workflow").run()
    _step(at, 6)


def test_w2_new_sample_recomputes_the_scorecard():
    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state["jev_mode"] = "mock"
    at.session_state["glm_mode"] = "mock"
    at.session_state["approaches"] = ["rules_b1", "rules_b2"]
    at.run()
    _step(at, 2)
    at.slider(key="plan_n").set_value(30).run()
    _button(at, label="Freeze sample").click().run()
    _step(at, 4)
    first = [k for k in at.session_state["sc_cache"]][0][0]
    _step(at, 2)
    at.session_state["plan_seed"] = 99
    at.slider(key="plan_n").set_value(31).run()
    _button(at, label="Freeze sample").click().run()
    _step(at, 4)
    second = [k for k in at.session_state["sc_cache"]][0][0]
    assert first != second and second == at.session_state["sample_id"]
