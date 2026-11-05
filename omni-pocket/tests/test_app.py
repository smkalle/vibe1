"""Streamlit AppTest smoke tests (mock engine, no browser)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def run_app():
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def test_app_boots_with_mock_engine():
    at = run_app()
    assert at.title[0].value.endswith("Omni Pocket")
    assert at.sidebar.selectbox[0].value == "mock"


def test_ask_text_logs_history_and_bench():
    at = run_app()
    at.radio[0].set_value("none")
    at.text_area[0].input("what is 2+2? costs $5")
    at.button[0].click().run()
    assert not at.exception, at.exception
    msgs = [m for m in at.chat_message]
    assert len(msgs) >= 2
    from omnipocket.config import Settings
    from omnipocket.history import logs

    hist, bench = logs(Settings.from_env().data_dir)
    assert hist.read()[-1]["engine"] == "mock"
    assert bench.read()[-1]["source"] == "ask"


def test_empty_ask_warns():
    at = run_app()
    at.radio[0].set_value("none")
    at.button[0].click().run()
    assert any("Add a photo" in w.value for w in at.warning)


def test_benchmarks_tab_after_run():
    at = run_app()
    at.radio[0].set_value("none")
    at.text_area[0].input("hello")
    at.button[0].click().run()
    at.run()
    assert not at.exception, at.exception
    assert any("Median" in s.value for s in at.subheader)
