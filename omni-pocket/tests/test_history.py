from omnipocket.history import Log, logs, summarize


def test_log_append_read_and_torn_line(tmp_path):
    log = Log(tmp_path / "x" / "h.jsonl")
    log.append({"a": 1})
    log.append({"a": 2})
    with log.path.open("a") as f:
        f.write('{"broken')
    rows = log.read()
    assert [r["a"] for r in rows] == [1, 2] and all("ts" in r for r in rows)
    assert log.read(limit=1)[0]["a"] == 2
    log.clear()
    assert log.read() == []


def test_summarize_medians(tmp_path):
    _, bench = logs(tmp_path)
    for t in (1.0, 2.0, 9.0):
        bench.append({"engine": "remote", "model": "m", "mode": "text", "ttft_s": t, "total_s": t * 2,
                      "tokens_per_s": 10 * t})
    bench.append({"engine": "webgpu", "model": "onnx", "mode": "image", "ttft_s": None, "total_s": 1.0})
    s = summarize(bench.read())
    assert s["remote | m | text"] == {"runs": 3, "ttft_s": 2.0, "total_s": 4.0, "tokens_per_s": 20.0}
    assert s["webgpu | onnx | image"]["ttft_s"] is None
