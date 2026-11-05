import json

from omnipocket.cli import main
from tests.conftest import make_jpeg


def test_doctor_json(capsys):
    main(["doctor", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert {"env", "total_ram_mb", "modules", "notes"} <= set(data)


def test_ask_mock_with_image(tmp_path, capsys):
    img = tmp_path / "p.jpg"
    img.write_bytes(make_jpeg(400, 300))
    main(["ask", "--engine", "mock", "--text", "hi", "--image", str(img)])
    assert "400x300" in capsys.readouterr().out


def test_bench_logs_runs(tmp_path, capsys):
    main(["bench", "--engine", "mock", "--text", "hi", "--runs", "2", "--warmup", "1"])
    out = capsys.readouterr().out
    assert out.count("run ") == 2 and '"runs": 2' in out
