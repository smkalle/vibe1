import io
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Never touch the real data dir, network endpoints or keys."""
    monkeypatch.setenv("OMNI_DATA_DIR", str(tmp_path / "omni_data"))
    monkeypatch.setenv("OMNI_ENGINE", "mock")
    for var in ("OMNI_REMOTE_URL", "OMNI_REMOTE_API_KEY", "OMNI_SERVER_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def make_wav(seconds=1.0, sr=44_100, channels=2, width=2, freq=440.0) -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    x = 0.5 * np.sin(2 * np.pi * freq * t)
    x = np.stack([x] * channels, axis=1).reshape(-1)
    if width == 2:
        raw = (x * 32767).astype("<i2").tobytes()
    elif width == 3:
        i = (x * 8388607).astype("<i4")
        raw = b"".join(int(v).to_bytes(3, "little", signed=True) for v in i)
    else:
        raw = (x * 2147483647).astype("<i4").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(sr)
        w.writeframes(raw)
    return buf.getvalue()


def make_jpeg(w=1600, h=1200, color=(200, 40, 40)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="JPEG")
    return buf.getvalue()
