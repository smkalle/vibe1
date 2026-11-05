import numpy as np
import pytest

from omnipocket.media import (Audio, data_url, decode_data_url, image_to_jpeg, normalize_speech,
                              prepare_image, read_wav, resample, to_wav_bytes)
from tests.conftest import make_jpeg, make_wav


@pytest.mark.parametrize("width", [2, 3, 4])
def test_read_wav_widths_and_downmix(width):
    a = read_wav(make_wav(seconds=0.5, sr=22_050, channels=2, width=width))
    assert a.sample_rate == 22_050
    assert len(a.samples) == 11_025
    assert 0.45 < np.max(np.abs(a.samples)) < 0.55


def test_resample_preserves_duration_and_pitch():
    a = read_wav(make_wav(seconds=1.0, sr=44_100, channels=1, freq=440))
    r = resample(a, 16_000)
    assert r.sample_rate == 16_000 and abs(len(r.samples) - 16_000) <= 1
    spectrum = np.abs(np.fft.rfft(r.samples))
    peak_hz = np.argmax(spectrum) * 16_000 / len(r.samples)
    assert abs(peak_hz - 440) < 3


def test_normalize_speech_trims_silence_and_peaks():
    tone = read_wav(make_wav(seconds=1.0, sr=16_000, channels=1)).samples * 0.2
    padded = np.concatenate([np.zeros(16_000), tone, np.zeros(16_000)]).astype(np.float32)
    out = normalize_speech(to_wav_bytes(Audio(padded, 16_000)))
    assert out.sample_rate == 16_000
    assert 1.0 <= out.seconds < 1.3
    assert np.max(np.abs(out.samples)) == pytest.approx(0.95, abs=0.01)


def test_wav_roundtrip():
    a = Audio(np.linspace(-1, 1, 1000, dtype=np.float32), 16_000)
    b = read_wav(to_wav_bytes(a))
    assert np.allclose(a.samples, b.samples, atol=1e-4)


def test_prepare_image_downscales_and_rgb():
    img = prepare_image(make_jpeg(1600, 1200))
    assert img.mode == "RGB" and max(img.size) == 768 and img.size == (768, 576)


def test_data_url_roundtrip():
    payload = image_to_jpeg(prepare_image(make_jpeg(64, 64)))
    mime, data = decode_data_url(data_url(payload, "image/jpeg"))
    assert mime == "image/jpeg" and data == payload
    with pytest.raises(ValueError):
        decode_data_url("https://example.com/x.png")
