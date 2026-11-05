"""Image / audio helpers that only need Pillow + numpy + stdlib (all available
as prebuilt packages on Termux), so no librosa / soundfile / ffmpeg needed."""

from __future__ import annotations

import base64
import io
import wave
from dataclasses import dataclass

import numpy as np

TARGET_SR = 16_000  # the audio encoder's expected rate
MAX_IMAGE_SIDE = 768  # phone photos are 12+ MP; the vision encoder tiles far smaller


@dataclass
class Audio:
    samples: np.ndarray  # float32 mono in [-1, 1]
    sample_rate: int

    @property
    def seconds(self) -> float:
        return len(self.samples) / float(self.sample_rate or 1)


def prepare_image(data: bytes, max_side: int = MAX_IMAGE_SIDE):
    """Decode, fix EXIF rotation, convert to RGB and downscale. Returns PIL.Image."""
    from PIL import Image, ImageOps

    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((max_side, max_side))
    return img


def image_to_jpeg(img, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def read_wav(data: bytes) -> Audio:
    """Parse 8/16/24/32-bit PCM WAV (what st.audio_input and browsers produce)."""
    with wave.open(io.BytesIO(data), "rb") as w:
        n_ch, width, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        i = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
        i = np.where(i & 0x800000, i - 0x1000000, i)
        x = i.astype(np.float32) / 8388608.0
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width: {width}")
    if n_ch > 1:
        x = x.reshape(-1, n_ch).mean(axis=1)
    return Audio(x.astype(np.float32), sr)


def resample(audio: Audio, target_sr: int = TARGET_SR) -> Audio:
    """Linear-interpolation resampler. Good enough for speech at 16 kHz."""
    if audio.sample_rate == target_sr or len(audio.samples) == 0:
        return Audio(audio.samples, target_sr if len(audio.samples) == 0 else audio.sample_rate)
    n_out = int(round(len(audio.samples) * target_sr / audio.sample_rate))
    t_in = np.arange(len(audio.samples)) / audio.sample_rate
    t_out = np.arange(n_out) / target_sr
    return Audio(np.interp(t_out, t_in, audio.samples).astype(np.float32), target_sr)


def to_wav_bytes(audio: Audio) -> bytes:
    pcm = (np.clip(audio.samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(audio.sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def normalize_speech(data: bytes) -> Audio:
    """WAV bytes -> mono float32 at 16 kHz, peak-normalised, silence-trimmed."""
    audio = resample(read_wav(data))
    x = audio.samples
    if len(x):
        peak = float(np.max(np.abs(x)))
        if peak > 1e-4:
            x = x / peak * 0.95
        # trim leading/trailing near-silence (keeps 100 ms padding)
        loud = np.flatnonzero(np.abs(x) > 0.02)
        if len(loud):
            pad = TARGET_SR // 10
            x = x[max(0, loud[0] - pad): loud[-1] + pad]
    return Audio(x.astype(np.float32), TARGET_SR)


def data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def decode_data_url(url: str) -> tuple[str, bytes]:
    if not url.startswith("data:") or "," not in url:
        raise ValueError("not a base64 data URL")
    header, payload = url.split(",", 1)
    mime = header[5:].split(";")[0] or "application/octet-stream"
    return mime, base64.b64decode(payload)
