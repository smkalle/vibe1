"""Task presets: practical phone use-cases mapped to a system prompt and the
inputs they expect. Picked in the sidebar of the Ask tab."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    key: str
    title: str
    system: str
    needs: tuple[str, ...]  # suggested inputs: image / audio / text
    hint: str
    speak: bool = False  # read the answer aloud by default


PRESETS: tuple[Preset, ...] = (
    Preset("ask", "Ask anything",
           "You are a concise, helpful on-device assistant. Answer in a few sentences.",
           ("text",), "Type, speak, or attach a photo."),
    Preset("look", "Spoken visual Q&A",
           "Answer the spoken question about the provided image. Respond with text.",
           ("image", "audio"), "Take a photo, then record your question.", speak=True),
    Preset("describe", "Describe my surroundings",
           "Describe the image for a blind or low-vision user: main objects, people, text, hazards. "
           "Lead with what matters most. Max 4 sentences.",
           ("image",), "Point the camera and capture.", speak=True),
    Preset("read", "Read text / label / receipt",
           "Transcribe all legible text in the image exactly, preserving line breaks. "
           "Then give a one-line summary (e.g. total, expiry date, dosage).",
           ("image",), "Photograph a label, sign, menu or receipt."),
    Preset("translate", "Translate a sign or menu",
           "Read the text in the image and translate it into {lang}. "
           "Output: original line -> translation, one per line.",
           ("image",), "Set the target language below."),
    Preset("transcribe", "Voice note -> clean notes",
           "Transcribe the audio, then rewrite it as tidy bullet-point notes with any action items at the end.",
           ("audio",), "Record a voice memo."),
    Preset("fix", "What's wrong here?",
           "Diagnose the problem shown in the image (plant, appliance, error screen, bike, etc.). "
           "List likely causes, then the safest next step.",
           ("image", "text"), "Photo + a short description."),
    Preset("cook", "What can I cook?",
           "List the ingredients visible in the image, then suggest two simple recipes using mostly those.",
           ("image",), "Photo of your fridge or pantry."),
)

BY_KEY = {p.key: p for p in PRESETS}


def system_prompt(preset: Preset, lang: str = "English", speak: bool = False) -> str:
    text = preset.system.replace("{lang}", lang or "English")
    if speak:
        text += " Keep it short and easy to listen to; no markdown or tables."
    return text
