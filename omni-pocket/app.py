"""Omni Pocket - Streamlit UI for LiquidAI LFM3-Omni-2B on an Android phone.

    streamlit run app.py          # then open http://localhost:8501 in Chrome
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import time

import streamlit as st

from omnipocket import __version__
from omnipocket.config import DEFAULT_GGUF, ENGINES, OFFICIAL_WEBGPU_DEMO, Settings, server_alive
from omnipocket.engines import OmniRequest, StreamTimer, make_engine
from omnipocket.history import logs, summarize
from omnipocket.media import normalize_speech, prepare_image
from omnipocket.platform import report
from omnipocket.presets import BY_KEY, PRESETS, system_prompt
from omnipocket.webgpu import webgpu_panel

st.set_page_config(page_title="Omni Pocket", page_icon="🎙️", layout="centered",
                   initial_sidebar_state="collapsed")

BASE = Settings.from_env()
HISTORY, BENCH = logs(BASE.data_dir)


def md(text: str) -> str:
    """Model text -> safe Markdown ($ would render as LaTeX)."""
    return text.replace("$", "\\$")


# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.header("Engine")
    default_engine = BASE.resolve_engine()
    engine_name = st.selectbox("Python engine (Ask tab)", ENGINES, index=ENGINES.index(default_engine),
                               help="llamacpp = llama.cpp server on this phone (recommended on Termux) · "
                                    "mock = offline demo · remote = your GPU box / any OpenAI-compatible server · "
                                    "transformers = run the model in this Python (UserLAnd / desktop).")
    settings = Settings.from_env()
    if engine_name == "llamacpp":
        settings.llama_url = st.text_input("llama-server URL", BASE.llama_url)
        if server_alive(settings.llama_url, timeout=1.0):
            st.success("llama-server is running", icon="✅")
        else:
            st.warning("llama-server is not running. In Termux:", icon="⚠️")
            st.code("./scripts/llama_server.sh", language="bash")
            st.caption(f"Downloads `{DEFAULT_GGUF}` (+ mmproj) once, then runs offline. "
                       "Or start everything with `./scripts/run.sh --llama`.")
    if engine_name == "remote":
        settings.remote_url = st.text_input("Server URL", BASE.remote_url,
                                            help="e.g. http://192.168.1.20:8088/v1 or a Tailscale name")
        settings.remote_model = st.text_input("Model name", BASE.remote_model)
        key_in = st.text_input("API key (optional)", type="password",
                               help="Kept in this session's memory only, never written to disk.")
        settings.remote_api_key = key_in or BASE.remote_api_key
    if engine_name == "transformers":
        settings.model_id = st.text_input("HF model id", BASE.model_id)
        settings.device = st.selectbox("Device", ["auto", "cpu", "cuda", "mps"])
        settings.dtype = st.selectbox("dtype", ["auto", "bfloat16", "float16", "float32"],
                                      help="auto = bfloat16 (~4.4 GB). float32 needs ~8.7 GB RAM.")
    max_new = st.slider("Max new tokens", 32, 1024, BASE.max_new_tokens, 32)
    temperature = st.slider("Temperature", 0.0, 1.2, 0.3, 0.1)

    st.header("Speech output")
    has_termux_tts = shutil.which("termux-tts-speak") is not None
    speech_modes = ["off", "phone TTS (browser)"] + (["termux-tts-speak"] if has_termux_tts else []) + ["model voice (experimental)"]
    speech_mode = st.selectbox("Read answers aloud with", speech_modes, index=1)

    st.header("In-browser (WebGPU)")
    onnx_id = st.text_input("ONNX model id", BASE.onnx_model_id)
    web_precision = st.selectbox(
        "Decoder", ["auto", "q4f16", "q4"],
        help="auto = q4f16 when the GPU has shader-f16, else q4. Both use the q4 bundle's side graphs "
             "(vision_encoder_q8, audio_encoder), as LiquidAI's onnx-export README 4.6 recommends.")
    max_image_tokens = st.select_slider("Max image tokens", [64, 128, 256, 512, 1024], value=256,
                                        help="Smaller = faster prefill. The processor allows up to 2048.")
    prefill_chunk = st.select_slider("Prefill chunk (tokens)", [256, 512, 1024], value=1024,
                                     help="README 4.6: prefill q4f16 decoders in chunks of at most 1024.")
    ort_version = st.text_input("onnxruntime-web version", BASE.ort_web_version)
    tjs_version = st.text_input("transformers.js version (tokenizer only)", BASE.transformers_js_version)
    st.caption(f"Omni Pocket v{__version__} · data in `{BASE.data_dir}/`")


@st.cache_resource(show_spinner=False)
def get_engine(name: str, fingerprint: str, _settings: Settings):
    return make_engine(name, _settings)


def engine_fingerprint(s: Settings) -> str:
    key_hash = hashlib.sha256(s.remote_api_key.encode()).hexdigest()[:8] if s.remote_api_key else ""
    return "|".join([s.model_id, s.device, s.dtype, s.remote_url, s.remote_model, key_hash])


def speak(text: str) -> None:
    if not text or speech_mode == "off":
        return
    if speech_mode == "termux-tts-speak":
        subprocess.Popen(["termux-tts-speak", text[:4000]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    import json

    import streamlit.components.v1 as components

    components.html(
        "<script>const u=new SpeechSynthesisUtterance(%s);u.rate=1.05;"
        "speechSynthesis.cancel();speechSynthesis.speak(u);</script>" % json.dumps(text[:4000]),
        height=0,
    )


# ------------------------------------------------------------------- header --
st.title("🎙️ Omni Pocket")
st.caption("LFM3-Omni-2B on your phone — one model for vision, speech and text. "
           f"Python engine: **{engine_name}**")

tab_ask, tab_live, tab_bench, tab_device = st.tabs(["Ask", "Live (WebGPU)", "Benchmarks", "Device"])

# ---------------------------------------------------------------------- Ask --
with tab_ask:
    preset = BY_KEY[st.selectbox("Task", [p.key for p in PRESETS], format_func=lambda k: BY_KEY[k].title)]
    st.caption(f"💡 {preset.hint}  ·  suggested inputs: {', '.join(preset.needs)}")
    lang = "English"
    if "{lang}" in preset.system:
        lang = st.text_input("Target language", "English")

    if "chat" not in st.session_state:
        st.session_state.chat = []  # [{"role", "content", "meta"}]

    image_bytes = None
    src = st.radio("Image", ["none", "camera", "upload"], horizontal=True,
                   index=1 if "image" in preset.needs else 0,
                   help="Camera asks for permission only when selected.")
    if src == "camera":
        shot = st.camera_input("Take a photo", label_visibility="collapsed")
        image_bytes = shot.getvalue() if shot else None
    elif src == "upload":
        up = st.file_uploader("Image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
        image_bytes = up.getvalue() if up else None

    audio_bytes = None
    audio_widget = getattr(st, "audio_input", None) or getattr(st, "experimental_audio_input", None)
    if audio_widget:
        clip = audio_widget("Record a question / voice note (optional)")
        audio_bytes = clip.getvalue() if clip else None
    else:
        up = st.file_uploader("Audio (WAV)", type=["wav"])
        audio_bytes = up.getvalue() if up else None

    text = st.text_area("Text (optional)", height=80, placeholder="Ask something, or leave empty for spoken questions")
    c1, c2 = st.columns([3, 2])
    keep_ctx = c2.toggle("Follow-up mode", value=False, help="Send earlier text turns as context.")
    go = c1.button("Ask", type="primary", use_container_width=True)

    for turn in st.session_state.chat:
        with st.chat_message(turn["role"]):
            st.markdown(md(turn["content"]))
            if turn.get("meta"):
                st.caption(turn["meta"])

    if go:
        want_speech = speech_mode.startswith("model voice")
        req = OmniRequest(system=system_prompt(preset, lang, speak=speech_mode != "off"),
                          text=text, max_new_tokens=max_new, temperature=temperature, want_speech=want_speech)
        try:
            if image_bytes:
                req.image = prepare_image(image_bytes)
            if audio_bytes:
                req.audio = normalize_speech(audio_bytes)
        except Exception as exc:
            st.error(f"Could not read the media: {exc}")
            st.stop()
        if req.is_empty():
            st.warning("Add a photo, a recording or some text first.")
            st.stop()
        if keep_ctx:
            req.history = [{"role": t["role"], "content": t["content"]} for t in st.session_state.chat[-6:]]

        mode = "+".join(req.modalities())
        user_line = text.strip() or "(" + mode + ")"
        with st.chat_message("user"):
            if req.image is not None:
                st.image(req.image, width=220)
            st.markdown(md(user_line))

        with st.chat_message("assistant"):
            try:
                with st.spinner("Loading engine..." if engine_name == "transformers" else "Connecting..."):
                    engine = get_engine(engine_name, engine_fingerprint(settings), settings)
                    if hasattr(engine, "load"):
                        engine.load()
                if want_speech:
                    with st.spinner("Generating text + speech..."):
                        res = engine.generate(req)
                    st.markdown(md(res.text))
                    answer, ttft, total, n_tok, tps = res.text, res.ttft_s, res.total_s, res.output_tokens, res.tokens_per_s
                    if res.audio_wav:
                        st.audio(res.audio_wav, format="audio/wav", autoplay=True)
                    for note in res.notes:
                        st.caption(note)
                else:
                    timer = StreamTimer(engine.stream(req))
                    st.write_stream(md(p) for p in timer)
                    answer, ttft, total, n_tok = timer.text.strip(), timer.ttft, timer.elapsed, timer.chunks
                    tps = n_tok / (total - ttft) if ttft is not None and total > ttft and n_tok > 1 else None
                    res = None
            except Exception as exc:
                st.error(f"{engine_name} engine failed: {exc.__class__.__name__}: {exc}")
                if engine_name == "llamacpp":
                    st.info("Start the model server first: `./scripts/llama_server.sh` (or `./scripts/run.sh --llama`).")
                if engine_name == "transformers":
                    st.info("No torch on this device? Use the **Live (WebGPU)** tab or the remote engine. "
                            "See guides/INSTALL_USERLAND_ARCH.md.")
                st.stop()
            meta = f"{engine_name} · {mode} · TTFT {ttft or 0:.2f}s · {total:.1f}s" + (f" · {tps:.1f} tok/s" if tps else "")
            st.caption(meta)

        if speech_mode != "off" and not (res is not None and res.audio_wav):
            speak(answer)
        st.session_state.chat += [{"role": "user", "content": user_line},
                                  {"role": "assistant", "content": answer, "meta": meta}]
        HISTORY.append({"engine": engine_name, "preset": preset.key, "mode": mode, "question": text.strip(),
                        "answer": answer, "image": list(req.image.size) if req.image is not None else None,
                        "audio_s": round(req.audio.seconds, 2) if req.audio is not None else None})
        BENCH.append({"engine": engine_name, "model": getattr(engine, "model", ""), "mode": mode, "source": "ask",
                      "ttft_s": round(ttft, 3) if ttft is not None else None, "total_s": round(total, 3),
                      "output_tokens": n_tok, "tokens_per_s": round(tps, 2) if tps else None})

    if st.session_state.chat and st.button("Clear conversation"):
        st.session_state.chat = []
        st.rerun()

# ---------------------------------------------------------------- Live tab --
with tab_live:
    st.markdown(
        "Runs the **ONNX** export of the model **inside Chrome on your phone's GPU** — nothing is sent to "
        "this Python server. Open the app as `http://localhost:8501`: camera, mic and WebGPU need a "
        "secure context."
    )
    st.info(
        "transformers.js cannot load LiquidAI's ONNX export (*Unsupported model type: lfm3_omni*; their "
        "README: \"not loadable by Transformers.js\"), so this tab runs **Omni Pocket's port of LiquidAI's "
        "own runtime** on onnxruntime-web/WebGPU: text answers to camera + voice + text. "
        "It is new: if it fails on your phone, the **llamacpp** engine runs the same model natively.",
        icon="ℹ️",
    )
    st.link_button("Open the official LFM3-Omni WebGPU demo", OFFICIAL_WEBGPU_DEMO, use_container_width=True)
    event = webgpu_panel(
        key="webgpu",
        transformers_js=tjs_version.strip(),
        ort_version=ort_version.strip(),
        onnx_model_id=onnx_id.strip(),
        precision=web_precision,
        max_image_tokens=max_image_tokens,
        prefill_chunk=prefill_chunk,
        system=system_prompt(BY_KEY["look"], speak=speech_mode != "off"),
        max_new_tokens=max_new,
        speak=speech_mode != "off",
    )
    if event and event.get("seq") != st.session_state.get("last_webgpu_seq"):
        st.session_state.last_webgpu_seq = event.get("seq")
        kind = event.get("kind")
        if kind == "probe":
            st.session_state.webgpu_probe = event
        elif kind == "gpu_bench":
            st.session_state.webgpu_bench = event["results"]
        elif kind == "answer":
            BENCH.append({"engine": "webgpu", "model": f"{onnx_id} {event.get('bundle', '')}".strip(),
                          "mode": event.get("mode", ""), "source": "live",
                          "ttft_s": event.get("ttft_s"), "total_s": event.get("total_s"),
                          "output_tokens": event.get("output_tokens"), "tokens_per_s": event.get("tokens_per_s")})
            HISTORY.append({"engine": "webgpu", "preset": "live", "mode": event.get("mode", ""),
                            "question": "", "answer": event.get("text", "")})
        elif kind == "error":
            st.session_state.webgpu_error = event

    if st.session_state.get("webgpu_error"):
        err = st.session_state.webgpu_error
        msg = str(err.get("message", ""))
        if "WebGPU is not available" in msg or "shader" in msg.lower():
            st.error(f"{msg} — use the **llamacpp** engine (`./scripts/run.sh --llama`) on this phone.")
        else:
            st.warning(f"Browser engine error during {err.get('where')}: {msg}")

# ----------------------------------------------------------- Benchmarks tab --
with tab_bench:
    rows = BENCH.read()
    if not rows:
        st.info("No runs yet. Every answer (Ask + Live) is logged here with TTFT and tokens/s. "
                "For controlled runs use `python -m omnipocket bench --text 'hi' --runs 5`.")
    else:
        import pandas as pd

        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["ts"], unit="s")
        st.subheader("Median by engine / modality")
        st.dataframe(pd.DataFrame(summarize(rows)).T, use_container_width=True)
        metric = st.radio("Chart", ["ttft_s", "tokens_per_s", "total_s"], horizontal=True)
        if metric in df and df[metric].notna().any():
            st.scatter_chart(df.dropna(subset=[metric]), x="time", y=metric, color="engine")
        with st.expander("Raw runs"):
            st.dataframe(df.drop(columns=["ts"]).iloc[::-1], use_container_width=True)
        c1, c2 = st.columns(2)
        c1.download_button("Export bench.jsonl", BENCH.path.read_bytes(), "bench.jsonl", use_container_width=True)
        if HISTORY.path.exists():
            c2.download_button("Export history.jsonl", HISTORY.path.read_bytes(), "history.jsonl",
                               use_container_width=True)
        if st.button("Delete all logs"):
            BENCH.clear()
            HISTORY.clear()
            st.rerun()
    if st.session_state.get("webgpu_bench"):
        st.subheader("Browser GPU matmul")
        st.table(st.session_state.webgpu_bench)

# --------------------------------------------------------------- Device tab --
with tab_device:
    r = report(".")
    c1, c2, c3 = st.columns(3)
    c1.metric("Environment", r.env)
    c2.metric("RAM", f"{r.total_ram_mb / 1024:.1f} GB" if r.total_ram_mb else "?")
    c3.metric("Free disk", f"{r.free_disk_gb} GB")
    st.write(f"**Arch** `{r.machine}` · **Python** `{r.python}` · **torch device** `{r.torch_device}`")
    st.write("**Modules** " + " · ".join(f"{k} {'✅' if v else '❌'}" for k, v in r.modules.items()))
    for note in r.notes:
        st.markdown(f"- {note}")
    probe = st.session_state.get("webgpu_probe")
    st.subheader("Browser (from the Live tab's check)")
    if probe:
        st.json({k: v for k, v in probe.items() if k not in ("seq", "at")})
    else:
        st.caption("Press **Check WebGPU** in the Live tab to fill this in.")
    st.subheader("Which engine should I use?")
    st.markdown(
        "| You have | Use |\n|---|---|\n"
        "| Termux (any phone, 6 GB+ RAM) | **llamacpp** — `./scripts/run.sh --llama` (GGUF Q4_K_M on the CPU) |\n"
        "| Termux + Chrome 121+ with WebGPU | **Live (WebGPU)** tab — LiquidAI's ONNX export on the phone GPU |\n"
        "| Termux + a PC with a GPU | **remote** engine → `python -m omnipocket serve` on the PC |\n"
        "| UserLAnd / proot Arch, ≥ 8 GB RAM | **transformers** engine (bfloat16, CPU, slow but fully local) |\n"
        "| Just trying the UI | **mock** engine |"
    )
    st.caption(f"Checked at {time.strftime('%H:%M:%S')}")
