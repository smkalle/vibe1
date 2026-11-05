# Tutorial: real-time, fully local multimodal AI with LFM3-Omni-2B (Python → WebGPU → phone)

This tutorial walks the six stages from the original plan. Each stage maps to
code in this repo, so you can read the explanation and run the matching piece.

---

## 1. Why this matters

A typical "talk to your camera" assistant chains four models:

```
mic ──ASR──► text ─┐
camera ──VLM──► caption ─┴─► LLM ──► text ──TTS──► speaker
```

Each hop adds latency, a failure point and lost information. The ASR throws
away tone. The caption throws away anything the LLM later wants to ask about.

**LFM3-Omni-2B** is a single ~2.17B end-to-end model:

- **Inputs**: text, images and audio, interleaved in one conversation.
- **Outputs**: text, or interleaved text + speech (audio-codec tokens decoded by the processor's `audio_tokenizer`).
- **Size**: small enough for on-device use. q4f16 ONNX is about 1.6 GB to download. In bf16 on CPU it needs about 4.4 GB of RAM for weights.

What that gets you:

- **Privacy**: photos of medicine labels and receipts, and voice notes, never leave the phone.
- **Latency**: no network round trip and no pipeline handoffs. A short spoken question about a photo is a single prefill.
- **Offline**: works on a plane or in a basement once the weights are cached.

In Omni Pocket, the **Ask** tab's presets ("Spoken visual Q&A", "Read text",
"Translate a sign"…) are each just a system prompt plus suggested inputs, all
sent to the same model (`omnipocket/presets.py`).

---

## 2. Model overview and the Python baseline

Before going to the edge, learn the interface in Python. It is the reference
the other runtimes mirror.

```bash
pip install "transformers>=5.20.2" accelerate torch torchvision torchcodec pillow soundfile librosa
```

```python
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "LiquidAI/LFM3-Omni-2B"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

model = AutoModelForMultimodalLM.from_pretrained(
    model_id, dtype=dtype, device_map=device, trust_remote_code=True
).eval()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
codec = processor.audio_tokenizer.to(device=device).eval()   # speech in/out codec

conversation = [
    {"role": "system", "content": "Answer questions about the provided image. Respond with text."},
    {"role": "user", "content": [
        {"type": "image", "image": "photo.jpg"},
        {"type": "audio", "audio": "question.wav"},
    ]},
]
inputs = processor.apply_chat_template(
    conversation, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
).to(model.device)

out = model.generate(**inputs, max_new_tokens=512, generation_mode="text")
reply = processor.decode(out.sequences[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(reply)
```

Things to notice:

- **One chat template for every modality.** Media are content parts in the user turn, not separate model calls.
- **`generation_mode`** picks text-only or interleaved output. Text mode is the cheapest and the right default for screen answers.
- **The codec belongs to the processor.** Speech output is codec tokens that `processor.audio_tokenizer` turns into a waveform.

**In this repo:** `omnipocket/engines/transformers_engine.py` wraps exactly
this. It adds streaming (`TextIteratorStreamer` on a worker thread), timing
(TTFT and tokens/s), and a bfloat16-on-CPU default. The model card uses
float32 on CPU, which is about 8.7 GB and would not fit most phones. Try it:

```bash
python -m omnipocket ask --engine transformers --image photo.jpg --audio question.wav --stream
```

Media prep (`omnipocket/media.py`): phone photos are EXIF-rotated and downscaled to ≤ 768 px. Audio is decoded from WAV, down-mixed, resampled to 16 kHz with numpy, peak-normalised and silence-trimmed. Fewer pixels and samples mean fewer prefill tokens, and prefill dominates latency on a phone.

---

## 3. The browser path: ONNX Runtime Web + WebGPU

**Try the official demo first:**
[huggingface.co/spaces/LiquidAI/LFM3-Omni-2B-WebGPU](https://huggingface.co/spaces/LiquidAI/LFM3-Omni-2B-WebGPU)
(about 1.6 GB download, then fully local). It works in Chrome on Android 12+ with a recent Adreno or Mali GPU.

The export lives at `LiquidAI/LFM3-Omni-2B-ONNX`. The model card's WebGPU notes say:

1. Use **`onnxruntime-web/webgpu`** (transformers.js wraps it).
2. Pair the **`decoder_q4f16`** graph with the **float32 side graphs** (embeddings, vision encoder, audio encoder). Quantising the small encoders saves little memory and costs accuracy.
3. **Chunk prefill to ≤ 1024 tokens.** Long image+audio prompts in one pass can blow WebGPU buffer limits on mobile.

The custom WGSL kernels Xenova used for the headline speeds (about 280 tok/s on an M4 Max) live at
[huggingface.co/webgpu-kernels](https://huggingface.co/webgpu-kernels) and load through `@huggingface/kernels`.

> **Status, Nov 2026:** the published transformers.js releases (≤ 4.3.2,
> 4.4.0-next.0) support `lfm3` and `lfm3_vl` but not `lfm3_omni`, so loading
> the ONNX repo fails with `Unsupported model type: lfm3_omni`. The official
> demo Space works because it ships its own build. Until a release adds the
> type, the practical on-phone path is llama.cpp (section 3b).

**In this repo:** the **Live (WebGPU)** tab is a bidirectional Streamlit
component (`omnipocket/webgpu/index.html`). It has no npm build, because it
speaks Streamlit's `postMessage` protocol directly. It does three things:

1. **Device check**: `navigator.gpu` adapter, vendor/architecture, `shader-f16` (required for q4f16), `maxBufferSize`, `maxStorageBufferBindingSize`, and secure-context status.
2. **Matmul benchmark**: a tiled 1024² WGSL matmul in f32 and f16, reported in GFLOP/s. Use it to compare phones and to confirm you really are on the GPU.
3. **Live agent**: dynamically imports `@huggingface/transformers@<version>`, loads the ONNX repo with `device: "webgpu"` and a per-graph `dtype` map, and streams the answer.

```js
const TJS = await import(`https://cdn.jsdelivr.net/npm/@huggingface/transformers@${ver}`);
const processor = await TJS.AutoProcessor.from_pretrained(modelId);
const model = await TJS[cls].from_pretrained(modelId, {
  device: "webgpu",
  dtype: { decoder_model_merged: "q4f16", embed_tokens: "fp32", vision_encoder: "fp32", audio_encoder: "fp32" },
});
```

The transformers.js version, model class and dtype map are sidebar settings,
not hard-coded, so you can match whatever the ONNX model card specifies. The
results (probe, GFLOP/s, TTFT, tok/s) flow back to Python and land in the
Benchmarks tab.

**Why Streamlit is still useful here:** Python serves the page from
`localhost` (a secure context, so camera, mic and WebGPU are allowed), keeps
the logs, and gives you the other engines for comparison. The heavy lifting
runs on the phone GPU in Chrome, so this works from **plain Termux** with no
PyTorch.

---

## 3b. The on-phone path that works today: llama.cpp + GGUF

llama.cpp added LFM3-Audio and LFM3-Omni in PR #31257 (released in v0.8.0,
Oct 28 2026). LiquidAI publishes `LiquidAI/LFM3-Omni-2B-GGUF` (Q4_K_M model
plus a Q8_0 multimodal projector). Termux packages llama.cpp, so the whole
stack is:

```bash
pkg install llama-cpp
llama-server -hf LiquidAI/LFM3-Omni-2B-GGUF:Q4_K_M --reasoning off --host 127.0.0.1 --port 8080
```

- `-hf repo:quant` downloads the model **and** its mmproj (vision + audio encoders) into `~/.cache/llama.cpp`.
- `--reasoning off`: LFM3-Omni is not a reasoning model. The server README says to run it this way so turns start with an empty think block.
- The OpenAI endpoint accepts `{"type":"image_url"}` and `{"type":"input_audio"}` content parts with data URLs or raw base64. That is exactly what Omni Pocket's `RemoteEngine` sends, so the `llamacpp` engine is the same client pointed at `127.0.0.1:8080`.
- Speech output: `POST /tts {"input": "..."}` returns WAV from the model's speech decoder. The app uses it for *model voice*. llama-server also has a realtime WebSocket endpoint that prefills audio as it arrives, a natural next step for a lower-latency voice loop.

`./scripts/run.sh --llama` wraps all of this. `scripts/llama_server.sh` sets
threads to cores − 2 and the context to 4096, and takes `OMNI_LLAMA_NGL` for
Vulkan/OpenCL offload.

---

## 4. The native/mobile path: react-native-webgpu

The browser carries overhead: tab memory caps, background throttling, and no
access to native audio sessions. Xenova showed the same model running
natively on iPhone through **React Native + `react-native-webgpu`**, a Dawn
(Chrome's WebGPU implementation) binding exposed to JS. The same WGSL kernels
and ONNX graphs run with:

- higher memory limits than a browser tab,
- native camera and audio pipelines (lower capture latency, echo cancellation),
- background audio and a proper app lifecycle.

Sketch of the migration from the browser component:

| Browser (this repo) | React Native |
|---|---|
| `navigator.gpu` | `import { gpu } from "react-native-webgpu"` (Dawn) |
| `getUserMedia` video | `react-native-vision-camera` frame processor → RGBA buffer |
| `MediaRecorder` + `OfflineAudioContext` | native recorder at 16 kHz mono PCM |
| Cache API for weights | app documents dir, downloaded once with resume |
| `speechSynthesis` | model speech via codec, or the platform TTS |

Since `index.html` already isolates model I/O (`grabFrame`, the 16 kHz
`Float32Array` audio and `ask()`), porting means swapping those three
adapters. The prompt and generation logic stays the same.

On Android without building an app, plain Chrome plus this repo's Live tab is
the closest equivalent.

---

## 5. Building a real-time agent: camera + microphone → interleaved response

The Live tab implements the loop:

```
[camera preview] ──grab frame (≤640 px)──┐
[hold-to-talk]  ──record → 16 kHz mono──┴─► chat template ─► prefill ─► stream tokens ─► speak
```

Latency budget on a phone, roughly in order of impact:

1. **Prefill**: image tokens plus audio tokens. Smaller frames (640 px) and short questions (< 30 s; the UI warns above that) cut it the most. This is where chunked prefill ≤ 1024 matters.
2. **First load**: a one-time download and shader compile. Weights are cached by the browser afterwards. Keep the tab open between questions.
3. **Decode**: memory-bandwidth-bound. q4f16 roughly halves bytes per token compared with q8.
4. **Speech**: starting the phone TTS on the full text is instant. Model-native interleaved speech is better quality but costs codec tokens.

Design choices in the code:

- **Push-to-talk, not always-on VAD.** It saves battery, avoids accidental prefills, and the release is a natural "go" signal (`auto_ask_after_speech`).
- **Frame grabbed at release time**, so you can aim while you speak.
- **Every answer is reported back** to Python with `ttft_s`, `tokens_per_s` and `mode`. That gives you a benchmark log for free across real use.

The Python **Ask** tab gives the same flow for the server-side engines with
`st.camera_input` + `st.audio_input`, which is useful when the phone's GPU is
too weak.

---

## 6. Optimisation and production notes

**Quantisation**

| Variant | Where | Notes |
|---|---|---|
| q4f16 decoder + fp32 side graphs | WebGPU | Card's recommendation. Needs `shader-f16` (the Device check shows it) |
| q4 (no f16) | WebGPU without `shader-f16` | Set dtype to `"q4"` in the sidebar |
| bfloat16 | CPU (UserLAnd), CUDA | ~4.4 GB. Default for the transformers engine here |
| float32 | CPU reference | ~8.7 GB. Correctness baseline only |

**Chunked prefill**: keep any single forward pass ≤ 1024 tokens on mobile
WebGPU. Mobile `maxStorageBufferBindingSize` is often 128–256 MB (the Device
check shows yours). Attention scratch for long prefills exceeds it before
weights do.

**Memory**: Android kills background apps aggressively. `run.sh` takes a
`termux-wake-lock`. Keep Chrome in the foreground while generating. On
UserLAnd, close other apps: proot shares the phone's RAM with everything else.

**Fleet-style benchmarking**: measure the same prompts across many devices and engines, and compare medians, not single runs.

```bash
python -m omnipocket bench --engine remote --image test.jpg --text "Read the total" --runs 10 --warmup 2
```

All runs (CLI, Ask, Live) append to `omni_data/bench.jsonl` with engine,
model, modality, TTFT and tok/s. The Benchmarks tab groups them by
engine | model | modality and charts them. Export the JSONL from each phone and
concatenate for a fleet view.

**Serving to many phones**: `python -m omnipocket serve` is a minimal
OpenAI-compatible endpoint that handles one generation at a time behind a
lock. A 2B model saturates a consumer GPU, so batching gives little on one
card. For more users, put several workers behind a load balancer, or use a
serving stack with continuous batching that supports the model, and keep
this app's `remote` engine as the client (it only speaks the OpenAI protocol).
