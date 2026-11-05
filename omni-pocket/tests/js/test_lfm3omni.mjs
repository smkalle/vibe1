// Compares omnipocket/webgpu/lfm3omni.js with LiquidAI's Python runtime (liquidonnx) on a tiny
// seeded LFM3-Omni export. Build the reference first (tests/js/make_reference.py), then:
//
//   OMNI_REF=/tmp/omni-ref ORT_WEB_DIR=/path/to/node_modules/onnxruntime-web node --test tests/js/test_lfm3omni.mjs
//
// Runs onnxruntime-web's wasm backend in Node: the fp32 bundle only (README 4.6 of onnx-export:
// the wasm EP has no GatherBlockQuantized kernel for the q4 decoders). The q4 bundle runs on
// WebGPU in tests/js/browser_check.mjs.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import * as R from "../../omnipocket/webgpu/lfm3omni.js";

const REF = process.env.OMNI_REF;
const skip = !REF || !fs.existsSync(path.join(REF ?? "", "reference.json"));
const ref = skip ? null : JSON.parse(fs.readFileSync(path.join(REF, "reference.json"), "utf8"));
const f32 = (name) => { const b = fs.readFileSync(path.join(REF, "data", name)); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); };
const maxDiff = (a, b) => { assert.equal(a.length, b.length, "length"); let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };

async function loadOrt() {
  const req = createRequire(path.join(process.env.ORT_WEB_DIR || process.cwd(), "package.json"));
  const ort = await import(pathToFileURL(req.resolve(process.env.ORT_WEB_DIR || "onnxruntime-web")).href);
  ort.env.wasm.numThreads = 1;
  return ort;
}

function localSource(dir) {
  return {
    async loadJson(name) { const p = path.join(dir, name); return fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, "utf8")) : null; },
    async loadModel(file) {
      const onnx = path.join(dir, "onnx"), stem = file.replace(/\.onnx$/, "");
      const externalData = fs.readdirSync(onnx).filter((f) => f === `${stem}.onnx_data` || f.startsWith(`${stem}.onnx_data_`))
        .map((f) => ({ path: f, data: new Uint8Array(fs.readFileSync(path.join(onnx, f))) }));
      return { model: new Uint8Array(fs.readFileSync(path.join(onnx, file))), externalData };
    },
  };
}

test("mel filters match transformers' slaney filter bank", { skip }, () => {
  assert.ok(maxDiff(R.melFilters(R.MEL), f32("mel_filters.f32")) < 1e-6);
});

test("log-mel features match the LFM3 log-mel recipe", { skip }, () => {
  const { data, frames } = R.logMel(f32("wave.f32"), R.MEL);
  assert.equal(frames, ref.mel_frames);
  assert.ok(maxDiff(data, f32("mel.f32")) < 1e-4);
  assert.equal(R.numAudioTokens(frames), ref.n_audio);
});

test("smart_resize matches transformers (banker's rounding included)", { skip }, () => {
  for (const [h, w, cap, eh, ew] of ref.smart_resize) assert.deepEqual(R.smartResize(h, w, 32, 65536, cap), [eh, ew], `${h}x${w} cap ${cap}`);
});

test("patchify matches Qwen2VLImageProcessor pixel_values", { skip }, () => {
  const rgb = new Uint8Array(fs.readFileSync(path.join(REF, "data", "image_rgb.u8")));
  const cfg = R.imageConfig(JSON.parse(fs.readFileSync(path.join(REF, "model", "processor_config.json"), "utf8")).image_processor);
  const { pixelValues, grid } = R.patchify(rgb, 320, 256, cfg, 3);
  assert.deepEqual(grid, ref.grid);
  assert.ok(maxDiff(pixelValues, f32("pixel_values.f32")) < 1e-6);
});

test("prompt expansion matches prompt_ids / expand_image_tokens", { skip }, () => {
  const rt = new R.LFM3OmniRuntime(null, {});
  rt.image = { merge: 2 };
  assert.deepEqual(rt.expandPrompt(ref.prompt, [ref.grid], [ref.n_audio]), ref.ids);
});

test("buildPrompt follows the LFM3-Omni chat template", () => {
  assert.equal(R.buildPrompt({ system: "S", images: 1, audios: 1, text: "Q" }),
    "<|startoftext|><|im_start|>system\nS<|im_end|>\n<|im_start|>user\n<|vision_start|><|image|><|vision_end|><|vision_start|><|audio|><|vision_end|>Q<|im_end|>\n<|im_start|>assistant\n<think></think>");
});

test("buildPrompt equals the real chat_template.jinja rendering", { skip }, () => {
  const t = ref.templates;
  assert.equal(R.buildPrompt({ system: "S", images: 1, audios: 1, text: "Q" }), t.full);
  assert.equal(R.buildPrompt({ text: "Hi there" }), t.text_only);
  assert.equal(R.buildPrompt({ system: "S", history: [{ role: "user", content: "a" }, { role: "assistant", content: "b" }], images: 1, text: "c" }), t.history);
});

test("fp16 decode", () => {
  assert.equal(R.f16ToF32(0x3c00), 1); assert.equal(R.f16ToF32(0xc000), -2); assert.equal(R.f16ToF32(0x0000), 0);
  assert.equal(R.argmaxSuppressed(new Float32Array([1, 5, 3]), new Set([1])), 2);
});

const noModel = skip || !fs.existsSync(path.join(REF, "model", "onnx"));
test("encoders and greedy generation match the Python runtime (fp32, wasm)", { skip: noModel, timeout: 600000 }, async () => {
  const ort = await loadOrt();
  const src = localSource(path.join(REF, "model"));
  const bundle = { decoder: "decoder.onnx", vision_encoder: "vision_encoder.onnx", audio_encoder: "audio_encoder.onnx" };
  const rt = new R.LFM3OmniRuntime(ort, { executionProvider: "wasm", bundle, ...src });
  await rt.load();
  const img = await rt.encodeImage(f32("pixel_values.f32"), ref.grid);
  assert.ok(maxDiff(img, f32("image_features.f32")) < 1e-3, "image features");
  const aud = await rt.encodeAudio(f32("wave.f32"));
  assert.ok(maxDiff(aud, f32("audio_features.f32")) < 1e-3, "audio features");
  const ids = rt.expandPrompt(ref.prompt, [ref.grid], [ref.n_audio]);
  const out = await rt.generate(ids, { imageFeatures: img, audioFeatures: aud, maxNewTokens: 24 });
  assert.deepEqual(out.tokens, ref.tokens_fp32);
  // chunked prefill (README 4.6) must give the same answer: chunks of 7 split the image/audio spans
  const rt7 = new R.LFM3OmniRuntime(ort, { executionProvider: "wasm", bundle, prefillChunk: 7, ...src });
  await rt7.load();
  const out7 = await rt7.generate(ids, { imageFeatures: img, audioFeatures: aud, maxNewTokens: 24 });
  assert.deepEqual(out7.tokens, ref.tokens_fp32);
});
