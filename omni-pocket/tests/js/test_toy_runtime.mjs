// LFM3OmniRuntime (omnipocket/webgpu/lfm3omni.js) on onnxruntime-web (wasm) against the toy bundle
// and numpy reference of tests/js/make_toy_model.py: cache handoff, external data, placeholder
// scatter, positions, suppression and chunked prefill.
//
//   python tests/js/make_toy_model.py /tmp/omni-toy
//   OMNI_TOY=/tmp/omni-toy ORT_WEB_DIR=/path/to/node_modules/onnxruntime-web node --test tests/js/test_toy_runtime.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import * as R from "../../omnipocket/webgpu/lfm3omni.js";

const TOY = process.env.OMNI_TOY;
const skip = !TOY || !fs.existsSync(path.join(TOY ?? "", "reference.json"));
const ref = skip ? null : JSON.parse(fs.readFileSync(path.join(TOY, "reference.json"), "utf8"));
const f32 = (n) => { const b = fs.readFileSync(path.join(TOY, "data", n)); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); };

async function loadOrt() {
  const dir = process.env.ORT_WEB_DIR || process.cwd();
  const req = createRequire(path.join(dir, "package.json"));
  const ort = await import(pathToFileURL(req.resolve(process.env.ORT_WEB_DIR ? dir : "onnxruntime-web")).href);
  ort.env.wasm.numThreads = 1;
  return ort;
}

const source = (dir) => ({
  async loadJson(n) { const p = path.join(dir, n); return fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, "utf8")) : null; },
  async loadModel(file) {
    const onnx = path.join(dir, "onnx"), stem = file.replace(/\.onnx$/, "");
    const externalData = fs.readdirSync(onnx).filter((f) => f === `${stem}.onnx_data` || f.startsWith(`${stem}.onnx_data_`))
      .map((f) => ({ path: f, data: new Uint8Array(fs.readFileSync(path.join(onnx, f))) }));
    return { model: new Uint8Array(fs.readFileSync(path.join(onnx, file))), externalData };
  },
});

test("toy bundle: greedy tokens match the numpy reference for every prefill chunk size", { skip, timeout: 120000 }, async () => {
  const ort = await loadOrt();
  const bundle = { decoder: "decoder.onnx", vision_encoder: "vision_encoder.onnx", audio_encoder: "audio_encoder.onnx" };
  for (const chunk of [1024, 5, 3, 1]) {
    const rt = new R.LFM3OmniRuntime(ort, { executionProvider: "wasm", bundle, prefillChunk: chunk, ...source(path.join(TOY, "model")) });
    await rt.load();
    assert.equal(rt.tokens.image, ref.tok.image, "token ids come from config.json");
    rt.image = { merge: 2 };
    const img = await rt.encodeImage(f32("pixel_values.f32"), ref.grid);
    const want = f32("image_features.f32");
    assert.equal(img.length, want.length);
    for (let i = 0; i < img.length; i++) assert.ok(Math.abs(img[i] - want[i]) < 1e-4, `image feature ${i}`);
    const ids = rt.expandPrompt(ref.prompt, [ref.grid], [f32("audio_features.f32").length / rt.hidden]);
    assert.deepEqual(ids, ref.ids);
    const seen = [];
    const out = await rt.generate(ids, { imageFeatures: img, audioFeatures: f32("audio_features.f32"), maxNewTokens: 24,
                                         onToken: (t, sofar) => seen.push(sofar.length) });
    assert.deepEqual(out.tokens, ref.tokens, `prefill chunk ${chunk}`);
    assert.deepEqual(seen, ref.tokens.map((_, i) => i + 1), "onToken gets the tokens so far");
  }
});

test("toy bundle: placeholder / feature row mismatch is rejected", { skip }, async () => {
  const ort = await loadOrt();
  const rt = new R.LFM3OmniRuntime(ort, { executionProvider: "wasm", bundle: { decoder: "decoder.onnx" }, ...source(path.join(TOY, "model")) });
  await rt.load();
  await assert.rejects(rt.generate(ref.ids, { imageFeatures: new Float32Array(8), audioFeatures: f32("audio_features.f32") }), /image placeholders/);
});

test("toy bundle: shouldStop ends generation early", { skip }, async () => {
  const ort = await loadOrt();
  const rt = new R.LFM3OmniRuntime(ort, { executionProvider: "wasm", bundle: { decoder: "decoder.onnx" }, ...source(path.join(TOY, "model")) });
  await rt.load();
  let n = 0;
  const out = await rt.generate(ref.ids, { imageFeatures: f32("image_features.f32"), audioFeatures: f32("audio_features.f32"),
                                           maxNewTokens: 24, shouldStop: () => ++n >= 3 });
  assert.equal(out.tokens.length, 3);
});
