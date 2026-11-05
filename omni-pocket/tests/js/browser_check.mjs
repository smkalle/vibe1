// Runs omnipocket/webgpu/lfm3omni.js on onnxruntime-web's WebGPU EP in headless Chromium against
// the tiny export of make_reference.py, and compares the greedy tokens with LiquidAI's Python
// runtime. Run with Playwright available (require('playwright')):
//
//   OMNI_REF=/tmp/omni-ref ORT_WEB_DIR=/path/to/node_modules/onnxruntime-web node tests/js/browser_check.mjs
//
// Headless Chromium's adapter is SwiftShader: no shader-f16 (so no q4f16 decoder here), and its
// 4-wide subgroups break onnxruntime-web 1.30.0's FlashAttention from 32 query positions (onnx-export
// README 4.6), so the prefill runs in chunks of 28.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const REF = process.env.OMNI_REF;
const ORT_DIR = path.join(process.env.ORT_WEB_DIR, "dist");
const MOD_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../omnipocket/webgpu");
const ref = JSON.parse(fs.readFileSync(path.join(REF, "reference.json"), "utf8"));
const roots = { "/ort/": ORT_DIR, "/mod/": MOD_DIR, "/model/": path.join(REF, "model"), "/data/": path.join(REF, "data") };
const types = { ".mjs": "text/javascript", ".js": "text/javascript", ".wasm": "application/wasm", ".json": "application/json", ".html": "text/html" };

const server = http.createServer((req, res) => {
  const url = decodeURIComponent(req.url.split("?")[0]);
  if (url === "/") { res.setHeader("content-type", "text/html"); return res.end("<!doctype html><title>check</title>"); }
  if (url === "/list") { res.setHeader("content-type", "application/json"); return res.end(JSON.stringify(fs.readdirSync(path.join(REF, "model", "onnx")))); }
  const root = Object.keys(roots).find((r) => url.startsWith(r));
  const file = root && path.join(roots[root], url.slice(root.length));
  if (!file || !fs.existsSync(file)) { res.statusCode = 404; return res.end(); }
  res.setHeader("content-type", types[path.extname(file)] || "application/octet-stream");
  fs.createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const base = `http://127.0.0.1:${server.address().port}`;

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage();
page.on("console", (m) => { if (m.type() === "error") console.log("[page]", m.text()); });
await page.goto(base);

const result = await page.evaluate(async ({ ref }) => {
  const ort = await import("/ort/ort.webgpu.min.mjs");
  ort.env.wasm.wasmPaths = "/ort/";
  ort.env.wasm.numThreads = 1;
  const L = await import("/mod/lfm3omni.js");
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) return { error: "no WebGPU adapter" };
  const list = await (await fetch("/list")).json();
  const bytes = async (u) => new Uint8Array(await (await fetch(u)).arrayBuffer());
  const f32 = async (n) => new Float32Array((await bytes(`/data/${n}`)).buffer);
  const src = {
    loadJson: async (n) => { const r = await fetch(`/model/${n}`); return r.ok ? r.json() : null; },
    loadModel: async (file) => {
      const stem = file.replace(/\.onnx$/, "");
      const ext = list.filter((f) => f === `${stem}.onnx_data` || f.startsWith(`${stem}.onnx_data_`));
      return { model: await bytes(`/model/onnx/${file}`), externalData: await Promise.all(ext.map(async (f) => ({ path: f, data: await bytes(`/model/onnx/${f}`) }))) };
    },
  };
  const out = { adapter: [adapter.info?.vendor, adapter.info?.architecture].join(" "), f16: adapter.features.has("shader-f16") };
  const pixels = await f32("pixel_values.f32"), wave = await f32("wave.f32");
  for (const [name, bundle] of Object.entries({
    fp32: { decoder: "decoder.onnx", vision_encoder: "vision_encoder.onnx", audio_encoder: "audio_encoder.onnx" },
    q4: { decoder: "decoder_q4.onnx", vision_encoder: "vision_encoder_q8.onnx", audio_encoder: "audio_encoder.onnx" },
  })) {
    try {
      const rt = new L.LFM3OmniRuntime(ort, { executionProvider: "webgpu", bundle, prefillChunk: 28, ...src });
      await rt.load();
      const img = await rt.encodeImage(pixels, ref.grid);
      const aud = await rt.encodeAudio(wave);
      const ids = rt.expandPrompt(ref.prompt, [ref.grid], [ref.n_audio]);
      const t0 = performance.now();
      const gen = await rt.generate(ids, { imageFeatures: img, audioFeatures: aud, maxNewTokens: 24 });
      out[name] = { tokens: gen.tokens, ms: Math.round(performance.now() - t0), imgHead: Array.from(img.slice(0, 4)) };
    } catch (e) { out[name] = { error: String(e.message || e) }; }
  }
  return out;
}, { ref });

await browser.close();
server.close();

console.log(`adapter: ${result.adapter || result.error} (shader-f16: ${result.f16})`);
let failed = !!result.error;
for (const [name, want] of [["fp32", ref.tokens_fp32], ["q4", ref.tokens_q4]]) {
  const got = result[name] || {};
  if (got.error) { console.log(`${name}: ERROR ${got.error}`); failed = true; continue; }
  const same = JSON.stringify(got.tokens) === JSON.stringify(want);
  let prefix = 0; while (prefix < want.length && got.tokens[prefix] === want[prefix]) prefix++;
  console.log(`${name}: ${same ? "MATCH" : "DIFF"} ${prefix}/${want.length} leading tokens equal to the Python reference (${got.ms} ms)`);
  if (!same) { console.log("  got ", JSON.stringify(got.tokens)); console.log("  want", JSON.stringify(want)); }
  if (name === "fp32" && !same) failed = true;  // q4: DP4A int8 activations on non-Apple GPUs may flip near-ties
}
process.exit(failed ? 1 : 0);
