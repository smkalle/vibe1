// The toy bundle (make_toy_model.py) through LFM3OmniRuntime on onnxruntime-web's WebGPU EP in
// headless Chromium: GPU-resident caches (preferredOutputLocation "gpu-buffer"), logits read back
// with getData, tensor disposal, chunked prefill. Needs Playwright (require('playwright')).
//
//   OMNI_TOY=/tmp/omni-toy ORT_WEB_DIR=/path/to/node_modules/onnxruntime-web node tests/js/browser_toy.mjs
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const TOY = process.env.OMNI_TOY;
const roots = { "/ort/": path.join(process.env.ORT_WEB_DIR, "dist"), "/mod/": path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../omnipocket/webgpu"), "/toy/": TOY };
const types = { ".mjs": "text/javascript", ".js": "text/javascript", ".wasm": "application/wasm", ".json": "application/json" };
const server = http.createServer((req, res) => {
  const url = decodeURIComponent(req.url.split("?")[0]);
  if (url === "/") { res.setHeader("content-type", "text/html"); return res.end("<!doctype html><title>toy</title>"); }
  if (url === "/list") { res.setHeader("content-type", "application/json"); return res.end(JSON.stringify(fs.readdirSync(path.join(TOY, "model", "onnx")))); }
  const root = Object.keys(roots).find((r) => url.startsWith(r));
  const file = root && path.join(roots[root], url.slice(root.length));
  if (!file || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.statusCode = 404; return res.end(); }
  res.setHeader("content-type", types[path.extname(file)] || "application/octet-stream");
  fs.createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const { chromium } = createRequire(import.meta.url)("playwright");
const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage();
page.on("console", (m) => { if (m.type() === "error") console.log("[page]", m.text()); });
await page.goto(`http://127.0.0.1:${server.address().port}/`);
const result = await page.evaluate(async () => {
  const ort = await import("/ort/ort.webgpu.min.mjs");
  ort.env.wasm.wasmPaths = "/ort/"; ort.env.wasm.numThreads = 1;
  const L = await import("/mod/lfm3omni.js");
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) return { error: "no WebGPU adapter" };
  const ref = await (await fetch("/toy/reference.json")).json();
  const list = await (await fetch("/list")).json();
  const bytes = async (u) => new Uint8Array(await (await fetch(u)).arrayBuffer());
  const f32 = async (n) => new Float32Array((await bytes(`/toy/data/${n}`)).buffer);
  const src = {
    loadJson: async (n) => { const r = await fetch(`/toy/model/${n}`); return r.ok ? r.json() : null; },
    loadModel: async (file) => {
      const stem = file.replace(/\.onnx$/, "");
      const ext = list.filter((f) => f === `${stem}.onnx_data` || f.startsWith(`${stem}.onnx_data_`));
      return { model: await bytes(`/toy/model/onnx/${file}`), externalData: await Promise.all(ext.map(async (f) => ({ path: f, data: await bytes(`/toy/model/onnx/${f}`) }))) };
    },
  };
  const out = { adapter: [adapter.info?.vendor, adapter.info?.architecture].filter(Boolean).join(" "), runs: {} };
  for (const chunk of [1024, 5]) {
    const rt = new L.LFM3OmniRuntime(ort, { executionProvider: "webgpu", bundle: { decoder: "decoder.onnx", vision_encoder: "vision_encoder.onnx", audio_encoder: "audio_encoder.onnx" }, prefillChunk: chunk, ...src });
    await rt.load();
    rt.image = { merge: 2 };
    const img = await rt.encodeImage(await f32("pixel_values.f32"), ref.grid);
    const gen = await rt.generate(ref.ids, { imageFeatures: img, audioFeatures: await f32("audio_features.f32"), maxNewTokens: 24 });
    out.runs[chunk] = { tokens: gen.tokens, match: JSON.stringify(gen.tokens) === JSON.stringify(ref.tokens) };
  }
  return out;
});
await browser.close(); server.close();
console.log(JSON.stringify(result));
process.exit(result.error || Object.values(result.runs).some((r) => !r.match) ? 1 : 0);
