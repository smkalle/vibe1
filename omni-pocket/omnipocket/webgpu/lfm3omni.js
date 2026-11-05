// LFM3-Omni text-answer runtime for onnxruntime-web (WebGPU or wasm).
//
// LiquidAI's ONNX exports (LiquidAI/LFM3-Omni-2B-ONNX) are NOT loadable by
// transformers.js: they run on plain onnxruntime with their own generation
// loop (github.com/Liquid4All/onnx-export, liquidonnx.lfm3_omni.session). This
// file ports the parts needed for a text answer to images + speech + text:
//
//   * log-mel front end      (lfm3_audio.session.log_mel / mel_filters)
//   * image preprocessing    (Qwen2VLImageProcessor: smart_resize, patchify)
//   * placeholder expansion  (<|image|> x grid tokens, <|audio|> x clip tokens)
//   * the cached decoder loop (lfm3.session.CachedDecoder + generate_ids, text mode)
//
// Graph choice follows the export README section 4.6 (WebGPU): the q4f16
// decoder with the q4 bundle's side graphs (vision_encoder_q8, audio_encoder),
// prefilled in chunks of at most 1024 tokens. Speech output (MaskGIT depth
// decoder + codec) is not ported; callers speak the text with platform TTS.
//
// Pure helpers are exported for tests (tests/js/*.mjs compare them with the
// Python reference).

export const DEFAULTS = {
  repo: "LiquidAI/LFM3-Omni-2B-ONNX",
  bundle: {
    decoder: "decoder_q4f16.onnx",
    vision_encoder: "vision_encoder_q8.onnx",
    audio_encoder: "audio_encoder.onnx",
  },
  maxImageTokens: 256, // the processor allows 2048; a phone prefill wants far fewer
  prefillChunk: 1024, // README 4.6: q4f16 decoders prefill in chunks of <= 1024 tokens
};

// Fallback token ids (LFM3-Omni-2B config.json); load() reads the real ones.
export const TOKENS = {
  bos: 124894, eos: 124900, pad: 124893,
  image: 125030, video: 125029, audio: 125031, audio_out: 125032,
  audio_start: 125012, text_end: 125014, mixed_start: 125015,
};

// ---------------------------------------------------------------- audio --

export const MEL = {
  sampling_rate: 16000, num_mel_bins: 128, n_fft: 400, win_length: 400, hop_length: 160,
  log_floor: -12.0, normalization_offset: 4.0, normalization_divisor: 4.0, encoder_stride: 8,
};

export function melConfig(featureExtractor = {}) {
  const f = featureExtractor;
  return {
    ...MEL,
    sampling_rate: f.sampling_rate ?? MEL.sampling_rate,
    num_mel_bins: f.feature_size ?? MEL.num_mel_bins,
    n_fft: f.n_fft ?? MEL.n_fft,
    win_length: f.win_length ?? MEL.win_length,
    hop_length: f.hop_length ?? MEL.hop_length,
    log_floor: f.log_floor ?? MEL.log_floor,
    normalization_offset: f.normalization_offset ?? MEL.normalization_offset,
    normalization_divisor: f.normalization_divisor ?? MEL.normalization_divisor,
    encoder_stride: f.encoder_stride ?? MEL.encoder_stride,
  };
}

const pyMod = (a, n) => ((a % n) + n) % n;

/** MelConfig.num_frames */
export function numFrames(numSamples, mel = MEL) {
  const padded = Math.max(mel.n_fft, numSamples + pyMod(-(numSamples - mel.n_fft), mel.hop_length));
  return 1 + Math.floor((padded - mel.n_fft) / mel.hop_length);
}

/** MelConfig.num_tokens: one encoder token per 8 mel frames (ceil). */
export function numAudioTokens(frames, mel = MEL) {
  return Math.ceil(frames / mel.encoder_stride);
}

const SLANEY = { minLogHz: 1000.0, minLogMel: 15.0, logstep: 27.0 / Math.log(6.4) };
export function hzToMel(f) {
  return f >= SLANEY.minLogHz ? SLANEY.minLogMel + Math.log(f / SLANEY.minLogHz) * SLANEY.logstep : (3.0 * f) / 200.0;
}
export function melToHz(m) {
  // the inverse step is ln(6.4) / 27 (transformers' mel_to_hertz)
  return m >= SLANEY.minLogMel ? SLANEY.minLogHz * Math.exp((Math.log(6.4) / 27.0) * (m - SLANEY.minLogMel)) : (200.0 * m) / 3.0;
}

/** transformers.audio_utils.mel_filter_bank(slaney, slaney): [nFreq * nMels], freq-major, float32. */
export function melFilters(mel = MEL) {
  const nFreq = Math.floor(mel.n_fft / 2) + 1, nMels = mel.num_mel_bins;
  const melMin = hzToMel(0), melMax = hzToMel(mel.sampling_rate / 2);
  const filterFreqs = new Float64Array(nMels + 2);
  for (let i = 0; i < nMels + 2; i++) filterFreqs[i] = melToHz(melMin + ((melMax - melMin) * i) / (nMels + 1));
  const half = Math.floor(mel.sampling_rate / 2);
  const out = new Float32Array(nFreq * nMels);
  for (let k = 0; k < nFreq; k++) {
    const fft = (half * k) / (nFreq - 1);
    for (let m = 0; m < nMels; m++) {
      const down = -(filterFreqs[m] - fft) / (filterFreqs[m + 1] - filterFreqs[m]);
      const up = (filterFreqs[m + 2] - fft) / (filterFreqs[m + 2] - filterFreqs[m + 1]);
      const enorm = 2.0 / (filterFreqs[m + 2] - filterFreqs[m]);
      out[k * nMels + m] = Math.max(0, Math.min(down, up)) * enorm;
    }
  }
  return out;
}

/** log_mel: input_features [num_mel_bins, T] (mel-major) of a 16 kHz mono clip. */
export function logMel(wave, mel = MEL, filters = null) {
  filters = filters || melFilters(mel);
  const T = numFrames(wave.length, mel), nfft = mel.n_fft, hop = mel.hop_length;
  const nFreq = Math.floor(nfft / 2) + 1, nMels = mel.num_mel_bins;
  const padded = new Float64Array((T - 1) * hop + nfft);
  for (let i = 0; i < wave.length; i++) padded[i] = wave[i];
  // periodic Hann (float32, as torch), centered in n_fft
  const window = new Float64Array(nfft), off = Math.floor((nfft - mel.win_length) / 2);
  for (let i = 0; i < mel.win_length; i++) window[off + i] = Math.fround(0.5 - 0.5 * Math.cos((2 * Math.PI * i) / mel.win_length));
  const cos = new Float64Array(nFreq * nfft), sin = new Float64Array(nFreq * nfft);
  for (let k = 0; k < nFreq; k++) for (let n = 0; n < nfft; n++) {
    const a = (2 * Math.PI * ((k * n) % nfft)) / nfft;
    cos[k * nfft + n] = Math.cos(a); sin[k * nfft + n] = Math.sin(a);
  }
  const out = new Float32Array(nMels * T), seg = new Float64Array(nfft), power = new Float64Array(nFreq);
  for (let t = 0; t < T; t++) {
    for (let n = 0; n < nfft; n++) seg[n] = padded[t * hop + n] * window[n];
    for (let k = 0; k < nFreq; k++) {
      let re = 0, im = 0;
      const base = k * nfft;
      for (let n = 0; n < nfft; n++) { re += seg[n] * cos[base + n]; im -= seg[n] * sin[base + n]; }
      power[k] = re * re + im * im;
    }
    for (let m = 0; m < nMels; m++) {
      let acc = 0;
      for (let k = 0; k < nFreq; k++) acc += filters[k * nMels + m] * power[k];
      const v = Math.max(Math.log10(acc), mel.log_floor);
      out[m * T + t] = (v + mel.normalization_offset) / mel.normalization_divisor;
    }
  }
  return { data: out, frames: T };
}

// ---------------------------------------------------------------- images --

/** Python's round(): halves to even. */
export function pyRound(x) {
  const f = Math.floor(x), d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

/** Qwen2-VL smart_resize -> [height, width], multiples of factor. */
export function smartResize(height, width, factor, minPixels, maxPixels) {
  if (Math.max(height, width) / Math.min(height, width) > 200) throw new Error("aspect ratio over 200");
  let h = pyRound(height / factor) * factor, w = pyRound(width / factor) * factor;
  if (h * w > maxPixels) {
    const beta = Math.sqrt((height * width) / maxPixels);
    h = Math.max(factor, Math.floor(height / beta / factor) * factor);
    w = Math.max(factor, Math.floor(width / beta / factor) * factor);
  } else if (h * w < minPixels) {
    const beta = Math.sqrt(minPixels / (height * width));
    h = Math.ceil((height * beta) / factor) * factor;
    w = Math.ceil((width * beta) / factor) * factor;
  }
  return [h, w];
}

export function imageConfig(imageProcessor = {}, maxImageTokens = DEFAULTS.maxImageTokens) {
  const p = imageProcessor;
  const patch = p.patch_size ?? 16, merge = p.merge_size ?? 2;
  const pixelsPerToken = (patch * merge) ** 2;
  let maxPixels = p.size?.longest_edge ?? 2097152;
  if (maxImageTokens) maxPixels = Math.min(maxPixels, maxImageTokens * pixelsPerToken);
  return {
    patch, merge, temporal: p.temporal_patch_size ?? 2,
    mean: p.image_mean ?? [0.5, 0.5, 0.5], std: p.image_std ?? [0.5, 0.5, 0.5],
    rescale: p.rescale_factor ?? 1 / 255,
    minPixels: p.size?.shortest_edge ?? 65536, maxPixels,
  };
}

/**
 * Qwen2VLImageProcessor's rescale + normalize + patchify of an image already
 * resized to (height, width) (multiples of patch * merge).
 * rgba: Uint8ClampedArray / Uint8Array, 4 bytes per pixel (canvas ImageData) or 3 with channels=3.
 * -> { pixelValues: Float32Array [gh*gw, C*temporal*patch*patch] (block-major rows), grid: [1, gh, gw] }
 */
export function patchify(rgba, width, height, cfg, channels = 4) {
  const { patch: ps, merge: m, temporal: tp } = cfg;
  const gh = height / ps, gw = width / ps;
  if (!Number.isInteger(gh) || !Number.isInteger(gw) || gh % m || gw % m) throw new Error(`size ${width}x${height} is not a multiple of ${ps * m}`);
  const C = 3, dim = C * tp * ps * ps;
  const out = new Float32Array(gh * gw * dim);
  const scale = [0, 1, 2].map((c) => cfg.rescale / cfg.std[c]);
  const shift = [0, 1, 2].map((c) => -cfg.mean[c] / cfg.std[c]);
  let row = 0;
  for (let bh = 0; bh < gh / m; bh++) for (let bw = 0; bw < gw / m; bw++) for (let mh = 0; mh < m; mh++) for (let mw = 0; mw < m; mw++) {
    const y0 = (bh * m + mh) * ps, x0 = (bw * m + mw) * ps, base = row * dim;
    for (let c = 0; c < C; c++) for (let t = 0; t < tp; t++) for (let py = 0; py < ps; py++) for (let px = 0; px < ps; px++) {
      const v = rgba[((y0 + py) * width + (x0 + px)) * channels + c];
      out[base + ((c * tp + t) * ps + py) * ps + px] = v * scale[c] + shift[c];
    }
    row++;
  }
  return { pixelValues: out, grid: [1, gh, gw] };
}

export function imageTokenCount(grid, merge) {
  return (grid[0] * grid[1] * grid[2]) / (merge * merge);
}

// ---------------------------------------------------------------- prompt --

/**
 * The LFM3-Omni chat template for one system + one user turn + generation
 * prompt, thinking off (the template's default):
 *   <|startoftext|><|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{images}{audio}{text}<|im_end|>\n<|im_start|>assistant\n<think></think>
 * history: [{role, content: string}] text turns before the user turn.
 */
export function buildPrompt({ system = "", images = 0, audios = 0, text = "", history = [] }) {
  let s = "<|startoftext|>";
  if (system) s += `<|im_start|>system\n${system}<|im_end|>\n`;
  for (const turn of history) {
    s += `<|im_start|>${turn.role}\n`;
    s += turn.role === "assistant" ? `<think></think>${turn.content}` : turn.content;
    s += "<|im_end|>\n";
  }
  s += "<|im_start|>user\n";
  s += "<|vision_start|><|image|><|vision_end|>".repeat(images);
  s += "<|vision_start|><|audio|><|vision_end|>".repeat(audios);
  s += `${text}<|im_end|>\n<|im_start|>assistant\n<think></think>`;
  return s;
}

/** Repeat each occurrence of `token` counts[i] times (expand_image_tokens / prompt_ids). */
export function expandTokens(ids, token, counts) {
  const n = ids.filter((t) => t === token).length;
  if (n !== counts.length) throw new Error(`prompt has ${n} placeholders ${token}, got ${counts.length} counts`);
  const out = [];
  let i = 0;
  for (const t of ids) {
    if (t === token) { for (let k = 0; k < counts[i]; k++) out.push(token); i++; }
    else out.push(t);
  }
  return out;
}

// ---------------------------------------------------------------- tensors --

export function f16ToF32(h) {
  const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * 2 ** -14 * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * 2 ** (e - 15) * (1 + f / 1024);
}

export function argmaxSuppressed(logits, suppress = new Set()) {
  let best = -1, bestV = -Infinity;
  const isF16 = logits instanceof Uint16Array;
  for (let i = 0; i < logits.length; i++) {
    if (suppress.has(i)) continue;
    const v = isF16 ? f16ToF32(logits[i]) : logits[i];
    if (v > bestV) { bestV = v; best = i; }
  }
  return best;
}

const ZERO_DTYPES = { float32: Float32Array, float16: Uint16Array, int64: BigInt64Array, int32: Int32Array, bool: Uint8Array };

/** Empty past_* caches as initialize_cache: symbolic dims -> 1, "sequence" dims -> 0, batch 1. */
export function emptyCaches(ort, session) {
  const caches = {};
  for (const meta of session.inputMetadata || []) {
    if (!meta.name.startsWith("past")) continue;
    const shape = meta.shape.map((d) => (typeof d === "number" && d >= 0 ? d : (String(d).toLowerCase().includes("sequence") ? 0 : 1)));
    shape[0] = 1;
    const Arr = ZERO_DTYPES[meta.type] || Float32Array;
    caches[meta.name] = new ort.Tensor(meta.type, new Arr(shape.reduce((a, b) => a * b, 1)), shape);
  }
  return caches;
}

/** present_conv.N -> past_conv.N, present.N.key -> past_key_values.N.key */
export function cacheInputName(output) {
  if (output.startsWith("present_conv.")) return output.replace("present_conv.", "past_conv.");
  if (output.startsWith("present.")) return output.replace("present.", "past_key_values.");
  return null;
}

const i64 = (arr) => BigInt64Array.from(arr, (v) => BigInt(v));

// ---------------------------------------------------------------- runtime --

export class LFM3OmniRuntime {
  /**
   * ort: the onnxruntime-web module ('onnxruntime-web/webgpu' for WebGPU).
   * opts.loadModel(name) -> {model: Uint8Array|string, externalData: [{path, data}]}
   * opts.loadJson(name) -> object|null (config.json, processor_config.json, ...)
   */
  constructor(ort, opts = {}) {
    this.ort = ort;
    this.ep = opts.executionProvider || "webgpu";
    this.bundle = { ...DEFAULTS.bundle, ...(opts.bundle || {}) };
    this.maxImageTokens = opts.maxImageTokens ?? DEFAULTS.maxImageTokens;
    this.prefillChunk = opts.prefillChunk ?? DEFAULTS.prefillChunk;
    this.loadModel = opts.loadModel;
    this.loadJson = opts.loadJson;
    this.sessions = {};
    this.tokens = { ...TOKENS };
  }

  async load(onStatus = () => {}) {
    const config = (await this.loadJson("config.json")) || {};
    const processor = (await this.loadJson("processor_config.json")) || {};
    const pre = (await this.loadJson("preprocessor_config.json")) || processor.image_processor || {};
    const t = config.text_config || {};
    const pick = (k, d) => (config[k] ?? d);
    this.tokens = {
      bos: t.bos_token_id ?? TOKENS.bos, eos: t.eos_token_id ?? TOKENS.eos, pad: t.pad_token_id ?? TOKENS.pad,
      image: pick("image_token_id", TOKENS.image), video: pick("video_token_id", TOKENS.video),
      audio: pick("audio_token_id", TOKENS.audio), audio_out: pick("audio_out_token_id", TOKENS.audio_out),
      audio_start: pick("audio_start_token_id", TOKENS.audio_start), text_end: pick("text_end_token_id", TOKENS.text_end),
      mixed_start: pick("mixed_start_token_id", TOKENS.mixed_start),
    };
    this.hidden = t.hidden_size ?? 2048;
    this.numCodebooks = config.num_codebooks ?? 12;
    this.mel = melConfig(processor.feature_extractor || {});
    this.melFilters = melFilters(this.mel);
    this.image = imageConfig(pre, this.maxImageTokens);
    this.onStatus = onStatus;
    await this.ensure("decoder");
    const dec = this.sessions.decoder;
    this.decoderInputs = new Set(dec.inputNames);
    this.cacheOutputs = dec.outputNames.filter((n) => cacheInputName(n));
    onStatus("ready");
  }

  /** Create a graph's session on first use (encoders load lazily: no audio, no audio encoder download). */
  async ensure(role) {
    if (this.sessions[role]) return this.sessions[role];
    const file = this.bundle[role];
    (this.onStatus || (() => {}))(`loading ${file}`);
    const { model, externalData } = await this.loadModel(file);
    const ep = this.ep === "webgpu" ? [{ name: "webgpu", enableMatmulFp32Accumulation: true }] : [this.ep];
    const options = { executionProviders: ep, graphOptimizationLevel: "all" };
    if (externalData?.length) options.externalData = externalData;
    if (role === "decoder" && this.ep === "webgpu") options.preferredOutputLocation = "gpu-buffer";
    this.sessions[role] = await this.ort.InferenceSession.create(model, options);
    return this.sessions[role];
  }

  /** Vision encoder: one run per grid -> Float32Array [tokens * hidden]. */
  async encodeImage(pixelValues, grid) {
    const m = this.image.merge, [, gh, gw] = grid, dim = pixelValues.length / (gh * gw);
    const input = new this.ort.Tensor("float32", pixelValues, [gh / m, gw / m, m, m, dim]);
    const sess = await this.ensure("vision_encoder");
    const out = await sess.run({ [sess.inputNames[0]]: input });
    return toFloat32(out[sess.outputNames[0]]);
  }

  /** Audio encoder on log-mel features -> Float32Array [tokens * hidden]. */
  async encodeAudio(wave16k) {
    const { data, frames } = logMel(wave16k, this.mel, this.melFilters);
    const sess = await this.ensure("audio_encoder");
    const out = await sess.run({ input_features: new this.ort.Tensor("float32", data, [1, this.mel.num_mel_bins, frames]) });
    const feats = toFloat32(out.audio_features ?? out[sess.outputNames[0]]);
    const tokens = feats.length / this.hidden;
    if (tokens !== numAudioTokens(frames, this.mel)) throw new Error(`${frames} mel frames gave ${tokens} audio features`);
    return feats;
  }

  /** Prompt ids with placeholders expanded. imageGrids: [[1,gh,gw],...]; audioTokens: [n,...]. */
  expandPrompt(ids, imageGrids = [], audioTokens = []) {
    let out = expandTokens(ids, this.tokens.image, imageGrids.map((g) => imageTokenCount(g, this.image.merge)));
    out = expandTokens(out, this.tokens.audio, audioTokens);
    return out;
  }

  /**
   * Greedy text answer (generate_ids in text mode: placeholders, <|mixed_start|>
   * and <|audio_start|> suppressed; stops at <|im_end|> or <|text_end|>).
   * imageFeatures / audioFeatures: Float32Array rows in prompt order.
   */
  async generate(ids, { imageFeatures = null, audioFeatures = null, maxNewTokens = 256, onToken = () => {}, shouldStop = () => false } = {}) {
    const ort = this.ort, H = this.hidden, T = this.tokens;
    const rows = (f) => (f ? f.length / H : 0);
    for (const [tok, f, name] of [[T.image, imageFeatures, "image"], [T.audio, audioFeatures, "audio"]]) {
      const n = ids.filter((t) => t === tok).length;
      if (n !== rows(f)) throw new Error(`${n} ${name} placeholders, ${rows(f)} feature rows`);
    }
    const suppress = new Set([T.image, T.video, T.audio, T.audio_out, T.mixed_start, T.audio_start]);
    const dec = this.sessions.decoder;
    let cache = emptyCaches(ort, dec);
    let past = 0, imgRow = 0, audRow = 0;
    const feats = (name, src, start, n) => {
      const data = src ? src.subarray(start * H, (start + n) * H) : new Float32Array(0);
      return new ort.Tensor("float32", data, [n, H]);
    };
    const step = async (chunk) => {
      const n = chunk.length;
      const nImg = chunk.filter((t) => t === T.image).length, nAud = chunk.filter((t) => t === T.audio).length;
      const feed = {
        input_ids: new ort.Tensor("int64", i64(chunk), [1, n]),
        attention_mask: new ort.Tensor("int64", new BigInt64Array(past + n).fill(1n), [1, past + n]),
        position_ids: new ort.Tensor("int64", i64(Array.from({ length: n }, (_, i) => past + i)), [1, n]),
        ...cache,
      };
      const side = {
        logits_to_keep: () => new ort.Tensor("int64", new BigInt64Array([1n]), [1]),
        image_features: () => feats("image", imageFeatures, imgRow, nImg),
        video_features: () => feats("video", null, 0, 0),
        audio_features: () => feats("audio", audioFeatures, audRow, nAud),
        audio_out_codes: () => new ort.Tensor("int64", new BigInt64Array(0), [0, this.numCodebooks]),
      };
      for (const [name, make] of Object.entries(side)) if (this.decoderInputs.has(name)) feed[name] = make();
      const out = await dec.run(feed, ["logits", ...this.cacheOutputs]);
      for (const name of this.cacheOutputs) {
        const key = cacheInputName(name);
        if (past > 0 || cache[key].location === "gpu-buffer") cache[key].dispose?.();
        cache[key] = out[name];
      }
      imgRow += nImg; audRow += nAud; past += n;
      const logits = out.logits;
      const data = logits.location === "gpu-buffer" ? await logits.getData(true) : logits.data;
      const V = logits.dims[logits.dims.length - 1];
      return data.subarray(data.length - V);
    };

    const t0 = performance.now();
    let logits;
    for (let i = 0; i < ids.length; i += this.prefillChunk) logits = await step(ids.slice(i, i + this.prefillChunk));
    const prefillMs = performance.now() - t0;
    const tokens = [];
    for (let s = 0; s < maxNewTokens; s++) {
      const token = argmaxSuppressed(logits, suppress);
      tokens.push(token);
      onToken(token, tokens);
      if (token === T.eos || token === T.text_end || shouldStop()) break;
      if (s === maxNewTokens - 1) break;
      logits = await step([token]);
    }
    for (const t of Object.values(cache)) t.dispose?.();
    return { tokens, prefillMs, promptTokens: ids.length };
  }
}

function toFloat32(tensor) {
  if (tensor.type === "float16") return Float32Array.from(tensor.data, f16ToF32);
  return tensor.data instanceof Float32Array ? tensor.data : Float32Array.from(tensor.data);
}

// ---------------------------------------------------------------- browser loading --

const CACHE_NAME = "omni-pocket-onnx-v1";

/** Fetch with the Cache API (models are reused across visits) and byte progress. */
export async function cachedFetchBytes(url, onProgress = () => {}) {
  let cache = null;
  try { cache = await caches.open(CACHE_NAME); } catch (_) { /* no Cache API */ }
  let resp = cache ? await cache.match(url) : null;
  const fromCache = !!resp;
  if (!resp) {
    resp = await fetch(url);
    if (!resp.ok) throw new Error(`${resp.status} for ${url}`);
  }
  const total = Number(resp.headers.get("content-length")) || 0;
  const reader = resp.body.getReader();
  const parts = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); loaded += value.length;
    onProgress(loaded, total, fromCache);
  }
  const bytes = new Uint8Array(loaded);
  let off = 0;
  for (const p of parts) { bytes.set(p, off); off += p.length; }
  if (cache && !fromCache) {
    try { await cache.put(url, new Response(bytes, { headers: { "content-length": String(loaded) } })); } catch (_) { /* quota */ }
  }
  return bytes;
}

/** loadModel/loadJson for a Hugging Face repo: lists onnx/ to find each graph's .onnx_data chunks. */
export function huggingFaceSource(repo, { revision = "main", onProgress = () => {} } = {}) {
  const base = `https://huggingface.co/${repo}/resolve/${revision}`;
  let listing = null;
  const list = async () => {
    if (!listing) {
      const r = await fetch(`https://huggingface.co/api/models/${repo}/tree/${revision}/onnx`);
      listing = r.ok ? (await r.json()).map((e) => e.path.split("/").pop()) : [];
    }
    return listing;
  };
  return {
    async loadJson(name) {
      const r = await fetch(`${base}/${name}`);
      return r.ok ? r.json() : null;
    },
    async loadModel(file) {
      const stem = file.replace(/\.onnx$/, "");
      const files = await list();
      if (files.length && !files.includes(file)) throw new Error(`${repo} has no onnx/${file}`);
      const dataFiles = files.filter((f) => f === `${stem}.onnx_data` || f.startsWith(`${stem}.onnx_data_`)).sort();
      const model = await cachedFetchBytes(`${base}/onnx/${file}`, (l, t, c) => onProgress(file, l, t, c));
      const externalData = [];
      for (const f of dataFiles) {
        externalData.push({ path: f, data: await cachedFetchBytes(`${base}/onnx/${f}`, (l, t, c) => onProgress(f, l, t, c)) });
      }
      return { model, externalData };
    },
  };
}
