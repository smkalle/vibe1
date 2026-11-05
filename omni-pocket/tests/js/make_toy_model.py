"""A toy LFM3-Omni-shaped ONNX bundle + a numpy reference of its greedy loop, to test the
plumbing of omnipocket/webgpu/lfm3omni.js on onnxruntime-web without the real weights:

* decoder.onnx has the real decoder's interface: input_ids / attention_mask / position_ids /
  logits_to_keep, image/video/audio_features scattered over their placeholder tokens (ScatterND
  at NonZero(ids == token), as LiquidAI's decoder does), audio_out_codes, a conv cache
  (past_conv.0 [1, H, L] -> present_conv.0) and a KV cache (past_key_values.1.key/value
  [1, 1, T, H] -> present.1.key/value), logits [1, 1, V] and hidden_states.
* vision_encoder.onnx: pixel_values [h/2, w/2, 2, 2, D] -> [h*w/4, H].
* audio_encoder.onnx: input_features [1, 128, T] -> audio_features [ceil(T / 8), H].

The logits depend on every cached position, the conv cache, the positions and the scattered
features, so a wrong cache handoff, chunk split, feature slice or position shows up as a different
token sequence. reference.json holds the numpy loop's tokens.

    python tests/js/make_toy_model.py OUT_DIR
    OMNI_TOY=OUT_DIR node --test tests/js/test_toy_runtime.mjs
"""

import json
import pathlib
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

H, V, L, D = 8, 64, 3, 16 * 16 * 3 * 2
TOK = {"bos": 1, "eos": 54, "text_end": 55, "mixed_start": 56, "audio_start": 57, "audio_out": 58,
       "video": 59, "image": 60, "audio": 61, "pad": 0}
RNG = np.random.default_rng(42)
E = RNG.standard_normal((V, H)).astype(np.float32)
W = RNG.standard_normal((H, V)).astype(np.float32)
WC = RNG.standard_normal((H,)).astype(np.float32)
POS = RNG.uniform(0.3, 1.7, (H,)).astype(np.float32)  # position frequencies
WV = (RNG.standard_normal((4 * D, H)) * 0.02).astype(np.float32)
WA = (RNG.standard_normal((128, H)) * 0.3).astype(np.float32)


def init(name, arr):
    return numpy_helper.from_array(arr, name)


def scatter(nodes, emb_in, emb_out, feats, token, tag):
    nodes += [
        helper.make_node("Constant", [], [f"{tag}_tok"], value=numpy_helper.from_array(np.array(token, np.int64))),
        helper.make_node("Equal", ["input_ids", f"{tag}_tok"], [f"{tag}_mask"]),
        helper.make_node("NonZero", [f"{tag}_mask"], [f"{tag}_nz"]),
        helper.make_node("Transpose", [f"{tag}_nz"], [f"{tag}_idx"], perm=[1, 0]),
        helper.make_node("ScatterND", [emb_in, f"{tag}_idx", feats], [emb_out]),
    ]


def decoder() -> onnx.ModelProto:
    n = []
    n += [helper.make_node("Gather", ["E", "input_ids"], ["emb0"], axis=0)]
    scatter(n, "emb0", "emb1", "image_features", TOK["image"], "img")
    scatter(n, "emb1", "emb2", "audio_features", TOK["audio"], "aud")
    scatter(n, "emb2", "emb3", "video_features", TOK["video"], "vid")
    # positions: emb += position * POS
    n += [helper.make_node("Cast", ["position_ids"], ["posf"], to=TensorProto.FLOAT),
          helper.make_node("Unsqueeze", ["posf", "axes_m1"], ["pos3"]),
          helper.make_node("Mul", ["pos3", "POS"], ["posarg"]),
          helper.make_node("Sin", ["posarg"], ["posv"]),
          helper.make_node("Add", ["emb3", "posv"], ["emb"]),
          helper.make_node("Identity", ["emb"], ["hidden_states"])]
    # KV cache [1, 1, T, H]
    n += [helper.make_node("Unsqueeze", ["emb", "axes_1"], ["emb4"]),
          helper.make_node("Concat", ["past_key_values.1.key", "emb4"], ["present.1.key"], axis=2),
          helper.make_node("Mul", ["emb4", "two"], ["emb4x2"]),
          helper.make_node("Concat", ["past_key_values.1.value", "emb4x2"], ["present.1.value"], axis=2)]
    # conv cache [1, H, L]: keep the last L columns of [past | emb^T]
    n += [helper.make_node("Transpose", ["emb"], ["embT"], perm=[0, 2, 1]),
          helper.make_node("Concat", ["past_conv.0", "embT"], ["convcat"], axis=2),
          helper.make_node("Slice", ["convcat", "minusL", "big", "axes_2"], ["present_conv.0"])]
    # logits from the last position: emb_last + mean(key cache) + 0.1 * sum(value) + conv . WC
    n += [helper.make_node("Slice", ["emb", "minus1", "big", "axes_1"], ["last"]),           # [1,1,H]
          helper.make_node("ReduceMean", ["present.1.key", "axes_2"], ["kmean"], keepdims=0),  # [1,1,H]
          helper.make_node("ReduceMean", ["present.1.value", "axes_2"], ["vsum"], keepdims=0),
          helper.make_node("Mul", ["vsum", "tenth"], ["vsum10"]),
          helper.make_node("Mul", ["present_conv.0", "WC3"], ["convw"]),
          helper.make_node("ReduceSum", ["convw", "axes_1"], ["convs"], keepdims=1),          # [1,1,L]
          helper.make_node("ReduceSum", ["convs", "axes_2"], ["convscalar"], keepdims=1),     # [1,1,1]
          helper.make_node("Add", ["last", "kmean"], ["h1"]),
          helper.make_node("Add", ["h1", "vsum10"], ["h2"]),
          helper.make_node("Add", ["h2", "convscalar"], ["h3"]),
          helper.make_node("MatMul", ["h3", "W"], ["logits"])]
    # unused-but-required inputs consumed harmlessly
    n += [helper.make_node("Shape", ["attention_mask"], ["am_shape"]),
          helper.make_node("Shape", ["logits_to_keep"], ["ltk_shape"]),
          helper.make_node("Shape", ["audio_out_codes"], ["aoc_shape"])]
    consts = [init("E", E), init("W", W), init("POS", POS.reshape(1, 1, H)), init("WC3", WC.reshape(1, H, 1)),
              init("axes_m1", np.array([-1], np.int64)), init("axes_1", np.array([1], np.int64)),
              init("axes_2", np.array([2], np.int64)), init("minusL", np.array([-L], np.int64)),
              init("minus1", np.array([-1], np.int64)), init("big", np.array([2**62], np.int64)),
              init("two", np.array(2.0, np.float32)), init("tenth", np.array(0.1, np.float32))]
    f, i64 = TensorProto.FLOAT, TensorProto.INT64
    inputs = [helper.make_tensor_value_info("input_ids", i64, [1, "sequence_length"]),
              helper.make_tensor_value_info("attention_mask", i64, [1, "total_sequence_length"]),
              helper.make_tensor_value_info("position_ids", i64, [1, "sequence_length"]),
              helper.make_tensor_value_info("logits_to_keep", i64, [1]),
              helper.make_tensor_value_info("image_features", f, ["num_image_tokens", H]),
              helper.make_tensor_value_info("video_features", f, ["num_video_tokens", H]),
              helper.make_tensor_value_info("audio_features", f, ["num_audio_tokens", H]),
              helper.make_tensor_value_info("audio_out_codes", i64, ["num_frames", 12]),
              helper.make_tensor_value_info("past_conv.0", f, ["batch_size", H, L]),
              helper.make_tensor_value_info("past_key_values.1.key", f, ["batch_size", 1, "past_sequence_length", H]),
              helper.make_tensor_value_info("past_key_values.1.value", f, ["batch_size", 1, "past_sequence_length", H])]
    outputs = [helper.make_tensor_value_info("logits", f, [1, 1, V]),
               helper.make_tensor_value_info("hidden_states", f, [1, "sequence_length", H]),
               helper.make_tensor_value_info("present_conv.0", f, ["batch_size", H, L]),
               helper.make_tensor_value_info("present.1.key", f, ["batch_size", 1, "total_sequence_length", H]),
               helper.make_tensor_value_info("present.1.value", f, ["batch_size", 1, "total_sequence_length", H])]
    graph = helper.make_graph(n, "toy_decoder", inputs, outputs, consts)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=8)


def vision() -> onnx.ModelProto:
    n = [helper.make_node("Constant", [], ["shape"], value=numpy_helper.from_array(np.array([-1, 4 * D], np.int64))),
         helper.make_node("Reshape", ["pixel_values", "shape"], ["flat"]),
         helper.make_node("MatMul", ["flat", "WV"], ["image_features"])]
    g = helper.make_graph(n, "toy_vision",
                          [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, ["h", "w", 2, 2, D])],
                          [helper.make_tensor_value_info("image_features", TensorProto.FLOAT, ["n", H])], [init("WV", WV)])
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)], ir_version=8)


def audio() -> onnx.ModelProto:
    n = [helper.make_node("AveragePool", ["input_features"], ["pooled"], kernel_shape=[8], strides=[8], ceil_mode=1),
         helper.make_node("Transpose", ["pooled"], ["pt"], perm=[0, 2, 1]),
         helper.make_node("MatMul", ["pt", "WA"], ["af3"]),
         helper.make_node("Squeeze", ["af3", "axes0"], ["audio_features"])]
    g = helper.make_graph(n, "toy_audio",
                          [helper.make_tensor_value_info("input_features", TensorProto.FLOAT, [1, 128, "frames"])],
                          [helper.make_tensor_value_info("audio_features", TensorProto.FLOAT, ["n", H])],
                          [init("WA", WA), init("axes0", np.array([0], np.int64))])
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)], ir_version=8)


# ---- numpy reference of the same math and the greedy loop (unchunked prefill) ----

def ref_vision(pixel_values):
    return pixel_values.reshape(-1, 4 * D) @ WV


def ref_audio(mel):  # mel [128, T]
    T = mel.shape[1]
    n = -(T // -8)
    pooled = np.stack([mel[:, 8 * i: min(T, 8 * i + 8)].mean(axis=1) for i in range(n)])  # ceil_mode averages the partial window
    return (pooled @ WA).astype(np.float32)


def ref_generate(ids, image_feats, audio_feats, max_new=24):
    keys, values, conv = np.zeros((0, H), np.float32), np.zeros((0, H), np.float32), np.zeros((H, L), np.float32)
    suppress = {TOK[k] for k in ("image", "video", "audio", "audio_out", "mixed_start", "audio_start")}

    def step(chunk, past, img, aud):
        nonlocal keys, values, conv
        emb = E[chunk].copy()
        for tok, feats in ((TOK["image"], img), (TOK["audio"], aud)):
            pos = [i for i, t in enumerate(chunk) if t == tok]
            if pos:
                emb[pos] = feats[: len(pos)]
        emb += np.sin(np.arange(past, past + len(chunk), dtype=np.float32)[:, None] * POS)
        keys = np.concatenate([keys, emb]); values = np.concatenate([values, 2 * emb])
        conv = np.concatenate([conv, emb.T], axis=1)[:, -L:]
        h = emb[-1] + keys.mean(axis=0) + 0.1 * values.mean(axis=0) + (conv * WC[:, None]).sum()
        return h @ W

    logits = step(ids, 0, image_feats, audio_feats)
    past, out = len(ids), []
    for s in range(max_new):
        masked = logits.copy()
        masked[list(suppress)] = -np.inf
        tok = int(np.argmax(masked))
        out.append(tok)
        if tok in (TOK["eos"], TOK["text_end"]) or s == max_new - 1:
            break
        logits = step([tok], past, None, None)
        past += 1
    return out


def main(out: pathlib.Path):
    onnx_dir = out / "model" / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx.save(decoder(), onnx_dir / "decoder.onnx", save_as_external_data=True, location="decoder.onnx_data",
              size_threshold=1024)
    onnx.save(vision(), onnx_dir / "vision_encoder.onnx")
    onnx.save(audio(), onnx_dir / "audio_encoder.onnx")
    cfg = {"model_type": "lfm3_omni", "image_token_id": TOK["image"], "video_token_id": TOK["video"],
           "audio_token_id": TOK["audio"], "audio_out_token_id": TOK["audio_out"],
           "audio_start_token_id": TOK["audio_start"], "text_end_token_id": TOK["text_end"],
           "mixed_start_token_id": TOK["mixed_start"], "num_codebooks": 12,
           "text_config": {"hidden_size": H, "bos_token_id": TOK["bos"], "eos_token_id": TOK["eos"], "pad_token_id": 0}}
    (out / "model" / "config.json").write_text(json.dumps(cfg))

    rng = np.random.default_rng(7)
    grid = [1, 4, 6]  # 6 image tokens
    pixel_values = rng.standard_normal((grid[1] * grid[2], D)).astype(np.float32)
    mel = rng.standard_normal((128, 45)).astype(np.float32)  # 45 frames -> 6 audio tokens
    img = ref_vision(pixel_values.reshape(grid[1] // 2, grid[2] // 2, 2, 2, D))
    aud = ref_audio(mel)
    prompt = [TOK["bos"], 3, 4, TOK["image"], 5, TOK["audio"], 6, 7, 8]
    ids = []
    for t in prompt:
        ids += [t] * {TOK["image"]: len(img), TOK["audio"]: len(aud)}.get(t, 1)
    tokens = ref_generate(ids, img, aud)

    # cross-check the numpy reference against onnxruntime on the same graphs
    import onnxruntime as ort
    dec = ort.InferenceSession(str(onnx_dir / "decoder.onnx"))
    feed = {"input_ids": np.array([ids], np.int64), "attention_mask": np.ones((1, len(ids)), np.int64),
            "position_ids": np.arange(len(ids), dtype=np.int64)[None], "logits_to_keep": np.array([1], np.int64),
            "image_features": img, "video_features": np.zeros((0, H), np.float32), "audio_features": aud,
            "audio_out_codes": np.zeros((0, 12), np.int64), "past_conv.0": np.zeros((1, H, L), np.float32),
            "past_key_values.1.key": np.zeros((1, 1, 0, H), np.float32),
            "past_key_values.1.value": np.zeros((1, 1, 0, H), np.float32)}
    suppress = [TOK[k] for k in ("image", "video", "audio", "audio_out", "mixed_start", "audio_start")]
    ort_tokens, past = [], len(ids)
    outs = dec.run(None, feed)
    for _ in range(len(tokens)):
        logits = outs[0][0, -1].copy()
        logits[suppress] = -np.inf
        tok = int(np.argmax(logits))
        ort_tokens.append(tok)
        if tok in (TOK["eos"], TOK["text_end"]):
            break
        names = [o.name for o in dec.get_outputs()]
        cache = {n.replace("present_conv.", "past_conv.").replace("present.", "past_key_values."): v
                 for n, v in zip(names, outs) if n.startswith("present")}
        feed = {**feed, **cache, "input_ids": np.array([[tok]], np.int64),
                "attention_mask": np.ones((1, past + 1), np.int64), "position_ids": np.array([[past]], np.int64),
                "image_features": np.zeros((0, H), np.float32), "audio_features": np.zeros((0, H), np.float32)}
        outs = dec.run(None, feed)
        past += 1
    assert ort_tokens == tokens, (ort_tokens, tokens)
    ort_aud = ort.InferenceSession(str(onnx_dir / "audio_encoder.onnx")).run(None, {"input_features": mel[None]})[0]
    np.testing.assert_allclose(ort_aud, aud, rtol=1e-5, atol=1e-5)

    data = out / "data"
    data.mkdir(exist_ok=True)
    pixel_values.tofile(data / "pixel_values.f32")
    mel.tofile(data / "mel.f32")
    img.astype(np.float32).tofile(data / "image_features.f32")
    aud.astype(np.float32).tofile(data / "audio_features.f32")
    (out / "reference.json").write_text(json.dumps({"grid": grid, "prompt": prompt, "ids": ids, "tokens": tokens,
                                                    "mel_frames": mel.shape[1], "tok": TOK}, indent=1))
    print("tokens", tokens)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
