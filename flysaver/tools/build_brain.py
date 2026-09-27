#!/usr/bin/env python3
"""Bake the live brain that flysaver embeds.

Reads the sub-net the original page settles (cadence-examples/fly-matrix/web/data/brain.json,
60,000 neurons of BANC release 888, CC BY) and the atlas (atlas.json, for soma positions), and
writes assets/brain.bin. Weights are not stored: they are rebuilt exactly as the Cadence library
composes them, (gain * count * exp(log_gain[pre])) * efficacy, which reproduces the payload's f64
weights bit for bit. A 64-bit FNV-1a of those weights is stored so flysaver can prove it.

Format (little endian; varint = unsigned LEB128):
    b"FLYB" u32 version=1 u32 n u32 edges
    f64 dt slope threshold gain amplitude leak
    u64 fnv1a64(weights as f64 LE)
    u8 k, k * f64 log_gain values, n * u8 log_gain class
    u32 m, m * (u32 neuron, f64 bias)                  -- nonzero biases
    n * varint row length
    edges * varint sender (first in a row absolute, then delta; senders are sorted per row)
    edges * varint synapse count
    ceil(edges/4) bytes of 2-bit signs (0 -> 0, 1 -> +1, 2 -> -1)
    u32 m, m * (u32 edge, f64 efficacy)                -- efficacies off their sign
    n * (u16 x, u16 y, u16 z, u8 region)               -- atlas soma position, [-1,1] -> [0,65535]
    u32 p, p * (u8 len, name, varint count, count * varint delta index)

Stdlib only:  python3 tools/build_brain.py path/to/web/data
"""

import array
import base64
import json
import math
import struct
import sys
from pathlib import Path


def dec(spec_or_b64, typecode):
    a = array.array(typecode)
    a.frombytes(base64.b64decode(spec_or_b64["b64"] if isinstance(spec_or_b64, dict) else spec_or_b64))
    return a


def varint(out, v):
    assert v >= 0
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return


def fnv1a64(data):
    h = 0xCBF29CE484222325
    for byte in data:
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def main():
    data = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out_path = Path(__file__).resolve().parent.parent / "assets" / "brain.bin"
    brain = json.loads((data / "brain.json").read_text())
    atlas = json.loads((data / "atlas.json").read_text())
    A, m = brain["arrays"], brain["model"]
    assert A.get("sign_dtype") == "int8" and not m.get("adaptation")
    n, E = brain["n"], brain["edges"]
    row_ptr, pre, weight = dec(A["row_ptr"], "i"), dec(A["pre"], "i"), dec(A["weight"], "d")
    count, sign = dec(A["count"], "H"), dec(A["sign"], "b")
    bias, log_gain = dec(A["bias"], "d"), dec(A["log_gain"], "d")
    eff_idx, eff_val = dec(A["efficacy_index"], "i"), dec(A["efficacy_value"], "d")
    members = dec(A["members"], "i")

    # The weights must be exactly what we rebuild at run time.
    efficacy = array.array("d", (float(s) for s in sign))
    for i, v in zip(eff_idx, eff_val):
        efficacy[i] = v
    gain = m["gain"]
    for e in range(E):
        assert (gain * count[e] * math.exp(log_gain[pre[e]])) * efficacy[e] == weight[e], f"weight {e} does not rebuild"
    checksum = fnv1a64(weight.tobytes())

    out = bytearray(b"FLYB")
    out += struct.pack("<III", 1, n, E)
    out += struct.pack("<6d", m["dt"], m["slope"], m["threshold"], gain, m["stimulus_amplitude"], m.get("leak") or 0.0)
    out += struct.pack("<Q", checksum)

    classes = sorted(set(log_gain))
    assert len(classes) < 256
    out += struct.pack("<B", len(classes)) + b"".join(struct.pack("<d", c) for c in classes)
    out += bytes(classes.index(g) for g in log_gain)

    nz = [(i, b) for i, b in enumerate(bias) if b != 0.0]
    out += struct.pack("<I", len(nz)) + b"".join(struct.pack("<Id", i, b) for i, b in nz)

    for i in range(n):
        varint(out, row_ptr[i + 1] - row_ptr[i])
    for i in range(n):
        last = 0
        for e in range(row_ptr[i], row_ptr[i + 1]):
            assert pre[e] >= last
            varint(out, pre[e] - last)
            last = pre[e]
    for e in range(E):
        varint(out, count[e])
    packed = bytearray((E + 3) // 4)
    for e, s in enumerate(sign):
        packed[e >> 2] |= {0: 0, 1: 1, -1: 2}[s] << (2 * (e & 3))
    out += packed

    out += struct.pack("<I", len(eff_idx)) + b"".join(struct.pack("<Id", i, v) for i, v in zip(eff_idx, eff_val))

    pos3, region = dec(atlas["positions3"], "f"), dec(atlas["region"], "H")
    scale = max(abs(v) for v in pos3) or 1.0
    q = lambda v: max(0, min(65535, round((v / scale + 1.0) * 0.5 * 65535)))
    for k in members:
        out += struct.pack("<HHHB", q(pos3[3 * k]), q(pos3[3 * k + 1]), q(pos3[3 * k + 2]), region[k])

    pops = brain["populations"]
    out += struct.pack("<I", len(pops))
    for name, idx in pops.items():
        b = name.encode()
        out += struct.pack("<B", len(b)) + b
        idx = sorted(idx)
        varint(out, len(idx))
        last = 0
        for i in idx:
            varint(out, i - last)
            last = i

    out_path.write_bytes(out)
    print(f"wrote {out_path}: {n} neurons, {E} synapse classes, {len(pops)} populations, {len(out) / 1e6:.2f} MB, weights fnv1a64 {checksum:016x}")


if __name__ == "__main__":
    main()
