#!/usr/bin/env python3
"""Bake the connectome point cloud that flysaver embeds.

Reads the atlas that ships with "A fly in the Matrix"
(cadence-examples/fly-matrix/web/data/atlas.json, BANC release 888, CC BY)
and writes assets/connectome.bin: a stratified downsample of the soma
positions, quantised so the binary stays small.

Format (little endian):
    b"FLYC"  u32 version=1  u32 count  u8 region_count
    region_count x { u8 name_len, name bytes }
    count x { u16 x, u16 y, u16 z, u8 region }
Coordinates map [-1, 1] to [0, 65535]; y is the long (head to nerve cord) axis.

Stdlib only:  python3 tools/build_connectome.py path/to/atlas.json [--points 20000]
"""

import argparse
import array
import base64
import json
import random
import struct
from pathlib import Path


def decode(spec, typecode):
    a = array.array(typecode)
    a.frombytes(base64.b64decode(spec["b64"]))
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("atlas")
    ap.add_argument("--points", type=int, default=20000)
    ap.add_argument("--min-per-region", type=int, default=40)
    ap.add_argument("--seed", type=int, default=888)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "assets" / "connectome.bin"))
    args = ap.parse_args()

    atlas = json.loads(Path(args.atlas).read_text())
    n = atlas["n"]
    pos = decode(atlas["positions3"], "f")
    region = decode(atlas["region"], "H")
    names = [r["name"] for r in atlas["regions"]]
    assert len(pos) == 3 * n and len(region) == n

    by_region = [[] for _ in names]
    for i in range(n):
        by_region[region[i]].append(i)

    # Proportional allocation with a floor, so small sensory and motor groups
    # (ocelli, halteres, motor neurons) still show up in the cloud.
    rng = random.Random(args.seed)
    picked = []
    for r, members in enumerate(by_region):
        want = max(args.min_per_region, round(args.points * len(members) / n))
        want = min(want, len(members))
        picked.extend(rng.sample(members, want))
    rng.shuffle(picked)  # runtime subsampling can then just take a prefix

    scale = max(abs(v) for v in pos) or 1.0
    q = lambda v: max(0, min(65535, round((v / scale + 1.0) * 0.5 * 65535)))

    out = bytearray(b"FLYC")
    out += struct.pack("<IIB", 1, len(picked), len(names))
    for name in names:
        b = name.encode()
        out += struct.pack("<B", len(b)) + b
    for i in picked:
        out += struct.pack("<HHHB", q(pos[3 * i]), q(pos[3 * i + 1]), q(pos[3 * i + 2]), region[i])

    Path(args.out).write_bytes(out)
    print(f"wrote {args.out}: {len(picked)} of {n} neurons, {len(out)} bytes")


if __name__ == "__main__":
    main()
