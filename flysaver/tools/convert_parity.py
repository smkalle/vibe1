#!/usr/bin/env python3
"""Convert the Cadence library's parity cases (cadence-examples/fly-matrix/tests/parity_cases.json,
produced by cadence.Brain) into a plain-text fixture the Rust parity test reads without a JSON crate.

    case <name>
    stim <population> <level>          (repeated)
    readouts <population> ...
    final_active <n>
    step <mean> <mean> ...             (one line per step, repr() so f64 round-trips exactly)
    end

usage: python3 tools/convert_parity.py path/to/parity_cases.json > tests/parity_cases.txt
"""
import json
import sys

d = json.load(open(sys.argv[1]))
print(f"# from cadence-examples/fly-matrix/tests/parity_cases.json (cadence.Brain), gain {d['gain']}")
for c in d["cases"]:
    print(f"case {c['name']}")
    for pop, level in c["stimulus"].items():
        print(f"stim {pop} {float(level)!r}")
    print("readouts " + " ".join(c["readouts"]))
    print(f"final_active {c['final_active']}")
    for row in c["per_step_means"]:
        print("step " + " ".join(repr(float(x)) for x in row))
    print("end")
