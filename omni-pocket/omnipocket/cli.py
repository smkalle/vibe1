"""Command line: ``python -m omnipocket {doctor,ask,bench,serve}``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ENGINES, Settings
from .engines import OmniRequest, make_engine
from .history import logs, summarize
from .media import normalize_speech, prepare_image
from .platform import report


def _engine(args, settings: Settings):
    if getattr(args, "remote_url", None):
        settings.remote_url = args.remote_url
    name = args.engine or settings.resolve_engine()
    return make_engine(name, settings)


def _request(args) -> OmniRequest:
    req = OmniRequest(text=args.text or "", max_new_tokens=args.max_new_tokens)
    if args.system:
        req.system = args.system
    if args.image:
        req.image = prepare_image(Path(args.image).read_bytes())
    if args.audio:
        req.audio = normalize_speech(Path(args.audio).read_bytes())
    if req.is_empty():
        sys.exit("give at least one of --text, --image, --audio")
    return req


def cmd_doctor(args, settings):
    r = report(str(settings.data_dir.parent if settings.data_dir.parent.exists() else "."))
    if args.json:
        print(json.dumps(r.as_dict(), indent=2))
        return
    print(f"environment : {r.env} ({r.machine}, Python {r.python})")
    print(f"RAM         : {r.total_ram_mb} MB total, {r.available_ram_mb} MB available")
    print(f"free disk   : {r.free_disk_gb} GB")
    print(f"torch device: {r.torch_device}")
    print("modules     : " + ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in r.modules.items()))
    print(f"auto engine : {settings.resolve_engine()}")
    for n in r.notes:
        print(f"  - {n}")


def cmd_ask(args, settings):
    engine = _engine(args, settings)
    req = _request(args)
    if args.stream:
        for piece in engine.stream(req):
            print(piece, end="", flush=True)
        print()
        return
    res = engine.generate(req)
    print(res.text)
    tps = f"{res.tokens_per_s:.1f} tok/s" if res.tokens_per_s else "n/a"
    print(f"\n[{res.engine} {res.model}] ttft={res.ttft_s or 0:.2f}s total={res.total_s:.2f}s {tps}",
          file=sys.stderr)
    if res.audio_wav and args.save_audio:
        Path(args.save_audio).write_bytes(res.audio_wav)


def cmd_bench(args, settings):
    engine = _engine(args, settings)
    req = _request(args)
    _, bench = logs(settings.data_dir)
    mode = "+".join(req.modalities())
    for i in range(args.warmup + args.runs):
        res = engine.generate(req)
        if i < args.warmup:
            print(f"warmup {i + 1}: {res.total_s:.2f}s")
            continue
        row = bench.append({"engine": res.engine, "model": res.model, "mode": mode, "source": "cli",
                            "ttft_s": res.ttft_s, "total_s": round(res.total_s, 3),
                            "output_tokens": res.output_tokens,
                            "tokens_per_s": round(res.tokens_per_s, 2) if res.tokens_per_s else None})
        print(f"run {i - args.warmup + 1}: ttft={row['ttft_s'] or 0:.2f}s total={row['total_s']:.2f}s "
              f"tok/s={row['tokens_per_s']}")
    print(json.dumps(summarize(bench.read()), indent=2))


def cmd_serve(args, settings):
    from .serve import run

    engine = make_engine(args.engine or "transformers", settings)
    run(engine, host=args.host, port=args.port)


def main(argv=None):
    settings = Settings.from_env()
    p = argparse.ArgumentParser(prog="omnipocket", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="check this device and recommend an engine")
    d.add_argument("--json", action="store_true")

    for name in ("ask", "bench"):
        s = sub.add_parser(name, help="one request" if name == "ask" else "repeat a request and log latency")
        s.add_argument("--engine", choices=ENGINES)
        s.add_argument("--remote-url")
        s.add_argument("--text")
        s.add_argument("--image")
        s.add_argument("--audio", help="WAV file")
        s.add_argument("--system")
        s.add_argument("--max-new-tokens", type=int, default=settings.max_new_tokens)
        if name == "ask":
            s.add_argument("--stream", action="store_true")
            s.add_argument("--save-audio", help="write model speech (if any) to this WAV")
        else:
            s.add_argument("--runs", type=int, default=3)
            s.add_argument("--warmup", type=int, default=1)

    v = sub.add_parser("serve", help="OpenAI-compatible server for the remote engine")
    v.add_argument("--engine", choices=ENGINES)
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8088)

    args = p.parse_args(argv)
    {"doctor": cmd_doctor, "ask": cmd_ask, "bench": cmd_bench, "serve": cmd_serve}[args.cmd](args, settings)


if __name__ == "__main__":
    main()
