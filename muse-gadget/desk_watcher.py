#!/usr/bin/env python3
"""Desk Watcher v1 — webcam motion snapshots for the Muse news/camera gadget.

Works standalone (no Meta SDK needed). When the Meta Muse Gadget Linux
daemon (`musegadget`) is installed + paired, motion events are forwarded
with `musegadget send-user-msg`; otherwise they are logged locally and
snapshots are kept in ./snapshots/.

Usage:
    python3 desk_watcher.py --once          # single test capture
    python3 desk_watcher.py --watch         # loop until Ctrl-C
    python3 desk_watcher.py --watch --no-send  # never call musegadget

Config: watcher_config.json (same directory), section "desk_watcher".
Camera verified on OMAKALLE: /dev/video0 Integrated_Webcam_HD.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "watcher_config.json"
SNAP_DIR = HERE / "snapshots"


def load_config() -> dict:
    defaults = {
        "camera_index": 0,
        "width": 640,
        "height": 480,
        "interval_sec": 2.0,
        "cooldown_sec": 60,
        "min_area": 2500,
        "diff_threshold": 25,
        "blur": 21,
        "snapshot_dir": "snapshots",
        "send_enabled": True,
    }
    try:
        cfg = json.loads(CONFIG_PATH.read_text())
        defaults.update(cfg.get("desk_watcher", {}))
    except FileNotFoundError:
        pass
    except json.JSONDecodeError as exc:
        print(f"WARN: bad {CONFIG_PATH}: {exc}, using defaults", file=sys.stderr)
    return defaults


def capture_frame(cfg: dict):
    """Return a BGR frame via OpenCV, or None on failure."""
    try:
        import cv2
    except ImportError:
        print("ERROR: opencv (cv2) not installed", file=sys.stderr)
        return None
    cap = cv2.VideoCapture(cfg["camera_index"])
    cap.set(3, cfg["width"])
    cap.set(4, cfg["height"])
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def save_snapshot(frame, tag: str, snap_dir: Path) -> Path:
    import cv2

    snap_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = snap_dir / f"{tag}-{ts}.jpg"
    cv2.imwrite(str(path), frame)
    return path


def motion_score(prev_gray, gray, cfg: dict) -> tuple[bool, float]:
    """Frame-diff motion test. Returns (triggered, changed_area)."""
    import cv2

    delta = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(delta, cfg["diff_threshold"], 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > cfg["min_area"])
    return area > cfg["min_area"], area


def maybe_send(message: str, send_enabled: bool) -> str:
    """Forward to Muse via musegadget if present, else log. Returns action taken."""
    exe = shutil.which("musegadget")
    if send_enabled and exe:
        try:
            subprocess.run([exe, "send-user-msg", message], check=True, timeout=15)
            return "sent via musegadget"
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            print(f"WARN: musegadget send failed: {exc}", file=sys.stderr)
            return "send-failed (logged)"
    log = HERE / "events.log"
    with log.open("a") as fh:
        fh.write(f"{datetime.now().isoformat()} {message}\n")
    return "logged locally (musegadget not installed)"


def cmd_once(cfg: dict, snap_dir: Path) -> int:
    import cv2

    frame = capture_frame(cfg)
    if frame is None:
        print("ERROR: camera capture failed (is /dev/video0 busy?)", file=sys.stderr)
        return 1
    path = save_snapshot(frame, "test", snap_dir)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(f"OK capture {frame.shape[1]}x{frame.shape[0]} -> {path} ({path.stat().st_size} bytes)")
    return 0


def cmd_watch(cfg: dict, snap_dir: Path, no_send: bool) -> int:
    import cv2

    print(f"Watching camera {cfg['camera_index']} every {cfg['interval_sec']}s "
          f"(cooldown {cfg['cooldown_sec']}s). Ctrl-C to stop.")
    prev_gray = None
    last_alert = 0.0
    # Warm-up frame
    frame = capture_frame(cfg)
    if frame is None:
        print("ERROR: camera capture failed", file=sys.stderr)
        return 1
    prev_gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                   (cfg["blur"], cfg["blur"]), 0)
    try:
        while True:
            time.sleep(cfg["interval_sec"])
            frame = capture_frame(cfg)
            if frame is None:
                print("WARN: capture failed, retrying", file=sys.stderr)
                continue
            gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                    (cfg["blur"], cfg["blur"]), 0)
            triggered, area = motion_score(prev_gray, gray, cfg)
            prev_gray = gray
            now = time.time()
            if triggered and (now - last_alert) >= cfg["cooldown_sec"]:
                last_alert = now
                path = save_snapshot(frame, "motion", snap_dir)
                msg = (f"Desk Watcher: motion detected (area {area:.0f}) "
                       f"snapshot {path.name}")
                action = maybe_send(msg, send_enabled=(cfg["send_enabled"] and not no_send))
                print(f"{datetime.now().isoformat()} MOTION area={area:.0f} {path.name} [{action}]")
            else:
                print(f"{datetime.now().isoformat()} quiet area={area:.0f}", flush=True)
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Desk Watcher v1")
    ap.add_argument("--once", action="store_true", help="single test capture and exit")
    ap.add_argument("--watch", action="store_true", help="loop motion detection")
    ap.add_argument("--no-send", action="store_true", help="never call musegadget")
    args = ap.parse_args(argv)

    cfg = load_config()
    snap_dir = HERE / cfg.get("snapshot_dir", "snapshots")

    if args.once or not args.watch:
        return cmd_once(cfg, snap_dir)
    return cmd_watch(cfg, snap_dir, no_send=args.no_send)


if __name__ == "__main__":
    sys.exit(main())
