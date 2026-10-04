#!/usr/bin/env python3
"""Health Sentinel v1 — sensors/disk/memory/uptime checks for this Linux box.

Stdlib only. Prints JSON, exits 0 when all OK, 2 when any threshold trips.
Optional --send forwards alerts via `musegadget send-user-msg` when the
Meta Muse Gadget daemon is installed; otherwise alerts just print.

Usage:
    python3 health_sentinel.py --check
    python3 health_sentinel.py --check --send
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "watcher_config.json"


def load_thresholds() -> dict:
    defaults = {
        "temp_crit": 90.0,
        "disk_min_free_gb": 5.0,
        "mem_min_avail_mb": 500,
    }
    try:
        cfg = json.loads(CONFIG_PATH.read_text())
        defaults.update(cfg.get("health_sentinel", {}))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return defaults


def read_temps() -> list[dict]:
    """Best-effort temps from /sys/class/hwmon + thermal zones."""
    out: list[dict] = []
    for hwmon in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        name = (hwmon / "name").read_text().strip() if (hwmon / "name").exists() else hwmon.name
        for temp_input in sorted(hwmon.glob("temp*_input")):
            try:
                milli = int(temp_input.read_text().strip())
                out.append({"chip": name, "sensor": temp_input.name, "celsius": milli / 1000.0})
            except (ValueError, OSError):
                continue
    if not out:
        for tz in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
            try:
                milli = int((tz / "temp").read_text().strip())
                out.append({"chip": tz.name, "sensor": "temp", "celsius": milli / 1000.0})
            except (ValueError, OSError):
                continue
    return out


def disk_free_gb(path: str = "/") -> float:
    st = shutil.disk_usage(path)
    return st.free / 1e9


def mem_avail_mb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


def uptime_str() -> str:
    try:
        secs = float(Path("/proc/uptime").read_text().split()[0])
        h, rem = divmod(int(secs), 3600)
        m, s = divmod(rem, 60)
        return f"{h}h{m:02d}m{s:02d}s"
    except (OSError, ValueError):
        return "unknown"


def check() -> dict:
    th = load_thresholds()
    temps = read_temps()
    hot = [t for t in temps if t["celsius"] >= th["temp_crit"]]
    disk_gb = disk_free_gb("/")
    mem_mb = mem_avail_mb()
    alerts: list[str] = []
    if hot:
        worst = max(hot, key=lambda t: t["celsius"])
        alerts.append(f"temperature {worst['celsius']:.1f}C on {worst['chip']} >= {th['temp_crit']}C")
    if disk_gb < th["disk_min_free_gb"]:
        alerts.append(f"disk free {disk_gb:.1f}GB < {th['disk_min_free_gb']}GB")
    if mem_mb is not None and mem_mb < th["mem_min_avail_mb"]:
        alerts.append(f"memory available {mem_mb:.0f}MB < {th['mem_min_avail_mb']}MB")
    return {
        "ok": not alerts,
        "alerts": alerts,
        "uptime": uptime_str(),
        "disk_free_gb": round(disk_gb, 1),
        "mem_avail_mb": round(mem_mb) if mem_mb is not None else None,
        "max_temp_c": max((t["celsius"] for t in temps), default=None),
        "temp_sensors": len(temps),
    }


def maybe_send(report: dict) -> None:
    exe = shutil.which("musegadget")
    if not exe:
        print("(musegadget not installed — alert stays local)")
        return
    msg = "Health Sentinel: " + ("all OK" if report["ok"] else "; ".join(report["alerts"]))
    subprocess.run([exe, "send-user-msg", msg], check=False, timeout=15)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Health Sentinel v1")
    ap.add_argument("--check", action="store_true", help="run checks and print JSON")
    ap.add_argument("--send", action="store_true", help="forward result via musegadget")
    args = ap.parse_args(argv)
    report = check()
    print(json.dumps(report, indent=2))
    if args.send:
        maybe_send(report)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
