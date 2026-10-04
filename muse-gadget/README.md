# muse-gadget — Desk Watcher + Health Sentinel (v1)

Turns this Omarchy Linux box into a **Muse Gadget**: webcam motion snapshots
plus system-health alerts, forwarded to Muse on your phone once the Meta
Linux Device SDK daemon (`musegadget`) is installed + paired.

Meta SDK: `github.com/facebookincubator/muse-gadget-sdk` (`linux/` dir),
tokens at `gadgets.muse.ai/settings/sdk-tokens`. Nothing here needs it yet —
both scripts run standalone and degrade to local logging.

## Install from GitHub

Box (Omarchy/Arch):

```bash
git clone git@github.com:smkalle/vibe1.git && cd vibe1/muse-gadget
./setup_muse_gadget.sh            # needs MUSE_SDK_TOKEN for pair step only
```

Phone (Termux): see [QUICKSTART_TERMUX.md](QUICKSTART_TERMUX.md) — one curl
+ one ssh key, then `./termux_controller.sh photo`.

## Files

- `desk_watcher.py` — OpenCV frame-diff motion detection on `/dev/video0`.
  `--once` test capture, `--watch` loop with cooldown. Snapshots → `snapshots/`,
  events → `events.log` until `musegadget` exists, then `send-user-msg`.
- `health_sentinel.py` — stdlib-only checks: temps (`/sys/class/hwmon`),
  disk free, mem available, uptime. JSON out, exit 2 on alert, `--send` forwards.
- `watcher_config.json` — tunables for both (intervals, thresholds).
- `snapshots/` — gitignored captures. `events.log` — local event log.

## Verified on OMAKALLE (Oct 4 2026)

- `--once`: 640x480 capture OK (~69KB). `--check`: 14 temp sensors,
  max 82°C, mem 2878MB avail, disk 4.7GB free → correctly alerts (< 5GB;
  `/` is 91% full — real, not a test artifact).
- Camera `/dev/video0` (Integrated_Webcam_HD), mic records, BT `OMAKALLE`
  pairable, no RTSP cams on LAN (port 554 closed on all neighbours).

## Run

```bash
python3 desk_watcher.py --once
python3 desk_watcher.py --watch            # Ctrl-C to stop
python3 health_sentinel.py --check
```

## Setup on this box (Omarchy/Arch port)

The official installer refuses Arch, so `setup_muse_gadget.sh` performs the
same steps with pacman (`/opt/musegadget` venv on system python,
BlueZ MTU 256 + battery plugin off, `musegadget.service` with commands as
limited user `muse-gadget`, plus `desk-watcher.service` and an hourly
`health-sentinel.timer` from `/opt/muse-gadget-watcher`):

```bash
export MUSE_SDK_TOKEN=mgst_...   # from gadgets.muse.ai/settings/sdk-tokens
./setup_muse_gadget.sh            # install + open BLE pairing
```

Status: installed, both services active (`MuseGadgetBBB1BA`, node
`homelink-bbb1ba`), watcher logging quiet frames. Remaining: token +
phone pairing (`Settings > Devices > Developer mode` > add `MuseGadgetBBB1BA`).
