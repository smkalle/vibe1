#!/usr/bin/env bash
# Install a Termux:Boot hook so Omni Pocket starts when the phone boots.
# Needs the Termux:Boot app (F-Droid), opened once. Remove: rm ~/.termux/boot/omni-pocket
source "$(dirname "$0")/common.sh"
[[ "$(detect_env)" == termux ]] || die "Termux only"
mkdir -p ~/.termux/boot
cat > ~/.termux/boot/omni-pocket <<HOOK
#!/data/data/com.termux/files/usr/bin/sh
termux-wake-lock
cd "$ROOT" && nohup ./scripts/run.sh --no-open > "$ROOT/omni_data/boot.log" 2>&1 &
HOOK
chmod +x ~/.termux/boot/omni-pocket
say "Installed ~/.termux/boot/omni-pocket - after reboot open http://localhost:8501"
