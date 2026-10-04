# Muse Gadget setup for Omarchy/Arch — Desk Watcher + Health Sentinel
#
# The official Meta installer only supports Debian/Ubuntu (it dies on Arch
# and needs apt-get/dpkg). This script performs the same steps with pacman
# equivalents. Step names mirror the official install.sh where possible.
#
#   Token comes ONLY from the environment, never from argv or files:
#       export MUSE_SDK_TOKEN=mgst_...
#       ./setup_muse_gadget.sh
#
#   Flags: --no-pair   install + wire services, skip opening BLE pairing
#          --uninstall remove services/venv (keeps /var/lib/musegadget unless --purge)
#          --purge     with --uninstall: also forget device identity + pairing
#
# Idempotent: safe to re-run (e.g. once the token is exported).

set -euo pipefail

PREFIX="/opt/musegadget"
VENV="$PREFIX/venv"
UNIT="/etc/systemd/system/musegadget.service"
STATE_DIR="/var/lib/musegadget"
BLUEZ_CONF="/etc/bluetooth/main.conf"
BLUEZ_DROPIN="/etc/systemd/system/bluetooth.service.d/zz-musegadget.conf"
SRC_DIR="$HOME/.local/src/muse-gadget-sdk/linux"
RUN_AS="muse-gadget"
TOKEN_RE='^mgst_[A-Za-z0-9_-]{42}[AEIMQUYcgkosw048]$'

NO_PAIR=0
UNINSTALL=0
PURGE=0
for arg in "$@"; do
    case "$arg" in
        --no-pair) NO_PAIR=1 ;;
        --uninstall) UNINSTALL=1 ;;
        --purge) PURGE=1 ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "error: unknown option: $arg" >&2; exit 1 ;;
    esac
done

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
as_root() { if [ "$(id -u)" -eq 0 ]; then "$@"; else sudo "$@"; fi; }

do_uninstall() {
    say "Removing musegadget (Arch port)"
    as_root systemctl disable --now musegadget.service >/dev/null 2>&1 || true
    as_root systemctl disable --now desk-watcher.service >/dev/null 2>&1 || true
    as_root systemctl disable --now health-sentinel.timer >/dev/null 2>&1 || true
    as_root rm -f "$UNIT" /etc/systemd/system/desk-watcher.service \
        /etc/systemd/system/health-sentinel.{service,timer} /usr/local/bin/musegadget
    as_root systemctl daemon-reload || true
    as_root rm -rf "$PREFIX"
    if [ "$PURGE" = 1 ]; then
        as_root rm -rf "$STATE_DIR"
        say "Removed device identity and pairing too. Remove the device in the Muse app as well."
    else
        say "Kept device identity and pairing in $STATE_DIR (rerun with --purge to forget them)."
    fi
    if [ -f "$BLUEZ_DROPIN" ]; then
        as_root rm -f "$BLUEZ_DROPIN"
        as_root systemctl daemon-reload
        as_root systemctl restart bluetooth || true
        say "Turned BlueZ's battery plugin back on."
    fi
}
[ "$UNINSTALL" = 1 ] && { do_uninstall; exit 0; }

# --- Preflight ---------------------------------------------------------------
say "Preflight"
[ "$(uname -s)" = Linux ] || die "needs Linux"
[ -d /run/systemd/system ] || die "systemd not running"
command -v git >/dev/null || die "git not found"
[ -x /usr/bin/python3 ] || die "/usr/bin/python3 missing"
[ -d /sys/class/bluetooth ] || warn "no Bluetooth adapter; pairing needs Bluetooth LE (found hci0 earlier, continuing)"
/usr/bin/python3 -c "import gi, dbus" 2>/dev/null || die "system python lacks gi/dbus (pacman: python-gobject python-dbus)"
UV="$(command -v uv || echo "$HOME/.local/share/mise/installs/uv/latest/uv-x86_64-unknown-linux-musl/uv")"
[ -x "$UV" ] || die "uv not found"
[ -d "$SRC_DIR" ] || die "SDK checkout missing at $SRC_DIR (git clone facebookincubator/muse-gadget-sdk there)"
need_refresh=0
for pkg in python-cryptography python-websockets python-opencv python-numpy; do
    pacman -Q "$pkg" >/dev/null 2>&1 || need_refresh=1
done
if [ "$need_refresh" = 1 ]; then
    say "Refreshing package database"
    as_root pacman -Sy --noconfirm
fi
for pkg in python-cryptography python-websockets python-opencv python-numpy; do
    pacman -Q "$pkg" >/dev/null 2>&1 || { say "Installing $pkg"; as_root pacman -S --needed --noconfirm "$pkg"; }
done

# --- Run-as account (limited, no sudo) ----------------------------------------
if ! id "$RUN_AS" >/dev/null 2>&1; then
    say "Creating limited account '$RUN_AS' (no sudo; Muse inherits only its rights)"
    as_root useradd -r -m -s /usr/bin/nologin -c "Muse gadget sandbox" "$RUN_AS"
fi
as_root usermod -aG video "$RUN_AS"   # webcam /dev/video0
if as_root sudo -n -l -U "$RUN_AS" 2>/dev/null | grep -qE '\(ALL( : ALL)?\) (NOPASSWD: )?ALL'; then
    die "account '$RUN_AS' has full sudo; pick one without admin rights"
fi
say "Muse will run commands as '$RUN_AS' (no sudo) — sysadmin via this box stays yours"

# --- Venv + package (mirrors install_musegadget) ------------------------------
if [ ! -x "$VENV/bin/python" ]; then
    say "Creating $VENV (system python, system-site-packages for gi/dbus)"
    as_root "$UV" venv --quiet --python /usr/bin/python3 --system-site-packages "$VENV"
fi
say "Installing musegadget from $SRC_DIR"
as_root "$VENV/bin/python" -c "import musegadget" 2>/dev/null || \
    as_root "$UV" pip install --quiet --python "$VENV/bin/python" --no-deps "$SRC_DIR"
as_root "$VENV/bin/python" -c "import musegadget, cryptography, websockets, gi, dbus; print('imports OK')"
as_root ln -sf "$VENV/bin/musegadget" /usr/local/bin/musegadget

# --- BlueZ tweaks (same two as official: MTU 256, no battery plugin) ----------
say "Setting BlueZ ExchangeMTU=256 (backup: $BLUEZ_CONF.pre-musegadget)"
as_root python3 - "$BLUEZ_CONF" <<'EOF'
import re, shutil, sys
path = sys.argv[1]
try: text = open(path).read()
except FileNotFoundError: text = ""
m = re.search(r"^\s*ExchangeMTU\s*=\s*(\d+)\s*$", text, re.M)
if m and int(m.group(1)) <= 256: print("unchanged"); sys.exit()
line = "ExchangeMTU = 256"
if m: text = text[:m.start()] + line + text[m.end():]
elif re.search(r"^\s*#\s*ExchangeMTU\s*=.*$", text, re.M):
    text = re.sub(r"^\s*#\s*ExchangeMTU\s*=.*$", line, text, count=1, flags=re.M)
elif re.search(r"^\[GATT\]\s*$", text, re.M):
    text = re.sub(r"^\[GATT\]\s*$", "[GATT]\n" + line, text, count=1, flags=re.M)
else: text = text.rstrip("\n") + ("\n\n" if text else "") + "[GATT]\n" + line + "\n"
try: shutil.copy2(path, path + ".pre-musegadget")
except FileNotFoundError: pass
open(path, "w").write(text); print("changed")
EOF
as_root systemctl restart bluetooth || warn "bluetooth restart failed"
if [ ! -f "$BLUEZ_DROPIN" ]; then
    say "Disabling BlueZ battery plugin (phones are never asked to bond)"
    as_root mkdir -p "$(dirname "$BLUEZ_DROPIN")"
    # shellcheck disable=SC2024
    as_root tee "$BLUEZ_DROPIN" >/dev/null <<'EOF'
# Written by muse-gadget setup (Arch port of Meta installer): never ask phones to bond.
[Service]
ExecStart=
ExecStart=/usr/lib/bluetooth/bluetoothd --noplugin=battery
EOF
    as_root systemctl daemon-reload
    as_root systemctl restart bluetooth || warn "bluetooth restart failed"
fi

# --- SDK token (env only, root-only file like official) -----------------------
TOKEN="${MUSE_SDK_TOKEN:-}"
if [ -z "$TOKEN" ]; then
    if ! as_root test -s "$STATE_DIR/sdk_token"; then
        warn "MUSE_SDK_TOKEN not set and no token saved yet."
        echo "  export MUSE_SDK_TOKEN=mgst_...   # from gadgets.muse.ai/settings/sdk-tokens"
        echo "  then re-run this script to finish (pair + test)."
    fi
else
    [[ "$TOKEN" =~ $TOKEN_RE ]] || die "that SDK token is not valid; copy it again from gadgets.muse.ai"
    say "Saving SDK token (root-only, never logged)"
    as_root install -d -m 0700 "$STATE_DIR"
    printf '%s\n' "$TOKEN" | as_root install -m 0600 /dev/stdin "$STATE_DIR/sdk_token"
fi

# --- musegadget service --------------------------------------------------------
say "Installing musegadget service (runs as root for BT; commands as $RUN_AS)"
sed "s/@RUN_AS@/$RUN_AS/" "$SRC_DIR/src/musegadget/data/musegadget.service" | as_root tee "$UNIT" >/dev/null
as_root systemctl daemon-reload
as_root systemctl enable --now musegadget.service >/dev/null 2>&1 || true
as_root systemctl restart musegadget.service
sleep 3
as_root musegadget info || warn "musegadget info failed; see: sudo journalctl -u musegadget -f"

# --- Wire Desk Watcher + Health Sentinel --------------------------------------
# Runtime lives on a native Linux path: the repo sits on NTFS (uid-mapped),
# which the limited service account cannot traverse. Repo stays source of
# truth; each run syncs scripts+config here.
HERE="$(cd "$(dirname "$0")" && pwd)"
RUNTIME_DIR="/opt/muse-gadget-watcher"
say "Wiring desk-watcher service + health-sentinel timer (as $RUN_AS)"
as_root install -d -m 0755 "$RUNTIME_DIR" "$RUNTIME_DIR/snapshots"
as_root install -m 0644 "$HERE/desk_watcher.py" "$HERE/health_sentinel.py" \
    "$HERE/watcher_config.json" "$RUNTIME_DIR/"
as_root chown -R "$RUN_AS:$RUN_AS" "$RUNTIME_DIR/snapshots"
as_root tee /etc/systemd/system/desk-watcher.service >/dev/null <<EOF
[Unit]
Description=Desk Watcher (Muse gadget webcam motion)
After=network-online.target
[Service]
User=$RUN_AS
SupplementaryGroups=video
ExecStart=/usr/bin/python3 $RUNTIME_DIR/desk_watcher.py --watch
Restart=always
RestartSec=10
[Install]
WantedBy=multi-user.target
EOF
as_root tee /etc/systemd/system/health-sentinel.service >/dev/null <<EOF
[Unit]
Description=Health Sentinel check (Muse gadget)
[Service]
Type=oneshot
User=$RUN_AS
ExecStart=/usr/bin/python3 $RUNTIME_DIR/health_sentinel.py --check --send
EOF
as_root tee /etc/systemd/system/health-sentinel.timer >/dev/null <<'EOF'
[Unit]
Description=Health Sentinel hourly
[Timer]
OnBootSec=5min
OnUnitActiveSec=1h
[Install]
WantedBy=timers.target
EOF
as_root systemctl daemon-reload
as_root systemctl enable --now desk-watcher.service health-sentinel.timer >/dev/null 2>&1 || true

# --- Pairing -------------------------------------------------------------------
if as_root test -s "$STATE_DIR/pairing.json"; then
    say "Already paired; the service will reconnect to your Muse."
else
    if [ "$NO_PAIR" = 1 ]; then
        say "Skipping pairing. Run 'sudo musegadget pair' when ready."
    elif [ -z "$TOKEN" ] && ! as_root test -s "$STATE_DIR/sdk_token"; then
        warn "No token, so pairing is not opened. Export MUSE_SDK_TOKEN and re-run."
    else
        cat <<'EOF'

Pair with your Muse (10-minute window once opened):
  1. Muse app > Settings > Devices > Developer mode ON.
  2. Add device (+) > choose the MuseGadget device named below.
  3. Wi-Fi prompt: pick the network shown; no password needed.

EOF
        as_root musegadget pair || warn "not paired. Run 'sudo musegadget pair' to try again."
    fi
fi

# --- Test ----------------------------------------------------------------------
if as_root test -s "$STATE_DIR/pairing.json"; then
    say "Sending test message to Muse"
    as_root musegadget send-user-msg "Desk Watcher online: motion + health sentinel armed on OMAKALLE." \
        && say "Test sent — check your Muse chat." \
        || warn "test send failed; check: sudo journalctl -u musegadget -f"
else
    warn "Not paired yet — test message skipped (it will work after pairing)."
fi

cat <<EOF

Service:   sudo systemctl status musegadget desk-watcher.service
Logs:      sudo journalctl -u musegadget -f
Watcher:   sudo journalctl -u desk-watcher -f
Remove:    ./setup_muse_gadget.sh --uninstall [--purge]
EOF
