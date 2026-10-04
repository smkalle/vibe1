#!/usr/bin/env bash
# Termux controller for the OMAKALLE desk gadget. Plain Termux (+openssh) is
# enough; inside an Alpine proot just `apk add openssh` first — same script.
#
# One-time setup on the phone:
#   pkg install -y openssh termux-api   # termux-api for photo viewing (optional)
#   ssh-keygen -t ed25519 -f ~/.ssh/gadget -N ""
#   ssh-copy-id -i ~/.ssh/gadget user@100.92.198.31   # Tailscale IP of OMAKALLE
#   (replace 'user' with your box login, e.g. smk)
#
# Then: ./termux_controller.sh status|photo|health|events|log
# Optional: put symlinks in ~/.shortcuts/ for Termux:Widget home-screen buttons.
set -uo pipefail

BOX="${GADGET_BOX:-user@100.92.198.31}"
KEY="${GADGET_KEY:-$HOME/.ssh/gadget}"
# Phone browser fallback when termux-open is unavailable (e.g. Alpine proot):
# served photo dir + port. localhost is shared between Termux and proots.
PHOTO_DIR="${GADGET_PHOTO_DIR:-$HOME/gadget}"
SERVE_PORT="${GADGET_SERVE_PORT:-8080}"
SSH="ssh -i $KEY -o BatchMode=yes -o ConnectTimeout=10 $BOX"

case "${1:-help}" in
    status) $SSH gadgetctl status ;;
    health) $SSH gadgetctl health ;;
    events) $SSH gadgetctl events ;;
    log)    $SSH gadgetctl log ;;
    photo)
        mkdir -p "$PHOTO_DIR"
        # gadgetctl snap prints the path; fetch + open it.
        remote="$($SSH gadgetctl snap)" || { echo "$remote"; exit 1; }
        path="${remote%% *}"  # strip " (latest, camera busy)" suffix if present
        scp -i "$KEY" "$BOX:$path" "$PHOTO_DIR/latest.jpg" && echo "saved $PHOTO_DIR/latest.jpg"
        if command -v termux-open >/dev/null; then
            termux-open "$PHOTO_DIR/latest.jpg"
        else
            # No termux-api in here (e.g. Alpine proot): serve over localhost,
            # which proot shares with Termux — open the URL in any phone browser.
            if ! pgrep -f "http.server $SERVE_PORT" >/dev/null 2>&1; then
                nohup python3 -m http.server "$SERVE_PORT" --directory "$PHOTO_DIR" \
                    >/dev/null 2>&1 &
                sleep 1
            fi
            echo "open http://127.0.0.1:$SERVE_PORT/latest.jpg in your phone browser"
        fi
        ;;
    *) echo "usage: $0 status|photo|health|events|log" ;;
esac
