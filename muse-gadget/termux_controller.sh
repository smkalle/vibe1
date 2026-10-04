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
SSH="ssh -i $KEY -o BatchMode=yes -o ConnectTimeout=10 $BOX"

case "${1:-help}" in
    status) $SSH gadgetctl status ;;
    health) $SSH gadgetctl health ;;
    events) $SSH gadgetctl events ;;
    log)    $SSH gadgetctl log ;;
    photo)
        mkdir -p ~/gadget
        # gadgetctl snap prints the path; fetch + open it.
        remote="$($SSH gadgetctl snap)" || { echo "$remote"; exit 1; }
        path="${remote%% *}"  # strip " (latest, camera busy)" suffix if present
        scp -i "$KEY" "$BOX:$path" ~/gadget/latest.jpg && echo "saved ~/gadget/latest.jpg"
        command -v termux-open >/dev/null && termux-open ~/gadget/latest.jpg || true
        ;;
    *) echo "usage: $0 status|photo|health|events|log" ;;
esac
