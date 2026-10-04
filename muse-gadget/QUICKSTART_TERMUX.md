# Desk gadget on your phone in ~5 minutes (Termux)

No Muse app, no region lock — just SSH over Tailscale.

## 1. Prerequisites (one time)

- Tailscale on phone + Linux box, same account (test: `ping 100.92.198.31`)
- Box: `gadgetctl` installed (it is — `/usr/local/bin/gadgetctl`)
- Your box login + Tailscale IP (default below assumes `smk@100.92.198.31`)

## 2. Install the controller (on the phone)

```bash
pkg install -y openssh termux-api
mkdir -p ~/gadget && cd ~/gadget
curl -fsSL https://raw.githubusercontent.com/smkalle/vibe1/main/muse-gadget/termux_controller.sh \
  -o termux_controller.sh
chmod +x termux_controller.sh
ssh-keygen -t ed25519 -f ~/.ssh/gadget -N ""
ssh-copy-id -i ~/.ssh/gadget smk@100.92.198.31
```

If your login or IP differs:

```bash
export GADGET_BOX=you@100.x.y.z
```

## 3. Use it

```bash
./termux_controller.sh status   # services + pairing state
./termux_controller.sh photo    # latest desk photo, opened automatically
./termux_controller.sh health   # temps / disk / memory / uptime
./termux_controller.sh events   # recent motion events
./termux_controller.sh log      # watcher service log
```

## 4. Optional: home-screen buttons

Install **Termux:Widget**, then:

```bash
mkdir -p ~/.shortcuts
ln -s ~/gadget/termux_controller.sh ~/.shortcuts/  # or per-command wrappers
```

## 5. Optional: lock the key down

On the box, prefix your phone's key in `~/.ssh/authorized_keys` so it can
only run the controller, never a shell:

```
command="/usr/local/bin/gadgetctl $SSH_ORIGINAL_COMMAND",no-agent-forwarding,no-port-forwarding ssh-ed25519 AAAA...
```

Note: with a forced command, `scp` in the `photo` command stops working —
use `./termux_controller.sh photo` alternatives or keep a second full key.
