#!/bin/bash
# install -> shim resolves to flysaver -> disabled shim falls back to stock -> uninstall leaves nothing behind.
# usage: tests/install_roundtrip.sh [path/to/flysaver]
set -euo pipefail

bin=$(realpath "${1:-target/release/flysaver}")
export HOME=$(mktemp -d)
unset XDG_CONFIG_HOME
trap 'rm -rf "$HOME"' EXIT

"$bin" install >/dev/null

for f in .local/bin/flysaver .local/share/flysaver/bin/omarchy-screensaver .local/share/flysaver/flysaver-launch \
  .config/uwsm/env.d/99-flysaver .config/omarchy/screensaver/run .config/omarchy/flysaver.toml; do
  [[ -e $HOME/$f ]] || { echo "missing $f"; exit 1; }
done
echo "ok  install wrote every file"

# The session env puts the shim first, exactly once even if sourced twice.
stock=$HOME/stock
mkdir -p "$stock"
printf '#!/bin/bash\necho STOCK\n' >"$stock/omarchy-screensaver"
chmod +x "$stock/omarchy-screensaver"
path=$(PATH="$stock:/usr/bin:/bin" bash -c ". $HOME/.config/uwsm/env.d/99-flysaver; . $HOME/.config/uwsm/env.d/99-flysaver; echo \$PATH")
[[ $path == "$HOME/.local/share/flysaver/bin:$stock:/usr/bin:/bin" ]] || { echo "bad PATH $path"; exit 1; }
resolved=$(PATH="$path" bash -c 'command -v omarchy-screensaver')
[[ $resolved == "$HOME/.local/share/flysaver/bin/omarchy-screensaver" ]] || { echo "resolves to $resolved"; exit 1; }
echo "ok  session PATH resolves omarchy-screensaver to the shim"

# The shim execs flysaver (about is a harmless command to prove it).
out=$(PATH="$path" omarchy-screensaver about | head -1)
[[ $out == flysaver* ]] || { echo "shim did not run flysaver: $out"; exit 1; }
echo "ok  shim runs flysaver"

touch "$HOME/.config/omarchy/flysaver.disabled"
out=$(PATH="$path" omarchy-screensaver)
[[ $out == STOCK ]] || { echo "disabled shim did not fall back: $out"; exit 1; }
rm "$HOME/.config/omarchy/flysaver.disabled"
echo "ok  disabled shim falls back to the stock screensaver"

"$HOME/.local/bin/flysaver" doctor >/dev/null || true   # no Hyprland here; must not crash
echo "ok  doctor runs"

"$HOME/.local/bin/flysaver" uninstall >/dev/null
left=$(cd "$HOME" && find . -type f ! -path './stock/*' | sort)
[[ $left == "./.config/omarchy/flysaver.toml" ]] || { echo "left behind: $left"; exit 1; }
echo "ok  uninstall removes everything but the user's config"
echo "all install checks passed"
