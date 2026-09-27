#!/bin/bash
# flysaver: build and deploy "A fly in the Matrix" as your Omarchy screensaver,
# straight from GitHub. Safe to re-run: it updates to the latest code.
#
#   curl -fsSL https://raw.githubusercontent.com/smkalle/vibe1/main/flysaver/deploy.sh | bash
#   curl -fsSL .../deploy.sh | bash -s -- --now          # also start it on every monitor now
#
# options:
#   --ref REF      branch, tag or commit to build (default: main)
#   --now          launch the screensaver right after installing
#   --test         run the test suite before installing
#   --yes          don't ask before installing missing packages
#   --uninstall    remove flysaver (keeps ~/.config/omarchy/flysaver.toml)
# environment:
#   FLYSAVER_REPO  git URL to build from (default: https://github.com/smkalle/vibe1.git)
#   FLYSAVER_SRC   where the checkout lives (default: ~/.local/src/flysaver)

set -euo pipefail

repo=${FLYSAVER_REPO:-https://github.com/smkalle/vibe1.git}
src=${FLYSAVER_SRC:-$HOME/.local/src/flysaver}
ref=main
now=0
test=0
yes=0
uninstall=0

while (($#)); do
  case $1 in
  --ref) ref=${2:?--ref needs a value}; shift ;;
  --now) now=1 ;;
  --test) test=1 ;;
  --yes | -y) yes=1 ;;
  --uninstall) uninstall=1 ;;
  -h | --help)
    echo "usage: deploy.sh [--ref REF] [--now] [--test] [--yes] [--uninstall]"
    echo "env:   FLYSAVER_REPO (default $repo), FLYSAVER_SRC (default $src)"
    exit 0
    ;;
  *) echo "flysaver deploy: unknown option $1" >&2; exit 2 ;;
  esac
  shift
done

say() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m==>\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m==>\033[0m %s\n' "$*" >&2; exit 1; }

# Read answers from the terminal, since stdin is this script under `curl | bash`.
confirm() {
  ((yes)) && return 0
  [[ -r /dev/tty ]] || die "$1 (re-run with --yes to allow this without a terminal)"
  local answer
  read -r -p "$1 [Y/n] " answer </dev/tty
  [[ -z $answer || $answer == [yY]* ]]
}

[[ $(uname -s) == Linux ]] || die "flysaver is an Omarchy (Linux) screensaver"
((EUID != 0)) || [[ ${FLYSAVER_ALLOW_ROOT:-} == 1 ]] || die "run this as your own user, not root: flysaver installs into \$HOME"

if ((uninstall)); then
  if [[ -x $HOME/.local/bin/flysaver ]]; then
    "$HOME/.local/bin/flysaver" uninstall
  else
    warn "flysaver is not installed"
  fi
  if [[ -d $src ]] && confirm "Also delete the source checkout in $src?"; then
    rm -rf -- "$src"
    say "removed $src"
  fi
  exit 0
fi

# --- dependencies -------------------------------------------------------------

missing=()
command -v git >/dev/null || missing+=(git)
command -v cargo >/dev/null || [[ -x $HOME/.cargo/bin/cargo ]] || missing+=(rust)
command -v socat >/dev/null || missing+=(socat) # flysaver launch waits on Hyprland's event socket
command -v jq >/dev/null || missing+=(jq)

if ((${#missing[@]})); then
  command -v pacman >/dev/null || die "missing: ${missing[*]}. Install them and re-run."
  confirm "Install ${missing[*]} with pacman (needs sudo)?" || die "missing: ${missing[*]}"
  sudo pacman -S --needed --noconfirm "${missing[@]}"
fi
[[ -x $HOME/.cargo/bin/cargo ]] && PATH="$HOME/.cargo/bin:$PATH"

# --- source -------------------------------------------------------------------
# $src is a checkout this script owns: it is always reset to the requested ref.

say "fetching $ref from $repo"
if [[ ! -d $src/.git ]]; then
  mkdir -p "$src"
  git -C "$src" init -q
  git -C "$src" remote add origin "$repo"
fi
git -C "$src" remote set-url origin "$repo"
git -C "$src" fetch -q --depth 1 origin "$ref"
git -C "$src" checkout -q --force --detach FETCH_HEAD
rev=$(git -C "$src" rev-parse --short HEAD)
[[ -f $src/flysaver/Cargo.toml ]] || die "$ref has no flysaver/ directory"

# --- build --------------------------------------------------------------------

cd "$src/flysaver"
say "building flysaver @ $rev (first build takes a minute)"
cargo build --release --locked --quiet

if ((test)); then
  say "running tests"
  cargo test --release --locked --quiet
  python3 tests/pty_lifecycle.py target/release/flysaver
  python3 tests/hypr_contract.py target/release/flysaver
  tests/install_roundtrip.sh target/release/flysaver
fi

# --- deploy -------------------------------------------------------------------

first_time=1
[[ -e $HOME/.config/uwsm/env.d/99-flysaver ]] && first_time=0

say "installing"
target/release/flysaver install >/dev/null
flysaver=$HOME/.local/bin/flysaver

say "checking the setup"
"$flysaver" doctor || true

if ((now)); then
  if [[ -n ${HYPRLAND_INSTANCE_SIGNATURE:-} ]]; then
    say "launching on every monitor (any key or mouse movement exits)"
    "$flysaver" launch || warn "launch failed; try: flysaver preview"
  else
    warn "not inside a Hyprland session; skipping --now"
  fi
fi

echo
say "flysaver @ $rev deployed to $flysaver"
if ((first_time)); then
  echo "    Log out and back in once so idle uses it. Until then: flysaver launch"
fi
echo "    Preview in a terminal: flysaver preview   Settings: ~/.config/omarchy/flysaver.toml"
echo "    Update: re-run this script.               Remove: re-run with --uninstall"
