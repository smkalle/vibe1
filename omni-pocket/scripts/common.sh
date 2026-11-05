# Shared helpers, sourced by the other scripts.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${OMNI_VENV:-$ROOT/.venv}"

say()  { printf '\033[1;35m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mxx\033[0m %s\n' "$*" >&2; exit 1; }

detect_env() {
  if [[ "${PREFIX:-}" == *com.termux* || -n "${TERMUX_VERSION:-}" ]]; then echo termux
  elif [[ -e /support/userland_profile ]] || env | grep -q '^PROOT' \
       || grep -qs proot "/proc/$(awk '/TracerPid/{print $2}' /proc/self/status)/comm"; then echo proot
  elif [[ "$(uname -s)" == Darwin ]]; then echo macos
  else echo linux; fi
}

free_gb() { df -Pk "${1:-$ROOT}" | awk 'NR==2{printf "%d", $4/1048576}'; }

activate() {
  [[ -f "$VENV/bin/activate" ]] || die "no virtualenv at $VENV - run the install script for your environment first"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
}
