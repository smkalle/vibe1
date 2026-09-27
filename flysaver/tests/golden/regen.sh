#!/bin/bash
# Regenerate the golden frames after an intentional visual change.
set -euo pipefail
cd "$(dirname "$0")/../.."
cargo build --release --quiet
export HOME=$(mktemp -d)  # ignore the developer's own flysaver.toml
for cam in follow room brain; do
  target/release/flysaver snapshot --seed 7 --size 100x30 --time 6 --camera "$cam" --palette matrix \
    >"tests/golden/${cam}_s7_100x30_t6.txt"
done
echo "regenerated tests/golden/*.txt"
