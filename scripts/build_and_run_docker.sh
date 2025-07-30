#!/usr/bin/env bash
set -euo pipefail
docker build -t litserve-mcp .
docker run --rm -p 8000:8000 litserve-mcp
