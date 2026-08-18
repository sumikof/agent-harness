#!/usr/bin/env bash
# Stop the vLLM service (model cache stays on the host).
set -euo pipefail
cd "$(dirname "$0")"
docker compose down
