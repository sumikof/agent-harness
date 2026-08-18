#!/usr/bin/env bash
# Wrapper around scripts/benchmark_inference.py against the local server.
#
#   ./benchmark.sh sweep      # concurrency 1/4/8/16 (+24/32 with --saturation)
#   ./benchmark.sh prefix     # prefix-cache ON impact (shared 20K/50K prefixes)
#   ./benchmark.sh soak       # long-running 8-16 concurrent agent-like load
#   ./benchmark.sh all        # sweep + prefix, then write the tuning profile
#
# Speculative (MTP) and server-side sweeps (max-num-seqs, batched tokens,
# gpu-memory-utilization) require a server RESTART per point: edit .env,
# ./start.sh, re-run `./benchmark.sh sweep --label mtp2` etc. The Python
# harness records per-run labels and merges results.
set -euo pipefail
cd "$(dirname "$0")"
[[ -f .env ]] && { set -a; source ./.env; set +a; }

BASE_URL="http://${HOST:-127.0.0.1}:${PORT:-8000}/v1"
MODEL_ALIAS="${SERVED_MODEL_NAME:-qwen3.6-27b-fp8}"
MODE="${1:-sweep}"; shift || true

exec python3 ../../scripts/benchmark_inference.py \
    --base-url "${BASE_URL}" \
    --model "${MODEL_ALIAS}" \
    --mode "${MODE}" \
    --output-dir ../../config \
    "$@"
