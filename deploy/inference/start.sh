#!/usr/bin/env bash
# Start the vLLM server for the agent harness (DGX Spark).
#
# Safety gates before anything serves:
#   - .env must exist and VLLM_IMAGE must be a pinned tag (never `latest`)
#   - the container's `vllm --version` must satisfy VLLM_MIN_VERSION
# Then launch.sh is generated from .env (the exact `vllm serve` command,
# speculative JSON included) and compose brings the service up.
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f .env ]]; then
    echo "ERROR: deploy/inference/.env not found. cp .env.example .env and edit it." >&2
    exit 1
fi
set -a; source ./.env; set +a

if [[ "${VLLM_IMAGE}" == *:latest || "${VLLM_IMAGE}" != *:* ]]; then
    echo "ERROR: VLLM_IMAGE must pin an explicitly verified tag (got '${VLLM_IMAGE}')." >&2
    echo "       'latest' is not reproducible and is rejected." >&2
    exit 1
fi

echo "==> verifying vLLM version inside ${VLLM_IMAGE} (>= ${VLLM_MIN_VERSION:-0.19})"
CONTAINER_VLLM_VERSION="$(docker run --rm --entrypoint vllm "${VLLM_IMAGE}" --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1 || true)"
if [[ -z "${CONTAINER_VLLM_VERSION}" ]]; then
    echo "ERROR: could not read 'vllm --version' from ${VLLM_IMAGE}." >&2
    exit 1
fi
if [[ "$(printf '%s\n%s\n' "${VLLM_MIN_VERSION:-0.19}" "${CONTAINER_VLLM_VERSION}" | sort -V | head -1)" != "${VLLM_MIN_VERSION:-0.19}" ]]; then
    echo "ERROR: container ships vLLM ${CONTAINER_VLLM_VERSION} < required ${VLLM_MIN_VERSION:-0.19}." >&2
    echo "       Use a newer container that officially supports Qwen3.6." >&2
    exit 1
fi
echo "    vLLM ${CONTAINER_VLLM_VERSION} OK"

if [[ -n "${SPECULATIVE_CONFIG:-}" ]]; then
    # Sourcing strips shell quoting, so a value that was not quoted in .env
    # arrives here as malformed JSON. Fail with the fix instead of handing
    # vLLM something it will reject at startup.
    if ! printf '%s' "${SPECULATIVE_CONFIG}" | python3 -c 'import json,sys; json.load(sys.stdin)' 2>/dev/null; then
        echo "ERROR: SPECULATIVE_CONFIG is not valid JSON after shell expansion:" >&2
        echo "       ${SPECULATIVE_CONFIG}" >&2
        echo "       Single-quote the whole value in .env, e.g." >&2
        echo "       SPECULATIVE_CONFIG='{\"method\":\"qwen3_next_mtp\",\"num_speculative_tokens\":2}'" >&2
        exit 1
    fi
fi

mkdir -p .generated
{
    echo '#!/usr/bin/env bash'
    echo 'set -euo pipefail'
    echo -n 'exec vllm serve '
    printf '%q ' "${MODEL}"
    printf -- '--served-model-name %q ' "${SERVED_MODEL_NAME}"
    printf -- '--host 0.0.0.0 --port 8000 '
    printf -- '--tensor-parallel-size %q ' "${TENSOR_PARALLEL_SIZE:-1}"
    # Text-only serving: no vision encoder memory; everything goes to
    # KV cache / batching. Multimodal agents get a separate profile.
    printf -- '--language-model-only '
    printf -- '--reasoning-parser %q ' "${REASONING_PARSER:-qwen3}"
    printf -- '--enable-auto-tool-choice --tool-call-parser %q ' "${TOOL_CALL_PARSER:-qwen3_coder}"
    # Prefix caching is EXPLICIT, never left to defaults; healthcheck.sh
    # verifies it is actually effective at runtime.
    printf -- '--enable-prefix-caching --enable-chunked-prefill '
    printf -- '--max-model-len %q ' "${MAX_MODEL_LEN:-65536}"
    printf -- '--max-num-seqs %q ' "${MAX_NUM_SEQS:-16}"
    printf -- '--max-num-batched-tokens %q ' "${MAX_NUM_BATCHED_TOKENS:-32768}"
    printf -- '--gpu-memory-utilization %q ' "${GPU_MEMORY_UTILIZATION:-0.80}"
    printf -- '--kv-cache-dtype %q ' "${KV_CACHE_DTYPE:-auto}"
    if [[ -n "${MODEL_REVISION:-}" ]]; then
        printf -- '--revision %q ' "${MODEL_REVISION}"
    fi
    if [[ -n "${SPECULATIVE_CONFIG:-}" ]]; then
        printf -- "--speculative-config %q " "${SPECULATIVE_CONFIG}"
    fi
    if [[ -n "${EXTRA_ARGS:-}" ]]; then
        printf -- '%s ' ${EXTRA_ARGS}
    fi
    echo ''
} > .generated/launch.sh
chmod +x .generated/launch.sh
echo "==> generated .generated/launch.sh:"
sed -n '3p' .generated/launch.sh | fold -s -w 100 | sed 's/^/    /'

docker compose up -d
echo "==> waiting for the server to become healthy (model load can take minutes)"
for _ in $(seq 1 120); do
    if curl -fsS "http://${HOST:-127.0.0.1}:${PORT:-8000}/health" >/dev/null 2>&1; then
        echo "==> vLLM is up at http://${HOST:-127.0.0.1}:${PORT:-8000}/v1"
        echo "==> run ./healthcheck.sh to verify tool calling / prefix cache / speculative decoding"
        exit 0
    fi
    sleep 5
done
echo "ERROR: server did not become healthy; check 'docker compose logs vllm'" >&2
exit 1
