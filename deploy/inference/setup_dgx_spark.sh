#!/usr/bin/env bash
# DGX Spark environment verification for the inference stack.
#
# Confirms the machine is what the deployment assumes (ARM64 + Blackwell +
# CUDA + Docker with NVIDIA GPU access) BEFORE anything is installed, so no
# x86-specific wheel or image ever lands on this host by accident.
set -uo pipefail

fail=0
check() {  # check <label> <command...>
    local label="$1"; shift
    echo "==> ${label}"
    if "$@"; then echo "    OK"; else echo "    FAILED: $*" >&2; fail=1; fi
}

echo "== DGX Spark environment check =="

ARCH="$(uname -m)"
echo "==> uname -m: ${ARCH}"
if [[ "${ARCH}" != "aarch64" && "${ARCH}" != "arm64" ]]; then
    echo "    WARNING: expected aarch64 (DGX Spark is ARM64/Grace). x86 images/wheels" >&2
    echo "             must NOT be pulled onto this host if it is a real DGX Spark." >&2
fi

check "nvidia-smi (driver + GPU visible)" nvidia-smi
check "nvcc --version (CUDA toolkit)" bash -c "nvcc --version || /usr/local/cuda/bin/nvcc --version"
check "docker --version" docker --version
check "docker info (daemon reachable)" docker info

echo "==> docker GPU access (container -> NVIDIA GPU)"
if docker run --rm --gpus all ubuntu:24.04 nvidia-smi >/dev/null 2>&1; then
    echo "    OK"
else
    echo "    FAILED: containers cannot reach the GPU. Install/configure the" >&2
    echo "    NVIDIA Container Toolkit (nvidia-ctk runtime configure) first." >&2
    fail=1
fi

if [[ -f "$(dirname "$0")/.env" ]]; then
    set -a; source "$(dirname "$0")/.env"; set +a
    echo "==> container image architecture + vllm version (${VLLM_IMAGE:-unset})"
    if [[ -n "${VLLM_IMAGE:-}" ]]; then
        IMG_ARCH="$(docker image inspect --format '{{.Architecture}}' "${VLLM_IMAGE}" 2>/dev/null || true)"
        if [[ -z "${IMG_ARCH}" ]]; then
            echo "    image not pulled yet (docker pull ${VLLM_IMAGE})"
        elif [[ "${IMG_ARCH}" != "arm64" && "${ARCH}" == "aarch64" ]]; then
            echo "    FAILED: image arch ${IMG_ARCH} does not match host aarch64" >&2
            fail=1
        else
            echo "    image arch: ${IMG_ARCH}"
        fi
        VER="$(docker run --rm --entrypoint vllm "${VLLM_IMAGE}" --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1 || true)"
        echo "    vllm --version: ${VER:-unreadable}"
        if [[ -n "${VER}" ]]; then
            MIN="${VLLM_MIN_VERSION:-0.19}"
            if [[ "$(printf '%s\n%s\n' "$MIN" "$VER" | sort -V | head -1)" != "$MIN" ]]; then
                echo "    FAILED: vLLM ${VER} < required ${MIN} (Qwen3.6 support)" >&2
                fail=1
            fi
        fi
    fi
else
    echo "==> (no .env yet — copy .env.example to also verify the container)"
fi

echo
if [[ $fail -eq 0 ]]; then
    echo "== environment check PASSED =="
else
    echo "== environment check FAILED (see messages above) ==" >&2
fi
exit $fail
