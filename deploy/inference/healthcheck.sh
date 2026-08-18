#!/usr/bin/env bash
# End-to-end serving verification. Passing CLI flags is NOT verification —
# each feature is exercised against the live server:
#
#   1. /health + /v1/models (alias served)
#   2. simple completion (+ usage reporting)
#   3. tool calling through the qwen3_coder parser
#   4. reasoning parser (reasoning_content separated from content)
#   5. prefix caching: two shared-prefix requests must move the
#      prefix-cache metrics (names discovered from /metrics, not hardcoded)
#   6. speculative decoding: spec-decode metrics present and advancing
set -euo pipefail
cd "$(dirname "$0")"
[[ -f .env ]] && { set -a; source ./.env; set +a; }

BASE="http://${HOST:-127.0.0.1}:${PORT:-8000}"
MODEL_ALIAS="${SERVED_MODEL_NAME:-qwen3.6-27b-fp8}"
fail=0

step() { echo "==> $1"; }
ok()   { echo "    OK${1:+: $1}"; }
bad()  { echo "    FAILED${1:+: $1}" >&2; fail=1; }

chat() {  # chat <json-messages> [extra-json-fields]
    curl -fsS "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"${MODEL_ALIAS}\",\"stream\":false,\"max_tokens\":64,\"temperature\":0,${2:+$2,}\"messages\":$1}"
}

metric_sum() {  # metric_sum <substring...> — sum of matching counters
    # Prints NOTHING when no series matches. An uninitialized accumulator
    # would print 0, making "metric absent" indistinguishable from "metric
    # present and zero" — and the absent-metric failures below unreachable.
    curl -fsS "${BASE}/metrics" 2>/dev/null | awk -v pats="$*" '
        BEGIN { n=split(pats, p, " "); found=0 }
        /^#/ { next }
        { name=$1; sub(/\{.*/, "", name)
          for (i=1;i<=n;i++) if (index(name, p[i])==0) next
          s += $NF; found=1 }
        END { if (found) printf "%.0f", s }'
}

step "1. liveness + model availability"
curl -fsS "${BASE}/health" >/dev/null && ok "/health" || bad "/health unreachable"
MODELS_JSON="$(curl -fsS "${BASE}/v1/models")"
echo "${MODELS_JSON}" | grep -q "\"${MODEL_ALIAS}\"" && ok "model '${MODEL_ALIAS}' served" \
    || bad "model '${MODEL_ALIAS}' not in /v1/models: ${MODELS_JSON}"

step "2. simple completion + usage"
RESP="$(chat '[{"role":"user","content":"Reply with the single word: ready"}]')" || bad "completion request failed"
echo "${RESP}" | grep -q '"completion_tokens"' && ok "usage reported" || bad "no usage in response"

step "3. tool calling (qwen3_coder parser)"
TOOLS='"tools":[{"type":"function","function":{"name":"read_file","description":"Read a file.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}],"tool_choice":"auto"'
RESP="$(chat '[{"role":"user","content":"Use the read_file tool to read README.md. Do not answer in text."}]' "${TOOLS}")" || bad "tool request failed"
echo "${RESP}" | grep -q '"tool_calls"' && echo "${RESP}" | grep -q '"read_file"' \
    && ok "read_file tool call emitted" || bad "no valid tool_calls in: $(echo "${RESP}" | head -c 400)"

step "4. reasoning parser"
RESP="$(chat '[{"role":"user","content":"What is 2+2? Think briefly."}]')" || true
if echo "${RESP}" | grep -q '"reasoning_content"'; then
    ok "reasoning_content separated"
else
    echo "    NOTE: no reasoning_content field (model may answer without thinking; check --reasoning-parser)"
fi

step "5. prefix caching (metrics must move, not just the CLI flag)"
FILLER="$(printf 'The harness verifies serving features before production use. %.0s' $(seq 1 400))"
BEFORE="$(metric_sum prefix_cache)"
chat "[{\"role\":\"user\",\"content\":\"${FILLER} Reply: one\"}]" >/dev/null || bad "prefix request 1 failed"
chat "[{\"role\":\"user\",\"content\":\"${FILLER} Reply: two\"}]" >/dev/null || bad "prefix request 2 failed"
AFTER="$(metric_sum prefix_cache)"
if [[ -z "${BEFORE}" || -z "${AFTER}" ]]; then
    bad "no prefix_cache metrics exposed — verify --enable-prefix-caching and the vLLM version"
elif [[ "${AFTER}" -gt "${BEFORE}" ]]; then
    ok "prefix cache metrics advanced (${BEFORE} -> ${AFTER})"
else
    bad "prefix cache metrics did not move (${BEFORE} -> ${AFTER}); caching may be inactive"
fi

step "6. speculative decoding (only if SPECULATIVE_CONFIG is set)"
if [[ -n "${SPECULATIVE_CONFIG:-}" ]]; then
    SPEC_BEFORE="$(metric_sum spec_decod)"
    chat '[{"role":"user","content":"Write a 3-line Python function that adds two numbers."}]' >/dev/null || true
    SPEC_AFTER="$(metric_sum spec_decod)"
    if [[ -z "${SPEC_AFTER}" ]]; then
        bad "no spec_decode metrics exposed — speculative decoding is configured but not active"
    elif [[ "${SPEC_AFTER}" -gt "${SPEC_BEFORE:-0}" ]]; then
        ok "speculative decode metrics advanced (${SPEC_BEFORE:-0} -> ${SPEC_AFTER})"
    else
        echo "    NOTE: spec metrics present but unchanged by one request; run benchmark.sh for a real check"
    fi
else
    echo "    skipped (speculative decoding disabled in .env)"
fi

echo
if [[ $fail -eq 0 ]]; then echo "== healthcheck PASSED =="; else echo "== healthcheck FAILED ==" >&2; fi
exit $fail
