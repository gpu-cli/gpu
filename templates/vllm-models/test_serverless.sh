#!/usr/bin/env bash
# =============================================================================
# vLLM Serverless — Full Lifecycle Test
#
# Tests: deploy → status → list → invoke → delete
#
# Usage:
#   cd ~/Development/gpu/templates/vllm-models
#   bash test_serverless.sh
#
# Override the binary path:
#   GPU_BIN=~/Development/cli/crates/target/debug/gpu bash test_serverless.sh
#
# Skip the deploy (use existing endpoint):
#   ENDPOINT_ID=abc123 bash test_serverless.sh
# =============================================================================

set -euo pipefail

GPU_BIN="${GPU_BIN:-gpu}"
ENDPOINT_ID="${ENDPOINT_ID:-}"
CREATED_ENDPOINT=false

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

PASSED=0
FAILED=0

pass() { echo -e "  ${GREEN}PASS${NC}: $1"; PASSED=$((PASSED + 1)); }
fail() { echo -e "  ${RED}FAIL${NC}: $1"; FAILED=$((FAILED + 1)); }
info() { echo -e "\n${CYAN}====${NC} $1 ${CYAN}====${NC}"; }
warn() { echo -e "  ${YELLOW}WARN${NC}: $1"; }

cleanup() {
    if [[ "$CREATED_ENDPOINT" == "true" && -n "$ENDPOINT_ID" ]]; then
        echo ""
        warn "Cleaning up: deleting endpoint $ENDPOINT_ID..."
        $GPU_BIN serverless delete "$ENDPOINT_ID" --force 2>/dev/null || true
    fi
}
trap cleanup EXIT

# -----------------------------------------------------------------------------
info "1. Verify CLI"
# -----------------------------------------------------------------------------

$GPU_BIN --version > /dev/null 2>&1 && pass "gpu CLI found" || fail "gpu CLI not found"

# -----------------------------------------------------------------------------
info "2. Deploy"
# -----------------------------------------------------------------------------

if [[ -z "$ENDPOINT_ID" ]]; then
    echo "  Deploying vLLM endpoint (this talks to RunPod API)..."

    # Capture stderr separately to avoid mixing with JSON stdout
    STDERR_FILE=$(mktemp)
    DEPLOY_OUTPUT=$($GPU_BIN serverless deploy \
        --template vllm \
        --min-workers 0 \
        --max-workers 1 \
        --idle-timeout 5 \
        --json 2>"$STDERR_FILE") || { fail "Deploy command failed"; cat "$STDERR_FILE" >&2; rm -f "$STDERR_FILE"; }

    if [[ -f "$STDERR_FILE" ]]; then
        # Show stderr as info (progress messages)
        while IFS= read -r line; do
            [[ -n "$line" ]] && echo "  $line"
        done < "$STDERR_FILE"
        rm -f "$STDERR_FILE"
    fi

    # Try to extract endpoint ID from JSON stdout
    ENDPOINT_ID=$(echo "$DEPLOY_OUTPUT" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('endpoint_id', data.get('id', '')))
except:
    print('')
" 2>/dev/null || echo "")

    if [[ -z "$ENDPOINT_ID" ]]; then
        # Fallback: grep for ID patterns
        ENDPOINT_ID=$(echo "$DEPLOY_OUTPUT" | grep -oE '[a-z0-9]{20,}' | head -1 || echo "")
    fi

    if [[ -n "$ENDPOINT_ID" ]]; then
        CREATED_ENDPOINT=true
        pass "Deployed endpoint: $ENDPOINT_ID"
    else
        fail "Could not extract endpoint ID from output"
        echo "  Output was: $DEPLOY_OUTPUT"
    fi
else
    pass "Using existing endpoint: $ENDPOINT_ID"
fi

# -----------------------------------------------------------------------------
info "3. Status"
# -----------------------------------------------------------------------------

if [[ -n "$ENDPOINT_ID" ]]; then
    STATUS_OUTPUT=$($GPU_BIN serverless status "$ENDPOINT_ID" 2>&1) \
        && pass "Status retrieved" \
        || fail "Status check failed"
    echo "$STATUS_OUTPUT" | sed 's/^/  /'
fi

# -----------------------------------------------------------------------------
info "4. List"
# -----------------------------------------------------------------------------

LIST_OUTPUT=$($GPU_BIN serverless list 2>&1) \
    && pass "List succeeded" \
    || fail "List failed"

if [[ -n "$ENDPOINT_ID" ]] && echo "$LIST_OUTPUT" | grep -q "$ENDPOINT_ID"; then
    pass "Endpoint visible in list"
fi

# JSON format
LIST_JSON=$($GPU_BIN serverless list --json 2>&1) || true
if echo "$LIST_JSON" | python3 -m json.tool > /dev/null 2>&1; then
    pass "List --json returns valid JSON"
else
    warn "List --json output is not valid JSON"
fi

# -----------------------------------------------------------------------------
info "5. Invoke (curl)"
# -----------------------------------------------------------------------------

if [[ -n "$ENDPOINT_ID" ]]; then
    # Try to get API key
    RUNPOD_API_KEY="${RUNPOD_API_KEY:-}"
    if [[ -z "$RUNPOD_API_KEY" ]]; then
        RUNPOD_API_KEY=$($GPU_BIN auth show --key runpod 2>/dev/null || echo "")
    fi

    if [[ -n "$RUNPOD_API_KEY" ]]; then
        echo "  Sending chat completion to endpoint..."

        # Use /openai/v1/chat/completions endpoint format (RunPod vLLM worker v2.12+)
        # Model name must match MODEL_NAME env var in gpu.jsonc
        RESPONSE=$(curl -sS --max-time 300 \
            -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/openai/v1/chat/completions" \
            -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
            -H "Content-Type: application/json" \
            -d '{
                "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "messages": [
                    {"role": "system", "content": "Reply in one sentence."},
                    {"role": "user", "content": "What is serverless GPU computing?"}
                ],
                "max_tokens": 100,
                "temperature": 0.7
            }' 2>&1) || true

        # Check if response is valid JSON
        if echo "$RESPONSE" | python3 -m json.tool > /dev/null 2>&1; then
            # Check for choices (direct OpenAI format) or status (RunPod wrapper)
            HAS_CHOICES=$(echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'choices' in data:
        print('direct')
    elif 'status' in data:
        print(data['status'])
    else:
        print('unknown')
except:
    print('parse_error')
" 2>/dev/null || echo "parse_error")

            case "$HAS_CHOICES" in
                direct)
                    pass "Invocation completed (direct OpenAI response)"
                    TEXT=$(echo "$RESPONSE" | python3 -c "
import sys, json
r = json.load(sys.stdin)
choices = r.get('choices', [])
if choices:
    print(choices[0].get('message', {}).get('content', '(empty)'))
else:
    print('(no choices)')
" 2>/dev/null || echo "(parse error)")
                    echo -e "  ${GREEN}Response:${NC} $TEXT"
                    ;;
                COMPLETED)
                    pass "Invocation completed"
                    TEXT=$(echo "$RESPONSE" | python3 -c "
import sys, json
r = json.load(sys.stdin)
output = r.get('output', {})
choices = output.get('choices', [])
if choices:
    print(choices[0].get('message', {}).get('content', '(empty)'))
else:
    print('(no choices)')
" 2>/dev/null || echo "(parse error)")
                    echo -e "  ${GREEN}Response:${NC} $TEXT"
                    ;;
                IN_QUEUE|IN_PROGRESS)
                    warn "Request queued/processing (cold start). Status: $HAS_CHOICES"
                    ;;
                FAILED)
                    fail "Invocation failed"
                    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null | sed 's/^/  /' || echo "  $RESPONSE"
                    ;;
                *)
                    warn "Unexpected response format: $HAS_CHOICES"
                    echo "$RESPONSE" | head -5 | sed 's/^/  /'
                    ;;
            esac
        else
            warn "Response is not valid JSON"
            echo "$RESPONSE" | head -5 | sed 's/^/  /'
        fi
    else
        warn "Skipping invoke — no RUNPOD_API_KEY found"
    fi
fi

# -----------------------------------------------------------------------------
info "6. Delete"
# -----------------------------------------------------------------------------

if [[ "$CREATED_ENDPOINT" == "true" && -n "$ENDPOINT_ID" ]]; then
    $GPU_BIN serverless delete "$ENDPOINT_ID" --force 2>&1 \
        && pass "Endpoint deleted" \
        || fail "Delete failed"
    ENDPOINT_ID=""
    CREATED_ENDPOINT=false
else
    warn "Skipping delete (endpoint was pre-existing or not created)"
fi

# -----------------------------------------------------------------------------
info "Results"
# -----------------------------------------------------------------------------

echo ""
echo -e "  ${GREEN}Passed${NC}: $PASSED"
[[ $FAILED -gt 0 ]] && echo -e "  ${RED}Failed${NC}: $FAILED"
echo ""

[[ $FAILED -eq 0 ]] && echo -e "${GREEN}All tests passed.${NC}" || { echo -e "${RED}Some tests failed.${NC}"; exit 1; }
