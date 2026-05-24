#!/usr/bin/env bash
# Unified OpenAI compatibility validation harness for qwen-proxy-rs.
# Runs all suites sequentially against a live proxy.
#
# Prerequisites:
#   - Proxy running with QWEN_TOKEN or ~/.qwen_session.json
#   - Node/bun deps installed in parent qtalt/ (npm install / bun install)
#
# Usage:
#   ./tests/run_compat.sh
#   PROXY_URL=http://127.0.0.1:8765 ./tests/run_compat.sh
#   ./tests/run_compat.sh --skip-perf   # skip live streaming perf suite

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROXY_RS="$(cd "$SCRIPT_DIR/.." && pwd)"
QTALT="$(cd "$PROXY_RS/.." && pwd)"

PROXY_URL="${PROXY_URL:-http://127.0.0.1:8765/v1}"
HEALTH_URL="${PROXY_URL%/v1}/health"
SKIP_PERF=0

for arg in "$@"; do
  case "$arg" in
    --skip-perf) SKIP_PERF=1 ;;
    -h|--help)
      echo "Usage: $0 [--skip-perf]"
      echo "  PROXY_URL  Base URL (default: http://127.0.0.1:8765/v1)"
      exit 0
      ;;
  esac
done

export PROXY_URL

declare -a SUITE_NAMES=()
declare -a SUITE_STATUS=()
declare -a SUITE_DETAIL=()

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

failures=0

record() {
  local name="$1"
  local status="$2"
  local detail="${3:-}"
  SUITE_NAMES+=("$name")
  SUITE_STATUS+=("$status")
  SUITE_DETAIL+=("$detail")
  if [[ "$status" == "FAIL" ]]; then
    failures=$((failures + 1))
  fi
}

run_suite() {
  local name="$1"
  shift
  echo ""
  echo -e "${BOLD}▶ ${name}${NC}"
  echo "  $*"
  if "$@"; then
    record "$name" "PASS" ""
    echo -e "  ${GREEN}✓ ${name} passed${NC}"
  else
    local code=$?
    record "$name" "FAIL" "exit $code"
    echo -e "  ${RED}✗ ${name} failed (exit $code)${NC}"
  fi
}

echo -e "${BOLD}Qwen Proxy — OpenAI Compatibility Validation${NC}"
echo "Proxy: $PROXY_URL"
echo "Health: $HEALTH_URL"

# Health check
if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
  record "health" "PASS" ""
  echo -e "${GREEN}✓ Proxy health OK${NC}"
else
  record "health" "FAIL" "unreachable"
  echo -e "${RED}✗ Proxy not reachable at $HEALTH_URL${NC}"
  echo "  Start: cd qwen-proxy-rs && cargo run --release"
fi

if [[ "${SUITE_STATUS[0]:-}" == "FAIL" ]]; then
  echo ""
  echo -e "${RED}Aborting: proxy must be running.${NC}"
  exit 1
fi

# Rust unit tests (no proxy traffic for most)
run_suite "cargo test (unit)" bash -c "cd '$PROXY_RS' && cargo test --quiet 2>&1"

# Rust HTTP contract integration tests
run_suite "openai_compat (ignored)" bash -c "cd '$PROXY_RS' && cargo test --test openai_compat -- --ignored --nocapture 2>&1"

# Node/Bun SDK suites (deps in qtalt/)
run_suite "openai_sdk_compat" bash -c "cd '$QTALT' && bun run '$PROXY_RS/tests/openai_sdk_compat_test.ts' 2>&1"

run_suite "ai_sdk" bash -c "cd '$QTALT' && node '$PROXY_RS/tests/ai_sdk_test.mjs' 2>&1"

run_suite "agents_sdk" bash -c "cd '$QTALT' && bun run '$PROXY_RS/tests/agents_sdk_test.ts' 2>&1"

run_suite "toolcall_stress" bash -c "cd '$QTALT' && node '$PROXY_RS/tests/toolcall_stress_test.mjs' 2>&1"

if [[ "$SKIP_PERF" -eq 0 ]]; then
  run_suite "live_streaming" bash -c "cd '$QTALT' && bun run '$PROXY_RS/tests/live_agents_streaming_test.ts' 2>&1"
else
  record "live_streaming" "SKIP" "--skip-perf"
  echo -e "${YELLOW}⊘ Skipping live_streaming (--skip-perf)${NC}"
fi

run_suite "pi_opencode_compat" bash -c "cd '$QTALT' && bun run '$PROXY_RS/tests/pi_opencode_compat_test.ts' 2>&1"

run_suite "contract_checklist" bash -c "cd '$QTALT' && node '$PROXY_RS/tests/contract_checklist_test.mjs' 2>&1"

# Summary table
echo ""
echo "$(printf '=%.0s' {1..60})"
echo -e "${BOLD}Compatibility Summary${NC}"
echo "$(printf '=%.0s' {1..60})"
printf "%-24s %-8s %s\n" "Suite" "Status" "Detail"
printf "%-24s %-8s %s\n" "-----" "------" "------"

for i in "${!SUITE_NAMES[@]}"; do
  name="${SUITE_NAMES[$i]}"
  status="${SUITE_STATUS[$i]}"
  detail="${SUITE_DETAIL[$i]}"
  case "$status" in
    PASS) color="$GREEN" ;;
    FAIL) color="$RED" ;;
    SKIP) color="$YELLOW" ;;
    *) color="$NC" ;;
  esac
  printf "${color}%-24s %-8s${NC} %s\n" "$name" "$status" "$detail"
done

echo "$(printf '=%.0s' {1..60})"

if [[ "$failures" -gt 0 ]]; then
  echo -e "${RED}${failures} suite(s) failed.${NC}"
  exit 1
fi

echo -e "${GREEN}All suites passed.${NC}"
exit 0
