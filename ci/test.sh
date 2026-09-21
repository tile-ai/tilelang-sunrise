#!/bin/bash
# Thin shim: ci/run.sh is the canonical test entry. Kept so existing callers of
# ci/test.sh keep working unchanged; runs TileLang's own list in CI flavor.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TILELANG_HOME="$(cd "$SCRIPT_DIR/.." && pwd)"
exec "$SCRIPT_DIR/run.sh" --ci --list "$TILELANG_HOME/ci_test_case_list_tilelang.txt"
