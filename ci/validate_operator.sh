#!/bin/bash
# Validate one vendored downstream project against this pipeline's TileLang wheel.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TILELANG_HOME="$(cd "$SCRIPT_DIR/.." && pwd)"

: "${OP:?OP must be set by the CI matrix}"
case "$OP" in
    tileops_sunrise) OP_HOME="$TILELANG_HOME/downstream/tileops_sunrise" ;;
    tilekernels_sunrise) OP_HOME="$TILELANG_HOME/downstream/tilekernels_sunrise" ;;
    *) echo "Unknown downstream project: $OP" >&2; exit 1 ;;
esac

export TILELANG_SOURCE_DIR="$TILELANG_HOME"
export TILELANG_WHL_DIR="$TILELANG_HOME/dist"
export TILELANG_CI_DIR="$SCRIPT_DIR"
export CI_FAILURE_REPORT_DIR="${CI_FAILURE_REPORT_DIR:-$TILELANG_HOME/ci_failure_reports}"

echo "Validated $OP source: $OP_HOME"
cd "$OP_HOME"
bash ci/test.sh
