#!/bin/bash
# Run TileKernels pre-commit checks with TileKernels as the logical Git repository.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GIT_ROOT="$(git -C "$PROJECT_ROOT" rev-parse --show-toplevel)"
RUN_ROOT="$PROJECT_ROOT"
SCOPED_ROOT=""

cleanup() {
    if [[ -n "$SCOPED_ROOT" && -d "$SCOPED_ROOT" ]]; then
        rm -rf "$SCOPED_ROOT"
    fi
}
trap cleanup EXIT

# A child pipeline checks out the TileLang-Sunrise monorepo. Export only the
# TileKernels subtree into a temporary Git repository so --all-files and
# hook-local config discovery cannot cross into the other projects.
if [[ "$PROJECT_ROOT" != "$GIT_ROOT" ]]; then
    project_rel="${PROJECT_ROOT#"$GIT_ROOT"/}"
    if [[ "$project_rel" == "$PROJECT_ROOT" || "$project_rel" == *".."* ]]; then
        echo "Unable to resolve TileKernels relative to Git root: $PROJECT_ROOT" >&2
        exit 1
    fi
    SCOPED_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/tilekernels-precommit.XXXXXX")"
    scoped_revision="${SCOPED_LINT_REVISION:-HEAD}"
    git -C "$GIT_ROOT" archive "$scoped_revision:$project_rel" | tar -xf - -C "$SCOPED_ROOT"
    git -C "$SCOPED_ROOT" init -q -b main
    git -C "$SCOPED_ROOT" -c user.name="TileKernels CI" -c user.email="ci@example.invalid" commit --allow-empty -q -m "ci: initialize scoped lint tree"
    git -C "$SCOPED_ROOT" add -A
    RUN_ROOT="$SCOPED_ROOT"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! "$PYTHON_BIN" -m pre_commit --version >/dev/null 2>&1; then
    "$PYTHON_BIN" -m pip install --user pre-commit
fi

cd "$RUN_ROOT"
"$PYTHON_BIN" -m pre_commit install-hooks
"$PYTHON_BIN" -m pre_commit run --all-files
