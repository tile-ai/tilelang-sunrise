#!/bin/bash
# Thin shim: ci/install.sh is the canonical build/install entry. Kept so existing
# callers of ci/build.sh (e.g. operator repos) keep working unchanged.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/install.sh" "$@"
