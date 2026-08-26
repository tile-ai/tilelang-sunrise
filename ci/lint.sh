#!/usr/bin/env bash
# Shared local + CI lint entry point.
#
# Usage:
#   ci/lint.sh changed   # blocking tier: lint only files changed vs the diff base
#   ci/lint.sh all       # audit tier: compileall + cpp-api audit (warn) + pre-commit --all-files
#
# CI's changed-files tier and all-files tier both run through this one script, so
# the two CI lint tiers cannot drift. format.sh is the separate local convenience
# wrapper (merge-base against the GitHub upstream) and may compute a different diff base.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

ZERO_SHA="0000000000000000000000000000000000000000"

log() { echo "[lint] $*"; }

# Interpreter used to run pre-commit. pre-commit builds each python-language
# hook env with the interpreter that runs it. All pinned hooks support Python
# >=3.9, matching the public CI runner baseline.
PY="python3"      # resolved by ensure_py39
PY_IN_CONDA=0     # 1 when PY lives in a conda env we created (skip pip --user)

py_ge_39() {
    "$1" -c 'import sys; raise SystemExit(0 if sys.version_info[:2]>=(3,9) else 1)' 2>/dev/null
}

# Resolve a Python >=3.9 interpreter for pre-commit and store it in PY.
# Tried in order: current python3 -> conda -> explicitly-versioned system Python.
ensure_py39() {
    local boot_err="LINT BOOTSTRAP ERROR: no Python >=3.9 interpreter available for pre-commit — this is NOT a lint failure"

    # 1. current python3 already satisfies >=3.9
    if py_ge_39 python3; then
        PY="python3"
        log "python interpreter : system python3 ($(python3 --version 2>&1)) — already >=3.9"
        return 0
    fi

    # 2. conda, when it is already available on PATH. Use a lightweight
    #    python=3.10-only env rather than ci_create_conda_env's full stack.
    if command -v conda >/dev/null 2>&1; then
        # Job-scoped prefix: lint_changed and lint_all run concurrently on one host.
        local lint_env="${CI_PROJECT_DIR:-.}/.lint-conda-${CI_JOB_ID:-local}"
        if [[ -x "$lint_env/bin/python" ]] && py_ge_39 "$lint_env/bin/python"; then
            PY="$lint_env/bin/python"; PY_IN_CONDA=1
            log "python interpreter : reusing conda env $lint_env ($("$PY" --version 2>&1))"
            return 0
        fi
        log "python interpreter : creating lightweight conda env $lint_env (python=3.9)"
        if conda create -y --prefix "$lint_env" python=3.9; then
            PY="$lint_env/bin/python"; PY_IN_CONDA=1
            return 0
        fi
        log "WARNING: conda create failed; falling through to system python3.1x probe"
    fi

    # 3. probe explicitly-versioned system interpreters (first found wins)
    local cand
    for cand in python3.13 python3.12 python3.11 python3.10 python3.9; do
        if command -v "$cand" >/dev/null 2>&1 && py_ge_39 "$cand"; then
            PY="$cand"
            log "python interpreter : $cand ($("$cand" --version 2>&1))"
            return 0
        fi
    done

    # 4. nothing >=3.9 anywhere — fail loudly, never silently pass
    local found=""
    for cand in python3.13 python3.12 python3.11 python3.10 python3.9; do
        command -v "$cand" >/dev/null 2>&1 && found+="$cand "
    done
    echo "$boot_err" >&2
    echo "[lint]   python3 version    : $(python3 --version 2>&1)" >&2
    echo "[lint]   conda on PATH      : $(command -v conda >/dev/null 2>&1 && echo yes || echo no)" >&2
    echo "[lint]   Python >=3.9 found : ${found:-none}" >&2
    return 1
}

# Ensure pre-commit is available; bootstrap minimally if missing.
# pre-commit self-installs the hook tools pinned in .pre-commit-config.yaml.
# Install only the pre-commit launcher (a user install, like format.sh) so lint
# does not couple to requirements-lint.txt version pins or need root on the
# shared runner. A bootstrap/network failure here is a runner-capability
# problem, NOT a lint violation, so it is reported with a distinct diagnostic.
ensure_pre_commit() {
    ensure_py39 || return 1
    local boot_err="LINT BOOTSTRAP ERROR: could not install/run pre-commit (tooling or network on the runner) — this is NOT a lint failure"
    if ! "$PY" -m pre_commit --version >/dev/null 2>&1; then
        log "pre-commit not found; installing"
        # In our own conda prefix a plain install lands in the env; otherwise a
        # user install (like format.sh) avoids needing root on the shared runner.
        if [[ $PY_IN_CONDA -eq 1 ]]; then
            "$PY" -m pip install pre-commit || { echo "$boot_err" >&2; return 1; }
        else
            "$PY" -m pip install --user pre-commit || { echo "$boot_err" >&2; return 1; }
        fi
    fi
    # Resolve/install hook environments up front so a network failure fetching
    # hook repos is flagged here distinctly instead of surfacing later as an
    # opaque failure inside the actual lint run.
    if ! "$PY" -m pre_commit install-hooks; then
        echo "$boot_err" >&2
        return 1
    fi
}

run_all_files() {
    log "running: pre-commit run --all-files"
    "$PY" -m pre_commit run --all-files
}

lint_changed() {
    ensure_pre_commit

    local base="" head="HEAD" reason=""

    if [[ -n "${CI_MERGE_REQUEST_DIFF_BASE_SHA:-}" ]]; then
        if git cat-file -e "${CI_MERGE_REQUEST_DIFF_BASE_SHA}^{commit}" 2>/dev/null; then
            base="$CI_MERGE_REQUEST_DIFF_BASE_SHA"
            reason="merge_request_event (CI_MERGE_REQUEST_DIFF_BASE_SHA)"
        else
            log "WARNING: MR diff base ${CI_MERGE_REQUEST_DIFF_BASE_SHA} is unreachable; falling through to merge-base / all-files"
        fi
    fi

    if [[ -z "$base" && -n "${CI_COMMIT_BEFORE_SHA:-}" && "${CI_COMMIT_BEFORE_SHA}" != "$ZERO_SHA" ]] \
        && git cat-file -e "${CI_COMMIT_BEFORE_SHA}^{commit}" 2>/dev/null; then
        base="$CI_COMMIT_BEFORE_SHA"
        reason="push (CI_COMMIT_BEFORE_SHA)"
    fi

    if [[ -z "$base" ]]; then
        local default_branch="${CI_DEFAULT_BRANCH:-develop}"
        local ref="origin/${default_branch}"
        if ! git rev-parse --verify --quiet "${ref}^{commit}" >/dev/null 2>&1; then
            log "local ref ${ref} missing; attempting fetch"
            git fetch --quiet origin "${default_branch}" 2>/dev/null || true
        fi
        if git rev-parse --verify --quiet "${ref}^{commit}" >/dev/null 2>&1; then
            base="$(git merge-base "${ref}" "$head" 2>/dev/null || true)"
            [[ -n "$base" ]] && reason="merge-base with ${ref}"
        fi
    fi

    if [[ -z "$base" ]]; then
        log "WARNING: diff base is UNRESOLVABLE (no MR base, no push before-sha, no reachable ${CI_DEFAULT_BRANCH:-develop})"
        log "WARNING: falling back to linting ALL files so nothing is skipped"
        run_all_files
        return $?
    fi

    log "diff base source : ${reason}"
    log "base SHA         : ${base}"
    log "head             : ${head} ($(git rev-parse "$head"))"

    local files=()
    while IFS= read -r file; do
        [[ -n "$file" ]] && files+=("$file")
    done < <(git diff --diff-filter=ACMR --name-only "$base" "$head")

    if [[ ${#files[@]} -eq 0 ]]; then
        log "no changed files to lint"
        return 0
    fi

    log "files checked:"
    printf '[lint]   %s\n' "${files[@]}"

    local rc=0
    "$PY" -m pre_commit run --files "${files[@]}" || rc=$?
    if [[ $rc -ne 0 ]]; then
        log "FAIL: pre-commit exited with code ${rc} (see the failing hook id above)"
    fi
    return $rc
}

lint_all() {
    # Create the report dir first so a bootstrap failure still leaves the
    # lint_reports/ artifact (with the bootstrap error captured in the log).
    local report_dir="${CI_PROJECT_DIR:-.}/lint_reports"
    local report="${report_dir}/lint_all.log"
    mkdir -p "$report_dir"

    # Run in this shell (NOT a pipe subshell) so the $PY resolved by ensure_py310
    # persists for compileall and pre-commit below; still tee output to the report.
    ensure_pre_commit > >(tee -a "$report") 2>&1 || true

    # compileall must not abort the job: capture its status so the cpp audit and
    # pre-commit --all-files still run and the full report is always written.
    local compile_rc=0
    log "running: python -m compileall -q -f tilelang (report -> ${report})"
    if ! "$PY" -m compileall -q -f tilelang 2>&1 | tee -a "$report"; then
        compile_rc=1
        log "compileall reported errors; continuing so the report is still produced"
    fi

    if [[ -f maint/scripts/audit_cpp_api_style.py ]]; then
        log "running (warning-only): audit_cpp_api_style.py --limit 20"
        python3 maint/scripts/audit_cpp_api_style.py --limit 20 2>&1 | tee -a "$report" || \
            log "cpp-api audit reported findings (warning-only, not failing)"
    fi

    log "running: pre-commit run --all-files (report -> ${report})"
    local precommit_rc=0
    "$PY" -m pre_commit run --all-files 2>&1 | tee -a "$report" || precommit_rc=$?
    if [[ $precommit_rc -ne 0 ]]; then
        log "audit tier found violations (exit ${precommit_rc}); this tier is non-blocking (allow_failure)"
    fi

    # Decide the final code only now, reflecting real failures (compileall or lint).
    if [[ $compile_rc -ne 0 || $precommit_rc -ne 0 ]]; then
        return 1
    fi
    return 0
}

main() {
    local mode="${1:-}"
    case "$mode" in
        changed) lint_changed ;;
        all)     lint_all ;;
        *)
            echo "usage: $0 {changed|all}" >&2
            exit 2
            ;;
    esac
}

main "$@"
