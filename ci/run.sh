#!/bin/bash
# Unified test-run entry for TileLang.
#
# Usage: ci/run.sh [--fresh] [--list FILE]... [--case SEL]... [--ci] [--dry-run]
#   --list FILE   a test-case list file (repeatable)
#   --case SEL    a single selector (repeatable): a .py path, a "python x.py"
#                 command, or a pytest nodeid (path.py::Class::test)
#   --ci          CI flavor: bring up the isolated conda env, install the wheel,
#                 export the audit-plugin env, and emit JUnit (self-contained)
#   --fresh       clear the script-managed tilelang cache before running
#   --dry-run     resolve + dedup + print the ordered plan, then exit without running
#
# Selection: with neither --list nor --case, defaults to ci_test_case_list_tilelang.txt.
# Any explicit source suppresses the default. All sources merge into one ordered
# list with stable first-seen dedup (on the resolved selector). Retry / tally /
# exit-code semantics are unchanged from ci_run_test_list ("any attempt passes =>
# success"; nonzero exit iff any final FAIL/TIMEOUT).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

INVOKE_CWD="$PWD"
TILELANG_HOME="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="$TILELANG_HOME"
DIST_DIR="$TILELANG_HOME/dist"
DEFAULT_LIST="$TILELANG_HOME/ci_test_case_list_tilelang.txt"
SUITE="tilelang"

# Distinct non-zero exit codes for actionable failures.
EXIT_USAGE=2
EXIT_LIST_MISSING=3
EXIT_CASE_INVALID=4
EXIT_EMPTY_PLAN=5

usage () {
    echo "Usage: ci/run.sh [--fresh] [--list FILE]... [--case SEL]... [--ci] [--dry-run]"
}

FRESH=0 CI_MODE=0 DRY_RUN=0
declare -a SRC_TYPE=() SRC_VAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fresh) FRESH=1; shift ;;
        --ci) CI_MODE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --list)
            [[ -n "${2:-}" ]] || { echo "ERROR: --list requires a FILE" >&2; usage >&2; exit $EXIT_USAGE; }
            SRC_TYPE+=("list"); SRC_VAL+=("$2"); shift 2 ;;
        --case)
            [[ -n "${2:-}" ]] || { echo "ERROR: --case requires a SELECTOR" >&2; usage >&2; exit $EXIT_USAGE; }
            SRC_TYPE+=("case"); SRC_VAL+=("$2"); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown flag: $1" >&2; usage >&2; exit $EXIT_USAGE ;;
    esac
done

# No explicit source -> default list. Any explicit source suppresses the default.
if [[ ${#SRC_TYPE[@]} -eq 0 ]]; then
    SRC_TYPE+=("list"); SRC_VAL+=("$DEFAULT_LIST")
fi

# In CI flavor the built pytest command carries the audit plugin; set it early so
# --dry-run also previews the real command form.
[[ $CI_MODE -eq 1 ]] && export PYTEST_AUDIT_PLUGIN_ARGS="-p pytest_node_audit"

# Echo a path made relative to the repo root when it is under it, else absolute.
_to_root_rel () {
    local abs="$1"
    case "$abs" in
        "$ROOT"/*) echo "${abs#"$ROOT"/}" ;;
        *) echo "$abs" ;;
    esac
}

# Normalize one selector to a plan token (root-relative when under the repo root).
# Accepted forms: a bare .py path, a `python x.py` command, and a pytest nodeid
# (path.py::...). A leading `pytest ` is NOT accepted (build_plan would turn it
# into a bogus path), nor is any other unrecognized shape.
# stdout: the token. return: 0 ok, 1 skip (list line the parser also ignores),
# 2 invalid (message on stderr).
normalize_selection () {
    local raw="$1" base="$2" origin="$3"
    raw="${raw#./}"
    # Command form: only `python <script.py>`.
    if [[ $raw =~ ^python[[:space:]] ]]; then
        local -a parts
        read -ra parts <<< "$raw"
        local script="${parts[1]:-}"
        if [[ ${#parts[@]} -eq 2 && "$script" == *.py ]]; then
            local abs="$script"; [[ "$abs" != /* ]] && abs="$base/$abs"
            if [[ "$origin" == "case" && ! -f "$abs" ]]; then
                echo "ERROR: --case script not found: $script" >&2; return 2
            fi
            echo "${parts[0]} $(_to_root_rel "$abs")"; return 0
        fi
        [[ "$origin" == "case" ]] && { echo "ERROR: invalid --case selector: $raw" >&2; return 2; }
        return 1
    fi
    # Bare .py path or pytest nodeid (path.py::...): a single whitespace-free token.
    if [[ ! $raw =~ [[:space:]] && $raw =~ \.py(::|$) ]]; then
        local file_part="${raw%%::*}" node_suffix=""
        [[ "$raw" == *::* ]] && node_suffix="::${raw#*::}"
        local abs="$file_part"; [[ "$abs" != /* ]] && abs="$base/$abs"
        if [[ ! -f "$abs" && "$origin" == "case" ]]; then
            echo "ERROR: --case file not found: $file_part" >&2; return 2
        fi
        echo "$(_to_root_rel "$abs")${node_suffix}"; return 0
    fi
    if [[ "$origin" == "case" ]]; then
        echo "ERROR: invalid --case selector: $raw" >&2; return 2
    fi
    return 1
}

declare -A SEEN=()
declare -a MERGED=() DEDUPED=() DROPPED=()
# Populated by ci_parse_list_into when it skips an unparsable line (build_plan).
declare -a CI_DROPPED_LINES=()
add_token () {
    local tok="$1"
    if [[ -n "${SEEN[$tok]:-}" ]]; then
        DEDUPED+=("$tok"); return 0
    fi
    SEEN[$tok]=1; MERGED+=("$tok")
}

# Expand every source in order into the merged, deduped plan.
for idx in "${!SRC_TYPE[@]}"; do
    stype="${SRC_TYPE[$idx]}"; sval="${SRC_VAL[$idx]}"
    if [[ "$stype" == "list" ]]; then
        if [[ ! -f "$sval" ]]; then
            echo "ERROR: --list file not found: $sval" >&2; exit $EXIT_LIST_MISSING
        fi
        lst_dir="$(cd "$(dirname "$sval")" && pwd)"
        ci_load_test_timeouts "$sval" pytest || exit $EXIT_CASE_INVALID
        while IFS= read -r line; do
            [[ $line =~ ^#.*$ ]] && continue
            [[ -z $line ]] && continue
            tok=$(normalize_selection "$line" "$lst_dir" "list"); rc=$?
            if [[ $rc -eq 1 ]]; then
                echo "WARNING: run.sh: skipping unparsable line in ${sval}: ${line}" >&2
                DROPPED+=("$line"); continue
            fi
            [[ $rc -eq 2 ]] && exit $EXIT_CASE_INVALID
            add_token "$tok"
        done < "$sval"
    else
        tok=$(normalize_selection "$sval" "$INVOKE_CWD" "case"); rc=$?
        [[ $rc -eq 2 ]] && exit $EXIT_CASE_INVALID
        [[ $rc -eq 1 ]] && { echo "ERROR: invalid --case selector: $sval" >&2; exit $EXIT_CASE_INVALID; }
        add_token "$tok"
    fi
done

if [[ ${#MERGED[@]} -eq 0 ]]; then
    echo "ERROR: empty plan — nothing to run" >&2; exit $EXIT_EMPTY_PLAN
fi

echo "Resolved plan: ${#MERGED[@]} case(s) (root=$ROOT, suite=$SUITE)"
[[ ${#DEDUPED[@]} -gt 0 ]] && echo "Deduped ${#DEDUPED[@]} duplicate selection(s)"
[[ ${#DROPPED[@]} -gt 0 ]] && echo "Dropped ${#DROPPED[@]} unparsable list line(s) (see warnings above)"

# Build the plan into CI_CMDS/CI_LABELS/CI_CASE_MODES from a merged list file.
build_plan () {
    local merged_file="$1"
    local default_timeout="${CASE_TIMEOUT:-900}"
    if ! [[ "$default_timeout" =~ ^[1-9][0-9]*$ ]] || [[ ${#default_timeout} -gt 9 ]]; then
        echo "ERROR: Invalid CASE_TIMEOUT=$default_timeout (need positive integer seconds)" >&2
        return $EXIT_CASE_INVALID
    fi
    printf '%s\n' "${MERGED[@]}" > "$merged_file"
    CI_CMDS=(); CI_LABELS=(); CI_CASE_MODES=()
    ci_parse_list_into "$merged_file" pytest "$ROOT"
}

if [[ $DRY_RUN -eq 1 ]]; then
    merged_file="$(mktemp "${TMPDIR:-/tmp}/ci_run_plan.XXXXXX")"
    build_plan "$merged_file" || { plan_ret=$?; rm -f "$merged_file"; exit "$plan_ret"; }
    rm -f "$merged_file"
    echo "==================== Dry-run plan (${#CI_CMDS[@]} case(s)) ===================="
    for i in "${!CI_CMDS[@]}"; do
        timeout_key=$(ci_timeout_key "${CI_LABELS[$i]}" "$ROOT") || exit $EXIT_CASE_INVALID
        printf '%3d  [%-7s]  %s (timeout=%ss)\n         -> %s\n' \
            "$((i+1))" "${CI_CASE_MODES[$i]}" "${CI_LABELS[$i]}" \
            "${CI_CASE_TIMEOUTS[$timeout_key]:-${CASE_TIMEOUT:-900}}" "${CI_CMDS[$i]}"
    done
    if [[ ${#DEDUPED[@]} -gt 0 ]]; then
        echo "-------------------- Deduped (dropped) selections --------------------"
        printf '     %s\n' "${DEDUPED[@]}"
    fi
    all_dropped=("${DROPPED[@]}" "${CI_DROPPED_LINES[@]}")
    if [[ ${#all_dropped[@]} -gt 0 ]]; then
        echo "-------------------- Unparsable (dropped) lines: ${#all_dropped[@]} --------------------"
        printf '     %s\n' "${all_dropped[@]}"
    fi
    exit 0
fi

ci_configure_ptcc || exit 1

# CI flavor: bring up the isolated env, install the wheel, export audit env.
if [[ $CI_MODE -eq 1 ]]; then
    time {
    ci_init_state "$TILELANG_HOME"
    ci_prepare_installed_wheel_env "$SCRIPT_DIR"
    ci_create_conda_env
    ci_export_tang_env
    ci_install_tilelang_whl "$DIST_DIR"
    check_exec ci_assert_installed_tilelang_wheel
    }
    export PYTEST_AUDIT_PLUGIN_ARGS="-p pytest_node_audit"
    export PYTEST_NODE_AUDIT_PATH="$(ci_failure_report_dir)/tilelang_nodes.jsonl"
    export PYTEST_NODE_AUDIT_SUITE="$SUITE"
    export PYTEST_LAUNCH_CWD="$CI_TMP_DIR"
    export PYTEST_NODE_AUDIT_TEST_CWD="$TILELANG_HOME"
    export PYTEST_NODE_AUDIT_SOURCE_ROOT="$TILELANG_HOME"
    export TILELANG_TEST_INSTALLED_WHEEL=1
    # The S2 runner carries CUDA toolchain metadata as well as PTPU, so auto target
    # detection would otherwise prefer CUDA even with no CUDA device visible.
    export TILELANG_DEFAULT_TARGET="${TILELANG_DEFAULT_TARGET:-tang}"
fi

if [[ $FRESH -eq 1 ]]; then
    fresh_cache="${TILELANG_CACHE_DIR:-$HOME/.tilelang/cache}"
    echo "--fresh: clearing tilelang cache ${fresh_cache}"
    rm -rf "$fresh_cache"
fi

# Run from the repo root so command-mode relative scripts resolve as in CI.
cd "$ROOT"
REPORT_DIR="$(ci_failure_report_dir)"

[[ $CI_MODE -eq 1 ]] && ci_assert_runtime_stack

merged_file="$(mktemp "${TMPDIR:-/tmp}/ci_run_plan.XXXXXX")"
build_plan "$merged_file" || { plan_ret=$?; rm -f "$merged_file"; exit "$plan_ret"; }

# Finalize from EXIT so a nested helper's deliberate `exit` (for example, an
# unusable device detected between retries) cannot bypass report synthesis.
# ci_init_state owns the normal CI cleanup trap; temporarily replace it here,
# then invoke cleanup explicitly so it receives the final exit code.
finalize_run () {
    local run_ret=$?
    local report_ret=0
    trap - EXIT

    python3 "$SCRIPT_DIR/junit_from_jsonl.py" --suite "$SUITE" --report-dir "$REPORT_DIR" \
        --expected-cases "${#CI_CMDS[@]}" || report_ret=$?
    if [[ $report_ret -ne 0 ]]; then
        echo "ERROR: JUnit synthesis failed (exit $report_ret)" >&2
        if [[ $run_ret -eq 0 ]]; then
            run_ret=$report_ret
        else
            echo "Preserving test/recording exit code $run_ret" >&2
        fi
    fi

    if [[ $CI_MODE -eq 1 ]]; then
        ci_cleanup_state "$run_ret"
    else
        rm -f "$merged_file"
    fi
    exit "$run_ret"
}

trap finalize_run EXIT
run_ret=0
ci_run_prepared_cases "$SUITE" "$ROOT" || run_ret=$?
exit "$run_ret"
