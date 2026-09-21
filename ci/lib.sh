# Shared CI library for TileLang and downstream operator repos.
# Source this file; it only defines variables + functions, never exits on its own.
# Public defaults are provided where portable. Vendor packages and host toolchain
# locations must be supplied explicitly by the caller.

# -------------------- Version pins (single source of truth) --------------------
TARGET_TORCH_VERSION="${TARGET_TORCH_VERSION:-2.10.0}"
TARGET_TORCH_PTPU_PKG="${TARGET_TORCH_PTPU_PKG:-}"
TARGET_TRITON_VERSION="${TARGET_TRITON_VERSION:-3.4.3+git7e2003b3}"
TARGET_TRITON_PKG="${TARGET_TRITON_PKG:-}"
TARGET_TORCH_PKG_URL="${TARGET_TORCH_PKG_URL:-https://download.pytorch.org/whl/cpu}"
export TARGET_TORCH_VERSION TARGET_TRITON_VERSION TARGET_TORCH_PKG_URL
export TARGET_TORCH_PTPU_PKG TARGET_TRITON_PKG
TARGET_TVM_FFI_VERSION="${TARGET_TVM_FFI_VERSION:-0.1.11+sunrise.1}"

# Keep GitLab defaults; restricted public runners opt into sudo-n and redacted logs.
TILELANG_CI_RESET_MODE="${TILELANG_CI_RESET_MODE:-password}"
TILELANG_CI_PUBLIC_LOGS="${TILELANG_CI_PUBLIC_LOGS:-0}"
export TILELANG_CI_RESET_MODE TILELANG_CI_PUBLIC_LOGS
declare -A CI_CASE_TIMEOUTS=()

# -------------------- Host toolchain paths (machine-specific) --------------------
# LLVM_HOME has no built-in default (a personal absolute path was removed); it must
# come from the environment (CI runner or local shell). Empty here simply means
# "unset unless the environment provides it".
LLVM_HOME="${LLVM_HOME:-}"
LLVM_VERSION_MAJOR="${LLVM_VERSION_MAJOR:-20}"
LLVM_VERSION_MINOR="${LLVM_VERSION_MINOR:-0}"
TANGRT_PATH="${TANGRT_PATH:-/usr/local/tangrt/}"
STPU_TANGRT_PATH="${STPU_TANGRT_PATH:-/usr/local/tangrt}"
TANGRT_LIB_PATH="${TANGRT_LIB_PATH:-${TANGRT_PATH%/}/targets/linux-x86_64/lib:${TANGRT_PATH%/}/lib/linux-x86_64:/usr/lib64}"
VENDOR_INCLUDE_DIRS="${VENDOR_INCLUDE_DIRS:-/usr/local/tangrt/include}"
PTCC_PATH="${PTCC_PATH:-}"
CI_PTCC_RESOLVER="$(cd "${BASH_SOURCE[0]%/*}" && pwd)/../tilelang/_ptcc.py"
CMAKE_PATH="${CMAKE_PATH:-/usr/local/tangrt/cmake}"
CMAKE_ROOT="${CMAKE_ROOT:-/usr/local/bin/cmake}"
TANG_CMAKE_PACKAGE_DIR="${TANG_CMAKE_PACKAGE_DIR:-${TANGRT_PATH}/targets/linux-x86_64/lib/cmake/TANG}"
TANGRT_CMAKE_PACKAGE_DIR="${TANGRT_CMAKE_PACKAGE_DIR:-${TANGRT_PATH}/targets/linux-x86_64/lib/cmake/TANGRT}"

# -------------------- Helpers --------------------
ci_configure_ptcc () {
    PTCC_PATH=$(PTCC_PATH="$PTCC_PATH" python3 "$CI_PTCC_RESOLVER" "$TANGRT_PATH") || return 1
    export PTCC_PATH
    echo "PTCC: $(readlink -f "$PTCC_PATH")"
    "$PTCC_PATH" --version || return 1
    echo "PTCC JIT profile: ${PTCC_JIT_PROFILE:-llvm20}"
    echo "TANG toolkit: $TANGRT_PATH; S2 build arch: stcu"
}

# GitLab job cancel (SIGTERM): child pid of the in-flight test, and shutdown latch.
CI_TEST_PID=""
CI_SHUTTING_DOWN=0
# timeout(1) SIGKILL grace after wall-clock SIGTERM (hung pytest/GPU).
CI_TIMEOUT_KILL_AFTER="${CI_TIMEOUT_KILL_AFTER:-30}"
# Cancel trap: max seconds to wait after SIGTERM before SIGKILL (keep short).
CI_CANCEL_KILL_AFTER="${CI_CANCEL_KILL_AFTER:-5}"

check_exec () {
    echo "Execute $@"
    "$@"
    local ret=$?
    if [ $ret -ne 0 ]; then
        echo "Run $@ failed"
        exit $ret
    fi
}

# GitLab Runner cancel sends SIGTERM to the job process group. Without special
# handling, bash waits for a foreground `timeout`/pytest to finish before
# exiting, and GNU `timeout` (no --foreground) puts the test in a new process
# group so cancel never reaches it. Run tests in the background + wait so the
# trap can interrupt immediately, and force SIGKILL after a short grace period.
ci_on_job_cancel () {
    if [[ ${CI_SHUTTING_DOWN:-0} -eq 1 ]]; then
        return
    fi
    CI_SHUTTING_DOWN=1
    # Avoid re-entrancy while we tear down children.
    trap - TERM INT
    echo "$(date) Job cancelled (SIGTERM/SIGINT), terminating child processes..."
    if [[ -n ${CI_TEST_PID} ]] && kill -0 "${CI_TEST_PID}" 2>/dev/null; then
        kill -TERM "${CI_TEST_PID}" 2>/dev/null || true
        pkill -TERM -P "${CI_TEST_PID}" 2>/dev/null || true
        local _i
        for ((_i = 0; _i < CI_CANCEL_KILL_AFTER; _i++)); do
            kill -0 "${CI_TEST_PID}" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "${CI_TEST_PID}" 2>/dev/null; then
            echo "$(date) Child pid=${CI_TEST_PID} still alive, sending SIGKILL"
            kill -KILL "${CI_TEST_PID}" 2>/dev/null || true
            pkill -KILL -P "${CI_TEST_PID}" 2>/dev/null || true
        fi
    fi
    local kids
    kids=$(jobs -p 2>/dev/null || true)
    if [[ -n ${kids} ]]; then
        kill -TERM ${kids} 2>/dev/null || true
        sleep 1
        kill -KILL ${kids} 2>/dev/null || true
    fi
    exit 143
}

# List ctx@pid client dir names under /proc/pt/ptpu{N}/ (e.g. 0@779457).
ci_card_ctx_clients () {
    local state_dir=$1
    ls -1 "${state_dir}" 2>/dev/null | grep -E '^[0-9]+@[0-9]+$' || true
}

# Return 0 when any ctx@pid client dir is present (card occupied).
ci_card_has_ctx () {
    local clients
    clients=$(ci_card_ctx_clients "$1")
    [[ -n ${clients} ]]
}

# Print device client dirs under /proc/pt/ptpu{N}/ named like 0@779457 (ctx@pid).
ci_list_card_clients () {
    local state_dir=$1
    if [[ "${TILELANG_CI_PUBLIC_LOGS:-0}" == "1" ]]; then
        echo "ci_check_card_state: client details redacted for public CI"
        return 0
    fi
    local clients c pid cmd
    clients=$(ci_card_ctx_clients "${state_dir}")
    if [[ -z ${clients} ]]; then
        echo "ci_check_card_state: no ctx@pid clients under ${state_dir}"
        return 0
    fi
    if [[ "${TILELANG_CI_PUBLIC_LOGS:-0}" == "1" ]]; then
        echo "ci_check_card_state: device has active clients (details redacted for public CI)"
        return 0
    fi
    echo "ci_check_card_state: clients under ${state_dir}:"
    while IFS= read -r c; do
        [[ -z $c ]] && continue
        pid=${c#*@}
        cmd=$(ps -p "${pid}" -o user=,pid=,cmd= 2>/dev/null || echo "(process gone)")
        echo "  ${c}  ${cmd}"
    done <<< "${clients}"
}

# Return 0 when every live ctx@pid client is a benign holder (heartbeat or pt_smi).
# Gone pids are ignored; no live clients -> return 1.
# On success sets CI_CARD_BENIGN_HOLDERS to the matched kinds ("heartbeat",
# "pt_smi" or "heartbeat,pt_smi") so callers can log which daemon collided.
ci_card_clients_are_benign_only () {
    local state_dir=$1
    local clients c pid cmd
    local found=0 hb=0 smi=0
    CI_CARD_BENIGN_HOLDERS=""
    clients=$(ci_card_ctx_clients "${state_dir}")
    [[ -z ${clients} ]] && return 1
    while IFS= read -r c; do
        [[ -z $c ]] && continue
        pid=${c#*@}
        cmd=$(ps -p "${pid}" -o cmd= 2>/dev/null) || continue
        found=1
        if [[ ${cmd} =~ [Hh]eartbeat ]]; then
            hb=1
        elif [[ ${cmd} =~ pt_smi ]]; then
            smi=1
        else
            return 1
        fi
    done <<< "${clients}"
    [[ $found -eq 1 ]] || return 1
    [[ $hb -eq 1 ]] && CI_CARD_BENIGN_HOLDERS="heartbeat"
    [[ $smi -eq 1 ]] && CI_CARD_BENIGN_HOLDERS="${CI_CARD_BENIGN_HOLDERS:+${CI_CARD_BENIGN_HOLDERS},}pt_smi"
    return 0
}

ci_dump_device_status () {
    if [[ "${TILELANG_CI_PUBLIC_LOGS:-0}" == "1" ]]; then
        ci_save_device_logs
        return $?
    fi
    echo "========== device status dump =========="
    echo "---- cat /proc/pt/ptpu*/state ----"
    if compgen -G "/proc/pt/ptpu*/state" >/dev/null 2>&1; then
        for state_file in /proc/pt/ptpu*/state; do
            echo "## ${state_file}"
            cat "${state_file}" 2>&1 || echo "cat ${state_file} failed (exit=$?)"
            ci_list_card_clients "$(dirname "${state_file}")"
        done
    else
        echo "no /proc/pt/ptpu*/state found"
    fi
    echo "---- lspci -d 1ecc: ----"
    lspci -d 1ecc: 2>&1 || echo "lspci -d 1ecc: failed (exit=$?)"
    echo "========================================"
}

# Task may run when /proc/pt/ptpu{dev_id}/state has:
#   state:       READY
#   fatal_error: 0
# and there is no ctx@pid client dir under /proc/pt/ptpu{dev_id}/.
# usage is informational only: some edge cases report usage != 0 with no ctx;
# that alone must not block the job.
# If ctx is held, wait and re-check: up to 10 times for heartbeat/pt_smi only,
# else up to 3 times. Wait 20s when holders are only heartbeat/pt_smi, else 10s.
# The lenient regime latches once seen, so a racy sample (client dir momentarily
# empty) cannot shrink the budget back to 3 mid-wait.
# Missing /proc/pt/ptpu{dev_id}/ (or state) is abnormal.
ci_check_card_state () {
    local card=${TANG_VISIBLE_DEVICES:-0}
    local state_dir="/proc/pt/ptpu${card}"
    local state_file="${state_dir}/state"
    local content state_val usage_val fatal_val
    local max_attempts=3
    local max_attempts_benign=10
    local wait_secs=10
    local heartbeat_wait_secs=20
    local attempt this_wait limit benign_seen=0 holders=""
    if [[ ! -d $state_dir ]]; then
        echo "ERROR: 设备异常 — directory ${state_dir} does not exist"
        ci_dump_device_status
        exit 1
    fi
    if [[ ! -e $state_file ]]; then
        echo "ERROR: 设备异常 — missing ${state_file}"
        ci_dump_device_status
        exit 1
    fi
    for (( attempt = 1; ; attempt++ )); do
        if [[ ${CI_SHUTTING_DOWN:-0} -eq 1 ]]; then
            exit 143
        fi
        content=$(cat "${state_file}" 2>&1) || {
            echo "ERROR: 设备异常 — cannot read ${state_file}"
            ci_dump_device_status
            exit 1
        }
        state_val=$(echo "${content}" | awk -F: '/^[[:space:]]*state:/{gsub(/[[:space:]]/,"",$2); print $2; exit}')
        usage_val=$(echo "${content}" | awk -F: '/^[[:space:]]*usage:/{gsub(/[[:space:]]/,"",$2); print $2; exit}')
        fatal_val=$(echo "${content}" | awk -F: '/^[[:space:]]*fatal_error:/{gsub(/[[:space:]]/,"",$2); print $2; exit}')
        if [[ $state_val != "READY" || $fatal_val != "0" ]]; then
            echo "ERROR: 设备异常 — ptpu${card} (expect state=READY fatal_error=0; got state=${state_val} usage=${usage_val} fatal_error=${fatal_val})"
            echo "---- ${state_file} ----"
            [[ "${TILELANG_CI_PUBLIC_LOGS:-0}" == "1" ]] || echo "${content}"
            ci_list_card_clients "${state_dir}"
            ci_dump_device_status
            exit 1
        fi
        # Free card: READY + fatal_error=0 + no ctx. usage != 0 alone is OK.
        if ! ci_card_has_ctx "${state_dir}"; then
            if [[ $usage_val != "0" ]]; then
                echo "ci_check_card_state: ptpu${card} OK (state=READY fatal_error=0 no ctx; usage=${usage_val} nonzero but ignored)"
            else
                echo "ci_check_card_state: ptpu${card} OK (state=READY usage=0 fatal_error=0 no ctx)"
            fi
            return 0
        fi
        if ci_card_clients_are_benign_only "${state_dir}"; then
            benign_seen=1
            holders=${CI_CARD_BENIGN_HOLDERS}
        fi
        # Latch the lenient regime: once a heartbeat/pt_smi-only collision is seen,
        # keep the 10-attempt budget for the rest of the loop. A later racy sample
        # (client dir momentarily empty, or pid just gone) must not shrink the
        # budget back to 3 and abort a job that only collides with the daemons.
        if [[ $benign_seen -eq 1 ]]; then
            this_wait=$heartbeat_wait_secs
            limit=$max_attempts_benign
        else
            this_wait=$wait_secs
            limit=$max_attempts
        fi
        # Busy card: print state + holders once per retry, keep the happy path quiet.
        echo "ci_check_card_state: ptpu${card} busy (ctx held, usage=${usage_val}, attempt ${attempt}/${limit})"
        echo "---- ${state_file} ----"
        [[ "${TILELANG_CI_PUBLIC_LOGS:-0}" == "1" ]] || echo "${content}"
        ci_list_card_clients "${state_dir}"
        if [[ $benign_seen -eq 1 ]]; then
            echo "ci_check_card_state: holders are ${holders:-heartbeat/pt_smi} only, wait up to ${limit} attempts (${this_wait}s each)"
        fi
        if [[ $attempt -ge $limit ]]; then
            break
        fi
        echo "ci_check_card_state: wait ${this_wait}s then retry"
        sleep "${this_wait}"
    done
    echo "ERROR: 设备异常 — ptpu${card} ctx still held after ${attempt} attempts (last usage=${usage_val})"
    ci_dump_device_status
    exit 1
}

# Directory where every suite drops its structured per-case report. Anchored to
# CI_PROJECT_DIR (always the tilelang root in the pipeline) so operator runs under
# .ci-operators/<op>/ still land in ONE dir collectible by a single artifacts:path.
# Falls back to PWD for local runs.
ci_failure_report_dir () {
    echo "${CI_FAILURE_REPORT_DIR:-${CI_PROJECT_DIR:-$PWD}/ci_failure_reports}"
}

# Append one JSONL record. Calls with attempt metadata emit a case_attempt;
# calls without it retain the existing final case_result schema. The inline
# python3 only JSON-encodes arbitrary log text; it does not run the test case.
# PASS/SKIPPED pass an empty logfile to omit log_tail.
# Args: suite case command status exit_code elapsed reason logfile timeout_seconds
#       [attempt max_attempts will_retry]
ci_record_case_result () {
    local suite="$1" case_id="$2" command="$3" status="$4" exit_code="$5" elapsed="$6" reason="$7" logfile="$8"
    local timeout_seconds="${9:-}"
    local report_dir; report_dir="$(ci_failure_report_dir)"
    if ! mkdir -p "$report_dir"; then
        echo "ERROR: failed to create case-report directory: $report_dir" >&2
        return 1
    fi
    local tail_chars="${CI_CASE_LOG_TAIL_CHARS:-6000}"
    local tail=""
    if [[ -n "$logfile" && -f "$logfile" ]] && ! tail="$(tail -c "$tail_chars" "$logfile")"; then
        echo "ERROR: failed to read case log tail: $logfile" >&2
        return 1
    fi
    python3 - "$report_dir/${suite}.jsonl" "$suite" "$case_id" "$command" "$status" \
              "$exit_code" "$elapsed" "$reason" "$tail" "$timeout_seconds" <<'PY'
import json, os, sys
out, suite, case_id, command, status, exit_code, elapsed, reason, tail = sys.argv[1:10]
timeout_seconds = sys.argv[10]
rec = {
    "schema_version": 2, "record_kind": "case_result",
    "suite": suite, "job": os.getenv("CI_JOB_NAME", ""), "case": case_id,
    "command": command, "cwd": os.getcwd(), "status": status,
    "exit_code": int(exit_code), "elapsed_seconds": int(elapsed),
    "failure_reason": reason, "log_tail": tail,
    "pipeline_id": os.getenv("CI_PIPELINE_ID", ""), "job_id": os.getenv("CI_JOB_ID", ""),
    "job_name": os.getenv("CI_JOB_NAME", ""), "commit_sha": os.getenv("CI_COMMIT_SHA", ""),
    "project_id": os.getenv("CI_PROJECT_ID", ""), "mr_iid": os.getenv("CI_MERGE_REQUEST_IID", ""),
}
if timeout_seconds:
    rec["timeout_seconds"] = int(timeout_seconds)
with open(out, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
PY
}

# Append one JSONL record per ATTEMPT (additive; the final attempt is still also
# recorded via ci_record_case_result). record_kind:"case_attempt" carries attempt
# / total_attempts so per-retry history is inspectable; downstream case_result
# consumers ignore it (they filter on record_kind). FAIL/TIMEOUT attempts carry a
# bounded log tail; PASS/SKIPPED pass an empty logfile to omit it.
# Args: suite case command status exit_code elapsed attempt total_attempts reason logfile
ci_record_case_attempt () {
    local suite="$1" case_id="$2" command="$3" status="$4" exit_code="$5" elapsed="$6"
    local attempt="$7" total="$8" reason="$9" logfile="${10}"
    local timeout_seconds="${11:-}"
    local report_dir; report_dir="$(ci_failure_report_dir)"
    if ! mkdir -p "$report_dir"; then
        echo "ERROR: failed to create case-report directory: $report_dir" >&2
        return 1
    fi
    local tail_chars="${CI_CASE_LOG_TAIL_CHARS:-6000}"
    local tail=""
    if [[ -n "$logfile" && -f "$logfile" ]] && ! tail="$(tail -c "$tail_chars" "$logfile")"; then
        echo "ERROR: failed to read case log tail: $logfile" >&2
        return 1
    fi
    python3 - "$report_dir/${suite}.jsonl" "$suite" "$case_id" "$command" "$status" \
              "$exit_code" "$elapsed" "$attempt" "$total" "$reason" "$tail" "$timeout_seconds" <<'PY'
import json, os, sys
out, suite, case_id, command, status, exit_code, elapsed, attempt, total, reason, tail = sys.argv[1:12]
timeout_seconds = sys.argv[12]
rec = {
    "schema_version": 2, "record_kind": "case_attempt",
    "suite": suite, "job": os.getenv("CI_JOB_NAME", ""), "case": case_id,
    "command": command, "cwd": os.getcwd(), "status": status,
    "exit_code": int(exit_code), "elapsed_seconds": int(elapsed),
    "attempt": int(attempt), "total_attempts": int(total),
    "failure_reason": reason, "log_tail": tail,
    "pipeline_id": os.getenv("CI_PIPELINE_ID", ""), "job_id": os.getenv("CI_JOB_ID", ""),
    "job_name": os.getenv("CI_JOB_NAME", ""), "commit_sha": os.getenv("CI_COMMIT_SHA", ""),
    "project_id": os.getenv("CI_PROJECT_ID", ""), "mr_iid": os.getenv("CI_MERGE_REQUEST_IID", ""),
}
if timeout_seconds:
    rec["timeout_seconds"] = int(timeout_seconds)
with open(out, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
PY
}

ci_record_device_recovery () {
    local suite="$1" case_id="$2" attempt="$3" device="$4" status="$5" exit_code="$6" reason="$7"
    local report_dir; report_dir="$(ci_failure_report_dir)"
    mkdir -p "$report_dir" || return 1
    python3 - "$report_dir/${suite}.jsonl" "$suite" "$case_id" "$attempt" "$device" \
              "$status" "$exit_code" "$reason" <<'PY'
import json, os, sys
out, suite, case_id, attempt, device, status, exit_code, reason = sys.argv[1:9]
rec = {
    "schema_version": 2, "record_kind": "device_recovery",
    "suite": suite, "job": os.getenv("CI_JOB_NAME", ""), "case": case_id,
    "attempt": int(attempt), "action": "reset", "device": device,
    "status": status, "exit_code": int(exit_code), "reason": reason,
    "pipeline_id": os.getenv("CI_PIPELINE_ID", ""), "job_id": os.getenv("CI_JOB_ID", ""),
    "job_name": os.getenv("CI_JOB_NAME", ""), "commit_sha": os.getenv("CI_COMMIT_SHA", ""),
    "project_id": os.getenv("CI_PROJECT_ID", ""), "mr_iid": os.getenv("CI_MERGE_REQUEST_IID", ""),
}
with open(out, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
PY
}

# Reset the Tang GPU after a case times out (exit 124); a hung card would
# otherwise cascade-fail every later case. Resume only after verified recovery.
# Bound sudo/pt_smi so a stuck reset cannot block GitLab cancel forever.
ci_reset_gpu_on_timeout () {
    local dev="${TANG_VISIBLE_DEVICES:-0}"
    local mode="${TILELANG_CI_RESET_MODE:-password}"
    local suite="${1:-unknown}" case_id="${2:-unknown}" attempt="${3:-0}"
    local ret=0
    if [[ ${CI_SHUTTING_DOWN:-0} -eq 1 ]]; then
        echo "ci_reset_gpu_on_timeout: skip (job cancelling)"
        ci_record_device_recovery "$suite" "$case_id" "$attempt" "$dev" SKIPPED 0 "job cancelling" || return $?
        return 1
    fi
    echo "Resetting Tang device $dev after timeout ..."
    case "$mode" in
        sudo-n)
            if [[ "$dev" != "0" ]]; then
                echo "WARNING: public runner reset is restricted to Tang device 0 (got $dev)"
                ci_record_device_recovery "$suite" "$case_id" "$attempt" "$dev" SKIPPED 0 \
                    "public reset is restricted to device 0" || return $?
                return 1
            fi
            /usr/bin/timeout --foreground --kill-after=10s 60 \
                /usr/bin/sudo -n /usr/bin/pt_smi -r -i 0 || ret=$?
            ;;
        password)
            if ! command -v pt_smi >/dev/null 2>&1; then
                echo "WARNING: pt_smi is unavailable; cannot reset timed-out Tang device"
                ci_record_device_recovery "$suite" "$case_id" "$attempt" "$dev" SKIPPED 0 \
                    "pt_smi is unavailable" || return $?
                return 1
            fi
            if [[ -z "${SUDO_MAGICWORD:-}" ]]; then
                echo "WARNING: SUDO_MAGICWORD is unavailable; cannot reset timed-out Tang device"
                ci_record_device_recovery "$suite" "$case_id" "$attempt" "$dev" SKIPPED 0 \
                    "reset credential is unavailable" || return $?
                return 1
            fi
            printf '%s\n' "$SUDO_MAGICWORD" | sudo -S -p '' \
                /usr/bin/timeout --foreground --kill-after=10s 60 pt_smi -r -i "$dev" || ret=$?
            ;;
        disabled)
            echo "WARNING: Tang reset is disabled; runner operator intervention may be required"
            ci_record_device_recovery "$suite" "$case_id" "$attempt" "$dev" SKIPPED 0 \
                "reset mode is disabled" || return $?
            return 1
            ;;
        *)
            echo "WARNING: unsupported TILELANG_CI_RESET_MODE=$mode; Tang device was not reset"
            ci_record_device_recovery "$suite" "$case_id" "$attempt" "$dev" SKIPPED 0 \
                "unsupported reset mode" || return $?
            return 1
            ;;
    esac
    if [[ $ret -ne 0 ]]; then
        echo "WARNING: Tang reset failed (exit=${ret}); runner operator intervention may be required"
        ci_record_device_recovery "$suite" "$case_id" "$attempt" "$dev" FAIL "$ret" \
            "pt_smi reset failed" || return $?
        ci_dump_device_status
        return 1
    fi
    if ! (ci_check_card_state); then
        ci_record_device_recovery "$suite" "$case_id" "$attempt" "$dev" FAIL 1 \
            "device health check failed after reset" || return $?
        return 1
    fi
    echo "ci_reset_gpu_on_timeout: pt_smi -r -i ${dev} OK"
    ci_record_device_recovery "$suite" "$case_id" "$attempt" "$dev" PASS 0 "" || return $?
}

# Best-effort pytest skip detection from one case's output. Used by the caller only
# when ret is 0 or 5. Returns 0 (true) when the case looks fully skipped / empty.
ci_pytest_is_skipped () {
    local logfile="$1" ret="$2"
    # exit 5 = pytest collected no tests (no tests ran)
    [[ "$ret" -eq 5 ]] && return 0
    # exit 0 with a summary mentioning "N skipped" but no passed/failed/error
    if [[ "$ret" -eq 0 ]] && grep -qE '[0-9]+ skipped' "$logfile" \
       && ! grep -qE '[0-9]+ (passed|failed|error)' "$logfile"; then
        return 0
    fi
    return 1
}

# Load conda's shell integration without reading the user's interactive shell
# startup files.  This keeps local callers' selected environment and aliases out
# of CI setup while still making conda activate available in a plain shell.
ci_ensure_conda_shell () {
    if [[ "$(type -t conda 2>/dev/null)" == "function" ]]; then
        return 0
    fi

    local conda_exe="${CONDA_EXE:-}" candidate
    if [[ -z "$conda_exe" ]]; then
        conda_exe="$(command -v conda 2>/dev/null || true)"
    fi
    if [[ -z "$conda_exe" ]]; then
        for candidate in \
            "$HOME/miniconda3/bin/conda" \
            "$HOME/miniconda3/condabin/conda" \
            "$HOME/anaconda3/bin/conda" \
            /opt/conda/bin/conda; do
            if [[ -x "$candidate" ]]; then
                conda_exe="$candidate"
                break
            fi
        done
    fi
    if [[ -z "$conda_exe" ]]; then
        echo "ERROR: conda is unavailable; put conda on PATH, set CONDA_EXE, or install it under a standard prefix" >&2
        return 1
    fi

    local conda_base conda_sh
    conda_base="$("$conda_exe" info --base)" || {
        echo "ERROR: failed to query the conda base directory" >&2
        return 1
    }
    conda_sh="$conda_base/etc/profile.d/conda.sh"
    if [[ ! -r "$conda_sh" ]]; then
        echo "ERROR: conda shell integration is unreadable: $conda_sh" >&2
        return 1
    fi
    source "$conda_sh"
}

# Prefer the configured CMake executable, but reject wrappers that cannot run in
# the isolated environment and fall back to another installed CMake binary.
ci_resolve_cmake () {
    local candidate resolved
    for candidate in "${CMAKE_ROOT:-}" cmake cmake3; do
        [[ -z "$candidate" ]] && continue
        if [[ "$candidate" == */* ]]; then
            resolved="$candidate"
        else
            resolved="$(command -v "$candidate" 2>/dev/null || true)"
        fi
        if [[ -n "$resolved" && -x "$resolved" ]] && "$resolved" --version >/dev/null 2>&1; then
            echo "$resolved"
            return 0
        fi
    done
    echo "ERROR: no working CMake executable found" >&2
    return 1
}

# Remove source-checkout overrides before installing or testing a wheel.  Keep
# only an explicitly requested PYTHONPATH entry (the pytest audit plugin in CI).
ci_prepare_installed_wheel_env () {
    local pythonpath="${1:-}"
    export -n TILELANG_HOME 2>/dev/null || true
    unset TVM_HOME TVM_PREBUILD_PATH TVM_SOURCE_DIR TVM_LIBRARY_PATH TL_TEMPLATE_PATH
    unset TVM_IMPORT_PYTHON_PATH TVM_USE_RUNTIME_LIB SKIP_LOADING_TILELANG_SO PYTHONHOME
    unset TILELANG_TEST_INSTALLED_WHEEL TILELANG_WHEEL_PREFIX
    export PYTHONNOUSERSITE=1
    if [[ -n "$pythonpath" ]]; then
        export PYTHONPATH="$pythonpath"
    else
        unset PYTHONPATH
    fi
}

# Import TileLang outside the checkout and prove that both Python and the loaded
# package come from the isolated environment populated by ci_install_tilelang_whl.
ci_assert_installed_tilelang_wheel () {
    local expected_prefix="${CONDA_ENV_PREFIX:-${CONDA_PREFIX:-}}"
    if [[ -z "$expected_prefix" || -z "${CI_TMP_DIR:-}" ]]; then
        echo "ERROR: isolated conda prefix or temp directory is unavailable" >&2
        return 1
    fi
    export TILELANG_WHEEL_PREFIX="$expected_prefix"
    (
        cd "$CI_TMP_DIR" || exit 1
        ci_prepare_installed_wheel_env
        TILELANG_EXPECTED_PREFIX="$expected_prefix" python -I - <<'PY'
import importlib.metadata
import os
from pathlib import Path
import sys

import tilelang
import tvm
import tvm.base
import tvm_ffi
import tvm_ffi.core

expected_prefix = Path(os.environ["TILELANG_EXPECTED_PREFIX"]).resolve()
active_prefix = Path(sys.prefix).resolve()
package_file = Path(tilelang.__file__).resolve()
distribution_root = Path(
    importlib.metadata.distribution("tilelang-sunrise").locate_file("")
).resolve()

if active_prefix != expected_prefix:
    raise SystemExit(
        f"Python prefix mismatch: expected {expected_prefix}, got {active_prefix}"
    )
if expected_prefix not in package_file.parents:
    raise SystemExit(
        f"tilelang.__file__ is outside the isolated environment: {package_file}"
    )
if distribution_root not in package_file.parents:
    raise SystemExit(
        "tilelang.__file__ is outside the installed tilelang-sunrise distribution: "
        f"{package_file} (distribution root: {distribution_root})"
    )
def installed_files(name):
    dist = importlib.metadata.distribution(name)
    if not dist.files:
        raise SystemExit(f"{name}: installed file manifest is missing")
    return {Path(dist.locate_file(entry)).resolve() for entry in dist.files}


def verify_path(label, value, files):
    path = Path(value).resolve(strict=True)
    if path not in files or expected_prefix not in path.parents:
        raise SystemExit(f"{label}: resolved outside installed wheel files: {path}")
    print(f"{label}: {path}")


tilelang_files = installed_files("tilelang-sunrise")
ffi_files = installed_files("apache-tvm-ffi")
verify_path("installed tilelang", tilelang.__file__, tilelang_files)
verify_path("installed tvm", tvm.__file__, tilelang_files)
verify_path("installed tvm-ffi", tvm_ffi.__file__, ffi_files)
verify_path("tvm-ffi extension", tvm_ffi.core.__file__, ffi_files)
verify_path("tilelang native library", tilelang._LIB._name, tilelang_files)
verify_path("tvm compiler library", tvm.base._LIB._name, tilelang_files)
verify_path("tvm runtime library", tvm.base._LIB_RUNTIME._name, tilelang_files)
verify_path("tvm-ffi native library", tvm_ffi.LIB._name, ffi_files)
if tvm.get_global_func("target.build.tilelang_tang", allow_missing=True) is None:
    raise SystemExit("target.build.tilelang_tang is not registered")
print("target.build.tilelang_tang: registered")
print(f"installed wheel: python={sys.executable} tilelang={package_file}")
PY
    )
}

# Init job-isolated state under $1 (default: CI project dir / PWD) and arm cleanup.
# $2 is the env role (build|test, default test); it tags the state path so a build
# env and a test env never collide. Sets CI_STATE_ROOT, CONDA_ENV_PREFIX, TILELANG_CACHE_DIR.
# Arm cancel trap early so SIGTERM during pip install / env setup also exits promptly.
ci_init_state () {
    local base="${1:-${CI_PROJECT_DIR:-$PWD}}"
    local role="${2:-test}"
    CI_RUN_KEY="${CI_JOB_ID:-local-$$}"
    CI_STATE_ROOT="${base}/.ci-state/${CI_RUN_KEY}-${role}"
    CONDA_ENV_PREFIX="${CI_STATE_ROOT}/conda"
    TILELANG_CACHE_DIR="${CI_STATE_ROOT}/tilelang-cache"
    CI_TMP_DIR="${CI_STATE_ROOT}/tmp"
    export TMPDIR="$CI_TMP_DIR"
    CI_TEST_PID=""
    CI_SHUTTING_DOWN=0
    trap ci_cleanup_state EXIT
    trap ci_on_job_cancel TERM INT
    rm -rf "$CI_STATE_ROOT"
    mkdir -p "$CI_STATE_ROOT" "$CI_TMP_DIR"
    echo "CI state root: ${CI_STATE_ROOT}"
    echo "Conda env prefix: ${CONDA_ENV_PREFIX}"
}

ci_cleanup_state () {
    local trapped_exit_code=$?
    local exit_code="${1:-$trapped_exit_code}"
    trap - EXIT TERM INT
    set +e
    # Abnormal exit: stage device logs for GitLab artifacts before tearing down state.
    if [[ $exit_code -ne 0 ]]; then
        ci_save_device_logs || true
    fi
    echo "Cleaning job-isolated CI state: ${CI_STATE_ROOT}"
    if [[ "${CONDA_PREFIX:-}" == "$CONDA_ENV_PREFIX" ]]; then
        conda deactivate
    fi
    if [[ -d "$CONDA_ENV_PREFIX" ]]; then
        conda env remove --prefix "$CONDA_ENV_PREFIX" -y
    fi
    rm -rf "$CI_STATE_ROOT"
    exit "$exit_code"
}

# On job failure/cancel: write dmesg.log and copy PT200 pt.log into the workspace
# so GitLab artifacts can collect them (paths must be under CI_PROJECT_DIR).
ci_save_device_logs () {
    local dest="${CI_PROJECT_DIR:-$PWD}"
    if [[ "${TILELANG_CI_PUBLIC_LOGS:-0}" == "1" ]]; then
        local dev="${TANG_VISIBLE_DEVICES:-0}"
        local state_file="/proc/pt/ptpu${dev}/state"
        local state_val="missing" usage_val="" fatal_val=""
        if [[ -r "$state_file" ]]; then
            state_val=$(awk -F: '/^[[:space:]]*state:/{gsub(/[[:space:]]/,"",$2); print $2; exit}' "$state_file")
            usage_val=$(awk -F: '/^[[:space:]]*usage:/{gsub(/[[:space:]]/,"",$2); print $2; exit}' "$state_file")
            fatal_val=$(awk -F: '/^[[:space:]]*fatal_error:/{gsub(/[[:space:]]/,"",$2); print $2; exit}' "$state_file")
        fi
        mkdir -p "$dest" || return 1
        python3 - "$dest/sunrise_device_summary.json" "$dev" "$state_val" "$usage_val" "$fatal_val" <<'PY' || return $?
import json
import os
import sys

path, device, state, usage, fatal_error = sys.argv[1:]
record = {
    "schema_version": 1,
    "device": device,
    "state": state,
    "usage": usage,
    "fatal_error": fatal_error,
    "commit_sha": os.getenv("CI_COMMIT_SHA", ""),
    "job_name": os.getenv("CI_JOB_NAME", ""),
    "run_id": os.getenv("CI_PIPELINE_ID", ""),
}
with open(path, "w", encoding="utf-8") as output:
    json.dump(record, output, ensure_ascii=False, sort_keys=True)
    output.write("\n")
PY
        echo "ci_save_device_logs: wrote sanitized public summary ${dest}/sunrise_device_summary.json"
        return $?
    fi
    local dmesg_log="${dest}/dmesg.log"
    local pt_log_src="/var/log/pt200/pt.log"
    local pt_log_dst="${dest}/pt.log"
    mkdir -p "${dest}"
    echo "ci_save_device_logs: writing ${dmesg_log}"
    if ! dmesg -T > "${dmesg_log}" 2>&1; then
        echo "ci_save_device_logs: dmesg -T failed (exit=$?)" | tee -a "${dmesg_log}"
    fi
    if [[ -r $pt_log_src ]]; then
        echo "ci_save_device_logs: copying ${pt_log_src} -> ${pt_log_dst}"
        cp -f "${pt_log_src}" "${pt_log_dst}" || echo "ci_save_device_logs: copy ${pt_log_src} failed"
    else
        echo "ci_save_device_logs: ${pt_log_src} missing or unreadable"
    fi
}

# Create + activate the isolated conda env with base deps, torch and torch_ptpu.
ci_create_conda_env () {
    check_exec ci_ensure_conda_shell
    : "${TARGET_TORCH_PTPU_PKG:?Set TARGET_TORCH_PTPU_PKG to an accessible torch_ptpu wheel}"
    : "${TARGET_TRITON_PKG:?Set TARGET_TRITON_PKG to an accessible Triton wheel}"
    local conda_exe=""
    if command -v conda >/dev/null 2>&1; then
        conda_exe="$(command -v conda)"
    elif [[ -n ${HOME:-} && -f "$HOME/.bashrc" ]]; then
        source "$HOME/.bashrc"
        if command -v conda >/dev/null 2>&1; then
            conda_exe="$(command -v conda)"
        fi
    fi
    if [[ -z "$conda_exe" && -n ${CONDA_EXE:-} && -x $CONDA_EXE ]]; then
        conda_exe="$CONDA_EXE"
    fi
    if [[ -z "$conda_exe" ]]; then
        echo "ERROR: conda is not available on PATH and CONDA_EXE is not executable"
        return 1
    fi
    local conda_base conda_init
    conda_base="$("$conda_exe" info --base)"
    conda_init="$conda_base/etc/profile.d/conda.sh"
    if [[ ! -f $conda_init ]]; then
        echo "ERROR: conda shell initialization script not found: $conda_init"
        return 1
    fi
    source "$conda_init"
    conda create --prefix "$CONDA_ENV_PREFIX" python=3.10 -y
    conda activate "$CONDA_ENV_PREFIX"
    if [[ "${CONDA_PREFIX:-}" != "$CONDA_ENV_PREFIX" ]]; then
        echo "ERROR: failed to activate conda prefix $CONDA_ENV_PREFIX (current: ${CONDA_PREFIX:-<none>})"
        exit 1
    fi
    python -m pip install --upgrade pip
    # Keep this aligned with pyproject.toml: the wheel targets the Python 3.8
    # Limited API, which Cython 3.3 no longer supports.
    conda install numpy psutil "cython>=3.1.0,<3.3" pytest -y
    pip install einops cloudpickle tqdm scipy matplotlib pytest-instafail
    pip install torch=="$TARGET_TORCH_VERSION" --index-url "$TARGET_TORCH_PKG_URL"
    check_exec pip3 install "$TARGET_TORCH_PTPU_PKG"
    check_exec pip3 install "$TARGET_TRITON_PKG"
    ci_assert_runtime_stack
}

ci_assert_runtime_stack () {
    python3 - <<'PY'
import importlib.metadata as metadata
import os
import sys

expected_torch = os.environ["TARGET_TORCH_VERSION"]
expected_ptpu = os.environ["TARGET_TORCH_PTPU_PKG"].split("/")[-1].removesuffix(".whl")
expected_ptpu = expected_ptpu.removeprefix("torch_ptpu-").split("-cp", 1)[0]
expected_triton = os.environ["TARGET_TRITON_VERSION"]

actual_torch = metadata.version("torch").split("+", 1)[0]
actual_ptpu = metadata.version("torch-ptpu")
actual_triton = metadata.version("triton")
print(f"runtime stack: python={sys.executable} torch={actual_torch} torch_ptpu={actual_ptpu} triton={actual_triton}")
print(f"runtime stack: expected_ptpu={expected_ptpu} expected_triton={expected_triton}")
if actual_torch != expected_torch:
    raise SystemExit(f"torch mismatch: expected {expected_torch}, got {actual_torch}")
if actual_ptpu != expected_ptpu:
    raise SystemExit(f"torch_ptpu mismatch: expected {expected_ptpu}, got {actual_ptpu}")
if actual_triton != expected_triton:
    raise SystemExit(f"triton mismatch: expected {expected_triton}, got {actual_triton}")
PY
    check_exec python3 -m pip check
}

# Export TANG/PTPU runtime + build env vars. Call after the conda env is active.
ci_export_tang_env () {
    local tang_cmake_prefix="${TANGRT_PATH%/}/targets/linux-x86_64"
    local conda_cmake_prefix="${CONDA_PREFIX:?A conda environment must be active}"
    export ENV_PATH=$CONDA_PREFIX
    if [[ -n "$LLVM_HOME" ]]; then
        export LLVM_HOME
    else
        unset LLVM_HOME
    fi
    export LLVM_VERSION_MAJOR LLVM_VERSION_MINOR
    export TANGRT_PATH STPU_TANGRT_PATH VENDOR_INCLUDE_DIRS PTCC_PATH CMAKE_PATH CMAKE_ROOT
    # Prefer the physical target prefix over /usr/local/tangrt-* symlinks.  The
    # vendor package computes imported library paths relative to its config
    # file, so resolving the symlink first would incorrectly produce /usr/targets.
    export CMAKE_PREFIX_PATH="${tang_cmake_prefix}:${conda_cmake_prefix}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
    export PYTORCH_DIR=$ENV_PATH/lib/python3.10/site-packages/
    export PYTHON_INCLUDE_DIR=$ENV_PATH/include/python3.10
    export PTPU_PATH=$ENV_PATH/lib/python3.10/site-packages/torch_ptpu
    export LD_LIBRARY_PATH="${TANGRT_LIB_PATH}:$ENV_PATH/lib"
    export LIBRARY_PATH="${tang_cmake_prefix}/lib:${TANGRT_LIB_PATH}"
}

# Build tvm-ffi + tvm from source inside $1 (tilelang home). Caller installs
# tilelang itself afterwards (wheel for release, editable for operator repos).
ci_build_tvm () {
    local tilelang_home="$1"
    local dist_dir="$2"
    if [[ -z "$dist_dir" ]]; then
        echo "ERROR: ci_build_tvm requires an explicit wheel output directory"
        return 1
    fi
    local cmake_cmd
    cmake_cmd="$(ci_resolve_cmake)" || return 1
    echo "Using CMake: $cmake_cmd"
    mkdir -p "$dist_dir"
    local tvm_home="$tilelang_home/3rdparty/tvm_sunrise"
    local tvm_ffi_home="$tvm_home/3rdparty/tvm-ffi"
    local vendored_path
    for vendored_path in \
        "$tvm_home/LICENSE" \
        "$tvm_ffi_home/LICENSE" \
        "$tvm_ffi_home/3rdparty/dlpack/LICENSE" \
        "$tvm_ffi_home/3rdparty/libbacktrace/LICENSE"; do
        if [[ ! -f "$vendored_path" ]]; then
            echo "ERROR: required vendored source is missing: $vendored_path"
            return 1
        fi
    done
    export TVM_HOME=$tvm_home
    export PYTHONPATH=$tvm_home/python:$tvm_home/ffi/python:$PYTHONPATH
    pushd "$tvm_home"
        pushd 3rdparty/tvm-ffi
            SETUPTOOLS_SCM_PRETEND_VERSION="$TARGET_TVM_FFI_VERSION" check_exec \
                pip wheel --no-deps . -w "$dist_dir" -v
            local ffi_whl
            ffi_whl=$(find "$dist_dir" -maxdepth 1 -name "apache_tvm_ffi-${TARGET_TVM_FFI_VERSION}-*.whl" -print -quit)
            if [[ -z "$ffi_whl" ]]; then
                echo "ERROR: tvm-ffi wheel $TARGET_TVM_FFI_VERSION was not produced in $dist_dir"
                return 1
            fi
            check_exec pip install "$ffi_whl"
        popd
        rm -rf build
        mkdir build
        pushd build
            check_exec "$cmake_cmd" .. -DUSE_TANG=1 -DUSE_TADNN=0 -DUSE_CUDA=OFF -DUSE_OPENCL=OFF -DUSE_CUTLASS=OFF \
                "-DCMAKE_TANG_COMPILER=${PTCC_PATH}" \
                "-DCMAKE_TANG_FLAGS=--tang-gpu-arch=stcu" \
                "-DTANG_TOOLKIT_ROOT_DIR=${TANGRT_PATH}" \
                "-DTANG_DIR=${TANG_CMAKE_PACKAGE_DIR}" \
                "-DTANGRT_DIR=${TANGRT_CMAKE_PACKAGE_DIR}" \
                -DCMAKE_MODULE_PATH=${CMAKE_PATH} \
                -DUSE_HEXAGON=0
            check_exec "$cmake_cmd" --build . --parallel "$(nproc)"
        popd
    popd
}

# Export the env tilelang's own setup.py needs to find the prebuilt tvm.
ci_set_tilelang_build_env () {
    local tilelang_home="$1"
    export TVM_HOME=$tilelang_home/3rdparty/tvm_sunrise
    export TVM_PREBUILD_PATH=$TVM_HOME/build
    export TVM_SOURCE_DIR=$TVM_HOME
    export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/ffi/python:$PYTHONPATH
    # scikit-build-core starts a separate CMake configure for the TileLang
    # wheel (including its embedded TVM); it does not inherit TVM's cache.
    # Its environment list uses semicolons, preserving spaces inside paths.
    local ptcc_cmake_args="-DCMAKE_TANG_COMPILER=${PTCC_PATH};-DCMAKE_TANG_FLAGS=--tang-gpu-arch=stcu"
    ptcc_cmake_args+=";-DTANG_TOOLKIT_ROOT_DIR=${TANGRT_PATH};-DTANG_DIR=${TANG_CMAKE_PACKAGE_DIR};-DTANGRT_DIR=${TANGRT_CMAKE_PACKAGE_DIR}"
    export SKBUILD_CMAKE_ARGS="${SKBUILD_CMAKE_ARGS:+${SKBUILD_CMAKE_ARGS};}${ptcc_cmake_args}"
}

# Install tilelang + tvm-ffi from prebuilt whls in $1 (dist dir).
ci_install_tilelang_whl () {
    local dist_dir="$1"
    local ffi_whl tl_whl
    ffi_whl=$(ls "$dist_dir"/apache_tvm_ffi-*.whl 2>/dev/null | head -1)
    tl_whl=$(ls "$dist_dir"/tilelang_sunrise-*.whl 2>/dev/null | head -1)
    if [ -z "$ffi_whl" ]; then echo "ERROR: tvm-ffi whl not found in $dist_dir/"; exit 1; fi
    if [ -z "$tl_whl" ]; then echo "ERROR: tilelang whl not found in $dist_dir/"; exit 1; fi
    echo "Found tvm-ffi whl: $ffi_whl"
    echo "Found tilelang whl: $tl_whl"
    check_exec pip install "$ffi_whl"
    check_exec pip install ml-dtypes "z3-solver>=4.13.0,<4.15.5"
    check_exec pip install --no-deps "$tl_whl"
}

# Verify the installed wheel from outside the source checkout.  This catches a
# missing native library or TANG registration without accidentally importing the
# in-tree Python package.
ci_assert_tilelang_tang_registration () {
    local launch_cwd="${CI_TMP_DIR:?ci_init_state must run before wheel verification}"
    (
        cd "$launch_cwd"
        python3 - <<'PY'
import tilelang
import tvm

print(f"installed tilelang: {tilelang.__file__}")
print(f"tvm source: {tvm.__file__}")
registered = tvm.get_global_func("target.build.tilelang_tang", allow_missing=True)
if registered is None:
    raise SystemExit("target.build.tilelang_tang is not registered")
print("target.build.tilelang_tang: registered")
PY
    )
}

# Run one puzzle script body (no timeout). Caller wraps with ci_run_timed.
# Fail on non-zero exit or "Results match: False" in the logfile.
ci_puzzle_postcheck () {
    local logfile="$1" ret="$2"
    if [[ $ret -eq 0 ]] && grep -qE 'Results match:[[:space:]]*False' "$logfile"; then
        echo "Detected failed puzzle result (Results match: False)"
        return 1
    fi
    return "$ret"
}

# Run $cmd under GNU timeout in the background + wait so GitLab cancel (SIGTERM)
# can interrupt promptly. --foreground keeps the test in the job process group;
# --kill-after SIGKILLs if the test ignores SIGTERM (common with GPU/pytest).
# Live-streams via `tail -f --pid` into the job log while also keeping $logfile.
# Optional $4 = launch cwd (e.g. PYTEST_LAUNCH_CWD). Returns the timeout/cmd exit.
ci_run_timed () {
    local case_timeout="$1" logfile="$2" cmd="$3" launch_cwd="${4:-}"
    local timeout_log
    timeout_log=$(mktemp) || return 125
    : > "$logfile" || { rm -f "$timeout_log"; return 125; }
    CI_LAST_RUN_TIMED_OUT=0
    if [[ ${CI_SHUTTING_DOWN:-0} -eq 1 ]]; then
        rm -f "$timeout_log"
        exit 143
    fi
    echo "Execute timeout --foreground --kill-after=${CI_TIMEOUT_KILL_AFTER}s ${case_timeout} ${cmd}"
    (
        [[ -z "$launch_cwd" ]] || cd "$launch_cwd" || exit 125
        # Separate timeout's own diagnostic from child output. A child exiting
        # 124/137/143 is not proof that the wall-clock deadline was reached.
        exec env LC_ALL=C timeout --foreground --verbose --kill-after="${CI_TIMEOUT_KILL_AFTER}s" \
            "$case_timeout" bash -c 'exec >"$2" 2>&1; eval "exec $1"' _ "$cmd" "$logfile"
    ) </dev/null 2>"$timeout_log" &
    CI_TEST_PID=$!
    tail -n +1 -f "$logfile" --pid="$CI_TEST_PID" 2>/dev/null &
    local tail_pid=$! ret=0
    wait "${CI_TEST_PID}" || ret=$?
    CI_TEST_PID=""
    wait "${tail_pid}" 2>/dev/null || true
    if grep -qE '^timeout: sending signal (TERM|KILL) to command' "$timeout_log"; then
        CI_LAST_RUN_TIMED_OUT=1
    fi
    cat "$timeout_log" | tee -a "$logfile"
    rm -f "$timeout_log"
    if [[ ${CI_SHUTTING_DOWN:-0} -eq 1 ]]; then
        exit 143
    fi
    return "$ret"
}

# GNU timeout normally returns 124. Tang/Python teardown may instead surface a
# child status. Signal exits at the limit, or any nonzero exit after the limit,
# are timeouts and require recovery; short-lived signal failures remain FAIL.
ci_result_is_timeout () {
    local ret="$1" elapsed="$2" case_timeout="$3"
    [[ "$ret" -eq 124 ]] && return 0
    if (( elapsed >= case_timeout )) && [[ "$ret" -eq 137 || "$ret" -eq 143 ]]; then
        return 0
    fi
    (( elapsed >= case_timeout ))
}

# Initialize the shared report directory once.  A job may run multiple lists
# (pytest files followed by direct scripts); later suites must not erase the
# evidence produced by earlier suites.
ci_prepare_failure_reports () {
    if [[ "${CI_FAILURE_REPORTS_INITIALIZED:-0}" == "1" ]]; then
        return 0
    fi
    local report_dir; report_dir="$(ci_failure_report_dir)"
    if ! rm -rf "$report_dir" || ! mkdir -p "$report_dir"; then
        echo "ERROR: failed to initialize case-report directory: $report_dir" >&2
        return 1
    fi
    CI_FAILURE_REPORTS_INITIALIZED=1
    export CI_FAILURE_REPORTS_INITIALIZED
}

# Warn (stderr) that a non-comment, non-blank list line matched no accepted form
# and was dropped, and record it in the caller-visible CI_DROPPED_LINES array so a
# preview (ci/run.sh --dry-run) can surface it. Additive: which lines RUN is
# unchanged; existing callers just see extra warnings.
_ci_warn_unparsable () {
    local line="$1" list_file="$2"
    echo "WARNING: ci_parse_list_into: skipping unparsable line in ${list_file}: ${line}" >&2
    CI_DROPPED_LINES+=("$line")
}

# Parse a test-case list file into the caller-visible arrays CI_CMDS / CI_LABELS
# / CI_CASE_MODES (APPENDS; caller resets first). $1=list file, $2=mode
# (command|pytest|puzzle), optional $3=list_root used to resolve relative paths
# (defaults to the list file's directory).
# In pytest mode: bare Python paths run under pytest, "python path.py" entries run
# as direct commands, and a pytest nodeid (path.py::Class::test) is passed through
# to pytest as-is (its file part is resolved against list_root).
# Honors TEST_MARKER and PYTEST_AUDIT_PLUGIN_ARGS.
ci_parse_list_into () {
    local list_file="$1" mode="${2:-pytest}"
    local list_root="${3:-}"
    if [[ ! -f "$list_file" ]]; then
        echo "Error: test list file not found: $list_file"; return 1
    fi
    [[ -z "$list_root" ]] && list_root="$(cd "$(dirname "$list_file")" && pwd)"
    local line
    while IFS= read -r line; do
        [[ $line =~ ^#.*$ ]] && continue
        [[ -z $line ]] && continue
        case "$mode" in
            command)
                [[ $line =~ ^(python|pytest)[[:space:]]+ ]] || { _ci_warn_unparsable "$line" "$list_file"; continue; }
                CI_CMDS+=("$line"); CI_LABELS+=("$(echo "$line" | awk '{print $2}')")
                CI_CASE_MODES+=("command") ;;
            pytest)
                if [[ $line =~ ^python[[:space:]]+.+\.py$ ]]; then
                    CI_CMDS+=("$line")
                    CI_LABELS+=("$(echo "$line" | awk '{print $2}')")
                    CI_CASE_MODES+=("command")
                else
                    # Accept a bare .py path or a pytest nodeid (path.py::...).
                    [[ $line =~ \.py(::|$) ]] || { _ci_warn_unparsable "$line" "$list_file"; continue; }
                    local file_part="${line%%::*}" node_suffix=""
                    [[ "$line" == *::* ]] && node_suffix="::${line#*::}"
                    [[ "$file_part" != /* ]] && file_part="$list_root/$file_part"
                    local case_path="${file_part}${node_suffix}"
                    if [ -n "${TEST_MARKER:-}" ]; then
                        CI_CMDS+=("python -m pytest -v -ra --instafail --import-mode=importlib ${PYTEST_AUDIT_PLUGIN_ARGS:-} -m ${TEST_MARKER} $case_path")
                    elif [ "$line" = "tests/test_base.py" ]; then
                        CI_CMDS+=("python $case_path")
                    else
                        CI_CMDS+=("python -m pytest -v -ra --instafail --import-mode=importlib ${PYTEST_AUDIT_PLUGIN_ARGS:-} $case_path")
                    fi
                    CI_LABELS+=("$line")
                    CI_CASE_MODES+=("pytest")
                fi
                ;;
            puzzle)
                [[ $line =~ \.py$ ]] || { _ci_warn_unparsable "$line" "$list_file"; continue; }
                CI_CMDS+=("python $line"); CI_LABELS+=("$line"); CI_CASE_MODES+=("puzzle") ;;
        esac
    done < "$list_file"
}

# Canonical selector identity shared by list overrides and run.sh's merged plan.
ci_timeout_key () {
    local label="$1" root="$2" file_part="${1%%::*}" suffix=""
    [[ "$label" == *::* ]] && suffix="::${label#*::}"
    [[ "$file_part" == /* ]] || file_part="$root/$file_part"
    local resolved
    resolved=$(realpath -m -- "$file_part") || return 1
    printf '%s%s\n' "$resolved" "$suffix"
}

# Load an optional sibling TSV into the caller's associative CI_CASE_TIMEOUTS.
# Reject duplicate entries, stale selectors, and conflicting merged-list values.
ci_load_test_timeouts () {
    local list_file="$1" mode="${2:-pytest}" timeout_file="${1%.txt}_timeouts.tsv"
    [[ -e "$timeout_file" ]] || return 0
    if [[ ! -f "$timeout_file" || ! -r "$timeout_file" ]]; then
        echo "ERROR: unreadable timeout mapping: $timeout_file" >&2
        return 1
    fi
    local list_root
    list_root=$(cd "$(dirname "$list_file")" && pwd) || return 1
    local -a CI_CMDS=() CI_LABELS=() CI_CASE_MODES=()
    ci_parse_list_into "$list_file" "$mode" "$list_root" || return 1
    local -A available=() seen=()
    local label key line value
    for label in "${CI_LABELS[@]}"; do
        key=$(ci_timeout_key "$label" "$list_root") || return 1
        available["$key"]=1
    done
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ "$line" =~ ^[[:space:]]*# || -z "$line" ]] && continue
        label="${line%%$'\t'*}"
        value="${line#*$'\t'}"
        if [[ "$label" == "$line" || -z "$label" || "$label" =~ [[:space:]] ||
              ! "$value" =~ ^[1-9][0-9]*$ || ${#value} -gt 9 ]]; then
            echo "ERROR: Invalid timeout mapping in $timeout_file: $line" >&2
            return 1
        fi
        key=$(ci_timeout_key "$label" "$list_root") || return 1
        if [[ -n "${seen[$key]:-}" ]]; then
            echo "ERROR: Duplicate timeout mapping for $label in $timeout_file" >&2
            return 1
        fi
        seen["$key"]=1
        if [[ -z "${available[$key]:-}" ]]; then
            echo "ERROR: Timeout mapping does not match a case in $list_file: $label" >&2
            return 1
        fi
        if [[ -n "${CI_CASE_TIMEOUTS[$key]:-}" && "${CI_CASE_TIMEOUTS[$key]}" != "$value" ]]; then
            echo "ERROR: Conflicting timeout mappings for $label in $timeout_file" >&2
            return 1
        fi
        CI_CASE_TIMEOUTS["$key"]="$value"
    done < "$timeout_file"
    return 0
}

# Run/retry/tally the cases already parsed into CI_CMDS / CI_LABELS / CI_CASE_MODES.
# $1=suite (report bucket). Honors CASE_TIMEOUT, CASE_REPEAT, TILELANG_CACHE_DIR,
# PYTEST_LAUNCH_CWD.
# CASE_REPEAT (default 3): per-case attempts until PASS/SKIPPED; the final attempt
# is tallied/reported as case_result, and every attempt is additionally recorded as
# case_attempt. Returns 1 if any case fails.
ci_run_prepared_cases () {
    local suite="$1"
    local list_root="${2:-$PWD}"
    local case_timeout="${CASE_TIMEOUT:-900}"
    local case_repeat="${CASE_REPEAT:-3}"
    local cache_dir="${TILELANG_CACHE_DIR:-$HOME/.tilelang/cache}"

    if ! [[ "$case_timeout" =~ ^[1-9][0-9]*$ ]] || [[ ${#case_timeout} -gt 9 ]]; then
        echo "ERROR: Invalid CASE_TIMEOUT=$case_timeout (need positive integer seconds)" >&2
        return 1
    fi
    if ! [[ $case_repeat =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: Invalid CASE_REPEAT=$case_repeat (need positive integer)"
        return 1
    fi

    local -n cmds=CI_CMDS
    local -n labels=CI_LABELS
    local -n case_modes=CI_CASE_MODES

    if [[ -f "$timeout_file" ]]; then
        local timeout_case timeout_value
        while IFS= read -r line || [[ -n $line ]]; do
            [[ $line =~ ^[[:space:]]*# ]] && continue
            [[ -z $line ]] && continue
            timeout_case="${line%%$'\t'*}"
            timeout_value="${line#*$'\t'}"
            if [[ "$timeout_case" == "$line" || -z "$timeout_case" || \
                  "$timeout_value" == *$'\t'* || ! "$timeout_value" =~ ^[1-9][0-9]*$ ]]; then
                echo "ERROR: Invalid timeout mapping in $timeout_file: $line"
                return 1
            fi
            if [[ -n "${case_timeouts[$timeout_case]+set}" ]]; then
                echo "ERROR: Duplicate timeout mapping for $timeout_case in $timeout_file"
                return 1
            fi
            case_timeouts["$timeout_case"]="$timeout_value"
        done < "$timeout_file"
    fi

    local num=${#cmds[@]}
    echo "Total test cases: $num   (marker: ${TEST_MARKER:-all}, repeat: ${case_repeat})"
    if [[ $num -eq 0 ]]; then
        echo "Error: no valid test cases found"; return 1
    fi

    if (( ${#case_timeouts[@]} > 0 )); then
        local override_case candidate found
        for override_case in "${!case_timeouts[@]}"; do
            found=0
            for candidate in "${labels[@]}"; do
                if [[ "$candidate" == "$override_case" ]]; then
                    found=1
                    break
                fi
            done
            if [[ $found -eq 0 ]]; then
                echo "ERROR: Timeout mapping does not match a case in $list_file: $override_case"
                return 1
            fi
        done
        echo "Per-case timeout mappings: $timeout_file"
    fi

    # Clear stale reports only before this job's first list.  Subsequent lists
    # append separate suite JSONL files into the same artifact directory.
    if ! ci_prepare_failure_reports; then
        return 2
    fi

    local -a results=()
    local success=0 fail=0 skipped=0 i ret start end elapsed line_result
    local attempt case_status reason rec_log current_mode launch_cwd logfile attempt_log current_timeout key
    local recording_failed=0 record_ret device_unusable=0
    for ((i=0; i<num; i++)); do
        echo ">>>>>>> running case $((i+1))/$num: ${labels[$i]} <<<<<<<<"
        key=$(ci_timeout_key "${labels[$i]}" "$list_root") || return 1
        current_timeout="${CI_CASE_TIMEOUTS[$key]:-$case_timeout}"
        echo "Case timeout: ${current_timeout}s"
        if [[ $device_unusable -eq 1 ]]; then
            ci_record_case_result "$suite" "${labels[$i]}" "${cmds[$i]}" NOT_RUN 1 0 \
                "not run: device recovery failed" "" "$current_timeout" || recording_failed=1
            results+=("NOT_RUN: ${labels[$i]} (device recovery failed)")
            fail=$((fail+1))
            continue
        fi
        current_mode="${case_modes[$i]}"
        launch_cwd=""
        if [[ "$current_mode" == "pytest" && -n "${PYTEST_LAUNCH_CWD:-}" ]]; then
            launch_cwd="$PYTEST_LAUNCH_CWD"
        fi

        # Per-case retry: up to CASE_REPEAT attempts until PASS/SKIPPED; tally once.
        case_status="" reason="" ret=1 elapsed=0 rec_log=""
        for (( attempt = 1; attempt <= case_repeat; attempt++ )); do
            needs_timeout_reset=0
            if [[ ${CI_SHUTTING_DOWN:-0} -eq 1 ]]; then
                exit 143
            fi
            rm -rf "$cache_dir"
            echo "$(date) [attempt ${attempt}/${case_repeat}]"
            ci_check_card_state

            logfile=$(mktemp)
            start=$(date +%s)
            # `|| ret=$?` keeps set -e callers from aborting before we classify the result.
            ret=0
            CI_LAST_RUN_TIMED_OUT=0
            ci_run_timed "$current_timeout" "$logfile" "${cmds[$i]}" "$launch_cwd" || ret=$?
            if [[ "$current_mode" == "puzzle" ]]; then
                ci_puzzle_postcheck "$logfile" "$ret" || ret=$?
            fi
            end=$(date +%s); elapsed=$((end - start))

            if [[ ${CI_LAST_RUN_TIMED_OUT:-0} -eq 1 ]]; then
                case_status="TIMEOUT"; reason="timed out after ${current_timeout}s (exit $ret)"; rec_log="$logfile"
            elif [[ "$current_mode" == "pytest" ]] && ci_pytest_is_skipped "$logfile" "$ret"; then
                case_status="SKIPPED"; reason="no tests ran / all skipped"; rec_log=""
            elif [[ $ret -eq 0 ]]; then
                case_status="PASS"; reason=""; rec_log=""
            elif [[ $ret -eq 137 ]]; then
                case_status="FAIL"; reason="SIGKILL (exit 137; possible OOM)"; rec_log="$logfile"
            elif [[ $ret -eq 143 ]]; then
                case_status="FAIL"; reason="SIGTERM (exit 143)"; rec_log="$logfile"
            else
                case_status="FAIL"; reason="exit code $ret"; rec_log="$logfile"
            fi

            # Record this attempt (additive). FAIL/TIMEOUT carry a bounded log tail.
            attempt_log=""
            [[ "$case_status" != "PASS" && "$case_status" != "SKIPPED" ]] && attempt_log="$logfile"
            ci_record_case_attempt "$suite" "${labels[$i]}" "${cmds[$i]}" "$case_status" \
                "$ret" "$elapsed" "$attempt" "$case_repeat" "$reason" "$attempt_log" "$current_timeout" || {
                record_ret=$?
                echo "ERROR: failed to record attempt ${attempt} for ${labels[$i]} (exit $record_ret)" >&2
                recording_failed=1
            }

            if [[ "$case_status" == "TIMEOUT" ]]; then
                ci_reset_gpu_on_timeout "$suite" "${labels[$i]}" "$attempt" || {
                    echo "ERROR: device recovery failed for ${labels[$i]}; stopping GPU execution" >&2
                    device_unusable=1
                }
            fi

            [[ $device_unusable -eq 1 ]] && break
            if [[ $case_status == "PASS" || $case_status == "SKIPPED" ]]; then
                [[ $attempt -gt 1 ]] && echo "passed on attempt ${attempt}/${case_repeat}: ${labels[$i]}"
                # Keep logfile only when recording a failure; PASS/SKIP drop it.
                [[ -z $rec_log ]] && rm -f "$logfile"
                break
            fi
            if [[ $attempt -lt $case_repeat ]]; then
                echo "failed (${case_status}) on attempt ${attempt}/${case_repeat}, retrying: ${labels[$i]}"
                rm -f "$logfile"
                rec_log=""
            else
                echo "failed (${case_status}) after ${case_repeat} attempt(s): ${labels[$i]}"
            fi
        done

        case "$case_status" in
            PASS)
                line_result="SUCCESS: ${labels[$i]} (${elapsed}s)"; success=$((success+1)) ;;
            SKIPPED)
                line_result="SKIPPED: ${labels[$i]} (${elapsed}s)"; skipped=$((skipped+1)) ;;
            TIMEOUT)
                line_result="TIMEOUT: ${labels[$i]} (${elapsed}s)"; fail=$((fail+1)) ;;
            *)
                line_result="FAILURE: ${labels[$i]} (exit $ret, ${elapsed}s)"; fail=$((fail+1)) ;;
        esac
        # Record every case (PASS/SKIPPED carry no log_tail to bound artifact size).
        ci_record_case_result "$suite" "${labels[$i]}" "${cmds[$i]}" "$case_status" "$ret" "$elapsed" "$reason" "$rec_log" "$current_timeout" || {
            record_ret=$?
            echo "ERROR: failed to record final result for ${labels[$i]} (exit $record_ret)" >&2
            recording_failed=1
        }
        [[ -n $rec_log ]] && rm -f "$rec_log"
        results+=("$line_result")
        echo "  -> $line_result"
        echo "---------------------------------------------------------------------"
    done

    echo "============================= Test Results Summary =============================="
    printf '%s\n' "${results[@]}"
    echo "============================= Statistics =============================="
    echo "Total: $num   Success: $success   Failed: $fail   Skipped: $skipped"
    awk "BEGIN {printf \"Success rate: %.1f%%\n\", $success/$num*100}"
    if [[ $fail -gt 0 ]]; then
        return 1
    fi
    [[ $recording_failed -gt 0 ]] && return 2
    return 0
}

# Run a test-case list. $1=list file, $2=mode (command|pytest|puzzle), $3=suite.
# Thin wrapper preserved for downstream callers: reset the arrays, parse the list,
# then run the prepared cases. Net behavior identical to before the parse/run split.
# In pytest mode, bare Python paths use pytest while "python path.py" entries run
# as direct commands.
# Honors TEST_MARKER, CASE_TIMEOUT, CASE_REPEAT, TILELANG_CACHE_DIR.
# CASE_REPEAT (default 3): per-case attempts until PASS/SKIPPED; only the final
# attempt is tallied/reported (no batching).
# Returns 1 if any case fails.
ci_run_test_list () {
    local list_file="$1" mode="${2:-pytest}" suite="${3:-}"
    local case_repeat="${CASE_REPEAT:-3}"
    local -A CI_CASE_TIMEOUTS=()

    if ! [[ $case_repeat =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: Invalid CASE_REPEAT=$case_repeat (need positive integer)"
        return 1
    fi

    # Operator repos call with 2 args; derive the suite from the list filename:
    #   ci_test_case_list_tileops.txt -> tileops ; ..._tilelang_puzzles.txt -> tilelang_puzzles
    if [[ -z "$suite" ]]; then
        suite="$(basename "$list_file")"; suite="${suite#ci_test_case_list_}"; suite="${suite%.txt}"
    fi

    if [[ ! -f "$list_file" ]]; then
        echo "Error: test list file not found: $list_file"; return 1
    fi

    ci_assert_runtime_stack

    CI_CMDS=(); CI_LABELS=(); CI_CASE_MODES=()
    ci_parse_list_into "$list_file" "$mode" || return 1
    ci_load_test_timeouts "$list_file" "$mode" || return 1
    ci_run_prepared_cases "$suite" "$(cd "$(dirname "$list_file")" && pwd)"
}
