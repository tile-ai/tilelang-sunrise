#!/bin/bash

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

# GitLab keeps the existing password-backed reset path.  The public GitHub
# runner selects sudo-n and only receives one exact NOPASSWD pt_smi command.
TILELANG_CI_RESET_MODE="${TILELANG_CI_RESET_MODE:-password}"
TILELANG_CI_PUBLIC_LOGS="${TILELANG_CI_PUBLIC_LOGS:-0}"
export TILELANG_CI_RESET_MODE TILELANG_CI_PUBLIC_LOGS

# -------------------- Host toolchain paths (machine-specific) --------------------
LLVM_HOME="${LLVM_HOME:-}"
LLVM_VERSION_MAJOR="${LLVM_VERSION_MAJOR:-20}"
LLVM_VERSION_MINOR="${LLVM_VERSION_MINOR:-0}"
TANGRT_PATH="${TANGRT_PATH:-/usr/local/tangrt/}"
STPU_TANGRT_PATH="${STPU_TANGRT_PATH:-/usr/local/tangrt}"
TANGRT_LIB_PATH="${TANGRT_LIB_PATH:-/usr/local/tangrt/lib/linux-x86_64:/usr/lib64}"
VENDOR_INCLUDE_DIRS="${VENDOR_INCLUDE_DIRS:-/usr/local/tangrt/include}"
PTCC_PATH="${PTCC_PATH:-/usr/local/tangrt/toolchains/llvm/prebuilt/linux-x86_64/bin/ptcc}"
CMAKE_PATH="${CMAKE_PATH:-/usr/local/tangrt/cmake}"
CMAKE_ROOT="${CMAKE_ROOT:-/usr/local/bin/cmake}"
TANG_CMAKE_PACKAGE_DIR="${TANG_CMAKE_PACKAGE_DIR:-${TANGRT_PATH}/targets/linux-x86_64/lib/cmake/TANG}"
TANGRT_CMAKE_PACKAGE_DIR="${TANGRT_CMAKE_PACKAGE_DIR:-${TANGRT_PATH}/targets/linux-x86_64/lib/cmake/TANGRT}"

# -------------------- Helpers --------------------
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
            echo "${content}"
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
        echo "${content}"
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
    local timeout_seconds="$9" attempt="${10:-}" max_attempts="${11:-}" will_retry="${12:-0}"
    local report_dir; report_dir="$(ci_failure_report_dir)"
    mkdir -p "$report_dir"
    local tail_chars="${CI_CASE_LOG_TAIL_CHARS:-6000}"
    local tail=""
    [[ -n "$logfile" && -f "$logfile" ]] && tail="$(tail -c "$tail_chars" "$logfile")"
    python3 - "$report_dir/${suite}.jsonl" "$suite" "$case_id" "$command" "$status" \
              "$exit_code" "$elapsed" "$reason" "$tail" "$timeout_seconds" "$attempt" "$max_attempts" "$will_retry" <<'PY'
import json, os, sys
out, suite, case_id, command, status, exit_code, elapsed, reason, tail, timeout_seconds, attempt, max_attempts, will_retry = sys.argv[1:14]
rec = {
    "schema_version": 2, "record_kind": "case_attempt" if attempt else "case_result",
    "suite": suite, "job": os.getenv("CI_JOB_NAME", ""), "case": case_id,
    "command": command, "cwd": os.getcwd(), "status": status,
    "exit_code": int(exit_code), "elapsed_seconds": int(elapsed),
    "timeout_seconds": int(timeout_seconds),
    "failure_reason": reason, "log_tail": tail,
    "pipeline_id": os.getenv("CI_PIPELINE_ID", ""), "job_id": os.getenv("CI_JOB_ID", ""),
    "job_name": os.getenv("CI_JOB_NAME", ""), "commit_sha": os.getenv("CI_COMMIT_SHA", ""),
    "project_id": os.getenv("CI_PROJECT_ID", ""), "mr_iid": os.getenv("CI_MERGE_REQUEST_IID", ""),
}
if attempt:
    rec.update({
        "attempt": int(attempt),
        "max_attempts": int(max_attempts),
        "will_retry": will_retry == "1",
    })
with open(out, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
PY
}

ci_record_device_recovery () {
    local suite="$1" case_id="$2" attempt="$3" device="$4" status="$5" exit_code="$6" reason="$7"
    local report_dir; report_dir="$(ci_failure_report_dir)"
    mkdir -p "$report_dir"
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
# otherwise cascade-fail every later case. Best-effort: never aborts the run.
# Bound sudo/pt_smi so a stuck reset cannot block GitLab cancel forever.
ci_reset_gpu_on_timeout () {
    local dev="${TANG_VISIBLE_DEVICES:-0}"
    local mode="${TILELANG_CI_RESET_MODE:-password}"
    local suite="${1:-unknown}" case_id="${2:-unknown}" attempt="${3:-0}"
    local ret=0
    if [[ ${CI_SHUTTING_DOWN:-0} -eq 1 ]]; then
        echo "ci_reset_gpu_on_timeout: skip (job cancelling)"
        ci_record_device_recovery "$suite" "$case_id" "$attempt" "$dev" SKIPPED 0 "job cancelling" || true
        return 0
    fi
    echo "Resetting Tang device $dev after timeout ..."
    case "$mode" in
        sudo-n)
            if [[ "$dev" != "0" ]]; then
                echo "WARNING: public runner reset is restricted to Tang device 0 (got $dev)"
                ci_record_device_recovery "$suite" "$case_id" "$attempt" "$dev" SKIPPED 0 \
                    "public reset is restricted to device 0" || true
                return 0
            fi
            /usr/bin/timeout --foreground --kill-after=10s 60 \
                /usr/bin/sudo -n /usr/bin/pt_smi -r -i 0 || ret=$?
            ;;
        password)
            if ! command -v pt_smi >/dev/null 2>&1; then
                echo "WARNING: pt_smi is unavailable; cannot reset timed-out Tang device"
                ci_record_device_recovery "$suite" "$case_id" "$attempt" "$dev" SKIPPED 0 \
                    "pt_smi is unavailable" || true
                return 0
            fi
            if [[ -z "${SUDO_MAGICWORD:-}" ]]; then
                echo "WARNING: SUDO_MAGICWORD is unavailable; cannot reset timed-out Tang device"
                ci_record_device_recovery "$suite" "$case_id" "$attempt" "$dev" SKIPPED 0 \
                    "reset credential is unavailable" || true
                return 0
            fi
            printf '%s\n' "$SUDO_MAGICWORD" | sudo -S -p '' \
                /usr/bin/timeout --foreground --kill-after=10s 60 pt_smi -r -i "$dev" || ret=$?
            ;;
        disabled)
            echo "WARNING: Tang reset is disabled; runner operator intervention may be required"
            ci_record_device_recovery "$suite" "$case_id" "$attempt" "$dev" SKIPPED 0 \
                "reset mode is disabled" || true
            return 0
            ;;
        *)
            echo "WARNING: unsupported TILELANG_CI_RESET_MODE=$mode; Tang device was not reset"
            ci_record_device_recovery "$suite" "$case_id" "$attempt" "$dev" SKIPPED 0 \
                "unsupported reset mode" || true
            return 0
            ;;
    esac
    if [[ $ret -ne 0 ]]; then
        echo "WARNING: Tang reset failed (exit=${ret}); runner operator intervention may be required"
        ci_record_device_recovery "$suite" "$case_id" "$attempt" "$dev" FAIL "$ret" \
            "pt_smi reset failed" || true
        ci_dump_device_status
        return 0
    fi
    echo "ci_reset_gpu_on_timeout: pt_smi -r -i ${dev} OK"
    ci_record_device_recovery "$suite" "$case_id" "$attempt" "$dev" PASS 0 "" || true
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
    local exit_code=$?
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
        mkdir -p "$dest"
        python3 - "$dest/sunrise_device_summary.json" "$dev" "$state_val" "$usage_val" "$fatal_val" <<'PY'
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
        return 0
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
            check_exec cmake .. -DUSE_TANG=1 -DUSE_TADNN=0 \
                -DUSE_CUDA=OFF \
                -DUSE_OPENCL=OFF \
                -DUSE_CUTLASS=OFF \
                -DCMAKE_TANG_COMPILER=${PTCC_PATH} \
                -DTANG_TOOLKIT_ROOT_DIR=${TANGRT_PATH} \
                -DTANG_DIR=${TANG_CMAKE_PACKAGE_DIR} \
                -DTANGRT_DIR=${TANGRT_CMAKE_PACKAGE_DIR} \
                -DCMAKE_MODULE_PATH=${CMAKE_PATH} \
                -DUSE_HEXAGON=0
            check_exec cmake --build . --parallel "$(nproc)"
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
    if [[ ${CI_SHUTTING_DOWN:-0} -eq 1 ]]; then
        exit 143
    fi
    echo "Execute timeout --foreground --kill-after=${CI_TIMEOUT_KILL_AFTER}s ${case_timeout} ${cmd}"
    if [[ -n "$launch_cwd" ]]; then
        # exec so CI_TEST_PID is timeout itself (not a wrapping subshell).
        (cd "$launch_cwd" && eval "exec timeout --foreground --kill-after=${CI_TIMEOUT_KILL_AFTER}s ${case_timeout} ${cmd}") \
            </dev/null >"$logfile" 2>&1 &
    else
        eval "timeout --foreground --kill-after=${CI_TIMEOUT_KILL_AFTER}s ${case_timeout} ${cmd}" \
            </dev/null >"$logfile" 2>&1 &
    fi
    CI_TEST_PID=$!
    # Live-stream while the test runs; --pid makes tail exit when the child dies.
    tail -n +1 -f "$logfile" --pid="$CI_TEST_PID" 2>/dev/null &
    local tail_pid=$!
    wait "${CI_TEST_PID}"
    local ret=$?
    CI_TEST_PID=""
    wait "${tail_pid}" 2>/dev/null || true
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
    rm -rf "$report_dir"
    mkdir -p "$report_dir"
    CI_FAILURE_REPORTS_INITIALIZED=1
    export CI_FAILURE_REPORTS_INITIALIZED
}

# Run a test-case list. $1=list file, $2=mode (command|pytest|puzzle).
# In pytest mode, bare Python paths use pytest while "python path.py" entries
# run as direct commands.
# Honors TEST_MARKER, CASE_TIMEOUT, CASE_REPEAT, TILELANG_CACHE_DIR. An optional
# sibling <list-name>_timeouts.tsv maps an exact case label to a larger/smaller
# timeout in seconds. Unknown, duplicate, or malformed mappings fail closed.
# CASE_REPEAT (default 3): per-case attempts until PASS/SKIPPED. Every attempt
# is reported; only the final case outcome is tallied (no batching).
# Returns 1 if any case fails.
ci_run_test_list () {
    local list_file="$1" mode="${2:-pytest}" suite="${3:-}"
    local case_timeout="${CASE_TIMEOUT:-900}"
    local case_repeat="${CASE_REPEAT:-3}"
    local cache_dir="${TILELANG_CACHE_DIR:-$HOME/.tilelang/cache}"
    local list_root; list_root="$(cd "$(dirname "$list_file")" && pwd)"
    local timeout_file="${CASE_TIMEOUT_FILE:-${list_file%.txt}_timeouts.tsv}"

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

    local -a cmds=() labels=() case_modes=()
    local -A case_timeouts=()
    local line
    while IFS= read -r line; do
        [[ $line =~ ^#.*$ ]] && continue
        [[ -z $line ]] && continue
        case "$mode" in
            command)
                [[ $line =~ ^(python|pytest)[[:space:]]+ ]] || continue
                cmds+=("$line"); labels+=("$(echo "$line" | awk '{print $2}')")
                case_modes+=("command") ;;
            pytest)
                if [[ $line =~ ^python[[:space:]]+.+\.py$ ]]; then
                    cmds+=("$line")
                    labels+=("$(echo "$line" | awk '{print $2}')")
                    case_modes+=("command")
                else
                    [[ $line =~ \.py$ ]] || continue
                    local case_path="$line"
                    [[ "$case_path" != /* ]] && case_path="$list_root/$case_path"
                    if [ -n "${TEST_MARKER}" ]; then
                        cmds+=("python -m pytest -v -ra --instafail --import-mode=importlib ${PYTEST_AUDIT_PLUGIN_ARGS:-} -m ${TEST_MARKER} $case_path")
                    elif [ "$line" = "tests/test_base.py" ]; then
                        cmds+=("python $case_path")
                    else
                        cmds+=("python -m pytest -v -ra --instafail --import-mode=importlib ${PYTEST_AUDIT_PLUGIN_ARGS:-} $case_path")
                    fi
                    labels+=("$line")
                    case_modes+=("pytest")
                fi
                ;;
            puzzle)
                [[ $line =~ \.py$ ]] || continue
                cmds+=("python $line"); labels+=("$line"); case_modes+=("puzzle") ;;
        esac
    done < "$list_file"

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
    echo "Total test cases: $num   (mode: $mode, marker: ${TEST_MARKER:-all}, repeat: ${case_repeat})"
    if [[ $num -eq 0 ]]; then
        echo "Error: no valid test cases found in $list_file"; return 1
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
    ci_prepare_failure_reports

    local -a results=()
    local success=0 fail=0 skipped=0 i ret start end elapsed line_result
    local attempt case_status reason rec_log current_mode launch_cwd logfile current_timeout
    local needs_timeout_reset will_retry
    for ((i=0; i<num; i++)); do
        echo ">>>>>>> running case $((i+1))/$num: ${labels[$i]} <<<<<<<<"
        current_timeout="${case_timeouts[${labels[$i]}]:-$case_timeout}"
        echo "Case timeout: ${current_timeout}s"
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
            ci_run_timed "$current_timeout" "$logfile" "${cmds[$i]}" "$launch_cwd" || ret=$?
            if [[ "$current_mode" == "puzzle" ]]; then
                ci_puzzle_postcheck "$logfile" "$ret" || ret=$?
            fi
            end=$(date +%s); elapsed=$((end - start))

            if [[ "$current_mode" == "pytest" ]] && ci_pytest_is_skipped "$logfile" "$ret"; then
                case_status="SKIPPED"; reason="no tests ran / all skipped"; rec_log=""
            elif [[ $ret -eq 0 ]]; then
                case_status="PASS"; reason=""; rec_log=""
            elif ci_result_is_timeout "$ret" "$elapsed" "$current_timeout"; then
                case_status="TIMEOUT"; reason="timed out after ${current_timeout}s (exit ${ret})"; rec_log="$logfile"
                needs_timeout_reset=1
            elif [[ $ret -eq 137 ]]; then
                case_status="FAIL"; reason="terminated by SIGKILL (exit 137)"; rec_log="$logfile"
            else
                case_status="FAIL"; reason="exit code $ret"; rec_log="$logfile"
            fi

            will_retry=0
            if [[ $case_status != "PASS" && $case_status != "SKIPPED" && $attempt -lt $case_repeat ]]; then
                will_retry=1
            fi
            ci_record_case_result "$suite" "${labels[$i]}" "${cmds[$i]}" "$case_status" \
                "$ret" "$elapsed" "$reason" "$rec_log" "$current_timeout" "$attempt" "$case_repeat" "$will_retry"
            if [[ $needs_timeout_reset -eq 1 ]]; then
                ci_reset_gpu_on_timeout "$suite" "${labels[$i]}" "$attempt"
            fi

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
        ci_record_case_result "$suite" "${labels[$i]}" "${cmds[$i]}" "$case_status" \
            "$ret" "$elapsed" "$reason" "$rec_log" "$current_timeout"
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
    [[ $fail -gt 0 ]] && return 1 || return 0
}
