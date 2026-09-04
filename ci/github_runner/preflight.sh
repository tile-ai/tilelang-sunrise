#!/usr/bin/env bash
# Run the exact public CI contract before this host is registered with GitHub.
set -Eeuo pipefail

: "${SOURCE_DIR:?SOURCE_DIR must be provided by the preflight service}"
: "${SOURCE_SHA:?SOURCE_SHA must be provided by the preflight service}"
: "${BASE_SHA:?BASE_SHA must be provided by the preflight service}"
: "${TILELANG_EVIDENCE_ROOT:?TILELANG_EVIDENCE_ROOT must be set}"
: "${TARGET_TORCH_PTPU_PKG:?TARGET_TORCH_PTPU_PKG must be set}"
: "${TARGET_TRITON_PKG:?TARGET_TRITON_PKG must be set}"
: "${CONDA_EXE:?CONDA_EXE must be set}"

if [[ "$(id -un)" != "tilelang-gh-runner" ]]; then
    echo "preflight must run as tilelang-gh-runner" >&2
    exit 2
fi
if [[ "${TANG_VISIBLE_DEVICES:-}" != "0" ]]; then
    echo "preflight requires TANG_VISIBLE_DEVICES=0" >&2
    exit 2
fi
if [[ -n "${LLVM_HOME:-}" ]]; then
    echo "preflight intentionally validates the TANG build with LLVM_HOME unset" >&2
    exit 2
fi

cd "$SOURCE_DIR"
actual_sha="$(git rev-parse HEAD)"
if [[ "$actual_sha" != "$SOURCE_SHA" ]]; then
    echo "source SHA mismatch: expected $SOURCE_SHA, got $actual_sha" >&2
    exit 2
fi
git cat-file -e "${BASE_SHA}^{commit}"
if [[ -n "$(git status --porcelain)" ]]; then
    echo "preflight source tree is not clean" >&2
    git status --short >&2
    exit 2
fi

evidence_dir="$TILELANG_EVIDENCE_ROOT/$SOURCE_SHA"
mkdir -p "$evidence_dir"
log_path="$evidence_dir/preflight.log"
summary_path="$evidence_dir/jobs.tsv"
success_path="$evidence_dir/PREFLIGHT_SUCCESS"
rm -f "$success_path"
exec > >(tee -a "$log_path") 2>&1

cleanup_source_device_summary () {
    local preserve="${1:-0}"
    if [[ -f "$SOURCE_DIR/sunrise_device_summary.json" ]]; then
        if [[ "$preserve" == "1" ]]; then
            cp -f "$SOURCE_DIR/sunrise_device_summary.json" "$evidence_dir/failure_device_summary.json"
        fi
        rm -f "$SOURCE_DIR/sunrise_device_summary.json"
    fi
}

preserve_failure_evidence () {
    local exit_code=$?
    trap - EXIT
    cleanup_source_device_summary 1
    exit "$exit_code"
}
trap preserve_failure_evidence EXIT

printf 'job\tstatus\texit_code\telapsed_seconds\n' > "$summary_path"
export CI_PROJECT_DIR="$SOURCE_DIR"
export CI_PROJECT_ID="tile-ai/tilelang-sunrise"
export CI_PIPELINE_ID="preflight-${SOURCE_SHA:0:12}"
export CI_COMMIT_SHA="$SOURCE_SHA"
export CI_COMMIT_BEFORE_SHA="$BASE_SHA"
export CI_MERGE_REQUEST_DIFF_BASE_SHA="$BASE_SHA"
export CI_DEFAULT_BRANCH="main"
export CI_PIPELINE_SOURCE="preflight"
export CI_FAILURE_REPORT_ROOT="$evidence_dir/ci_failure_reports"
export CASE_REPEAT="3"
export TILELANG_CI_RESET_MODE="sudo-n"
export TILELANG_CI_PUBLIC_LOGS="1"
export TILELANG_DEFAULT_TARGET="tang"

record_job () {
    local name="$1" status="$2" exit_code="$3" elapsed="$4"
    printf '%s\t%s\t%s\t%s\n' "$name" "$status" "$exit_code" "$elapsed" >> "$summary_path"
}

run_job () {
    local name="$1"
    shift
    local start end elapsed ret=0
    start=$(date +%s)
    export CI_JOB_NAME="$name"
    export CI_JOB_ID="preflight-${SOURCE_SHA:0:12}-${name}"
    export CI_FAILURE_REPORT_DIR="$CI_FAILURE_REPORT_ROOT/$name"
    echo "================ preflight job: $name ================"
    # Calling a shell function as the left side of `||` disables errexit inside
    # that function.  Run it in an independently strict subshell so an early
    # failed contract check cannot be masked by a later successful command.
    set +e
    (set -Eeuo pipefail; "$@")
    ret=$?
    set -e
    end=$(date +%s)
    elapsed=$((end - start))
    if [[ $ret -eq 0 ]]; then
        record_job "$name" PASS 0 "$elapsed"
        echo "preflight job $name: PASS (${elapsed}s)"
        return 0
    fi
    record_job "$name" FAIL "$ret" "$elapsed"
    echo "preflight job $name: FAIL exit=$ret (${elapsed}s)" >&2
    return "$ret"
}

expect_proxy_success () {
    local url="$1"
    curl --silent --show-error --location --max-time 30 --output /dev/null "$url"
}

expect_direct_failure () {
    local url="$1"
    if curl --noproxy '*' --insecure --silent --location \
            --connect-timeout 5 --max-time 10 --output /dev/null "$url"; then
        echo "network isolation failure: direct request unexpectedly succeeded: $url" >&2
        return 1
    fi
}

expect_tcp_blocked () {
    local host="$1" port="$2"
    if timeout --foreground 5 bash -c 'exec 3<>"/dev/tcp/${1}/${2}"' -- "$host" "$port" \
            2>/dev/null; then
        echo "network isolation failure: direct TCP connection unexpectedly succeeded: ${host}:${port}" >&2
        return 1
    fi
}

check_network_policy () {
    local url code
    for url in \
        https://github.com/ \
        https://api.github.com/ \
        https://results-receiver.actions.githubusercontent.com/ \
        https://objects.githubusercontent.com/ \
        https://conda.anaconda.org/ \
        https://pypi.org/ \
        https://download.pytorch.org/; do
        expect_proxy_success "$url"
    done

    expect_direct_failure https://github.com/
    expect_direct_failure http://169.254.169.254/
    # These services already listen on the host.  Connection-level probes prove
    # that the runner UID cannot reach SSH, RPC, or SMB through loopback.
    expect_tcp_blocked 127.0.0.1 22
    expect_tcp_blocked 127.0.0.1 111
    expect_tcp_blocked 127.0.0.1 139
    expect_tcp_blocked 127.0.0.1 445

    code=$(curl --proxy "${HTTPS_PROXY:?}" --insecure --silent --output /dev/null \
        --write-out '%{http_connect}' --connect-timeout 5 --max-time 10 https://10.0.0.1/ || true)
    if [[ "$code" != "403" ]]; then
        echo "proxy private-address policy returned CONNECT ${code:-<none>}, expected 403" >&2
        return 1
    fi
}

check_host_contract () {
    local no_new_privs ptcc_version_output
    /usr/bin/python3 - <<'PY'
import os
import socket
import stat
import sys

reachable = []
socket_types = [socket.SOCK_STREAM, socket.SOCK_DGRAM]
if hasattr(socket, "SOCK_SEQPACKET"):
    socket_types.append(socket.SOCK_SEQPACKET)

for root, _, files in os.walk("/run"):
    for name in files:
        path = os.path.join(root, name)
        try:
            if not stat.S_ISSOCK(os.stat(path, follow_symlinks=False).st_mode):
                continue
        except OSError:
            continue
        for socket_type in socket_types:
            candidate = socket.socket(socket.AF_UNIX, socket_type)
            candidate.settimeout(1)
            try:
                candidate.connect(path)
            except OSError:
                pass
            else:
                reachable.append(path)
                break
            finally:
                candidate.close()

if reachable:
    print("host contract found reachable host service sockets:", file=sys.stderr)
    print("\n".join(sorted(set(reachable))), file=sys.stderr)
    sys.exit(1)
PY
    [[ -c /dev/ptpu0 ]]
    [[ -c /dev/ptpuctrl ]]
    [[ -x /usr/bin/pt_smi ]]
    [[ "$PTCC_PATH" == "/usr/local/tangrt/toolchains/llvm/prebuilt/linux-x86_64/bin/ptcc" ]]
    [[ -x "$PTCC_PATH" ]]
    echo "762879026fa89dd3b5dd6b48aebc2f7abba239180187a603899d3a8b8335ebc2  $PTCC_PATH" | sha256sum -c -
    ptcc_version_output=$("$PTCC_PATH" --version 2>&1)
    grep -Fxq "ptcc version 2.2.9 (ceb3571d0)" <<< "$ptcc_version_output"
    [[ -x "$CONDA_EXE" ]]
    "$CONDA_EXE" --version
    echo "82a9af8d019e59565761946a9b810880764a4ca416287ef8095b6eeaf36bc463  $TARGET_TORCH_PTPU_PKG" | sha256sum -c -
    echo "c010c33d294061d1774f058787ace3d5df4d9a0da3210e44241ae52250b86ce6  $TARGET_TRITON_PKG" | sha256sum -c -
    no_new_privs="$(awk '$1 == "NoNewPrivs:" { print $2 }' /proc/self/status)"
    if [[ "$no_new_privs" != "0" ]]; then
        echo "host contract requires NoNewPrivs=0 for the exact sudo reset command" >&2
        return 1
    fi
    /usr/bin/sudo -n -l | grep -F '/usr/bin/pt_smi -r -i 0' || return 1
    source ci/lib.sh
    ci_check_card_state || return 1
}

run_contract_tests () {
    bash -n \
        ci/build.sh \
        ci/github_actions.sh \
        ci/github_runner/manage.sh \
        ci/github_runner/preflight.sh \
        ci/lib.sh \
        ci/lint.sh \
        ci/test.sh \
        ci/validate_operator.sh
    /usr/bin/python3 -m pytest -q testing/python/test_sunrise_ci_config.py
}

run_tilekernels () {
    OP=tilekernels_sunrise bash ci/validate_operator.sh
}

run_tileops () {
    OP=tileops_sunrise bash ci/validate_operator.sh
}

echo "source_sha=$SOURCE_SHA"
echo "base_sha=$BASE_SHA"
echo "source_dir=$SOURCE_DIR"
echo "python=$(command -v python3)"
echo "ptcc=$PTCC_PATH"
echo "runner_user=$(id)"

run_job host_contract check_host_contract
run_job network_policy check_network_policy
run_job ci_contract run_contract_tests
run_job lint_changed bash ci/lint.sh changed
run_job lint_all bash ci/lint.sh all
run_job tilelang_build bash ci/build.sh

# Build is a prerequisite for all three hardware validation jobs, but one
# validation failure must not hide the outcome of the remaining suites.  Keep
# using the same card serially, collect every result in jobs.tsv, and fail the
# preflight only after TileLang, TileKernels, and TileOps have all run.
validation_failed=0
run_job tilelang_test bash ci/test.sh || validation_failed=1
run_job tilekernels_sunrise run_tilekernels || validation_failed=1
run_job tileops_sunrise run_tileops || validation_failed=1

source ci/lib.sh
ci_check_card_state
CI_PROJECT_DIR="$evidence_dir" ci_save_device_logs
mv "$evidence_dir/sunrise_device_summary.json" "$evidence_dir/final_device_summary.json"
cleanup_source_device_summary "$validation_failed"
git status --short > "$evidence_dir/final_git_status.txt"
if [[ -n "$(cat "$evidence_dir/final_git_status.txt")" ]]; then
    echo "preflight changed the source tree; inspect final_git_status.txt" >&2
    exit 1
fi
if (( validation_failed != 0 )); then
    echo "One or more hardware validation jobs failed; inspect jobs.tsv and the structured reports" >&2
    exit 1
fi
printf 'SOURCE_SHA=%s\n' "$SOURCE_SHA" > "$success_path"
echo "All six logical CI jobs passed for $SOURCE_SHA"
trap - EXIT
