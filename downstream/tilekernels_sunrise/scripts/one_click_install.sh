#!/usr/bin/env bash
# ============================================================
# TileKernels Full Install Script (optimized)
# ============================================================
# Phases:
#   0. Ensure conda is available
#   1. (optional) Clean caches
#   2. Create / activate conda env, install python deps
#   3. Clone & build tilelang (with bundled TVM and tvm-ffi)
#   4. Install TileKernels (editable, with [dev] extras)
#   5. Run tests
#
# Toggle envs (all default to "0"/unset = run normally):
#   SKIP_CONDA_INSTALL=1     skip miniconda bootstrap
#   SKIP_DEPS_INSTALL=1      skip pip install -r requirements.txt
#   SKIP_TVM_BUILD=1         reuse existing TVM build dir
#   SKIP_TILELANG_CLONE=1    reuse existing tilelang checkout
#   SKIP_TILELANG_INSTALL=1  skip `pip install -e .` for tilelang
#   SKIP_TILEKERNELS_INSTALL=1
#   SKIP_PRE_COMMIT=1        skip installing pre-commit hooks
#   SKIP_TESTS=1             skip pytest phase
#   CLEAN_CACHE=1            run `conda clean` and `pip cache purge`
#   MAX_JOBS=N               parallelism for cmake build (default: nproc)
#   USE_CCACHE=1             enable ccache if available
#   TEST_TARGETS="..."       pytest targets (default: tests/moe/test_topk_gate.py)
#   PYTEST_ARGS="..."        extra pytest args (default: --run-benchmark)
#   SKIP_BENCHMARK=1         skip phase 6 (performance tests)
#   BENCHMARK_TARGETS="..."  benchmark pytest targets (default: tests)
#   BENCHMARK_PYTEST_ARGS="..."  extra pytest args for benchmark phase
#   BENCHMARK_OUTPUT=path    JSONL output (default: benchmark_results_<ts>.jsonl)
#   BENCHMARK_REGRESSION_THRESHOLD=0.1   regression fail threshold (10%)
#   BENCHMARK_VERBOSE=1      pass --benchmark-verbose
#   BENCHMARK_ALLOW_FAILURE=1 do not fail script on benchmark regressions
#   SKIP_STRESS=1            skip phase 7 (stress tests)
#   STRESS_TARGETS="..."     stress pytest targets (default: tests)
#   STRESS_COUNT=N           repeat each test N times (default: 50)
#   STRESS_PARALLEL=N|auto   xdist workers (default: auto)
#   STRESS_TIMEOUT=SECONDS   wall-clock cap (default: 3600)
#   STRESS_PYTEST_ARGS="..." extra pytest args
#   STRESS_LOG=path.log      tee output to log file
#   STRESS_ALLOW_FAILURE=1   do not fail script on stress failures
# ============================================================

set -Eeuo pipefail

# -------------------- Logging & error handling --------------------
log() {
    local level="${1:-INFO}"; shift || true
    printf '[%(%Y-%m-%d %H:%M:%S)T] [%s] %s\n' -1 "${level}" "$*" 2>/dev/null \
        || echo "[$(date '+%F %T')] [${level}] $*"
}

on_error() {
    local exit_code=$?
    local line_no=${1:-?}
    log ERROR "Script failed at line ${line_no} (exit=${exit_code})"
    log ERROR "Last command: ${BASH_COMMAND}"
    exit "${exit_code}"
}
trap 'on_error $LINENO' ERR

# -------------------- Configuration --------------------
CONDA_ENV=${CONDA_ENV:-tilekernels_dev}
TILEKERNELS_HOME=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)
log INFO "TILEKERNELS_HOME: ${TILEKERNELS_HOME}"
TILELANG_SOURCE_DIR=${TILELANG_SOURCE_DIR:-$(cd "${TILEKERNELS_HOME}/../.." && pwd)}
TILELANG_HOME="${TILELANG_SOURCE_DIR}"

PYTHON_VERSION=${PYTHON_VERSION:-3.10}
MAX_JOBS=${MAX_JOBS:-$(nproc)}

# Tang / LLVM toolchain
TANGRT_LIB_PATH="/usr/local/tangrt/lib/linux-x86_64:/usr/lib64"
: "${LLVM_HOME:?Set LLVM_HOME to an LLVM installation compatible with the TANG toolchain}"
export LLVM_HOME
export LLVM_VERSION_MAJOR=${LLVM_VERSION_MAJOR:-20}
export LLVM_VERSION_MINOR=${LLVM_VERSION_MINOR:-0}
export PTCC_PATH=${PTCC_PATH:-/usr/local/tangrt/toolchains/llvm/prebuilt/linux-x86_64/bin/ptcc}
export TANGRT_PATH=${TANGRT_PATH:-/usr/local/tangrt/}
export CMAKE_PATH=${CMAKE_PATH:-/usr/local/tangrt/cmake}
export STPU_TANGRT_PATH=${STPU_TANGRT_PATH:-/usr/local/tangrt}
export VENDOR_INCLUDE_DIRS=${VENDOR_INCLUDE_DIRS:-/usr/local/tangrt/include}
export LD_LIBRARY_PATH="${TANGRT_LIB_PATH}:/usr/local/tangrt/targets/linux-x86_64/lib:${LD_LIBRARY_PATH:-}"

is_true() { [[ "${1:-0}" =~ ^(1|true|TRUE|yes|YES)$ ]]; }

# ============================================================
# Phase 0: Conda
# ============================================================
phase_conda() {
    log INFO "=========== Phase 0: Conda Installer Download ==========="
    if is_true "${SKIP_CONDA_INSTALL:-0}"; then
        log INFO "SKIP_CONDA_INSTALL=1, skipping miniconda bootstrap"
    elif ! command -v conda &>/dev/null && [ ! -x "${HOME}/miniconda3/bin/conda" ]; then
        log INFO "Conda not found. Installing Miniconda3..."
        local tmp; tmp=$(mktemp -d)
        trap 'rm -rf "${tmp}"' RETURN
        wget -q https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh \
            -O "${tmp}/miniconda.sh"
        bash "${tmp}/miniconda.sh" -b -p "${HOME}/miniconda3"
    fi

    export PATH="${HOME}/miniconda3/bin:${PATH}"
    local conda_base
    if command -v conda &>/dev/null; then
        conda_base=$(conda info --base)
    else
        conda_base=$("${HOME}/miniconda3/bin/conda" info --base)
    fi
    # shellcheck disable=SC1091
    source "${conda_base}/etc/profile.d/conda.sh"
    log OK "conda available: $(conda --version)"
}

# ============================================================
# Phase 1: Optional cache clean
# ============================================================
phase_clean_cache() {
    if is_true "${CLEAN_CACHE:-0}"; then
        log INFO "Cleaning conda + pip caches..."
        conda clean --all -y -q || true
        pip cache purge || true
    else
        log INFO "Phase 1: cache clean skipped (set CLEAN_CACHE=1 to enable)"
    fi
}

# ============================================================
# Phase 2: Conda env + python deps
# ============================================================
phase_env() {
    log INFO "=========== Phase 2: Conda env '${CONDA_ENV}' ==========="

    if conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -qx "${CONDA_ENV}"; then
        log INFO "env '${CONDA_ENV}' already exists, reusing"
    else
        conda create -n "${CONDA_ENV}" "python=${PYTHON_VERSION}" -y
    fi
    conda activate "${CONDA_ENV}"

    python -m pip install --upgrade pip
    local need_pkgs=()
    command -v cmake  >/dev/null || need_pkgs+=(cmake)
    command -v ninja  >/dev/null || need_pkgs+=(ninja)
    if [ ${#need_pkgs[@]} -gt 0 ]; then
        conda install -y "${need_pkgs[@]}"
    fi

    if is_true "${SKIP_DEPS_INSTALL:-0}"; then
        log INFO "SKIP_DEPS_INSTALL=1, skipping requirements.txt"
    else
        log INFO "Installing dependencies from requirements.txt"
        pip install -r "${TILEKERNELS_HOME}/requirements.txt" --progress-bar on
    fi

    # Derived env (after activation so CONDA_PREFIX is set)
    export TILEKERNELS_HOME
    export ENV_PATH="${CONDA_PREFIX}"
    export CMAKE_PREFIX_PATH="${CONDA_PREFIX}"
    local cmake_xy
    cmake_xy=$(cmake --version | awk 'NR==1{split($3,a,"."); print a[1]"."a[2]}')
    export CMAKE_ROOT="${CONDA_PREFIX}/share/cmake-${cmake_xy}"
    export PYTORCH_DIR="${ENV_PATH}/lib/python${PYTHON_VERSION}/site-packages/"
    export PYTHON_INCLUDE_DIR="${ENV_PATH}/include/python${PYTHON_VERSION}"
    export PTPU_PATH="${ENV_PATH}/lib/python${PYTHON_VERSION}/site-packages/torch_ptpu"
    export LD_LIBRARY_PATH="${ENV_PATH}/lib:${LD_LIBRARY_PATH}"

    if is_true "${USE_CCACHE:-0}" && command -v ccache &>/dev/null; then
        export CMAKE_C_COMPILER_LAUNCHER=ccache
        export CMAKE_CXX_COMPILER_LAUNCHER=ccache
        log INFO "ccache enabled ($(ccache --version | head -1))"
    fi
}

# ============================================================
# Phase 3: Build tilelang (+ TVM, tvm-ffi)
# ============================================================
patch_libbacktrace() {
    local target="3rdparty/libbacktrace/elf.c"
    [ -f "${target}" ] || return 0
    if grep -q "^#define BACKTRACE_ELF_SIZE 64" "${target}"; then
        return 0
    fi
    local line_num
    line_num=$(grep -n "BACKTRACE_ELF_SIZE != 64" "${target}" | head -1 | cut -d: -f1 || true)
    if [ -n "${line_num}" ]; then
        sed -i "${line_num}i#define BACKTRACE_ELF_SIZE 64" "${target}"
        log INFO "Patched ${target} (line ${line_num})"
    fi
}

phase_tilelang() {
    log INFO "=========== Phase 3: tilelang ==========="

    if [ ! -f "${TILELANG_HOME}/VENDORED_COMPONENTS.md" ]; then
        log ERROR "TileLang-Sunrise source tree not found at ${TILELANG_HOME}"
        exit 1
    fi
    log INFO "Using vendored TileLang-Sunrise source at ${TILELANG_HOME}"

    export TVM_HOME="${TILELANG_HOME}/3rdparty/tvm_sunrise"
    export TVM_PREBUILD_PATH="${TVM_HOME}/build"
    export TVM_SOURCE_DIR="${TVM_HOME}"
    export PYTHONPATH="${TVM_HOME}/python:${TVM_HOME}/3rdparty/tvm-ffi/python:${PYTHONPATH:-}"

    pushd "${TVM_HOME}" >/dev/null
        patch_libbacktrace

        log INFO "------ Building tvm-ffi ------"
        if [ ! -d "3rdparty/tvm-ffi" ]; then
            log ERROR "tvm-ffi directory not found"
            exit 1
        fi
        pushd 3rdparty/tvm-ffi >/dev/null
            pip install -e . -v
        popd >/dev/null

        log INFO "------ Building TVM (jobs=${MAX_JOBS}) ------"
        if is_true "${SKIP_TVM_BUILD:-0}" && [ -f "${TVM_HOME}/build/CMakeCache.txt" ]; then
            log INFO "SKIP_TVM_BUILD=1, incremental rebuild only"
        else
            rm -rf "${TVM_HOME}/build"
            mkdir -p "${TVM_HOME}/build"
        fi
        pushd build >/dev/null
            cmake .. -G Ninja \
                -DUSE_TANG=1 -DUSE_TADNN=0 -DUSE_HEXAGON=0 \
                -DUSE_CUDA=OFF -DUSE_OPENCL=OFF -DUSE_CUTLASS=OFF \
                -DCMAKE_TANG_COMPILER="${PTCC_PATH}" \
                -DTANG_TOOLKIT_ROOT_DIR="${TANGRT_PATH}" \
                -DCMAKE_MODULE_PATH="${CMAKE_PATH}"
            cmake --build . --parallel "${MAX_JOBS}"
        popd >/dev/null
    popd >/dev/null
    log OK "TVM build complete"

    if is_true "${SKIP_TILELANG_INSTALL:-0}"; then
        log INFO "SKIP_TILELANG_INSTALL=1, skipping tilelang pip install"
    else
        log INFO "------ Installing tilelang ------"
        pushd "${TILELANG_HOME}" >/dev/null
            USE_TANG=ON pip install -e . -v
        popd >/dev/null
        log OK "tilelang installed"
    fi
}

# ============================================================
# Phase 4: TileKernels
# ============================================================
phase_tilekernels() {
    if is_true "${SKIP_TILEKERNELS_INSTALL:-0}"; then
        log INFO "SKIP_TILEKERNELS_INSTALL=1, skipping"
        return
    fi
    log INFO "=========== Phase 4: TileKernels ==========="
    pushd "${TILEKERNELS_HOME}" >/dev/null
        pip install -e ".[dev]" -v
    popd >/dev/null
    log OK "TileKernels installed"
}

# ============================================================
# Phase 4.5: pre-commit hooks
# ============================================================
phase_pre_commit() {
    if is_true "${SKIP_PRE_COMMIT:-0}"; then
        log INFO "SKIP_PRE_COMMIT=1, skipping pre-commit setup"
        return
    fi
    log INFO "=========== Phase 4.5: pre-commit ==========="
    pip install pre-commit
    pushd "${TILEKERNELS_HOME}" >/dev/null
        if [ -f ".pre-commit-config.yaml" ] && [ -d ".git" ]; then
            pre-commit install
            log OK "pre-commit hooks installed"
        else
            log INFO "no .pre-commit-config.yaml or .git directory, skipping hook install"
        fi
    popd >/dev/null
}

# ============================================================
# Phase 5: Tests
# ============================================================
phase_tests() {
    if is_true "${SKIP_TESTS:-0}"; then
        log INFO "SKIP_TESTS=1, skipping pytest"
        return
    fi
    log INFO "=========== Phase 5: tests ==========="
    # Allow overriding test targets and extra pytest args, e.g.:
    #   TEST_TARGETS="tests/moe tests/foo" PYTEST_ARGS="-x -k bar" ./one_click_install.sh
    local targets=(${TEST_TARGETS:-tests/moe/test_topk_gate.py})
    # local extra_args=(${PYTEST_ARGS:---run-benchmark})
    pushd "${TILEKERNELS_HOME}" >/dev/null
        # pytest "${targets[@]}" "${extra_args[@]}"
        pytest "${targets[@]}"
    popd >/dev/null
}

# ============================================================
# Phase 6: Benchmarks (performance tests)
# ============================================================
# Driven by repo's tests/pytest_benchmark_plugin.py:
#   --run-benchmark                     enable benchmark tests
#   --benchmark-output PATH             write JSONL results
#   --benchmark-regression-threshold X  fail if regression > X (e.g. 0.1 = 10%)
#   --benchmark-verbose                 extra columns in report
#
# Toggle envs:
#   SKIP_BENCHMARK=1                    skip this phase
#   BENCHMARK_TARGETS="tests/moe ..."   pytest targets (default: tests)
#   BENCHMARK_PYTEST_ARGS="-k foo"      extra pytest args
#   BENCHMARK_OUTPUT=path.jsonl         override output path
#   BENCHMARK_REGRESSION_THRESHOLD=0.1  default 0.1 (10%)
#   BENCHMARK_VERBOSE=1                 add --benchmark-verbose
#   BENCHMARK_ALLOW_FAILURE=1           never fail the script on regressions
phase_benchmark() {
    if is_true "${SKIP_BENCHMARK:-0}"; then
        log INFO "SKIP_BENCHMARK=1, skipping performance tests"
        return
    fi
    log INFO "=========== Phase 6: benchmark ==========="

    local ts; ts=$(date '+%Y%m%d_%H%M%S')
    local out_default="${TILEKERNELS_HOME}/benchmark_results_${ts}.jsonl"
    local out_path="${BENCHMARK_OUTPUT:-${out_default}}"
    local threshold="${BENCHMARK_REGRESSION_THRESHOLD:-0.1}"

    local targets=(${BENCHMARK_TARGETS:-tests})
    local extra_args=(${BENCHMARK_PYTEST_ARGS:-})

    local bench_args=(
        --run-benchmark
        --benchmark-output "${out_path}"
        --benchmark-regression-threshold "${threshold}"
    )
    if is_true "${BENCHMARK_VERBOSE:-0}"; then
        bench_args+=(--benchmark-verbose)
    fi

    log INFO "targets         : ${targets[*]}"
    log INFO "output          : ${out_path}"
    log INFO "threshold       : ${threshold}"
    log INFO "extra pytest    : ${extra_args[*]:-<none>}"

    local rc=0
    pushd "${TILEKERNELS_HOME}" >/dev/null
        if is_true "${BENCHMARK_ALLOW_FAILURE:-0}"; then
            pytest "${targets[@]}" "${bench_args[@]}" "${extra_args[@]}" || rc=$?
            if [ "${rc}" -ne 0 ]; then
                log INFO "benchmark exited with rc=${rc}, ignoring (BENCHMARK_ALLOW_FAILURE=1)"
                rc=0
            fi
        else
            pytest "${targets[@]}" "${bench_args[@]}" "${extra_args[@]}"
        fi
    popd >/dev/null

    if [ -f "${out_path}" ]; then
        log OK "benchmark results written to: ${out_path}"
    else
        log INFO "benchmark output not found at ${out_path} (no benchmark tests collected?)"
    fi
}

# ============================================================
# Main
# ============================================================
main() {
    log INFO "TILEKERNELS_HOME=${TILEKERNELS_HOME}"
    log INFO "CONDA_ENV=${CONDA_ENV}  PYTHON=${PYTHON_VERSION}  MAX_JOBS=${MAX_JOBS}"

    phase_conda
    phase_clean_cache
    phase_env
    phase_tilelang
    phase_tilekernels
    phase_pre_commit
    phase_tests
    phase_benchmark

    log OK "All phases finished successfully"
}

main "$@"
