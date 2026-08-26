#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║                  TileOps — One-Click Install Script                  ║
# ╠══════════════════════════════════════════════════════════════════════╣
# ║  Phases                                                              ║
# ║    0. Ensure conda is available                                      ║
# ║    1. (optional) Clean caches                                        ║
# ║    2. Create / activate conda env, install python deps               ║
# ║    3. Clone & build tilelang (with bundled TVM and tvm-ffi)          ║
# ║    4. Install TileOps (editable, with [dev] extras)                  ║
# ║    5. Run tests                                                      ║
# ╠══════════════════════════════════════════════════════════════════════╣
# ║  Toggles (env vars; default = run normally)                          ║
# ║    SKIP_CONDA_INSTALL=1     skip miniconda bootstrap                 ║
# ║    SKIP_TVM_BUILD=1         reuse existing TVM build dir             ║
# ║    SKIP_TILELANG_CLONE=1    reuse existing tilelang checkout         ║
# ║    SKIP_TILELANG_INSTALL=1  skip `pip install -e .` for tilelang     ║
# ║    SKIP_TILEOPS_INSTALL=1   skip TileOps install                     ║
# ║    SKIP_TESTS=1             skip pytest phase                        ║
# ║    CLEAN_CACHE=1            run `conda clean` and `pip cache purge`  ║
# ║    MAX_JOBS=N               cmake build parallelism (default: nproc) ║
# ║    USE_CCACHE=1             enable ccache (auto-on if installed)     ║
# ║    USE_HTTPS_FALLBACK=1     fallback git@ -> https:// on clone fail  ║
# ║    TEST_TARGETS="..."       pytest targets                           ║
# ║    PYTEST_ARGS="..."        extra pytest args                        ║
# ║    VERBOSE=1                stream sub-command output to terminal    ║
# ║    LOG_DIR=/path            override log directory                   ║
# ║                                                                      ║
# ║  Run with -h or --help for the same listing.                         ║
# ╚══════════════════════════════════════════════════════════════════════╝

set -Eeuo pipefail

# ──────────────────────────────────────────────────────────────────────
#  Colors / TTY detection
# ──────────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    C_RESET=$'\033[0m'
    C_DIM=$'\033[2m'
    C_BOLD=$'\033[1m'
    C_INFO=$'\033[1;38;5;39m'    # cyan-blue
    C_OK=$'\033[1;38;5;42m'      # green
    C_WARN=$'\033[1;38;5;214m'   # orange
    C_ERR=$'\033[1;38;5;203m'    # red
    C_ACCENT=$'\033[1;38;5;141m' # purple
    C_MUTED=$'\033[38;5;245m'    # grey
    TTY=1
else
    C_RESET='' C_DIM='' C_BOLD='' C_INFO='' C_OK=''
    C_WARN='' C_ERR='' C_ACCENT='' C_MUTED=''
    TTY=0
fi
readonly C_RESET C_DIM C_BOLD C_INFO C_OK C_WARN C_ERR C_ACCENT C_MUTED TTY

if [[ "${LANG:-}${LC_ALL:-}" =~ [Uu][Tt][Ff] ]]; then
    GLYPH_OK='✔' GLYPH_FAIL='✘' GLYPH_SKIP='∘' GLYPH_RUN='▶' GLYPH_ARROW='›'
else
    GLYPH_OK='[OK]' GLYPH_FAIL='[X]' GLYPH_SKIP='[-]' GLYPH_RUN='>' GLYPH_ARROW='>'
fi
readonly GLYPH_OK GLYPH_FAIL GLYPH_SKIP GLYPH_RUN GLYPH_ARROW

# ──────────────────────────────────────────────────────────────────────
#  Tiny helpers
# ──────────────────────────────────────────────────────────────────────
is_true() { [[ "${1:-0}" =~ ^(1|true|TRUE|yes|YES)$ ]]; }
have()    { command -v "$1" >/dev/null 2>&1; }

fmt_dur() {
    local s=${1:-0}
    if   (( s < 60 ));   then printf '%ds' "$s"
    elif (( s < 3600 )); then printf '%dm%02ds' $((s/60)) $((s%60))
    else                      printf '%dh%02dm%02ds' $((s/3600)) $(((s%3600)/60)) $((s%60))
    fi
}

now_s() { printf '%(%s)T' -1; }

log() {
    local level="${1:-INFO}"; shift || true
    local color sym
    case "${level}" in
        OK)    color="${C_OK}";   sym="${GLYPH_OK}"   ;;
        WARN)  color="${C_WARN}"; sym="!"             ;;
        ERROR) color="${C_ERR}";  sym="${GLYPH_FAIL}" ;;
        STEP)  color="${C_ACCENT}"; sym="${GLYPH_ARROW}" ;;
        *)     color="${C_INFO}"; sym="${GLYPH_RUN}"  ;;
    esac
    local ts
    printf -v ts '%(%H:%M:%S)T' -1 2>/dev/null || ts=$(date '+%T')
    printf '%s%s%s %s%s%s %s\n' \
        "${C_MUTED}" "${ts}" "${C_RESET}" \
        "${color}" "${sym}" "${C_RESET}" "$*"
}

usage() {
    sed -n '2,/^# ╚/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
}

to_https_url() {
    local url="$1"
    if [[ "${url}" =~ ^git@([^:]+):(.+)$ ]]; then
        printf 'https://%s/%s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
    else
        printf '%s\n' "${url}"
    fi
}

# Split a string into an array safely (whitespace-separated).
split_into() {
    local -n _arr=$1
    local _val="${2:-}"
    if [ -n "${_val}" ]; then
        # shellcheck disable=SC2206
        read -ra _arr <<<"${_val}"
    else
        shift 2
        _arr=("$@")
    fi
}

# ──────────────────────────────────────────────────────────────────────
#  Banner / ASCII logo
# ──────────────────────────────────────────────────────────────────────
print_logo() {
    local g1=$'\033[1;38;5;208m' g2=$'\033[1;38;5;214m' g3=$'\033[1;38;5;220m'
    local b1=$'\033[1;38;5;39m'  b2=$'\033[1;38;5;45m'  b3=$'\033[1;38;5;51m' b4=$'\033[1;38;5;141m'
    if [[ $TTY == 0 ]]; then
        g1='' g2='' g3='' b1='' b2='' b3='' b4=''
    fi
    cat <<EOF
${g1}   _____                  _          ${C_RESET}
${g1}  / ____|                (_)         ${C_RESET}
${g2} | (___  _   _ _ __  _ __ _ ___  ___ ${C_RESET}
${g2}  \___ \| | | | '_ \| '__| / __|/ _ \\${C_RESET}
${g3}  ____) | |_| | | | | |  | \__ \  __/${C_RESET}
${g3} |_____/ \__,_|_| |_|_|  |_|___/\___|${C_RESET}
${b1}  _______ _ _       ___              ${C_RESET}
${b2} |__   __(_) |     / _ \\             ${C_RESET}
${b3}    | |   _| | ___| | | |_ __  ___   ${C_RESET}
${b3}    | |  | | |/ _ \\ | | | '_ \\/ __|  ${C_RESET}
${b4}    | |  | | |  __/ |_| | |_) \\__ \\  ${C_RESET}
${b4}    |_|  |_|_|\\___|\\___/| .__/|___/  ${C_RESET}
${b4}                        | |          ${C_RESET}
${b4}                        |_|          ${C_RESET}
EOF
}

banner() {
    local title="$*"
    local w=68 line
    printf -v line '%*s' "$w" ''; line=${line// /═}
    printf '%s╔%s╗%s\n' "${C_INFO}" "${line}" "${C_RESET}"
    printf '%s║%s %s%-*s%s %s║%s\n' \
        "${C_INFO}" "${C_RESET}" "${C_BOLD}" $((w-2)) "${title}" "${C_RESET}" \
        "${C_INFO}" "${C_RESET}"
    printf '%s╚%s╝%s\n' "${C_INFO}" "${line}" "${C_RESET}"
}

phase_header() {
    local idx=$1 total=$2 title=$3
    local w=68 line
    printf -v line '%*s' "$w" ''; line=${line// /─}
    printf '\n%s┌%s┐%s\n' "${C_ACCENT}" "${line}" "${C_RESET}"
    printf '%s│%s %s[%d/%d]%s %s%-*s%s %s│%s\n' \
        "${C_ACCENT}" "${C_RESET}" \
        "${C_BOLD}" "${idx}" "${total}" "${C_RESET}" \
        "${C_BOLD}" $((w-2-${#idx}-${#total}-3)) "${title}" "${C_RESET}" \
        "${C_ACCENT}" "${C_RESET}"
    printf '%s└%s┘%s\n' "${C_ACCENT}" "${line}" "${C_RESET}"
}

# ──────────────────────────────────────────────────────────────────────
#  Phase tracking & spinner
# ──────────────────────────────────────────────────────────────────────
SCRIPT_START_TS=$(now_s)
PHASES_TOTAL=6
PHASE_IDX=0
declare -a PHASE_NAMES=() PHASE_STATUS=() PHASE_DUR=() PHASE_LOG=()
CURRENT_PHASE_NAME=""
CURRENT_PHASE_START=0
CURRENT_PHASE_LOG=""

LOG_DIR="${LOG_DIR:-/tmp/tileops_install_$(date +%Y%m%d_%H%M%S)_$$}"
mkdir -p "${LOG_DIR}"

start_phase() {
    PHASE_IDX=$((PHASE_IDX+1))
    CURRENT_PHASE_NAME="$1"
    CURRENT_PHASE_START=$(now_s)
    CURRENT_PHASE_LOG="${LOG_DIR}/phase${PHASE_IDX}_$(echo "$1" | tr ' /' '__' | tr -cd '[:alnum:]_').log"
    : > "${CURRENT_PHASE_LOG}"
    phase_header "${PHASE_IDX}" "${PHASES_TOTAL}" "${CURRENT_PHASE_NAME}"
    printf '%s   log: %s%s\n' "${C_MUTED}" "${CURRENT_PHASE_LOG}" "${C_RESET}"
}

end_phase() {
    local status="${1:-OK}"
    local dur=$(( $(now_s) - CURRENT_PHASE_START ))
    PHASE_NAMES+=("${CURRENT_PHASE_NAME}")
    PHASE_STATUS+=("${status}")
    PHASE_DUR+=("${dur}")
    PHASE_LOG+=("${CURRENT_PHASE_LOG}")
    local color sym
    case "${status}" in
        OK)   color="${C_OK}";   sym="${GLYPH_OK}"   ;;
        SKIP) color="${C_MUTED}";sym="${GLYPH_SKIP}" ;;
        FAIL) color="${C_ERR}";  sym="${GLYPH_FAIL}" ;;
        *)    color="${C_WARN}"; sym='?'             ;;
    esac
    printf '%s%s %s — %s (%s)%s\n' \
        "${color}" "${sym}" "${CURRENT_PHASE_NAME}" "${status}" "$(fmt_dur "${dur}")" "${C_RESET}"
}

SPINNER_FRAMES='⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏'

# run_logged "Description" -- cmd args...
run_logged() {
    local desc="$1"; shift
    [[ "$1" == "--" ]] && shift
    local logfile="${CURRENT_PHASE_LOG:-/dev/null}"

    {
        printf '\n=== [%s] %s ===\n' "$(date '+%F %T')" "${desc}"
        printf '$ %s\n' "$*"
    } >> "${logfile}"

    if is_true "${VERBOSE:-0}" || [[ $TTY == 0 ]]; then
        log STEP "${desc}"
        if "$@" 2>&1 | tee -a "${logfile}"; then
            return 0
        else
            local rc=${PIPESTATUS[0]}
            log ERROR "${desc} failed (exit=${rc})"
            log ERROR "  log: ${logfile}"
            return "${rc}"
        fi
    fi

    local start=$(now_s)
    "$@" >>"${logfile}" 2>&1 &
    local pid=$!
    local frames=(${SPINNER_FRAMES})
    local i=0
    printf '\033[?25l'
    while kill -0 "$pid" 2>/dev/null; do
        local elapsed=$(( $(now_s) - start ))
        printf '\r%s%s%s %s %s(%s)%s\033[K' \
            "${C_ACCENT}" "${frames[i]}" "${C_RESET}" \
            "${desc}" \
            "${C_MUTED}" "$(fmt_dur "${elapsed}")" "${C_RESET}"
        i=$(( (i+1) % ${#frames[@]} ))
        sleep 0.1
    done
    wait "$pid"
    local rc=$?
    local elapsed=$(( $(now_s) - start ))
    printf '\r\033[K\033[?25h'
    if (( rc == 0 )); then
        printf '%s%s%s %s %s(%s)%s\n' \
            "${C_OK}" "${GLYPH_OK}" "${C_RESET}" \
            "${desc}" "${C_MUTED}" "$(fmt_dur "${elapsed}")" "${C_RESET}"
    else
        printf '%s%s%s %s %s(%s, exit=%d)%s\n' \
            "${C_ERR}" "${GLYPH_FAIL}" "${C_RESET}" \
            "${desc}" "${C_MUTED}" "$(fmt_dur "${elapsed}")" "${rc}" "${C_RESET}"
        log ERROR "Last 30 lines of ${logfile}:"
        printf '%s' "${C_DIM}"
        tail -n 30 "${logfile}" | sed 's/^/    /'
        printf '%s' "${C_RESET}"
    fi
    return "${rc}"
}

# ──────────────────────────────────────────────────────────────────────
#  Final summary
# ──────────────────────────────────────────────────────────────────────
final_summary() {
    local total=$(( $(now_s) - SCRIPT_START_TS ))
    local w=72 line
    printf -v line '%*s' "$w" ''; line=${line// /═}
    printf '\n%s╔%s╗%s\n' "${C_ACCENT}" "${line}" "${C_RESET}"
    printf '%s║%s %s%-*s%s %s║%s\n' \
        "${C_ACCENT}" "${C_RESET}" "${C_BOLD}" $((w-2)) "Install summary" "${C_RESET}" \
        "${C_ACCENT}" "${C_RESET}"
    printf '%s╠%s╣%s\n' "${C_ACCENT}" "${line}" "${C_RESET}"
    printf '%s║%s %-32s %-7s %-10s %-18s %s║%s\n' \
        "${C_ACCENT}" "${C_RESET}" "Phase" "Status" "Duration" "Log" "${C_ACCENT}" "${C_RESET}"
    printf '%s╟%s╢%s\n' "${C_ACCENT}" "${line//═/─}" "${C_RESET}"
    local i color sym
    for i in "${!PHASE_NAMES[@]}"; do
        case "${PHASE_STATUS[$i]}" in
            OK)   color="${C_OK}";   sym="${GLYPH_OK}"   ;;
            SKIP) color="${C_MUTED}";sym="${GLYPH_SKIP}" ;;
            FAIL) color="${C_ERR}";  sym="${GLYPH_FAIL}" ;;
            *)    color="${C_WARN}"; sym='?'             ;;
        esac
        printf '%s║%s %-32s %s%s %-5s%s %-10s %-18s %s║%s\n' \
            "${C_ACCENT}" "${C_RESET}" \
            "${PHASE_NAMES[$i]:0:32}" \
            "${color}" "${sym}" "${PHASE_STATUS[$i]}" "${C_RESET}" \
            "$(fmt_dur "${PHASE_DUR[$i]}")" \
            "$(basename "${PHASE_LOG[$i]}" | cut -c1-18)" \
            "${C_ACCENT}" "${C_RESET}"
    done
    printf '%s╠%s╣%s\n' "${C_ACCENT}" "${line}" "${C_RESET}"
    printf '%s║%s %-32s %s%-7s%s %-10s %-18s %s║%s\n' \
        "${C_ACCENT}" "${C_RESET}" "TOTAL" \
        "${C_BOLD}" "" "${C_RESET}" \
        "$(fmt_dur "${total}")" "${LOG_DIR##*/}" "${C_ACCENT}" "${C_RESET}"
    printf '%s╚%s╝%s\n' "${C_ACCENT}" "${line}" "${C_RESET}"
    printf '%sFull logs: %s%s\n\n' "${C_MUTED}" "${LOG_DIR}" "${C_RESET}"
}

# ──────────────────────────────────────────────────────────────────────
#  Error / exit traps
# ──────────────────────────────────────────────────────────────────────
on_error() {
    local exit_code=$?
    local line_no=${1:-?}
    log ERROR "Script failed at line ${line_no} (exit=${exit_code})"
    log ERROR "Last command: ${BASH_COMMAND}"
    if [[ -n "${CURRENT_PHASE_NAME}" ]] && (( ${#PHASE_NAMES[@]} < PHASE_IDX )); then
        end_phase FAIL
    fi
    exit "${exit_code}"
}
on_exit() {
    [[ $TTY == 1 ]] && printf '\033[?25h'
    if (( ${#PHASE_NAMES[@]} > 0 )); then
        final_summary
    fi
}
trap 'on_error $LINENO' ERR
trap 'on_exit' EXIT

# ──────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────
CONDA_ENV=${CONDA_ENV:-tilelang_dev}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TILEOPS_HOME=${TILEOPS_HOME:-$(cd "${SCRIPT_DIR}/.." && pwd)}
TILELANG_SOURCE_DIR=${TILELANG_SOURCE_DIR:-$(cd "${TILEOPS_HOME}/../.." && pwd)}
TILELANG_HOME=${TILELANG_HOME:-"${TILELANG_SOURCE_DIR}"}

PYTHON_VERSION=${PYTHON_VERSION:-3.10}
_DETECTED_JOBS=$(nproc 2>/dev/null || echo 4)
(( _DETECTED_JOBS > 32 )) && _DETECTED_JOBS=32
MAX_JOBS=${MAX_JOBS:-${_DETECTED_JOBS}}
unset _DETECTED_JOBS

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
export LD_LIBRARY_PATH="${TANGRT_LIB_PATH}:/usr/local/tangrt/targets/linux-x86_64/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# ══════════════════════════════════════════════════════════════════════
#  Phase 0 — Conda bootstrap
# ══════════════════════════════════════════════════════════════════════
phase_conda() {
    start_phase "Conda bootstrap"

    if is_true "${SKIP_CONDA_INSTALL:-0}"; then
        log INFO "SKIP_CONDA_INSTALL=1 — skipping miniconda bootstrap"
    elif ! have conda && [[ ! -x "${HOME}/miniconda3/bin/conda" ]]; then
        log INFO "Conda not found — installing Miniconda3..."
        local tmp installer
        tmp=$(mktemp -d)
        # shellcheck disable=SC2064
        trap "rm -rf '${tmp}'; on_exit" EXIT
        installer="${tmp}/miniconda.sh"

        run_logged "Download Miniconda installer" -- \
            wget -q --tries=3 --timeout=60 \
                https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh \
                -O "${installer}"
        run_logged "Run Miniconda installer" -- \
            bash "${installer}" -b -p "${HOME}/miniconda3"
        rm -rf "${tmp}"
        trap 'on_exit' EXIT
    fi

    export PATH="${HOME}/miniconda3/bin:${PATH}"

    local conda_bin conda_base
    if have conda; then
        conda_bin=$(command -v conda)
    else
        conda_bin="${HOME}/miniconda3/bin/conda"
    fi
    conda_base=$("${conda_bin}" info --base)

    # shellcheck disable=SC1091
    source "${conda_base}/etc/profile.d/conda.sh"
    log OK "conda available — $(conda --version)"
    end_phase OK
}

# ══════════════════════════════════════════════════════════════════════
#  Phase 1 — (optional) Cache cleaning
# ══════════════════════════════════════════════════════════════════════
phase_clean_cache() {
    start_phase "Cache clean"
    if is_true "${CLEAN_CACHE:-0}"; then
        if have conda; then
            run_logged "conda clean --all" -- conda clean --all -y -q || true
        fi
        if have pip; then
            run_logged "pip cache purge" -- pip cache purge || true
        fi
        end_phase OK
    else
        log INFO "Cache clean skipped (set CLEAN_CACHE=1 to enable)"
        end_phase SKIP
    fi
}

# ══════════════════════════════════════════════════════════════════════
#  Phase 2 — Conda env + python deps
# ══════════════════════════════════════════════════════════════════════
phase_env() {
    start_phase "Conda env '${CONDA_ENV}' + deps"

    if conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -qx "${CONDA_ENV}"; then
        log INFO "env '${CONDA_ENV}' already exists — reusing"
    else
        run_logged "conda create -n ${CONDA_ENV} python=${PYTHON_VERSION}" -- \
            conda create -n "${CONDA_ENV}" "python=${PYTHON_VERSION}" -y
    fi
    set +u
    conda activate "${CONDA_ENV}"
    set -u

    run_logged "Upgrade pip" -- python -m pip install --upgrade pip --quiet

    local need_pkgs=()
    have cmake || need_pkgs+=(cmake)
    have ninja || need_pkgs+=(ninja)
    if is_true "${USE_CCACHE:-0}" && ! have ccache; then
        need_pkgs+=(ccache)
    fi
    if (( ${#need_pkgs[@]} > 0 )); then
        run_logged "conda install ${need_pkgs[*]}" -- \
            conda install -y -c conda-forge "${need_pkgs[@]}"
    fi

    export TILEOPS_HOME
    export ENV_PATH="${CONDA_PREFIX}"
    export CMAKE_PREFIX_PATH="${CONDA_PREFIX}"

    if ! have cmake; then
        log ERROR "cmake not found in PATH after env setup"
        end_phase FAIL
        return 1
    fi

    local cmake_xy
    cmake_xy=$(cmake --version | awk 'NR==1{split($3,a,"."); print a[1]"."a[2]}')
    export CMAKE_ROOT="${CONDA_PREFIX}/share/cmake-${cmake_xy}"
    export PYTORCH_DIR="${ENV_PATH}/lib/python${PYTHON_VERSION}/site-packages/"
    export PYTHON_INCLUDE_DIR="${ENV_PATH}/include/python${PYTHON_VERSION}"
    export PTPU_PATH="${ENV_PATH}/lib/python${PYTHON_VERSION}/site-packages/torch_ptpu"
    export LD_LIBRARY_PATH="${ENV_PATH}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

    if is_true "${USE_CCACHE:-0}" && have ccache; then
        export CMAKE_C_COMPILER_LAUNCHER=ccache
        export CMAKE_CXX_COMPILER_LAUNCHER=ccache
        export CCACHE_DIR="${CCACHE_DIR:-${HOME}/.ccache}"
        export CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-20G}"
        ccache -o max_size="${CCACHE_MAXSIZE}" >/dev/null 2>&1 || true
        log OK "ccache enabled — $(ccache --version | head -1) · dir=${CCACHE_DIR} · max=${CCACHE_MAXSIZE}"
    fi
    end_phase OK
}

# ══════════════════════════════════════════════════════════════════════
#  Phase 3 — Build tilelang (+ TVM, tvm-ffi)
# ══════════════════════════════════════════════════════════════════════
patch_libbacktrace() {
    local target="3rdparty/libbacktrace/elf.c"
    [[ -f "${target}" ]] || return 0
    if grep -q "^#define BACKTRACE_ELF_SIZE 64" "${target}"; then
        return 0
    fi
    local line_num
    line_num=$(grep -n "BACKTRACE_ELF_SIZE != 64" "${target}" | head -1 | cut -d: -f1 || true)
    if [[ -n "${line_num}" ]]; then
        sed -i "${line_num}i#define BACKTRACE_ELF_SIZE 64" "${target}"
        log INFO "Patched ${target} (line ${line_num})"
    fi
}

clone_tilelang() {
    if [[ ! -f "${TILELANG_HOME}/VENDORED_COMPONENTS.md" ]]; then
        log ERROR "TileLang-Sunrise source tree not found at ${TILELANG_HOME}"
        return 1
    fi
    log INFO "Using vendored TileLang-Sunrise source at ${TILELANG_HOME}"
}

build_tvm_ffi() {
    if [[ ! -d "3rdparty/tvm-ffi" ]]; then
        log ERROR "tvm-ffi directory not found"
        return 1
    fi
    pushd 3rdparty/tvm-ffi >/dev/null
        run_logged "pip install -e tvm-ffi" -- pip install -e . -v
    popd >/dev/null
}

build_tvm() {
    local cache_file="${TVM_HOME}/build/CMakeCache.txt"
    if is_true "${SKIP_TVM_BUILD:-0}" && [[ -f "${cache_file}" ]]; then
        log INFO "SKIP_TVM_BUILD=1 — incremental rebuild only"
    else
        rm -rf "${TVM_HOME}/build"
    fi
    mkdir -p "${TVM_HOME}/build"

    pushd "${TVM_HOME}/build" >/dev/null
        if [[ ! -f CMakeCache.txt ]]; then
            run_logged "cmake configure TVM" -- \
                cmake .. -G Ninja \
                    -DCMAKE_BUILD_TYPE=Release \
                    -DUSE_TANG=1 -DUSE_TADNN=0 -DUSE_HEXAGON=0 \
                    -DUSE_CUDA=OFF -DUSE_OPENCL=OFF -DUSE_CUTLASS=OFF \
                    -DCMAKE_TANG_COMPILER="${PTCC_PATH}" \
                    -DTANG_TOOLKIT_ROOT_DIR="${TANGRT_PATH}" \
                    -DCMAKE_MODULE_PATH="${CMAKE_PATH}"
        fi
        run_logged "cmake build TVM (jobs=${MAX_JOBS})" -- \
            cmake --build . --parallel "${MAX_JOBS}"
    popd >/dev/null
}

phase_tilelang() {
    start_phase "Build & install tilelang"

    clone_tilelang

    export TVM_HOME="${TILELANG_HOME}/3rdparty/tvm_sunrise"
    export TVM_PREBUILD_PATH="${TVM_HOME}/build"
    export TVM_SOURCE_DIR="${TVM_HOME}"
    export PYTHONPATH="${TVM_HOME}/python:${TVM_HOME}/3rdparty/tvm-ffi/python:${PYTHONPATH:-}"

    pushd "${TVM_HOME}" >/dev/null
        patch_libbacktrace
        build_tvm_ffi
        build_tvm
    popd >/dev/null
    log OK "TVM build complete"

    if have ccache; then
        ccache -s 2>/dev/null | sed 's/^/[ccache] /' || true
    fi

    if is_true "${SKIP_TILELANG_INSTALL:-0}"; then
        log INFO "SKIP_TILELANG_INSTALL=1 — skipping tilelang pip install"
    else
        pushd "${TILELANG_HOME}" >/dev/null
            run_logged "pip install -e tilelang" -- \
                env USE_TANG=ON pip install -e . -v
        popd >/dev/null
        log OK "tilelang installed"
    fi
    end_phase OK
}

# ══════════════════════════════════════════════════════════════════════
#  Phase 4 — TileOps install
# ══════════════════════════════════════════════════════════════════════
phase_tileops() {
    start_phase "Install TileOps"
    if is_true "${SKIP_TILEOPS_INSTALL:-0}"; then
        log INFO "SKIP_TILEOPS_INSTALL=1 — skipping"
        end_phase SKIP
        return
    fi
    pushd "${TILEOPS_HOME}" >/dev/null
        run_logged "pip install -e .[dev]" -- \
            env PIP_NO_BUILD_ISOLATION=1 pip install -e '.[dev]' -v
        run_logged "pre-commit install" -- pre-commit install || \
            log WARN "pre-commit install failed (continuing)"
    popd >/dev/null
    log OK "TileOps installed"
    end_phase OK
}

# ══════════════════════════════════════════════════════════════════════
#  Phase 5 — Tests
# ══════════════════════════════════════════════════════════════════════
phase_tests() {
    start_phase "Run tests"
    if is_true "${SKIP_TESTS:-0}"; then
        log INFO "SKIP_TESTS=1 — skipping pytest"
        end_phase SKIP
        return
    fi
    local targets extra_args
    split_into targets    "${TEST_TARGETS:-}" tests/ops/test_fused_add_rms_norm.py
    split_into extra_args "${PYTEST_ARGS:-}"
    log INFO "targets    : ${targets[*]}"
    log INFO "extra args : ${extra_args[*]:-<none>}"
    pushd "${TILEOPS_HOME}" >/dev/null
        run_logged "pytest ${targets[*]}" -- pytest "${targets[@]}" "${extra_args[@]}"
    popd >/dev/null
    end_phase OK
}

# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════
main() {
    case "${1:-}" in
        -h|--help) usage ;;
    esac

    print_logo
    banner "TileOps One-Click Install"

    if [[ ! -f "${TILEOPS_HOME}/pyproject.toml" ]]; then
        log ERROR "pyproject.toml not found under ${TILEOPS_HOME}; wrong repo layout?"
        exit 1
    fi

    printf '  %sSCRIPT_DIR%s    = %s\n'    "${C_MUTED}" "${C_RESET}" "${SCRIPT_DIR}"
    printf '  %sTILEOPS_HOME%s  = %s\n'    "${C_MUTED}" "${C_RESET}" "${TILEOPS_HOME}"
    printf '  %sTILELANG_HOME%s = %s\n'    "${C_MUTED}" "${C_RESET}" "${TILELANG_HOME}"
    printf '  %sCONDA_ENV%s     = %s\n'    "${C_MUTED}" "${C_RESET}" "${CONDA_ENV}"
    printf '  %sPYTHON%s        = %s\n'    "${C_MUTED}" "${C_RESET}" "${PYTHON_VERSION}"
    printf '  %sMAX_JOBS%s      = %s\n'    "${C_MUTED}" "${C_RESET}" "${MAX_JOBS}"
    printf '  %sLOG_DIR%s       = %s\n\n'  "${C_MUTED}" "${C_RESET}" "${LOG_DIR}"

    phase_conda
    phase_clean_cache
    phase_env
    phase_tilelang
    phase_tileops
    phase_tests

    log OK "All phases finished successfully"
}

main "$@"
