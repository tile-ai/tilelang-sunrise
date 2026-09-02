#!/usr/bin/env bash
# Root-only lifecycle manager for the dedicated TileLang-Sunrise repository runner.
set -Eeuo pipefail
umask 027

readonly RUNNER_USER="tilelang-gh-runner"
readonly RUNNER_GROUP="tilelang-gh-runner"
readonly REPOSITORY_URL="https://github.com/tile-ai/tilelang-sunrise"
readonly RUNNER_NAME="tilelang-s2-runner-01"
readonly BASE_DIR="/home/github-actions/tilelang-sunrise"
readonly RUNNER_DIR="$BASE_DIR/runner"
readonly WORK_DIR="$BASE_DIR/work"
readonly CACHE_DIR="$BASE_DIR/cache"
readonly WHEEL_DIR="$BASE_DIR/vendor-wheels"
readonly TOOLCHAIN_DIR="$BASE_DIR/vendor-toolchain"
readonly EVIDENCE_DIR="$BASE_DIR/evidence"
readonly SOURCE_ROOT="$BASE_DIR/source"
readonly RUNNER_HOME="$BASE_DIR/home"
readonly TOOLCACHE_DIR="$BASE_DIR/toolcache"
readonly MINIFORGE_DIR="$TOOLCACHE_DIR/miniforge"
readonly ADMIN_TMP="$BASE_DIR/admin-tmp"
readonly CONFIG_DIR="/etc/tilelang-gh-runner"
readonly RUNNER_ENV="$CONFIG_DIR/runner.env"
readonly PREFLIGHT_ENV="$CONFIG_DIR/preflight.env"
readonly PUBLIC_DISCLOSURE_APPROVAL_FILE="$CONFIG_DIR/ptcc-public-disclosure-approved"
readonly NETWORK_HELPER="/usr/local/libexec/tilelang-gh-runner-network"
readonly PREFLIGHT_LAUNCHER="/usr/local/libexec/tilelang-gh-preflight-launcher"
readonly PROXY_SERVICE="tilelang-gh-proxy.service"
readonly NETWORK_SERVICE="tilelang-gh-network.service"
readonly RUNNER_SERVICE="tilelang-gh-runner.service"
readonly PREFLIGHT_SERVICE="tilelang-gh-preflight.service"
readonly SUDOERS_FILE="/etc/sudoers.d/tilelang-gh-runner"

readonly RUNNER_VERSION="2.337.0"
readonly RUNNER_ARCHIVE="actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
readonly RUNNER_URL="https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${RUNNER_ARCHIVE}"
readonly RUNNER_SHA256="70920811a4f8ad4328818682bca5c6469c1c942fab52448868071d0063816613"
readonly MINIFORGE_VERSION="26.5.3-0"
readonly MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-Linux-x86_64.sh"
readonly MINIFORGE_SHA256="14db468222ad564658656f769506056209b6dc375f5e7dfd31eb5ebbf08fa529"

readonly TORCH_WHEEL_NAME="torch_ptpu-0.2.3+torch2.10-cp310-cp310-linux_x86_64.whl"
readonly TORCH_WHEEL_SHA256="82a9af8d019e59565761946a9b810880764a4ca416287ef8095b6eeaf36bc463"
readonly TRITON_WHEEL_NAME="triton-3.4.0.3+git7e2003b3-cp310-cp310-linux_x86_64.whl"
readonly TRITON_WHEEL_SHA256="c010c33d294061d1774f058787ace3d5df4d9a0da3210e44241ae52250b86ce6"
readonly SYSTEM_PTCC_PATH="/usr/local/tangrt/toolchains/llvm/prebuilt/linux-x86_64/bin/ptcc"
readonly PINNED_PTCC_DIR="$TOOLCHAIN_DIR/ptcc-2.2.9-ceb3571d0"
readonly PINNED_PTCC_PATH="$PINNED_PTCC_DIR/ptcc"
readonly PINNED_PTCC_SHA256="762879026fa89dd3b5dd6b48aebc2f7abba239180187a603899d3a8b8335ebc2"
readonly PINNED_PTCC_VERSION="ptcc version 2.2.9 (ceb3571d0)"
readonly PUBLIC_DISCLOSURE_APPROVAL_VALUE="PTCC_PUBLIC_DISCLOSURE_APPROVED=tile-ai/tilelang-sunrise:$PINNED_PTCC_SHA256"

die () {
    echo "ERROR: $*" >&2
    exit 1
}

usage () {
    cat <<'EOF'
usage: manage.sh ACTION [ARGS]

actions:
  check
  provision
  stage-ptcc PTCC_BINARY_PATH
  refresh-units
  stage BUNDLE_PATH SOURCE_SHA BASE_SHA
  reset-test
  preflight
  register
  unregister
  start
  stop
  status
  rollback-list
  rollback
EOF
}

require_root () {
    [[ ${EUID:-$(id -u)} -eq 0 ]] || die "this action must run as root"
}

assert_safe_layout () {
    [[ "$BASE_DIR" == "/home/github-actions/tilelang-sunrise" ]] || die "refusing unexpected base path: $BASE_DIR"
    [[ "$CONFIG_DIR" == "/etc/tilelang-gh-runner" ]] || die "refusing unexpected config path: $CONFIG_DIR"
}

ensure_admin_tmp () {
    install -d -o root -g root -m 0700 "$ADMIN_TMP"
}

write_file () {
    local destination="$1" mode="$2"
    local temporary="$ADMIN_TMP/write.$$.${RANDOM}"
    cat > "$temporary"
    install -o root -g root -m "$mode" "$temporary" "$destination"
    rm -f "$temporary"
}

download_checked () {
    local url="$1" expected_sha="$2" destination="$3"
    local temporary="$ADMIN_TMP/download.$$.${RANDOM}"
    curl --fail --location --proto '=https' --tlsv1.2 "$url" --output "$temporary"
    echo "$expected_sha  $temporary" | sha256sum -c -
    mv -f "$temporary" "$destination"
}

create_account_and_directories () {
    install -d -o root -g root -m 0755 /home/github-actions
    install -d -o root -g root -m 0750 "$BASE_DIR"
    if ! getent passwd "$RUNNER_USER" >/dev/null; then
        useradd --system --user-group --home-dir "$RUNNER_HOME" --no-create-home --shell /sbin/nologin "$RUNNER_USER"
    fi
    [[ "$(id -gn "$RUNNER_USER")" == "$RUNNER_GROUP" ]] || die "$RUNNER_USER has an unexpected primary group"
    if id -nG "$RUNNER_USER" | tr ' ' '\n' | grep -Fxq docker; then
        die "$RUNNER_USER must not belong to the docker group"
    fi

    install -d -o root -g "$RUNNER_GROUP" -m 0750 "$BASE_DIR"
    install -d -o root -g root -m 0755 "$RUNNER_DIR"
    install -d -o root -g "$RUNNER_GROUP" -m 0750 "$TOOLCACHE_DIR"
    install -d -o root -g "$RUNNER_GROUP" -m 0550 "$WHEEL_DIR"
    install -d -o root -g "$RUNNER_GROUP" -m 0550 "$TOOLCHAIN_DIR"
    install -d -o "$RUNNER_USER" -g "$RUNNER_GROUP" -m 0700 \
        "$WORK_DIR" "$CACHE_DIR" "$EVIDENCE_DIR" "$SOURCE_ROOT" "$RUNNER_HOME"
    install -d -o "$RUNNER_USER" -g "$RUNNER_GROUP" -m 0700 \
        "$CACHE_DIR/conda-pkgs" "$CACHE_DIR/pip" "$CACHE_DIR/pre-commit" \
        "$CACHE_DIR/tilelang" "$CACHE_DIR/xdg" "$CACHE_DIR/ccache"
    install -d -o root -g root -m 0755 "$CONFIG_DIR" /usr/local/libexec
    ensure_admin_tmp
}

restrict_runtime_tree () {
    local path="$1"
    [[ -d "$path" ]] || die "runtime tree is missing: $path"
    chown -R root:"$RUNNER_GROUP" "$path"
    chmod -R g-w,o-rwx "$path"
    chmod -R g+rX "$path"
}

install_runner_distribution () {
    local marker="$RUNNER_DIR/.tilelang-runner-version"
    if [[ -f "$marker" ]]; then
        [[ "$(cat "$marker")" == "$RUNNER_VERSION" ]] || die "runner version marker does not match $RUNNER_VERSION"
        return 0
    fi
    if [[ -n "$(find "$RUNNER_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        die "$RUNNER_DIR is non-empty without a version marker"
    fi
    local archive="$ADMIN_TMP/$RUNNER_ARCHIVE"
    download_checked "$RUNNER_URL" "$RUNNER_SHA256" "$archive"
    tar -xzf "$archive" -C "$RUNNER_DIR"
    (cd "$RUNNER_DIR" && bash bin/installdependencies.sh)
    printf '%s\n' "$RUNNER_VERSION" > "$marker"
    chown -R root:root "$RUNNER_DIR"
    chmod -R go-w "$RUNNER_DIR"
    rm -f "$archive"
}

install_miniforge_distribution () {
    local marker="$MINIFORGE_DIR/.tilelang-miniforge-version"
    if [[ -f "$marker" ]]; then
        [[ "$(cat "$marker")" == "$MINIFORGE_VERSION" ]] || die "Miniforge version marker does not match $MINIFORGE_VERSION"
        restrict_runtime_tree "$MINIFORGE_DIR"
        return 0
    fi
    if [[ -d "$MINIFORGE_DIR" && -n "$(find "$MINIFORGE_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        die "$MINIFORGE_DIR is non-empty without a version marker"
    fi
    local installer="$ADMIN_TMP/Miniforge3-Linux-x86_64.sh"
    download_checked "$MINIFORGE_URL" "$MINIFORGE_SHA256" "$installer"
    bash "$installer" -b -p "$MINIFORGE_DIR"
    printf '%s\n' "$MINIFORGE_VERSION" > "$marker"
    restrict_runtime_tree "$MINIFORGE_DIR"
    rm -f "$installer"
}

stage_vendor_wheel () {
    local source_variable="$1" destination_name="$2" expected_sha="$3"
    local destination="$WHEEL_DIR/$destination_name"
    local source_value="${!source_variable:-}"
    chmod 0750 "$WHEEL_DIR"
    if [[ -f "$destination" ]] && echo "$expected_sha  $destination" | sha256sum -c - >/dev/null; then
        chmod 0440 "$destination"
        chmod 0550 "$WHEEL_DIR"
        return 0
    fi
    [[ -n "$source_value" ]] || die "$source_variable must name a local file or HTTPS URL"
    local temporary="$ADMIN_TMP/$destination_name.$$.${RANDOM}"
    case "$source_value" in
        https://*)
            curl --fail --location --proto '=https' --tlsv1.2 "$source_value" --output "$temporary"
            ;;
        /*)
            [[ -f "$source_value" ]] || die "$source_variable does not exist: $source_value"
            cp -f "$source_value" "$temporary"
            ;;
        *)
            die "$source_variable must be an absolute local path or HTTPS URL"
            ;;
    esac
    echo "$expected_sha  $temporary" | sha256sum -c -
    install -o root -g "$RUNNER_GROUP" -m 0440 "$temporary" "$destination"
    rm -f "$temporary"
    chmod 0550 "$WHEEL_DIR"
}

pinned_ptcc_matches () {
    local path="$1" output
    [[ -f "$path" && -x "$path" ]] || return 1
    echo "$PINNED_PTCC_SHA256  $path" | sha256sum -c - >/dev/null 2>&1 || return 1
    output=$("$path" --version 2>&1) || return 1
    grep -Fxq "$PINNED_PTCC_VERSION" <<< "$output"
}

verify_pinned_ptcc () {
    pinned_ptcc_matches "$PINNED_PTCC_PATH" || \
        die "pinned PTCC checksum or version does not match $PINNED_PTCC_VERSION"
}

stage_pinned_ptcc () {
    local source_value="${PTCC_BINARY_SOURCE:-}"
    chmod 0750 "$TOOLCHAIN_DIR"
    install -d -o root -g "$RUNNER_GROUP" -m 0750 "$PINNED_PTCC_DIR"
    if pinned_ptcc_matches "$PINNED_PTCC_PATH"; then
        chown root:"$RUNNER_GROUP" "$PINNED_PTCC_PATH"
        chmod 0550 "$PINNED_PTCC_PATH" "$PINNED_PTCC_DIR" "$TOOLCHAIN_DIR"
        return 0
    fi
    [[ "$source_value" == /* ]] || die "PTCC_BINARY_SOURCE must name an absolute local file"
    [[ -f "$source_value" ]] || die "PTCC_BINARY_SOURCE does not exist: $source_value"
    local temporary_dir="$ADMIN_TMP/ptcc-stage.$$.${RANDOM}"
    local temporary="$temporary_dir/ptcc"
    install -d -o root -g root -m 0700 "$temporary_dir"
    install -o root -g root -m 0700 "$source_value" "$temporary"
    if ! pinned_ptcc_matches "$temporary"; then
        rm -f "$temporary"
        rmdir "$temporary_dir"
        die "PTCC_BINARY_SOURCE checksum or version does not match $PINNED_PTCC_VERSION"
    fi
    install -o root -g "$RUNNER_GROUP" -m 0550 "$temporary" "$PINNED_PTCC_PATH"
    rm -f "$temporary"
    rmdir "$temporary_dir"
    chmod 0550 "$PINNED_PTCC_DIR" "$TOOLCHAIN_DIR"
}

stage_ptcc () {
    require_root
    assert_safe_layout
    local source_value="${1:-}"
    [[ $# -eq 1 ]] || die "stage-ptcc requires exactly one PTCC binary path"
    [[ -d "$BASE_DIR" ]] || die "runner base directory is not provisioned"
    getent passwd "$RUNNER_USER" >/dev/null || die "$RUNNER_USER is not provisioned"
    [[ -x "$SYSTEM_PTCC_PATH" ]] || die "PTCC bind-mount target is unavailable"
    install -d -o root -g "$RUNNER_GROUP" -m 0750 "$TOOLCHAIN_DIR"
    local PTCC_BINARY_SOURCE="$source_value"
    stage_pinned_ptcc
    verify_pinned_ptcc
    echo "Staged isolated $PINNED_PTCC_VERSION without changing the host compiler"
}

public_disclosure_is_approved () {
    [[ -f "$PUBLIC_DISCLOSURE_APPROVAL_FILE" && ! -L "$PUBLIC_DISCLOSURE_APPROVAL_FILE" ]] || return 1
    [[ "$(stat -c '%u:%g:%a' "$PUBLIC_DISCLOSURE_APPROVAL_FILE")" == "0:0:600" ]] || return 1
    [[ "$(cat "$PUBLIC_DISCLOSURE_APPROVAL_FILE")" == "$PUBLIC_DISCLOSURE_APPROVAL_VALUE" ]]
}

require_public_disclosure_approval () {
    public_disclosure_is_approved || \
        die "GitHub activation is blocked until the repository administrator explicitly approves PTCC public disclosure"
}

write_runner_environment () {
    write_file "$RUNNER_ENV" 0644 <<EOF
HOME=$RUNNER_HOME
PATH=$MINIFORGE_DIR/bin:$(dirname "$SYSTEM_PTCC_PATH"):/usr/local/bin:/usr/bin
CONDA_EXE=$MINIFORGE_DIR/bin/conda
HTTP_PROXY=http://127.0.0.1:3128
HTTPS_PROXY=http://127.0.0.1:3128
http_proxy=http://127.0.0.1:3128
https_proxy=http://127.0.0.1:3128
NO_PROXY=
no_proxy=
CONDARC=$CONFIG_DIR/condarc
CONDA_PKGS_DIRS=$CACHE_DIR/conda-pkgs
PIP_CACHE_DIR=$CACHE_DIR/pip
PRE_COMMIT_HOME=$CACHE_DIR/pre-commit
XDG_CACHE_HOME=$CACHE_DIR/xdg
CCACHE_DIR=$CACHE_DIR/ccache
TILELANG_CACHE_DIR=$CACHE_DIR/tilelang
TILELANG_EVIDENCE_ROOT=$EVIDENCE_DIR
TARGET_TORCH_VERSION=2.10.0
TARGET_TORCH_PTPU_PKG=$WHEEL_DIR/$TORCH_WHEEL_NAME
TARGET_TRITON_VERSION=3.4.3+git7e2003b3
TARGET_TRITON_PKG=$WHEEL_DIR/$TRITON_WHEEL_NAME
TARGET_TORCH_PKG_URL=https://download.pytorch.org/whl/cpu
TANG_VISIBLE_DEVICES=0
TILELANG_DEFAULT_TARGET=tang
TILELANG_CI_RESET_MODE=sudo-n
TILELANG_CI_PUBLIC_LOGS=1
TANGRT_PATH=/usr/local/tangrt/
STPU_TANGRT_PATH=/usr/local/tangrt
TANGRT_LIB_PATH=/usr/local/tangrt/lib/linux-x86_64:/usr/lib64
VENDOR_INCLUDE_DIRS=/usr/local/tangrt/include
PTCC_PATH=$SYSTEM_PTCC_PATH
CMAKE_PATH=/usr/local/tangrt/cmake
CMAKE_ROOT=/usr/local/bin/cmake
EOF

    write_file "$CONFIG_DIR/condarc" 0644 <<'EOF'
channels:
  - conda-forge
channel_priority: strict
auto_activate_base: false
show_channel_urls: true
EOF
}

write_proxy_configuration () {
    write_file "$CONFIG_DIR/squid.conf" 0644 <<'EOF'
http_port 127.0.0.1:3128
visible_hostname tilelang-gh-proxy
pid_filename /run/tilelang-gh-proxy/squid.pid
coredump_dir /var/lib/tilelang-gh-proxy
access_log none
cache_store_log none
cache_log stdio:/var/log/tilelang-gh-proxy/cache.log
cache_mem 64 MB
cache deny all

acl local_runner src 127.0.0.1/32
acl CONNECT method CONNECT
acl SSL_ports port 443
acl private_v4 dst 0.0.0.0/8 10.0.0.0/8 100.64.0.0/10 127.0.0.0/8 169.254.0.0/16 172.16.0.0/12 192.0.0.0/24 192.168.0.0/16 198.18.0.0/15 224.0.0.0/4 240.0.0.0/4
acl allowed_domains dstdomain .github.com .githubusercontent.com .actions.githubusercontent.com .blob.core.windows.net repo.anaconda.com conda.anaconda.org api.anaconda.org anaconda.org binstar-cio-packages-prod.s3.amazonaws.com pypi.org files.pythonhosted.org download.pytorch.org download-r2.pytorch.org

http_access deny !local_runner
http_access deny !SSL_ports
http_access deny private_v4
http_access allow local_runner CONNECT allowed_domains
http_access deny all
EOF
}

write_network_helper () {
    write_file "$NETWORK_HELPER" 0755 <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
runner_uid="$(id -u tilelang-gh-runner)"
case "${1:-}" in
    apply)
        /usr/sbin/nft list table inet tilelang_gh_runner >/dev/null 2>&1 && \
            /usr/sbin/nft delete table inet tilelang_gh_runner
        /usr/sbin/nft -f - <<NFT
table inet tilelang_gh_runner {
    chain output {
        type filter hook output priority -50; policy accept;
        meta skuid ${runner_uid} ip daddr 127.0.0.1 tcp dport 3128 accept
        meta skuid ${runner_uid} reject
    }
}
NFT
        ;;
    remove)
        if /usr/sbin/nft list table inet tilelang_gh_runner >/dev/null 2>&1; then
            /usr/sbin/nft delete table inet tilelang_gh_runner
        fi
        ;;
    *)
        echo "usage: $0 {apply|remove}" >&2
        exit 2
        ;;
esac
EOF
}

write_preflight_launcher () {
    write_file "$PREFLIGHT_LAUNCHER" 0755 <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
set -a
source $PREFLIGHT_ENV
set +a
exec "\$SOURCE_DIR/ci/github_runner/preflight.sh"
EOF
}

write_sudoers_policy () {
    local temporary="$ADMIN_TMP/sudoers.$$.${RANDOM}"
    cat > "$temporary" <<EOF
Defaults:$RUNNER_USER secure_path=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin
$RUNNER_USER ALL=(root) NOPASSWD: /usr/bin/pt_smi -r -i 0
EOF
    chmod 0440 "$temporary"
    visudo -cf "$temporary"
    install -o root -g root -m 0440 "$temporary" "$SUDOERS_FILE"
    rm -f "$temporary"
}

write_systemd_units () {
    write_file "/etc/systemd/system/$PROXY_SERVICE" 0644 <<EOF
[Unit]
Description=TileLang-Sunrise GitHub runner HTTPS allowlist proxy
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=squid
Group=squid
RuntimeDirectory=tilelang-gh-proxy
StateDirectory=tilelang-gh-proxy
LogsDirectory=tilelang-gh-proxy
ExecStartPre=/usr/sbin/squid -k parse -f $CONFIG_DIR/squid.conf
ExecStart=/usr/sbin/squid -N -f $CONFIG_DIR/squid.conf
Restart=on-failure
RestartSec=5
NoNewPrivileges=true
PrivateDevices=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true
RestrictAddressFamilies=AF_UNIX AF_INET
LockPersonality=true

[Install]
WantedBy=multi-user.target
EOF

    write_file "/etc/systemd/system/$NETWORK_SERVICE" 0644 <<EOF
[Unit]
Description=TileLang-Sunrise GitHub runner UID egress policy
After=network-online.target $PROXY_SERVICE
Requires=$PROXY_SERVICE
Before=$RUNNER_SERVICE $PREFLIGHT_SERVICE

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=$NETWORK_HELPER apply
ExecStop=$NETWORK_HELPER remove

[Install]
WantedBy=multi-user.target
EOF

    write_file "/etc/systemd/system/$RUNNER_SERVICE" 0644 <<EOF
[Unit]
Description=TileLang-Sunrise repository GitHub Actions runner
After=network-online.target $PROXY_SERVICE $NETWORK_SERVICE
Requires=$PROXY_SERVICE $NETWORK_SERVICE
ConditionPathExists=$RUNNER_DIR/.runner
ConditionPathExists=$PINNED_PTCC_PATH

[Service]
Type=simple
User=$RUNNER_USER
Group=$RUNNER_GROUP
WorkingDirectory=$RUNNER_DIR
EnvironmentFile=$RUNNER_ENV
ExecStart=$RUNNER_DIR/run.sh
Restart=always
RestartSec=10
KillMode=mixed
TimeoutStopSec=30
UMask=0077
PrivateTmp=true
ProtectSystem=strict
ProtectHome=tmpfs
ProtectControlGroups=true
IPAddressDeny=any
IPAddressAllow=127.0.0.1
DevicePolicy=closed
DeviceAllow=/dev/ptpu0 rw
DeviceAllow=/dev/ptpuctrl rw
TemporaryFileSystem=/mnt:ro
BindPaths=$BASE_DIR
BindReadOnlyPaths=$PINNED_PTCC_PATH:$SYSTEM_PTCC_PATH
InaccessiblePaths=-/run/avahi-daemon -/run/cups -/run/dbus -/run/gssproxy.sock -/run/.heim_org.h5l.kcm-socket -/run/irqbalance -/run/lsm -/run/mcelog-client -/run/rpcbind.sock -/run/systemd/io.system.ManagedOOM -/run/systemd/journal -/run/systemd/notify -/run/systemd/userdb /media /srv /root /var/log/pt200 $EVIDENCE_DIR $SOURCE_ROOT
ReadOnlyPaths=$RUNNER_DIR $WHEEL_DIR $TOOLCHAIN_DIR
ReadWritePaths=$RUNNER_DIR/_diag $RUNNER_DIR/run-helper.sh $WORK_DIR $CACHE_DIR $RUNNER_HOME

[Install]
WantedBy=multi-user.target
EOF

    write_file "/etc/systemd/system/$PREFLIGHT_SERVICE" 0644 <<EOF
[Unit]
Description=Pre-registration TileLang-Sunrise CI validation
After=network-online.target $PROXY_SERVICE $NETWORK_SERVICE
Requires=$PROXY_SERVICE $NETWORK_SERVICE
ConditionPathExists=$PREFLIGHT_ENV
ConditionPathExists=$PINNED_PTCC_PATH

[Service]
Type=oneshot
User=$RUNNER_USER
Group=$RUNNER_GROUP
WorkingDirectory=$SOURCE_ROOT
EnvironmentFile=$RUNNER_ENV
ExecStart=$PREFLIGHT_LAUNCHER
KillMode=mixed
TimeoutStopSec=30
UMask=0077
PrivateTmp=true
ProtectSystem=strict
ProtectHome=tmpfs
ProtectControlGroups=true
IPAddressDeny=any
IPAddressAllow=127.0.0.1
DevicePolicy=closed
DeviceAllow=/dev/ptpu0 rw
DeviceAllow=/dev/ptpuctrl rw
TemporaryFileSystem=/mnt:ro
BindPaths=$BASE_DIR
BindReadOnlyPaths=$PINNED_PTCC_PATH:$SYSTEM_PTCC_PATH
InaccessiblePaths=-/run/avahi-daemon -/run/cups -/run/dbus -/run/gssproxy.sock -/run/.heim_org.h5l.kcm-socket -/run/irqbalance -/run/lsm -/run/mcelog-client -/run/rpcbind.sock -/run/systemd/io.system.ManagedOOM -/run/systemd/journal -/run/systemd/notify -/run/systemd/userdb /media /srv /root /var/log/pt200
ReadOnlyPaths=$RUNNER_DIR $WHEEL_DIR $TOOLCHAIN_DIR
ReadWritePaths=$RUNNER_DIR/_diag $WORK_DIR $CACHE_DIR $EVIDENCE_DIR $RUNNER_HOME $SOURCE_ROOT
EOF
}

refresh_units () {
    require_root
    assert_safe_layout
    [[ -d "$BASE_DIR" ]] || die "runner base directory is not provisioned"
    [[ -r "$RUNNER_ENV" ]] || die "runner environment is not provisioned"
    [[ -x "$MINIFORGE_DIR/bin/conda" ]] || die "pinned Miniforge conda executable is missing"
    verify_pinned_ptcc
    ensure_admin_tmp
    restrict_runtime_tree "$MINIFORGE_DIR"
    if [[ -e "$RUNNER_DIR/.runner" ]]; then
        prepare_runner_helper
    fi
    write_runner_environment
    write_preflight_launcher
    write_systemd_units
    systemd-analyze verify "/etc/systemd/system/$PROXY_SERVICE" \
        "/etc/systemd/system/$NETWORK_SERVICE" \
        "/etc/systemd/system/$RUNNER_SERVICE" \
        "/etc/systemd/system/$PREFLIGHT_SERVICE"
    systemctl daemon-reload
    echo "Refreshed systemd units without reprovisioning packages, vendor wheels, or PTCC"
}

provision () {
    require_root
    assert_safe_layout
    [[ -c /dev/ptpu0 && -c /dev/ptpuctrl ]] || die "S2 device 0 is not available"
    [[ -x /usr/bin/pt_smi ]] || die "/usr/bin/pt_smi is unavailable"
    [[ -x "$SYSTEM_PTCC_PATH" ]] || die "PTCC bind-mount target is unavailable"

    dnf install -y squid nftables curl tar gzip git shadow-utils sudo
    create_account_and_directories
    install_runner_distribution
    install -d -o "$RUNNER_USER" -g "$RUNNER_GROUP" -m 0700 "$RUNNER_DIR/_diag"
    install_miniforge_distribution
    stage_vendor_wheel TORCH_PTPU_WHEEL_SOURCE "$TORCH_WHEEL_NAME" "$TORCH_WHEEL_SHA256"
    stage_vendor_wheel TRITON_WHEEL_SOURCE "$TRITON_WHEEL_NAME" "$TRITON_WHEEL_SHA256"
    stage_pinned_ptcc
    write_runner_environment
    write_proxy_configuration
    write_network_helper
    write_preflight_launcher
    write_sudoers_policy
    write_systemd_units

    systemd-analyze verify "/etc/systemd/system/$PROXY_SERVICE" \
        "/etc/systemd/system/$NETWORK_SERVICE" \
        "/etc/systemd/system/$RUNNER_SERVICE" \
        "/etc/systemd/system/$PREFLIGHT_SERVICE"
    systemctl daemon-reload
    systemctl enable --now "$PROXY_SERVICE" "$NETWORK_SERVICE"
    echo "Provisioned host isolation without registering or starting the GitHub runner"
}

load_runner_environment () {
    [[ -r "$RUNNER_ENV" ]] || die "runner environment is not provisioned"
    RUNNER_ENVIRONMENT=()
    local line
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*$ ]] || \
            die "runner environment contains an invalid line"
        RUNNER_ENVIRONMENT+=("$line")
    done < "$RUNNER_ENV"
    [[ ${#RUNNER_ENVIRONMENT[@]} -gt 0 ]] || die "runner environment is empty"
}

run_as_runner () {
    load_runner_environment
    runuser -u "$RUNNER_USER" -- /usr/bin/env -i "${RUNNER_ENVIRONMENT[@]}" "$@"
}

run_as_runner_with_token () {
    local token="$1"
    shift
    load_runner_environment
    printf '%s\n' "$token" | \
        runuser -u "$RUNNER_USER" -- /usr/bin/env -i "${RUNNER_ENVIRONMENT[@]}" \
        /usr/bin/bash -c \
        'IFS= read -r ACTIONS_RUNNER_INPUT_TOKEN; export ACTIONS_RUNNER_INPUT_TOKEN; exec "$@"' \
        tilelang-runner-token "$@"
}

stage_source () {
    require_root
    local bundle="${1:-}" source_sha="${2:-}" base_sha="${3:-}"
    [[ -f "$bundle" ]] || die "bundle does not exist: ${bundle:-<empty>}"
    [[ "$source_sha" =~ ^[0-9a-f]{40}$ ]] || die "SOURCE_SHA must be a full lowercase SHA"
    [[ "$base_sha" =~ ^[0-9a-f]{40}$ ]] || die "BASE_SHA must be a full lowercase SHA"
    local destination="$SOURCE_ROOT/$source_sha"
    if [[ -d "$destination/.git" ]]; then
        [[ "$(git -c safe.directory="$destination" -C "$destination" rev-parse HEAD)" == "$source_sha" ]] || die "existing staged source has the wrong SHA"
    else
        [[ ! -e "$destination" ]] || die "staging destination already exists and is not a Git checkout"
        git clone "$bundle" "$destination"
        git -c safe.directory="$destination" -C "$destination" checkout --detach "$source_sha"
    fi
    git -c safe.directory="$destination" -C "$destination" cat-file -e "${base_sha}^{commit}"
    [[ -z "$(git -c safe.directory="$destination" -C "$destination" status --porcelain)" ]] || die "staged source is not clean"
    chown -R "$RUNNER_USER:$RUNNER_GROUP" "$destination"
    write_file "$PREFLIGHT_ENV" 0644 <<EOF
SOURCE_DIR=$destination
SOURCE_SHA=$source_sha
BASE_SHA=$base_sha
EOF
    echo "Staged $source_sha with base $base_sha at $destination"
}

reset_test () {
    require_root
    [[ -r /proc/pt/ptpu0/state ]] || die "PTPU state file is unavailable"
    grep -Eq '^[[:space:]]*state:[[:space:]]*READY' /proc/pt/ptpu0/state || die "device 0 is not READY"
    grep -Eq '^[[:space:]]*fatal_error:[[:space:]]*0' /proc/pt/ptpu0/state || die "device 0 has a fatal error"
    if compgen -G '/proc/pt/ptpu0/[0-9]*@[0-9]*' >/dev/null; then
        die "device 0 has active clients; reset is forbidden"
    fi
    echo "About to run the only allowed root command: /usr/bin/pt_smi -r -i 0"
    read -r -p "Type RESET-S2-0 to continue: " confirmation
    [[ "$confirmation" == "RESET-S2-0" ]] || die "reset confirmation did not match"
    local evidence="$EVIDENCE_DIR/reset-test-$(date -u +%Y%m%dT%H%M%SZ).log"
    {
        echo "before:"
        cat /proc/pt/ptpu0/state
        run_as_runner /usr/bin/sudo -n /usr/bin/pt_smi -r -i 0
        sleep 5
        echo "after:"
        cat /proc/pt/ptpu0/state
    } | tee "$evidence"
    grep -Eq '^[[:space:]]*state:[[:space:]]*READY' /proc/pt/ptpu0/state || die "device 0 is not READY after reset"
    grep -Eq '^[[:space:]]*fatal_error:[[:space:]]*0' /proc/pt/ptpu0/state || die "device 0 has a fatal error after reset"
    echo "Reset test evidence: $evidence"
}

start_preflight () {
    require_root
    [[ -r "$PREFLIGHT_ENV" ]] || die "stage a source bundle before preflight"
    systemctl stop "$RUNNER_SERVICE" 2>/dev/null || true
    local source_sha
    source_sha=$(awk -F= '$1=="SOURCE_SHA"{print $2}' "$PREFLIGHT_ENV")
    [[ "$source_sha" =~ ^[0-9a-f]{40}$ ]] || die "invalid staged source SHA"
    rm -f "$EVIDENCE_DIR/$source_sha/PREFLIGHT_SUCCESS"
    systemctl reset-failed "$PREFLIGHT_SERVICE" 2>/dev/null || true
    systemctl start --no-block "$PREFLIGHT_SERVICE"
    echo "Preflight started; monitor with: journalctl -fu $PREFLIGHT_SERVICE"
}

require_preflight_pass () {
    [[ -r "$PREFLIGHT_ENV" ]] || die "preflight metadata is missing"
    local source_sha
    source_sha=$(awk -F= '$1=="SOURCE_SHA"{print $2}' "$PREFLIGHT_ENV")
    [[ "$source_sha" =~ ^[0-9a-f]{40}$ ]] || die "invalid staged source SHA"
    local summary="$EVIDENCE_DIR/$source_sha/jobs.tsv"
    local pass_marker="$EVIDENCE_DIR/$source_sha/PREFLIGHT_SUCCESS"
    [[ -f "$summary" ]] || die "preflight summary is missing for $source_sha"
    local job
    for job in lint_changed lint_all tilelang_build tilelang_test tilekernels_sunrise tileops_sunrise; do
        awk -F '\t' -v name="$job" '$1==name && $2=="PASS" && $3==0 {found=1} END {exit !found}' "$summary" || \
            die "preflight job did not pass: $job"
    done
    [[ -f "$pass_marker" ]] || die "preflight success marker is missing for $source_sha"
    [[ "$(cat "$pass_marker")" == "SOURCE_SHA=$source_sha" ]] || die "preflight success marker is invalid"
}

prepare_runner_helper () {
    # The upstream run.sh copies run-helper.sh.template on every startup.
    # Keep the runner distribution read-only except for this one generated
    # helper, which must be writable by the unprivileged runner account.
    local helper="$RUNNER_DIR/run-helper.sh"
    [[ ! -L "$helper" && ! -d "$helper" ]] || die "runner helper path is not a regular file"
    if [[ ! -e "$helper" ]]; then
        install -o "$RUNNER_USER" -g "$RUNNER_GROUP" -m 0700 /dev/null "$helper"
    else
        chown "$RUNNER_USER:$RUNNER_GROUP" "$helper"
        chmod 0700 "$helper"
    fi
    local state_file state_path
    for state_file in .env .path; do
        state_path="$RUNNER_DIR/$state_file"
        [[ ! -L "$state_path" && ! -d "$state_path" ]] || die "runner state path is not a regular file: $state_file"
        if [[ -e "$state_path" ]]; then
            chown "$RUNNER_USER:$RUNNER_GROUP" "$state_path"
            chmod 0600 "$state_path"
        fi
    done
}

lock_runner_distribution () {
    chown -R root:root "$RUNNER_DIR"
    chmod -R go-w "$RUNNER_DIR"
    local path
    for path in .runner .credentials .credentials_rsaparams .service; do
        if [[ -e "$RUNNER_DIR/$path" ]]; then
            chown "$RUNNER_USER:$RUNNER_GROUP" "$RUNNER_DIR/$path"
            chmod 0600 "$RUNNER_DIR/$path"
        fi
    done
    if [[ -d "$RUNNER_DIR/_diag" ]]; then
        chown -R "$RUNNER_USER:$RUNNER_GROUP" "$RUNNER_DIR/_diag"
        chmod 0700 "$RUNNER_DIR/_diag"
    fi
    prepare_runner_helper
}

register_runner () {
    require_root
    require_public_disclosure_approval
    require_preflight_pass
    [[ ! -e "$RUNNER_DIR/.runner" ]] || die "runner is already configured"
    systemctl is-active --quiet "$PROXY_SERVICE" "$NETWORK_SERVICE" || die "proxy and network policy must be active"
    ! systemctl is-active --quiet "$PREFLIGHT_SERVICE" || die "preflight is still running"
    read -r -s -p "GitHub one-time repository runner registration token: " registration_token
    echo
    [[ -n "$registration_token" ]] || die "registration token is empty"
    chown -R "$RUNNER_USER:$RUNNER_GROUP" "$RUNNER_DIR"
    run_as_runner_with_token "$registration_token" \
        "$RUNNER_DIR/config.sh" --unattended \
        --url "$REPOSITORY_URL" \
        --name "$RUNNER_NAME" \
        --labels sunrise-s2,tilelang-sunrise \
        --work "$WORK_DIR" \
        --disableupdate
    registration_token=""
    unset registration_token
    lock_runner_distribution
    echo "Runner registered but not started"
}

unregister_runner () {
    require_root
    [[ -e "$RUNNER_DIR/.runner" ]] || { echo "Runner is not configured"; return 0; }
    systemctl stop "$RUNNER_SERVICE" 2>/dev/null || true
    read -r -s -p "GitHub one-time repository runner removal token: " removal_token
    echo
    [[ -n "$removal_token" ]] || die "removal token is empty"
    chown -R "$RUNNER_USER:$RUNNER_GROUP" "$RUNNER_DIR"
    run_as_runner_with_token "$removal_token" "$RUNNER_DIR/config.sh" remove
    removal_token=""
    unset removal_token
    lock_runner_distribution
    echo "Runner unregistered"
}

start_runner () {
    require_root
    require_public_disclosure_approval
    require_preflight_pass
    [[ -e "$RUNNER_DIR/.runner" ]] || die "register the runner before start"
    systemctl enable --now "$RUNNER_SERVICE"
}

stop_runner () {
    require_root
    systemctl disable --now "$RUNNER_SERVICE" 2>/dev/null || true
}

check_host () {
    require_root
    assert_safe_layout
    echo "host=$(hostname)"
    uname -a
    df -h / /home
    /usr/bin/pt_smi -i 0
    getent passwd "$RUNNER_USER" || true
    [[ ! -e "$RUNNER_DIR/.runner" ]] || echo "runner_registration=present"
    for service in "$PROXY_SERVICE" "$NETWORK_SERVICE" "$RUNNER_SERVICE" "$PREFLIGHT_SERVICE"; do
        echo "$service active=$(systemctl is-active "$service" 2>/dev/null || true) enabled=$(systemctl is-enabled "$service" 2>/dev/null || true)"
    done
    /usr/sbin/nft list table inet tilelang_gh_runner 2>/dev/null || true
    [[ ! -f "$WHEEL_DIR/$TORCH_WHEEL_NAME" ]] || echo "$TORCH_WHEEL_SHA256  $WHEEL_DIR/$TORCH_WHEEL_NAME" | sha256sum -c -
    [[ ! -f "$WHEEL_DIR/$TRITON_WHEEL_NAME" ]] || echo "$TRITON_WHEEL_SHA256  $WHEEL_DIR/$TRITON_WHEEL_NAME" | sha256sum -c -
    verify_pinned_ptcc
    if public_disclosure_is_approved; then
        echo "ptcc_public_disclosure_approval=present"
    else
        echo "ptcc_public_disclosure_approval=absent (GitHub activation blocked)"
    fi
    run_as_runner "$MINIFORGE_DIR/bin/conda" --version
    [[ ! -r "$RUNNER_ENV" ]] || run_as_runner curl --silent --show-error --location --max-time 30 --output /dev/null https://github.com/
}

show_status () {
    require_root
    for service in "$PROXY_SERVICE" "$NETWORK_SERVICE" "$RUNNER_SERVICE" "$PREFLIGHT_SERVICE"; do
        systemctl --no-pager --full status "$service" 2>/dev/null || true
    done
    if [[ -r "$PREFLIGHT_ENV" ]]; then
        local source_sha
        source_sha=$(awk -F= '$1=="SOURCE_SHA"{print $2}' "$PREFLIGHT_ENV")
        [[ ! -f "$EVIDENCE_DIR/$source_sha/jobs.tsv" ]] || cat "$EVIDENCE_DIR/$source_sha/jobs.tsv"
    fi
}

rollback_list () {
    cat <<EOF
Rollback will stop and archive only these targets:
  services: $RUNNER_SERVICE $PREFLIGHT_SERVICE $NETWORK_SERVICE $PROXY_SERVICE
  config:   $CONFIG_DIR
  sudoers:  $SUDOERS_FILE
  helpers:  $NETWORK_HELPER $PREFLIGHT_LAUNCHER
  units:    /etc/systemd/system/$RUNNER_SERVICE
            /etc/systemd/system/$PREFLIGHT_SERVICE
            /etc/systemd/system/$NETWORK_SERVICE
            /etc/systemd/system/$PROXY_SERVICE
  account:  $RUNNER_USER (without recursive home deletion)
  data:     $BASE_DIR (moved to a timestamped sibling archive)
  nftables: table inet tilelang_gh_runner

Installed RPM packages are not removed. Rollback refuses while .runner exists;
run unregister first so GitHub does not retain a stale runner record.
EOF
}

rollback () {
    require_root
    assert_safe_layout
    [[ ! -e "$RUNNER_DIR/.runner" ]] || die "runner is still registered; run unregister first"
    rollback_list
    read -r -p "Type ARCHIVE-RUNNER to continue: " confirmation
    [[ "$confirmation" == "ARCHIVE-RUNNER" ]] || die "rollback confirmation did not match"
    local timestamp archive_root
    timestamp=$(date -u +%Y%m%dT%H%M%SZ)
    archive_root="/var/lib/tilelang-gh-runner-rollback/$timestamp"
    install -d -o root -g root -m 0700 "$archive_root"
    systemctl disable --now "$RUNNER_SERVICE" "$PREFLIGHT_SERVICE" 2>/dev/null || true
    systemctl disable --now "$NETWORK_SERVICE" "$PROXY_SERVICE" 2>/dev/null || true
    [[ ! -x "$NETWORK_HELPER" ]] || "$NETWORK_HELPER" remove
    local path
    for path in \
        "$CONFIG_DIR" \
        "$SUDOERS_FILE" \
        "$NETWORK_HELPER" \
        "$PREFLIGHT_LAUNCHER" \
        "/etc/systemd/system/$RUNNER_SERVICE" \
        "/etc/systemd/system/$PREFLIGHT_SERVICE" \
        "/etc/systemd/system/$NETWORK_SERVICE" \
        "/etc/systemd/system/$PROXY_SERVICE"; do
        [[ ! -e "$path" ]] || mv "$path" "$archive_root/$(basename "$path")"
    done
    if getent passwd "$RUNNER_USER" >/dev/null; then
        userdel "$RUNNER_USER"
    fi
    if [[ -d "$BASE_DIR" ]]; then
        mv "$BASE_DIR" "/home/github-actions/tilelang-sunrise-rollback-$timestamp"
    fi
    systemctl daemon-reload
    echo "Configuration archive: $archive_root"
    echo "Data archive: /home/github-actions/tilelang-sunrise-rollback-$timestamp"
}

main () {
    require_root
    assert_safe_layout
    local action="${1:-}"
    shift || true
    case "$action" in
        check) check_host "$@" ;;
        provision) provision "$@" ;;
        stage-ptcc) stage_ptcc "$@" ;;
        refresh-units) refresh_units "$@" ;;
        stage) stage_source "$@" ;;
        reset-test) reset_test "$@" ;;
        preflight) start_preflight "$@" ;;
        register) register_runner "$@" ;;
        unregister) unregister_runner "$@" ;;
        start) start_runner "$@" ;;
        stop) stop_runner "$@" ;;
        status) show_status "$@" ;;
        rollback-list) rollback_list "$@" ;;
        rollback) rollback "$@" ;;
        *) usage; exit 2 ;;
    esac
}

main "$@"
