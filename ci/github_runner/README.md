# Sunrise S2 GitHub runner operations

This directory contains the public, secret-free host contract for the dedicated
`tile-ai/tilelang-sunrise` repository runner. The runner is not a general host
service: it has one S2 device, one repository, one exact reset command, and a
localhost HTTPS allowlist proxy.

The root operator must inspect every command before running it. Do not put a
sudo password, GitHub PAT, runner token, or private package URL in this
repository. `register` and `unregister` read one-time GitHub tokens without
echoing or persisting them.

## Provision without registering

Run on the intended S2 host from a clean checkout. Supply the two vendor wheels
as existing local paths or HTTPS URLs, and supply the audited PTCC binary as an
existing absolute local path. The PTCC input deliberately cannot be a URL:

```bash
sudo env \
  TORCH_PTPU_WHEEL_SOURCE=/secure/staging/torch_ptpu.whl \
  TRITON_WHEEL_SOURCE=/secure/staging/triton.whl \
  PTCC_BINARY_SOURCE=/secure/staging/ptcc-2.2.9-ceb3571d0 \
  bash ci/github_runner/manage.sh provision
sudo bash ci/github_runner/manage.sh check
```

`provision` installs the pinned runner and Miniforge distributions, creates the
dedicated account, stages checksum-verified wheels and PTCC 2.2.9
(`ceb3571d0`), and starts only the proxy and UID network policy. The PTCC binary
is root-owned and read-only. The runner and preflight services bind it over the
usual PTCC path inside their private mount namespaces, leaving the host's
system compiler unchanged. `provision` does not contact the GitHub runner
registration API.

After an audited management-script update, refresh only the systemd launcher
and units, the secret-free runtime environment, and root-owned Miniforge access
without downloading packages or restaging vendor artifacts. This command also
fails closed unless the pinned PTCC checksum and version still match:

```bash
sudo bash ci/github_runner/manage.sh stage-ptcc \
  /secure/staging/ptcc-2.2.9-ceb3571d0
sudo bash ci/github_runner/manage.sh refresh-units
```

`stage-ptcc` is intended for an already provisioned host. It accepts one local
absolute path, verifies the pinned binary before executing it, and changes only
the runner's isolated, root-owned toolchain copy. It does not replace the host
compiler used by other accounts.

Miniforge is owned by root, readable and executable only by the dedicated
runner group, and exposed through both `PATH` and an explicit `CONDA_EXE`.
Shared CI keeps its existing PATH and shell-initialization discovery first, so
the explicit executable is a GitHub host fallback rather than a GitLab-specific
execution path.

The active workflows intentionally have no `pull_request` trigger. Same-repo
feature branches are checked on `push`; fork pull requests do not run base-repo
Actions. Do not approve a fork workflow while this repository-scoped runner is
registered: GitHub approvals do not prevent a fork from changing runner labels.

## Stage and run the pre-registration validation

Create a bundle containing the exact source and base commits, transfer it to a
root-readable staging path on the host, then run:

```bash
sudo bash ci/github_runner/manage.sh stage \
  /secure/staging/tilelang-sunrise-ci.bundle SOURCE_SHA BASE_SHA
sudo bash ci/github_runner/manage.sh reset-test
sudo bash ci/github_runner/manage.sh preflight
sudo bash ci/github_runner/manage.sh status
```

The reset test requires typing `RESET-S2-0`. The preflight service uses the same
account, proxy, nftables policy, filesystem restrictions, wheel paths, and S2
device as the final runner. Evidence is stored under
`/home/github-actions/tilelang-sunrise/evidence/SOURCE_SHA/`.

The hardware services deliberately omit systemd settings that implicitly set
`NoNewPrivileges=yes` for a `User=` service. On systemd 252 this includes
`ProtectKernel*`, `ProtectHostname`, `ProtectClock`,
`RestrictAddressFamilies`, and `LockPersonality`; any of them would disable the
single setuid transition needed by the exact `sudo -n /usr/bin/pt_smi -r -i 0`
policy. The services retain read-only filesystem protection, explicit writable
paths, device cgroup restrictions, localhost-only systemd IP policy, the UID
nftables policy, inaccessible host service sockets, and the allowlist proxy.
Preflight also verifies the effective
`NoNewPrivs` value from `/proc/self/status` before accepting the sudo policy.

## Register only after preflight passes

Pre-registration debugging and the complete preflight are allowed without a
public-disclosure decision. `register` and `start`, however, fail closed unless
the repository administrator has explicitly approved public PTCC disclosure
and root has created
`/etc/tilelang-gh-runner/ptcc-public-disclosure-approved` as a non-symlink,
root-owned `0600` file containing exactly the repository name and pinned PTCC
SHA required by `manage.sh`. The lifecycle manager never creates this approval
file. Do not create it based only on a successful preflight.

Obtain a one-time repository runner registration token from GitHub, then run:

```bash
sudo bash ci/github_runner/manage.sh register
sudo bash ci/github_runner/manage.sh start
sudo bash ci/github_runner/manage.sh status
```

Before rollback, obtain a one-time removal token and run `unregister`. Always
inspect `rollback-list` before `rollback`; rollback archives the runner data and
specific configuration files instead of recursively deleting them.
