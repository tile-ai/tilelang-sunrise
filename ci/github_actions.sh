#!/usr/bin/env bash
# GitHub Actions adapter for the shared GitLab-oriented CI entry points.
set -euo pipefail

command_name="${1:-}"
job_name="${2:-}"

require_github_file () {
    if [[ -z "${GITHUB_EVENT_PATH:-}" || ! -f "$GITHUB_EVENT_PATH" ]]; then
        echo "GITHUB_EVENT_PATH is missing or unreadable" >&2
        exit 2
    fi
}

trust_boundary () {
    require_github_file
    python3 - "$GITHUB_EVENT_PATH" "${GITHUB_EVENT_NAME:-}" "${GITHUB_REPOSITORY:-}" <<'PY'
import json
import sys

event_path, event_name, repository = sys.argv[1:]
with open(event_path, encoding="utf-8") as source:
    event = json.load(source)

event_repository = event.get("repository", {}).get("full_name", "")
if event_name not in {"push", "workflow_dispatch"}:
    raise SystemExit(f"refusing untrusted GitHub event: {event_name or '<empty>'}")
if not repository or event_repository != repository:
    raise SystemExit(
        f"refusing repository mismatch: event={event_repository or '<empty>'} expected={repository or '<empty>'}"
    )
print(f"trusted GitHub event: {event_name} in {repository}")
PY
}

export_environment () {
    require_github_file
    if [[ -z "$job_name" ]]; then
        echo "job name is required for the export command" >&2
        exit 2
    fi
    if [[ -z "${GITHUB_ENV:-}" ]]; then
        echo "GITHUB_ENV is required for the export command" >&2
        exit 2
    fi
    python3 - "$GITHUB_EVENT_PATH" "$GITHUB_ENV" "$job_name" <<'PY'
import json
import os
import subprocess
import sys

event_path, output_path, job_name = sys.argv[1:]
with open(event_path, encoding="utf-8") as source:
    event = json.load(source)

event_name = os.environ.get("GITHUB_EVENT_NAME", "")
if event_name not in {"push", "workflow_dispatch"}:
    raise SystemExit(f"unsupported GitHub event for CI export: {event_name or '<empty>'}")

repository = event.get("repository", {})
default_branch = repository.get("default_branch") or "main"
before_sha = event.get("before") or "0" * 40
diff_base = ""
if event_name == "push" and os.environ.get("GITHUB_REF") != f"refs/heads/{default_branch}":
    try:
        diff_base = subprocess.check_output(
            ["git", "merge-base", f"origin/{default_branch}", "HEAD"],
            text=True,
        ).strip()
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"cannot resolve diff base against origin/{default_branch}") from error

run_id = os.environ.get("GITHUB_RUN_ID", "")
run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "1")
workspace = os.environ.get("GITHUB_WORKSPACE", "")
values = {
    "CI_PROJECT_DIR": workspace,
    "CI_PROJECT_ID": os.environ.get("GITHUB_REPOSITORY_ID", ""),
    "CI_PIPELINE_ID": run_id,
    "CI_JOB_ID": f"{run_id}-{run_attempt}-{job_name}",
    "CI_JOB_NAME": job_name,
    "CI_COMMIT_SHA": os.environ.get("GITHUB_SHA", ""),
    "CI_COMMIT_BEFORE_SHA": before_sha,
    "CI_MERGE_REQUEST_DIFF_BASE_SHA": diff_base,
    "CI_MERGE_REQUEST_IID": "",
    "CI_DEFAULT_BRANCH": default_branch,
    "CI_PIPELINE_SOURCE": "push" if event_name == "push" else "web",
    "CI_FAILURE_REPORT_DIR": os.path.join(workspace, "ci_failure_reports"),
    "CASE_REPEAT": "3",
    "TANG_VISIBLE_DEVICES": "0",
    "TILELANG_DEFAULT_TARGET": "tang",
    "TILELANG_CI_RESET_MODE": "sudo-n",
    "TILELANG_CI_PUBLIC_LOGS": "1",
}

for key, value in values.items():
    if "\n" in value or "\r" in value:
        raise SystemExit(f"refusing newline in GitHub environment value {key}")
with open(output_path, "a", encoding="utf-8") as output:
    for key, value in values.items():
        output.write(f"{key}={value}\n")
PY
}

build_wheels () {
    if [[ "${GITHUB_REF:-}" != "refs/heads/main" ]]; then
        bash ci/build.sh
        return
    fi

    local commit
    commit="$(git rev-parse HEAD)"
    if [[ "$commit" != "${GITHUB_SHA:-}" ]]; then
        echo "checkout does not match GITHUB_SHA for the formal wheel build" >&2
        return 2
    fi

    NO_VERSION_LABEL=1 bash ci/build.sh
    python3 -I - "$commit" <<'PY'
import subprocess
import sys
from email import message_from_bytes
from pathlib import Path
from zipfile import ZipFile

commit = sys.argv[1]
expected = subprocess.check_output(["git", "show", f"{commit}:VERSION"], text=True).strip()
wheels = list(Path("dist").glob("tilelang_sunrise-*.whl"))
if len(wheels) != 1:
    raise SystemExit(f"expected one TileLang-Sunrise wheel, found {len(wheels)}")
with ZipFile(wheels[0]) as archive:
    entries = [entry for entry in archive.namelist() if entry.endswith(".dist-info/METADATA")]
    if len(entries) != 1:
        raise SystemExit("expected one METADATA entry in the TileLang-Sunrise wheel")
    metadata = message_from_bytes(archive.read(entries[0]))
names = metadata.get_all("Name")
versions = metadata.get_all("Version")
if names != ["tilelang-sunrise"] or versions != [expected]:
    raise SystemExit(f"expected TileLang-Sunrise wheel version {expected}, got Name={names}, Version={versions}")
print(f"Verified TileLang-Sunrise wheel version {expected} from {commit}")
PY
}

case "$command_name" in
    trust) trust_boundary ;;
    export) export_environment ;;
    build) build_wheels ;;
    *)
        echo "usage: $0 {trust|export JOB_NAME|build}" >&2
        exit 2
        ;;
esac
