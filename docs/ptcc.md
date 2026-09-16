# Selecting PTCC for S2

LLVM20 PTCC (for example 2.2.9) remains the default for daily S2 use. LLVM22 is
an opt-in compatibility-validation configuration, not a performance upgrade.
For an LLVM22 validation run, set these variables before starting Python or the
unified scripts:

```bash
export PTCC_PATH=/path/to/llvm22/bin/ptcc
export PTCC_JIT_PROFILE=llvm22
bash ci/install.sh
bash ci/run.sh --case examples/elementwise/test_example_elementwise.py
```

Use an existing PTCC installation. `ci/install.sh` builds and installs TileLang,
not PTCC.

Relative paths are resolved against the invocation directory. An invalid explicit
path is an error. Without an override, discovery checks `PATH`, then `bin/ptcc`
and `toolchains/llvm/prebuilt/linux-<machine>/bin/ptcc` under the TANG toolkit.
Existing toolkit configuration is unchanged: Python uses `TANG_HOME`/`TANG_PATH`
and the CI build scripts use `TANGRT_PATH`. `PTCC_PATH` selects only the compiler,
not runtime libraries or toolkit headers.

`PTCC_JIT_PROFILE` selects `llvm20` (default) or `llvm22` JIT options explicitly;
it neither selects nor detects the compiler. Set it to match the chosen PTCC.
LLVM20 keeps the existing options. LLVM22 uses a separate compatibility
configuration; the remaining optimization options are unchanged. Their defaults
live in `tilelang/_ptcc.py` and can be adjusted independently without duplicating
the TVM-FFI and library-generator compilation paths. Existing per-kernel flag
overrides remain at their callers; this profile does not change host/CMake flags.
Effective default options also participate in the kernel cache key, so future
profile changes cannot reuse binaries compiled with different defaults.

S2 device compilation uses `--tang-gpu-arch=stcu`, including with compilers that
default to S3. The PTCC executable's real path and SHA256 participate in TANG
kernel cache keys. Restart Python after changing the compiler path or replacing
the toolchain. Do not replace files in the maintainer-managed directory. The
fingerprint does not cover headers or device libraries: clear the kernel cache
if those change without replacing PTCC. S3 ISS retains its separate
`TANG_S3_PTCC_PATH` interface.

For GitLab, ensure the managed compiler path is available on each runner and set
`PTCC_PATH` and `PTCC_JIT_PROFILE=llvm22` only for explicit LLVM22 validation
pipelines. Keep the daily LLVM20 configuration unchanged. Build and test jobs
must receive the same selection. The scripts log the selected
compiler version and path; they do not download or install PTCC. Record the
compiler SHA256 and previous variable value before rollout. Restore that value
(or remove the override) to roll back. GPU selection remains runner-managed.
