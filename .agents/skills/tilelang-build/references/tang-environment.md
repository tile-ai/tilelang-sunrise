# TANG/PTPU Build Environment

Use this reference when building or testing TileLang for TANG/PTPU.

## Source Of Truth Scripts

Inspect these repository entry points before copying commands or package pins:

- `ci/build.sh`: backend build entry point
- `ci/test.sh`: test entry point
- `ci/lib.sh`: shared build, runtime, cache, and test helpers
- `ci/s3/lib_s3.sh` and `ci/s3/test_s3.sh`: S3-specific overrides

Version pins and environment names may drift; use the files in the current checkout. Vendor wheel
locations and host toolchain paths must be supplied explicitly and must not be committed to the
repository.

## Common Environment Variables

- Expose one assigned device with `TANG_VISIBLE_DEVICES`; if it is unset, stop and ask which device
  to use.
- Run one TANG hardware process at a time.
- Build with `USE_TANG=ON` or `USE_TANG=1`.
- Give correctness work a task-specific `TILELANG_CACHE_DIR`. For codegen, JIT, ABI, or generated
  kernel changes, run at least one focused check with `TILELANG_DISABLE_CACHE=1`.

## Environment Checks

Verify Python and driver state:

```bash
python -c "import torch; import torch_ptpu; print(torch.ptpu.is_available())"
ls -la /dev/tang*
```

Import `torch_ptpu` before accessing `torch.ptpu`; it registers the private-use backend. If a
sandbox alone reports missing TANG devices or runtime initialization failure, repeat the preflight
outside the sandbox before classifying the hardware as unavailable.

Record the source SHA, build directory and CMake options, Python executable, `tilelang.__file__`,
`tvm.__file__`, installed package version, and relevant runtime/compiler/driver versions together.

## Debugging Generated TANG Code

Useful locations:

- TANG codegen implementation: `src/tang/codegen/codegen_tang.cc`
- TANG runtime module: `src/tang/codegen/rt_mod_tang.cc`
- TANG templates: `src/tl_templates/tang/`

Do not delete the shared default cache as a routine diagnostic. Use an isolated cache or disable the
cache for a focused rerun; ask before deleting user caches or build trees.
