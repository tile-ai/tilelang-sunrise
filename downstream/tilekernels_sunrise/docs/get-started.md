# Get Started with TileKernels

This guide walks you through setting up TileKernels from scratch, building the
required toolchain (TileLang + bundled TVM), running the test suite, and
performing benchmarks and stress tests.

For a high-level project overview, see [README.md](./README.md).

______________________________________________________________________

## 1. Prerequisites

| Item         | Version / Note                                   |
| ------------ | ------------------------------------------------ |
| OS           | Linux x86_64                                     |
| Python       | 3.10 (managed by conda below)                    |
| PyTorch      | 2.10+                                            |
| TileLang     | 0.1.9+ (built from source by the install script) |
| GPU          | NVIDIA SM90 / SM100, or Sunrise Tang accelerator |
| CUDA Toolkit | 13.1+ (CUDA path) / Tang Runtime (Tang path)     |
| Disk         | ~10 GB free for conda env, TVM build, wheels     |

For the Tang backend, the script expects:

- `ptcc` at `/usr/local/tangrt/toolchains/llvm/prebuilt/linux-x86_64/bin/ptcc`
- Tang runtime at `/usr/local/tangrt/`

All paths can be overridden via environment variables; see
[Toolchain overrides](#toolchain-overrides) below.

______________________________________________________________________

## 2. One-click installation

The fastest path is the bundled installer. It will:

1. Bootstrap Miniconda3 if `conda` is not available.
1. Create / reuse a conda env (`tilekernels_dev` by default).
1. Install Python deps from `requirements.txt`.
1. Clone & build `tilelang` (with the bundled `TVM` and `tvm-ffi`).
1. Install TileKernels in editable mode with `[dev]` extras.
1. Run a smoke test, then benchmarks, then stress tests.

```bash
git clone <repo-url> tilekernels
cd tilekernels
bash ./one_click_install.sh
```

A successful run ends with:

```
[INFO] All phases finished successfully
```

### Phase toggles

Every phase has a skip switch so you can iterate quickly:

| Variable                     | Effect                                         |
| ---------------------------- | ---------------------------------------------- |
| `SKIP_CONDA_INSTALL=1`       | Don't bootstrap Miniconda                      |
| `SKIP_DEPS_INSTALL=1`        | Skip `pip install -r requirements.txt`         |
| `SKIP_TILELANG_CLONE=1`      | Reuse existing `./tilelang/` checkout          |
| `SKIP_TVM_BUILD=1`           | Incremental TVM build instead of clean rebuild |
| `SKIP_TILELANG_INSTALL=1`    | Don't reinstall tilelang Python package        |
| `SKIP_TILEKERNELS_INSTALL=1` | Don't reinstall tile-kernels                   |
| `SKIP_TESTS=1`               | Skip Phase 5 (functional pytest)               |
| `SKIP_BENCHMARK=1`           | Skip Phase 6 (benchmark)                       |
| `SKIP_STRESS=1`              | Skip Phase 7 (stress)                          |
| `CLEAN_CACHE=1`              | Run `conda clean --all` and `pip cache purge`  |

Build performance:

| Variable       | Default    | Effect                      |
| -------------- | ---------- | --------------------------- |
| `MAX_JOBS`     | `$(nproc)` | Parallelism for cmake/ninja |
| `USE_CCACHE=1` | off        | Enable ccache if installed  |

### Toolchain overrides

```bash
LLVM_HOME=/opt/llvm-20 \
PTCC_PATH=/opt/tang/bin/ptcc \
TANGRT_PATH=/opt/tang \
CONDA_ENV=my_env \
PYTHON_VERSION=3.10 \
bash ./one_click_install.sh
```

______________________________________________________________________

## 3. Manual installation (without the installer)

If you already have a working Python env and tilelang built:

```bash
# inside an activated python>=3.10 env
pip install -e ".[dev]"
```

To install a published release:

```bash
pip install tile-kernels
```

______________________________________________________________________

## 4. Running tests

The `tests/` tree is organized by kernel family:

```
tests/
├── moe/        # routing / gating
├── quant/      # FP8/FP4/E5M6
├── transpose/  # batched transpose
├── engram/     # engram gate
└── mhc/        # manifold hyper-connection
```

### Functional tests

```bash
# single file, 4 parallel workers
pytest tests/transpose/test_transpose.py -n 4

# whole suite
pytest tests -n auto
```

Via the installer:

```bash
TEST_TARGETS="tests/moe tests/quant" \
PYTEST_ARGS="-x -k topk" \
SKIP_BENCHMARK=1 SKIP_STRESS=1 \
bash ./one_click_install.sh
```

### Benchmarks

The repo ships a benchmark plugin (`tests/pytest_benchmark_plugin.py`) that:

- Skips `@pytest.mark.benchmark` tests unless `--run-benchmark` is passed.
- Writes results to a JSONL file via `--benchmark-output`.
- Compares against `tests/benchmark_baselines.jsonl` and fails the run if any
  result regresses by more than `--benchmark-regression-threshold`.

Direct pytest:

```bash
pytest tests/transpose/test_transpose.py \
    --run-benchmark \
    --benchmark-output bench.jsonl \
    --benchmark-regression-threshold 0.1 \
    --benchmark-verbose
```

Via the installer (Phase 6):

```bash
SKIP_TESTS=1 SKIP_STRESS=1 \
BENCHMARK_TARGETS="tests" \
BENCHMARK_REGRESSION_THRESHOLD=0.1 \
BENCHMARK_VERBOSE=1 \
BENCHMARK_OUTPUT=bench.jsonl \
bash ./one_click_install.sh
```

| Variable                         | Default                               |
| -------------------------------- | ------------------------------------- |
| `BENCHMARK_TARGETS`              | `tests`                               |
| `BENCHMARK_REGRESSION_THRESHOLD` | `0.1` (10%)                           |
| `BENCHMARK_OUTPUT`               | `benchmark_results_<timestamp>.jsonl` |
| `BENCHMARK_VERBOSE=1`            | adds `--benchmark-verbose`            |
| `BENCHMARK_PYTEST_ARGS`          | extra pytest args                     |
| `BENCHMARK_ALLOW_FAILURE=1`      | never fail on regressions             |

### Stress tests

Stress mode repeats every selected test `N` times across multiple xdist
workers (uses `pytest-repeat` + `pytest-xdist`, both already in `[dev]`):

```bash
# direct pytest
TK_FULL_TEST=1 pytest tests -n 4 --count 50

# via the installer (Phase 7)
SKIP_TESTS=1 SKIP_BENCHMARK=1 \
STRESS_TARGETS="tests/moe" \
STRESS_COUNT=100 \
STRESS_PARALLEL=4 \
STRESS_TIMEOUT=7200 \
bash ./one_click_install.sh
```

| Variable                 | Default                          |
| ------------------------ | -------------------------------- |
| `STRESS_TARGETS`         | `tests`                          |
| `STRESS_COUNT`           | `50`                             |
| `STRESS_PARALLEL`        | `auto`                           |
| `STRESS_TIMEOUT`         | `3600` seconds                   |
| `STRESS_LOG`             | `stress_results_<timestamp>.log` |
| `STRESS_PYTEST_ARGS`     | extra pytest args                |
| `STRESS_ALLOW_FAILURE=1` | never fail on stress failures    |

The phase fails fast if `pytest-repeat` or `pytest-xdist` is missing.

______________________________________________________________________

## 5. Quick recipes

| Goal                           | Command                                                              |
| ------------------------------ | -------------------------------------------------------------------- |
| Full clean install + all tests | `bash ./one_click_install.sh`                                        |
| Iterate on code, skip rebuilds | `SKIP_TILELANG_CLONE=1 SKIP_TVM_BUILD=1 bash ./one_click_install.sh` |
| Only functional tests          | `SKIP_BENCHMARK=1 SKIP_STRESS=1 bash ./one_click_install.sh`         |
| Only benchmarks                | `SKIP_TESTS=1 SKIP_STRESS=1 bash ./one_click_install.sh`             |
| Only stress                    | `SKIP_TESTS=1 SKIP_BENCHMARK=1 bash ./one_click_install.sh`          |
| CI-style, fail-fast            | `PYTEST_ARGS="-x" bash ./one_click_install.sh`                       |

______________________________________________________________________

## 6. Troubleshooting

**`conda: command not found` after install**
Re-source your shell: `source ~/miniconda3/etc/profile.d/conda.sh`.

**TVM build fails on `BACKTRACE_ELF_SIZE`**
The installer auto-patches `3rdparty/libbacktrace/elf.c`. If you build TVM
manually, add `#define BACKTRACE_ELF_SIZE 64` near the related macro check.

**`tvm-ffi directory not found`**
Use the bundled TileLang-Sunrise source tree and verify that
`3rdparty/tvm_sunrise/3rdparty/tvm-ffi/` is present.

**Benchmark exits non-zero on first run**
There is no baseline yet, or the threshold is too tight. Either set
`BENCHMARK_ALLOW_FAILURE=1` for the first run, or commit the new
`benchmark_results_*.jsonl` as a baseline.

**Stress test times out**
Raise `STRESS_TIMEOUT`, or narrow the scope with `STRESS_TARGETS`.

______________________________________________________________________

## 7. Next steps

- Read [`README.md`](./README.md) for the kernel catalogue.
- Browse `tile_kernels/` to see kernel implementations.
- Check `tests/` for usage patterns of each kernel.
