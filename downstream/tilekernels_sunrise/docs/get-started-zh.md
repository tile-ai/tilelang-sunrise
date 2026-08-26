# TileKernels 快速上手

本指南带你从零搭建 TileKernels：安装依赖、编译工具链（TileLang + 内置 TVM）、运行测试、跑性能测试和压力测试。

项目整体介绍请见 [README.md](./README.md)。

______________________________________________________________________

## 1. 环境要求

| 项目         | 版本 / 说明                                         |
| ------------ | --------------------------------------------------- |
| 操作系统     | Linux x86_64                                        |
| Python       | 3.10（由下文的 conda 自动管理）                     |
| PyTorch      | 2.10 及以上                                         |
| TileLang     | 0.1.9 及以上（安装脚本会自动从源码编译）            |
| GPU          | NVIDIA SM90 / SM100，或 Sunrise Tang 加速器         |
| CUDA Toolkit | 13.1 及以上（CUDA 路线）/ Tang Runtime（Tang 路线） |
| 磁盘空间     | 约 10 GB（conda 环境 + TVM 构建 + wheel）           |

如使用 Tang 后端，脚本默认期望以下路径：

- `ptcc` 位于 `/usr/local/tangrt/toolchains/llvm/prebuilt/linux-x86_64/bin/ptcc`
- Tang runtime 位于 `/usr/local/tangrt/`

所有路径均可通过环境变量覆盖，详见下文 [工具链路径覆盖](#%E5%B7%A5%E5%85%B7%E9%93%BE%E8%B7%AF%E5%BE%84%E8%A6%86%E7%9B%96)。

______________________________________________________________________

## 2. 一键安装

最快的方式是使用仓库自带的安装脚本。它会依次：

1. 若没有 `conda`，自动安装 Miniconda3
1. 创建或复用 conda 环境（默认名 `tilekernels_dev_tang_ci`）
1. 从 `requirements.txt` 安装 Python 依赖
1. 克隆并编译 `tilelang`（含内置的 `TVM` 和 `tvm-ffi`）
1. 以 editable 方式安装 TileKernels（包含 `[dev]` 依赖）
1. 依次执行：功能冒烟测试 → 性能测试 → 压力测试

```bash
git clone <repo-url> tilekernels
cd tilekernels
bash ./one_click_install.sh
```

成功结束时会输出：

```
[INFO] All phases finished successfully
```

### 阶段开关

每个阶段都有跳过开关，方便迭代调试：

| 变量                         | 作用                                          |
| ---------------------------- | --------------------------------------------- |
| `SKIP_CONDA_INSTALL=1`       | 跳过 Miniconda 引导安装                       |
| `SKIP_DEPS_INSTALL=1`        | 跳过 `pip install -r requirements.txt`        |
| `SKIP_TILELANG_CLONE=1`      | 复用已有的 `./tilelang/` 检出                 |
| `SKIP_TVM_BUILD=1`           | TVM 增量编译（不清空 build 目录）             |
| `SKIP_TILELANG_INSTALL=1`    | 不重装 tilelang Python 包                     |
| `SKIP_TILEKERNELS_INSTALL=1` | 不重装 tile-kernels                           |
| `SKIP_TESTS=1`               | 跳过 Phase 5（功能测试）                      |
| `SKIP_BENCHMARK=1`           | 跳过 Phase 6（性能测试）                      |
| `SKIP_STRESS=1`              | 跳过 Phase 7（压力测试）                      |
| `CLEAN_CACHE=1`              | 执行 `conda clean --all` 与 `pip cache purge` |

构建性能：

| 变量           | 默认值     | 作用                        |
| -------------- | ---------- | --------------------------- |
| `MAX_JOBS`     | `$(nproc)` | cmake/ninja 的并行度        |
| `USE_CCACHE=1` | 关闭       | 启用 ccache（前提是已安装） |

### 工具链路径覆盖

```bash
LLVM_HOME=/opt/llvm-20 \
PTCC_PATH=/opt/tang/bin/ptcc \
TANGRT_PATH=/opt/tang \
CONDA_ENV=my_env \
PYTHON_VERSION=3.10 \
bash ./one_click_install.sh
```

______________________________________________________________________

## 3. 手动安装（不使用一键脚本）

如果你已经有可用的 Python 环境且 tilelang 已编译好：

```bash
# 在已激活的 python>=3.10 环境内执行
pip install -e ".[dev]"
```

安装已发布的 release 版本：

```bash
pip install tile-kernels
```

______________________________________________________________________

## 4. 运行测试

`tests/` 目录按 kernel 类别组织：

```
tests/
├── moe/        # 路由 / 门控
├── quant/      # FP8/FP4/E5M6 量化
├── transpose/  # 批量转置
├── engram/     # Engram gate
└── mhc/        # Manifold HyperConnection
```

### 功能测试

```bash
# 单文件，4 个并行 worker
pytest tests/transpose/test_transpose.py -n 4

# 全部用例
pytest tests -n auto
```

通过一键脚本：

```bash
TEST_TARGETS="tests/moe tests/quant" \
PYTEST_ARGS="-x -k topk" \
SKIP_BENCHMARK=1 SKIP_STRESS=1 \
bash ./one_click_install.sh
```

### 性能测试 (Benchmark)

仓库自带性能测试插件 `tests/pytest_benchmark_plugin.py`，能力：

- `@pytest.mark.benchmark` 标记的用例默认跳过，需加 `--run-benchmark` 才会执行
- 通过 `--benchmark-output` 把结果写成 JSONL
- 与 `tests/benchmark_baselines.jsonl` 对比，超出 `--benchmark-regression-threshold` 即判为回归并以非零退出码失败

直接使用 pytest：

```bash
pytest tests/transpose/test_transpose.py \
    --run-benchmark \
    --benchmark-output bench.jsonl \
    --benchmark-regression-threshold 0.1 \
    --benchmark-verbose
```

通过一键脚本（Phase 6）：

```bash
SKIP_TESTS=1 SKIP_STRESS=1 \
BENCHMARK_TARGETS="tests" \
BENCHMARK_REGRESSION_THRESHOLD=0.1 \
BENCHMARK_VERBOSE=1 \
BENCHMARK_OUTPUT=bench.jsonl \
bash ./one_click_install.sh
```

| 变量                             | 默认值                             |
| -------------------------------- | ---------------------------------- |
| `BENCHMARK_TARGETS`              | `tests`                            |
| `BENCHMARK_REGRESSION_THRESHOLD` | `0.1`（即 10%）                    |
| `BENCHMARK_OUTPUT`               | `benchmark_results_<时间戳>.jsonl` |
| `BENCHMARK_VERBOSE=1`            | 等价于追加 `--benchmark-verbose`   |
| `BENCHMARK_PYTEST_ARGS`          | 额外 pytest 参数                   |
| `BENCHMARK_ALLOW_FAILURE=1`      | 性能回归不让脚本失败               |

### 压力测试 (Stress)

压力测试将每条用例**重复 N 次**，并通过 xdist **多 worker 并行**执行，用以暴露 flaky、竞态、内存泄漏等问题（依赖 `pytest-repeat` + `pytest-xdist`，已在 `[dev]` 中）。

直接使用 pytest：

```bash
TK_FULL_TEST=1 pytest tests -n 4 --count 50
```

通过一键脚本（Phase 7）：

```bash
SKIP_TESTS=1 SKIP_BENCHMARK=1 \
STRESS_TARGETS="tests/moe" \
STRESS_COUNT=100 \
STRESS_PARALLEL=4 \
STRESS_TIMEOUT=7200 \
bash ./one_click_install.sh
```

| 变量                     | 默认值                        |
| ------------------------ | ----------------------------- |
| `STRESS_TARGETS`         | `tests`                       |
| `STRESS_COUNT`           | `50`                          |
| `STRESS_PARALLEL`        | `auto`                        |
| `STRESS_TIMEOUT`         | `3600` 秒                     |
| `STRESS_LOG`             | `stress_results_<时间戳>.log` |
| `STRESS_PYTEST_ARGS`     | 额外 pytest 参数              |
| `STRESS_ALLOW_FAILURE=1` | 压测失败不让脚本失败          |

启动前会校验 `pytest-repeat` / `pytest-xdist` 是否安装，缺失会立即报错。

______________________________________________________________________

## 5. 常用命令速查

| 目标                     | 命令                                                                 |
| ------------------------ | -------------------------------------------------------------------- |
| 全量干净安装并跑全部测试 | `bash ./one_click_install.sh`                                        |
| 迭代代码，跳过重复构建   | `SKIP_TILELANG_CLONE=1 SKIP_TVM_BUILD=1 bash ./one_click_install.sh` |
| 仅跑功能测试             | `SKIP_BENCHMARK=1 SKIP_STRESS=1 bash ./one_click_install.sh`         |
| 仅跑性能测试             | `SKIP_TESTS=1 SKIP_STRESS=1 bash ./one_click_install.sh`             |
| 仅跑压力测试             | `SKIP_TESTS=1 SKIP_BENCHMARK=1 bash ./one_click_install.sh`          |
| CI 风格、fail-fast       | `PYTEST_ARGS="-x" bash ./one_click_install.sh`                       |

______________________________________________________________________

## 6. 故障排查

**安装完 conda 后 `conda: command not found`**
重新加载 shell 配置：`source ~/miniconda3/etc/profile.d/conda.sh`。

**TVM 编译报 `BACKTRACE_ELF_SIZE` 错误**
一键脚本会自动给 `3rdparty/libbacktrace/elf.c` 打补丁。如果你手动构建 TVM，需在相关宏检查附近加入 `#define BACKTRACE_ELF_SIZE 64`。

**报错 `tvm-ffi directory not found`**
请使用内置依赖的 TileLang-Sunrise 源码树，并确认
`3rdparty/tvm_sunrise/3rdparty/tvm-ffi/` 目录存在。

**第一次跑 benchmark 就退出非零**
原因通常是没有 baseline 或阈值太严格。可以先用 `BENCHMARK_ALLOW_FAILURE=1` 跑一遍，把生成的 `benchmark_results_*.jsonl` 提交为新的 baseline。

**压力测试超时**
调大 `STRESS_TIMEOUT`，或用 `STRESS_TARGETS` 缩小范围。

**`ptcc` / `tangrt` 路径找不到**
通过环境变量覆盖：`PTCC_PATH=...`、`TANGRT_PATH=...`、`LLVM_HOME=...`。

______________________________________________________________________

## 7. 后续阅读

- 阅读 [`README.md`](./README.md) 了解 kernel 列表与项目特性
- 浏览 `tile_kernels/` 查看 kernel 实现
- 浏览 `tests/` 了解每个 kernel 的典型用法
