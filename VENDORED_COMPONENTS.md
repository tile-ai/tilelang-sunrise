# Vendored Components

This repository contains source snapshots of the components below so that the
TileLang-Sunrise source tree does not depend on nested Git repositories. The
recorded revision is the exact base source revision used for this distribution;
distribution-specific edits carry explicit modification notices where required.

| Component | Repository path | Public project | Base revision | License |
| --- | --- | --- | --- | --- |
| TVM | `3rdparty/tvm_sunrise/` | <https://github.com/tile-ai/tvm> | `415f44b0ddba3dff19631eeda1f559bf7951e4f0` | Apache-2.0 |
| Apache TVM FFI | `3rdparty/tvm_sunrise/3rdparty/tvm-ffi/` | <https://github.com/apache/tvm-ffi> | `15e1e8d2682369199704433dde43d62c885b0014` | Apache-2.0 |
| DLPack | `3rdparty/tvm_sunrise/3rdparty/tvm-ffi/3rdparty/dlpack/` | <https://github.com/dmlc/dlpack> | `5cfd3fb7adf6b1a56a26e3408a887a6cca73aec8` | Apache-2.0 |
| libbacktrace | `3rdparty/tvm_sunrise/3rdparty/tvm-ffi/3rdparty/libbacktrace/` | <https://github.com/ianlancetaylor/libbacktrace> | `793921876c981ce49759114d7bb89bb89b2d3a2d` | BSD-3-Clause |
| TileOPs | `downstream/tileops_sunrise/` | <https://github.com/tile-ai/TileOPs> | `040be8d3d684f51e5afffce62ef18636b5122b6c` | MIT |
| TileKernels | `downstream/tilekernels_sunrise/` | <https://github.com/deepseek-ai/TileKernels> | `1f8eba24eb8e9723143aa5dc7ed6f7a5fd49586f` | MIT |

Each component retains its own license, attribution files, and source copyright
notices in its repository path. TVM and Apache TVM FFI also retain their
`NOTICE` and `licenses/` trees. These projects and their licensors do not
endorse TileLang-Sunrise.
