# Vendored Components

This repository contains source snapshots of the components below so that the
TileLang-Sunrise source tree does not depend on nested Git repositories. The
recorded revision is the exact base source revision used for this distribution;
distribution-specific edits carry explicit modification notices where required.

| Component | Repository path | Public project | Base revision | License |
| --- | --- | --- | --- | --- |
| TVM | `3rdparty/tvm_sunrise/` | <https://github.com/tile-ai/tvm> | `501200faa70879869f1632b3e468cd1576262b61` | Apache-2.0 |
| Apache TVM FFI | `3rdparty/tvm_sunrise/3rdparty/tvm-ffi/` | <https://github.com/apache/tvm-ffi> | `66400b2f22dc713f4c999b8f19ce3c5a273a9719` | Apache-2.0 |
| DLPack | `3rdparty/tvm_sunrise/3rdparty/tvm-ffi/3rdparty/dlpack/` | <https://github.com/dmlc/dlpack> | `5cfd3fb7adf6b1a56a26e3408a887a6cca73aec8` | Apache-2.0 |
| libbacktrace | `3rdparty/tvm_sunrise/3rdparty/tvm-ffi/3rdparty/libbacktrace/` | <https://github.com/ianlancetaylor/libbacktrace> | `793921876c981ce49759114d7bb89bb89b2d3a2d` | BSD-3-Clause |
| TileOPs | `downstream/tileops_sunrise/` | <https://github.com/tile-ai/TileOPs> | `1f8e299d272d10471b01a94560122583e8009a2e` | MIT |
| TileKernels | `downstream/tilekernels_sunrise/` | <https://github.com/deepseek-ai/TileKernels> | `b1a149cb8daa25b280b2bde6df50a99c0e3b8592` | MIT |

Each component retains its own license, attribution files, and source copyright
notices in its repository path. TVM and Apache TVM FFI also retain their
`NOTICE` and `licenses/` trees. These projects and their licensors do not
endorse TileLang-Sunrise.
