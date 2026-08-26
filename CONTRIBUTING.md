# Contributing

That would be awesome if you want to contribute something to TileLang!

## Table of Contents  <!-- omit in toc --> <!-- markdownlint-disable heading-increment -->

- [Report Bugs](#report-bugs)
- [Ask Questions](#ask-questions)
- [Submit Pull Requests](#submit-pull-requests)
- [Coding Style](#coding-style)
- [Setup Development Environment](#setup-development-environment)
- [Install Develop Version](#install-develop-version)
- [Lint Check](#lint-check)
- [Test Locally](#test-locally)
- [Build Wheels](#build-wheels)
- [Documentation](#documentation)

## Report Bugs

If you run into any weird behavior while using TileLang, feel free to open a new issue in this repository! Please run a **search before opening** a new issue, to make sure that someone else hasn't already reported or solved the bug you've found.

Any issue you open must include:

- Code snippet that reproduces the bug with a minimal setup.
- A clear explanation of what the issue is.

## Ask Questions

Please ask questions in issues.

## Submit Pull Requests

All pull requests are welcome and greatly appreciated. Check the repository's
open issues if you are looking for somewhere to start.

If you're new to contributing to TileLang, you can follow the following guidelines before submitting a pull request.

> [!NOTE]
> Please include tests and docs with every pull request if applicable!

## Coding Style

Run the repository formatter on the files touched by your pull request:

```bash
bash format.sh --files <changed-file>...
```

Python code is checked with Ruff through pre-commit. C and C++ code is checked
with clang-format, and TileLang-owned C++ APIs should follow the
[C++ Style Guide](docs/developer_guide/cpp_style.md) for naming, TVM object
usage, header boundaries, and incremental migration.

For a focused C++ API naming or header-boundary cleanup, run the advisory audit
locally:

```bash
python3 maint/scripts/audit_cpp_api_style.py
```

The audit is a review aid, not a blanket rename command. Check each finding for
FFI visibility, generated code, backend shims, and external API constraints
before changing it.

## Setup Development Environment

Before contributing to TileLang, please follow the instructions below to setup.

1. Fork this repository and clone your fork. All compiler dependencies needed
   by the source tree are already vendored.

    ```bash
    git clone <your-fork-url> tilelang-sunrise
    cd tilelang-sunrise
    ```

2. Setup a development environment:

    ```bash
    uv venv --seed .venv  # use `python3 -m venv .venv` if you don't have `uv`

    source .venv/bin/activate
    python3 -m pip install --upgrade pip setuptools wheel "build[uv]"
    uv pip install --requirements requirements-dev.txt
    ```

3. Setup the [`pre-commit`](https://pre-commit.com) hooks:

    ```bash
    pre-commit install --install-hooks
    ```

Then you are ready to rock. Thanks for contributing to TileLang!

## Install Develop Version

Build the vendored compiler stack as described in [README.md](README.md), then
install TileLang-Sunrise in editable mode with the same TANG build environment:

```bash
python3 -m pip install --no-build-isolation --verbose --editable .
```

in the main directory. This installation is removable by:

```bash
python3 -m pip uninstall tilelang
```

We also recommend installing TileLang in a more manual way for better control over the build process, by compiling the C++ extensions first and set the `PYTHONPATH`. See [Working from Source via `PYTHONPATH`](https://tilelang.com/get_started/Installation.html#working-from-source-via-pythonpath) for detailed instructions.

## Lint Check

To check the linting, run:

```bash
pre-commit run --all-files
```

This command checks the TileLang root project only. Vendored dependencies under
`3rdparty/` and the independent projects under `downstream/` are excluded
from TileLang's hook configuration. Run each downstream project's checks with
its own pinned hooks:

```bash
bash downstream/tileops_sunrise/ci/lint.sh
bash downstream/tilekernels_sunrise/ci/lint.sh
```

## Test Locally

To run the tests, start by building the project as described in the [Setup Development Environment](#setup-development-environment) section.

Then you can rerun the tests with:

```bash
python3 -m pytest testing
```

## Build Wheels

_TBA_

## Documentation

_TBA_
