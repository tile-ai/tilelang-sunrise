from __future__ import annotations

import ast
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_sunrise_distribution_metadata():
    metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]

    assert (REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip() == "0.1.13+sunrise.1.0.0"
    assert project["name"] == "tilelang-sunrise"
    assert project["requires-python"] == ">=3.10"
    assert "apache-tvm-ffi==0.1.11+sunrise.1" in project["dependencies"]
    assert not any(dependency.startswith("torch-c-dlpack-ext") for dependency in project["dependencies"])
    assert "torch" not in project["dependencies"]
    assert "apache-tvm-ffi" not in (REPO_ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    assert (REPO_ROOT / "3rdparty" / "tvm_sunrise" / "3rdparty" / "tvm-ffi" / "LICENSE").is_file()


def test_sunrise_wheel_package_scope():
    metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    packages = metadata["tool"]["scikit-build"]["wheel"]["packages"]

    assert packages["tilelang/src/tl_templates"] == "src/tl_templates"
    assert packages["examples"] == "examples"
    assert "tilelang/src" not in packages
    assert "tilelang/3rdparty/tvm_sunrise/src" not in packages
    assert not any("cutlass" in path or "composable_kernel" in path for path in packages)


def test_vendored_downstream_packages_build_without_nested_git_metadata():
    tileops = tomllib.loads((REPO_ROOT / "downstream" / "tileops_sunrise" / "pyproject.toml").read_text(encoding="utf-8"))
    tilekernels = tomllib.loads((REPO_ROOT / "downstream" / "tilekernels_sunrise" / "pyproject.toml").read_text(encoding="utf-8"))

    version_range = "tilelang-sunrise>=0.1.13,<0.1.14"
    assert version_range in tileops["project"]["dependencies"]
    assert version_range in tilekernels["project"]["dependencies"]
    assert tilekernels["tool"]["setuptools_scm"]["fallback_version"] == "0.1.0"


def test_upstream_version_marker():
    module = ast.parse((REPO_ROOT / "tilelang" / "__init__.py").read_text(encoding="utf-8"))
    assignments = {
        target.id: ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id == "__upstream_version__"
    }

    assert assignments["__upstream_version__"] == "0.1.13"
