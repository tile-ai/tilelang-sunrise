import subprocess
import sys


def test_autodd_module_help_runs_with_light_import(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "tilelang.autodd", "--help"],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "Delta-debug the provided Python source" in result.stdout
