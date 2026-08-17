from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_editable_install_imports_package_from_arbitrary_checkout_name(tmp_path):
    project = Path(__file__).resolve().parents[1]
    environment = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
    pip = environment / "bin" / "pip"
    python = environment / "bin" / "python"
    subprocess.run(
        [str(pip), "install", "--no-deps", "--editable", str(project)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        [
            str(python),
            "-c",
            "import dms_final_system; "
            "from dms_final_system.runtime.app import main; "
            "print(dms_final_system.__version__)",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip()
    assert (environment / "bin" / "dms-runtime").is_file()
