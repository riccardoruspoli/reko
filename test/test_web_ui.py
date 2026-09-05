from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_web_ui_scenarios() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.fail("Node.js is required to run the local web UI scenarios.")

    result = subprocess.run(
        [node, "test/web_ui_harness.mjs"],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 0, result.stderr
    assert "web UI scenarios passed" in result.stdout
