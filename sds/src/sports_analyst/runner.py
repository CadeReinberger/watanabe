"""Subprocess execution of AI-generated Python scripts."""

import os
import sys
import subprocess
import tempfile
from dataclasses import dataclass


@dataclass
class RunResult:
    stdout: str
    stderr: str
    returncode: int

    @property
    def success(self) -> bool:
        return self.returncode == 0


def run_script(code: str, timeout: int = 180) -> RunResult:
    """
    Write *code* to a temp file, execute it with the current Python interpreter,
    and return stdout/stderr/returncode.

    A 3-minute timeout is generous enough for most sports API calls.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        return RunResult(
            stdout=proc.stdout.strip(),
            stderr=proc.stderr.strip(),
            returncode=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return RunResult(
            stdout="",
            stderr=f"Script timed out after {timeout} seconds.",
            returncode=1,
        )
    finally:
        os.unlink(path)
