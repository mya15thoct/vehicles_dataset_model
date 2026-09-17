"""Subprocess helpers for scripts that shell out to other scripts in this repo."""

from __future__ import annotations

import subprocess
from pathlib import Path


def run(command: list[str]) -> None:
    """Run a command to completion, echoing it first. Raises on non-zero exit."""
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def stream_to_log(command: list[str], log_path: Path) -> None:
    """Run a command, tee-ing its combined stdout/stderr to console and a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running: {' '.join(command)}", flush=True)
    print(f"Log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
