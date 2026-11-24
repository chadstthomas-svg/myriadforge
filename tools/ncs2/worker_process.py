"""Bound a legacy worker, including the child of a Windows venv launcher."""

import subprocess
import sys


def run_worker(command, timeout):
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, errors="replace") as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            if sys.platform == "win32" and process.poll() is None:
                subprocess.run(["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                               capture_output=True, check=False)
            if process.poll() is None:
                process.kill()
            error.output, error.stderr = process.communicate()
            raise
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
