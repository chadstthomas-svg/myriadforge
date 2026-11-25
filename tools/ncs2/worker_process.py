"""Bound a legacy worker, including the child of a Windows venv launcher."""

import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep


def stop_worker(process):
    if sys.platform == "win32" and process.poll() is None:
        subprocess.run(["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                       capture_output=True, check=False)
    if process.poll() is None:
        process.kill()


def run_worker(command, timeout):
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, errors="replace") as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            stop_worker(process)
            error.output, error.stderr = process.communicate()
            raise
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def watch_decoder(command, result_path, compile_timeout=120):
    """Relay child progress and bound each decoder compile/load independently."""
    with TemporaryDirectory(prefix="ncs2-decoder-") as directory:
        scratch = Path(directory)
        log, result = scratch / "worker.log", scratch / "result.json"
        command = [*command, "--out", str(result), "--worker"]
        phase, start, limit = "initialization", monotonic(), compile_timeout
        with log.open("wb") as writer, subprocess.Popen(
            command, stdout=writer, stderr=subprocess.STDOUT,
        ) as process, log.open("r", encoding="utf8", errors="replace") as reader:
            while True:
                for line in reader.readlines():
                    print(line, end="", flush=True)
                    if line.startswith(("BUILD ", "COMPILE ")):
                        phase, start, limit = line.strip(), monotonic(), compile_timeout
                    elif line.startswith(("LOADED ", "CASE ", "REPLAY ")):
                        phase, start, limit = line.strip(), monotonic(), 180
                if process.poll() is not None:
                    break
                if monotonic() - start >= limit:
                    stop_worker(process)
                    process.wait()
                    return 1, {"status": "timeout", "stage": phase, "timeout_sec": limit}
                sleep(.25)
            if result.exists():
                result_path.write_bytes(result.read_bytes())
                return process.returncode, None
            return process.returncode or 1, {"status": "error", "stage": phase, "detail": log.read_text(
                encoding="utf8", errors="replace",
            )[-1800:]}
