"""Bound native driver stalls by supervising the batch worker's JSONL pipe."""
import json
import os
from queue import Empty, Queue
import subprocess
from threading import Thread


class BatchWorker:
    def __init__(self, command):
        self.child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.requests = Queue()
        self.responses = Queue()
        self.thread = Thread(target=self.exchange, daemon=True)
        self.thread.start()

    def exchange(self):
        try:
            with self.child.stdin as request_pipe, self.child.stdout as response_pipe:
                while True:
                    request = self.requests.get()
                    if request is None:
                        return
                    request_pipe.write(request if request.endswith(b"\n") else request + b"\n")
                    request_pipe.flush()
                    reply = response_pipe.readline()
                    if not reply:
                        break
                    self.responses.put(reply)
        except OSError:
            pass
        self.responses.put(None)

    def request(self, raw, timeout):
        self.requests.put(raw)
        try:
            response = self.responses.get(timeout=timeout)
        except Empty as exc:
            raise RuntimeError("Batch timed out. The worker will restart for the next batch.") from exc
        if response is None:
            raise RuntimeError("Batch worker exited before returning a result")
        return response

    def close(self, force=False):
        self.requests.put(None)
        if not force:
            try:
                self.child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        if self.child.poll() is None:
            if os.name == "nt":
                # The venv launcher can create a child holding the native driver.
                subprocess.run(["taskkill.exe", "/PID", str(self.child.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=8)
            else:
                self.child.kill()
            self.child.wait(timeout=5)
        self.thread.join(timeout=1)


def supervise(command, stdin, output, timeout):
    worker = None
    try:
        for line in stdin:
            if worker is None:
                worker = BatchWorker(command)
            try:
                response = worker.request(line, timeout).decode("utf8").rstrip("\r\n") + "\n"
            except RuntimeError as exc:
                worker.close(force=True)
                worker = None
                try:
                    request = json.loads(line)
                except ValueError:
                    request = None
                request_id = request.get("id") if isinstance(request, dict) else None
                response = json.dumps({"id": request_id, "error": str(exc)}) + "\n"
            output.write(response)
            output.flush()
    finally:
        if worker is not None:
            worker.close()
