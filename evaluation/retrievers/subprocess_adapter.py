from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Sequence

from evaluation.retrievers.base import RetrievalBundle, RetrievalInfrastructureError


RESULT_PREFIX = "SCIENCEWORLD_RETRIEVAL_RESULT\t"


class PersistentSubprocessRetriever:
    def __init__(
        self,
        method: str,
        command: Sequence[str],
        *,
        timeout_seconds: float = 180.0,
    ) -> None:
        self.method = method
        self.command = list(command)
        self.timeout_seconds = timeout_seconds
        self._process: subprocess.Popen[str] | None = None
        self._stdout_queue: queue.Queue[str] = queue.Queue()
        self._stderr_tail: deque[str] = deque(maxlen=80)
        self._request_lock = threading.Lock()

    def _start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        python_path = Path(self.command[0]).resolve()
        if python_path.name.lower() in {"python.exe", "python"} and python_path.exists():
            prefix = python_path.parent
            runtime_paths = [prefix, prefix / "Scripts", prefix / "Library" / "bin"]
            env["PATH"] = os.pathsep.join(str(path) for path in runtime_paths) + os.pathsep + env.get("PATH", "")
            env["CONDA_PREFIX"] = str(prefix)
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        assert self._process.stdout is not None
        assert self._process.stderr is not None
        threading.Thread(
            target=self._drain_stdout,
            args=(self._process.stdout,),
            daemon=True,
            name=f"{self.method}-retriever-stdout",
        ).start()
        threading.Thread(
            target=self._drain_stderr,
            args=(self._process.stderr,),
            daemon=True,
            name=f"{self.method}-retriever-stderr",
        ).start()

    def _drain_stdout(self, stream: Any) -> None:
        for line in stream:
            self._stdout_queue.put(line.rstrip("\r\n"))

    def _drain_stderr(self, stream: Any) -> None:
        for line in stream:
            self._stderr_tail.append(line.rstrip("\r\n"))

    def _failure_context(self) -> str:
        if not self._stderr_tail:
            return "no stderr captured"
        return " | ".join(self._stderr_tail)

    def retrieve(
        self,
        query: str,
        *,
        top_n: int,
        max_chars_per_skill: int,
        max_context_chars: int,
    ) -> RetrievalBundle:
        request_id = uuid.uuid4().hex
        request = {
            "id": request_id,
            "operation": "retrieve",
            "query": query,
            "top_n": top_n,
            "max_chars_per_skill": max_chars_per_skill,
            "max_context_chars": max_context_chars,
        }
        with self._request_lock:
            self._start()
            assert self._process is not None
            assert self._process.stdin is not None
            try:
                self._process.stdin.write(json.dumps(request, ensure_ascii=True) + "\n")
                self._process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise RetrievalInfrastructureError(
                    f"{self.method} retrieval worker pipe failed: {exc}. "
                    f"Worker stderr: {self._failure_context()}"
                ) from exc

            deadline = time.monotonic() + self.timeout_seconds
            while True:
                if self._process.poll() is not None and self._stdout_queue.empty():
                    raise RetrievalInfrastructureError(
                        f"{self.method} retrieval worker exited with code "
                        f"{self._process.returncode}. Worker stderr: {self._failure_context()}"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RetrievalInfrastructureError(
                        f"{self.method} retrieval timed out after {self.timeout_seconds}s. "
                        f"Worker stderr: {self._failure_context()}"
                    )
                try:
                    line = self._stdout_queue.get(timeout=min(remaining, 0.5))
                except queue.Empty:
                    continue
                if not line.startswith(RESULT_PREFIX):
                    continue
                try:
                    response = json.loads(line[len(RESULT_PREFIX) :])
                except json.JSONDecodeError as exc:
                    raise RetrievalInfrastructureError(
                        f"{self.method} worker returned malformed JSON."
                    ) from exc
                if response.get("id") != request_id:
                    continue
                if not response.get("ok"):
                    error_type = response.get("error_type", "RetrievalError")
                    error = response.get("error", "unknown retrieval failure")
                    raise RetrievalInfrastructureError(
                        f"{self.method} {error_type}: {error}"
                    )
                bundle = RetrievalBundle.from_dict(response["bundle"])
                if bundle.method != self.method:
                    raise RetrievalInfrastructureError("Retriever method identity mismatch.")
                if bundle.query != query:
                    raise RetrievalInfrastructureError("Retriever changed the raw query.")
                if bundle.requested_top_n != top_n:
                    raise RetrievalInfrastructureError(
                        f"{self.method} changed requested top_n from {top_n} "
                        f"to {bundle.requested_top_n}."
                    )
                if len(bundle.skills) > top_n:
                    raise RetrievalInfrastructureError(
                        f"{self.method} returned {len(bundle.skills)} skills; top_n is {top_n}."
                    )
                if len(bundle.rendered_context) > max_context_chars:
                    raise RetrievalInfrastructureError(
                        f"{self.method} returned {len(bundle.rendered_context)} context chars; "
                        f"budget is {max_context_chars}."
                    )
                return bundle

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None and process.stdin is not None:
            try:
                process.stdin.write(
                    json.dumps({"id": uuid.uuid4().hex, "operation": "shutdown"}) + "\n"
                )
                process.stdin.flush()
                process.wait(timeout=5)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()

    def __enter__(self) -> "PersistentSubprocessRetriever":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
