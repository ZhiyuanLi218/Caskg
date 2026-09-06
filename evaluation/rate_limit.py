from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None


def _rpm_limit() -> float:
    raw_value = os.environ.get("CASKG_LLM_RPM_LIMIT", "50")
    try:
        return max(float(raw_value), 0.0)
    except (TypeError, ValueError):
        return 50.0


def _state_path() -> Path:
    raw_path = os.environ.get("CASKG_LLM_RATE_LIMIT_FILE")
    if raw_path:
        return Path(raw_path).expanduser()
    return Path(tempfile.gettempdir()) / "caskg_llm_rate_limit.json"


def _scope_key(model: str) -> str:
    scope = os.environ.get("CASKG_LLM_RATE_LIMIT_SCOPE", "global").lower()
    if scope == "model":
        return f"model:{model}"
    return "global"


def _cooldown_seconds() -> float:
    raw_value = os.environ.get("CASKG_LLM_RATE_LIMIT_COOLDOWN_SECS", "45")
    try:
        return max(float(raw_value), 0.0)
    except (TypeError, ValueError):
        return 45.0


def _is_rate_limit_error(error: object) -> bool:
    text = f"{type(error).__name__}: {error}"
    return (
        "429" in text
        or "RateLimit" in text
        or "GLOBAL_CONCURRENT_LIMIT_EXCEEDED" in text
    )


def acquire_llm_request_slot(model: str = "") -> None:
    """Throttle LLM request starts across evaluation worker processes.

    ALFWorld evaluation uses process-level parallelism, so per-process retry is
    not enough to respect provider RPM. This small file-locked token scheduler
    serializes request starts while still allowing environment work to run with
    high worker counts.
    """
    rpm = _rpm_limit()
    if rpm <= 0:
        return

    state_path = _state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    interval = 60.0 / rpm
    key = _scope_key(model)

    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                state = {}
            if not isinstance(state, dict):
                state = {}

            now = time.time()
            next_allowed = float(state.get(key, 0.0) or 0.0)
            wait_seconds = next_allowed - now
            if wait_seconds > 0:
                time.sleep(wait_seconds)
                now = time.time()

            state[key] = max(next_allowed, now) + interval
            state_path.write_text(json.dumps(state), encoding="utf-8")
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def cooldown_after_llm_error(model: str, error: object) -> None:
    """Extend the shared request-start schedule after provider rate limits.

    The regular scheduler spaces request starts, but some providers hold a
    global concurrency slot briefly after returning 429. Adding a shared
    cooldown keeps retry workers from immediately re-entering the same window.
    """
    if not _is_rate_limit_error(error):
        return

    cooldown = _cooldown_seconds()
    if cooldown <= 0:
        return

    state_path = _state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    key = _scope_key(model)

    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                state = {}
            if not isinstance(state, dict):
                state = {}

            now = time.time()
            state[key] = max(float(state.get(key, 0.0) or 0.0), now + cooldown)
            state_path.write_text(json.dumps(state), encoding="utf-8")
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    time.sleep(cooldown)
