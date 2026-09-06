from __future__ import annotations

import json
import random
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


class ChatInfrastructureError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatResponse:
    content: str
    usage: dict[str, int] = field(default_factory=dict)
    response_id: str = ""
    latency_seconds: float = 0.0
    request_attempt_count: int = 1
    retry_errors: tuple[str, ...] = ()
    router_provider: str = ""
    router_policy: str = ""
    router_request_id: str = ""


def _message_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return ""


class OpenAICompatibleChatClient:
    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model: str,
        temperature: float,
        max_completion_tokens: int,
        timeout_seconds: float,
        attempts: int,
        retry_base_delay_seconds: float,
        retry_max_delay_seconds: float,
        router_provider: str = "",
        router_policy: str = "",
    ) -> None:
        if not api_base.strip():
            raise ValueError("Chat API base URL is empty.")
        if not api_key.strip():
            raise ValueError("Chat API key is empty.")
        self.endpoint = api_base.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.timeout_seconds = timeout_seconds
        self.attempts = attempts
        self.retry_base_delay_seconds = retry_base_delay_seconds
        self.retry_max_delay_seconds = retry_max_delay_seconds
        self.router_provider = router_provider.strip()
        self.router_policy = router_policy.strip()

    def complete(self, messages: list[dict[str, str]]) -> ChatResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_completion_tokens,
            "stream": False,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error = "unknown chat error"
        retry_errors: list[str] = []
        for attempt in range(1, self.attempts + 1):
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "ScienceWorld-Paired-Evaluator/1.0",
            }
            if self.router_provider:
                headers["X-Router-Provider"] = self.router_provider
            if self.router_policy:
                headers["X-Router-Policy"] = self.router_policy
            request = urllib.request.Request(
                self.endpoint,
                data=body,
                method="POST",
                headers=headers,
            )
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                data = json.loads(raw)
                choices = data.get("choices") or []
                if not choices:
                    raise ChatInfrastructureError("Chat response has no choices.")
                content = _message_content(choices[0].get("message", {}).get("content"))
                if not content.strip():
                    raise ChatInfrastructureError("Chat response content is empty.")
                usage = {
                    str(key): int(value)
                    for key, value in (data.get("usage") or {}).items()
                    if isinstance(value, (int, float))
                }
                return ChatResponse(
                    content=content,
                    usage=usage,
                    response_id=str(data.get("id", "")),
                    latency_seconds=time.perf_counter() - started,
                    request_attempt_count=attempt,
                    retry_errors=tuple(retry_errors),
                    router_provider=str(response.headers.get("X-Router-Provider", "")),
                    router_policy=str(response.headers.get("X-Router-Policy", "")),
                    router_request_id=str(response.headers.get("X-Router-Request-ID", "")),
                )
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")[:2000]
                last_error = f"HTTP {exc.code}: {error_body}"
                retryable = exc.code in {408, 409, 425, 429} or 500 <= exc.code <= 599
                if not retryable:
                    raise ChatInfrastructureError(last_error) from exc
            except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            except json.JSONDecodeError as exc:
                last_error = f"Malformed chat JSON: {exc}"
            except ChatInfrastructureError as exc:
                last_error = str(exc)

            if attempt >= self.attempts:
                break
            retry_errors.append(last_error)
            delay = min(
                self.retry_base_delay_seconds * (2 ** (attempt - 1)),
                self.retry_max_delay_seconds,
            )
            time.sleep(delay * random.uniform(0.9, 1.1))
        raise ChatInfrastructureError(
            f"Chat request exhausted {self.attempts} attempts: {last_error}"
        )
