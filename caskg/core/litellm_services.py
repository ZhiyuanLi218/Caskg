from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import re
import time
from typing import Any, Type
import asyncio

import litellm
import numpy as np
from json_repair import repair_json
from pydantic import BaseModel

from fast_graphrag._llm._base import BaseEmbeddingService, BaseLLMService, T_model
from fast_graphrag._models import BaseModelAlias


JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# Global LLM rate limiter
# ---------------------------------------------------------------------------
# fast_graphrag drives information extraction by firing one LLM call per
# document/chunk through asyncio with a default concurrency of 1024. Those calls
# go through THIS service (not fast_graphrag's built-in OpenAI client), so
# fast_graphrag's own CONCURRENT_TASK_LIMIT semaphore does NOT throttle them —
# 200+ requests hit the provider at once and trip its RPM limit (429).
#
# This module-level limiter caps BOTH the number of in-flight requests and the
# request rate (requests/minute) across every caller in the process, so no code
# path — indexing, candidate induction, the LLM judge, query rewrite — can
# burst past the provider quota. Tunable via env vars for different quotas.
_LLM_MAX_CONCURRENCY = int(os.getenv("CASKG_LLM_MAX_CONCURRENCY", "2"))
_LLM_MAX_RPM = int(os.getenv("CASKG_LLM_MAX_RPM", "60"))  # half of provider RPM=120
_LLM_CALL_TIMEOUT = float(os.getenv("CASKG_LLM_CALL_TIMEOUT", "60"))  # per-call hard cap (s)

_llm_semaphore: asyncio.Semaphore | None = None
_llm_rate_lock: asyncio.Lock | None = None
_llm_call_times: list[float] = []


def _get_llm_semaphore() -> asyncio.Semaphore:
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(_LLM_MAX_CONCURRENCY)
    return _llm_semaphore


def _get_llm_rate_lock() -> asyncio.Lock:
    global _llm_rate_lock
    if _llm_rate_lock is None:
        _llm_rate_lock = asyncio.Lock()
    return _llm_rate_lock


async def _throttle_llm_rate() -> None:
    """Block until issuing a new call keeps us under _LLM_MAX_RPM (sliding 60s window)."""
    if _LLM_MAX_RPM <= 0:
        return
    async with _get_llm_rate_lock():
        now = time.monotonic()
        # Drop timestamps older than 60s.
        cutoff = now - 60.0
        while _llm_call_times and _llm_call_times[0] < cutoff:
            _llm_call_times.pop(0)
        if len(_llm_call_times) >= _LLM_MAX_RPM:
            # Wait until the oldest call ages out of the window.
            sleep_for = 60.0 - (now - _llm_call_times[0]) + 0.05
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
        _llm_call_times.append(time.monotonic())


def extract_json_text(content: str) -> str:
    fenced = JSON_BLOCK_PATTERN.search(content)
    if fenced:
        return fenced.group(1).strip()
    return content.strip()



def validate_response_model(response_model: Type[T_model], content: str) -> T_model:
    cleaned = extract_json_text(content)
    repaired = repair_json(cleaned)

    if issubclass(response_model, BaseModelAlias):
        parsed = response_model.Model.model_validate_json(repaired)
        return parsed.to_dataclass(parsed)

    return response_model.model_validate_json(repaired)


@dataclass
class LiteLLMService(BaseLLMService):
    temperature: float = field(default=0.0)

    async def send_message(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
        response_model: Type[T_model] | None = None,
        **kwargs: Any,
    ) -> tuple[T_model, list[dict[str, str]]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        # Global throttle: cap concurrency + request rate across all callers so
        # fast_graphrag's high-fan-out extraction can't burst past the provider
        # RPM limit. Retry transient rate-limit errors with backoff.
        last_exc: Exception | None = None
        for attempt in range(5):
            await _throttle_llm_rate()
            async with _get_llm_semaphore():
                try:
                    # Hard timeout: without it, a single stalled request holds a
                    # concurrency slot forever (litellm's default request_timeout
                    # is ~6000s). Under concurrency=2 that wedges the whole run.
                    response = await asyncio.wait_for(
                        litellm.acompletion(
                            model=self.model,
                            messages=messages,
                            api_key=self.api_key,
                            base_url=self.base_url,
                            timeout=_LLM_CALL_TIMEOUT,
                            temperature=kwargs.pop("temperature", self.temperature)
                            if attempt == 0
                            else self.temperature,
                            **(kwargs if attempt == 0 else {}),
                        ),
                        timeout=_LLM_CALL_TIMEOUT,
                    )
                    break
                except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
                    last_exc = exc
                    is_timeout = isinstance(exc, asyncio.TimeoutError)
                    if is_timeout or "429" in str(exc) or "RateLimit" in type(exc).__name__:
                        await asyncio.sleep(2.0 * (attempt + 1))
                        continue
                    raise
        else:
            raise last_exc if last_exc else RuntimeError("LLM call failed")

        content = response.choices[0].message.content or ""
        if response_model is None:
            parsed_response = content
        else:
            parsed_response = validate_response_model(response_model, content)

        updated_history = messages + [{"role": "assistant", "content": content}]
        return parsed_response, updated_history


@dataclass
class LiteLLMEmbeddingService(BaseEmbeddingService):
    # Batch size per API call. Keep small to respect RPM/TPM limits.
    embedding_batch_size: int = field(default=20)

    async def _encode_batch(self, batch: list[str], model: str) -> list[list[float]]:
        # Share the global LLM rate limiter so embedding bursts (fast_graphrag
        # may call encode from concurrent insert paths) also stay under quota.
        last_exc: Exception | None = None
        for attempt in range(5):
            await _throttle_llm_rate()
            async with _get_llm_semaphore():
                try:
                    response = await asyncio.wait_for(
                        litellm.aembedding(
                            model=model,
                            input=batch,
                            api_key=self.api_key,
                            api_base=self.base_url or None,
                            timeout=_LLM_CALL_TIMEOUT,
                        ),
                        timeout=_LLM_CALL_TIMEOUT,
                    )
                    break
                except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
                    last_exc = exc
                    is_timeout = isinstance(exc, asyncio.TimeoutError)
                    if is_timeout or "429" in str(exc) or "RateLimit" in type(exc).__name__:
                        await asyncio.sleep(2.0 * (attempt + 1))
                        continue
                    raise
        else:
            raise last_exc if last_exc else RuntimeError("Embedding call failed")

        vectors = []
        for item in response.data:
            if isinstance(item, dict):
                vectors.append(item["embedding"])
            else:
                vectors.append(item.embedding)
        return vectors

    async def encode(
        self,
        texts: list[str],
        model: str | None = None,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        resolved_model = model or self.model
        batches = [
            texts[i : i + self.embedding_batch_size]
            for i in range(0, len(texts), self.embedding_batch_size)
        ]
        # Process batches sequentially with delay to respect rate limits (RPM=120)
        all_vectors: list[list[float]] = []
        for batch in batches:
            vectors = await self._encode_batch(batch, resolved_model)
            all_vectors.extend(vectors)
            if len(batches) > 1:
                await asyncio.sleep(0.5)
        return np.array(all_vectors, dtype=np.float32)
