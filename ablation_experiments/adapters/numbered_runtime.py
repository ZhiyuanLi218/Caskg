"""Runtime isolation for the numbered A1--A4 CaSKG ablation.

This module wraps, rather than edits, the production retrieval engine.  It adds
one shared SQLite query-embedding cache, fails closed on embedding errors, turns
A1's normal ``async_retrieve`` entry point into vector-only retrieval, and writes
per-process retrieval audit records under the ablation directory.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import types
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ablation_experiments.adapters.causal_graph_views import sha256_file
from ablation_experiments.adapters.numbered_graph_views import (
    A1_VARIANT,
    PROTOCOL_ID,
    VARIANTS,
)


CACHE_SCHEMA_VERSION = 1


class InfrastructureRetrievalError(BaseException):
    """Fail-closed retrieval error that bypasses lexical fallback handlers."""


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _serialize_array(value: np.ndarray[Any, Any]) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return buffer.getvalue()


def _deserialize_array(value: bytes) -> np.ndarray[Any, Any]:
    return np.load(io.BytesIO(value), allow_pickle=False)


class SharedSQLiteEmbeddingCache:
    """Process-safe, exact-vector cache around a CaSKG embedding service."""

    def __init__(
        self,
        inner: Any,
        cache_path: str | Path,
        *,
        timeout_seconds: float = 600.0,
    ) -> None:
        self.inner = inner
        self.cache_path = Path(cache_path).expanduser().resolve()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = float(timeout_seconds)
        self.model = str(getattr(inner, "model", ""))
        self.embedding_dim = int(getattr(inner, "embedding_dim", 0) or 0)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.cache_path,
            timeout=self.timeout_seconds,
            isolation_level=None,
        )
        connection.execute(f"PRAGMA busy_timeout={int(self.timeout_seconds * 1000)}")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL,
                    model TEXT NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    text_sha256 TEXT NOT NULL,
                    vector_npy BLOB NOT NULL,
                    vector_sha256 TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS embeddings_text_sha256 "
                "ON embeddings(text_sha256)"
            )

    def _resolved_model(self, model: str | None) -> str:
        return str(model or self.model)

    def _key(self, text: str, model: str) -> tuple[str, str]:
        text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return (
            _canonical_hash(
                {
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "model": model,
                    "embedding_dim": self.embedding_dim,
                    "text_sha256": text_sha256,
                }
            ),
            text_sha256,
        )

    @staticmethod
    def _load_cached(
        connection: sqlite3.Connection,
        cache_key: str,
    ) -> np.ndarray[Any, Any] | None:
        row = connection.execute(
            "SELECT vector_npy, vector_sha256 FROM embeddings WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        blob = bytes(row[0])
        if hashlib.sha256(blob).hexdigest() != str(row[1]):
            raise InfrastructureRetrievalError(
                "InfrastructureRetrievalError: embedding cache checksum mismatch."
            )
        vector = _deserialize_array(blob)
        if vector.ndim != 1:
            raise InfrastructureRetrievalError(
                "InfrastructureRetrievalError: cached embedding is not one-dimensional."
            )
        return vector

    async def encode(
        self,
        texts: list[str],
        model: str | None = None,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        resolved_model = self._resolved_model(model)
        keys = [self._key(str(text), resolved_model) for text in texts]

        try:
            with closing(self._connect()) as connection:
                cached = [self._load_cached(connection, key) for key, _ in keys]
            if all(vector is not None for vector in cached):
                return np.stack(cached).astype(np.float32, copy=False)  # type: ignore[arg-type]

            missing_indices = [
                index for index, vector in enumerate(cached) if vector is None
            ]
            missing_texts = [str(texts[index]) for index in missing_indices]
            generated = await self.inner.encode(missing_texts, model=resolved_model)
            generated_array = np.asarray(generated, dtype=np.float32)
            if generated_array.ndim != 2:
                raise ValueError("Embedding backend returned a non-matrix result.")
            if generated_array.shape[0] != len(missing_indices):
                raise ValueError("Embedding backend returned the wrong batch size.")
            if self.embedding_dim and generated_array.shape[1] != self.embedding_dim:
                raise ValueError(
                    "Embedding backend returned dimension "
                    f"{generated_array.shape[1]} instead of {self.embedding_dim}."
                )

            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                for row_index, source_index in enumerate(missing_indices):
                    vector = generated_array[row_index].copy()
                    blob = _serialize_array(vector)
                    cache_key, text_sha256 = keys[source_index]
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO embeddings (
                            cache_key, schema_version, model, embedding_dim,
                            text_sha256, vector_npy, vector_sha256, created_at_utc
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            cache_key,
                            CACHE_SCHEMA_VERSION,
                            resolved_model,
                            int(vector.shape[0]),
                            text_sha256,
                            blob,
                            hashlib.sha256(blob).hexdigest(),
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )
                connection.execute("COMMIT")
            except BaseException:
                try:
                    connection.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
            finally:
                connection.close()

            # Re-read after INSERT OR IGNORE so concurrent misses always return
            # the single vector that won the database race.
            with closing(self._connect()) as connection:
                cached = [self._load_cached(connection, key) for key, _ in keys]
        except InfrastructureRetrievalError:
            raise
        except Exception as exc:
            raise InfrastructureRetrievalError(
                "InfrastructureRetrievalError: shared embedding request/cache failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if any(vector is None for vector in cached):
            raise InfrastructureRetrievalError(
                "InfrastructureRetrievalError: embedding cache left an unresolved vector."
            )
        return np.stack(cached).astype(np.float32, copy=False)  # type: ignore[arg-type]

    def stats(self) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            count = int(connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "path": str(self.cache_path),
            "entry_count": count,
            "model": self.model,
            "embedding_dim": self.embedding_dim,
        }


def _seed_record(seed: Any) -> dict[str, Any]:
    return {
        "name": str(getattr(seed, "name", "")),
        "seed_weight": float(getattr(seed, "seed_weight", 0.0) or 0.0),
        "semantic_rank": getattr(seed, "semantic_rank", None),
    }


def _write_retrieval_audit(
    audit_dir: Path,
    *,
    variant: str,
    workspace: Path,
    query: str,
    result: Any,
) -> None:
    audit_dir.mkdir(parents=True, exist_ok=True)
    budget = getattr(result, "budget", None)
    skills = list(getattr(result, "skills", []) or [])
    seeds = list(getattr(result, "seeds", []) or [])
    relations = list(getattr(result, "relations", []) or [])
    rendered_context = str(getattr(result, "rendered_context", "") or "")
    record = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "variant_id": variant,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "query": query,
        "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        "workspace": str(workspace),
        "workspace_graph_sha256": sha256_file(workspace / "graph_igraph_data.pklz"),
        "skill_names": [str(getattr(skill, "name", "")) for skill in skills],
        "seeds": [_seed_record(seed) for seed in seeds],
        "relation_count": len(relations),
        "rendered_context_chars": len(rendered_context),
        "rendered_context_sha256": hashlib.sha256(
            rendered_context.encode("utf-8")
        ).hexdigest(),
        "budget": {
            "top_n": getattr(budget, "top_n", None),
            "seed_top_k": getattr(budget, "seed_top_k", None),
            "ppr_damping": getattr(budget, "ppr_damping", None),
            "max_chars_per_skill": getattr(budget, "max_chars_per_skill", None),
            "max_context_chars": getattr(budget, "max_context_chars", None),
        },
    }
    path = audit_dir / f"retrieval_pid_{os.getpid()}.jsonl"
    try:
        with path.open("a", encoding="ascii", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
            handle.write("\n")
    except Exception as exc:
        raise InfrastructureRetrievalError(
            "InfrastructureRetrievalError: retrieval audit write failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def configure_numbered_engine(
    engine: Any,
    *,
    variant: str,
    workspace: str | Path,
    embedding_cache: str | Path,
    audit_dir: str | Path,
) -> Any:
    """Install a numbered ablation view on one already-created engine."""

    if variant not in VARIANTS:
        raise ValueError(f"Unsupported numbered ablation variant: {variant}")
    workspace_path = Path(workspace).expanduser().resolve()
    cache_path = Path(embedding_cache).expanduser().resolve()
    audit_path = Path(audit_dir).expanduser().resolve()
    if not (workspace_path / "graph_igraph_data.pklz").is_file():
        raise FileNotFoundError(f"Invalid numbered workspace: {workspace_path}")

    current_embedding = engine.config.embedding_service
    if not isinstance(current_embedding, SharedSQLiteEmbeddingCache):
        engine.config.embedding_service = SharedSQLiteEmbeddingCache(
            current_embedding,
            cache_path,
        )
    original_hybrid = engine.async_retrieve
    original_vector = engine.async_retrieve_vector

    async def numbered_async_retrieve(
        _self: Any,
        query: str,
        *,
        top_n: int | None = None,
        seed_top_k: int | None = None,
        max_chars_per_skill: int | None = None,
        max_context_chars: int | None = None,
    ) -> Any:
        del seed_top_k
        if variant == A1_VARIANT:
            result = await original_vector(
                query,
                top_n=top_n,
                max_chars_per_skill=max_chars_per_skill,
                max_context_chars=max_context_chars,
            )
        else:
            result = await original_hybrid(
                query,
                top_n=top_n,
                max_chars_per_skill=max_chars_per_skill,
                max_context_chars=max_context_chars,
            )
        _write_retrieval_audit(
            audit_path,
            variant=variant,
            workspace=workspace_path,
            query=query,
            result=result,
        )
        return result

    engine.async_retrieve = types.MethodType(numbered_async_retrieve, engine)
    engine._numbered_ablation_runtime = {  # noqa: SLF001
        "protocol_id": PROTOCOL_ID,
        "variant_id": variant,
        "embedding_cache": str(cache_path),
        "audit_dir": str(audit_path),
        "vector_only": variant == A1_VARIANT,
    }
    return engine


def cache_stats(cache_path: str | Path) -> dict[str, Any]:
    path = Path(cache_path).expanduser().resolve()
    if not path.is_file():
        return {"path": str(path), "entry_count": 0, "exists": False}
    with closing(sqlite3.connect(path)) as connection:
        count = int(connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])
    return {"path": str(path), "entry_count": count, "exists": True}


def audit_files(audit_dir: str | Path) -> Iterable[Path]:
    return sorted(Path(audit_dir).glob("retrieval_pid_*.jsonl"))
