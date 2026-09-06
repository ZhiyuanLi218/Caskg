from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ablation_experiments.adapters.numbered_graph_views import (
    A0_CONTROL_VARIANT,
    A1_VARIANT,
    A2_VARIANT,
)
from ablation_experiments.adapters.numbered_runtime import (
    InfrastructureRetrievalError,
    SharedSQLiteEmbeddingCache,
    configure_numbered_engine,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = REPOSITORY_ROOT / "ablation_experiments" / "cache"
WORKSPACE_ROOT = (
    REPOSITORY_ROOT
    / "ablation_experiments"
    / "graph_views"
    / "generated"
    / "a1-a4-main-parity-v1"
)


class FakeEmbedding:
    model = "openai/Qwen/Qwen3-Embedding-8B"
    embedding_dim = 4

    def __init__(self, value: float = 1.0, *, fail: bool = False) -> None:
        self.value = value
        self.fail = fail
        self.calls = 0

    async def encode(self, texts: list[str], model: str | None = None) -> np.ndarray:
        del model
        self.calls += 1
        if self.fail:
            raise RuntimeError("backend unavailable")
        return np.full((len(texts), self.embedding_dim), self.value, dtype=np.float32)


def _result(label: str, ppr_damping: float) -> SimpleNamespace:
    return SimpleNamespace(
        label=label,
        budget=SimpleNamespace(
            top_n=2,
            seed_top_k=2,
            ppr_damping=ppr_damping,
            max_chars_per_skill=100,
            max_context_chars=500,
        ),
        skills=[SimpleNamespace(name=f"{label}-skill")],
        seeds=[
            SimpleNamespace(name=f"{label}-seed", seed_weight=1.0, semantic_rank=1)
        ],
        relations=[] if label == "vector" else [SimpleNamespace()],
        rendered_context=label,
    )


class FakeEngine:
    def __init__(self) -> None:
        self.config = SimpleNamespace(embedding_service=FakeEmbedding())
        self.hybrid_calls = 0
        self.vector_calls = 0

    async def async_retrieve(self, query: str, **kwargs):
        del query, kwargs
        self.hybrid_calls += 1
        return _result("hybrid", 0.2)

    async def async_retrieve_vector(self, query: str, **kwargs):
        del query, kwargs
        self.vector_calls += 1
        return _result("vector", 0.0)


class NumberedRuntimeTests(unittest.TestCase):
    def test_shared_cache_reuses_the_database_vector(self) -> None:
        with tempfile.TemporaryDirectory(dir=CACHE_ROOT) as temporary:
            cache_path = Path(temporary) / "embeddings.sqlite3"
            first_inner = FakeEmbedding(1.0)
            second_inner = FakeEmbedding(2.0)
            first = SharedSQLiteEmbeddingCache(first_inner, cache_path)
            second = SharedSQLiteEmbeddingCache(second_inner, cache_path)
            first_value = asyncio.run(first.encode(["same query"]))
            second_value = asyncio.run(second.encode(["same query"]))
            np.testing.assert_array_equal(first_value, second_value)
            self.assertEqual(first_inner.calls, 1)
            self.assertEqual(second_inner.calls, 0)
            self.assertEqual(first.stats()["entry_count"], 1)

    def test_embedding_failure_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir=CACHE_ROOT) as temporary:
            cache = SharedSQLiteEmbeddingCache(
                FakeEmbedding(fail=True),
                Path(temporary) / "embeddings.sqlite3",
            )
            with self.assertRaises(InfrastructureRetrievalError):
                asyncio.run(cache.encode(["uncached query"]))

    def test_a1_replaces_only_the_retrieval_path(self) -> None:
        with tempfile.TemporaryDirectory(dir=CACHE_ROOT) as temporary:
            engine = FakeEngine()
            configure_numbered_engine(
                engine,
                variant=A1_VARIANT,
                workspace=WORKSPACE_ROOT / A1_VARIANT / "workspace",
                embedding_cache=Path(temporary) / "embeddings.sqlite3",
                audit_dir=Path(temporary) / "audit",
            )
            result = asyncio.run(engine.async_retrieve("task", top_n=2))
            self.assertEqual(result.label, "vector")
            self.assertEqual(engine.hybrid_calls, 0)
            self.assertEqual(engine.vector_calls, 1)
            audit_file = next((Path(temporary) / "audit").glob("*.jsonl"))
            audit = json.loads(audit_file.read_text(encoding="ascii").strip())
            self.assertEqual(audit["budget"]["ppr_damping"], 0.0)
            self.assertEqual(audit["relation_count"], 0)

    def test_p0_keeps_the_full_hybrid_path_on_the_a1_workspace(self) -> None:
        with tempfile.TemporaryDirectory(dir=CACHE_ROOT) as temporary:
            engine = FakeEngine()
            configure_numbered_engine(
                engine,
                variant=A0_CONTROL_VARIANT,
                workspace=WORKSPACE_ROOT / A1_VARIANT / "workspace",
                embedding_cache=Path(temporary) / "embeddings.sqlite3",
                audit_dir=Path(temporary) / "audit",
            )
            result = asyncio.run(engine.async_retrieve("task", top_n=2))
            self.assertEqual(result.label, "hybrid")
            self.assertEqual(engine.hybrid_calls, 1)
            self.assertEqual(engine.vector_calls, 0)
            audit_file = next((Path(temporary) / "audit").glob("*.jsonl"))
            audit = json.loads(audit_file.read_text(encoding="ascii").strip())
            self.assertEqual(audit["variant_id"], A0_CONTROL_VARIANT)
            self.assertGreater(audit["budget"]["ppr_damping"], 0.0)
            self.assertGreater(audit["relation_count"], 0)

    def test_a2_keeps_the_normal_hybrid_path(self) -> None:
        with tempfile.TemporaryDirectory(dir=CACHE_ROOT) as temporary:
            engine = FakeEngine()
            configure_numbered_engine(
                engine,
                variant=A2_VARIANT,
                workspace=WORKSPACE_ROOT / A2_VARIANT / "workspace",
                embedding_cache=Path(temporary) / "embeddings.sqlite3",
                audit_dir=Path(temporary) / "audit",
            )
            result = asyncio.run(engine.async_retrieve("task", top_n=2))
            self.assertEqual(result.label, "hybrid")
            self.assertEqual(engine.hybrid_calls, 1)
            self.assertEqual(engine.vector_calls, 0)


if __name__ == "__main__":
    unittest.main()
