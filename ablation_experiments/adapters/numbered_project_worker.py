"""Read-only ScienceWorld retrieval worker for numbered CaSKG ablations."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ablation_experiments.adapters.numbered_graph_views import VARIANTS  # noqa: E402
from ablation_experiments.adapters.numbered_runtime import (  # noqa: E402
    InfrastructureRetrievalError,
    configure_numbered_engine,
)


RESULT_PREFIX = "SCIENCEWORLD_RETRIEVAL_RESULT\t"


def _workspace_fingerprint(workspace: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in workspace.rglob("*") if item.is_file()):
        stat = path.stat()
        relative = path.relative_to(workspace).as_posix()
        digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def _jsonable(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    if hasattr(model, "dict"):
        return model.dict()
    raise TypeError(f"Unsupported retrieval result type: {type(model).__name__}")


def _load_engine(
    project_root: Path,
    workspace: Path,
    *,
    variant: str,
    embedding_cache: Path,
    audit_dir: Path,
) -> Any:
    if not workspace.is_dir():
        raise FileNotFoundError(f"Workspace does not exist: {workspace}")
    sys.path.insert(0, str(project_root))
    module = importlib.import_module("caskg.interfaces.claude_code")
    engine = module._get_engine(str(workspace))
    if bool(engine.config.enable_query_rewrite):
        raise RuntimeError("CaSKG query rewriting is enabled; raw-query parity failed.")
    return configure_numbered_engine(
        engine,
        variant=variant,
        workspace=workspace,
        embedding_cache=embedding_cache,
        audit_dir=audit_dir,
    )


async def _retrieve(
    engine: Any,
    workspace: Path,
    request: dict[str, Any],
) -> dict[str, Any]:
    query = str(request["query"])
    before = _workspace_fingerprint(workspace)
    started = time.perf_counter()
    result = await engine.async_retrieve(
        query,
        top_n=int(request["top_n"]),
        max_chars_per_skill=int(request["max_chars_per_skill"]),
        max_context_chars=int(request["max_context_chars"]),
    )
    latency = time.perf_counter() - started
    after = _workspace_fingerprint(workspace)
    if before != after:
        raise RuntimeError("Workspace changed during numbered retrieval.")

    payload = _jsonable(result)
    skills = list(payload.get("skills", []))
    return {
        "method": "caskg",
        "query": query,
        "status": "SKILL_HIT" if skills else "NO_SKILL",
        "requested_top_n": int(request["top_n"]),
        "rendered_context": str(payload.get("rendered_context", "")),
        "skills": skills,
        "relations": list(payload.get("relations", [])),
        "seeds": list(payload.get("seeds", [])),
        "latency_seconds": latency,
        "workspace": str(workspace),
        "workspace_fingerprint": before,
    }


def _emit(value: dict[str, Any]) -> None:
    print(RESULT_PREFIX + json.dumps(value, ensure_ascii=True), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    workspace = args.workspace.expanduser().resolve()
    os.chdir(project_root)
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    try:
        engine = _load_engine(
            project_root,
            workspace,
            variant=args.variant,
            embedding_cache=args.embedding_cache,
            audit_dir=args.audit_dir,
        )
    except InfrastructureRetrievalError as exc:
        print(f"Failed to initialize caskg retriever: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Failed to initialize caskg retriever: {exc}", file=sys.stderr)
        return 2

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        request_id = "unknown"
        try:
            request = json.loads(raw_line)
            request_id = str(request.get("id", "unknown"))
            operation = request.get("operation")
            if operation == "shutdown":
                return 0
            if operation != "retrieve":
                raise ValueError(f"Unsupported operation: {operation}")
            bundle = asyncio.run(_retrieve(engine, workspace, request))
            _emit({"id": request_id, "ok": True, "bundle": bundle})
        except InfrastructureRetrievalError as exc:
            _emit(
                {
                    "id": request_id,
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        except Exception as exc:
            _emit(
                {
                    "id": request_id,
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
