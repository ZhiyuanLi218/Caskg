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


def _load_engine(project: str, project_root: Path, workspace: Path) -> Any:
    if not workspace.is_dir():
        raise FileNotFoundError(f"Workspace does not exist: {workspace}")
    sys.path.insert(0, str(project_root))
    module = importlib.import_module(f"{project}.interfaces.claude_code")
    engine = module._get_engine(str(workspace))
    if bool(engine.config.enable_query_rewrite):
        raise RuntimeError(
            f"{project} query rewriting is enabled, but protocol v1 requires raw queries."
        )
    return engine


async def _retrieve(
    engine: Any,
    project: str,
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
        raise RuntimeError(
            "Workspace metadata changed during retrieval; refusing to treat the run as read-only."
        )

    payload = _jsonable(result)
    skills = list(payload.get("skills", []))
    return {
        "method": project,
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
    parser = argparse.ArgumentParser(description="Read-only GoS/CaSKG JSONL worker.")
    parser.add_argument("--project", choices=("gos", "caskg"), required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()

    os.chdir(args.project_root)
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    try:
        engine = _load_engine(args.project, args.project_root, args.workspace)
    except Exception as exc:
        print(f"Failed to initialize {args.project} retriever: {exc}", file=sys.stderr)
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
            bundle = asyncio.run(_retrieve(engine, args.project, args.workspace, request))
            _emit({"id": request_id, "ok": True, "bundle": bundle})
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
