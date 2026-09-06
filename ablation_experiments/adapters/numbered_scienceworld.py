"""ScienceWorld retriever adapter pointing at the isolated numbered worker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation.retrievers.subprocess_adapter import PersistentSubprocessRetriever

from ablation_experiments.adapters.numbered_graph_views import PROTOCOL_ID, VARIANTS


class NumberedScienceWorldRetriever(PersistentSubprocessRetriever):
    def __init__(
        self,
        config_path: str | Path,
        *,
        worker_path: str | Path,
        timeout_seconds: float = 180.0,
    ) -> None:
        del worker_path
        config_path = Path(config_path).expanduser().resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("protocol_id") != PROTOCOL_ID:
            raise ValueError("Numbered ScienceWorld config protocol mismatch.")
        method: dict[str, Any] = config["methods"]["caskg"]
        variant = str(method["variant"])
        if variant not in VARIANTS:
            raise ValueError(f"Unsupported numbered variant: {variant}")
        project_root = Path(str(method["project_root"])).expanduser().resolve()
        numbered_worker = (
            project_root
            / "ablation_experiments"
            / "adapters"
            / "numbered_project_worker.py"
        )
        command = [
            str(method["python"]),
            "-u",
            str(numbered_worker),
            "--project-root",
            str(project_root),
            "--workspace",
            str(method["workspace"]),
            "--variant",
            variant,
            "--embedding-cache",
            str(method["embedding_cache"]),
            "--audit-dir",
            str(method["audit_dir"]),
        ]
        super().__init__("caskg", command, timeout_seconds=timeout_seconds)
