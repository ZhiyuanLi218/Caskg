from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation.retrievers.subprocess_adapter import PersistentSubprocessRetriever


class GoSRetriever(PersistentSubprocessRetriever):
    def __init__(
        self,
        config_path: str | Path,
        *,
        worker_path: str | Path,
        timeout_seconds: float = 180.0,
    ) -> None:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        method: dict[str, Any] = config["methods"]["gos"]
        command = [
            str(method["python"]),
            "-u",
            str(Path(worker_path).resolve()),
            "--project",
            "gos",
            "--project-root",
            str(method["project_root"]),
            "--workspace",
            str(method["workspace"]),
        ]
        super().__init__("gos", command, timeout_seconds=timeout_seconds)
