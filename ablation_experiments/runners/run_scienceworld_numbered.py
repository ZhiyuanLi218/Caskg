"""Run the formal P0/A1--A4 ScienceWorld conditions with the accepted evaluator."""

from __future__ import annotations

import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCIENCE_ROOT = REPOSITORY_ROOT
sys.path.insert(0, str(SCIENCE_ROOT))
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(1, str(REPOSITORY_ROOT))

from evaluation import scienceworld_eto211_run as evaluator  # noqa: E402
from ablation_experiments.adapters.causal_graph_views import sha256_file  # noqa: E402
from ablation_experiments.adapters.numbered_scienceworld import (  # noqa: E402
    NumberedScienceWorldRetriever,
)


HISTORICAL_A0_EVALUATOR_FINGERPRINT = (
    "a5aac3d1988ad4a0050f2bedb3ad8759fa998123a604db414d7686627562ae92"
)
FORMAL_EVALUATOR_FINGERPRINT = (
    "6973cd44cf0a64bc829648250d54f1364b194212fcb5dd222148e279d17c023a"
)


def _argument_value(name: str) -> str:
    try:
        index = sys.argv.index(name)
    except ValueError as exc:
        raise ValueError(f"Missing required argument: {name}") from exc
    if index + 1 >= len(sys.argv):
        raise ValueError(f"Missing value for argument: {name}")
    return sys.argv[index + 1]


def _cache_stats(path_value: str) -> dict[str, object]:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        return {"path": str(path), "entry_count": 0, "exists": False}
    with closing(sqlite3.connect(path)) as connection:
        count = int(connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])
    return {"path": str(path), "entry_count": count, "exists": True}


def main() -> int:
    actual_fingerprint = evaluator._evaluator_fingerprint(SCIENCE_ROOT)
    if actual_fingerprint != FORMAL_EVALUATOR_FINGERPRINT:
        raise RuntimeError(
            "ScienceWorld evaluator no longer matches the accepted formal evaluator: "
            f"{actual_fingerprint}"
        )
    config_path = Path(_argument_value("--retriever-config")).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    method = config["methods"]["caskg"]
    output_root = Path(_argument_value("--output-dir")).resolve()
    model = _argument_value("--model")

    evaluator.CaSKGRetriever = NumberedScienceWorldRetriever
    exit_code = evaluator.main()

    summary_path = (
        output_root
        / "eto_skillnet_unseen211"
        / "caskg"
        / model
        / "summary.json"
    )
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(
            {
                "formal_evaluator_fingerprint": FORMAL_EVALUATOR_FINGERPRINT,
                "historical_a0_evaluator_fingerprint": (
                    HISTORICAL_A0_EVALUATOR_FINGERPRINT
                ),
                "numbered_runtime_adapter_sha256": sha256_file(
                    REPOSITORY_ROOT
                    / "ablation_experiments"
                    / "adapters"
                    / "numbered_runtime.py"
                ),
                "numbered_project_worker_sha256": sha256_file(
                    REPOSITORY_ROOT
                    / "ablation_experiments"
                    / "adapters"
                    / "numbered_project_worker.py"
                ),
                "embedding_cache": _cache_stats(method["embedding_cache"]),
                "retrieval_audit_dir": str(method["audit_dir"]),
            }
        )
        temporary = summary_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
            newline="\n",
        )
        temporary.replace(summary_path)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
