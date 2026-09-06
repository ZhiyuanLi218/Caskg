from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class ProtocolError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Protocol:
    path: Path
    data: dict[str, Any]
    sha256: str

    @classmethod
    def load(cls, path: str | Path) -> "Protocol":
        resolved = Path(path).resolve()
        data = json.loads(resolved.read_text(encoding="utf-8"))
        protocol = cls(path=resolved, data=data, sha256=sha256_file(resolved))
        protocol.validate()
        return protocol

    @property
    def protocol_id(self) -> str:
        return str(self.data["protocol_id"])

    @property
    def benchmark(self) -> dict[str, Any]:
        return self.data["benchmark"]

    @property
    def selection(self) -> dict[str, Any]:
        return self.data["selection"]

    @property
    def retrieval(self) -> dict[str, Any]:
        return self.data["retrieval"]

    @property
    def agent(self) -> dict[str, Any]:
        return self.data["agent"]

    @property
    def reliability(self) -> dict[str, Any]:
        return self.data["reliability"]

    def validate(self) -> None:
        if self.data.get("schema_version") != 1:
            raise ProtocolError("Unsupported protocol schema version.")
        if self.benchmark.get("name") != "ScienceWorld":
            raise ProtocolError("Protocol benchmark must be ScienceWorld.")
        if self.benchmark.get("version") != "1.2.3":
            raise ProtocolError("ScienceWorld version must remain pinned to 1.2.3.")
        if self.benchmark.get("split") != "test":
            raise ProtocolError("Formal protocol must use the official test split.")
        if self.benchmark.get("simplification") != "":
            raise ProtocolError("Protocol v1 requires the default, unsimplified mode.")
        if self.benchmark.get("env_step_limit") != 100:
            raise ProtocolError("Protocol v1 requires env_step_limit=100.")
        if self.retrieval.get("query_source") != "raw_task_description":
            raise ProtocolError("Both retrievers must receive the raw task description.")
        if self.retrieval.get("query_rewrite") is not False:
            raise ProtocolError("External query rewriting is disabled in protocol v1.")
        if self.retrieval.get("trigger") != "episode_start_once":
            raise ProtocolError("Protocol v1 performs retrieval exactly once per episode.")


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    protocol_id: str
    task_name: str
    task_index: int
    variation_idx: int
    split: str
    simplification: str
    env_step_limit: int
    selection_rank: int
    selection_digest: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EpisodeSpec":
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "protocol_id": self.protocol_id,
            "task_name": self.task_name,
            "task_index": self.task_index,
            "variation_idx": self.variation_idx,
            "split": self.split,
            "simplification": self.simplification,
            "env_step_limit": self.env_step_limit,
            "selection_rank": self.selection_rank,
            "selection_digest": self.selection_digest,
        }


def load_manifest(
    path: str | Path,
    protocol: Protocol,
    checksum_path: str | Path | None = None,
) -> list[EpisodeSpec]:
    resolved = Path(path).resolve()
    if checksum_path is not None:
        checksum_file = Path(checksum_path).resolve()
        expected = checksum_file.read_text(encoding="ascii").strip().split()[0].lower()
        actual = sha256_file(resolved)
        if actual != expected:
            raise ProtocolError(
                f"Manifest checksum mismatch: expected {expected}, found {actual}."
            )

    rows = [
        EpisodeSpec.from_dict(json.loads(line))
        for line in resolved.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _validate_manifest_rows(rows, protocol)
    return rows


def _validate_manifest_rows(rows: Iterable[EpisodeSpec], protocol: Protocol) -> None:
    materialized = list(rows)
    expected_count = int(protocol.selection["expected_episode_count"])
    if len(materialized) != expected_count:
        raise ProtocolError(
            f"Manifest has {len(materialized)} episodes; expected {expected_count}."
        )

    ids: set[str] = set()
    task_names: set[str] = set()
    per_task: dict[str, int] = {}
    for row in materialized:
        if row.episode_id in ids:
            raise ProtocolError(f"Duplicate episode id: {row.episode_id}")
        ids.add(row.episode_id)
        task_names.add(row.task_name)
        per_task[row.task_name] = per_task.get(row.task_name, 0) + 1
        if row.protocol_id != protocol.protocol_id:
            raise ProtocolError(f"Protocol mismatch for {row.episode_id}.")
        if row.split != protocol.benchmark["split"]:
            raise ProtocolError(f"Split mismatch for {row.episode_id}.")
        if row.simplification != protocol.benchmark["simplification"]:
            raise ProtocolError(f"Simplification mismatch for {row.episode_id}.")
        if row.env_step_limit != protocol.benchmark["env_step_limit"]:
            raise ProtocolError(f"Step-limit mismatch for {row.episode_id}.")

    expected_tasks = int(protocol.benchmark["task_count"])
    if len(task_names) != expected_tasks:
        raise ProtocolError(
            f"Manifest covers {len(task_names)} tasks; expected {expected_tasks}."
        )
    cap = int(protocol.selection["max_variations_per_task"])
    if any(count > cap for count in per_task.values()):
        raise ProtocolError("A task exceeds the configured variation cap.")
