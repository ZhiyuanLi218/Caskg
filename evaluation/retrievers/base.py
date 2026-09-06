from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class RetrievalInfrastructureError(RuntimeError):
    """Retrieval failed for reasons outside the evaluated agent."""


@dataclass(frozen=True)
class RetrievalBundle:
    method: str
    query: str
    status: str
    rendered_context: str
    requested_top_n: int = 0
    skills: list[dict[str, Any]] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    seeds: list[dict[str, Any]] = field(default_factory=list)
    latency_seconds: float = 0.0
    workspace: str = ""
    workspace_fingerprint: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RetrievalBundle":
        return cls(
            method=str(value["method"]),
            query=str(value["query"]),
            status=str(value["status"]),
            rendered_context=str(value.get("rendered_context", "")),
            requested_top_n=int(value.get("requested_top_n", 0)),
            skills=list(value.get("skills", [])),
            relations=list(value.get("relations", [])),
            seeds=list(value.get("seeds", [])),
            latency_seconds=float(value.get("latency_seconds", 0.0)),
            workspace=str(value.get("workspace", "")),
            workspace_fingerprint=str(value.get("workspace_fingerprint", "")),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "query": self.query,
            "status": self.status,
            "requested_top_n": self.requested_top_n,
            "rendered_context_chars": len(self.rendered_context),
            "rendered_context": self.rendered_context,
            "skill_names": [str(skill.get("name", "")) for skill in self.skills],
            "skills": self.skills,
            "relations": self.relations,
            "seeds": self.seeds,
            "latency_seconds": self.latency_seconds,
            "workspace": self.workspace,
            "workspace_fingerprint": self.workspace_fingerprint,
        }


class Retriever(Protocol):
    method: str

    def retrieve(
        self,
        query: str,
        *,
        top_n: int,
        max_chars_per_skill: int,
        max_context_chars: int,
    ) -> RetrievalBundle: ...

    def close(self) -> None: ...


class NullRetriever:
    method = "none"

    def retrieve(
        self,
        query: str,
        *,
        top_n: int,
        max_chars_per_skill: int,
        max_context_chars: int,
    ) -> RetrievalBundle:
        del top_n, max_chars_per_skill, max_context_chars
        return RetrievalBundle(
            method=self.method,
            query=query,
            status="NO_RETRIEVER",
            rendered_context="",
        )

    def close(self) -> None:
        return None
