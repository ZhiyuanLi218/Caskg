"""Execution trace storage for CaSKG causal inference."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ExecutionTrace:
    """A single recorded execution trace from an agent run."""

    trace_id: str
    task_id: str
    environment: str
    task_type: str
    skills_used: list[str]
    skills_available: list[str]
    outcome: float  # 0-1
    steps: int
    timestamp: str
    is_intervention: bool = False
    intervention_edge: Optional[tuple[str, str]] = None
    intervention_type: Optional[str] = None
    skill_sequence: list[str] = field(default_factory=list)


class TraceStore:
    """Append-only JSON-lines store for execution traces.

    Each line in the storage file is a JSON-serialized ExecutionTrace.
    The store handles missing files, empty files, and malformed lines
    gracefully by skipping bad records.
    """

    def __init__(self, storage_path: str) -> None:
        self._path = Path(storage_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, trace: ExecutionTrace) -> None:
        """Append a trace to the store.

        NOTE: This method is NOT safe for concurrent writers. Concurrent
        appends from multiple processes may interleave partial JSON lines,
        producing corrupt data. Use external file locking (e.g. fcntl.flock)
        or write-to-temp-then-rename if multi-process safety is required.
        """
        data = asdict(trace)
        # Convert tuple to list for JSON serialization
        if data["intervention_edge"] is not None:
            data["intervention_edge"] = list(data["intervention_edge"])
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def recent(self, n: int = 100) -> list[ExecutionTrace]:
        """Return the last n traces."""
        return self._load_recent(n)

    def load_all(self) -> list[ExecutionTrace]:
        """Return all traces from the storage file.

        Public wrapper around the internal _load_all() method.
        """
        return self._load_all()

    def by_edge(self, source: str, target: str) -> list[ExecutionTrace]:
        """Return traces where intervention_edge matches (source, target)."""
        results = []
        for trace in self._load_all():
            if trace.intervention_edge == (source, target):
                results.append(trace)
        return results

    def by_environment(self, env: str) -> list[ExecutionTrace]:
        """Return traces matching the given environment."""
        results = []
        for trace in self._load_all():
            if trace.environment == env:
                results.append(trace)
        return results

    def cooccurrence_matrix(
        self, skill_names: list[str], window: int = 200
    ) -> dict[tuple[str, str], int]:
        """Count co-occurrences of skill pairs in recent successful traces.

        Only considers traces with outcome > 0.5 (successful).
        Returns a dict mapping (skill_a, skill_b) -> count where
        skill_a < skill_b lexicographically to avoid duplicates.
        """
        traces = self._load_recent(window)
        counts: dict[tuple[str, str], int] = {}

        skill_set = set(skill_names)
        for trace in traces:
            if trace.outcome <= 0.5:
                continue
            used = sorted(set(trace.skills_used) & skill_set)
            for i in range(len(used)):
                for j in range(i + 1, len(used)):
                    pair = (used[i], used[j])
                    counts[pair] = counts.get(pair, 0) + 1

        return counts

    def temporal_precedence(self, source: str, target: str) -> float:
        """Compute P(source appears before target in skills_used lists).

        Returns the fraction of traces containing both skills where
        source appears at an earlier index than target. Returns 0.5
        if no traces contain both skills.
        """
        both_count = 0
        source_first_count = 0

        for trace in self._load_all():
            if source in trace.skills_used and target in trace.skills_used:
                both_count += 1
                source_idx = trace.skills_used.index(source)
                target_idx = trace.skills_used.index(target)
                if source_idx < target_idx:
                    source_first_count += 1

        if both_count == 0:
            return 0.5
        return source_first_count / both_count

    def _load_all(self) -> list[ExecutionTrace]:
        """Read all traces from the storage file."""
        if not self._path.exists():
            return []
        traces = []
        try:
            with self._path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    trace = self._parse_line(line)
                    if trace is not None:
                        traces.append(trace)
        except (OSError, IOError):
            return []
        return traces

    def _load_recent(self, n: int) -> list[ExecutionTrace]:
        """Read the last n traces from the storage file."""
        all_traces = self._load_all()
        return all_traces[-n:] if len(all_traces) > n else all_traces

    @staticmethod
    def _parse_line(line: str) -> Optional[ExecutionTrace]:
        """Parse a single JSON line into an ExecutionTrace, or None on failure."""
        try:
            data = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return None

        try:
            # Convert intervention_edge from list back to tuple
            edge = data.get("intervention_edge")
            if edge is not None:
                if isinstance(edge, (list, tuple)) and len(edge) == 2:
                    edge = (str(edge[0]), str(edge[1]))
                else:
                    edge = None

            return ExecutionTrace(
                trace_id=str(data["trace_id"]),
                task_id=str(data["task_id"]),
                environment=str(data["environment"]),
                task_type=str(data["task_type"]),
                skills_used=list(data.get("skills_used", [])),
                skills_available=list(data.get("skills_available", [])),
                outcome=float(data["outcome"]),
                steps=int(data["steps"]),
                timestamp=str(data["timestamp"]),
                is_intervention=bool(data.get("is_intervention", False)),
                intervention_edge=edge,
                intervention_type=data.get("intervention_type"),
                skill_sequence=list(data.get("skill_sequence", [])),
            )
        except (KeyError, TypeError, ValueError):
            return None
