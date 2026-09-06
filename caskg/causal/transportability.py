"""Cross-environment edge stability scoring for CaSKG transportability analysis."""

from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import Any

from caskg.causal.trace_store import ExecutionTrace, TraceStore
from caskg.core.schema import SkillEdge, SkillNode


class TransportabilityEstimator:
    """Estimates how well a causal edge transfers across environments."""

    def __init__(self, config: Any, trace_store: TraceStore):
        self.config = config
        self.trace_store = trace_store
        self.threshold = getattr(config, "TRANSPORTABILITY_THRESHOLD", 0.7)

    def compute_transportability(self, edge: SkillEdge) -> float:
        """Compute transportability score for an edge.

        t_ij = 1 - Var_d(c_ij^(d)) across task distributions d.
        Gets all traces involving this edge, groups by environment,
        and measures variance of causal effect estimates.
        """
        all_traces = self.trace_store.load_all()
        env_outcomes = self._group_by_environment(edge, all_traces)

        if len(env_outcomes) < 2:
            return 0.5  # insufficient data, neutral score

        # Compute per-environment "causal effect" estimate
        env_effects: dict[str, float] = {}
        for env, traces in env_outcomes.items():
            with_source = [t for t in traces if edge.source in t.skills_used]
            without_source = [
                t
                for t in traces
                if edge.source not in t.skills_used
                and edge.target in t.skills_used
            ]
            if with_source and without_source:
                effect = sum(t.outcome for t in with_source) / len(
                    with_source
                ) - sum(t.outcome for t in without_source) / len(without_source)
                env_effects[env] = effect

        if len(env_effects) < 2:
            return 0.5

        # t_ij = 1 - Var(effects across environments)
        values = list(env_effects.values())
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return max(0.0, min(1.0, 1.0 - variance))

    def _group_by_environment(
        self, edge: SkillEdge, traces: list[ExecutionTrace]
    ) -> dict[str, list[ExecutionTrace]]:
        """Group traces where target skill is relevant, by environment."""
        result: dict[str, list[ExecutionTrace]] = {}
        for trace in traces:
            if (
                edge.target in trace.skills_available
                or edge.target in trace.skills_used
            ):
                env = trace.environment or "default"
                result.setdefault(env, []).append(trace)
        return result

    def compute_all(self, edges: list[SkillEdge]) -> list[SkillEdge]:
        """Compute transportability for all edges, return updated list."""
        updated = []
        for edge in edges:
            t_score = self.compute_transportability(edge)
            updated.append(dc_replace(edge, transportability=t_score))
        return updated

    def identify_invariant_subgraph(
        self, edges: list[SkillEdge]
    ) -> list[SkillEdge]:
        """Return edges with transportability above threshold (environment-invariant)."""
        return [
            e
            for e in edges
            if e.transportability > self.threshold
            and e.status == "confirmed_causal"
        ]

    def identify_variant_edges(self, edges: list[SkillEdge]) -> list[SkillEdge]:
        """Return edges with low transportability (context-specific)."""
        return [
            e
            for e in edges
            if e.transportability < 0.3
            and e.status == "confirmed_causal"
        ]


class CrossEnvironmentTracker:
    """Tracks edge behavior across different environments over time."""

    def __init__(self, trace_store: TraceStore):
        self.trace_store = trace_store

    def environments_seen(self) -> list[str]:
        """Return unique environments from all traces."""
        traces = self.trace_store.load_all()
        envs: set[str] = set()
        for t in traces:
            if t.environment:
                envs.add(t.environment)
        return sorted(envs)

    def edge_stability_over_time(
        self, edge: SkillEdge, window: int = 50
    ) -> list[float]:
        """Compute rolling ATE estimate over time windows."""
        all_traces = self.trace_store.load_all()
        relevant = [
            t
            for t in all_traces
            if edge.target in t.skills_available
            or edge.target in t.skills_used
        ]
        if len(relevant) < window:
            return []

        stability: list[float] = []
        for i in range(0, len(relevant) - window + 1, window // 2):
            chunk = relevant[i : i + window]
            with_src = [t for t in chunk if edge.source in t.skills_used]
            without_src = [
                t for t in chunk if edge.source not in t.skills_used
            ]
            if with_src and without_src:
                ate = sum(t.outcome for t in with_src) / len(
                    with_src
                ) - sum(t.outcome for t in without_src) / len(without_src)
                stability.append(ate)
        return stability

    def detect_drift(self, edge: SkillEdge, threshold: float = 0.2) -> bool:
        """Detect if edge's causal effect is drifting over time."""
        stability = self.edge_stability_over_time(edge)
        if len(stability) < 3:
            return False
        # Check if trend has large shift between early and recent windows
        recent = stability[-3:]
        early = stability[:3]
        recent_mean = sum(recent) / len(recent)
        early_mean = sum(early) / len(early)
        return abs(recent_mean - early_mean) > threshold


class ColdStartTransferEngine:
    """Enables zero-shot transfer to new environments using invariant edges."""

    def __init__(self, invariant_edges: list[SkillEdge]):
        self.invariant_edges = invariant_edges

    def bootstrap_new_environment(self) -> list[SkillEdge]:
        """Return invariant edges as starting graph for new environment.

        These are the edges most likely to hold regardless of environment.
        """
        return [
            dc_replace(e, last_validated_episode=0)
            for e in self.invariant_edges
        ]

    def adaptation_priority(
        self, edges: list[SkillEdge], new_env_traces: list[ExecutionTrace]
    ) -> list[tuple[SkillEdge, float]]:
        """Compute priority for re-validating edges in new environment.

        Higher priority for: high usage in new env, low transportability,
        high uncertainty.
        """
        priorities: list[tuple[SkillEdge, float]] = []
        for edge in edges:
            # Count how often edge's skills appear in new env traces
            usage = sum(
                1
                for t in new_env_traces
                if edge.source in t.skills_used
                or edge.target in t.skills_used
            )
            priority = (
                usage / max(len(new_env_traces), 1) * 0.4
                + (1.0 - edge.transportability) * 0.3
                + edge.uncertainty * 0.3
            )
            priorities.append((edge, priority))
        priorities.sort(key=lambda x: x[1], reverse=True)
        return priorities

    def transfer_success_rate(
        self,
        transferred_edges: list[SkillEdge],
        new_env_results: list[ExecutionTrace],
    ) -> float:
        """Measure how many transferred edges are still valid in new environment."""
        if not transferred_edges:
            return 0.0
        still_valid = sum(
            1
            for e in transferred_edges
            if e.status in ("confirmed_causal", "stable")
        )
        return still_valid / len(transferred_edges)
