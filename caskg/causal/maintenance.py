"""Causal graph maintenance operations for CaSKG.

Implements health diagnostics, merge/retire/split operations, and the
full evolution cycle that keeps the skill graph causally valid over time.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, replace as dc_replace
from typing import Any

from caskg.causal.trace_store import ExecutionTrace, TraceStore
from caskg.core.schema import GraphHealthReport, SkillEdge, SkillNode


class CausalHealthDiagnostics:
    """Multi-dimensional health diagnostics for the causal skill graph."""

    def __init__(self, config: Any, trace_store: TraceStore):
        self.config = config
        self.trace_store = trace_store

    def diagnose(
        self, edges: list[SkillEdge], nodes: list[SkillNode]
    ) -> GraphHealthReport:
        """Run a full diagnostic pass across four dimensions."""
        # Dimension 1: Causal validity - find edges whose causal_score dropped
        recent_traces = self.trace_store.recent(200)
        decayed = self._find_decayed_edges(edges, recent_traces)

        # Dimension 2: Coverage - frequently co-occurring pairs without edges
        missing = self._find_missing_candidates(nodes, edges, recent_traces)

        # Dimension 3: Structural - check for anomalies
        anomalies = self._check_structural_health(edges, nodes)

        # Dimension 4: Cross-env consistency
        variants = self._find_variant_edges(edges, recent_traces)

        coverage = 1.0 - len(missing) / max(len(edges) + len(missing), 1)
        return GraphHealthReport(
            decayed_edges=[
                {"source": e.source, "target": e.target, "causal_score": e.causal_score}
                for e in decayed
            ],
            missing_edge_candidates=[
                {"source": pair[0], "target": pair[1]} for pair in missing
            ],
            structural_anomalies=anomalies,
            variant_edges=[
                {"source": e.source, "target": e.target, "causal_score": e.causal_score}
                for e in variants
            ],
            coverage_score=max(0.0, min(1.0, coverage)),
        )

    def _find_decayed_edges(
        self, edges: list[SkillEdge], traces: list[ExecutionTrace]
    ) -> list[SkillEdge]:
        """Find edges that were confirmed but recent traces suggest invalidity.

        Heuristic: if in recent traces, the source skill appears and succeeds
        but the target still fails (or is unused), the edge may have decayed.
        """
        decayed: list[SkillEdge] = []

        # Build a map of skill success/failure from recent traces
        skill_success_count: dict[str, int] = defaultdict(int)
        skill_failure_count: dict[str, int] = defaultdict(int)

        for trace in traces:
            for skill in trace.skills_used:
                if trace.outcome > 0.5:
                    skill_success_count[skill] += 1
                else:
                    skill_failure_count[skill] += 1

        for edge in edges:
            if edge.status not in ("confirmed_causal", "stable"):
                continue
            source_successes = skill_success_count.get(edge.source, 0)
            target_failures = skill_failure_count.get(edge.target, 0)
            target_successes = skill_success_count.get(edge.target, 0)

            # If source succeeds often but target still fails frequently,
            # the dependency edge may no longer be valid
            if source_successes > 3 and target_failures > 0:
                target_total = target_successes + target_failures
                if target_total > 0:
                    failure_rate = target_failures / target_total
                    if failure_rate > 0.5:
                        decayed.append(edge)

        return decayed

    def _find_missing_candidates(
        self,
        nodes: list[SkillNode],
        edges: list[SkillEdge],
        traces: list[ExecutionTrace],
    ) -> list[tuple[str, str]]:
        """Find pairs that co-occur frequently in successful traces but have no edge."""
        # Build set of existing edge pairs
        existing_pairs: set[tuple[str, str]] = set()
        for edge in edges:
            existing_pairs.add((edge.source, edge.target))
            existing_pairs.add((edge.target, edge.source))

        # Count co-occurrences in successful traces
        node_names = {n.name for n in nodes}
        cooccurrence: dict[tuple[str, str], int] = defaultdict(int)

        for trace in traces:
            if trace.outcome <= 0.5:
                continue
            used_in_graph = sorted(set(trace.skills_used) & node_names)
            for i in range(len(used_in_graph)):
                for j in range(i + 1, len(used_in_graph)):
                    pair = (used_in_graph[i], used_in_graph[j])
                    cooccurrence[pair] += 1

        # Threshold: pairs that co-occur at least 3 times without an edge
        threshold = getattr(self.config, "COOCCURRENCE_THRESHOLD", 3)
        missing: list[tuple[str, str]] = []
        for pair, count in cooccurrence.items():
            if count >= threshold and pair not in existing_pairs:
                missing.append(pair)

        return missing

    def _check_structural_health(
        self, edges: list[SkillEdge], nodes: list[SkillNode]
    ) -> list[str]:
        """Check for structural anomalies in the graph.

        Detects:
        - Dependency chains longer than 5
        - Isolated confirmed edges (source or target not in node list)
        - Potential cycles
        """
        anomalies: list[str] = []
        node_names = {n.name for n in nodes}

        # Check for edges referencing unknown nodes
        for edge in edges:
            if edge.source not in node_names:
                anomalies.append(
                    f"Edge {edge.source}->{edge.target}: source not in node set"
                )
            if edge.target not in node_names:
                anomalies.append(
                    f"Edge {edge.source}->{edge.target}: target not in node set"
                )

        # Build adjacency for chain length and cycle detection
        adjacency: dict[str, list[str]] = defaultdict(list)
        for edge in edges:
            if edge.status in ("confirmed_causal", "stable", "unverified"):
                adjacency[edge.source].append(edge.target)

        # Check for long chains via DFS from each node
        def _max_chain_from(start: str) -> int:
            visited: set[str] = set()
            stack: list[tuple[str, int]] = [(start, 0)]
            max_depth = 0
            while stack:
                current, depth = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                max_depth = max(max_depth, depth)
                for neighbor in adjacency.get(current, []):
                    if neighbor not in visited:
                        stack.append((neighbor, depth + 1))
            return max_depth

        for node in nodes:
            chain_len = _max_chain_from(node.name)
            if chain_len > 5:
                anomalies.append(
                    f"Long dependency chain (depth={chain_len}) starting from {node.name}"
                )

        # Check for cycles using iterative DFS with coloring
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {n.name: WHITE for n in nodes}

        def _has_cycle_from(start: str) -> bool:
            stack: list[tuple[str, bool]] = [(start, False)]
            while stack:
                node_name, backtrack = stack.pop()
                if backtrack:
                    color[node_name] = BLACK
                    continue
                if color[node_name] == GRAY:
                    return True
                if color[node_name] == BLACK:
                    continue
                color[node_name] = GRAY
                stack.append((node_name, True))
                for neighbor in adjacency.get(node_name, []):
                    if color.get(neighbor, WHITE) == GRAY:
                        return True
                    if color.get(neighbor, WHITE) == WHITE:
                        stack.append((neighbor, False))
            return False

        for node in nodes:
            if color.get(node.name, WHITE) == WHITE:
                if _has_cycle_from(node.name):
                    anomalies.append(f"Potential cycle detected involving {node.name}")

        return anomalies

    def _find_variant_edges(
        self, edges: list[SkillEdge], traces: list[ExecutionTrace]
    ) -> list[SkillEdge]:
        """Find edges with very different causal outcomes across environments.

        An edge is variant if the outcome when both source and target are used
        differs significantly across environments.
        """
        variants: list[SkillEdge] = []

        # Group traces by environment
        env_traces: dict[str, list[ExecutionTrace]] = defaultdict(list)
        for trace in traces:
            env_key = trace.environment or "default"
            env_traces[env_key].append(trace)

        if len(env_traces) < 2:
            return variants

        for edge in edges:
            if edge.status not in ("confirmed_causal", "stable"):
                continue

            # Compute average outcome per environment for traces using both skills
            env_outcomes: dict[str, list[float]] = {}
            for env, env_trace_list in env_traces.items():
                outcomes = [
                    t.outcome
                    for t in env_trace_list
                    if edge.source in t.skills_used and edge.target in t.skills_used
                ]
                if outcomes:
                    env_outcomes[env] = outcomes

            if len(env_outcomes) < 2:
                continue

            # Compute variance of per-environment means
            env_means = [sum(v) / len(v) for v in env_outcomes.values()]
            overall_mean = sum(env_means) / len(env_means)
            variance = sum((m - overall_mean) ** 2 for m in env_means) / len(env_means)

            threshold = getattr(self.config, "VARIANT_VARIANCE_THRESHOLD", 0.1)
            if variance > threshold:
                variants.append(edge)

        return variants


class CausalMergeOperation:
    """Determines if two skill nodes can be merged based on substitutability."""

    def __init__(self, config: Any):
        self.config = config

    def compute_substitutability(
        self, node_i: SkillNode, node_j: SkillNode, trace_store: TraceStore
    ) -> float:
        """Compute I_merge(i,j) = 1 - avg(|P(Y|do(si)) - P(Y|do(sj))|).

        Uses traces where one skill was used vs the other to estimate
        interventional outcome differences.
        """
        recent = trace_store.recent(500)
        traces_i = [t for t in recent if node_i.name in t.skills_used]
        traces_j = [t for t in recent if node_j.name in t.skills_used]
        if not traces_i or not traces_j:
            return 0.0
        avg_i = sum(t.outcome for t in traces_i) / len(traces_i)
        avg_j = sum(t.outcome for t in traces_j) / len(traces_j)
        return 1.0 - abs(avg_i - avg_j)

    def should_merge(
        self, node_i: SkillNode, node_j: SkillNode, trace_store: TraceStore
    ) -> bool:
        """Return True if the two nodes are substitutable enough to merge."""
        score = self.compute_substitutability(node_i, node_j, trace_store)
        threshold = getattr(self.config, "MERGE_SUBSTITUTABILITY_THRESHOLD", 0.9)
        return score >= threshold


class CausalRetireOperation:
    """Determines if a skill node should be retired based on marginal contribution."""

    def __init__(self, config: Any):
        self.config = config

    def compute_marginal_contribution(
        self, node: SkillNode, trace_store: TraceStore
    ) -> float:
        """Compute R_i = E[P(Y|do(si=1)) - P(Y|do(si=0))].

        Estimates the causal effect of using a skill by comparing outcomes
        in traces where the skill was used vs traces where it was available
        but not used.
        """
        recent = trace_store.recent(500)
        traces_with = [t for t in recent if node.name in t.skills_used]
        traces_without = [
            t
            for t in recent
            if node.name in t.skills_available and node.name not in t.skills_used
        ]
        if not traces_with:
            return 0.0
        avg_with = sum(t.outcome for t in traces_with) / len(traces_with)
        avg_without = (
            sum(t.outcome for t in traces_without) / len(traces_without)
            if traces_without
            else 0.5
        )
        return avg_with - avg_without

    def should_retire(self, node: SkillNode, trace_store: TraceStore) -> bool:
        """Return True if the node's marginal contribution is below threshold."""
        contribution = self.compute_marginal_contribution(node, trace_store)
        threshold = getattr(self.config, "RETIRE_MARGINAL_THRESHOLD", 0.05)
        return contribution < threshold


class CausalSplitOperation:
    """Determines if a skill node should be split based on context variance."""

    def __init__(self, config: Any):
        self.config = config

    def detect_context_variance(
        self, node: SkillNode, trace_store: TraceStore
    ) -> dict[str, float]:
        """Group traces by environment/task_type and compute per-context outcome.

        Returns a mapping from context label to average outcome for that context.
        """
        traces = [t for t in trace_store.recent(500) if node.name in t.skills_used]
        context_outcomes: dict[str, list[float]] = {}
        for t in traces:
            ctx = t.task_type or t.environment
            context_outcomes.setdefault(ctx, []).append(t.outcome)
        return {
            ctx: sum(outcomes) / len(outcomes)
            for ctx, outcomes in context_outcomes.items()
            if outcomes
        }

    def should_split(self, node: SkillNode, trace_store: TraceStore) -> bool:
        """Return True if the skill performs very differently across contexts."""
        ctx_scores = self.detect_context_variance(node, trace_store)
        if len(ctx_scores) < 2:
            return False
        values = list(ctx_scores.values())
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        threshold = getattr(self.config, "SPLIT_VARIANCE_THRESHOLD", 0.1)
        return variance > threshold


class EvolutionCycle:
    """Orchestrates a full graph evolution cycle: diagnose, decay, discover, reinforce, prune."""

    def __init__(self, config: Any, trace_store: TraceStore):
        self.config = config
        self.trace_store = trace_store
        self.diagnostics = CausalHealthDiagnostics(config, trace_store)
        self.merge_op = CausalMergeOperation(config)
        self.retire_op = CausalRetireOperation(config)
        self.split_op = CausalSplitOperation(config)

    def execute(
        self, edges: list[SkillEdge], nodes: list[SkillNode], episode: int
    ) -> tuple[list[SkillEdge], GraphHealthReport, float]:
        """Execute one full evolution cycle.

        Steps:
        1. Diagnose current graph health
        2. Decay edges that are no longer supported by traces
        3. Discover new candidate edges for missing pairs
        4. Reinforce stable edges
        5. Prune long-rejected edges
        6. Condition variant edges with environment context
        7. Compute unverified_ratio for p_explore adjustment

        Returns the updated edge list, the health report, and the
        unverified_ratio (fraction of edges still unverified) so the
        caller can adjust p_explore accordingly.
        """
        # Step 1: Diagnose
        report = self.diagnostics.diagnose(edges, nodes)

        # Step 2: DECAY - move decayed edges back to unverified
        edges = self._apply_decay(edges, report)

        # Step 3: DISCOVER - create new candidate edges for missing pairs
        new_candidates = self._discover_candidates(report, nodes)
        edges.extend(new_candidates)

        # Step 4: REINFORCE - boost stable edges
        edges = self._reinforce_stable(edges)

        # Step 5: PRUNE - remove long-rejected edges
        edges = self._prune_rejected(edges, episode)

        # Step 6: CONDITION - attach context_condition to variant edges
        edges = self._condition_variant_edges(edges, report)

        # Step 7: Compute unverified_ratio for p_explore adjustment
        unverified_ratio = self._compute_unverified_ratio(edges)

        return edges, report, unverified_ratio

    def _apply_decay(
        self, edges: list[SkillEdge], report: GraphHealthReport
    ) -> list[SkillEdge]:
        """Set status='unverified' for decayed edges identified in the report."""
        decayed_keys: set[tuple[str, str]] = set()
        for entry in report.decayed_edges:
            decayed_keys.add((entry["source"], entry["target"]))

        result: list[SkillEdge] = []
        for edge in edges:
            if (edge.source, edge.target) in decayed_keys:
                result.append(
                    dc_replace(edge, status="unverified", causal_score=edge.causal_score * 0.5)
                )
            else:
                result.append(edge)
        return result

    def _discover_candidates(
        self, report: GraphHealthReport, nodes: list[SkillNode]
    ) -> list[SkillEdge]:
        """Create new unverified edges for missing pairs found during diagnosis."""
        node_names = {n.name for n in nodes}
        new_edges: list[SkillEdge] = []

        for candidate in report.missing_edge_candidates:
            source = candidate["source"]
            target = candidate["target"]
            if source in node_names and target in node_names:
                new_edges.append(
                    SkillEdge(
                        source=source,
                        target=target,
                        description=f"Auto-discovered candidate: {source} -> {target}",
                        type="dependency",
                        weight=0.5,
                        confidence=0.3,
                        causal_score=0.0,
                        uncertainty=1.0,
                        status="unverified",
                    )
                )
        return new_edges

    def _reinforce_stable(self, edges: list[SkillEdge]) -> list[SkillEdge]:
        """For confirmed edges with recent trace evidence, boost confidence.

        Only reinforces edges where both source and target appear together in
        recent successful traces. Edges without recent trace support are left
        unchanged rather than blindly boosted.
        """
        factor = getattr(self.config, "STABLE_REINFORCEMENT_FACTOR", 1.02)

        # Build set of (source, target) pairs seen together in recent successful traces
        recent_traces = self.trace_store.recent(200)
        supported_pairs: set[tuple[str, str]] = set()
        for trace in recent_traces:
            if trace.outcome <= 0.5:
                continue
            skills_set = set(trace.skills_used)
            for edge in edges:
                if edge.source in skills_set and edge.target in skills_set:
                    supported_pairs.add((edge.source, edge.target))

        result: list[SkillEdge] = []
        for edge in edges:
            if edge.status in ("confirmed_causal", "stable"):
                # Only reinforce if recent traces support this edge
                if (edge.source, edge.target) in supported_pairs:
                    new_confidence = min(0.99, edge.confidence * factor)
                    result.append(dc_replace(edge, confidence=new_confidence))
                else:
                    result.append(edge)
            else:
                result.append(edge)
        return result

    def _prune_rejected(self, edges: list[SkillEdge], episode: int) -> list[SkillEdge]:
        """Remove edges in rejected status for more than PRUNE_AGE_THRESHOLD episodes."""
        prune_age = getattr(self.config, "PRUNE_AGE_THRESHOLD", 50)
        result: list[SkillEdge] = []
        for edge in edges:
            if edge.status == "rejected_non_causal":
                age = episode - edge.last_validated_episode
                if age > prune_age:
                    # Skip this edge (prune it)
                    continue
            result.append(edge)
        return result

    def _condition_variant_edges(
        self, edges: list[SkillEdge], report: GraphHealthReport
    ) -> list[SkillEdge]:
        """Attach context_condition to variant edges identified in the health report.

        For each variant edge, determines which environments it has a positive
        causal effect in, and records that as a JSON context_condition field.
        Uses the same environment-grouped trace analysis as _find_variant_edges.
        """
        variant_keys: set[tuple[str, str]] = set()
        for entry in report.variant_edges:
            variant_keys.add((entry["source"], entry["target"]))

        if not variant_keys:
            return edges

        # Group recent traces by environment (mirrors _find_variant_edges logic)
        recent_traces = self.trace_store.recent(200)
        env_traces: dict[str, list[ExecutionTrace]] = defaultdict(list)
        for trace in recent_traces:
            env_key = trace.environment or "default"
            env_traces[env_key].append(trace)

        result: list[SkillEdge] = []
        for edge in edges:
            if (edge.source, edge.target) not in variant_keys:
                result.append(edge)
                continue

            # Determine which environments this edge has positive effect in
            valid_envs: list[str] = []
            for env, env_trace_list in env_traces.items():
                outcomes_with = [
                    t.outcome
                    for t in env_trace_list
                    if edge.source in t.skills_used and edge.target in t.skills_used
                ]
                outcomes_without = [
                    t.outcome
                    for t in env_trace_list
                    if edge.target in t.skills_used and edge.source not in t.skills_used
                ]
                if outcomes_with:
                    avg_with = sum(outcomes_with) / len(outcomes_with)
                    avg_without = (
                        sum(outcomes_without) / len(outcomes_without)
                        if outcomes_without
                        else 0.5
                    )
                    # Positive effect: using both skills together improves outcome
                    if avg_with > avg_without:
                        valid_envs.append(env)

            condition = json.dumps({"valid_envs": valid_envs})
            result.append(dc_replace(edge, context_condition=condition))

        return result

    def _compute_unverified_ratio(self, edges: list[SkillEdge]) -> float:
        """Compute the fraction of edges that are still unverified.

        This ratio is used by the caller to adjust p_explore: higher
        unverified_ratio means more exploration is needed.
        """
        if not edges:
            return 0.0
        unverified_count = sum(1 for e in edges if e.status == "unverified")
        return unverified_count / len(edges)
