"""Active intervention scheduler for CaSKG.

Implements edge value scoring, active scheduling with exploration/exploitation,
group intervention planning, and opportunistic probing during normal tasks.
"""

from __future__ import annotations

import random
from math import sqrt
from typing import Any

from caskg.causal.trace_store import TraceStore
from caskg.core.schema import SkillEdge, SkillNode


class EdgeValueFunction:
    """Computes the value of testing a given edge via multi-factor scoring.

    V(e) = alpha*usage + beta*centrality + gamma*uncertainty
           + delta*failure_exposure + epsilon*maintenance_impact
    """

    def __init__(self, config: Any) -> None:
        self.config = config

    def compute(
        self,
        edge: SkillEdge,
        usage: float,
        centrality: float,
        failure_exposure: float,
        maintenance_impact: float,
    ) -> float:
        """Compute intervention value V(e) for a single edge."""
        alpha = getattr(self.config, "SCHEDULER_ALPHA_USAGE", 0.3)
        beta = getattr(self.config, "SCHEDULER_BETA_CENTRALITY", 0.2)
        gamma = getattr(self.config, "SCHEDULER_GAMMA_UNCERTAINTY", 0.25)
        delta = getattr(self.config, "SCHEDULER_DELTA_FAILURE", 0.15)
        epsilon = getattr(self.config, "SCHEDULER_EPSILON_MAINTENANCE", 0.1)

        uncertainty = edge.uncertainty

        value = (
            alpha * usage
            + beta * centrality
            + gamma * uncertainty
            + delta * failure_exposure
            + epsilon * maintenance_impact
        )
        return value

    def compute_batch(
        self,
        edges: list[SkillEdge],
        trace_store: TraceStore,
        betweenness: dict[str, float] | None = None,
    ) -> list[tuple[SkillEdge, float]]:
        """Compute V(e) for all edges, deriving usage/failure from trace_store.

        Parameters
        ----------
        edges:
            List of edges to score.
        trace_store:
            Trace store for computing usage and failure metrics.
        betweenness:
            Optional precomputed betweenness centrality dict (node_name -> score).
            If None, centrality defaults to 0.0 for all nodes.

        Returns
        -------
        List of (edge, value) tuples sorted descending by value.
        """
        if betweenness is None:
            betweenness = {}

        recent_traces = trace_store.recent(200)

        # Precompute usage counts: how often each skill pair co-occurs
        edge_usage: dict[tuple[str, str], int] = {}
        edge_failure: dict[tuple[str, str], int] = {}
        for trace in recent_traces:
            skills_set = set(trace.skills_used)
            for e in edges:
                if e.source in skills_set and e.target in skills_set:
                    key = (e.source, e.target)
                    edge_usage[key] = edge_usage.get(key, 0) + 1
                    if trace.outcome < 0.5:
                        edge_failure[key] = edge_failure.get(key, 0) + 1

        total_traces = max(len(recent_traces), 1)

        scored: list[tuple[SkillEdge, float]] = []
        for edge in edges:
            key = (edge.source, edge.target)

            # Usage: fraction of recent traces involving this edge's skills
            usage = edge_usage.get(key, 0) / total_traces

            # Centrality: average betweenness of source and target
            src_centrality = betweenness.get(edge.source, 0.0)
            tgt_centrality = betweenness.get(edge.target, 0.0)
            centrality = (src_centrality + tgt_centrality) / 2.0

            # Failure exposure: fraction of involving traces that failed
            involving = edge_usage.get(key, 0)
            failure_exposure = (
                edge_failure.get(key, 0) / involving if involving > 0 else 0.0
            )

            # Maintenance impact: proxy based on intervention count and uncertainty
            # Edges tested many times with still-high uncertainty need maintenance
            maintenance_impact = (
                edge.uncertainty / sqrt(edge.intervention_count + 1)
            )

            value = self.compute(
                edge, usage, centrality, failure_exposure, maintenance_impact
            )
            scored.append((edge, value))

        # Sort descending by value
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored


class ActiveScheduler:
    """Manages the active intervention queue with exploration/exploitation balance."""

    def __init__(self, config: Any, trace_store: TraceStore) -> None:
        self.config = config
        self.trace_store = trace_store
        self.value_fn = EdgeValueFunction(config)
        self._queue: list[tuple[float, SkillEdge]] = []
        self._p_explore: float = getattr(config, "P_EXPLORE_INITIAL", 0.3)
        self._total_probes: int = 0

    @property
    def exploration_rate(self) -> float:
        """Current exploration probability."""
        return self._p_explore

    @property
    def budget_remaining(self) -> int:
        """Number of intervention probes remaining in the budget."""
        return max(0, getattr(self.config, "MAX_INTERVENTION_BUDGET", 1000) - self._total_probes)

    def rebuild_queue(
        self,
        edges: list[SkillEdge],
        betweenness: dict[str, float] | None = None,
    ) -> None:
        """Recompute values for all unverified edges and rebuild the queue.

        Only unverified edges are candidates for intervention scheduling.
        """
        unverified = [e for e in edges if e.status == "unverified"]
        scored = self.value_fn.compute_batch(
            unverified, self.trace_store, betweenness
        )
        self._queue = [(v, e) for e, v in scored]

    def select_top_k(self, k: int) -> list[SkillEdge]:
        """Pop the top-k highest-value edges from the queue.

        Returns up to k edges, removing them from the internal queue.
        """
        selected: list[SkillEdge] = []
        for _ in range(min(k, len(self._queue))):
            if self._queue:
                _, edge = self._queue.pop(0)
                selected.append(edge)
        return selected

    def should_explore(self) -> bool:
        """Decide whether to perform an exploratory probe.

        Returns True with probability p_explore, provided budget remains.
        """
        return random.random() < self._p_explore and self.budget_remaining > 0

    def update_exploration_rate(self, unverified_ratio: float) -> None:
        """Adapt exploration rate based on the fraction of unverified edges.

        p_explore = max(P_EXPLORE_MIN, P_EXPLORE_INITIAL * unverified_ratio)
        """
        self._p_explore = max(
            getattr(self.config, "P_EXPLORE_MIN", 0.05),
            getattr(self.config, "P_EXPLORE_INITIAL", 0.3) * unverified_ratio,
        )

    def record_probe_used(self) -> None:
        """Record that one intervention probe has been consumed."""
        self._total_probes += 1


class GroupInterventionPlanner:
    """Plans grouped interventions for efficient batch causal testing.

    Uses a group-and-split strategy: if removing an entire group of source
    skills does not affect the target, all edges in the group are non-causal.
    If it does affect the target, binary splitting narrows down the causal edges.
    """

    def __init__(self) -> None:
        pass

    def plan_groups(
        self,
        target: str,
        candidate_sources: list[str],
        max_group_size: int = 8,
    ) -> list[list[str]]:
        """Split candidates into groups of max_group_size for batch testing.

        Parameters
        ----------
        target:
            The target skill under investigation.
        candidate_sources:
            Source skills that might causally influence the target.
        max_group_size:
            Maximum number of sources per group.

        Returns
        -------
        List of groups, each containing up to max_group_size source names.
        """
        groups: list[list[str]] = []
        for i in range(0, len(candidate_sources), max_group_size):
            groups.append(candidate_sources[i : i + max_group_size])
        return groups

    def split_group(self, group: list[str]) -> tuple[list[str], list[str]]:
        """Binary split a group for recursive significance testing.

        Parameters
        ----------
        group:
            A group of source skills that collectively showed significance.

        Returns
        -------
        Two halves of the group for further testing.
        """
        mid = len(group) // 2
        return group[:mid], group[mid:]

    def resolve_group_result(
        self, group: list[str], target: str, significant: bool
    ) -> list[tuple[str, str, bool]]:
        """Interpret the result of a group intervention test.

        Parameters
        ----------
        group:
            The group of source skills that were simultaneously removed.
        target:
            The target skill being tested.
        significant:
            Whether removing the group significantly affected the target.

        Returns
        -------
        - If not significant: all (source, target, False) -- non-causal edges.
        - If significant and len==1: [(source, target, True)] -- confirmed causal.
        - If significant and len>1: empty list -- needs further binary splitting.
        """
        if not significant:
            return [(s, target, False) for s in group]
        if len(group) == 1:
            return [(group[0], target, True)]
        return []

    def recursive_resolve(
        self,
        group: list[str],
        target: str,
        test_fn: Any,
    ) -> list[tuple[str, str, bool]]:
        """Full binary search resolution of a significant group.

        Implements the complete binary search algorithm from the proposal:
        1. Test the full group. If not significant, all edges are non-causal.
        2. If significant and |group| == 1, the single source is causal.
        3. If significant and |group| > 1, split in half and recurse on each half.
        4. Collect and return all resolved (source, target, is_causal) results.

        Parameters
        ----------
        group:
            Source skills to test as a batch against the target.
        target:
            The target skill being tested.
        test_fn:
            Callable(group: list[str], target: str) -> bool that runs the
            intervention experiment and returns True if removing the group
            significantly affects the target.

        Returns
        -------
        List of (source, target, is_causal) for every source in the group,
        fully resolved via recursive binary splitting.
        """
        significant = test_fn(group, target)

        if not significant:
            # Entire group is non-causal
            return [(s, target, False) for s in group]

        if len(group) == 1:
            # Base case: single confirmed causal source
            return [(group[0], target, True)]

        # Binary split and recurse on both halves
        left, right = self.split_group(group)
        results: list[tuple[str, str, bool]] = []
        results.extend(self.recursive_resolve(left, target, test_fn))
        results.extend(self.recursive_resolve(right, target, test_fn))
        return results


class OpportunisticProber:
    """Selects intervention probes that piggyback on normal task execution.

    When the agent is already executing a task involving certain skills,
    this prober identifies high-value unverified edges that can be tested
    with minimal additional cost.
    """

    def __init__(self, scheduler: ActiveScheduler) -> None:
        self.scheduler = scheduler

    def select_probe_for_task(
        self,
        task_skills: list[str],
        unverified_edges: list[SkillEdge],
    ) -> SkillEdge | None:
        """Find the highest-priority unverified edge relevant to the current task.

        Per proposal section 6.5: an edge is relevant if its target (B) is among
        the skills being used in the current task ("if B in relevant_skills(task)").
        Among relevant edges, the one with the highest value from the full
        EdgeValueFunction is selected as the best probe candidate.

        Parameters
        ----------
        task_skills:
            Skills involved in the current task execution.
        unverified_edges:
            All unverified edges available for probing.

        Returns
        -------
        The best edge to probe, or None if no relevant edge exists.
        """
        task_skill_set = set(task_skills)
        relevant = [
            e
            for e in unverified_edges
            if e.target in task_skill_set
        ]
        if not relevant:
            return None
        # Use the full value function via the scheduler's compute_batch for
        # multi-factor ranking (usage, centrality, uncertainty, failure, maintenance)
        scored = self.scheduler.value_fn.compute_batch(
            relevant, self.scheduler.trace_store
        )
        if scored:
            return scored[0][0]
        return relevant[0]
