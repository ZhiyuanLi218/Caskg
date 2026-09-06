"""Causal-aware retrieval scoring for the Graph-of-Skills system.

Implements CaSKG retrieval: scoring individual skills using causal edge
information, scoring skill bundles via counterfactual sufficiency, planning
execution paths using causal edge weights, and backward traversal for
prerequisite discovery.
"""

from __future__ import annotations

from typing import Any
import re

from caskg.core.schema import SkillEdge, SkillNode


CONFIRMED_EDGE_STATUSES = {"confirmed_causal", "stable"}
WEAK_SUPPORT_EDGE_STATUSES = {"deferred_uncertain"}
SUPPORT_EDGE_STATUSES = CONFIRMED_EDGE_STATUSES | WEAK_SUPPORT_EDGE_STATUSES

PREREQUISITE_EDGE_TYPES = {
    "dependency",
    "prereq",
    "prerequisite",
    "workflow",
    "enhance",
    "data-flow",
    "data_flow",
    "repair-support",
    "repair_support",
}

REDUNDANCY_EDGE_TYPES = {"similar", "alternative"}

CONFLICT_EDGE_TYPES = {"conflict", "conflicts_with"}


def _edge_type(edge: SkillEdge) -> str:
    return (getattr(edge, "type", "") or "").lower()


def _edge_status(edge: SkillEdge) -> str:
    return (getattr(edge, "status", "") or "").lower()


def _skill_namespace(skill_name: str) -> str:
    match = re.match(r"^([a-z][a-z0-9]{2,})-", (skill_name or "").lower())
    return match.group(1) if match else ""


def _same_skill_namespace(source: str, target: str) -> bool:
    source_namespace = _skill_namespace(source)
    target_namespace = _skill_namespace(target)
    return not source_namespace or not target_namespace or source_namespace == target_namespace


def _edge_reliability(edge: SkillEdge) -> float:
    status = _edge_status(edge)
    if status in CONFIRMED_EDGE_STATUSES:
        return 1.0
    if status in WEAK_SUPPORT_EDGE_STATUSES:
        if not _same_skill_namespace(edge.source, edge.target):
            return 0.0
        return 0.25
    return 0.0


def _edge_score(edge: SkillEdge) -> float:
    value = getattr(edge, "causal_score", None)
    if value is None:
        value = getattr(edge, "weight", 0.0)
    return float(value or 0.0) * _edge_reliability(edge)



def _edge_uncertainty(edge: SkillEdge) -> float:
    value = getattr(edge, "uncertainty", None)
    uncertainty = float(value if value is not None else 1.0)
    if _edge_status(edge) in WEAK_SUPPORT_EDGE_STATUSES:
        return max(uncertainty, 0.5)
    return uncertainty


def _edge_transportability(edge: SkillEdge) -> float:
    value = getattr(edge, "transportability", None)
    same_namespace = _same_skill_namespace(edge.source, edge.target)
    if value is None:
        return 1.0 if same_namespace else 0.0
    numeric = float(value or 0.0)
    if (
        numeric == 0.0
        and is_support_status_edge(edge)
        and _edge_type(edge) in PREREQUISITE_EDGE_TYPES
        and same_namespace
    ):
        # Legacy same-workspace validation states used the dataclass default
        # 0.0 even when no cross-environment transportability pass had run.
        # Keep those same-namespace support edges traversable, but require
        # explicit positive transportability before crossing skill namespaces.
        return 1.0
    return numeric


def is_confirmed_edge(edge: SkillEdge) -> bool:
    return _edge_status(edge) in CONFIRMED_EDGE_STATUSES


def is_support_status_edge(edge: SkillEdge) -> bool:
    return _edge_status(edge) in SUPPORT_EDGE_STATUSES


def is_prerequisite_edge(edge: SkillEdge) -> bool:
    """Edges that can be traversed as causal prerequisites."""
    return is_support_status_edge(edge) and _edge_type(edge) in PREREQUISITE_EDGE_TYPES


def is_redundancy_edge(edge: SkillEdge) -> bool:
    return is_confirmed_edge(edge) and _edge_type(edge) in REDUNDANCY_EDGE_TYPES


def is_conflict_edge(edge: SkillEdge) -> bool:
    return is_confirmed_edge(edge) and _edge_type(edge) in CONFLICT_EDGE_TYPES


class CausalScorer:
    """Scores individual skills using causal edge information."""

    def __init__(self, config: Any):
        self.lambda_rel = getattr(config, "CAUSAL_RETRIEVAL_LAMBDA_REL", 0.3)
        self.lambda_struct = getattr(config, "CAUSAL_RETRIEVAL_LAMBDA_STRUCT", 0.25)
        self.lambda_causalnec = getattr(config, "CAUSAL_RETRIEVAL_LAMBDA_CAUSALNEC", 0.35)
        self.lambda_unc = getattr(config, "CAUSAL_RETRIEVAL_LAMBDA_UNC", 0.1)

    def score(
        self,
        node: SkillNode,
        relevance: float,
        structural: float,
        causal_edges: list[SkillEdge],
    ) -> float:
        """Compute causal-aware score for a single skill node.

        score = lambda_rel*Rel + lambda_struct*Struct + lambda_causalnec*CausalNec - lambda_unc*Unc

        CausalNec: average causal_score of confirmed incoming support edges to this node.
        Unc: max uncertainty among connected confirmed edges.
        Similar/alternative edges are redundancy evidence, not causal necessity.
        """
        confirmed = [
            e
            for e in causal_edges
            if is_prerequisite_edge(e) and e.target == node.name
        ]
        causal_nec = sum(_edge_score(e) for e in confirmed) / max(len(confirmed), 1)
        max_unc = max((_edge_uncertainty(e) for e in confirmed), default=0.0)

        return (
            self.lambda_rel * relevance
            + self.lambda_struct * structural
            + self.lambda_causalnec * causal_nec
            - self.lambda_unc * max_unc
        )


class CausalBundleScorer:
    """Scores skill bundles using counterfactual sufficiency."""

    def __init__(self, config: Any):
        self.config = config

    def score_bundle(
        self,
        bundle: list[SkillNode],
        relevance_scores: dict[str, float],
        causal_edges: list[SkillEdge],
        task_goals: list[str] | None = None,
    ) -> float:
        """Compute bundle score: Score(B) = Rel(B) + Cov(B) + CausalSuff(B) - Cost(B)."""
        rel = self._bundle_relevance(bundle, relevance_scores)
        cov = self._coverage(bundle, task_goals or [])
        suff = self._causal_sufficiency(bundle, causal_edges)
        cost = self._cost(bundle)
        redundancy = self._redundancy_penalty(bundle, causal_edges)
        conflict = self._conflict_penalty(bundle, causal_edges)
        return rel + cov + suff - cost - redundancy - conflict

    def _bundle_relevance(self, bundle: list[SkillNode], scores: dict[str, float]) -> float:
        """Average relevance of skills in bundle."""
        if not bundle:
            return 0.0
        return sum(scores.get(n.name, 0.0) for n in bundle) / len(bundle)

    def _coverage(self, bundle: list[SkillNode], goals: list[str]) -> float:
        """Fraction of task goals covered by bundle skills."""
        if not goals:
            return 1.0
        covered = 0
        for goal in goals:
            goal_lower = goal.lower()
            for node in bundle:
                text = f"{node.description} {node.domain_tags} {node.one_line_capability}".lower()
                if goal_lower in text:
                    covered += 1
                    break
        return covered / len(goals)

    def _causal_sufficiency(self, bundle: list[SkillNode], causal_edges: list[SkillEdge]) -> float:
        """CausalSuff(B) = sum_i [ w_i * marginal_contribution(s_i, B) ]

        Leave-one-out sufficiency estimate. For each skill in bundle:
        - Find confirmed causal edges from OTHER bundle members TO this skill
          (these represent causal dependencies on this skill within the bundle).
        - Also find confirmed causal edges FROM this skill to other bundle members
          (this skill causally supports others in the bundle).
        - marginal_contribution = average causal_score of edges FROM this skill
          TO other bundle members (approximates interventional drop if removed).
        - If a skill has NO outgoing causal edges to other bundle members, it
          contributes independently (not through causal chains) and gets a LOW
          sufficiency contribution (subtracted from 1.0).

        Skills that ARE causally depended upon get HIGH contribution.
        Skills with NO causal dependencies from within the bundle get LOW
        contribution (they might be redundant).
        """
        support_edges = [e for e in causal_edges if is_prerequisite_edge(e)]
        bundle_names = {n.name for n in bundle}
        if not bundle:
            return 0.0

        total_contribution = 0.0
        weight = 1.0 / len(bundle)

        for node in bundle:
            # Edges FROM this skill TO other bundle members (this skill supports others)
            outgoing_to_bundle = [
                e
                for e in support_edges
                if e.source == node.name
                and e.target in bundle_names
                and e.target != node.name
            ]
            # Edges FROM other bundle members TO this skill (others depend on this)
            incoming_from_bundle = [
                e
                for e in support_edges
                if e.target == node.name
                and e.source in bundle_names
                and e.source != node.name
            ]

            if outgoing_to_bundle:
                # This skill is causally depended upon: high contribution
                # marginal_contribution = avg causal_score of outgoing edges
                marginal = sum(_edge_score(e) for e in outgoing_to_bundle) / len(
                    outgoing_to_bundle
                )
            elif incoming_from_bundle:
                # This skill depends on others but nothing depends on it
                # Moderate contribution: it uses the causal chain but removing
                # it would not break other skills
                marginal = (
                    sum(_edge_score(e) for e in incoming_from_bundle)
                    / len(incoming_from_bundle)
                    * 0.5
                )
            else:
                # No causal links within the bundle: low sufficiency contribution.
                # An independent skill adds nothing to causal sufficiency —
                # it may be redundant or not causally necessary for other bundle members.
                marginal = 0.0

            total_contribution += weight * marginal

        return total_contribution

    def _redundancy_penalty(
        self, bundle: list[SkillNode], causal_edges: list[SkillEdge]
    ) -> float:
        """Penalize co-selecting skills that the graph marks as alternatives/similar."""
        bundle_names = {n.name for n in bundle}
        if len(bundle_names) < 2:
            return 0.0
        redundant_pairs = {
            tuple(sorted((e.source, e.target)))
            for e in causal_edges
            if is_redundancy_edge(e)
            and e.source in bundle_names
            and e.target in bundle_names
            and e.source != e.target
        }
        return 0.05 * len(redundant_pairs)

    def _conflict_penalty(
        self, bundle: list[SkillNode], causal_edges: list[SkillEdge]
    ) -> float:
        """Penalize co-selecting skills connected by explicit conflict edges."""
        bundle_names = {n.name for n in bundle}
        if len(bundle_names) < 2:
            return 0.0
        conflicts = [
            e
            for e in causal_edges
            if is_conflict_edge(e)
            and e.source in bundle_names
            and e.target in bundle_names
            and e.source != e.target
        ]
        return 0.2 * len(conflicts)

    def _cost(self, bundle: list[SkillNode]) -> float:
        """Normalized cost: penalize large bundles."""
        return 0.02 * len(bundle)


class CausalPathPlanner:
    """Plans execution order using causal edge weights."""

    def __init__(self, nodes: list[SkillNode], edges: list[SkillEdge]):
        self.nodes = nodes
        self.edges = edges
        self._name_to_node = {n.name: n for n in nodes}

    def plan_execution_path(self, bundle: list[SkillNode]) -> list[str]:
        """Topological sort of bundle nodes based on confirmed causal edges.

        Prefers order where high-causal-score edges go first.
        Uses Kahn's algorithm with sorted queues for determinism.
        """
        bundle_names = {n.name for n in bundle}
        adj: dict[str, list[str]] = {n.name: [] for n in bundle}
        in_degree: dict[str, int] = {n.name: 0 for n in bundle}

        for edge in self.edges:
            if (
                edge.source in bundle_names
                and edge.target in bundle_names
                and is_prerequisite_edge(edge)
            ):
                adj[edge.source].append(edge.target)
                in_degree[edge.target] = in_degree.get(edge.target, 0) + 1

        # Kahn's algorithm with priority (sorted lexicographically for determinism)
        queue = sorted([n for n in bundle_names if in_degree.get(n, 0) == 0])
        result: list[str] = []
        while queue:
            node = queue.pop(0)
            result.append(node)
            for neighbor in sorted(adj.get(node, [])):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
                    queue.sort()

        # Add any remaining nodes (cycles or disconnected)
        remaining = [n.name for n in bundle if n.name not in result]
        result.extend(remaining)
        return result

    def path_score(self, path: list[str]) -> float:
        """Score(pi) = sum(c_ij) - lambda*sum(u_ij) along path edges."""
        score = 0.0
        edge_lookup = {
            (e.source, e.target): e
            for e in self.edges
            if is_prerequisite_edge(e)
        }
        for i in range(len(path) - 1):
            edge = edge_lookup.get((path[i], path[i + 1]))
            if edge:
                score += _edge_score(edge) - 0.1 * _edge_uncertainty(edge)
        return score

    def detect_causal_conflicts(self, bundle: list[SkillNode]) -> list[tuple[str, str, str]]:
        """Find pairs where one skill's effects might destroy another's preconditions.

        Returns list of (source_skill, target_skill, reason) triples.
        """
        conflicts: list[tuple[str, str, str]] = []
        for i, node_i in enumerate(bundle):
            for j, node_j in enumerate(bundle):
                if i == j:
                    continue
                # Check if conflict edges exist between the pair
                for edge in self.edges:
                    if (
                        edge.source == node_i.name
                        and edge.target == node_j.name
                        and is_conflict_edge(edge)
                    ):
                        conflicts.append((node_i.name, node_j.name, "conflict_edge"))
                # Check if effects text mentions something that contradicts preconditions
                if node_i.effects and node_j.preconditions:
                    effects_tokens = set(node_i.effects.lower().split())
                    precond_tokens = set(node_j.preconditions.lower().split())
                    # Heuristic: if destructive verbs in effects and shared object tokens
                    if (
                        effects_tokens & {"remove", "delete", "clear", "reset"}
                        and effects_tokens & precond_tokens
                    ):
                        conflicts.append(
                            (node_i.name, node_j.name, "effect_destroys_precondition")
                        )
        return conflicts


class CausalBackwardTraversal:
    """Backward BFS along confirmed causal edges to find prerequisites."""

    def __init__(self, edges: list[SkillEdge], config: Any = None):
        self.edges = edges
        self._config = config
        # Build reverse adjacency for backward traversal
        self._reverse_adj: dict[str, list[tuple[str, float, float]]] = {}
        for e in edges:
            if is_prerequisite_edge(e):
                causal_score = _edge_score(e)
                if causal_score <= 0.0:
                    continue
                transportability = _edge_transportability(e)
                self._reverse_adj.setdefault(e.target, []).append(
                    (e.source, causal_score, transportability)
                )

    def traverse(
        self,
        seeds: list[str],
        max_depth: int = 3,
        min_transportability: float | None = None,
    ) -> dict[str, float]:
        """BFS backward from seeds, accumulating product of causal_scores along path.

        Only follows edges with transportability >= min_transportability.
        If min_transportability is None, uses config.TRANSPORTABILITY_THRESHOLD
        (default 0.7) to filter out non-transportable edges.
        Returns a mapping of prerequisite skill name to accumulated path score.
        """
        if min_transportability is None:
            min_transportability = getattr(
                self._config, "TRANSPORTABILITY_THRESHOLD", 0.7
            ) if self._config else 0.7

        prerequisites: dict[str, float] = {}
        queue: list[tuple[str, float, int]] = [(s, 1.0, 0) for s in seeds]
        visited: set[str] = set(seeds)

        while queue:
            current, path_score, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            for source, c_score, t_score in self._reverse_adj.get(current, []):
                if t_score < min_transportability:
                    continue
                new_score = path_score * c_score
                if source in visited:
                    # Update if better path found and re-enqueue for propagation
                    if source in prerequisites and prerequisites[source] >= new_score:
                        continue
                    prerequisites[source] = new_score
                    queue.append((source, new_score, depth + 1))
                else:
                    visited.add(source)
                    prerequisites[source] = new_score
                    queue.append((source, new_score, depth + 1))

        return prerequisites
