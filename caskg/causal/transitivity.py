"""Transitivity-based pruning for CaSKG intervention scheduling.

Uses confirmed causal edges to infer transitive relationships (forward
inference) and to exclude unlikely edges (backward exclusion), reducing
the number of expensive intervention experiments needed.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from caskg.core.schema import SkillEdge, SkillNode


class TransitivityPruner:
    """Prunes the intervention queue using causal transitivity reasoning.

    Given confirmed and rejected causal edges, the pruner:
    - Infers transitive edges (A->B, B->C => A->C with discount)
    - Excludes edges unlikely to be direct prereqs (backward exclusion)
    - Reduces the intervention queue to only edges needing actual testing
    """

    def __init__(self, nodes: list[SkillNode], edges: list[SkillEdge]) -> None:
        self.nodes = nodes
        self.edges = edges
        self._name_to_idx: dict[str, int] = {n.name: i for i, n in enumerate(nodes)}
        self._build_adjacency()

    def _build_adjacency(self) -> None:
        """Build adjacency structures from confirmed and rejected causal edges."""
        self.adj: dict[str, set[str]] = {}
        self.confirmed: set[tuple[str, str]] = set()
        self.non_causal: set[tuple[str, str]] = set()

        # Also store causal scores for confirmed edges
        self._edge_scores: dict[tuple[str, str], float] = {}

        for e in self.edges:
            if e.status == "confirmed_causal":
                self.adj.setdefault(e.source, set()).add(e.target)
                self.confirmed.add((e.source, e.target))
                self._edge_scores[(e.source, e.target)] = e.causal_score
            elif e.status == "rejected_non_causal":
                self.non_causal.add((e.source, e.target))

    def forward_inference(self) -> list[tuple[str, str, float]]:
        """Infer transitive causal edges with discounted scores.

        If A->B is confirmed with score c_AB and B->C is confirmed with
        score c_BC, infer A->C with score c_AB * c_BC * 0.8 (discount
        per hop to account for indirect relationships being weaker).

        Returns
        -------
        list[tuple[str, str, float]]
            List of (source, target, inferred_score) for inferred edges.
            Only includes pairs not already confirmed or rejected.
        """
        inferred: list[tuple[str, str, float]] = []
        discount_per_hop = 0.8

        # For each confirmed edge A->B, look at B's outgoing confirmed edges
        for a in list(self.adj.keys()):
            for b in list(self.adj.get(a, set())):
                score_ab = self._edge_scores.get((a, b), 0.5)
                for c in list(self.adj.get(b, set())):
                    # Skip self-loops
                    if c == a:
                        continue
                    pair = (a, c)
                    # Skip if already confirmed or rejected
                    if pair in self.confirmed or pair in self.non_causal:
                        continue
                    score_bc = self._edge_scores.get((b, c), 0.5)
                    inferred_score = score_ab * score_bc * discount_per_hop
                    inferred.append((a, c, inferred_score))

        # Extend to longer paths via BFS from each node
        # For paths longer than 2, apply discount per hop
        for start in list(self.adj.keys()):
            visited: set[str] = {start}
            # (current_node, accumulated_score, hop_count)
            queue: deque[tuple[str, float, int]] = deque()
            for direct_target in self.adj.get(start, set()):
                score = self._edge_scores.get((start, direct_target), 0.5)
                queue.append((direct_target, score, 1))
                visited.add(direct_target)

            while queue:
                current, path_score, hops = queue.popleft()
                if hops >= 4:  # Cap at 4 hops to avoid explosion
                    continue
                for next_node in self.adj.get(current, set()):
                    if next_node in visited:
                        continue
                    visited.add(next_node)
                    edge_score = self._edge_scores.get((current, next_node), 0.5)
                    new_path_score = path_score * edge_score * discount_per_hop
                    pair = (start, next_node)
                    if pair not in self.confirmed and pair not in self.non_causal:
                        # Only add if not already found with higher score
                        inferred.append((start, next_node, new_path_score))
                    queue.append((next_node, new_path_score, hops + 1))

        # Deduplicate: keep the highest score for each pair
        best_scores: dict[tuple[str, str], float] = {}
        for src, tgt, score in inferred:
            key = (src, tgt)
            if key not in best_scores or score > best_scores[key]:
                best_scores[key] = score

        return [(s, t, sc) for (s, t), sc in best_scores.items()]

    def backward_exclusion(self) -> set[tuple[str, str]]:
        """Identify edges to deprioritize based on rejected non-causal edges.

        If A->B is non-causal, then for all C reachable from B via confirmed
        edges (full multi-hop BFS downstream), A->C is unlikely to be a direct
        prerequisite (since A does not causally influence B, and B->...->C may
        be the real path).

        Returns
        -------
        set[tuple[str, str]]
            Set of (source, target) edge pairs to deprioritize in testing.
        """
        excluded: set[tuple[str, str]] = set()

        for a, b in self.non_causal:
            # BFS over full downstream of B via confirmed edges
            visited: set[str] = set()
            queue: deque[str] = deque()
            for initial_target in self.adj.get(b, set()):
                queue.append(initial_target)
                visited.add(initial_target)

            while queue:
                current = queue.popleft()
                pair = (a, current)
                # Only exclude if not already confirmed
                if pair not in self.confirmed:
                    excluded.add(pair)
                # Continue BFS to further downstream nodes
                for downstream in self.adj.get(current, set()):
                    if downstream not in visited:
                        visited.add(downstream)
                        queue.append(downstream)

        return excluded

    def has_directed_path(self, source: str, target: str, max_depth: int = 5) -> bool:
        """Check if a directed path exists from source to target via BFS.

        Only traverses confirmed causal edges. Useful as a topological
        constraint check.

        Parameters
        ----------
        source : str
            Starting node name.
        target : str
            Destination node name.
        max_depth : int
            Maximum BFS depth to prevent explosion on large graphs.

        Returns
        -------
        bool
            True if a directed path exists within max_depth hops.
        """
        if source == target:
            return True
        if source not in self.adj:
            return False

        visited: set[str] = {source}
        queue: deque[tuple[str, int]] = deque([(source, 0)])

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for neighbor in self.adj.get(current, set()):
                if neighbor == target:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        return False

    def reduce_queue(self, queue: list[SkillEdge]) -> list[SkillEdge]:
        """Remove edges from the intervention queue that can be resolved via transitivity.

        Filters out edges that are either:
        - Inferrable via forward transitivity (confirmed path exists)
        - Deprioritized via backward exclusion (non-causal upstream)
        - Topologically impossible as prereq (Rule 3): if no directed path
          exists in either direction between source and target, the edge
          cannot be a prerequisite relationship. However, it is kept if the
          edge type is not 'prereq' (e.g., co-occurrence or association).

        Parameters
        ----------
        queue : list[SkillEdge]
            Candidate edges awaiting intervention testing.

        Returns
        -------
        list[SkillEdge]
            Reduced queue containing only edges that need actual testing.
        """
        inferred = set((s, t) for s, t, _ in self.forward_inference())
        excluded = self.backward_exclusion()

        reduced: list[SkillEdge] = []
        for e in queue:
            pair = (e.source, e.target)
            if pair in inferred:
                continue
            if pair in excluded:
                continue
            # Rule 3: topological constraint check for prereq edges
            # If no directed path exists in either direction, the edge
            # cannot represent a prerequisite relationship
            if e.type == "prereq":
                if (
                    not self.has_directed_path(e.source, e.target)
                    and not self.has_directed_path(e.target, e.source)
                ):
                    continue
            reduced.append(e)

        return reduced
