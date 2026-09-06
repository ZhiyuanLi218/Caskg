"""Top-level CaSKG orchestrator binding all causal components into a cohesive lifecycle."""

from pathlib import Path
from typing import Any
from dataclasses import replace as dc_replace
import math
import re

from caskg.core.schema import (
    SkillNode,
    SkillEdge,
    GraphHealthReport,
)
from caskg.causal.trace_store import TraceStore, ExecutionTrace
from caskg.causal.candidate_induction import CandidateGraphInducer
from caskg.causal.interventions import InterventionEngine
from caskg.causal.validator import CounterfactualValidator
from caskg.causal.scheduler import ActiveScheduler, GroupInterventionPlanner, OpportunisticProber
from caskg.causal.maintenance import EvolutionCycle, CausalHealthDiagnostics
from caskg.causal.retrieval import (
    CausalScorer,
    CausalBundleScorer,
    CausalPathPlanner,
    CausalBackwardTraversal,
    _edge_score,
    _edge_transportability,
    is_conflict_edge,
    is_prerequisite_edge,
    is_redundancy_edge,
)
from caskg.causal.transportability import (
    TransportabilityEstimator,
    CrossEnvironmentTracker,
    ColdStartTransferEngine,
)
from caskg.causal.transitivity import TransitivityPruner


class CausalGraphEngine:
    """Top-level orchestrator for the CaSKG lifecycle.

    Binds all causal components (candidate induction, counterfactual validation,
    active scheduling, maintenance, causal retrieval, transportability) into a
    closed-loop system that operates on top of an existing CaSKG SkillGraphRAG engine.
    """

    def __init__(self, config: Any, workspace_path: str = ""):
        self.config = config
        workspace = workspace_path or getattr(config, "WORKING_DIR", "./caskg_workspace")
        self.workspace_path = workspace
        trace_path = str(Path(workspace) / "causal_traces.jsonl")

        # Core components
        self.trace_store = TraceStore(trace_path)
        self.intervention_engine = InterventionEngine(self.trace_store, config)
        self.validator = CounterfactualValidator(self.intervention_engine, config)
        self.scheduler = ActiveScheduler(config, self.trace_store)
        self.group_planner = GroupInterventionPlanner()
        self.opportunistic = OpportunisticProber(self.scheduler)
        self.evolution = EvolutionCycle(config, self.trace_store)
        self.diagnostics = CausalHealthDiagnostics(config, self.trace_store)
        self.scorer = CausalScorer(config)
        self.bundle_scorer = CausalBundleScorer(config)
        self.transportability = TransportabilityEstimator(config, self.trace_store)
        self.env_tracker = CrossEnvironmentTracker(self.trace_store)

        # State
        self._edges: list[SkillEdge] = []
        self._nodes: list[SkillNode] = []
        self._episode: int = 0

    def load_graph(self, nodes: list[SkillNode], edges: list[SkillEdge]) -> None:
        """Load existing graph state and initialize edge priors."""
        self._nodes = list(nodes)
        self._edges = list(edges)
        self._initialize_edge_priors()

    def _initialize_edge_priors(self) -> None:
        """Initialize Bayesian estimator priors from association_score for unverified edges.

        For each unverified edge with a positive association_score, creates an
        informative prior in the intervention engine's estimator. The association
        score shifts the Beta prior toward causal (higher alpha) so that edges
        with strong observational evidence start closer to confirmation and require
        fewer intervention probes.
        """
        for edge in self._edges:
            if edge.status == "unverified" and edge.association_score > 0:
                # Scale association_score to an informative prior.
                # A score of 1.0 adds 2.0 to alpha (strong prior toward causal).
                # The base prior remains Beta(1,1) -- uniform -- for edges with no signal.
                prior_boost = edge.association_score * 2.0
                estimator = self.intervention_engine.get_estimator(edge.source, edge.target)
                # Only set the prior if the estimator is still at its default (uninformative)
                if estimator.alpha == 1.0 and estimator.beta == 1.0:
                    estimator.alpha += prior_boost

    @property
    def confirmed_edges(self) -> list[SkillEdge]:
        return [e for e in self._edges if e.status in {"confirmed_causal", "stable"}]

    @property
    def unverified_edges(self) -> list[SkillEdge]:
        return [e for e in self._edges if e.status == "unverified"]

    @property
    def stats(self) -> dict[str, Any]:
        status_counts: dict[str, int] = {}
        for e in self._edges:
            status_counts[e.status] = status_counts.get(e.status, 0) + 1
        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "edge_status": status_counts,
            "episode": self._episode,
            "exploration_rate": self.scheduler.exploration_rate,
            "budget_remaining": self.scheduler.budget_remaining,
            "environments_seen": self.env_tracker.environments_seen(),
        }

    async def build_candidate_graph(
        self, embedding_service=None, llm_service=None
    ) -> int:
        """Phase B: Generate candidate edges with association scores."""
        inducer = CandidateGraphInducer(self.trace_store, self.config)
        # Enable resume + progress logging via checkpoint file
        import os
        inducer._checkpoint_path = os.path.join(
            self.workspace_path, "candidate_checkpoint.jsonl"
        )
        new_edges = await inducer.induce_candidates(
            self._nodes,
            self._edges,
            embedding_service=embedding_service,
            llm_service=llm_service,
        )
        # Merge with existing edges. For pairs already present (e.g. base
        # dependency edges built by the core engine with association_score=0.0),
        # UPDATE their scoring fields from the freshly-computed candidate instead
        # of skipping — otherwise the re-scored value is silently dropped and the
        # base edge keeps its default 0.0 association_score, which buries genuine
        # I/O-dependency edges below spurious lexical matches at validation time.
        existing_by_pair: dict[tuple[str, str], SkillEdge] = {
            (e.source, e.target): e for e in self._edges
        }
        merged = 0
        for edge in new_edges:
            key = (edge.source, edge.target)
            current = existing_by_pair.get(key)
            if current is None:
                self._edges.append(edge)
                existing_by_pair[key] = edge
            elif current.status in (None, "unverified"):
                # Only refresh edges not yet validated. Preserve the candidate
                # type when it is more specific than the base "dependency" tag.
                current.association_score = edge.association_score
                current.weight = edge.weight
                current.confidence = edge.confidence
                if edge.type and edge.type not in ("dependency", "none", None):
                    current.type = edge.type
                if edge.description:
                    current.description = edge.description
                merged += 1
        return len(new_edges)

    def design_probes_for_episode(self, task_skills: list[str]) -> list[Any]:
        """Phase C+D: Select edges to probe and design interventions."""
        if not self.scheduler.should_explore():
            return []

        # Apply transitivity pruning to reduce the unverified queue before selection.
        # This eliminates edges that can be inferred or excluded via transitive reasoning,
        # saving expensive intervention experiments.
        pruner = TransitivityPruner(self._nodes, self._edges)
        pruned_queue = pruner.reduce_queue(self.unverified_edges)

        # Rebuild scheduler queue with pruned candidates
        self.scheduler.rebuild_queue(pruned_queue)

        # Select edges to test
        edges_to_test = self.scheduler.select_top_k(3)

        # Also check opportunistic probe
        opp = self.opportunistic.select_probe_for_task(
            task_skills, pruned_queue
        )
        if opp and opp not in edges_to_test:
            edges_to_test.append(opp)

        # Design probes
        probes = []
        for edge in edges_to_test:
            alternatives = self._find_alternatives(edge)
            updated_edge, edge_probes = self.validator.validate_edge(
                edge, task_skills, alternatives
            )
            # Apply any pre-validation edge modifications back to self._edges
            for i, existing in enumerate(self._edges):
                if existing.source == updated_edge.source and existing.target == updated_edge.target:
                    self._edges[i] = updated_edge
                    break
            probes.extend(edge_probes)

        return probes

    def record_probe_results(self, probe_results: list[dict]) -> None:
        """Record intervention outcomes and update edge states."""
        # Track which edges were updated so we can finalize their scores
        updated_edge_indices: set[int] = set()

        for result in probe_results:
            source = result["source"]
            target = result["target"]
            control_outcome = result["control_outcome"]
            treated_outcome = result["treated_outcome"]
            intervention_type = result["intervention_type"]

            # Find and update the edge
            for i, edge in enumerate(self._edges):
                if edge.source == source and edge.target == target:
                    from caskg.causal.interventions import InterventionProbe

                    probe = InterventionProbe(
                        edge_source=source,
                        edge_target=target,
                        intervention_type=intervention_type,
                        task_id=str(self._episode),
                        control_skills=[],
                        treated_skills=[],
                    )
                    delta = self.intervention_engine.record_outcome(
                        probe, control_outcome, treated_outcome
                    )
                    # Update edge
                    updated = self.validator.update_edge_from_probe(
                        edge, delta, intervention_type,
                        episode=self._episode,
                    )
                    self._edges[i] = updated
                    updated_edge_indices.add(i)
                    self.scheduler.record_probe_used()
                    break

        # Finalize composite causal_score for all edges that received probes
        for i in updated_edge_indices:
            self._edges[i] = self.validator.finalize_edge_score(self._edges[i])

    def record_execution(self, trace: ExecutionTrace) -> None:
        """Record a task execution trace."""
        self.trace_store.record(trace)
        self._episode += 1

        # Periodic evolution
        interval = getattr(self.config, "EVOLUTION_CYCLE_INTERVAL", 50)
        if self._episode % interval == 0:
            self._run_evolution()

    def _run_evolution(self) -> GraphHealthReport:
        """Phase E: Run evolution cycle."""
        updated_edges, report, unverified_ratio = self.evolution.execute(
            self._edges, self._nodes, self._episode
        )
        self._edges = updated_edges

        # Update scheduler with the ratio computed by the evolution cycle
        self.scheduler.update_exploration_rate(unverified_ratio)
        self.scheduler.rebuild_queue(self._edges)

        return report

    def retrieve_causal(
        self,
        query: str,
        relevance_scores: dict[str, float],
        top_n: int = 8,
    ) -> dict[str, Any]:
        """Phase F: Causal-aware skill retrieval."""
        # Backward traversal to find prerequisites
        seed_names = sorted(
            relevance_scores, key=lambda k: relevance_scores.get(k, 0.0), reverse=True
        )[:5]
        traversal = CausalBackwardTraversal(self._edges, config=self.config)
        prerequisites = traversal.traverse(
            seed_names,
            max_depth=3,
            min_transportability=getattr(
                self.config, "TRANSPORTABILITY_THRESHOLD", 0.7
            ),
        )

        # Merge candidates: seeds + prerequisites
        all_candidates = dict(relevance_scores)
        for name, score in prerequisites.items():
            all_candidates[name] = max(all_candidates.get(name, 0), score * 0.5)

        query_namespaces = self._infer_query_namespaces(query, relevance_scores)
        self._augment_namespace_signature_candidates(
            query,
            all_candidates,
            query_namespaces,
        )
        self._augment_causal_neighborhood_candidates(
            seed_names,
            all_candidates,
            query_namespaces,
        )
        self._augment_terminal_sink_candidates(
            seed_names,
            all_candidates,
            query_namespaces,
        )
        self._augment_role_phase_candidates(
            query,
            all_candidates,
            query_namespaces,
        )

        # Score each candidate
        name_to_node = {n.name: n for n in self._nodes}
        scored = []
        for name, rel_score in all_candidates.items():
            node = name_to_node.get(name)
            if not node:
                continue
            node_edges = [
                e for e in self._edges if e.target == name or e.source == name
            ]
            structural = self._structural_candidate_score(name, seed_names)
            score = self.scorer.score(node, rel_score, structural, node_edges)
            score *= self._namespace_affinity_multiplier(node.name, query_namespaces)
            scored.append((node, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        bundle_nodes = self._select_compact_bundle(
            scored,
            top_n,
            preferred_namespaces=query_namespaces,
            query=query,
        )

        # Bundle scoring
        bundle_score = self.bundle_scorer.score_bundle(
            bundle_nodes, relevance_scores, self._edges
        )

        # Path planning
        planner = CausalPathPlanner(self._nodes, self._edges)
        execution_path = planner.plan_execution_path(bundle_nodes)
        conflicts = planner.detect_causal_conflicts(bundle_nodes)
        bundle_names = {n.name for n in bundle_nodes}

        return {
            "skills": [n.name for n in bundle_nodes],
            "scores": {n.name: s for n, s in scored if n.name in bundle_names},
            "execution_path": execution_path,
            "bundle_sufficiency": bundle_score,
            "conflicts": conflicts,
            "prerequisites_found": list(prerequisites.keys()),
        }

    def _select_compact_bundle(
        self,
        scored: list[tuple[SkillNode, float]],
        top_n: int,
        preferred_namespaces: set[str] | None = None,
        query: str = "",
    ) -> list[SkillNode]:
        """Select a compact, causally coherent bundle.

        Prerequisite/support edges are handled by traversal and scoring. During
        final selection, similar/alternative edges mean "pick one", while
        conflict edges mean "do not co-select". The greedy objective is purely
        graph-theoretic: relevance plus marginal support-connectivity to the
        partial bundle, without mapping query words to specific skills.
        """
        if top_n <= 0:
            return []

        preferred_namespaces = preferred_namespaces or set()
        selected: list[SkillNode] = []
        skipped_redundant: list[SkillNode] = []
        skipped_namespace: list[SkillNode] = []
        eligible: list[SkillNode] = []
        score_by_name = {node.name: score for node, score in scored}
        required_roles = {
            role
            for role, _priority in self._required_role_phases(
                query,
                preferred_namespaces,
                score_by_name,
            )
        }

        for node, _score in scored:
            if (
                preferred_namespaces
                and not self._namespace_matches(node.name, preferred_namespaces)
                and not self._has_transportable_namespace_bridge(
                    node.name, preferred_namespaces
                )
            ):
                skipped_namespace.append(node)
                continue
            eligible.append(node)

        while len(selected) < top_n and eligible:
            selected_names = {item.name for item in selected}
            best_node: SkillNode | None = None
            best_score = float("-inf")
            for node in eligible:
                if node.name in selected_names:
                    continue
                if self._conflicts_with_any(node.name, selected_names):
                    continue
                if self._redundant_with_any(node.name, selected_names):
                    skipped_redundant.append(node)
                    continue
                if (
                    required_roles
                    and
                    selected_names
                    and not self._node_adds_required_role(node, selected, required_roles)
                    and self._bundle_connectivity_gain(node.name, selected_names) <= 0.0
                ):
                    skipped_redundant.append(node)
                    continue
                marginal = score_by_name.get(node.name, 0.0)
                marginal += self._bundle_connectivity_gain(node.name, selected_names)
                if marginal > best_score:
                    best_score = marginal
                    best_node = node
            if best_node is None:
                break
            selected.append(best_node)
            eligible = [node for node in eligible if node.name != best_node.name]

        if len(selected) < top_n:
            selected_names = {item.name for item in selected}
            for node in eligible:
                if len(selected) >= top_n:
                    break
                if node.name in selected_names:
                    continue
                if self._conflicts_with_any(node.name, selected_names):
                    continue
                if (
                    required_roles
                    and
                    selected_names
                    and not self._node_adds_required_role(node, selected, required_roles)
                    and self._bundle_connectivity_gain(node.name, selected_names) <= 0.0
                ):
                    continue
                selected.append(node)
                selected_names.add(node.name)

        if len(selected) < top_n:
            selected_names = {item.name for item in selected}
            for node in skipped_redundant:
                if len(selected) >= top_n:
                    break
                if (
                    node.name not in selected_names
                    and not self._conflicts_with_any(node.name, selected_names)
                    and (
                        not required_roles
                        or
                        self._node_adds_required_role(node, selected, required_roles)
                        or self._bundle_connectivity_gain(node.name, selected_names) > 0.0
                    )
                ):
                    selected.append(node)
                    selected_names.add(node.name)

        if len(selected) < top_n and not preferred_namespaces:
            selected_names = {item.name for item in selected}
            for node in skipped_namespace:
                if len(selected) >= top_n:
                    break
                if (
                    node.name not in selected_names
                    and not self._conflicts_with_any(node.name, selected_names)
                    and (
                        not required_roles
                        or
                        self._node_adds_required_role(node, selected, required_roles)
                        or self._bundle_connectivity_gain(node.name, selected_names) > 0.0
                    )
                ):
                    selected.append(node)
                    selected_names.add(node.name)

        if len(selected) < top_n and not preferred_namespaces:
            selected_names = {item.name for item in selected}
            for node, _score in scored:
                if len(selected) >= top_n:
                    break
                if (
                    node.name not in selected_names
                    and not self._conflicts_with_any(node.name, selected_names)
                    and (
                        not required_roles
                        or
                        self._node_adds_required_role(node, selected, required_roles)
                        or self._bundle_connectivity_gain(node.name, selected_names) > 0.0
                    )
                ):
                    selected.append(node)
                    selected_names.add(node.name)

        selected = self._ensure_terminal_sink_coverage(
            selected[:top_n],
            scored,
            top_n,
            preferred_namespaces,
            score_by_name,
        )
        selected = self._ensure_support_chain_coverage(
            selected[:top_n],
            scored,
            top_n,
            preferred_namespaces,
            score_by_name,
        )
        selected = self._ensure_role_phase_coverage(
            selected[:top_n],
            scored,
            top_n,
            preferred_namespaces,
            score_by_name,
            query,
        )
        return selected[:top_n]

    def _required_role_phases(
        self,
        query: str,
        preferred_namespaces: set[str],
        current_candidates: dict[str, float] | None = None,
    ) -> list[tuple[str, float]]:
        """Infer abstract workflow phases needed by the current retrieval.

        This is intentionally domain-agnostic. It only reasons over broad skill
        roles such as locate, acquire, transform, deliver, and verify. Concrete
        skill names, benchmark names, object ids, and environment-specific
        commands are never encoded here.
        """
        lower = str(query or "").lower()
        roles: list[tuple[str, float]] = []

        def add(role: str, priority: float) -> None:
            for idx, (existing, existing_priority) in enumerate(roles):
                if existing == role:
                    roles[idx] = (existing, max(existing_priority, priority))
                    return
            roles.append((role, priority))

        candidate_roles: set[str] = set()
        name_to_node = {node.name: node for node in self._nodes}
        for name in current_candidates or {}:
            node = name_to_node.get(name)
            if node is not None:
                candidate_roles.update(self._node_phase_roles(node))

        transform_terms = {
            "transform",
            "modify",
            "change",
            "convert",
            "normalize",
            "prepare",
            "generate",
            "update",
            "edit",
            "process",
            "clean",
            "cool",
            "heat",
            "hot",
            "cold",
        }
        deliver_terms = {
            "deliver",
            "place",
            "move",
            "store",
            "save",
            "write",
            "export",
            "return",
            "publish",
            "output",
            "deposit",
        }
        inspect_terms = {
            "inspect",
            "examine",
            "verify",
            "validate",
            "check",
            "audit",
            "test",
            "look",
            "compare",
            "evaluate",
        }
        tool_terms = {
            "tool",
            "device",
            "instrument",
            "interface",
            "api",
            "service",
            "client",
            "executor",
            "using",
            "with",
        }
        context_terms = {
            "context",
            "scan",
            "observe",
            "observation",
            "explore",
            "survey",
            "inspect",
        }

        query_tokens = set(re.findall(r"[a-z0-9_]+", lower))
        needs_context = bool(query_tokens & context_terms) or "context_scan" in candidate_roles
        needs_transform = bool(query_tokens & transform_terms) or "state_transform" in candidate_roles
        needs_delivery = bool(query_tokens & deliver_terms) or "place_output" in candidate_roles
        needs_inspection = bool(query_tokens & inspect_terms) or "verify_inspect" in candidate_roles
        needs_tool = bool(query_tokens & tool_terms) or "tool_device" in candidate_roles

        if needs_context:
            add("context_scan", 0.46)
        if needs_transform or needs_delivery or needs_inspection or needs_tool:
            add("target_locate", 0.54)
        if needs_transform or needs_delivery:
            add("acquire_inventory", 0.50)
        if needs_tool:
            add("tool_device", 0.48)
        if needs_transform:
            add("state_transform", 0.59)
        if needs_delivery:
            add("place_output", 0.56)
        if needs_inspection:
            add("verify_inspect", 0.52)

        phase_order = {
            "task_parse": 0,
            "context_scan": 1,
            "target_locate": 2,
            "acquire_inventory": 3,
            "tool_device": 4,
            "state_transform": 5,
            "place_output": 6,
            "verify_inspect": 7,
        }
        return sorted(roles, key=lambda item: phase_order.get(item[0], 99))

    def _augment_role_phase_candidates(
        self,
        query: str,
        candidates: dict[str, float],
        preferred_namespaces: set[str],
    ) -> None:
        phases = self._required_role_phases(
            query,
            preferred_namespaces,
            candidates,
        )
        if not phases:
            return

        active_namespaces = preferred_namespaces or {
            self._skill_namespace(name)
            for name in candidates
            if self._skill_namespace(name)
        }
        name_to_node = {node.name: node for node in self._nodes}
        covered_roles = {
            role
            for name in candidates
            if (node := name_to_node.get(name)) is not None
            for role in self._node_phase_roles(node)
        }
        for role, base_score in phases:
            if role in covered_roles:
                continue
            matches = [
                node
                for node in self._nodes
                if role in self._node_phase_roles(node)
                and self._candidate_allowed_for_namespace(node.name, active_namespaces)
            ]
            matches.sort(
                key=lambda node: (
                    self._namespace_matches(node.name, active_namespaces),
                    candidates.get(node.name, 0.0),
                    self._role_specificity(node, role),
                    node.name,
                ),
                reverse=True,
            )
            for offset, node in enumerate(matches[:3]):
                score = max(0.08, base_score - offset * 0.025)
                candidates[node.name] = max(candidates.get(node.name, 0.0), score)

    def _ensure_role_phase_coverage(
        self,
        selected: list[SkillNode],
        scored: list[tuple[SkillNode, float]],
        top_n: int,
        preferred_namespaces: set[str],
        score_by_name: dict[str, float],
        query: str,
    ) -> list[SkillNode]:
        candidate_scores = {node.name: score for node, score in scored}
        phases = self._required_role_phases(
            query,
            preferred_namespaces,
            candidate_scores,
        )
        if not phases or top_n <= 0:
            return selected

        candidate_by_name = {node.name: node for node, _score in scored}
        updated = list(selected)

        for role, _base_score in phases:
            selected_names = {node.name for node in updated}
            if any(role in self._node_phase_roles(node) for node in updated):
                continue

            choices = [
                node
                for node in candidate_by_name.values()
                if role in self._node_phase_roles(node)
                and not self._conflicts_with_any(node.name, selected_names)
                and self._candidate_allowed_for_namespace(node.name, preferred_namespaces)
            ]
            if not choices:
                continue

            choice = max(
                choices,
                key=lambda node: (
                    score_by_name.get(node.name, 0.0),
                    self._role_specificity(node, role),
                    node.name,
                ),
            )
            if len(updated) < top_n:
                updated.append(choice)
                continue

            replace_index = self._role_phase_replacement_index(
                updated,
                [role_name for role_name, _priority in phases],
                score_by_name,
            )
            if replace_index is None:
                continue
            updated[replace_index] = choice

        return updated

    def _role_phase_replacement_index(
        self,
        selected: list[SkillNode],
        roles: list[str],
        score_by_name: dict[str, float],
    ) -> int | None:
        if not selected:
            return None

        role_present: list[set[str]] = []
        for role in roles:
            role_present.append(
                {
                    node.name
                    for node in selected
                    if role in self._node_phase_roles(node)
                }
            )

        candidates: list[tuple[int, int, float, int]] = []
        for index, node in enumerate(selected):
            sole_role_coverage = sum(
                1 for present in role_present if present == {node.name}
            )
            any_role_coverage = sum(
                1 for present in role_present if node.name in present
            )
            candidates.append(
                (
                    sole_role_coverage,
                    any_role_coverage,
                    score_by_name.get(node.name, 0.0),
                    index,
                )
            )
        return min(candidates, key=lambda item: (item[0], item[1], item[2], item[3]))[3]

    def _node_phase_roles(self, node: SkillNode) -> set[str]:
        metadata_roles = self._metadata_phase_roles(node)
        if metadata_roles:
            return metadata_roles

        text = self._node_role_text(node)
        roles: set[str] = set()
        if self._contains_any(
            text,
            {
                "parse",
                "planner",
                "plan",
                "decompose",
                "intent",
                "goal",
                "objective",
                "workflow",
            },
        ):
            roles.add("task_parse")
        if self._contains_any(
            text,
            {
                "context",
                "scan",
                "observe",
                "observation",
                "explore",
                "navigation",
                "environment",
                "evidence",
                "snapshot",
            },
        ):
            roles.add("context_scan")
        if self._contains_any(
            text,
            {
                "locate",
                "find",
                "finder",
                "search",
                "entity",
                "identifier",
                "select",
                "match",
                "candidate",
            },
        ):
            roles.add("target_locate")
        if self._contains_any(
            text,
            {
                "acquire",
                "collect",
                "gather",
                "fetch",
                "take",
                "load",
                "read",
                "ingest",
                "input",
                "source",
                "inventory",
                "held",
            },
        ):
            roles.add("acquire_inventory")
        if self._contains_any(
            text,
            {
                "tool",
                "device",
                "operator",
                "appliance",
                "instrument",
                "equipment",
                "interface",
                "api",
                "client",
                "service",
                "executor",
                "automation",
            },
        ):
            roles.add("tool_device")
        if self._contains_any(
            text,
            {
                "state",
                "transform",
                "modify",
                "change",
                "convert",
                "normalize",
                "prepare",
                "generate",
                "update",
                "edit",
                "process",
                "clean",
                "cool",
                "heat",
            },
        ):
            roles.add("state_transform")
        if self._contains_any(
            text,
            {
                "deliver",
                "delivery",
                "place",
                "deposit",
                "output",
                "result",
                "artifact",
                "export",
                "write",
                "save",
                "store",
                "publish",
                "return",
                "final",
                "sink",
                "destination",
            },
        ):
            roles.add("place_output")
        if self._contains_any(
            text,
            {
                "inspect",
                "examine",
                "verify",
                "check",
                "validate",
                "test",
                "assert",
                "evaluate",
                "quality",
                "audit",
                "compare",
            },
        ):
            roles.add("verify_inspect")
        return roles

    def _metadata_phase_roles(self, node: SkillNode) -> set[str]:
        raw_roles: list[str] = []
        for container in (getattr(node, "metadata", {}), getattr(node, "context_profile", {})):
            if not isinstance(container, dict):
                continue
            for key in ("roles", "role", "workflow_roles", "capability_roles"):
                raw_roles.extend(self._coerce_text_list(container.get(key)))

        allowed = {
            "task_parse",
            "context_scan",
            "target_locate",
            "acquire_inventory",
            "tool_device",
            "state_transform",
            "place_output",
            "verify_inspect",
        }
        normalized = {
            str(role).strip().lower().replace("-", "_").replace(" ", "_")
            for role in raw_roles
        }
        return {role for role in normalized if role in allowed}

    @classmethod
    def _node_role_text(cls, node: SkillNode) -> str:
        return " ".join(
            (
                node.name or "",
                node.description or "",
                node.one_line_capability or "",
                node.inputs or "",
                node.outputs or "",
                node.domain_tags or "",
                node.tooling or "",
                node.preconditions or "",
                node.effects or "",
                cls._flatten_metadata_text(getattr(node, "metadata", {})),
                cls._flatten_metadata_text(getattr(node, "context_profile", {})),
            )
        ).lower()

    @classmethod
    def _flatten_metadata_text(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, dict):
            return " ".join(
                str(part)
                for key, item in value.items()
                for part in (key, cls._flatten_metadata_text(item))
                if str(part).strip()
            )
        if isinstance(value, (list, tuple, set)):
            return " ".join(cls._flatten_metadata_text(item) for item in value)
        return str(value)

    @staticmethod
    def _coerce_text_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value]
        return [str(value)]

    @staticmethod
    def _contains_any(text: str, needles: set[str]) -> bool:
        haystack = str(text or "").lower()
        return any(re.search(rf"\b{re.escape(needle)}\b", haystack) for needle in needles)

    def _role_specificity(self, node: SkillNode, role: str) -> float:
        roles = self._node_phase_roles(node)
        if role not in roles:
            return 0.0
        return 1.0 / max(len(roles), 1)

    def _node_adds_required_role(
        self,
        node: SkillNode,
        selected: list[SkillNode],
        required_roles: set[str],
    ) -> bool:
        if not required_roles:
            return False
        selected_roles = {
            role
            for selected_node in selected
            for role in self._node_phase_roles(selected_node)
        }
        return bool((self._node_phase_roles(node) & required_roles) - selected_roles)

    def _ensure_support_chain_coverage(
        self,
        selected: list[SkillNode],
        scored: list[tuple[SkillNode, float]],
        top_n: int,
        preferred_namespaces: set[str],
        score_by_name: dict[str, float],
    ) -> list[SkillNode]:
        """Prefer complete support chains over disconnected high-score nodes.

        Vector relevance often lands on the middle or end of a procedure. The
        graph can then tell us which predecessor skills make that node usable.
        This final pass is deliberately graph-only: it follows live support
        edges among scored candidates and tries to cover missing immediate
        predecessors before spending shortlist slots on isolated nodes.
        """
        if top_n <= 1 or not selected:
            return selected

        candidate_by_name = {node.name: node for node, _score in scored}
        if not candidate_by_name:
            return selected

        threshold = getattr(self.config, "TRANSPORTABILITY_THRESHOLD", 0.7)
        min_edge_score = getattr(
            self.config,
            "CAUSAL_RETRIEVAL_CHAIN_COVERAGE_MIN_EDGE_SCORE",
            0.05,
        )
        support_edges = [
            edge
            for edge in self._edges
            if is_prerequisite_edge(edge)
            and _edge_transportability(edge) >= threshold
            and _edge_score(edge) >= min_edge_score
            and edge.source in candidate_by_name
            and edge.target in candidate_by_name
        ]
        if not support_edges:
            return selected

        outgoing_count: dict[str, int] = {}
        for edge in support_edges:
            outgoing_count[edge.source] = outgoing_count.get(edge.source, 0) + 1

        updated = list(selected)
        for _iteration in range(max(1, top_n)):
            selected_names = {node.name for node in updated}
            best: tuple[float, SkillEdge] | None = None
            for edge in support_edges:
                if edge.target not in selected_names or edge.source in selected_names:
                    continue
                if not self._candidate_allowed_for_namespace(
                    edge.source,
                    preferred_namespaces,
                ):
                    continue
                if self._conflicts_with_any(edge.source, selected_names):
                    continue
                if self._redundant_with_any(edge.source, selected_names):
                    continue
                priority = (
                    _edge_score(edge) * (0.75 + max(score_by_name.get(edge.target, 0.0), 0.0))
                    + max(score_by_name.get(edge.source, 0.0), 0.0)
                    + min(outgoing_count.get(edge.source, 0) * 0.03, 0.12)
                )
                if best is None or priority > best[0]:
                    best = (priority, edge)

            if best is None:
                break

            priority, edge = best
            source_node = candidate_by_name[edge.source]
            if len(updated) < top_n:
                updated.append(source_node)
                continue

            replace_index = self._support_chain_replacement_index(
                updated,
                protected_names={edge.target},
                score_by_name=score_by_name,
                support_edges=support_edges,
            )
            if replace_index is None:
                break

            replacement_cost = self._bundle_node_removal_cost(
                updated[replace_index].name,
                {node.name for idx, node in enumerate(updated) if idx != replace_index},
                score_by_name,
                support_edges,
            )
            if priority <= replacement_cost + 0.03:
                break
            updated[replace_index] = source_node

        return updated

    def _candidate_allowed_for_namespace(
        self,
        skill_name: str,
        preferred_namespaces: set[str],
    ) -> bool:
        if not preferred_namespaces:
            return True
        return self._namespace_matches(
            skill_name,
            preferred_namespaces,
        ) or self._has_transportable_namespace_bridge(
            skill_name,
            preferred_namespaces,
        )

    def _support_chain_replacement_index(
        self,
        selected: list[SkillNode],
        protected_names: set[str],
        score_by_name: dict[str, float],
        support_edges: list[SkillEdge],
    ) -> int | None:
        selected_names = {node.name for node in selected}
        candidates: list[tuple[float, int]] = []
        for index, node in enumerate(selected):
            if node.name in protected_names:
                continue
            remaining = selected_names - {node.name}
            cost = self._bundle_node_removal_cost(
                node.name,
                remaining,
                score_by_name,
                support_edges,
            )
            candidates.append((cost, index))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    @staticmethod
    def _bundle_node_removal_cost(
        skill_name: str,
        remaining_names: set[str],
        score_by_name: dict[str, float],
        support_edges: list[SkillEdge],
    ) -> float:
        cost = max(score_by_name.get(skill_name, 0.0), 0.0)
        for edge in support_edges:
            if edge.source == skill_name and edge.target in remaining_names:
                cost += _edge_score(edge) * 0.35
            elif edge.target == skill_name and edge.source in remaining_names:
                cost += _edge_score(edge) * 0.25
        return cost

    def _ensure_terminal_sink_coverage(
        self,
        selected: list[SkillNode],
        scored: list[tuple[SkillNode, float]],
        top_n: int,
        preferred_namespaces: set[str],
        score_by_name: dict[str, float],
    ) -> list[SkillNode]:
        if top_n <= 0 or not preferred_namespaces:
            return selected

        terminal_priorities = self._terminal_sink_priorities(preferred_namespaces)
        if not terminal_priorities:
            return selected
        terminal_names = set(terminal_priorities)

        selected_names = {node.name for node in selected}
        terminal_candidates = [
            (node, score)
            for node, score in scored
            if node.name in terminal_names
            and not self._conflicts_with_any(node.name, selected_names)
        ]
        if not terminal_candidates:
            return selected

        terminal, _terminal_score = max(
            terminal_candidates,
            key=lambda item: (terminal_priorities[item[0].name], item[1]),
        )
        selected_terminal_indices = [
            idx for idx, node in enumerate(selected) if node.name in terminal_names
        ]
        if selected_terminal_indices:
            best_selected_idx = max(
                selected_terminal_indices,
                key=lambda idx: (
                    terminal_priorities[selected[idx].name],
                    score_by_name.get(selected[idx].name, 0.0),
                ),
            )
            if (
                terminal_priorities[selected[best_selected_idx].name],
                score_by_name.get(selected[best_selected_idx].name, 0.0),
            ) >= (terminal_priorities[terminal.name], _terminal_score):
                return selected

        if len(selected) < top_n:
            return selected + [terminal]

        if selected_terminal_indices:
            replace_index = min(
                selected_terminal_indices,
                key=lambda idx: (
                    terminal_priorities[selected[idx].name],
                    score_by_name.get(selected[idx].name, 0.0),
                ),
            )
        else:
            replace_index = min(
                range(len(selected)),
                key=lambda idx: score_by_name.get(selected[idx].name, 0.0),
            )
        updated = list(selected)
        updated[replace_index] = terminal
        return updated

    def _terminal_sink_names(self, preferred_namespaces: set[str]) -> set[str]:
        return set(self._terminal_sink_priorities(preferred_namespaces))

    def _terminal_sink_priorities(
        self,
        preferred_namespaces: set[str],
    ) -> dict[str, tuple[int, float, int, float]]:
        threshold = getattr(self.config, "TRANSPORTABILITY_THRESHOLD", 0.7)
        incoming: dict[str, list[float]] = {}
        outgoing: dict[str, int] = {}
        for edge in self._edges:
            if not is_prerequisite_edge(edge):
                continue
            if _edge_transportability(edge) < threshold:
                continue
            edge_score = _edge_score(edge)
            if edge_score <= 0.0:
                continue
            incoming.setdefault(edge.target, []).append(edge_score)
            outgoing[edge.source] = outgoing.get(edge.source, 0) + 1

        priorities: dict[str, tuple[int, float, int, float]] = {}
        for name, scores in incoming.items():
            if not self._namespace_matches(name, preferred_namespaces):
                continue
            in_degree = len(scores)
            out_degree = outgoing.get(name, 0)
            sinkness = in_degree / (in_degree + out_degree + 1)
            if out_degree == 0 or sinkness >= 0.75:
                avg_score = sum(scores) / in_degree
                priorities[name] = (
                    1 if out_degree == 0 else 0,
                    sinkness,
                    in_degree,
                    avg_score,
                )
        return priorities

    def _augment_causal_neighborhood_candidates(
        self,
        seed_names: list[str],
        candidates: dict[str, float],
        preferred_namespaces: set[str],
    ) -> None:
        if not seed_names:
            return

        threshold = getattr(self.config, "TRANSPORTABILITY_THRESHOLD", 0.7)
        adjacency: dict[str, list[tuple[str, float]]] = {}
        for edge in self._edges:
            if not is_prerequisite_edge(edge):
                continue
            if _edge_transportability(edge) < threshold:
                continue
            edge_score = _edge_score(edge)
            if edge_score <= 0.0:
                continue
            adjacency.setdefault(edge.source, []).append((edge.target, edge_score))
            adjacency.setdefault(edge.target, []).append((edge.source, edge_score * 0.8))

        queue: list[tuple[str, float, int]] = [
            (name, max(float(candidates.get(name, 0.0)), 0.1), 0)
            for name in seed_names
        ]
        best_seen = {name: score for name, score, _depth in queue}
        while queue:
            current, path_score, depth = queue.pop(0)
            if depth >= 2:
                continue
            for neighbor, edge_score in adjacency.get(current, []):
                if (
                    preferred_namespaces
                    and not self._namespace_matches(neighbor, preferred_namespaces)
                    and not self._has_transportable_namespace_bridge(
                        neighbor, preferred_namespaces
                    )
                ):
                    continue
                propagated = path_score * edge_score * 0.65
                if propagated <= best_seen.get(neighbor, 0.0):
                    continue
                best_seen[neighbor] = propagated
                candidates[neighbor] = max(candidates.get(neighbor, 0.0), propagated)
                queue.append((neighbor, propagated, depth + 1))

    def _augment_terminal_sink_candidates(
        self,
        seed_names: list[str],
        candidates: dict[str, float],
        preferred_namespaces: set[str],
    ) -> None:
        """Promote same-community terminal sinks from the validated graph.

        This is a graph-closure signal, not a query-word mapping: when the
        current seed set lands in a skill namespace, terminal nodes in that
        namespace with live prerequisite support represent likely workflow
        completion/checkpoint skills.
        """
        if not seed_names:
            return

        seed_namespaces = {
            self._skill_namespace(name)
            for name in seed_names
            if self._skill_namespace(name)
        }
        active_namespaces = preferred_namespaces or seed_namespaces
        if not active_namespaces:
            return

        threshold = getattr(self.config, "TRANSPORTABILITY_THRESHOLD", 0.7)
        incoming: dict[str, list[float]] = {}
        outgoing: dict[str, int] = {}
        for edge in self._edges:
            if not is_prerequisite_edge(edge):
                continue
            if _edge_transportability(edge) < threshold:
                continue
            edge_score = _edge_score(edge)
            if edge_score <= 0.0:
                continue
            incoming.setdefault(edge.target, []).append(edge_score)
            outgoing[edge.source] = outgoing.get(edge.source, 0) + 1

        for target, scores in incoming.items():
            if not self._namespace_matches(target, active_namespaces):
                continue
            target_out = outgoing.get(target, 0)
            sinkness = len(scores) / (len(scores) + target_out + 1)
            if target_out > 0 and sinkness < 0.75:
                continue
            mean_score = sum(scores) / len(scores)
            terminal_score = min(0.4, max(0.18, mean_score * sinkness * 1.5))
            candidates[target] = max(candidates.get(target, 0.0), terminal_score)

    def _structural_candidate_score(self, skill_name: str, seed_names: list[str]) -> float:
        if not seed_names:
            return 0.0

        threshold = getattr(self.config, "TRANSPORTABILITY_THRESHOLD", 0.7)
        best = 0.0
        seeds = set(seed_names)
        for edge in self._edges:
            if not is_prerequisite_edge(edge):
                continue
            if _edge_transportability(edge) < threshold:
                continue
            if edge.source == skill_name and edge.target in seeds:
                best = max(best, _edge_score(edge) * 0.9)
            elif edge.target == skill_name and edge.source in seeds:
                best = max(best, _edge_score(edge))
        return best

    def _bundle_connectivity_gain(
        self, skill_name: str, selected_names: set[str]
    ) -> float:
        if not selected_names:
            return 0.0

        threshold = getattr(self.config, "TRANSPORTABILITY_THRESHOLD", 0.7)
        gain = 0.0
        for edge in self._edges:
            if not is_prerequisite_edge(edge):
                continue
            if _edge_transportability(edge) < threshold:
                continue
            if edge.source == skill_name and edge.target in selected_names:
                gain += _edge_score(edge) * 0.18
            elif edge.target == skill_name and edge.source in selected_names:
                gain += _edge_score(edge) * 0.22
        return min(gain, 0.35)

    @staticmethod
    def _skill_namespace(skill_name: str) -> str:
        """Infer a coarse skill namespace from the name.

        The skill library convention uses stable name prefixes for domain or
        tool families. Treating the prefix as a namespace prevents unrelated
        families from crowding a short bundle unless the graph has strong
        support that they are transportable bridges.
        """
        match = re.match(r"^([a-z][a-z0-9]{2,})-", (skill_name or "").lower())
        return match.group(1) if match else ""

    def _infer_query_namespaces(
        self,
        query: str,
        relevance_scores: dict[str, float],
    ) -> set[str]:
        signature_scores = self._namespace_signature_scores(query)
        signature_best = None
        if signature_scores:
            signature_best = max(
                signature_scores.items(),
                key=lambda item: (item[1][0], item[1][1]),
            )

        if not relevance_scores:
            if signature_best and self._strong_namespace_signature(signature_best[1]):
                return {signature_best[0]}
            return set()

        ranked = sorted(
            relevance_scores.items(), key=lambda item: item[1], reverse=True
        )[:8]
        namespace_mass: dict[str, float] = {}
        namespace_count: dict[str, int] = {}
        total_mass = 0.0
        for name, score in ranked:
            namespace = self._skill_namespace(name)
            if not namespace:
                continue
            mass = max(float(score or 0.0), 0.0)
            namespace_mass[namespace] = namespace_mass.get(namespace, 0.0) + mass
            namespace_count[namespace] = namespace_count.get(namespace, 0) + 1
            total_mass += mass

        if not namespace_mass or total_mass <= 0.0:
            if signature_best and self._strong_namespace_signature(signature_best[1]):
                return {signature_best[0]}
            return set()

        best_namespace, best_mass = max(
            namespace_mass.items(), key=lambda item: item[1]
        )
        best_share = best_mass / total_mass

        if signature_best:
            signature_namespace, signature_value = signature_best
            candidate_signature_value = signature_scores.get(best_namespace, (0.0, 0))[0]
            if (
                signature_namespace != best_namespace
                and self._strong_namespace_signature(signature_value)
                and signature_value[0] >= max(candidate_signature_value * 1.6, 6.0)
            ):
                return {signature_namespace}

        if namespace_count.get(best_namespace, 0) < 2 or best_share < 0.4:
            if signature_best and self._strong_namespace_signature(signature_best[1]):
                return {signature_best[0]}
            return set()

        # Runtime tasks normally belong to one skill namespace. Secondary
        # namespaces should enter through explicit transportability bridges,
        # not because a noisy vector/PPR shortlist gave them modest mass.
        return {best_namespace}

    @staticmethod
    def _strong_namespace_signature(signature: tuple[float, int]) -> bool:
        score, support_count = signature
        return support_count >= 2 and score >= 4.0

    @staticmethod
    def _signature_tokens(values: list[str] | tuple[str, ...]) -> set[str]:
        stopwords = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "that",
            "this",
            "into",
            "onto",
            "inside",
            "outside",
            "there",
            "here",
            "task",
            "your",
            "you",
            "are",
            "was",
            "were",
            "have",
            "has",
            "had",
            "action",
            "observation",
            "recent",
            "runtime",
            "current",
            "need",
            "needs",
            "use",
            "uses",
            "using",
            "when",
            "first",
            "initial",
            "some",
            "none",
            "one",
            "two",
            "put",
            "puts",
            "place",
            "placed",
            "middle",
            "room",
            "looking",
            "quickly",
            "around",
            "see",
            "sees",
            "seen",
        }
        tokens: set[str] = set()
        for value in values:
            for token in re.findall(r"[a-z][a-z0-9_]{2,}", str(value or "").lower()):
                if token not in stopwords:
                    tokens.add(token)
        return tokens

    def _node_signature_tokens(self, node: SkillNode) -> set[str]:
        return self._signature_tokens(
            [
                node.name,
                node.description,
                node.one_line_capability,
                node.inputs,
                node.outputs,
                node.domain_tags,
                node.tooling,
                node.example_tasks,
                node.script_entrypoints,
                node.rendered_snippet,
                node.preconditions,
                node.effects,
            ]
        )

    def _namespace_signature_scores(
        self,
        query: str,
    ) -> dict[str, tuple[float, int]]:
        query_tokens = self._signature_tokens([query])
        if not query_tokens:
            return {}

        namespace_node_tokens: dict[str, list[set[str]]] = {}
        for node in self._nodes:
            namespace = self._skill_namespace(node.name)
            if not namespace:
                continue
            namespace_node_tokens.setdefault(namespace, []).append(
                self._node_signature_tokens(node)
            )
        if not namespace_node_tokens:
            return {}

        token_namespace_count: dict[str, int] = {}
        for token in query_tokens:
            token_namespace_count[token] = sum(
                1
                for node_tokens_list in namespace_node_tokens.values()
                if any(token in node_tokens for node_tokens in node_tokens_list)
            )

        namespace_count = len(namespace_node_tokens)
        scores: dict[str, tuple[float, int]] = {}
        for namespace, node_tokens_list in namespace_node_tokens.items():
            node_scores: list[float] = []
            for node_tokens in node_tokens_list:
                overlap = query_tokens & node_tokens
                if not overlap:
                    continue
                node_score = 0.0
                for token in overlap:
                    token_df = token_namespace_count.get(token, 0)
                    node_score += 1.0 + max(
                        0.0,
                        math.log((namespace_count + 1.0) / (token_df + 1.0)),
                    )
                node_score *= 1.0 + min(len(overlap), 6) * 0.05
                node_scores.append(node_score)
            if not node_scores:
                continue
            top_scores = sorted(node_scores, reverse=True)[:12]
            scores[namespace] = (
                sum(top_scores) + min(len(node_scores), 20) * 0.15,
                len(node_scores),
            )
        return scores

    def _augment_namespace_signature_candidates(
        self,
        query: str,
        candidates: dict[str, float],
        preferred_namespaces: set[str],
    ) -> None:
        if not preferred_namespaces:
            return

        existing_preferred = sum(
            1
            for name in candidates
            if self._namespace_matches(name, preferred_namespaces)
        )
        if existing_preferred >= 2:
            return

        query_tokens = self._signature_tokens([query])
        if not query_tokens:
            return

        scored_nodes: list[tuple[float, str]] = []
        for node in self._nodes:
            if not self._namespace_matches(node.name, preferred_namespaces):
                continue
            overlap = query_tokens & self._node_signature_tokens(node)
            if not overlap:
                continue
            score = float(len(overlap))
            score += min(len(overlap), 6) * 0.08
            scored_nodes.append((score, node.name))

        if not scored_nodes:
            return

        scored_nodes.sort(key=lambda item: item[0], reverse=True)
        top_k = max(
            1,
            int(
                getattr(
                    self.config,
                    "CAUSAL_RETRIEVAL_NAMESPACE_SIGNATURE_TOP_K",
                    12,
                )
            ),
        )
        max_score = scored_nodes[0][0]
        for raw_score, name in scored_nodes[:top_k]:
            normalized = raw_score / max(max_score, 1.0)
            candidates[name] = max(
                candidates.get(name, 0.0),
                min(0.55, 0.12 + 0.38 * normalized),
            )

    def _namespace_matches(
        self, skill_name: str, preferred_namespaces: set[str]
    ) -> bool:
        namespace = self._skill_namespace(skill_name)
        return bool(namespace and namespace in preferred_namespaces)

    def _namespace_affinity_multiplier(
        self, skill_name: str, preferred_namespaces: set[str]
    ) -> float:
        if not preferred_namespaces:
            return 1.0
        if self._namespace_matches(skill_name, preferred_namespaces):
            return 1.08
        if self._has_transportable_namespace_bridge(skill_name, preferred_namespaces):
            return 0.85
        return 0.35

    def _has_transportable_namespace_bridge(
        self, skill_name: str, preferred_namespaces: set[str]
    ) -> bool:
        if not preferred_namespaces:
            return True

        threshold = getattr(self.config, "TRANSPORTABILITY_THRESHOLD", 0.7)
        skill_namespace = self._skill_namespace(skill_name)
        for edge in self._edges:
            if not is_prerequisite_edge(edge):
                continue
            if skill_name not in {edge.source, edge.target}:
                continue
            other = edge.target if edge.source == skill_name else edge.source
            other_namespace = self._skill_namespace(other)
            if other_namespace not in preferred_namespaces:
                continue
            transportability = getattr(edge, "transportability", None)
            transportability = float(transportability or 0.0)
            if (
                transportability == 0.0
                and skill_namespace
                and skill_namespace == other_namespace
                and getattr(edge, "status", "") in {
                    "confirmed_causal",
                    "stable",
                    "deferred_uncertain",
                }
            ):
                transportability = 1.0
            causal_score = float(getattr(edge, "causal_score", 0.0) or 0.0)
            if transportability >= threshold and causal_score >= 0.45:
                return True
        return False

    def _redundant_with_any(self, skill_name: str, selected_names: set[str]) -> bool:
        if not selected_names:
            return False
        return any(
            is_redundancy_edge(edge)
            and skill_name in {edge.source, edge.target}
            and bool(({edge.source, edge.target} - {skill_name}) & selected_names)
            for edge in self._edges
        )

    def _conflicts_with_any(self, skill_name: str, selected_names: set[str]) -> bool:
        if not selected_names:
            return False
        return any(
            is_conflict_edge(edge)
            and skill_name in {edge.source, edge.target}
            and bool(({edge.source, edge.target} - {skill_name}) & selected_names)
            for edge in self._edges
        )

    def compute_transportability(self) -> int:
        """Phase G: Update transportability scores for all edges."""
        self._edges = self.transportability.compute_all(self._edges)
        return len(self._edges)

    def get_invariant_subgraph(self) -> list[SkillEdge]:
        """Get edges stable across environments."""
        return self.transportability.identify_invariant_subgraph(self._edges)

    def get_health_report(self) -> GraphHealthReport:
        """Get current graph health."""
        return self.diagnostics.diagnose(self._edges, self._nodes)

    def _find_alternatives(self, edge: SkillEdge) -> list[str]:
        """Find alternative skills for substitution probes."""
        # Skills with similar edges or in same domain as edge.source
        source_node = next(
            (n for n in self._nodes if n.name == edge.source), None
        )
        if not source_node:
            return []
        alternatives = []
        for node in self._nodes:
            if node.name == edge.source:
                continue
            # Check if similar domain/capability
            node_tags = {t for t in node.domain_tags.split("\n") if t.strip()}
            source_tags = {t for t in source_node.domain_tags.split("\n") if t.strip()}
            if (
                node_tags
                and source_tags
                and node_tags & source_tags
            ):
                alternatives.append(node.name)
        return alternatives[:3]

    def load_state(self, state_dict: dict[str, Any]) -> None:
        """Restore causal graph state from a previously exported state dict.

        Loads episode count and updates edge causal metadata (causal_score,
        uncertainty, status, alpha_posterior, beta_posterior, intervention_count,
        transportability, association_score) from a caskg_state.json file
        previously written by export_state().
        """
        self._episode = state_dict.get("episode", 0)

        # Build a lookup of saved edge metadata keyed by (source, target)
        saved_edges: dict[tuple[str, str], dict[str, Any]] = {}
        for edge_data in state_dict.get("edges", []):
            key = (edge_data.get("source", ""), edge_data.get("target", ""))
            saved_edges[key] = edge_data

        # Update existing edges with saved causal metadata
        updated_edges: list[SkillEdge] = []
        for edge in self._edges:
            key = (edge.source, edge.target)
            if key in saved_edges:
                data = saved_edges[key]
                edge = dc_replace(
                    edge,
                    causal_score=data.get("causal_score", edge.causal_score),
                    uncertainty=data.get("uncertainty", edge.uncertainty),
                    status=data.get("status", edge.status),
                    association_score=data.get("association_score", edge.association_score),
                    transportability=data.get("transportability", edge.transportability),
                    intervention_count=data.get("intervention_count", edge.intervention_count),
                    alpha_posterior=data.get("alpha_posterior", edge.alpha_posterior),
                    beta_posterior=data.get("beta_posterior", edge.beta_posterior),
                    last_validated_episode=data.get("last_validated_episode", edge.last_validated_episode),
                )
            updated_edges.append(edge)

        # Also add edges that exist in saved state but not in current graph
        existing_pairs = {(e.source, e.target) for e in updated_edges}
        for key, data in saved_edges.items():
            if key not in existing_pairs:
                updated_edges.append(SkillEdge(
                    source=data.get("source", ""),
                    target=data.get("target", ""),
                    description=f"Restored from caskg_state: {data.get('source', '')} -> {data.get('target', '')}",
                    type=data.get("type", "dependency"),
                    causal_score=data.get("causal_score", 0.0),
                    uncertainty=data.get("uncertainty", 1.0),
                    status=data.get("status", "unverified"),
                    association_score=data.get("association_score", 0.0),
                    transportability=data.get("transportability", 0.0),
                    intervention_count=data.get("intervention_count", 0),
                    alpha_posterior=data.get("alpha_posterior", 1.0),
                    beta_posterior=data.get("beta_posterior", 1.0),
                    last_validated_episode=data.get("last_validated_episode", 0),
                ))

        self._edges = updated_edges

        # Restore Bayesian estimators in the InterventionEngine from the
        # saved alpha/beta posteriors on edges.  Without this, the
        # InterventionEngine would create fresh estimators from scratch,
        # diverging from the saved posteriors used for status decisions.
        for edge in self._edges:
            if edge.alpha_posterior != 1.0 or edge.beta_posterior != 1.0:
                estimator = self.intervention_engine.get_estimator(
                    edge.source, edge.target
                )
                estimator.alpha = edge.alpha_posterior
                estimator.beta = edge.beta_posterior

    def export_state(self) -> dict[str, Any]:
        """Export full causal graph state for persistence."""
        return {
            "episode": self._episode,
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "type": e.type,
                    "causal_score": e.causal_score,
                    "uncertainty": e.uncertainty,
                    "transportability": e.transportability,
                    "status": e.status,
                    "association_score": e.association_score,
                    "intervention_count": e.intervention_count,
                    "alpha_posterior": e.alpha_posterior,
                    "beta_posterior": e.beta_posterior,
                    "last_validated_episode": e.last_validated_episode,
                }
                for e in self._edges
            ],
            "stats": self.stats,
        }
