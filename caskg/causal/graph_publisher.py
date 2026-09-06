"""Publish CaSKG candidate/validated state into the retrieval graph.

The offline causal state is stored in ``caskg_state.json``. Runtime retrieval,
however, reads ``graph_igraph_data.pklz`` through ``SkillGraphRAG``. This module
keeps those two representations aligned without adding evaluation-time rerank
logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from loguru import logger

from caskg.core.engine import SkillGraphRAG
from caskg.core.schema import SkillEdge


LIVE_STATUSES = {"confirmed_causal", "stable", "deferred_uncertain"}
REJECTED_STATUSES = {"rejected_non_causal", "decayed", "pruned"}
SCAFFOLD_STATUSES = {"deferred_unvalidated"}

TYPE_ALIASES = {
    "prereq": "dependency",
    "prerequisite": "dependency",
    "data-flow": "dependency",
    "data_flow": "dependency",
    "enhance": "workflow",
    "repair-support": "workflow",
    "repair_support": "workflow",
    "similar": "semantic",
    "cooccur": "semantic",
}

STATUS_WEIGHT_SCALE = {
    "confirmed_causal": 1.0,
    "stable": 1.0,
    "deferred_uncertain": 0.55,
    "deferred_unvalidated": 0.18,
    "unverified": 0.35,
}

SCAFFOLD_TYPE_PRIORITY = {
    "dependency": 5,
    "prereq": 5,
    "prerequisite": 5,
    "data-flow": 5,
    "data_flow": 5,
    "workflow": 4,
    "enhance": 4,
    "repair-support": 4,
    "repair_support": 4,
    "semantic": 3,
    "similar": 2,
    "alternative": 1,
    "cooccur": 1,
}


def _normalize_type(edge_type: str) -> str:
    normalized = str(edge_type or "semantic").strip().lower()
    return TYPE_ALIASES.get(normalized, normalized or "semantic")


def _edge_score(edge: dict[str, Any]) -> float:
    status = str(edge.get("status") or "unverified").strip().lower()
    causal_score = float(edge.get("causal_score") or 0.0)
    association_score = float(edge.get("association_score") or 0.0)
    weight = float(edge.get("weight") or edge.get("confidence") or 0.0)

    base = max(causal_score, association_score, weight, 0.01)
    scale = STATUS_WEIGHT_SCALE.get(status, 0.0)
    return max(0.01, min(1.0, base * scale))


def _is_publishable(edge: dict[str, Any], *, include_unverified: bool) -> bool:
    status = str(edge.get("status") or "unverified").strip().lower()
    if status in REJECTED_STATUSES or status in SCAFFOLD_STATUSES:
        return False
    if status in LIVE_STATUSES:
        return True
    return include_unverified and status == "unverified"


def _namespace(name: str) -> str:
    prefix = (name or "").split("-", 1)[0]
    return prefix if len(prefix) >= 3 else ""


def _same_namespace(source: str, target: str) -> bool:
    source_namespace = _namespace(source)
    target_namespace = _namespace(target)
    return bool(
        source_namespace
        and target_namespace
        and source_namespace == target_namespace
    )


def _raw_edge_priority(edge: dict[str, Any]) -> tuple[int, int, float, float, str, str]:
    edge_type = _normalize_type(str(edge.get("type") or "semantic"))
    same_namespace = _same_namespace(
        str(edge.get("source") or ""),
        str(edge.get("target") or ""),
    )
    return (
        SCAFFOLD_TYPE_PRIORITY.get(edge_type, 0),
        1 if same_namespace else 0,
        float(edge.get("association_score") or 0.0),
        float(
            edge.get("causal_score")
            or edge.get("weight")
            or edge.get("confidence")
            or 0.0
        ),
        str(edge.get("source") or ""),
        str(edge.get("target") or ""),
    )


def _make_edge(raw: dict[str, Any]) -> SkillEdge | None:
    source = str(raw.get("source") or "").strip()
    target = str(raw.get("target") or "").strip()
    if not source or not target or source == target:
        return None

    edge_type = _normalize_type(str(raw.get("type") or "semantic"))
    status = str(raw.get("status") or "unverified").strip().lower()
    weight = _edge_score(raw)
    description = str(raw.get("description") or "").strip()
    if status in SCAFFOLD_STATUSES:
        description = (
            f"CaSKG low-weight {edge_type} scaffold edge "
            f"(association={float(raw.get('association_score') or 0.0):.3f})"
        )
    elif not description:
        description = (
            f"CaSKG {status} {edge_type} edge "
            f"(association={float(raw.get('association_score') or 0.0):.3f}, "
            f"causal={float(raw.get('causal_score') or 0.0):.3f})"
        )

    return SkillEdge(
        source=source,
        target=target,
        description=description,
        type=edge_type,
        weight=weight,
        confidence=weight,
        causal_score=float(raw.get("causal_score") or 0.0),
        uncertainty=float(
            raw.get("uncertainty")
            if raw.get("uncertainty") is not None
            else 1.0
        ),
        association_score=float(raw.get("association_score") or 0.0),
        transportability=float(raw.get("transportability") or 0.0),
        status=status,
        intervention_count=int(raw.get("intervention_count") or 0),
        alpha_posterior=float(raw.get("alpha_posterior") or 1.0),
        beta_posterior=float(raw.get("beta_posterior") or 1.0),
        last_validated_episode=int(raw.get("last_validated_episode") or 0),
    )


def _add_edge(
    by_key: dict[tuple[str, str, str], SkillEdge],
    edge: SkillEdge,
) -> bool:
    key = (edge.source, edge.target, edge.type)
    current = by_key.get(key)
    if current is not None and (current.weight, current.association_score) >= (
        edge.weight,
        edge.association_score,
    ):
        return False
    by_key[key] = edge
    return True


def _select_scaffold_edges(
    state_edges: Iterable[dict[str, Any]],
    *,
    valid_node_names: set[str],
    existing_keys: set[tuple[str, str, str]],
    initial_degree: dict[str, int],
    max_out_per_node: int,
    max_in_per_node: int,
    max_total: int,
) -> list[SkillEdge]:
    candidates: list[dict[str, Any]] = []
    for raw in state_edges:
        status = str(raw.get("status") or "unverified").strip().lower()
        if status not in SCAFFOLD_STATUSES:
            continue
        source = str(raw.get("source") or "").strip()
        target = str(raw.get("target") or "").strip()
        if (
            not source
            or not target
            or source == target
            or source not in valid_node_names
            or target not in valid_node_names
        ):
            continue
        edge_type = _normalize_type(str(raw.get("type") or "semantic"))
        if SCAFFOLD_TYPE_PRIORITY.get(edge_type, 0) <= 0:
            continue
        if (source, target, edge_type) in existing_keys:
            continue
        candidates.append(raw)

    candidates.sort(key=_raw_edge_priority, reverse=True)

    out_counts: dict[str, int] = {}
    in_counts: dict[str, int] = {}
    degree = dict(initial_degree)
    selected: list[SkillEdge] = []
    selected_keys: set[tuple[str, str, str]] = set()

    def can_add(edge: SkillEdge) -> bool:
        if len(selected) >= max_total:
            return False
        if (edge.source, edge.target, edge.type) in selected_keys:
            return False
        if out_counts.get(edge.source, 0) >= max_out_per_node:
            return False
        if in_counts.get(edge.target, 0) >= max_in_per_node:
            return False
        return True

    def add(edge: SkillEdge) -> bool:
        if not can_add(edge):
            return False
        selected.append(edge)
        selected_keys.add((edge.source, edge.target, edge.type))
        out_counts[edge.source] = out_counts.get(edge.source, 0) + 1
        in_counts[edge.target] = in_counts.get(edge.target, 0) + 1
        degree[edge.source] = degree.get(edge.source, 0) + 1
        degree[edge.target] = degree.get(edge.target, 0) + 1
        return True

    # First pass: prefer edges that attach isolated nodes to the published graph.
    for raw in candidates:
        edge = _make_edge(raw)
        if edge is None:
            continue
        if degree.get(edge.source, 0) > 0 and degree.get(edge.target, 0) > 0:
            continue
        add(edge)

    # Second pass: fill a bounded local scaffold without letting any node dominate.
    for raw in candidates:
        edge = _make_edge(raw)
        if edge is None:
            continue
        add(edge)

    return selected


def build_published_edges(
    state_edges: Iterable[dict[str, Any]],
    *,
    valid_node_names: set[str],
    include_unverified: bool = False,
    include_scaffold: bool = True,
    scaffold_max_out_per_node: int = 4,
    scaffold_max_in_per_node: int = 4,
    scaffold_max_edges: int | None = None,
) -> list[SkillEdge]:
    """Convert causal state edges into deduplicated retrieval graph edges.

    Confirmed/stable/deferred-uncertain edges are published as causal support.
    Deferred-unvalidated edges are not causal evidence, but a bounded low-weight
    scaffold keeps the runtime graph connected enough for PPR over large skill
    libraries. Causal retrieval still ignores those scaffold edges by status.
    """
    by_key: dict[tuple[str, str, str], SkillEdge] = {}
    state_edges = list(state_edges)
    degree: dict[str, int] = {}

    for raw in state_edges:
        source = str(raw.get("source") or "").strip()
        target = str(raw.get("target") or "").strip()
        if not source or not target or source == target:
            continue
        if source not in valid_node_names or target not in valid_node_names:
            continue
        if not _is_publishable(raw, include_unverified=include_unverified):
            continue

        edge = _make_edge(raw)
        if edge is not None and _add_edge(by_key, edge):
            degree[edge.source] = degree.get(edge.source, 0) + 1
            degree[edge.target] = degree.get(edge.target, 0) + 1

    if include_scaffold:
        scaffold_limit = scaffold_max_edges
        if scaffold_limit is None:
            scaffold_limit = max(len(valid_node_names) * 4, len(by_key))
        scaffold_edges = _select_scaffold_edges(
            state_edges,
            valid_node_names=valid_node_names,
            existing_keys=set(by_key),
            initial_degree=degree,
            max_out_per_node=max(1, scaffold_max_out_per_node),
            max_in_per_node=max(1, scaffold_max_in_per_node),
            max_total=max(0, scaffold_limit),
        )
        for edge in scaffold_edges:
            _add_edge(by_key, edge)

    return sorted(by_key.values(), key=lambda e: (e.source, e.target, e.type))


async def publish_state_edges_to_workspace(
    workspace: str | Path,
    state: dict[str, Any],
    *,
    include_unverified: bool = False,
) -> dict[str, int]:
    """Replace the workspace retrieval graph edges with publishable state edges."""
    workspace = Path(workspace)
    rag = SkillGraphRAG(
        config=SkillGraphRAG.Config(
            working_dir=str(workspace),
            prebuilt_working_dir=str(workspace),
        )
    )

    await rag.state_manager.insert_start()
    try:
        target = rag.state_manager.graph_storage
        nodes = await rag._load_all_nodes()
        valid_node_names = {node.name for node in nodes}
        published_edges = build_published_edges(
            state.get("edges", []),
            valid_node_names=valid_node_names,
            include_unverified=include_unverified,
        )

        old_edge_count = await target.edge_count()
        if old_edge_count:
            await target.delete_edges_by_index(range(old_edge_count))
        if published_edges:
            await target.insert_edges(published_edges)
    finally:
        await rag.state_manager.insert_done()

    stats = {
        "nodes": len(valid_node_names),
        "old_edges": old_edge_count,
        "published_edges": len(published_edges),
    }
    logger.info(
        (
            "Published CaSKG state edges to retrieval graph: "
            "nodes={} old_edges={} published_edges={}"
        ),
        stats["nodes"],
        stats["old_edges"],
        stats["published_edges"],
    )
    return stats
