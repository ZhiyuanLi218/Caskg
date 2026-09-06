"""Build the isolated A1--A4 graph views for the numbered CaSKG ablation.

The source Skill1000 workspace is treated as immutable.  A1 is a byte-identical
runtime copy used by the vector-only runtime adapter.  A2 rewires targets with
degree-preserving directed edge swaps within each edge type.  A3 keeps the A0
topology and replaces only the online PPR weight/confidence with the frozen
Phase-1 association score.  A4 keeps the A0 topology and moves complete Phase-2
evidence packages to matched incorrect edges.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ablation_experiments.adapters.causal_graph_views import (
    EDGE_ATTRIBUTE_FIELDS,
    EDGE_DEFAULTS,
    EVIDENCE_PACKAGE_FIELDS,
    EXPECTED_CHECKPOINT_SHA256,
    RUNTIME_WORKSPACE_FILES,
    VALIDATED_STATUSES,
    AblationAuditError,
    _materialize_graph,
    _package_multiset,
    _records_sha256,
    _runtime_edge_records,
    _topology_multiset,
    load_a0_reference,
    load_graph,
    load_phase1_checkpoint,
    match_c3_evidence_packages,
    normalize_type,
    sha256_file,
)


PROTOCOL_ID = "caskg-s1000-a1-a4-main-parity-v1"
EXPECTED_SOURCE_GRAPH_SHA256 = (
    "ff64ad00ef7b35e29586ac59ed8bec07f0a1452555a9ff19f89712fd85654770"
)
EXPECTED_NODE_COUNT = 1000
EXPECTED_EDGE_COUNT = 3292
MIN_A2_ENDPOINT_CHANGE_RATE = 0.95

A0_CONTROL_VARIANT = "p0-api2-full-control"
A1_VARIANT = "a1-vector-only-no-ppr"
A2_VARIANT = "a2-matched-rewired-s7302"
A3_VARIANT = "a3-fixed-topology-phase1-weights"
A4_VARIANT = "a4-phase2-evidence-shuffle-s7301"
VARIANTS = (A0_CONTROL_VARIANT, A1_VARIANT, A2_VARIANT, A3_VARIANT, A4_VARIANT)


def _json_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
        newline="\n",
    )


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for record in records:
            handle.write(_json_key(record))
            handle.write("\n")


def _runtime_hashes(workspace: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for filename in RUNTIME_WORKSPACE_FILES:
        path = workspace / filename
        if not path.is_file():
            raise AblationAuditError(f"Missing runtime workspace asset: {path}")
        hashes[filename] = sha256_file(path)
    return hashes


def _copy_runtime_workspace(
    source: Path,
    destination: Path,
    *,
    include_graph: bool,
) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for filename in RUNTIME_WORKSPACE_FILES:
        if filename == "graph_igraph_data.pklz" and not include_graph:
            continue
        source_path = source / filename
        if not source_path.is_file():
            raise AblationAuditError(f"Missing source runtime asset: {source_path}")
        shutil.copy2(source_path, destination / filename)


def _atomic_replace(destination: Path, staged: Path, *, force: bool) -> None:
    if destination.exists() and not force:
        raise AblationAuditError(f"Refusing to overwrite graph views: {destination}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staged), str(destination))


def _canonical_pair(source: str, target: str) -> tuple[str, str]:
    return tuple(sorted((str(source), str(target))))


def _topology_counter(
    edges: Iterable[Mapping[str, Any]],
) -> Counter[tuple[str, str, str]]:
    return Counter(
        (*_canonical_pair(str(edge["source"]), str(edge["target"])), str(edge["type"]))
        for edge in edges
    )


def _degree_by_type(edges: Iterable[Mapping[str, Any]]) -> Counter[tuple[str, str]]:
    degree: Counter[tuple[str, str]] = Counter()
    for edge in edges:
        edge_type = str(edge["type"])
        source = str(edge["source"])
        target = str(edge["target"])
        degree[(source, edge_type)] += 1
        degree[(target, edge_type)] += 1
    return degree


def _attribute_package_multiset(
    edges: Iterable[Mapping[str, Any]],
) -> Counter[str]:
    return Counter(
        _json_key(
            {
                field: edge.get(field, EDGE_DEFAULTS[field])
                for field in EDGE_ATTRIBUTE_FIELDS
                if field not in {"type"}
            }
        )
        for edge in edges
    )


def _stable_type_seed(seed: int, edge_type: str) -> int:
    digest = hashlib.sha256(f"{seed}|{edge_type}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def build_a2_rewired_edges(
    a0_edges: Sequence[Mapping[str, Any]],
    *,
    seed: int = 7302,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rewire endpoints while preserving every node's degree within each type.

    Each successful directed double-edge swap exchanges the targets of two rows.
    Consequently source and target multisets are both exact, which also preserves
    undirected per-node degree-by-type.  Edge attribute packages remain attached
    to their rows, so their global multiset is unchanged.
    """

    rewired = [copy.deepcopy(dict(edge)) for edge in a0_edges]
    indices_by_type: dict[str, list[int]] = defaultdict(list)
    for index, edge in enumerate(rewired):
        indices_by_type[str(edge["type"])].append(index)

    type_audits: dict[str, Any] = {}
    for edge_type, indices in sorted(indices_by_type.items()):
        rng = random.Random(_stable_type_seed(seed, edge_type))
        pair_counts = Counter(
            _canonical_pair(rewired[index]["source"], rewired[index]["target"])
            for index in indices
        )
        max_parallel_multiplicity = max(pair_counts.values())

        target_successes = max(200, len(indices) * 40)
        max_attempts = target_successes * 100
        successes = 0
        attempts = 0
        while successes < target_successes and attempts < max_attempts:
            attempts += 1
            first, second = rng.sample(indices, 2)
            first_edge = rewired[first]
            second_edge = rewired[second]
            source_a = str(first_edge["source"])
            target_a = str(first_edge["target"])
            source_b = str(second_edge["source"])
            target_b = str(second_edge["target"])
            if target_a == target_b:
                continue

            old_a = _canonical_pair(source_a, target_a)
            old_b = _canonical_pair(source_b, target_b)
            new_a = _canonical_pair(source_a, target_b)
            new_b = _canonical_pair(source_b, target_a)
            if source_a == target_b or source_b == target_a:
                continue
            old_pair_delta = Counter((old_a, old_b))
            new_pair_delta = Counter((new_a, new_b))
            if old_pair_delta == new_pair_delta:
                continue
            remaining = pair_counts.copy()
            remaining.subtract(old_pair_delta)
            if any(value < 0 for value in remaining.values()):
                raise AblationAuditError("A2 pair accounting became negative.")
            if any(
                remaining[pair] + increment > max_parallel_multiplicity
                for pair, increment in new_pair_delta.items()
            ):
                continue

            pair_counts = remaining
            pair_counts.update(new_pair_delta)
            pair_counts += Counter()
            first_edge["target"] = target_b
            second_edge["target"] = target_a
            successes += 1

        if successes < target_successes:
            raise AblationAuditError(
                f"A2 could not complete deterministic swaps for {edge_type}: "
                f"{successes}/{target_successes}."
            )
        if sum(pair_counts.values()) != len(indices):
            raise AblationAuditError(f"A2 changed the {edge_type} edge multiplicity.")
        type_audits[edge_type] = {
            "edge_count": len(indices),
            "successful_target_swaps": successes,
            "swap_attempts": attempts,
            "maximum_parallel_multiplicity": max_parallel_multiplicity,
        }

    original_topology = _topology_counter(a0_edges)
    rewired_topology = _topology_counter(rewired)
    overlap = sum((original_topology & rewired_topology).values())
    changed = len(rewired) - overlap
    change_rate = changed / len(rewired)
    if change_rate < MIN_A2_ENDPOINT_CHANGE_RATE:
        raise AblationAuditError(
            f"A2 endpoint change rate is too weak: {change_rate:.6f}."
        )
    if any(str(edge["source"]) == str(edge["target"]) for edge in rewired):
        raise AblationAuditError("A2 created a self-loop.")
    if _degree_by_type(a0_edges) != _degree_by_type(rewired):
        raise AblationAuditError("A2 changed per-node degree-by-type.")
    if _attribute_package_multiset(a0_edges) != _attribute_package_multiset(rewired):
        raise AblationAuditError("A2 changed the edge attribute-package multiset.")

    for edge_type, indices in sorted(indices_by_type.items()):
        original_sources = Counter(str(a0_edges[index]["source"]) for index in indices)
        original_targets = Counter(str(a0_edges[index]["target"]) for index in indices)
        rewired_sources = Counter(str(rewired[index]["source"]) for index in indices)
        rewired_targets = Counter(str(rewired[index]["target"]) for index in indices)
        if original_sources != rewired_sources or original_targets != rewired_targets:
            raise AblationAuditError(f"A2 changed endpoint marginals for {edge_type}.")
        original_type_topology = Counter(
            (*_canonical_pair(a0_edges[index]["source"], a0_edges[index]["target"]), edge_type)
            for index in indices
        )
        rewired_type_topology = Counter(
            (*_canonical_pair(rewired[index]["source"], rewired[index]["target"]), edge_type)
            for index in indices
        )
        type_overlap = sum((original_type_topology & rewired_type_topology).values())
        type_audits[edge_type].update(
            {
                "original_topology_overlap": type_overlap,
                "endpoint_change_rate": 1.0 - (type_overlap / len(indices)),
                "source_multiset_exact": True,
                "target_multiset_exact": True,
            }
        )

    return rewired, {
        "seed": seed,
        "method": "within_type_directed_double_edge_swap",
        "edge_count": len(rewired),
        "original_topology_overlap": overlap,
        "changed_edge_count": changed,
        "endpoint_change_rate": change_rate,
        "minimum_required_endpoint_change_rate": MIN_A2_ENDPOINT_CHANGE_RATE,
        "degree_by_type_exact": True,
        "source_target_marginals_by_type_exact": True,
        "attribute_package_multiset_exact": True,
        "self_loop_count": 0,
        "parallel_edges_allowed_as_in_a0_multigraph": True,
        "by_type": type_audits,
    }


def _phase1_map(
    checkpoint_path: Path,
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[str, Any]]:
    records, audit = load_phase1_checkpoint(
        checkpoint_path,
        expected_sha256=EXPECTED_CHECKPOINT_SHA256,
    )
    mapping: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        key = (
            str(record["source"]),
            str(record["target"]),
            normalize_type(record["type"]),
        )
        if key in mapping:
            raise AblationAuditError(f"Duplicate normalized Phase-1 edge: {key}")
        mapping[key] = record
    return mapping, audit


def build_a3_phase1_weight_edges(
    a0_edges: Sequence[Mapping[str, Any]],
    checkpoint_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates, checkpoint_audit = _phase1_map(checkpoint_path)
    a3_edges: list[dict[str, Any]] = []
    changed_weight_count = 0
    for original in a0_edges:
        key = (
            str(original["source"]),
            str(original["target"]),
            normalize_type(original["type"]),
        )
        candidate = candidates.get(key)
        if candidate is None:
            raise AblationAuditError(f"A3 A0 edge is absent from Phase 1: {key}")
        association = float(candidate["association_score"])
        if abs(float(original["association_score"]) - association) > 1e-12:
            raise AblationAuditError(f"A3 association score mismatch: {key}")
        rewritten = copy.deepcopy(dict(original))
        if (
            abs(float(rewritten["weight"]) - association) > 1e-12
            or abs(float(rewritten["confidence"]) - association) > 1e-12
        ):
            changed_weight_count += 1
        rewritten["weight"] = association
        rewritten["confidence"] = association
        a3_edges.append(rewritten)

    if _topology_multiset(a0_edges, directed=True) != _topology_multiset(
        a3_edges, directed=True
    ):
        raise AblationAuditError("A3 changed the A0 topology.")
    for original, rewritten in zip(a0_edges, a3_edges):
        for field in original:
            if field in {"weight", "confidence"}:
                continue
            if original[field] != rewritten[field]:
                raise AblationAuditError(f"A3 changed forbidden field: {field}")

    return a3_edges, {
        "edge_count": len(a3_edges),
        "topology_exact": True,
        "phase1_checkpoint": checkpoint_audit,
        "a0_edges_matched_to_phase1": len(a3_edges),
        "a0_edges_missing_from_phase1": 0,
        "changed_weight_or_confidence_count": changed_weight_count,
        "only_changed_fields": ["confidence", "weight"],
        "weight_equals_phase1_association_for_all_edges": True,
        "confidence_equals_phase1_association_for_all_edges": True,
    }


def build_a4_shuffled_evidence_edges(
    a0_edges: Sequence[Mapping[str, Any]],
    *,
    seed: int = 7301,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    original = [copy.deepcopy(dict(edge)) for edge in a0_edges]
    shuffled = [copy.deepcopy(dict(edge)) for edge in a0_edges]
    matches, matching_audit = match_c3_evidence_packages(original, seed=seed)
    matching_records: list[dict[str, Any]] = []

    for validated_index, scaffold_index, tier in matches:
        validated_package = {
            field: copy.deepcopy(original[validated_index][field])
            for field in EVIDENCE_PACKAGE_FIELDS
        }
        scaffold_package = {
            field: copy.deepcopy(original[scaffold_index][field])
            for field in EVIDENCE_PACKAGE_FIELDS
        }
        shuffled[validated_index].update(scaffold_package)
        shuffled[scaffold_index].update(validated_package)
        matching_records.append(
            {
                "tier": tier,
                "validated_index": validated_index,
                "scaffold_index": scaffold_index,
                "validated_edge": {
                    "source": original[validated_index]["source"],
                    "target": original[validated_index]["target"],
                    "type": original[validated_index]["type"],
                },
                "scaffold_edge": {
                    "source": original[scaffold_index]["source"],
                    "target": original[scaffold_index]["target"],
                    "type": original[scaffold_index]["type"],
                },
            }
        )

    if _topology_multiset(original, directed=True) != _topology_multiset(
        shuffled, directed=True
    ):
        raise AblationAuditError("A4 changed the A0 topology.")
    if _package_multiset(original) != _package_multiset(shuffled):
        raise AblationAuditError("A4 changed the evidence-package multiset.")
    for original_edge, shuffled_edge in zip(original, shuffled):
        for field in original_edge:
            if field in EVIDENCE_PACKAGE_FIELDS:
                continue
            if original_edge[field] != shuffled_edge[field]:
                raise AblationAuditError(f"A4 changed forbidden field: {field}")

    displaced = sum(
        any(
            original[index][field] != shuffled[index][field]
            for field in EVIDENCE_PACKAGE_FIELDS
        )
        for index, edge in enumerate(original)
        if str(edge["status"]) in VALIDATED_STATUSES
    )
    if displaced != 285:
        raise AblationAuditError(f"A4 displaced {displaced}/285 validated packages.")

    return shuffled, matching_records, {
        **matching_audit,
        "edge_count": len(shuffled),
        "topology_exact": True,
        "evidence_package_multiset_exact": True,
        "validated_evidence_displaced": displaced,
        "validated_evidence_displacement_rate": displaced / 285.0,
        "description_fixed_at_original_endpoint": True,
        "chunks_fixed_at_original_endpoint": True,
        "only_changed_fields": sorted(EVIDENCE_PACKAGE_FIELDS),
    }


def _write_variant(
    root: Path,
    *,
    source_workspace: Path,
    source_graph: Any,
    source_edges: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
    copy_graph: bool,
    matching_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    edges_path = root / "edges.jsonl"
    _write_jsonl(edges_path, edges)
    if matching_records is not None:
        _write_jsonl(root / "matching.jsonl", matching_records)

    workspace = root / "workspace"
    _copy_runtime_workspace(
        source_workspace,
        workspace,
        include_graph=copy_graph,
    )
    graph_path = workspace / "graph_igraph_data.pklz"
    if not copy_graph:
        _materialize_graph(source_graph, edges, graph_path)

    materialized_graph = load_graph(graph_path)
    materialized_edges = _runtime_edge_records(materialized_graph)
    if materialized_graph.vcount() != EXPECTED_NODE_COUNT:
        raise AblationAuditError(f"Materialized node count drifted for {root.name}.")
    if materialized_graph.ecount() != EXPECTED_EDGE_COUNT:
        raise AblationAuditError(f"Materialized edge count drifted for {root.name}.")
    if _topology_multiset(materialized_edges, directed=False) != _topology_multiset(
        edges, directed=False
    ):
        raise AblationAuditError(f"Materialized topology drifted for {root.name}.")
    if _degree_by_type(materialized_edges) != _degree_by_type(edges):
        raise AblationAuditError(f"Materialized degree-by-type drifted for {root.name}.")

    final_audit = {
        **dict(audit),
        "protocol_id": PROTOCOL_ID,
        "variant_id": root.name,
        "source_graph_sha256": EXPECTED_SOURCE_GRAPH_SHA256,
        "node_count": materialized_graph.vcount(),
        "edge_count": materialized_graph.ecount(),
        "graph_directed": bool(materialized_graph.is_directed()),
        "edges_jsonl_sha256": sha256_file(edges_path),
        "edge_records_sha256": _records_sha256(edges),
        "workspace_runtime_files_sha256": _runtime_hashes(workspace),
        "materialized_topology_exact": True,
        "materialized_degree_by_type_exact": True,
        "type_counts": dict(
            sorted(Counter(str(edge["type"]) for edge in materialized_edges).items())
        ),
    }
    if copy_graph:
        final_audit["graph_sha256_identical_to_source"] = (
            sha256_file(graph_path) == EXPECTED_SOURCE_GRAPH_SHA256
        )
        if not final_audit["graph_sha256_identical_to_source"]:
            raise AblationAuditError("A1 graph is not byte-identical to A0.")
    if matching_records is not None:
        final_audit["matching_jsonl_sha256"] = sha256_file(root / "matching.jsonl")
    _write_json(root / "audit.json", final_audit)
    return final_audit


def build_numbered_graph_views(
    source_workspace: str | Path,
    output_root: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    source_workspace = Path(source_workspace).resolve()
    output_root = Path(output_root).resolve()
    source_hashes_before = _runtime_hashes(source_workspace)
    source_graph_path = source_workspace / "graph_igraph_data.pklz"
    checkpoint_path = source_workspace / "candidate_checkpoint.jsonl"
    if sha256_file(source_graph_path) != EXPECTED_SOURCE_GRAPH_SHA256:
        raise AblationAuditError("Skill1000 source graph checksum mismatch.")
    if sha256_file(checkpoint_path) != EXPECTED_CHECKPOINT_SHA256:
        raise AblationAuditError("Skill1000 Phase-1 checkpoint checksum mismatch.")

    source_graph, source_edges, source_reference_audit = load_a0_reference(
        source_workspace
    )
    if source_graph.is_directed():
        raise AblationAuditError("The A0 source graph must be undirected.")
    if source_graph.vcount() != EXPECTED_NODE_COUNT:
        raise AblationAuditError("The A0 source graph must contain 1000 nodes.")
    if len(source_edges) != EXPECTED_EDGE_COUNT:
        raise AblationAuditError("The A0 source graph must contain 3292 edges.")
    source_topology = _topology_counter(source_edges)
    source_parallel_excess = sum(count - 1 for count in source_topology.values())

    a1_edges = [copy.deepcopy(edge) for edge in source_edges]
    a2_edges, a2_audit = build_a2_rewired_edges(source_edges, seed=7302)
    a3_edges, a3_audit = build_a3_phase1_weight_edges(source_edges, checkpoint_path)
    a4_edges, a4_matches, a4_audit = build_a4_shuffled_evidence_edges(
        source_edges,
        seed=7301,
    )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_root.name}-", dir=str(output_root.parent)
    ) as temporary_root:
        staged_root = Path(temporary_root) / output_root.name
        staged_root.mkdir()
        variant_audits = {
            A1_VARIANT: _write_variant(
                staged_root / A1_VARIANT,
                source_workspace=source_workspace,
                source_graph=source_graph,
                source_edges=source_edges,
                edges=a1_edges,
                audit={
                    "treatment": "vector_only_runtime_no_ppr",
                    "graph_role": "byte_identical_a0_control",
                    "ppr_disabled_by_runtime_adapter": True,
                    "source_parallel_typed_edge_excess": source_parallel_excess,
                },
                copy_graph=True,
            ),
            A2_VARIANT: _write_variant(
                staged_root / A2_VARIANT,
                source_workspace=source_workspace,
                source_graph=source_graph,
                source_edges=source_edges,
                edges=a2_edges,
                audit={
                    "treatment": "matched_degree_preserving_topology_rewire",
                    **a2_audit,
                },
                copy_graph=False,
            ),
            A3_VARIANT: _write_variant(
                staged_root / A3_VARIANT,
                source_workspace=source_workspace,
                source_graph=source_graph,
                source_edges=source_edges,
                edges=a3_edges,
                audit={
                    "treatment": "fixed_a0_topology_phase1_runtime_weights",
                    **a3_audit,
                },
                copy_graph=False,
            ),
            A4_VARIANT: _write_variant(
                staged_root / A4_VARIANT,
                source_workspace=source_workspace,
                source_graph=source_graph,
                source_edges=source_edges,
                edges=a4_edges,
                audit={
                    "treatment": "fixed_a0_topology_phase2_evidence_package_shuffle",
                    **a4_audit,
                },
                copy_graph=False,
                matching_records=a4_matches,
            ),
        }
        manifest = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "source_workspace": str(source_workspace),
            "source_workspace_modified": False,
            "source_runtime_files_sha256_before": source_hashes_before,
            "source_graph_sha256": EXPECTED_SOURCE_GRAPH_SHA256,
            "phase1_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "a0_publisher_runtime_audit": source_reference_audit,
            "variants": variant_audits,
        }
        _write_json(staged_root / "asset_manifest.json", manifest)
        _atomic_replace(output_root, staged_root, force=force)

    source_hashes_after = _runtime_hashes(source_workspace)
    if source_hashes_after != source_hashes_before:
        raise AblationAuditError("Source workspace changed during graph-view generation.")
    manifest_path = output_root / "asset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    manifest["source_runtime_files_sha256_after"] = source_hashes_after
    manifest["source_workspace_modified"] = False
    _write_json(manifest_path, manifest)
    return manifest
