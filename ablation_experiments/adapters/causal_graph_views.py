"""Build audited C2/C3 graph views without mutating the CaSKG source workspace."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import pickle
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PHASE1_FIELDS = frozenset(
    {
        "source",
        "target",
        "type",
        "association_score",
        "weight",
        "confidence",
        "description",
    }
)
PHASE2_FIELDS = frozenset(
    {
        "status",
        "causal_score",
        "uncertainty",
        "alpha_posterior",
        "beta_posterior",
        "intervention_count",
        "transportability",
        "last_validated_episode",
    }
)
EVIDENCE_PACKAGE_FIELDS = (
    "status",
    "causal_score",
    "uncertainty",
    "alpha_posterior",
    "beta_posterior",
    "intervention_count",
    "transportability",
    "last_validated_episode",
    "weight",
    "confidence",
)
EDGE_ATTRIBUTE_FIELDS = (
    "description",
    "chunks",
    "type",
    "weight",
    "confidence",
    "causal_score",
    "uncertainty",
    "association_score",
    "transportability",
    "status",
    "context_condition",
    "intervention_count",
    "alpha_posterior",
    "beta_posterior",
    "last_validated_episode",
)
VALIDATED_STATUSES = frozenset({"confirmed_causal", "deferred_uncertain"})
SCAFFOLD_STATUS = "deferred_unvalidated"
EXPECTED_CHECKPOINT_SHA256 = (
    "dd46baaab214491ebcf75b769d4608e067abc5aefbf8b97253edad7924816be0"
)
EXPECTED_C3_TIER_COUNTS = {1: 160, 2: 15, 3: 50, 4: 55, 5: 5}
RUNTIME_WORKSPACE_FILES = (
    "chunks_kv_data.pkl",
    "entities_hnsw_index_4096.bin",
    "entities_hnsw_metadata.pkl",
    "graph_igraph_data.pklz",
    "map_e2r_blob_data.pkl",
    "map_r2c_blob_data.pkl",
)
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
EDGE_DEFAULTS: dict[str, Any] = {
    "description": "",
    "chunks": [],
    "type": "semantic",
    "weight": 0.0,
    "confidence": 0.0,
    "causal_score": 0.0,
    "uncertainty": 1.0,
    "association_score": 0.0,
    "transportability": 0.0,
    "status": "unverified",
    "context_condition": "",
    "intervention_count": 0,
    "alpha_posterior": 1.0,
    "beta_posterior": 1.0,
    "last_validated_episode": 0,
}


class AblationAuditError(RuntimeError):
    """Raised when an ablation invariant cannot be satisfied exactly."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _records_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(_json_key(record).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _workspace_file_hashes(path: Path) -> dict[str, str]:
    return {
        item.name: sha256_file(item)
        for item in sorted(path.iterdir(), key=lambda candidate: candidate.name)
        if item.is_file()
    }


def normalize_type(value: Any) -> str:
    edge_type = str(value or "semantic").strip().lower()
    return TYPE_ALIASES.get(edge_type, edge_type or "semantic")


def target_namespace(name: str) -> str:
    prefix = str(name or "").split("-", 1)[0]
    return prefix if len(prefix) >= 3 else ""


def edge_key(edge: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(edge["source"]), str(edge["target"]), str(edge["type"]))


def quota_key(edge: Mapping[str, Any]) -> tuple[str, str]:
    return (str(edge["source"]), str(edge["type"]))


def _edge_record_from_object(edge: Any) -> dict[str, Any]:
    record = {
        "source": str(edge.source),
        "target": str(edge.target),
    }
    for field in EDGE_ATTRIBUTE_FIELDS:
        record[field] = copy.deepcopy(getattr(edge, field, EDGE_DEFAULTS[field]))
    record["type"] = normalize_type(record["type"])
    return record


def _runtime_edge_records(graph: Any) -> list[dict[str, Any]]:
    names = list(graph.vs["name"])
    records: list[dict[str, Any]] = []
    for graph_edge in graph.es:
        source_index, target_index = graph_edge.tuple
        attrs = graph_edge.attributes()
        record = {
            "source": str(names[source_index]),
            "target": str(names[target_index]),
        }
        for field in EDGE_ATTRIBUTE_FIELDS:
            record[field] = copy.deepcopy(attrs.get(field, EDGE_DEFAULTS[field]))
        record["type"] = normalize_type(record["type"])
        records.append(record)
    return records


def _undirected_runtime_signature(edge: Mapping[str, Any]) -> str:
    source, target = sorted((str(edge["source"]), str(edge["target"])))
    payload = {"source": source, "target": target}
    for field in EDGE_ATTRIBUTE_FIELDS:
        payload[field] = edge.get(field, EDGE_DEFAULTS[field])
    return _json_key(payload)


def _undirected_runtime_multiset(edges: Iterable[Mapping[str, Any]]) -> Counter[str]:
    return Counter(_undirected_runtime_signature(edge) for edge in edges)


def _topology_multiset(edges: Iterable[Mapping[str, Any]], *, directed: bool) -> Counter[str]:
    values: Counter[str] = Counter()
    for edge in edges:
        source = str(edge["source"])
        target = str(edge["target"])
        if not directed:
            source, target = sorted((source, target))
        values[_json_key((source, target, str(edge["type"])))] += 1
    return values


def _package_multiset(edges: Iterable[Mapping[str, Any]]) -> Counter[str]:
    return Counter(
        _json_key({field: edge[field] for field in EVIDENCE_PACKAGE_FIELDS})
        for edge in edges
    )


def load_graph(path: str | Path) -> Any:
    with gzip.open(Path(path), "rb") as handle:
        return pickle.load(handle)


def load_phase1_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str = EXPECTED_CHECKPOINT_SHA256,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoint_path = Path(path)
    actual_sha256 = sha256_file(checkpoint_path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise AblationAuditError(
            f"Phase 1 checkpoint hash mismatch: {actual_sha256} != {expected_sha256}"
        )

    rows: list[dict[str, Any]] = []
    observed_field_sets: Counter[tuple[str, ...]] = Counter()
    with checkpoint_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            fields = frozenset(raw)
            observed_field_sets[tuple(sorted(fields))] += 1
            if fields != PHASE1_FIELDS:
                raise AblationAuditError(
                    f"Checkpoint row {line_number} fields differ from the Phase 1 schema: "
                    f"{sorted(fields)}"
                )
            if fields & PHASE2_FIELDS:
                raise AblationAuditError(
                    f"Checkpoint row {line_number} contains forbidden Phase 2 fields."
                )
            rows.append({field: copy.deepcopy(raw[field]) for field in PHASE1_FIELDS})

    return rows, {
        "path": str(checkpoint_path),
        "sha256": actual_sha256,
        "row_count": len(rows),
        "field_sets": {
            "|".join(fields): count for fields, count in sorted(observed_field_sets.items())
        },
        "phase2_field_count": 0,
    }


def load_a0_reference(source_workspace: str | Path) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    workspace = Path(source_workspace)
    graph_path = workspace / "graph_igraph_data.pklz"
    state_path = workspace / "caskg_state.json"
    graph = load_graph(graph_path)
    if graph.is_directed():
        raise AblationAuditError("The frozen A0 graph was expected to be an undirected igraph.")

    with state_path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)

    # A0 reconstruction is isolated from the C2 selector. C2 receives only the
    # resulting source-by-type quota table, never these Phase 2 edge records.
    from caskg.causal.graph_publisher import build_published_edges

    node_names = {str(name) for name in graph.vs["name"]}
    rebuilt_objects = build_published_edges(
        state["edges"],
        valid_node_names=node_names,
        include_unverified=False,
    )
    rebuilt_edges = [_edge_record_from_object(edge) for edge in rebuilt_objects]
    runtime_edges = _runtime_edge_records(graph)
    rebuilt_multiset = _undirected_runtime_multiset(rebuilt_edges)
    runtime_multiset = _undirected_runtime_multiset(runtime_edges)
    if rebuilt_multiset != runtime_multiset:
        raise AblationAuditError(
            "Current publisher does not reproduce the frozen A0 runtime edge multiset."
        )

    quotas = Counter(quota_key(edge) for edge in rebuilt_edges)
    type_counts = Counter(str(edge["type"]) for edge in rebuilt_edges)
    status_counts = Counter(str(edge["status"]) for edge in rebuilt_edges)
    audit = {
        "source_workspace": str(workspace),
        "source_graph_sha256": sha256_file(graph_path),
        "source_state_sha256": sha256_file(state_path),
        "graph_directed": bool(graph.is_directed()),
        "node_count": len(node_names),
        "edge_count": len(rebuilt_edges),
        "undirected_parallel_signature_count": len(runtime_multiset),
        "publisher_runtime_multiset_intersection": sum(
            (rebuilt_multiset & runtime_multiset).values()
        ),
        "publisher_runtime_multiset_exact": rebuilt_multiset == runtime_multiset,
        "quota_group_count": len(quotas),
        "type_counts": dict(sorted(type_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "node_names_sha256": hashlib.sha256(
            "\n".join(sorted(node_names)).encode("utf-8")
        ).hexdigest(),
    }
    return graph, rebuilt_edges, audit


def structural_quotas(a0_edges: Iterable[Mapping[str, Any]]) -> Counter[tuple[str, str]]:
    return Counter(quota_key(edge) for edge in a0_edges)


def _phase1_candidate_pool(
    checkpoint_rows: Iterable[Mapping[str, Any]],
    node_names: set[str],
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], int]:
    deduplicated: dict[tuple[str, str, str], dict[str, Any]] = {}
    duplicate_count = 0
    for raw in checkpoint_rows:
        if frozenset(raw) != PHASE1_FIELDS:
            raise AblationAuditError("C2 received a row outside the frozen Phase 1 schema.")
        if frozenset(raw) & PHASE2_FIELDS:
            raise AblationAuditError("C2 received forbidden Phase 2 fields.")
        source = str(raw["source"]).strip()
        target = str(raw["target"]).strip()
        edge_type = normalize_type(raw["type"])
        if not source or not target or source == target:
            continue
        if source not in node_names or target not in node_names:
            continue
        candidate = {
            "source": source,
            "target": target,
            "type": edge_type,
            "association_score": float(raw["association_score"]),
        }
        key = edge_key(candidate)
        current = deduplicated.get(key)
        if current is not None:
            duplicate_count += 1
            if (
                candidate["association_score"],
                candidate["target"],
            ) <= (
                current["association_score"],
                current["target"],
            ):
                continue
        deduplicated[key] = candidate

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in deduplicated.values():
        grouped[quota_key(candidate)].append(candidate)
    for candidates in grouped.values():
        candidates.sort(key=lambda edge: (-edge["association_score"], edge["target"]))
    return dict(grouped), duplicate_count


def _coverage_repair(
    selected: list[dict[str, Any]],
    grouped_candidates: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    quotas: Mapping[tuple[str, str], int],
    node_names: set[str],
) -> list[dict[str, Any]]:
    repairs: list[dict[str, Any]] = []

    while True:
        degree = Counter(
            endpoint
            for edge in selected
            for endpoint in (str(edge["source"]), str(edge["target"]))
        )
        missing_nodes = sorted(node_names - set(degree))
        if not missing_nodes:
            return repairs

        missing = missing_nodes[0]
        selected_keys = {edge_key(edge) for edge in selected}
        options: list[dict[str, Any]] = []
        for group, candidates in grouped_candidates.items():
            if quotas.get(group, 0) <= 0:
                continue
            for raw in candidates:
                candidate = dict(raw)
                if missing not in (candidate["source"], candidate["target"]):
                    continue
                if edge_key(candidate) in selected_keys:
                    continue
                options.append(candidate)
        options.sort(
            key=lambda edge: (
                -edge["association_score"],
                edge["source"],
                edge["target"],
                edge["type"],
            )
        )

        applied = False
        for candidate in options:
            group = quota_key(candidate)
            removable = sorted(
                (edge for edge in selected if quota_key(edge) == group),
                key=lambda edge: (edge["association_score"], edge["target"]),
            )
            for removed in removable:
                trial_degree = degree.copy()
                trial_degree[str(removed["source"])] -= 1
                trial_degree[str(removed["target"])] -= 1
                trial_degree[str(candidate["source"])] += 1
                trial_degree[str(candidate["target"])] += 1
                if any(trial_degree[node] <= 0 for node in node_names):
                    continue
                selected.remove(removed)
                selected.append(candidate)
                repairs.append(
                    {
                        "missing_node": missing,
                        "added": copy.deepcopy(candidate),
                        "removed": copy.deepcopy(removed),
                        "quota_group": {"source": group[0], "type": group[1]},
                    }
                )
                applied = True
                break
            if applied:
                break

        if not applied:
            raise AblationAuditError(
                f"C2 cannot repair coverage for {missing!r} without changing a quota."
            )


def _c2_runtime_edge(candidate: Mapping[str, Any]) -> dict[str, Any]:
    association_score = float(candidate["association_score"])
    record = {
        "source": str(candidate["source"]),
        "target": str(candidate["target"]),
        **copy.deepcopy(EDGE_DEFAULTS),
    }
    record.update(
        {
            # The core retriever treats "is" as an internal identity-edge
            # sentinel and excludes it from relations/rendered context.
            "description": "is",
            "chunks": [],
            "type": normalize_type(candidate["type"]),
            "weight": association_score,
            "confidence": association_score,
            "causal_score": 0.0,
            "uncertainty": 1.0,
            "association_score": association_score,
            "transportability": 0.0,
            "status": "unverified",
            "context_condition": "",
            "intervention_count": 0,
            "alpha_posterior": 1.0,
            "beta_posterior": 1.0,
            "last_validated_episode": 0,
        }
    )
    return record


def build_c2_edges(
    checkpoint_rows: Iterable[Mapping[str, Any]],
    quotas: Mapping[tuple[str, str], int],
    node_names: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select C2 edges using only Phase 1 rows and structural quota counts."""
    grouped, duplicate_count = _phase1_candidate_pool(checkpoint_rows, node_names)
    selected: list[dict[str, Any]] = []
    shortages: list[dict[str, Any]] = []
    for group, count in sorted(quotas.items()):
        candidates = grouped.get(group, [])
        if len(candidates) < count:
            shortages.append(
                {
                    "source": group[0],
                    "type": group[1],
                    "required": int(count),
                    "available": len(candidates),
                }
            )
            continue
        selected.extend(copy.deepcopy(candidates[:count]))
    if shortages:
        raise AblationAuditError(
            f"C2 structural quotas are short in {len(shortages)} source/type groups."
        )

    pre_repair_selected = copy.deepcopy(selected)
    pre_repair_covered = {
        endpoint
        for edge in pre_repair_selected
        for endpoint in (edge["source"], edge["target"])
    }
    repairs = _coverage_repair(selected, grouped, quotas, node_names)
    selected.sort(key=lambda edge: edge_key(edge))
    c2_edges = [_c2_runtime_edge(candidate) for candidate in selected]

    resulting_quotas = Counter(quota_key(edge) for edge in c2_edges)
    covered = {
        endpoint
        for edge in c2_edges
        for endpoint in (str(edge["source"]), str(edge["target"]))
    }
    if resulting_quotas != Counter(quotas):
        raise AblationAuditError("C2 source/type quotas changed during selection or repair.")
    if covered != node_names:
        raise AblationAuditError("C2 does not cover every frozen A0 node.")

    return c2_edges, {
        "selection_input": "phase1_checkpoint_rows_plus_structural_quotas_only",
        "phase2_fields_visible_to_selector": [],
        "quota_group_count": len(quotas),
        "quota_shortage_count": 0,
        "checkpoint_normalized_duplicate_count": duplicate_count,
        "edge_count": len(c2_edges),
        "coverage_before_repair": len(pre_repair_covered),
        "missing_nodes_before_repair": sorted(node_names - pre_repair_covered),
        "_pre_repair_edge_keys": [list(edge_key(edge)) for edge in pre_repair_selected],
        "coverage_after_repair": len(covered),
        "coverage_repairs": repairs,
        "type_counts": dict(sorted(Counter(edge["type"] for edge in c2_edges).items())),
        "status_counts": dict(sorted(Counter(edge["status"] for edge in c2_edges).items())),
    }


def _matching_tier(validated: Mapping[str, Any], scaffold: Mapping[str, Any]) -> int:
    same_source = validated["source"] == scaffold["source"]
    same_type = validated["type"] == scaffold["type"]
    same_target_namespace = target_namespace(str(validated["target"])) == target_namespace(
        str(scaffold["target"])
    )
    if same_source and same_type and same_target_namespace:
        return 1
    if same_source and same_type:
        return 2
    if same_source:
        return 3
    if same_type and same_target_namespace:
        return 4
    if same_type:
        return 5
    return 99


def _matching_jitter(seed: int, validated: Mapping[str, Any], scaffold: Mapping[str, Any]) -> float:
    payload = "|".join(
        (
            str(seed),
            str(validated["source"]),
            str(validated["target"]),
            str(validated["type"]),
            str(scaffold["source"]),
            str(scaffold["target"]),
            str(scaffold["type"]),
        )
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def match_c3_evidence_packages(
    a0_edges: Sequence[Mapping[str, Any]],
    *,
    seed: int = 7301,
) -> tuple[list[tuple[int, int, int]], dict[str, Any]]:
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise AblationAuditError(
            "C3 matching requires numpy and scipy in the ablation build environment."
        ) from exc

    validated_indices = sorted(
        (
            index
            for index, edge in enumerate(a0_edges)
            if str(edge["status"]) in VALIDATED_STATUSES
        ),
        key=lambda index: edge_key(a0_edges[index]),
    )
    scaffold_indices = sorted(
        (
            index
            for index, edge in enumerate(a0_edges)
            if str(edge["status"]) == SCAFFOLD_STATUS
        ),
        key=lambda index: edge_key(a0_edges[index]),
    )
    if len(validated_indices) != 285 or len(scaffold_indices) != 3007:
        raise AblationAuditError(
            "C3 expected 285 validated and 3007 scaffold packages, got "
            f"{len(validated_indices)} and {len(scaffold_indices)}."
        )

    tiers = np.empty((len(validated_indices), len(scaffold_indices)), dtype=np.int16)
    costs = np.empty((len(validated_indices), len(scaffold_indices)), dtype=np.float64)
    for row, validated_index in enumerate(validated_indices):
        validated = a0_edges[validated_index]
        for column, scaffold_index in enumerate(scaffold_indices):
            scaffold = a0_edges[scaffold_index]
            tier = _matching_tier(validated, scaffold)
            tiers[row, column] = tier
            costs[row, column] = float(tier) + (
                _matching_jitter(seed, validated, scaffold) * 1e-6
            )

    rows, columns = linear_sum_assignment(costs)
    matches: list[tuple[int, int, int]] = []
    for row, column in zip(rows.tolist(), columns.tolist()):
        tier = int(tiers[row, column])
        if tier > 5:
            raise AblationAuditError("C3 required a forbidden global matching fallback.")
        matches.append(
            (validated_indices[row], scaffold_indices[column], tier)
        )
    matches.sort(key=lambda item: edge_key(a0_edges[item[0]]))

    tier_counts = Counter(tier for _, _, tier in matches)
    if dict(sorted(tier_counts.items())) != EXPECTED_C3_TIER_COUNTS:
        raise AblationAuditError(
            f"C3 tier counts changed: {dict(sorted(tier_counts.items()))}"
        )
    if len({scaffold_index for _, scaffold_index, _ in matches}) != len(matches):
        raise AblationAuditError("C3 scaffold packages were not matched uniquely.")

    return matches, {
        "seed": seed,
        "optimizer": "scipy.optimize.linear_sum_assignment",
        "objective": "minimum_total_tier_cost",
        "validated_package_count": len(validated_indices),
        "scaffold_package_count": len(scaffold_indices),
        "matched_package_count": len(matches),
        "unique_scaffold_match_count": len(
            {scaffold_index for _, scaffold_index, _ in matches}
        ),
        "tier_counts": {str(key): value for key, value in sorted(tier_counts.items())},
        "global_fallback_count": 0,
    }


def build_c3_edges(
    a0_edges: Sequence[Mapping[str, Any]],
    *,
    seed: int = 7301,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    c3_edges = [copy.deepcopy(dict(edge)) for edge in a0_edges]
    original_edges = [copy.deepcopy(dict(edge)) for edge in a0_edges]
    matches, matching_audit = match_c3_evidence_packages(original_edges, seed=seed)
    matching_records: list[dict[str, Any]] = []

    for validated_index, scaffold_index, tier in matches:
        validated_original = original_edges[validated_index]
        scaffold_original = original_edges[scaffold_index]
        validated_package = {
            field: copy.deepcopy(validated_original[field])
            for field in EVIDENCE_PACKAGE_FIELDS
        }
        scaffold_package = {
            field: copy.deepcopy(scaffold_original[field])
            for field in EVIDENCE_PACKAGE_FIELDS
        }
        c3_edges[validated_index].update(scaffold_package)
        c3_edges[scaffold_index].update(validated_package)
        matching_records.append(
            {
                "tier": tier,
                "validated_edge": {
                    "source": validated_original["source"],
                    "target": validated_original["target"],
                    "type": validated_original["type"],
                },
                "scaffold_edge": {
                    "source": scaffold_original["source"],
                    "target": scaffold_original["target"],
                    "type": scaffold_original["type"],
                },
            }
        )

    for edge in c3_edges:
        edge["description"] = "is"
        edge["chunks"] = []

    if _topology_multiset(original_edges, directed=True) != _topology_multiset(
        c3_edges, directed=True
    ):
        raise AblationAuditError("C3 changed the directed edge topology.")
    if _package_multiset(original_edges) != _package_multiset(c3_edges):
        raise AblationAuditError("C3 changed the global evidence package multiset.")
    for original, shuffled in zip(original_edges, c3_edges):
        for field in ("source", "target", "type", "association_score"):
            if original[field] != shuffled[field]:
                raise AblationAuditError(f"C3 changed fixed field {field!r}.")

    displaced = sum(
        any(
            original_edges[index][field] != c3_edges[index][field]
            for field in EVIDENCE_PACKAGE_FIELDS
        )
        for index, edge in enumerate(original_edges)
        if str(edge["status"]) in VALIDATED_STATUSES
    )
    if displaced != 285:
        raise AblationAuditError(
            f"C3 displaced evidence from {displaced}/285 validated edges."
        )

    audit = {
        **matching_audit,
        "edge_count": len(c3_edges),
        "directed_topology_exact": True,
        "undirected_topology_exact": _topology_multiset(
            original_edges, directed=False
        )
        == _topology_multiset(c3_edges, directed=False),
        "evidence_package_multiset_exact": True,
        "association_score_fixed": True,
        "validated_evidence_displaced": displaced,
        "validated_evidence_displacement_rate": displaced / 285.0,
        "type_counts": dict(sorted(Counter(edge["type"] for edge in c3_edges).items())),
        "status_counts": dict(sorted(Counter(edge["status"] for edge in c3_edges).items())),
    }
    return c3_edges, matching_records, audit


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for record in records:
            handle.write(_json_key(record))
            handle.write("\n")


def _materialize_graph(source_graph: Any, edges: Sequence[Mapping[str, Any]], path: Path) -> None:
    graph = source_graph.copy()
    if graph.ecount():
        graph.delete_edges(range(graph.ecount()))
    name_to_index = {str(name): index for index, name in enumerate(graph.vs["name"])}
    graph.add_edges(
        [
            (name_to_index[str(edge["source"])], name_to_index[str(edge["target"])])
            for edge in edges
        ]
    )
    for field in EDGE_ATTRIBUTE_FIELDS:
        graph.es[field] = [
            copy.deepcopy(edge.get(field, EDGE_DEFAULTS[field])) for edge in edges
        ]
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_handle, mtime=0
        ) as compressed_handle:
            pickle.dump(graph, compressed_handle, protocol=pickle.HIGHEST_PROTOCOL)


def _copy_runtime_workspace(source_workspace: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for filename in RUNTIME_WORKSPACE_FILES:
        source = source_workspace / filename
        if not source.is_file():
            raise AblationAuditError(f"Required runtime asset is missing: {source}")
        if filename == "graph_igraph_data.pklz":
            continue
        shutil.copy2(source, destination / filename)


def _write_variant_directory(
    destination: Path,
    *,
    source_workspace: Path,
    source_graph: Any,
    edges: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
    matching_records: Sequence[Mapping[str, Any]] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        raise AblationAuditError(f"Refusing to overwrite existing graph view: {destination}")

    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-", dir=str(destination.parent)
    ) as temporary_root:
        temporary = Path(temporary_root) / destination.name
        temporary.mkdir()
        edges_path = temporary / "edges.jsonl"
        _write_jsonl(edges_path, edges)
        if matching_records is not None:
            _write_jsonl(temporary / "matching.jsonl", matching_records)

        workspace = temporary / "workspace"
        _copy_runtime_workspace(source_workspace, workspace)
        derived_graph_path = workspace / "graph_igraph_data.pklz"
        _materialize_graph(source_graph, edges, derived_graph_path)
        materialized_graph = load_graph(derived_graph_path)
        materialized_edges = _runtime_edge_records(materialized_graph)
        if _undirected_runtime_multiset(materialized_edges) != _undirected_runtime_multiset(
            edges
        ):
            raise AblationAuditError("Materialized workspace does not match its edge view.")

        final_audit = dict(audit)
        final_audit.update(
            {
                "edges_jsonl_sha256": sha256_file(edges_path),
                "edge_records_sha256": _records_sha256(edges),
                "workspace_graph_sha256": sha256_file(derived_graph_path),
                "workspace_runtime_files": list(RUNTIME_WORKSPACE_FILES),
                "workspace_contains_phase2_state": False,
                "workspace_contains_validation_log": False,
                "materialized_node_count": materialized_graph.vcount(),
                "materialized_edge_count": materialized_graph.ecount(),
                "materialized_graph_directed": bool(materialized_graph.is_directed()),
                "materialized_runtime_multiset_exact": True,
            }
        )
        if matching_records is not None:
            final_audit["matching_jsonl_sha256"] = sha256_file(
                temporary / "matching.jsonl"
            )
        _write_json(temporary / "audit.json", final_audit)

        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(temporary), str(destination))
    return final_audit


def _write_a0_reference(
    destination: Path,
    quotas: Mapping[tuple[str, str], int],
    audit: Mapping[str, Any],
    *,
    force: bool,
) -> dict[str, Any]:
    if destination.exists() and not force:
        raise AblationAuditError(f"Refusing to overwrite existing A0 audit: {destination}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    quota_records = [
        {"source": source, "type": edge_type, "count": int(count)}
        for (source, edge_type), count in sorted(quotas.items())
    ]
    quota_path = destination / "structural_quotas.json"
    _write_json(quota_path, quota_records)
    final_audit = dict(audit)
    final_audit.update(
        {
            "structural_quotas_sha256": sha256_file(quota_path),
            "structural_quota_total": sum(quotas.values()),
        }
    )
    _write_json(destination / "audit.json", final_audit)
    return final_audit


def build_all_graph_views(
    source_workspace: str | Path,
    output_root: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    source_workspace = Path(source_workspace).resolve()
    output_root = Path(output_root).resolve()
    source_hashes_before = _workspace_file_hashes(source_workspace)
    graph, a0_edges, a0_audit = load_a0_reference(source_workspace)
    node_names = {str(name) for name in graph.vs["name"]}
    quotas = structural_quotas(a0_edges)

    checkpoint_path = source_workspace / "candidate_checkpoint.jsonl"
    phase1_rows, checkpoint_audit = load_phase1_checkpoint(checkpoint_path)
    c2_edges, c2_audit = build_c2_edges(phase1_rows, quotas, node_names)
    c2_frozen_hash = _records_sha256(c2_edges)

    a0_keys = {edge_key(edge) for edge in a0_edges}
    c2_keys = {edge_key(edge) for edge in c2_edges}
    pre_repair_keys = {
        tuple(key) for key in c2_audit.pop("_pre_repair_edge_keys")
    }
    pre_repair_overlap = len(a0_keys & pre_repair_keys)
    post_repair_overlap = len(a0_keys & c2_keys)
    c2_audit.update(
        {
            "checkpoint": checkpoint_audit,
            "c2_frozen_before_posthoc_sha256": c2_frozen_hash,
            "pre_repair_a0_directed_edge_overlap": pre_repair_overlap,
            "pre_repair_a0_directed_edges_replaced": len(pre_repair_keys - a0_keys),
            "pre_repair_a0_directed_edge_jaccard": pre_repair_overlap
            / len(a0_keys | pre_repair_keys),
            "post_repair_a0_directed_edge_overlap": post_repair_overlap,
            "post_repair_a0_directed_edges_replaced": len(c2_keys - a0_keys),
            "post_repair_a0_directed_edge_jaccard": post_repair_overlap
            / len(a0_keys | c2_keys),
            "relation_text_exposed": False,
        }
    )

    # This rejected-edge count is computed only after the C2 view is frozen and
    # cannot influence selection, tie-breaking, or coverage repair.
    with (source_workspace / "caskg_state.json").open("r", encoding="utf-8") as handle:
        state_for_posthoc = json.load(handle)
    rejected_keys = {
        (
            str(edge["source"]),
            str(edge["target"]),
            normalize_type(edge["type"]),
        )
        for edge in state_for_posthoc["edges"]
        if str(edge.get("status")) == "rejected_non_causal"
    }
    c2_audit["pre_repair_posthoc_rejected_edge_count"] = len(
        pre_repair_keys & rejected_keys
    )
    c2_audit["post_repair_posthoc_rejected_edge_count"] = len(
        c2_keys & rejected_keys
    )
    c2_audit["posthoc_only_after_view_freeze"] = True
    if _records_sha256(c2_edges) != c2_frozen_hash:
        raise AblationAuditError("Post-hoc C2 audit changed the frozen C2 edge view.")

    c3_edges, matching_records, c3_audit = build_c3_edges(a0_edges, seed=7301)
    c3_audit["relation_text_exposed"] = False

    a0_structural_output = _write_a0_reference(
        output_root / "a0-reference", quotas, a0_audit, force=force
    )
    a0_control_edges = [copy.deepcopy(edge) for edge in a0_edges]
    for edge in a0_control_edges:
        edge["description"] = "is"
        edge["chunks"] = []
    a0_control_audit = {
        **a0_audit,
        "relation_text_exposed": False,
        "relation_text_control": "core_identity_edge_sentinel_is",
        "topology_and_ppr_weight_identity": True,
        "structural_reference_sha256": a0_structural_output[
            "structural_quotas_sha256"
        ],
    }
    a0_output = _write_variant_directory(
        output_root / "a0-full-reference",
        source_workspace=source_workspace,
        source_graph=graph,
        edges=a0_control_edges,
        audit=a0_control_audit,
        force=force,
    )
    c2_output = _write_variant_directory(
        output_root / "c2-no-counterfactual-matched",
        source_workspace=source_workspace,
        source_graph=graph,
        edges=c2_edges,
        audit=c2_audit,
        force=force,
    )
    c3_output = _write_variant_directory(
        output_root / "c3-shuffled-causal-evidence-s7301",
        source_workspace=source_workspace,
        source_graph=graph,
        edges=c3_edges,
        audit=c3_audit,
        matching_records=matching_records,
        force=force,
    )

    source_hashes_after = _workspace_file_hashes(source_workspace)
    if source_hashes_before != source_hashes_after:
        raise AblationAuditError("The source workspace changed during graph-view generation.")

    manifest = {
        "schema_version": 1,
        "protocol_id": "caskg-s1000-graph-ablation-v1",
        "source_workspace": str(source_workspace),
        "output_root": str(output_root),
        "source_workspace_modified": False,
        "source_workspace_hashes_before": source_hashes_before,
        "source_workspace_hashes_after": source_hashes_after,
        "a0": a0_output,
        "a0_structural_reference": a0_structural_output,
        "c2": c2_output,
        "c3": c3_output,
    }
    _write_json(output_root / "build_manifest.json", manifest)
    return manifest
