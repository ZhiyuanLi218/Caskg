"""Materialize A0/C2 workspaces for the main-protocol counterfactual ablation.

A0 is a byte-identical runtime copy of the published CaSKG workspace. C2 keeps
the previously audited Phase-1 topology selection while restoring only the
Phase-1 candidate descriptions expected by the normal retrieval renderer.
The source workspace and the frozen one-shot ablation assets are never edited.
"""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from ablation_experiments.adapters.causal_graph_views import (
    EDGE_DEFAULTS,
    EXPECTED_CHECKPOINT_SHA256,
    PHASE1_FIELDS,
    PHASE2_FIELDS,
    RUNTIME_WORKSPACE_FILES,
    AblationAuditError,
    _materialize_graph,
    _runtime_edge_records,
    _topology_multiset,
    load_graph,
    normalize_type,
    sha256_file,
)


PROTOCOL_ID = "caskg-s1000-c2-main-parity-v1"
A0_VARIANT = "a0-full-caskg-original"
C2_VARIANT = "c2-no-counterfactual-main-parity"


def _json_line(record: Mapping[str, Any]) -> str:
    return json.dumps(
        record,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
        newline="\n",
    )


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for record in records:
            handle.write(_json_line(record))
            handle.write("\n")


def _runtime_hashes(workspace: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for filename in RUNTIME_WORKSPACE_FILES:
        path = workspace / filename
        if not path.is_file():
            raise AblationAuditError(f"Missing runtime workspace file: {path}")
        hashes[filename] = sha256_file(path)
    return hashes


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AblationAuditError(
                    f"Invalid JSONL record at {path}:{line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise AblationAuditError(
                    f"Expected a JSON object at {path}:{line_number}"
                )
            records.append(record)
    return records


def _candidate_key(record: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(record["source"]),
        str(record["target"]),
        normalize_type(record["type"]),
    )


def _load_phase1_candidates(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    if sha256_file(path) != EXPECTED_CHECKPOINT_SHA256:
        raise AblationAuditError("Phase-1 candidate checkpoint checksum mismatch.")

    candidates: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in _load_jsonl(path):
        fields = frozenset(record)
        if fields != PHASE1_FIELDS or fields & PHASE2_FIELDS:
            raise AblationAuditError(
                "The main-parity C2 builder received non-Phase-1 candidate data."
            )
        key = _candidate_key(record)
        if key in candidates:
            raise AblationAuditError(f"Duplicate normalized Phase-1 edge: {key}")
        candidates[key] = record
    return candidates


def _restore_phase1_evidence(
    selected_edges: list[dict[str, Any]],
    candidates: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    restored: list[dict[str, Any]] = []
    for selected in selected_edges:
        key = _candidate_key(selected)
        candidate = candidates.get(key)
        if candidate is None:
            raise AblationAuditError(
                f"Selected C2 edge is absent from the Phase-1 checkpoint: {key}"
            )
        association = float(candidate["association_score"])
        if abs(float(selected["association_score"]) - association) > 1e-12:
            raise AblationAuditError(f"C2 association score drifted for edge: {key}")

        edge = copy.deepcopy(EDGE_DEFAULTS)
        edge.update(
            {
                "source": key[0],
                "target": key[1],
                "type": key[2],
                "association_score": association,
                "weight": association,
                "confidence": association,
                "description": str(candidate["description"]),
                "chunks": [],
                "status": "unverified",
                "causal_score": 0.0,
                "uncertainty": 1.0,
                "alpha_posterior": 1.0,
                "beta_posterior": 1.0,
                "intervention_count": 0,
                "transportability": 0.0,
                "last_validated_episode": 0,
            }
        )
        restored.append(edge)
    return restored


def _copy_runtime_files(source: Path, destination: Path, *, include_graph: bool) -> None:
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
        raise AblationAuditError(f"Refusing to overwrite graph view: {destination}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staged), str(destination))


def build_main_parity_views(
    source_workspace: str | Path,
    frozen_c2_view: str | Path,
    output_root: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    source_workspace = Path(source_workspace).resolve()
    frozen_c2_view = Path(frozen_c2_view).resolve()
    output_root = Path(output_root).resolve()

    source_hashes_before = _runtime_hashes(source_workspace)
    source_graph = load_graph(source_workspace / "graph_igraph_data.pklz")
    source_edges = _runtime_edge_records(source_graph)
    selected_edges = _load_jsonl(frozen_c2_view / "edges.jsonl")
    candidates = _load_phase1_candidates(source_workspace / "candidate_checkpoint.jsonl")
    c2_edges = _restore_phase1_evidence(selected_edges, candidates)

    if len(source_edges) != 3292 or len(c2_edges) != 3292:
        raise AblationAuditError("Main-parity A0/C2 must each contain exactly 3292 edges.")
    if source_graph.vcount() != 1000:
        raise AblationAuditError("Main-parity A0 must contain exactly 1000 nodes.")
    if _topology_multiset(c2_edges, directed=True) != _topology_multiset(
        selected_edges, directed=True
    ):
        raise AblationAuditError("Restoring Phase-1 descriptions changed C2 topology.")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_root.name}-", dir=str(output_root.parent)
    ) as temporary_root:
        temporary = Path(temporary_root) / output_root.name
        temporary.mkdir()

        a0_root = temporary / A0_VARIANT
        a0_workspace = a0_root / "workspace"
        a0_root.mkdir()
        _copy_runtime_files(source_workspace, a0_workspace, include_graph=True)
        a0_hashes = _runtime_hashes(a0_workspace)
        if a0_hashes != source_hashes_before:
            raise AblationAuditError("A0 runtime copy is not byte-identical to the source.")
        a0_audit = {
            "protocol_id": PROTOCOL_ID,
            "variant_id": A0_VARIANT,
            "source_workspace": str(source_workspace),
            "runtime_files_sha256": a0_hashes,
            "node_count": source_graph.vcount(),
            "edge_count": source_graph.ecount(),
            "graph_sha256_identical_to_source": True,
            "evaluator_or_retrieval_code_modified": False,
            "dynamic_retrieval_enabled": True,
        }
        _write_json(a0_root / "audit.json", a0_audit)

        c2_root = temporary / C2_VARIANT
        c2_workspace = c2_root / "workspace"
        c2_root.mkdir()
        _copy_runtime_files(source_workspace, c2_workspace, include_graph=False)
        _materialize_graph(
            source_graph,
            c2_edges,
            c2_workspace / "graph_igraph_data.pklz",
        )
        materialized = load_graph(c2_workspace / "graph_igraph_data.pklz")
        materialized_edges = _runtime_edge_records(materialized)
        if _topology_multiset(materialized_edges, directed=False) != _topology_multiset(
            c2_edges, directed=False
        ):
            raise AblationAuditError("Materialized C2 topology differs from selected C2.")
        if any(str(edge["status"]) != "unverified" for edge in materialized_edges):
            raise AblationAuditError("C2 materialization exposed a Phase-2 status.")
        if any(float(edge["causal_score"]) != 0.0 for edge in materialized_edges):
            raise AblationAuditError("C2 materialization exposed a causal score.")

        _write_jsonl(c2_root / "edges.jsonl", c2_edges)
        c2_hashes = _runtime_hashes(c2_workspace)
        c2_audit = {
            "protocol_id": PROTOCOL_ID,
            "variant_id": C2_VARIANT,
            "source_checkpoint": str(source_workspace / "candidate_checkpoint.jsonl"),
            "source_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "selection_source": str(frozen_c2_view / "edges.jsonl"),
            "selection_source_sha256": sha256_file(frozen_c2_view / "edges.jsonl"),
            "runtime_files_sha256": c2_hashes,
            "node_count": materialized.vcount(),
            "edge_count": materialized.ecount(),
            "type_counts": dict(
                sorted(Counter(str(edge["type"]) for edge in materialized_edges).items())
            ),
            "status_counts": {"unverified": len(materialized_edges)},
            "phase2_fields_visible_to_selector": [],
            "causal_score_zero_for_all_edges": True,
            "phase1_descriptions_restored": True,
            "dynamic_retrieval_enabled": True,
        }
        _write_json(c2_root / "audit.json", c2_audit)

        manifest = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "source_runtime_files_sha256": source_hashes_before,
            "variants": {
                A0_VARIANT: a0_audit,
                C2_VARIANT: c2_audit,
            },
        }
        _write_json(temporary / "asset_manifest.json", manifest)
        _atomic_replace(output_root, temporary, force=force)

    if _runtime_hashes(source_workspace) != source_hashes_before:
        raise AblationAuditError("Source workspace changed while building graph views.")
    return json.loads((output_root / "asset_manifest.json").read_text(encoding="ascii"))
