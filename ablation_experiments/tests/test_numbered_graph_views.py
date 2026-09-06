from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

from ablation_experiments.adapters.causal_graph_views import (
    EVIDENCE_PACKAGE_FIELDS,
    _package_multiset,
    _records_sha256,
    load_a0_reference,
    sha256_file,
)
from ablation_experiments.adapters.numbered_graph_views import (
    A1_VARIANT,
    A2_VARIANT,
    A3_VARIANT,
    A4_VARIANT,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_SOURCE_GRAPH_SHA256,
    PROTOCOL_ID,
    build_a2_rewired_edges,
    build_a3_phase1_weight_edges,
    build_a4_shuffled_evidence_edges,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    REPOSITORY_ROOT
    / "data"
    / "caskg_workspace"
    / "skills_1000_v32_scaffold_publish_gospath"
)
VIEW_ROOT = (
    REPOSITORY_ROOT
    / "ablation_experiments"
    / "graph_views"
    / "generated"
    / "a1-a4-main-parity-v1"
)


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="ascii").splitlines()
        if line.strip()
    ]


def _degree_by_type(edges: list[dict]) -> Counter[tuple[str, str]]:
    values: Counter[tuple[str, str]] = Counter()
    for edge in edges:
        values[(str(edge["source"]), str(edge["type"]))] += 1
        values[(str(edge["target"]), str(edge["type"]))] += 1
    return values


class NumberedGraphViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(
            (VIEW_ROOT / "asset_manifest.json").read_text(encoding="ascii")
        )
        cls.a0 = _read_jsonl(VIEW_ROOT / A1_VARIANT / "edges.jsonl")

    def test_source_was_not_modified(self) -> None:
        self.assertEqual(self.manifest["protocol_id"], PROTOCOL_ID)
        self.assertFalse(self.manifest["source_workspace_modified"])
        self.assertEqual(
            self.manifest["source_runtime_files_sha256_before"],
            self.manifest["source_runtime_files_sha256_after"],
        )
        self.assertEqual(
            sha256_file(SOURCE / "graph_igraph_data.pklz"),
            EXPECTED_SOURCE_GRAPH_SHA256,
        )
        self.assertEqual(
            sha256_file(SOURCE / "candidate_checkpoint.jsonl"),
            EXPECTED_CHECKPOINT_SHA256,
        )

    def test_a1_uses_the_exact_a0_graph(self) -> None:
        audit = self.manifest["variants"][A1_VARIANT]
        self.assertTrue(audit["graph_sha256_identical_to_source"])
        self.assertTrue(audit["ppr_disabled_by_runtime_adapter"])
        self.assertEqual(
            sha256_file(VIEW_ROOT / A1_VARIANT / "workspace" / "graph_igraph_data.pklz"),
            EXPECTED_SOURCE_GRAPH_SHA256,
        )

    def test_a2_preserves_marginals_degree_and_attributes(self) -> None:
        rewired = _read_jsonl(VIEW_ROOT / A2_VARIANT / "edges.jsonl")
        audit = self.manifest["variants"][A2_VARIANT]
        self.assertGreaterEqual(audit["endpoint_change_rate"], 0.95)
        self.assertTrue(audit["degree_by_type_exact"])
        self.assertTrue(audit["attribute_package_multiset_exact"])
        self.assertFalse(any(edge["source"] == edge["target"] for edge in rewired))
        self.assertEqual(_degree_by_type(self.a0), _degree_by_type(rewired))
        for edge_type in {edge["type"] for edge in self.a0}:
            original_type = [edge for edge in self.a0 if edge["type"] == edge_type]
            rewired_type = [edge for edge in rewired if edge["type"] == edge_type]
            self.assertEqual(
                Counter(edge["source"] for edge in original_type),
                Counter(edge["source"] for edge in rewired_type),
            )
            self.assertEqual(
                Counter(edge["target"] for edge in original_type),
                Counter(edge["target"] for edge in rewired_type),
            )

    def test_a3_changes_only_runtime_weight_and_confidence(self) -> None:
        phase1_weighted = _read_jsonl(VIEW_ROOT / A3_VARIANT / "edges.jsonl")
        audit = self.manifest["variants"][A3_VARIANT]
        self.assertEqual(audit["a0_edges_matched_to_phase1"], 3292)
        self.assertEqual(audit["only_changed_fields"], ["confidence", "weight"])
        for original, rewritten in zip(self.a0, phase1_weighted):
            self.assertEqual(rewritten["weight"], rewritten["association_score"])
            self.assertEqual(rewritten["confidence"], rewritten["association_score"])
            for field in original:
                if field not in {"weight", "confidence"}:
                    self.assertEqual(original[field], rewritten[field])

    def test_a4_moves_only_complete_evidence_packages(self) -> None:
        shuffled = _read_jsonl(VIEW_ROOT / A4_VARIANT / "edges.jsonl")
        audit = self.manifest["variants"][A4_VARIANT]
        self.assertEqual(audit["validated_evidence_displaced"], 285)
        self.assertTrue(audit["description_fixed_at_original_endpoint"])
        self.assertTrue(audit["chunks_fixed_at_original_endpoint"])
        self.assertEqual(_package_multiset(self.a0), _package_multiset(shuffled))
        for original, rewritten in zip(self.a0, shuffled):
            for field in original:
                if field not in EVIDENCE_PACKAGE_FIELDS:
                    self.assertEqual(original[field], rewritten[field])

    def test_numbered_builds_are_deterministic(self) -> None:
        _graph, source_edges, _audit = load_a0_reference(SOURCE)
        a2_edges, _ = build_a2_rewired_edges(source_edges, seed=7302)
        a3_edges, _ = build_a3_phase1_weight_edges(
            source_edges,
            SOURCE / "candidate_checkpoint.jsonl",
        )
        a4_edges, _matches, _ = build_a4_shuffled_evidence_edges(
            source_edges,
            seed=7301,
        )
        self.assertEqual(
            _records_sha256(a2_edges),
            self.manifest["variants"][A2_VARIANT]["edge_records_sha256"],
        )
        self.assertEqual(
            _records_sha256(a3_edges),
            self.manifest["variants"][A3_VARIANT]["edge_records_sha256"],
        )
        self.assertEqual(
            _records_sha256(a4_edges),
            self.manifest["variants"][A4_VARIANT]["edge_records_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
