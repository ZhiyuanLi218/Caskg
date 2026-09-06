from __future__ import annotations

import unittest

from ablation_experiments.adapters.causal_graph_views import (
    AblationAuditError,
    PHASE1_FIELDS,
    _matching_tier,
    build_c2_edges,
    normalize_type,
)


def phase1_edge(source: str, target: str, score: float) -> dict[str, object]:
    value: dict[str, object] = {
        "source": source,
        "target": target,
        "type": "similar",
        "association_score": score,
        "weight": score,
        "confidence": score,
        "description": "Phase 1 association",
    }
    assert frozenset(value) == PHASE1_FIELDS
    return value


class CausalGraphViewTests(unittest.TestCase):
    def test_type_normalization_matches_publisher_families(self) -> None:
        self.assertEqual(normalize_type("prereq"), "dependency")
        self.assertEqual(normalize_type("enhance"), "workflow")
        self.assertEqual(normalize_type("similar"), "semantic")
        self.assertEqual(normalize_type("alternative"), "alternative")

    def test_c2_preserves_quota_while_repairing_node_coverage(self) -> None:
        rows = [
            phase1_edge("a-skill", "b-skill", 0.9),
            phase1_edge("a-skill", "c-skill", 0.5),
            phase1_edge("b-skill", "a-skill", 0.8),
        ]
        quotas = {("a-skill", "semantic"): 1, ("b-skill", "semantic"): 1}
        edges, audit = build_c2_edges(rows, quotas, {"a-skill", "b-skill", "c-skill"})

        self.assertEqual(len(edges), 2)
        self.assertEqual(audit["coverage_before_repair"], 2)
        self.assertEqual(audit["coverage_after_repair"], 3)
        self.assertEqual(len(audit["coverage_repairs"]), 1)
        self.assertEqual({edge["status"] for edge in edges}, {"unverified"})
        self.assertTrue(all(edge["weight"] == edge["association_score"] for edge in edges))

    def test_c2_rejects_rows_with_phase2_fields(self) -> None:
        invalid = phase1_edge("a-skill", "b-skill", 0.9)
        invalid["status"] = "confirmed_causal"
        with self.assertRaises(AblationAuditError):
            build_c2_edges(
                [invalid],
                {("a-skill", "semantic"): 1},
                {"a-skill", "b-skill"},
            )

    def test_c3_tier_order(self) -> None:
        validated = {
            "source": "alfworld-a",
            "target": "scienceworld-b",
            "type": "workflow",
        }
        self.assertEqual(
            _matching_tier(
                validated,
                {
                    "source": "alfworld-a",
                    "target": "scienceworld-c",
                    "type": "workflow",
                },
            ),
            1,
        )
        self.assertEqual(
            _matching_tier(
                validated,
                {
                    "source": "alfworld-a",
                    "target": "other-c",
                    "type": "workflow",
                },
            ),
            2,
        )
        self.assertEqual(
            _matching_tier(
                validated,
                {
                    "source": "alfworld-a",
                    "target": "other-c",
                    "type": "semantic",
                },
            ),
            3,
        )
        self.assertEqual(
            _matching_tier(
                validated,
                {
                    "source": "other-a",
                    "target": "scienceworld-c",
                    "type": "workflow",
                },
            ),
            4,
        )
        self.assertEqual(
            _matching_tier(
                validated,
                {
                    "source": "other-a",
                    "target": "other-c",
                    "type": "workflow",
                },
            ),
            5,
        )


if __name__ == "__main__":
    unittest.main()
