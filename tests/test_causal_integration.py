"""Integration tests for CaSKG (Counterfactual-Causal Skill Graph) modules."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

import pytest


class TestTraceStore:
    def setup_method(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store_path = str(Path(self.tmp.name) / "traces.jsonl")

    def teardown_method(self):
        self.tmp.cleanup()

    def test_record_and_retrieve(self):
        from caskg.causal.trace_store import ExecutionTrace, TraceStore

        store = TraceStore(self.store_path)
        trace = ExecutionTrace(
            trace_id="t1",
            task_id="task1",
            environment="alfworld",
            task_type="pick",
            skills_used=["skill_a", "skill_b"],
            skills_available=["skill_a", "skill_b", "skill_c"],
            outcome=1.0,
            steps=5,
            timestamp="2026-01-01T00:00:00",
            is_intervention=False,
            intervention_edge=None,
            intervention_type=None,
        )
        store.record(trace)
        recent = store.recent(10)
        assert len(recent) == 1
        assert recent[0].task_id == "task1"
        assert recent[0].outcome == 1.0

    def test_empty_store(self):
        from caskg.causal.trace_store import TraceStore

        store = TraceStore(self.store_path)
        assert store.recent(10) == []

    def test_cooccurrence_matrix(self):
        from caskg.causal.trace_store import ExecutionTrace, TraceStore

        store = TraceStore(self.store_path)
        for i in range(5):
            store.record(
                ExecutionTrace(
                    trace_id=f"t{i}",
                    task_id=f"task{i}",
                    environment="alfworld",
                    task_type="pick",
                    skills_used=["A", "B"],
                    skills_available=["A", "B", "C"],
                    outcome=1.0,
                    steps=3,
                    timestamp=f"2026-01-0{i + 1}",
                    is_intervention=False,
                    intervention_edge=None,
                    intervention_type=None,
                )
            )
        matrix = store.cooccurrence_matrix(["A", "B", "C"])
        assert ("A", "B") in matrix or ("B", "A") in matrix

    def test_temporal_precedence(self):
        from caskg.causal.trace_store import ExecutionTrace, TraceStore

        store = TraceStore(self.store_path)
        for i in range(10):
            store.record(
                ExecutionTrace(
                    trace_id=f"t{i}",
                    task_id=f"task{i}",
                    environment="alfworld",
                    task_type="pick",
                    skills_used=["A", "B"],
                    skills_available=["A", "B"],
                    outcome=1.0,
                    steps=3,
                    timestamp=f"2026-01-{i + 1:02d}",
                    is_intervention=False,
                    intervention_edge=None,
                    intervention_type=None,
                )
            )
        precedence = store.temporal_precedence("A", "B")
        assert precedence >= 0.5

    def test_by_edge_filter(self):
        from caskg.causal.trace_store import ExecutionTrace, TraceStore

        store = TraceStore(self.store_path)
        store.record(
            ExecutionTrace(
                trace_id="t1",
                task_id="task1",
                environment="alfworld",
                task_type="pick",
                skills_used=["A"],
                skills_available=["A", "B"],
                outcome=1.0,
                steps=3,
                timestamp="2026-01-01",
                is_intervention=True,
                intervention_edge=("A", "B"),
                intervention_type="removal",
            )
        )
        results = store.by_edge("A", "B")
        assert len(results) == 1
        assert results[0].is_intervention is True


class TestBayesianEstimator:
    def test_initial_state(self):
        from caskg.causal.interventions import BayesianEdgeEstimator

        est = BayesianEdgeEstimator(1.0, 1.0)
        assert abs(est.mean - 0.5) < 0.01

    def test_convergence_to_true_rate(self):
        from caskg.causal.interventions import BayesianEdgeEstimator

        est = BayesianEdgeEstimator(1.0, 1.0)
        for _ in range(80):
            est.update(True)
        for _ in range(20):
            est.update(False)
        assert abs(est.mean - 0.8) < 0.05

    def test_variance_decreases_with_data(self):
        from caskg.causal.interventions import BayesianEdgeEstimator

        est = BayesianEdgeEstimator(1.0, 1.0)
        initial_var = est.variance
        for _ in range(50):
            est.update(True)
        assert est.variance < initial_var

    def test_confidence_thresholds(self):
        from caskg.causal.interventions import BayesianEdgeEstimator

        est = BayesianEdgeEstimator(1.0, 1.0)
        for _ in range(50):
            est.update(True)
        assert est.confident_causal(0.6)
        assert not est.confident_non_causal(0.2)


class TestCausalSchemaExtensions:
    def test_skill_edge_causal_fields_defaults(self):
        from caskg.core.schema import SkillEdge

        edge = SkillEdge(source="A", target="B")
        assert edge.causal_score == 0.0
        assert edge.uncertainty == 1.0
        assert edge.status == "unverified"
        assert edge.alpha_posterior == 1.0
        assert edge.beta_posterior == 1.0
        assert edge.transportability == 0.0
        assert edge.intervention_count == 0

    def test_skill_node_extended_fields(self):
        from caskg.core.schema import SkillNode

        node = SkillNode(name="test", preconditions="cond1\ncond2", effects="eff1")
        assert "cond1" in node.preconditions_list
        assert "cond2" in node.preconditions_list
        assert "eff1" in node.effects_list

    def test_causal_edge_status_values(self):
        from caskg.core.schema import CausalEdgeStatus

        assert CausalEdgeStatus.CONFIRMED_CAUSAL == "confirmed_causal"
        assert CausalEdgeStatus.UNVERIFIED == "unverified"
        assert CausalEdgeStatus.REJECTED_NON_CAUSAL == "rejected_non_causal"
        assert CausalEdgeStatus.PRUNED == "pruned"

    def test_edge_to_attrs_includes_causal_fields(self):
        from caskg.core.schema import SkillEdge

        edge = SkillEdge(
            source="A",
            target="B",
            causal_score=0.8,
            uncertainty=0.1,
            status="confirmed_causal",
        )
        attrs = SkillEdge.to_attrs(edge=edge)
        assert attrs["causal_score"] == 0.8
        assert attrs["uncertainty"] == 0.1
        assert attrs["status"] == "confirmed_causal"


class TestSignals:
    def test_lexical_signal(self):
        from caskg.causal.signals import LexicalSignal
        from caskg.core.schema import SkillNode

        node_a = SkillNode(name="mesh-parser", description="Parse 3D mesh files")
        node_b = SkillNode(name="mesh-exporter", description="Export 3D mesh to OBJ")
        signal = LexicalSignal()
        score = signal.compute(node_a, node_b)
        assert 0.0 <= score <= 1.0
        assert score > 0.0

    def test_io_compatibility_signal(self):
        from caskg.causal.signals import IOCompatibilitySignal
        from caskg.core.schema import SkillNode

        node_a = SkillNode(name="parser", inputs="raw_text", outputs="parsed_json")
        node_b = SkillNode(name="processor", inputs="parsed_json", outputs="result")
        signal = IOCompatibilitySignal()
        score = signal.compute(node_a, node_b)
        assert score > 0.3

    def test_no_overlap_gives_zero(self):
        from caskg.causal.signals import LexicalSignal
        from caskg.core.schema import SkillNode

        node_a = SkillNode(name="astronomy-calc", description="Calculate star positions")
        node_b = SkillNode(name="cooking-timer", description="Set kitchen timers")
        signal = LexicalSignal()
        score = signal.compute(node_a, node_b)
        assert score < 0.3

    def test_association_score_weighted(self):
        from caskg.causal.signals import compute_association_score

        signals = {"sem": 0.8, "lex": 0.3, "io": 0.9}
        weights = {"sem": 0.25, "lex": 0.10, "io": 0.25}
        score = compute_association_score(signals, weights)
        assert 0.0 <= score <= 1.0


class TestScheduler:
    def _make_scheduler(self):
        import tempfile

        from caskg.causal.scheduler import ActiveScheduler
        from caskg.causal.trace_store import TraceStore

        class FakeConfig:
            P_EXPLORE_INITIAL = 0.3
            P_EXPLORE_MIN = 0.05
            MAX_INTERVENTION_BUDGET = 100
            SCHEDULER_ALPHA_USAGE = 0.25
            SCHEDULER_BETA_CENTRALITY = 0.20
            SCHEDULER_GAMMA_UNCERTAINTY = 0.30
            SCHEDULER_DELTA_FAILURE = 0.15
            SCHEDULER_EPSILON_MAINTENANCE = 0.10

        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        store = TraceStore(tmp.name)
        return ActiveScheduler(FakeConfig(), store)

    def test_initial_exploration_rate(self):
        scheduler = self._make_scheduler()
        assert scheduler.exploration_rate == 0.3

    def test_exploration_rate_decay(self):
        scheduler = self._make_scheduler()
        scheduler.update_exploration_rate(0.1)
        assert scheduler.exploration_rate < 0.1
        assert scheduler.exploration_rate >= 0.05

    def test_budget_tracking(self):
        scheduler = self._make_scheduler()
        assert scheduler.budget_remaining == 100
        scheduler.record_probe_used()
        assert scheduler.budget_remaining == 99


class TestGroupIntervention:
    def test_plan_groups(self):
        from caskg.causal.scheduler import GroupInterventionPlanner

        planner = GroupInterventionPlanner()
        groups = planner.plan_groups("target", ["A", "B", "C", "D"], max_group_size=2)
        assert len(groups) == 2
        assert groups[0] == ["A", "B"]
        assert groups[1] == ["C", "D"]

    def test_resolve_non_significant(self):
        from caskg.causal.scheduler import GroupInterventionPlanner

        planner = GroupInterventionPlanner()
        result = planner.resolve_group_result(["A", "B"], "target", significant=False)
        assert len(result) == 2
        assert all(not causal for _, _, causal in result)

    def test_resolve_single_significant(self):
        from caskg.causal.scheduler import GroupInterventionPlanner

        planner = GroupInterventionPlanner()
        result = planner.resolve_group_result(["A"], "target", significant=True)
        assert result == [("A", "target", True)]


class TestTransitivity:
    def test_forward_inference(self):
        from caskg.causal.transitivity import TransitivityPruner
        from caskg.core.schema import SkillEdge, SkillNode

        nodes = [SkillNode(name=n) for n in ["A", "B", "C"]]
        edges = [
            SkillEdge(source="A", target="B", status="confirmed_causal", causal_score=0.8),
            SkillEdge(source="B", target="C", status="confirmed_causal", causal_score=0.9),
        ]
        pruner = TransitivityPruner(nodes, edges)
        inferred = pruner.forward_inference()
        assert any(s == "A" and t == "C" for s, t, _ in inferred)

    def test_has_directed_path(self):
        from caskg.causal.transitivity import TransitivityPruner
        from caskg.core.schema import SkillEdge, SkillNode

        nodes = [SkillNode(name=n) for n in ["A", "B", "C", "D"]]
        edges = [
            SkillEdge(source="A", target="B", status="confirmed_causal", causal_score=0.8),
            SkillEdge(source="B", target="C", status="confirmed_causal", causal_score=0.9),
        ]
        pruner = TransitivityPruner(nodes, edges)
        assert pruner.has_directed_path("A", "C")
        assert not pruner.has_directed_path("A", "D")

    def test_reduce_queue(self):
        from caskg.causal.transitivity import TransitivityPruner
        from caskg.core.schema import SkillEdge, SkillNode

        nodes = [SkillNode(name=n) for n in ["A", "B", "C"]]
        confirmed = [
            SkillEdge(source="A", target="B", status="confirmed_causal", causal_score=0.8),
            SkillEdge(source="B", target="C", status="confirmed_causal", causal_score=0.9),
        ]
        queue = [SkillEdge(source="A", target="C", status="unverified")]
        pruner = TransitivityPruner(nodes, confirmed)
        reduced = pruner.reduce_queue(queue)
        assert len(reduced) < len(queue)


class TestCausalRetrieval:
    def test_causal_scorer(self):
        from caskg.causal.retrieval import CausalScorer
        from caskg.core.schema import SkillEdge, SkillNode

        class FakeConfig:
            CAUSAL_RETRIEVAL_LAMBDA_REL = 0.3
            CAUSAL_RETRIEVAL_LAMBDA_STRUCT = 0.25
            CAUSAL_RETRIEVAL_LAMBDA_CAUSALNEC = 0.35
            CAUSAL_RETRIEVAL_LAMBDA_UNC = 0.1

        scorer = CausalScorer(FakeConfig())
        node = SkillNode(name="test-skill")
        edges = [
            SkillEdge(
                source="prereq",
                target="test-skill",
                status="confirmed_causal",
                causal_score=0.8,
                uncertainty=0.1,
            )
        ]
        score = scorer.score(node, relevance=0.7, structural=0.5, causal_edges=edges)
        assert score > 0.0

    def test_backward_traversal(self):
        from caskg.causal.retrieval import CausalBackwardTraversal
        from caskg.core.schema import SkillEdge

        edges = [
            SkillEdge(
                source="A", target="B", status="confirmed_causal",
                causal_score=0.9, transportability=0.8,
            ),
            SkillEdge(
                source="B", target="C", status="confirmed_causal",
                causal_score=0.7, transportability=0.8,
            ),
        ]
        traversal = CausalBackwardTraversal(edges)
        prereqs = traversal.traverse(["C"], max_depth=3)
        assert "B" in prereqs
        assert "A" in prereqs
        assert prereqs["B"] > prereqs["A"]

    def test_path_planner(self):
        from caskg.causal.retrieval import CausalPathPlanner
        from caskg.core.schema import SkillEdge, SkillNode

        nodes = [SkillNode(name=n) for n in ["setup", "process", "finalize"]]
        edges = [
            SkillEdge(source="setup", target="process", status="confirmed_causal", causal_score=0.9),
            SkillEdge(source="process", target="finalize", status="confirmed_causal", causal_score=0.8),
        ]
        planner = CausalPathPlanner(nodes, edges)
        path = planner.plan_execution_path(nodes)
        assert path.index("setup") < path.index("process")
        assert path.index("process") < path.index("finalize")


class TestMaintenanceEvolution:
    def test_evolution_cycle(self):
        import tempfile

        from caskg.causal.maintenance import EvolutionCycle
        from caskg.causal.trace_store import TraceStore
        from caskg.core.schema import SkillEdge, SkillNode

        class FakeConfig:
            STABLE_REINFORCEMENT_FACTOR = 1.05
            PRUNE_AGE_THRESHOLD = 100
            CAUSAL_CONFIRM_THRESHOLD = 0.6

        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        store = TraceStore(tmp.name)
        cycle = EvolutionCycle(FakeConfig(), store)

        nodes = [SkillNode(name="A"), SkillNode(name="B")]
        edges = [
            SkillEdge(
                source="A",
                target="B",
                status="confirmed_causal",
                causal_score=0.8,
                confidence=0.7,
            )
        ]

        updated_edges, report, unverified_ratio = cycle.execute(edges, nodes, episode=50)
        assert len(updated_edges) >= 1
        assert updated_edges[0].confidence >= 0.7
        assert 0.0 <= unverified_ratio <= 1.0


class TestValidatorCompositeScore:
    def test_composite_score_prereq_weights(self):
        """Prereq edges should weight removal and reordering heavily."""
        from caskg.causal.validator import CounterfactualValidator
        from caskg.causal.interventions import InterventionEngine
        from caskg.causal.trace_store import TraceStore
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        store = TraceStore(tmp.name)

        class FakeConfig:
            CAUSAL_CONFIRM_THRESHOLD = 0.6
            CAUSAL_REJECT_THRESHOLD = 0.2

        engine = InterventionEngine(store, FakeConfig())
        validator = CounterfactualValidator(engine, FakeConfig())

        # Prereq: alpha=0.5, beta=0.1, gamma=0.4
        deltas = {"removal": 0.8, "substitution": 0.5, "reordering": 0.7}
        score = validator.compute_composite_score(deltas, "prereq")
        expected = 0.5 * 0.8 + 0.1 * 0.5 + 0.4 * 0.7  # 0.4 + 0.05 + 0.28 = 0.73
        assert abs(score - expected) < 0.01

    def test_composite_score_similar_weights(self):
        """Similar edges should weight substitution heavily."""
        from caskg.causal.validator import CounterfactualValidator
        from caskg.causal.interventions import InterventionEngine
        from caskg.causal.trace_store import TraceStore
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        store = TraceStore(tmp.name)

        class FakeConfig:
            CAUSAL_CONFIRM_THRESHOLD = 0.6
            CAUSAL_REJECT_THRESHOLD = 0.2

        engine = InterventionEngine(store, FakeConfig())
        validator = CounterfactualValidator(engine, FakeConfig())

        # Similar: alpha=0.2, beta=0.7, gamma=0.1
        deltas = {"removal": 0.8, "substitution": 0.9, "reordering": 0.3}
        score = validator.compute_composite_score(deltas, "similar")
        expected = 0.2 * 0.8 + 0.7 * 0.9 + 0.1 * 0.3  # 0.16 + 0.63 + 0.03 = 0.82
        assert abs(score - expected) < 0.01


class TestRepairSignalTemporal:
    def test_detects_failure_then_recovery(self):
        """RepairSignal should detect temporal: target failed without source → later succeeded with source."""
        from caskg.causal.signals import RepairSignal
        from caskg.causal.trace_store import ExecutionTrace, TraceStore
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        store = TraceStore(tmp.name)

        # Trace 1: target "B" fails without source "A"
        store.record(ExecutionTrace(
            trace_id="t1", task_id="task1", environment="alfworld",
            task_type="clean", skills_used=["B"],
            skills_available=["A", "B"], outcome=0.2, steps=10,
            timestamp="2026-01-01T01:00:00",
            is_intervention=False, intervention_edge=None, intervention_type=None,
        ))
        # Trace 2: same task_type, target "B" succeeds WITH source "A"
        store.record(ExecutionTrace(
            trace_id="t2", task_id="task2", environment="alfworld",
            task_type="clean", skills_used=["A", "B"],
            skills_available=["A", "B"], outcome=0.9, steps=5,
            timestamp="2026-01-01T02:00:00",
            is_intervention=False, intervention_edge=None, intervention_type=None,
        ))

        signal = RepairSignal()
        score = signal.compute("A", "B", store)
        assert score > 0.0  # Should detect the recovery pattern

    def test_no_signal_without_temporal_order(self):
        """If recovery happens BEFORE failure, no repair signal should be detected."""
        from caskg.causal.signals import RepairSignal
        from caskg.causal.trace_store import ExecutionTrace, TraceStore
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        store = TraceStore(tmp.name)

        # Recovery trace comes FIRST (timestamp earlier)
        store.record(ExecutionTrace(
            trace_id="t1", task_id="task1", environment="alfworld",
            task_type="clean", skills_used=["A", "B"],
            skills_available=["A", "B"], outcome=0.9, steps=5,
            timestamp="2026-01-01T01:00:00",
            is_intervention=False, intervention_edge=None, intervention_type=None,
        ))
        # Failure trace comes AFTER
        store.record(ExecutionTrace(
            trace_id="t2", task_id="task2", environment="alfworld",
            task_type="clean", skills_used=["B"],
            skills_available=["A", "B"], outcome=0.2, steps=10,
            timestamp="2026-01-01T02:00:00",
            is_intervention=False, intervention_edge=None, intervention_type=None,
        ))

        signal = RepairSignal()
        score = signal.compute("A", "B", store)
        # The failure has no LATER recovery, so score should be 0
        assert score == 0.0


class TestInformativePrior:
    def test_high_association_creates_informative_prior(self):
        """Edges with high association_score should get a prior leaning toward causal."""
        from caskg.causal.engine import CausalGraphEngine
        from caskg.core.schema import SkillEdge, SkillNode

        class FakeConfig:
            WORKING_DIR = "/tmp/test_caskg"
            P_EXPLORE_INITIAL = 0.3
            P_EXPLORE_MIN = 0.05
            MAX_INTERVENTION_BUDGET = 100
            SCHEDULER_ALPHA_USAGE = 0.25
            SCHEDULER_BETA_CENTRALITY = 0.20
            SCHEDULER_GAMMA_UNCERTAINTY = 0.30
            SCHEDULER_DELTA_FAILURE = 0.15
            SCHEDULER_EPSILON_MAINTENANCE = 0.10
            EVOLUTION_CYCLE_INTERVAL = 50

        import tempfile
        tmp = tempfile.mkdtemp()
        engine = CausalGraphEngine(FakeConfig(), workspace_path=tmp)

        nodes = [SkillNode(name="A"), SkillNode(name="B")]
        edges = [SkillEdge(source="A", target="B", status="unverified", association_score=0.8)]
        engine.load_graph(nodes, edges)

        # The estimator for this edge should have alpha > 1 (informative prior)
        est = engine.intervention_engine.get_estimator("A", "B")
        assert est.alpha > 1.0  # Boosted by association_score
        assert est.mean > 0.5   # Prior leans toward causal


class TestBackwardTraversalThreshold:
    def test_low_transportability_edges_filtered(self):
        """Backward traversal should skip edges with transportability below threshold."""
        from caskg.causal.retrieval import CausalBackwardTraversal
        from caskg.core.schema import SkillEdge

        edges = [
            SkillEdge(source="A", target="B", status="confirmed_causal",
                     causal_score=0.9, transportability=0.9),
            SkillEdge(source="X", target="B", status="confirmed_causal",
                     causal_score=0.8, transportability=0.2),  # Below threshold
        ]

        class FakeConfig:
            TRANSPORTABILITY_THRESHOLD = 0.7

        traversal = CausalBackwardTraversal(edges, config=FakeConfig())
        prereqs = traversal.traverse(["B"], max_depth=3)
        # Only "A" should be found (t=0.9 > 0.7), not "X" (t=0.2 < 0.7)
        assert "A" in prereqs
        assert "X" not in prereqs


class TestCausalMetrics:
    def test_success_rate(self):
        from evaluation.causal_metrics import success_rate

        results = [{"reward": 1.0}, {"reward": 0.0}, {"reward": 1.0}]
        assert abs(success_rate(results) - 2.0 / 3.0) < 0.01

    def test_edge_precision_recall(self):
        from evaluation.causal_metrics import edge_f1, edge_precision, edge_recall

        predicted = [("A", "B"), ("B", "C"), ("X", "Y")]
        ground_truth = [("A", "B"), ("B", "C"), ("C", "D")]
        assert edge_precision(predicted, ground_truth) == pytest.approx(2.0 / 3.0)
        assert edge_recall(predicted, ground_truth) == pytest.approx(2.0 / 3.0)

    def test_probe_efficiency(self):
        from evaluation.causal_metrics import probe_efficiency

        assert probe_efficiency(10, 8) == pytest.approx(0.8)
        assert probe_efficiency(0, 0) == 0.0

    def test_edge_stability(self):
        from evaluation.causal_metrics import edge_stability

        # All same score = perfect stability
        assert edge_stability({"env1": 0.8, "env2": 0.8, "env3": 0.8}) == 1.0
