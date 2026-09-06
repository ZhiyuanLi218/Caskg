"""Counterfactual intervention engine for CaSKG causal inference.

Designs and records causal intervention probes (removal, substitution,
reordering) against skill graph edges, and maintains Bayesian estimators
for edge causal strength.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from caskg.causal.trace_store import ExecutionTrace, TraceStore
from caskg.core.schema import SkillEdge


# ---------------------------------------------------------------------------
# Intervention types (string constants for simplicity)
# ---------------------------------------------------------------------------

REMOVAL = "removal"
SUBSTITUTION = "substitution"
REORDERING = "reordering"


class InterventionType:
    """Namespace for intervention type constants."""

    REMOVAL = REMOVAL
    SUBSTITUTION = SUBSTITUTION
    REORDERING = REORDERING


# ---------------------------------------------------------------------------
# Intervention probe
# ---------------------------------------------------------------------------


@dataclass
class InterventionProbe:
    """Describes a single counterfactual intervention experiment."""

    edge_source: str
    edge_target: str
    intervention_type: str
    task_id: str
    control_skills: list[str]  # skills available in control condition
    treated_skills: list[str]  # skills available in treatment condition
    substitute_skill: str = ""  # for substitution probes


# ---------------------------------------------------------------------------
# Bayesian edge estimator
# ---------------------------------------------------------------------------


class BayesianEdgeEstimator:
    """Beta-distribution estimator for edge causal strength.

    Maintains alpha (positive evidence) and beta (negative evidence)
    parameters of a Beta distribution. Provides point estimate (mean),
    uncertainty (variance), and credible-interval-based confidence checks.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        self.alpha = alpha
        self.beta = beta

    def update(self, positive: bool, magnitude: float = 1.0) -> None:
        """Update the posterior with an observation.

        Args:
            positive: True if the intervention confirmed causal influence
                      (removal hurt performance), False otherwise.
            magnitude: The strength of the evidence (default 1.0). Larger
                       deltas contribute more to the posterior update.
        """
        if positive:
            self.alpha += magnitude
        else:
            self.beta += magnitude

    @property
    def mean(self) -> float:
        """Posterior mean of the Beta distribution."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Posterior variance of the Beta distribution."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def confident_causal(self, threshold: float = 0.5) -> bool:
        """Check if the lower bound of 95% CI exceeds threshold.

        Uses the normal approximation: lower bound = mean - 2*std.
        Returns True if the edge is confidently causal.
        """
        std = math.sqrt(self.variance)
        lower_bound = self.mean - 2.0 * std
        return lower_bound > threshold

    def confident_non_causal(self, threshold: float = 0.5) -> bool:
        """Check if the upper bound of 95% CI is below threshold.

        Uses the normal approximation: upper bound = mean + 2*std.
        Returns True if the edge is confidently non-causal.
        """
        std = math.sqrt(self.variance)
        upper_bound = self.mean + 2.0 * std
        return upper_bound < threshold


# ---------------------------------------------------------------------------
# Intervention engine
# ---------------------------------------------------------------------------


class InterventionEngine:
    """Designs, executes, and records counterfactual intervention probes.

    Manages a collection of BayesianEdgeEstimators keyed by (source, target)
    edge pairs, and interfaces with the TraceStore for persistence.
    """

    # Minimum performance delta to count as a positive causal signal.
    DEFAULT_EPSILON = 0.05

    def __init__(self, trace_store: TraceStore, config: Any = None) -> None:
        self.trace_store = trace_store
        self.config = config or {}
        self._estimators: dict[tuple[str, str], BayesianEdgeEstimator] = {}

    @property
    def _epsilon(self) -> float:
        """Minimum delta to treat as positive causal evidence."""
        if isinstance(self.config, dict):
            return float(self.config.get("epsilon", self.DEFAULT_EPSILON))
        return getattr(self.config, "epsilon", self.DEFAULT_EPSILON)

    # ------------------------------------------------------------------
    # Estimator management
    # ------------------------------------------------------------------

    def get_estimator(
        self, source: str, target: str, edge: SkillEdge | None = None
    ) -> BayesianEdgeEstimator:
        """Get or create an estimator for an edge.

        If no estimator exists for the given edge, creates one. If prior
        traces exist for this edge, initializes from their accumulated
        alpha/beta posteriors. If no traces exist but an edge with a
        positive association_score is provided, uses an informative prior
        derived from that score (Algorithm 1 Step 2).
        """
        key = (source, target)
        if key not in self._estimators:
            # Try to bootstrap from existing traces for this edge
            traces = self.trace_store.by_edge(source, target)
            if traces:
                # Count positive/negative outcomes from historical data
                alpha = 1.0
                beta = 1.0
                for trace in traces:
                    if trace.is_intervention and trace.intervention_edge == (source, target):
                        # Use outcome > 0.5 as heuristic for "causal confirmed"
                        if trace.outcome > 0.5:
                            alpha += 1.0
                        else:
                            beta += 1.0
                self._estimators[key] = BayesianEdgeEstimator(alpha=alpha, beta=beta)
            elif edge is not None and edge.association_score > 0:
                # Informative prior from association signal (Algorithm 1 Step 2).
                # Scale weakly (factor 3) so the prior is suggestive but
                # easily overridden by intervention data.
                a_ij = edge.association_score
                alpha_prior = 1.0 + a_ij * 3.0
                beta_prior = 1.0 + (1.0 - a_ij) * 3.0
                self._estimators[key] = BayesianEdgeEstimator(
                    alpha=alpha_prior, beta=beta_prior
                )
            else:
                self._estimators[key] = BayesianEdgeEstimator()
        return self._estimators[key]

    # ------------------------------------------------------------------
    # Probe design
    # ------------------------------------------------------------------

    def design_removal_probe(
        self, edge: SkillEdge, available_skills: list[str]
    ) -> InterventionProbe:
        """Design a removal intervention probe.

        Control condition: all available skills.
        Treatment condition: available skills minus the edge source.
        """
        control_skills = list(available_skills)
        treated_skills = [s for s in available_skills if s != edge.source]

        return InterventionProbe(
            edge_source=edge.source,
            edge_target=edge.target,
            intervention_type=REMOVAL,
            task_id=self._generate_task_id(),
            control_skills=control_skills,
            treated_skills=treated_skills,
        )

    def design_substitution_probe(
        self,
        edge: SkillEdge,
        available_skills: list[str],
        alternatives: list[str],
    ) -> InterventionProbe | None:
        """Design a substitution intervention probe.

        Control condition: available skills with the original source.
        Treatment condition: available skills with source replaced by the
        first alternative.

        Returns None if no alternatives are available.
        """
        if not alternatives:
            return None

        substitute = alternatives[0]
        control_skills = list(available_skills)
        treated_skills = [
            substitute if s == edge.source else s for s in available_skills
        ]
        # If source was not in available_skills, append the substitute anyway
        if edge.source not in available_skills:
            treated_skills.append(substitute)

        return InterventionProbe(
            edge_source=edge.source,
            edge_target=edge.target,
            intervention_type=SUBSTITUTION,
            task_id=self._generate_task_id(),
            control_skills=control_skills,
            treated_skills=treated_skills,
            substitute_skill=substitute,
        )

    def design_reordering_probe(
        self, edge: SkillEdge, available_skills: list[str]
    ) -> InterventionProbe:
        """Design a reordering intervention probe.

        Control condition: available skills in normal order (source before target).
        Treatment condition: available skills with target placed before source.
        """
        # Ensure control has source before target
        control_skills = list(available_skills)
        source_idx = None
        target_idx = None
        for i, s in enumerate(control_skills):
            if s == edge.source and source_idx is None:
                source_idx = i
            if s == edge.target and target_idx is None:
                target_idx = i

        # If source comes after target in control, swap them for control ordering
        if source_idx is not None and target_idx is not None and source_idx > target_idx:
            control_skills[source_idx], control_skills[target_idx] = (
                control_skills[target_idx],
                control_skills[source_idx],
            )

        # Treatment: target before source
        treated_skills = list(control_skills)
        source_idx_t = None
        target_idx_t = None
        for i, s in enumerate(treated_skills):
            if s == edge.source and source_idx_t is None:
                source_idx_t = i
            if s == edge.target and target_idx_t is None:
                target_idx_t = i

        if (
            source_idx_t is not None
            and target_idx_t is not None
            and source_idx_t < target_idx_t
        ):
            treated_skills[source_idx_t], treated_skills[target_idx_t] = (
                treated_skills[target_idx_t],
                treated_skills[source_idx_t],
            )

        return InterventionProbe(
            edge_source=edge.source,
            edge_target=edge.target,
            intervention_type=REORDERING,
            task_id=self._generate_task_id(),
            control_skills=control_skills,
            treated_skills=treated_skills,
        )

    # ------------------------------------------------------------------
    # Outcome recording
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        probe: InterventionProbe,
        control_outcome: float,
        treated_outcome: float,
    ) -> float:
        """Record the outcome of an intervention probe.

        Computes delta = control_outcome - treated_outcome. A positive delta
        means removing/altering the source skill hurt performance, confirming
        causal influence.

        Updates the Bayesian estimator and records the intervention trace.

        Returns:
            The computed delta value.
        """
        delta = control_outcome - treated_outcome

        # Update the estimator with magnitude-weighted evidence
        estimator = self.get_estimator(probe.edge_source, probe.edge_target)
        positive = delta > self._epsilon
        estimator.update(positive, magnitude=abs(delta))

        # Record intervention trace
        # skills_available = full set (control condition, all skills available)
        # skills_used = the treated/experimental condition that was actually run
        trace = ExecutionTrace(
            trace_id=str(uuid.uuid4()),
            task_id=probe.task_id,
            environment="intervention",
            task_type=probe.intervention_type,
            skills_used=probe.treated_skills,
            skills_available=probe.control_skills,
            outcome=max(0.0, min(1.0, 0.5 + delta)),  # normalize around 0.5
            steps=1,
            timestamp=datetime.now(timezone.utc).isoformat(),
            is_intervention=True,
            intervention_edge=(probe.edge_source, probe.edge_target),
            intervention_type=probe.intervention_type,
        )
        self.trace_store.record(trace)

        return delta

    # ------------------------------------------------------------------
    # Causal score computation
    # ------------------------------------------------------------------

    def compute_causal_scores(self, edge: SkillEdge) -> tuple[float, float]:
        """Compute the causal score and uncertainty for an edge.

        Returns:
            A tuple of (causal_score, uncertainty) where causal_score is
            the posterior mean and uncertainty is the posterior variance.
        """
        estimator = self.get_estimator(edge.source, edge.target, edge=edge)
        return estimator.mean, estimator.variance

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_task_id() -> str:
        """Generate a unique task identifier for a probe."""
        return f"intervention-{uuid.uuid4().hex[:12]}"
