"""Counterfactual validator for CaSKG edge causal verification.

Orchestrates counterfactual intervention probes and updates edge causal
metadata based on intervention outcomes. The validator selects appropriate
intervention types per edge, computes composite causal scores, and manages
the edge lifecycle between unverified/confirmed/rejected states.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from caskg.causal.interventions import (
    InterventionEngine,
    InterventionProbe,
    REMOVAL,
    SUBSTITUTION,
    REORDERING,
)
from caskg.core.schema import SkillEdge


class CounterfactualValidator:
    """Validates causal edges via counterfactual intervention probes.

    Given a candidate edge, the validator:
    1. Selects appropriate intervention types based on edge semantics
    2. Designs probes for external execution
    3. Updates edge metadata from probe results
    4. Computes composite causal scores with type-specific weighting
    """

    def __init__(self, intervention_engine: InterventionEngine, config: Any) -> None:
        self.engine = intervention_engine
        self.config = config
        # Accumulated deltas per edge keyed by (source, target) -> {intervention_type: delta}
        self._edge_deltas: dict[tuple[str, str], dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Intervention type selection
    # ------------------------------------------------------------------

    def select_intervention_types(self, edge: SkillEdge) -> list[str]:
        """Select intervention types appropriate for an edge based on its type.

        Parameters
        ----------
        edge : SkillEdge
            The edge to select interventions for.

        Returns
        -------
        list[str]
            Ordered list of intervention type constants to apply.
        """
        edge_type = edge.type.lower() if edge.type else ""

        if edge_type in ("prereq", "prerequisite", "dependency"):
            return [REMOVAL, REORDERING]
        elif edge_type in ("enhance", "workflow"):
            return [REMOVAL]
        elif edge_type in ("similar", "alternative"):
            return [SUBSTITUTION]
        elif edge_type == "conflict":
            # Co-load negative test: removing a conflicting skill should help
            return [REMOVAL]
        else:
            return [REMOVAL]

    # ------------------------------------------------------------------
    # Composite score computation
    # ------------------------------------------------------------------

    def compute_composite_score(self, deltas: dict[str, float], edge_type: str) -> float:
        """Compute a weighted composite causal score from intervention deltas.

        Each intervention type contributes to the final score with weights
        that depend on the edge type semantics.

        Parameters
        ----------
        deltas : dict[str, float]
            Mapping from intervention type to observed delta value.
            Missing types default to 0.0.
        edge_type : str
            The semantic type of the edge (prereq, enhance, similar, etc.).

        Returns
        -------
        float
            Composite causal score in [0, 1], clipped.
        """
        edge_type_lower = edge_type.lower() if edge_type else ""

        # Define weight profiles per edge type
        weight_profiles: dict[str, tuple[float, float, float]] = {
            # (removal_weight, substitution_weight, reordering_weight)
            "prereq": (0.5, 0.1, 0.4),
            "prerequisite": (0.5, 0.1, 0.4),
            "dependency": (0.5, 0.1, 0.4),
            "enhance": (0.8, 0.1, 0.1),
            "workflow": (0.8, 0.1, 0.1),
            "similar": (0.2, 0.7, 0.1),
            "alternative": (0.2, 0.7, 0.1),
        }

        # Default weight profile
        w_removal, w_substitution, w_reordering = weight_profiles.get(
            edge_type_lower, (0.6, 0.2, 0.2)
        )

        removal_delta = deltas.get(REMOVAL, 0.0)
        substitution_delta = deltas.get(SUBSTITUTION, 0.0)
        reordering_delta = deltas.get(REORDERING, 0.0)

        score = (
            w_removal * removal_delta
            + w_substitution * substitution_delta
            + w_reordering * reordering_delta
        )

        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Edge update from probe result
    # ------------------------------------------------------------------

    def update_edge_from_probe(
        self, edge: SkillEdge, delta: float, intervention_type: str,
        episode: int = 0,
    ) -> SkillEdge:
        """Update an edge's causal metadata after an intervention probe result.

        Accumulates the delta for the given intervention_type in a per-edge
        lookup. Updates the Bayesian posterior (alpha/beta) for status
        thresholding. The causal_score is NOT set here -- call
        finalize_edge_score() after all probes for an edge are collected to
        compute the composite causal score.

        Parameters
        ----------
        edge : SkillEdge
            The edge to update (not modified in place).
        delta : float
            The observed performance delta from the intervention.
        intervention_type : str
            The type of intervention that produced this delta.
        episode : int
            The current episode number, used to update last_validated_episode
            so that pruning in the evolution cycle uses correct timing.

        Returns
        -------
        SkillEdge
            A new SkillEdge instance with updated Bayesian and status fields.
        """
        # Accumulate delta per intervention type for later composite scoring
        edge_key = (edge.source, edge.target)
        if edge_key not in self._edge_deltas:
            self._edge_deltas[edge_key] = {}
        self._edge_deltas[edge_key][intervention_type] = delta

        # Determine if this is positive causal evidence
        # Positive delta means removing/altering source hurt target performance
        epsilon = 0.05
        positive_evidence = delta > epsilon
        magnitude = abs(delta)

        # Update Bayesian posterior with magnitude-weighted evidence
        new_alpha = edge.alpha_posterior + (magnitude if positive_evidence else 0.0)
        new_beta = edge.beta_posterior + (0.0 if positive_evidence else magnitude)

        # Beta posterior mean -- used ONLY for status thresholding
        posterior_mean = new_alpha / (new_alpha + new_beta)

        # Compute updated uncertainty (Beta posterior variance)
        new_uncertainty = (new_alpha * new_beta) / (
            (new_alpha + new_beta) ** 2 * (new_alpha + new_beta + 1)
        )

        # Update intervention count
        new_intervention_count = edge.intervention_count + 1

        # Determine status based on Beta posterior mean thresholds
        confirm_threshold = getattr(
            self.config, "CAUSAL_CONFIRM_THRESHOLD", 0.6
        )
        reject_threshold = getattr(
            self.config, "CAUSAL_REJECT_THRESHOLD", 0.2
        )

        if posterior_mean > confirm_threshold:
            new_status = "confirmed_causal"
        elif posterior_mean < reject_threshold:
            new_status = "rejected_non_causal"
        else:
            new_status = "unverified"

        # Create updated edge -- causal_score is left unchanged here;
        # finalize_edge_score() will set it from the composite formula.
        updated_edge = replace(
            edge,
            uncertainty=new_uncertainty,
            alpha_posterior=new_alpha,
            beta_posterior=new_beta,
            intervention_count=new_intervention_count,
            status=new_status,
            last_validated_episode=episode,
        )

        return updated_edge

    # ------------------------------------------------------------------
    # Composite score finalization
    # ------------------------------------------------------------------

    def finalize_edge_score(self, edge: SkillEdge) -> SkillEdge:
        """Compute and apply the composite causal score to an edge.

        Should be called after all intervention probes for the edge have been
        recorded via update_edge_from_probe(). Uses the accumulated deltas and
        the edge type to produce:
            c_ij = alpha*Delta_remove + beta*Delta_subst + gamma*Delta_order

        The composite score is written to edge.causal_score. The status and
        uncertainty remain as set by update_edge_from_probe() (driven by Beta
        posterior mean).

        Parameters
        ----------
        edge : SkillEdge
            The edge whose composite score should be finalized.

        Returns
        -------
        SkillEdge
            A new SkillEdge with causal_score set to the composite value.
        """
        edge_key = (edge.source, edge.target)
        deltas = self._edge_deltas.get(edge_key, {})

        composite = self.compute_composite_score(deltas, edge.type)

        return replace(edge, causal_score=composite)

    # ------------------------------------------------------------------
    # Full edge validation
    # ------------------------------------------------------------------

    def validate_edge(
        self,
        edge: SkillEdge,
        available_skills: list[str],
        alternatives: list[str] | None = None,
    ) -> tuple[SkillEdge, list[InterventionProbe]]:
        """Design intervention probes for validating a candidate edge.

        Selects intervention types based on edge semantics, designs the
        corresponding probes, and returns them for external execution.
        The actual execution happens in the scheduler/environment loop.

        Parameters
        ----------
        edge : SkillEdge
            The edge to validate.
        available_skills : list[str]
            Skills available in the execution environment.
        alternatives : list[str] or None
            Alternative skills that can substitute for edge.source.

        Returns
        -------
        tuple[SkillEdge, list[InterventionProbe]]
            The (possibly updated) edge and a list of probes to execute.
        """
        if alternatives is None:
            alternatives = []

        intervention_types = self.select_intervention_types(edge)
        probes: list[InterventionProbe] = []

        for itype in intervention_types:
            if itype == REMOVAL:
                probe = self.engine.design_removal_probe(edge, available_skills)
                probes.append(probe)
            elif itype == SUBSTITUTION:
                probe = self.engine.design_substitution_probe(
                    edge, available_skills, alternatives
                )
                if probe is not None:
                    probes.append(probe)
            elif itype == REORDERING:
                probe = self.engine.design_reordering_probe(edge, available_skills)
                probes.append(probe)

        return (edge, probes)
