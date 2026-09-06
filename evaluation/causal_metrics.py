"""
CaSKG Evaluation Metrics

Comprehensive evaluation metrics for the Causal Skill Knowledge Graph (CaSKG)
proposal, covering task performance, retrieval quality, graph truth, causal-specific
evaluation, maintenance, and generalization dimensions.
"""

from typing import Any


# =============================================================================
# Task Performance Metrics
# =============================================================================


def success_rate(results: list[dict]) -> float:
    """Compute the proportion of successful episodes.

    Args:
        results: List of episode result dicts, each containing a "reward" field
                 with value 0 (failure) or 1 (success).

    Returns:
        Proportion of results with reward == 1. Returns 0.0 if results is empty.
    """
    if not results:
        return 0.0
    successes = sum(1 for r in results if r["reward"] == 1)
    return successes / len(results)


def avg_steps(results: list[dict]) -> float:
    """Compute the mean number of steps across episodes.

    Args:
        results: List of episode result dicts, each containing a "steps" field
                 (integer number of steps taken).

    Returns:
        Mean step count. Returns 0.0 if results is empty.
    """
    if not results:
        return 0.0
    return sum(r["steps"] for r in results) / len(results)


def token_cost(results: list[dict]) -> float:
    """Compute the mean token usage across episodes.

    Args:
        results: List of episode result dicts, each containing a "tokens" field
                 (integer token count consumed).

    Returns:
        Mean token cost. Returns 0.0 if results is empty.
    """
    if not results:
        return 0.0
    return sum(r["tokens"] for r in results) / len(results)


def repair_cost(results: list[dict]) -> float:
    """Compute the mean repair steps across episodes.

    Args:
        results: List of episode result dicts, each containing a "repair_steps"
                 field (0 if no repair was needed, positive integer otherwise).

    Returns:
        Mean repair step count. Returns 0.0 if results is empty.
    """
    if not results:
        return 0.0
    return sum(r["repair_steps"] for r in results) / len(results)


# =============================================================================
# Retrieval Quality Metrics
# =============================================================================


def recall_at_k(retrieved: list[str], ground_truth: list[str], k: int) -> float:
    """Compute Recall@K for a single retrieval query.

    Measures the proportion of ground-truth items that appear in the top-k
    retrieved results.

    Args:
        retrieved: Ordered list of retrieved skill identifiers.
        ground_truth: List of relevant skill identifiers (the gold set).
        k: Cutoff rank for evaluation.

    Returns:
        Fraction of ground_truth items found in retrieved[:k].
        Returns 0.0 if ground_truth is empty.
    """
    if not ground_truth:
        return 0.0
    top_k = set(retrieved[:k])
    hits = sum(1 for item in ground_truth if item in top_k)
    return hits / len(ground_truth)


def mrr(retrieved_lists: list[list[str]], ground_truths: list[list[str]]) -> float:
    """Compute Mean Reciprocal Rank across multiple queries.

    For each query, the reciprocal rank is 1/rank of the first relevant item
    in the retrieved list (0 if no relevant item is found).

    Args:
        retrieved_lists: List of retrieved rankings, one per query.
        ground_truths: List of ground-truth sets, one per query (parallel to
                       retrieved_lists).

    Returns:
        Mean of reciprocal ranks across all queries.
        Returns 0.0 if no queries are provided.
    """
    if not retrieved_lists:
        return 0.0

    total_rr = 0.0
    for retrieved, truth in zip(retrieved_lists, ground_truths):
        truth_set = set(truth)
        rr = 0.0
        for rank, item in enumerate(retrieved, start=1):
            if item in truth_set:
                rr = 1.0 / rank
                break
        total_rr += rr

    return total_rr / len(retrieved_lists)


def bundle_sufficiency(bundle: list[str], required: list[str]) -> float:
    """Compute the fraction of required skills present in a skill bundle.

    Measures whether a retrieved bundle covers all skills needed to complete
    a task.

    Args:
        bundle: List of skill identifiers in the retrieved bundle.
        required: List of skill identifiers required for the task.

    Returns:
        Fraction of required skills found in bundle.
        Returns 1.0 if required is empty (vacuous truth).
    """
    if not required:
        return 1.0
    bundle_set = set(bundle)
    covered = sum(1 for skill in required if skill in bundle_set)
    return covered / len(required)


def prerequisite_coverage(bundle: list[str], prerequisites: dict[str, list[str]]) -> float:
    """Compute how well a bundle satisfies prerequisite constraints.

    For each skill in the bundle that has prerequisites defined, checks whether
    those prerequisites are also present in the bundle.

    Args:
        bundle: List of skill identifiers in the bundle.
        prerequisites: Mapping from skill identifier to its list of prerequisite
                       skill identifiers.

    Returns:
        Average prerequisite satisfaction ratio across all skills in the bundle
        that have prerequisites. Returns 1.0 if no skill in the bundle has
        prerequisites defined.
    """
    bundle_set = set(bundle)
    ratios: list[float] = []

    for skill in bundle:
        prereqs = prerequisites.get(skill, [])
        if not prereqs:
            continue
        satisfied = sum(1 for p in prereqs if p in bundle_set)
        ratios.append(satisfied / len(prereqs))

    if not ratios:
        return 1.0
    return sum(ratios) / len(ratios)


# =============================================================================
# Graph Truth Metrics
# =============================================================================


def edge_precision(
    predicted_edges: list[tuple[str, str]],
    ground_truth_edges: list[tuple[str, str]],
) -> float:
    """Compute edge precision: fraction of predicted edges that are correct.

    Args:
        predicted_edges: List of (source, target) tuples predicted by the model.
        ground_truth_edges: List of (source, target) tuples in the gold graph.

    Returns:
        Fraction of predicted edges found in ground truth.
        Returns 0.0 if predicted_edges is empty.
    """
    if not predicted_edges:
        return 0.0
    gt_set = set(ground_truth_edges)
    correct = sum(1 for edge in predicted_edges if edge in gt_set)
    return correct / len(predicted_edges)


def edge_recall(
    predicted_edges: list[tuple[str, str]],
    ground_truth_edges: list[tuple[str, str]],
) -> float:
    """Compute edge recall: fraction of ground-truth edges that are predicted.

    Args:
        predicted_edges: List of (source, target) tuples predicted by the model.
        ground_truth_edges: List of (source, target) tuples in the gold graph.

    Returns:
        Fraction of ground truth edges found in predictions.
        Returns 0.0 if ground_truth_edges is empty.
    """
    if not ground_truth_edges:
        return 0.0
    pred_set = set(predicted_edges)
    found = sum(1 for edge in ground_truth_edges if edge in pred_set)
    return found / len(ground_truth_edges)


def edge_f1(
    predicted_edges: list[tuple[str, str]],
    ground_truth_edges: list[tuple[str, str]],
) -> float:
    """Compute edge F1: harmonic mean of edge precision and recall.

    Args:
        predicted_edges: List of (source, target) tuples predicted by the model.
        ground_truth_edges: List of (source, target) tuples in the gold graph.

    Returns:
        F1 score combining precision and recall. Returns 0.0 if both are zero.
    """
    p = edge_precision(predicted_edges, ground_truth_edges)
    r = edge_recall(predicted_edges, ground_truth_edges)
    if p + r == 0.0:
        return 0.0
    return 2.0 * p * r / (p + r)


def spurious_edge_rate(
    confirmed_edges: list[tuple[str, str]],
    ground_truth_edges: list[tuple[str, str]],
) -> float:
    """Compute the fraction of confirmed edges that are spurious (not in ground truth).

    This metric identifies false positives among edges that passed the causal
    confirmation process.

    Args:
        confirmed_edges: List of (source, target) tuples that were confirmed
                         by the causal verification pipeline.
        ground_truth_edges: List of (source, target) tuples in the gold graph.

    Returns:
        Fraction of confirmed edges NOT in ground truth.
        Returns 0.0 if confirmed_edges is empty.
    """
    if not confirmed_edges:
        return 0.0
    gt_set = set(ground_truth_edges)
    spurious = sum(1 for edge in confirmed_edges if edge not in gt_set)
    return spurious / len(confirmed_edges)


# =============================================================================
# Causal-Specific Metrics
# =============================================================================


def avg_counterfactual_delta(causal_scores: list[float]) -> float:
    """Compute the mean of causal effect scores (Average Treatment Effect estimates).

    Each score represents the estimated causal effect of including a skill,
    computed via counterfactual comparison (with-skill minus without-skill
    performance).

    Args:
        causal_scores: List of ATE estimates for individual edges.

    Returns:
        Mean causal score. Returns 0.0 if list is empty.
    """
    if not causal_scores:
        return 0.0
    return sum(causal_scores) / len(causal_scores)


def edge_calibration_error(
    predicted_scores: list[float],
    actual_outcomes: list[float],
) -> float:
    """Compute mean absolute calibration error between predicted and actual effects.

    Measures how well the predicted causal scores align with observed outcomes.
    A well-calibrated model produces scores close to actual performance deltas.

    Args:
        predicted_scores: List of predicted causal effect scores for edges.
        actual_outcomes: List of observed performance effects (parallel to
                         predicted_scores).

    Returns:
        Mean absolute difference between predicted and actual.
        Returns 0.0 if lists are empty.
    """
    if not predicted_scores:
        return 0.0
    total_error = sum(
        abs(pred - actual)
        for pred, actual in zip(predicted_scores, actual_outcomes)
    )
    return total_error / len(predicted_scores)


def uncertainty_reduction(
    before_uncertainties: list[float],
    after_uncertainties: list[float],
) -> float:
    """Compute the average proportional reduction in uncertainty after probing.

    Measures how effectively causal probes reduce uncertainty about edge
    existence or strength.

    Args:
        before_uncertainties: Uncertainty values before probing (parallel lists).
        after_uncertainties: Uncertainty values after probing.

    Returns:
        Average proportional reduction: mean of (before - after) / before.
        Entries where before == 0 are skipped (no reduction possible).
        Returns 0.0 if no valid entries exist.
    """
    if not before_uncertainties:
        return 0.0

    reductions: list[float] = []
    for before, after in zip(before_uncertainties, after_uncertainties):
        if before == 0.0:
            continue
        reduction = (before - after) / before
        reductions.append(reduction)

    if not reductions:
        return 0.0
    return sum(reductions) / len(reductions)


def probe_efficiency(probes_used: int, edges_resolved: int) -> float:
    """Compute probe efficiency: edges resolved per probe used.

    Higher values indicate more efficient causal discovery (fewer probes
    needed to confirm or reject edge hypotheses).

    Args:
        probes_used: Total number of counterfactual probes executed.
        edges_resolved: Number of edges confirmed or rejected by those probes.

    Returns:
        Ratio of edges_resolved / probes_used.
        Returns 0.0 if probes_used is 0.
    """
    if probes_used == 0:
        return 0.0
    return edges_resolved / probes_used


# =============================================================================
# Maintenance Metrics
# =============================================================================


def harmful_merge_rate(merge_outcomes: list[dict]) -> float:
    """Compute the fraction of skill merges that caused harmful performance drops.

    A merge is considered harmful if it causes a performance delta below -0.05
    (i.e., more than 5% degradation).

    Args:
        merge_outcomes: List of outcome dicts, each containing a
                        "performance_delta" field (float).

    Returns:
        Fraction of merges with performance_delta < -0.05.
        Returns 0.0 if merge_outcomes is empty.
    """
    if not merge_outcomes:
        return 0.0
    harmful = sum(1 for m in merge_outcomes if m["performance_delta"] < -0.05)
    return harmful / len(merge_outcomes)


def safe_retire_rate(retire_outcomes: list[dict]) -> float:
    """Compute the fraction of skill retirements that were safe.

    A retirement is considered safe if the performance delta is >= -0.02
    (i.e., no more than 2% degradation).

    Args:
        retire_outcomes: List of outcome dicts, each containing a
                         "performance_delta" field (float).

    Returns:
        Fraction of retirements with performance_delta >= -0.02.
        Returns 0.0 if retire_outcomes is empty.
    """
    if not retire_outcomes:
        return 0.0
    safe = sum(1 for r in retire_outcomes if r["performance_delta"] >= -0.02)
    return safe / len(retire_outcomes)


def graph_drift_recovery_time(
    health_scores: list[float],
    threshold: float = 0.9,
) -> int:
    """Compute the number of episodes until graph health recovers after a drop.

    Scans the health_scores sequence for the first drop below threshold, then
    counts how many episodes pass until the score recovers above threshold.

    Args:
        health_scores: Time-ordered list of graph health scores (0.0 to 1.0).
        threshold: Recovery threshold (default 0.9).

    Returns:
        Number of episodes from first drop below threshold to first recovery
        above threshold. Returns 0 if no drop occurs. Returns len(health_scores)
        if the score never recovers.
    """
    drop_index = -1
    for i, score in enumerate(health_scores):
        if score < threshold:
            drop_index = i
            break

    if drop_index == -1:
        return 0

    for i in range(drop_index + 1, len(health_scores)):
        if health_scores[i] >= threshold:
            return i - drop_index

    return len(health_scores) - drop_index


# =============================================================================
# Generalization Metrics
# =============================================================================


def cross_task_transfer(source_success: float, target_success: float) -> float:
    """Compute cross-task transfer retention ratio.

    Measures how well skills learned on a source task transfer to a target task.
    A ratio of 1.0 means perfect retention; above 1.0 means positive transfer.

    Args:
        source_success: Success rate on the source (training) task.
        target_success: Success rate on the target (transfer) task.

    Returns:
        Retention ratio: target_success / max(source_success, 0.01).
        The denominator is clamped to avoid division by zero.
    """
    return target_success / max(source_success, 0.01)


def cross_env_transfer(source_success: float, target_success: float) -> float:
    """Compute cross-environment transfer retention ratio.

    Measures how well skills transfer across different environments (e.g.,
    from one ALFWorld house to another).

    Args:
        source_success: Success rate in the source environment.
        target_success: Success rate in the target environment.

    Returns:
        Retention ratio: target_success / max(source_success, 0.01).
        The denominator is clamped to avoid division by zero.
    """
    return target_success / max(source_success, 0.01)


def edge_stability(per_env_scores: dict[str, float]) -> float:
    """Compute edge stability (transportability) across environments.

    Measures how consistent a causal edge's strength is across different
    environments. High stability (close to 1.0) indicates the edge represents
    a robust, transportable causal relationship.

    Args:
        per_env_scores: Mapping from environment name to the causal edge score
                        observed in that environment.

    Returns:
        1 - variance of scores across environments. Returns 1.0 if fewer than
        2 environments are provided (no variance measurable).
    """
    scores = list(per_env_scores.values())
    if len(scores) < 2:
        return 1.0

    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    return 1.0 - variance
