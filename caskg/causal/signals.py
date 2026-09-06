"""Signal computation classes for CaSKG candidate edge scoring.

Each signal class computes a normalized [0, 1] score for a directed pair
(source_skill, target_skill).  Signals may be synchronous (lexical, IO) or
asynchronous (semantic embedding, LLM judge).

``compute_association_score`` combines individual signal scores using
configurable weights into a single association score a_ij.
"""

from __future__ import annotations

import json
import re
from typing import Any

from caskg.core.schema import SkillNode
from caskg.causal.trace_store import TraceStore


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

TOKEN_STOPWORDS: set[str] = {
    "a",
    "an",
    "and",
    "any",
    "arg",
    "args",
    "array",
    "bool",
    "boolean",
    "data",
    "dict",
    "file",
    "float",
    "for",
    "from",
    "in",
    "input",
    "int",
    "json",
    "list",
    "object",
    "of",
    "on",
    "or",
    "output",
    "path",
    "record",
    "result",
    "set",
    "str",
    "string",
    "text",
    "the",
    "to",
    "value",
}


def _signature_tokens(values: list[str]) -> set[str]:
    """Tokenize a list of strings into a canonical set of tokens.

    Splits on non-alphanumeric characters, lowercases, strips trailing 's'
    (simple plural removal), and filters out stopwords and short tokens.
    """
    tokens: set[str] = set()
    for value in values:
        lowered = value.lower()
        normalized = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
        if normalized:
            tokens.add(normalized)
        for token in re.findall(r"[a-z0-9]+", lowered):
            token = token.rstrip("s")
            if len(token) < 3 or token in TOKEN_STOPWORDS:
                continue
            tokens.add(token)
    return tokens


# ---------------------------------------------------------------------------
# Signal classes
# ---------------------------------------------------------------------------


class SemanticSignal:
    """Computes semantic similarity between two skill nodes via embeddings."""

    async def compute(
        self,
        node_i: SkillNode,
        node_j: SkillNode,
        embedding_service: Any,
    ) -> float:
        """Encode both node descriptions and return cosine similarity in [0, 1].

        Args:
            node_i: First skill node.
            node_j: Second skill node.
            embedding_service: Service with an encode(texts) -> list[list[float]] method.

        Returns:
            Cosine similarity clamped to [0, 1].
        """
        text_i = f"{node_i.name}: {node_i.description}"
        text_j = f"{node_j.name}: {node_j.description}"

        try:
            embeddings = await embedding_service.encode([text_i, text_j])
            vec_i = embeddings[0]
            vec_j = embeddings[1]
            return self._cosine_similarity(vec_i, vec_j)
        except Exception:
            return 0.0

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors, clamped to [0, 1]."""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        similarity = dot / (norm_a * norm_b)
        # Clamp to [0, 1] (cosine similarity can be negative for dissimilar vectors)
        return max(0.0, min(1.0, similarity))


class LexicalSignal:
    """Computes token-level overlap between two skill nodes.

    Uses the _signature_tokens pattern: split on non-alphanumeric,
    lowercase, strip plurals, filter stopwords.
    """

    def compute(self, node_i: SkillNode, node_j: SkillNode) -> float:
        """Compute lexical overlap between name, description, tooling, and domain tags.

        Uses Jaccard similarity: |intersection| / |union|.

        Args:
            node_i: First skill node.
            node_j: Second skill node.

        Returns:
            Overlap score in [0, 1].
        """
        tokens_i = self._collect_tokens(node_i)
        tokens_j = self._collect_tokens(node_j)

        if not tokens_i or not tokens_j:
            return 0.0

        intersection = tokens_i & tokens_j
        union = tokens_i | tokens_j

        if not union:
            return 0.0

        return len(intersection) / len(union)

    @staticmethod
    def _collect_tokens(node: SkillNode) -> set[str]:
        """Collect signature tokens from node name, description, tooling, domain_tags."""
        values: list[str] = [
            node.name,
            node.description,
        ]
        values.extend(node.tooling_list)
        values.extend(node.domain_tags_list)
        return _signature_tokens(values)


class IOCompatibilitySignal:
    """Computes input/output schema compatibility between two skill nodes.

    Checks if outputs of node_i overlap with inputs of node_j (forward)
    or vice versa (reverse), returning the max of both directions.
    Uses token-level matching like the existing _schema_overlap_score.
    """

    def compute(self, node_i: SkillNode, node_j: SkillNode) -> float:
        """Check if outputs of one node overlap with inputs of another.

        Computes both forward (i outputs -> j inputs) and reverse
        (j outputs -> i inputs) compatibility, returning the maximum.
        Falls back to raw inputs/outputs fields when output_types/input_types
        are empty (handles skills without explicit JSON schema annotations).

        Args:
            node_i: First skill node.
            node_j: Second skill node.

        Returns:
            Max of forward and reverse IO compatibility scores in [0, 1].
        """
        i_outputs = node_i.output_types or [s.strip() for s in node_i.outputs.split("\n") if s.strip()]
        j_inputs = node_j.input_types or [s.strip() for s in node_j.inputs.split("\n") if s.strip()]
        j_outputs = node_j.output_types or [s.strip() for s in node_j.outputs.split("\n") if s.strip()]
        i_inputs = node_i.input_types or [s.strip() for s in node_i.inputs.split("\n") if s.strip()]

        forward_score = self._schema_overlap_score(i_outputs, j_inputs)
        reverse_score = self._schema_overlap_score(j_outputs, i_inputs)
        return max(forward_score, reverse_score)

    @staticmethod
    def _schema_overlap_score(
        producer_values: list[str],
        consumer_values: list[str],
    ) -> float:
        """Token-level schema overlap between producer outputs and consumer inputs.

        Mirrors the _schema_overlap_score pattern from engine.py.
        """
        if not producer_values or not consumer_values:
            return 0.0

        best_score = 0.0

        for producer in producer_values:
            producer_norm = re.sub(r"[^a-z0-9]+", "_", producer.lower()).strip("_")
            producer_tokens = _signature_tokens([producer])
            for consumer in consumer_values:
                consumer_norm = re.sub(r"[^a-z0-9]+", "_", consumer.lower()).strip("_")
                consumer_tokens = _signature_tokens([consumer])

                # Exact normalized match
                if producer_norm and producer_norm == consumer_norm:
                    return 1.0
                # Substring containment
                if producer_norm and consumer_norm and (
                    producer_norm in consumer_norm or consumer_norm in producer_norm
                ):
                    return 0.85

                overlap = producer_tokens & consumer_tokens
                if not overlap:
                    continue

                union = producer_tokens | consumer_tokens
                score = max(0.5, len(overlap) / max(len(union), 1))
                if score > best_score:
                    best_score = score

        return best_score


class CooccurrenceSignal:
    """Computes co-occurrence signal from execution traces.

    Uses the trace store's cooccurrence_matrix to compute a normalized
    co-occurrence score, factoring in temporal precedence.
    """

    def compute(
        self,
        source: str,
        target: str,
        trace_store: TraceStore,
        window: int = 200,
    ) -> float:
        """Compute co-occurrence score between two skills from trace data.

        Combines normalized co-occurrence count with temporal precedence.
        Higher score if source consistently precedes target.

        Args:
            source: Name of the source skill.
            target: Name of the target skill.
            trace_store: The trace store to query.
            window: Number of recent traces to consider.

        Returns:
            Co-occurrence signal in [0, 1].
        """
        # Get co-occurrence matrix for the pair
        matrix = trace_store.cooccurrence_matrix([source, target], window=window)

        # Normalize pair key (lexicographic ordering as used by cooccurrence_matrix)
        pair_key = (min(source, target), max(source, target))
        pair_count = matrix.get(pair_key, 0)

        if pair_count == 0:
            return 0.0

        # Find max co-occurrence in window for normalization
        max_cooccurrence = max(matrix.values()) if matrix else 1
        if max_cooccurrence == 0:
            max_cooccurrence = 1

        # Normalized co-occurrence frequency
        frequency_score = pair_count / max_cooccurrence

        # Factor in temporal precedence (higher if source consistently precedes target)
        precedence = trace_store.temporal_precedence(source, target)

        # Combine: weight frequency at 0.6, temporal precedence at 0.4
        combined = 0.6 * frequency_score + 0.4 * precedence

        return max(0.0, min(1.0, combined))


class RepairSignal:
    """Computes temporal repair signal: does adding source fix target failures?

    Detects the temporal pattern: target FAILED (outcome < 0.5) without source
    in skills_used, followed by a LATER trace with the same (or similar)
    task_type where target SUCCEEDED (outcome >= 0.5) and source WAS in
    skills_used.  The repair signal is (count of such recovery pairs) /
    (count of target failures without source).
    """

    def compute(
        self,
        source: str,
        target: str,
        trace_store: TraceStore,
    ) -> float:
        """Compute temporal repair signal between source and target skills.

        Sorts traces by timestamp (falling back to trace_id ordering) and
        looks for failure-then-recovery pairs on the same task_type.

        Args:
            source: Name of the potential prerequisite skill.
            target: Name of the skill that may need the source.
            trace_store: The trace store to query.

        Returns:
            Repair signal in [0, 1]. Higher means stronger evidence that
            source causally enables target.
        """
        traces = trace_store.recent(n=500)

        if not traces:
            return 0.0

        # Sort traces temporally: by timestamp first, then trace_id as tiebreaker
        sorted_traces = sorted(traces, key=lambda t: (t.timestamp, t.trace_id))

        # Phase 1: Collect failure traces (target failed, source absent)
        # Grouped by task_type so we can match them against later recoveries.
        # Each entry: list of indices into sorted_traces
        failures_by_task_type: dict[str, list[int]] = {}

        for idx, trace in enumerate(sorted_traces):
            if target not in trace.skills_used:
                continue
            if trace.outcome < 0.5 and source not in trace.skills_used:
                task_type = trace.task_type
                if task_type not in failures_by_task_type:
                    failures_by_task_type[task_type] = []
                failures_by_task_type[task_type].append(idx)

        if not failures_by_task_type:
            return 0.0

        # Phase 2: For each recovery trace (target succeeded with source),
        # check if there is an earlier failure on the same task_type.
        # A failure can only be "repaired" once (greedy matching).
        repaired_failures: set[tuple[str, int]] = set()  # (task_type, failure_idx)

        for idx, trace in enumerate(sorted_traces):
            if target not in trace.skills_used:
                continue
            if trace.outcome >= 0.5 and source in trace.skills_used:
                task_type = trace.task_type
                if task_type not in failures_by_task_type:
                    continue
                # Find the earliest unrepaired failure before this trace
                for fail_idx in failures_by_task_type[task_type]:
                    if fail_idx >= idx:
                        break  # No earlier failures remain
                    key = (task_type, fail_idx)
                    if key not in repaired_failures:
                        repaired_failures.add(key)
                        break  # One repair per recovery trace

        # Total number of target failures without source
        total_failures = sum(len(idxs) for idxs in failures_by_task_type.values())

        if total_failures == 0:
            return 0.0

        repair_rate = len(repaired_failures) / total_failures
        return max(0.0, min(1.0, repair_rate))


class LLMJudgeSignal:
    """Uses an LLM to judge causal dependency strength between two skills.

    Imports prompts from caskg.causal.prompts and calls llm_service.send_message.
    Raises on error — callers must handle failures explicitly rather than
    receiving a misleading neutral score.
    """

    async def compute(
        self,
        node_i: SkillNode,
        node_j: SkillNode,
        llm_service: Any,
    ) -> float:
        """Query an LLM judge for causal dependency strength.

        Args:
            node_i: Source skill node.
            node_j: Target skill node.
            llm_service: Service with a send_message(prompt, system_prompt) method.

        Returns:
            LLM-judged causal score in [0, 1].

        Raises:
            RuntimeError: If the LLM call fails (network, auth, timeout).
        """
        from caskg.causal.prompts import CAUSAL_LLM_JUDGE_SYSTEM, CAUSAL_LLM_JUDGE_PROMPT

        source_description = node_i.to_str()
        target_description = node_j.to_str()

        prompt = CAUSAL_LLM_JUDGE_PROMPT.format(
            source_skill=source_description,
            target_skill=target_description,
        )

        try:
            response = await llm_service.send_message(
                prompt=prompt,
                system_prompt=CAUSAL_LLM_JUDGE_SYSTEM,
            )

            # send_message returns (content, history); unpack the content.
            if isinstance(response, tuple):
                response = response[0]

            score = self._parse_score(response)
            return max(0.0, min(1.0, score))
        except Exception as exc:
            raise RuntimeError(
                f"LLM judge failed for edge {node_i.name} -> {node_j.name}: {exc}"
            ) from exc

    @staticmethod
    def _parse_score(response: Any) -> float:
        """Parse a float score from the LLM response.

        Handles both raw string responses and structured responses.
        Tries to parse JSON first, then falls back to regex extraction.
        """
        text = str(response) if not isinstance(response, str) else response

        # Try JSON parsing first
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "score" in data:
                return float(data["score"])
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Try to find JSON embedded in the text
        json_match = re.search(
            r"\{[^}]*\"score\"\s*:\s*([0-9]*\.?[0-9]+)[^}]*\}", text
        )
        if json_match:
            try:
                return float(json_match.group(1))
            except ValueError:
                pass

        # Fall back to finding any float at end of response
        float_match = re.search(r"(?:^|\s)([01](?:\.\d+)?)\s*$", text)
        if float_match:
            try:
                return float(float_match.group(1))
            except ValueError:
                pass

        # Last resort: find any number between 0 and 1
        numbers = re.findall(r"(\d+\.?\d*)", text)
        for num_str in numbers:
            try:
                val = float(num_str)
                if 0.0 <= val <= 1.0:
                    return val
            except ValueError:
                continue

        return 0.5


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------


def compute_association_score(
    signals: dict[str, float],
    weights: dict[str, float],
) -> float:
    """Compute a weighted linear combination of signal scores.

    Args:
        signals: Mapping of signal name -> score (each in [0, 1]).
        weights: Mapping of signal name -> weight. Keys should match
            signal names. The result is normalized by total weight.

    Returns:
        Weighted combination clamped to [0, 1].
    """
    total_weight = 0.0
    weighted_sum = 0.0

    for name, score in signals.items():
        weight = weights.get(name, 0.0)
        weighted_sum += score * weight
        total_weight += weight

    if total_weight == 0.0:
        return 0.0

    raw_score = weighted_sum / total_weight
    return max(0.0, min(1.0, raw_score))
