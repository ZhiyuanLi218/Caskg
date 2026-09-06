"""Multi-signal candidate graph inducer for CaSKG.

Generates high-recall candidate edges between skill nodes using a combination
of lexical overlap, IO compatibility, co-occurrence traces, semantic similarity,
repair patterns, and LLM judge signals. Each candidate edge receives an
association score computed as a weighted combination of these signals.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

from caskg.causal.signals import (
    CooccurrenceSignal,
    IOCompatibilitySignal,
    LLMJudgeSignal,
    LexicalSignal,
    RepairSignal,
    SemanticSignal,
    compute_association_score,
)
from caskg.causal.trace_store import TraceStore
from caskg.core.schema import SkillEdge, SkillNode

logger = logging.getLogger(__name__)


class CandidateGraphInducer:
    """Induces candidate causal edges from multi-signal evidence.

    The inducer follows a two-phase approach:
    1. Generate a high-recall set of candidate pairs using fast heuristics
       (lexical overlap, IO compatibility, co-occurrence, existing edges).
    2. Score each candidate pair with all available signals and compute a
       weighted association score.

    Parameters
    ----------
    trace_store : TraceStore
        Execution trace store for co-occurrence and repair signals.
    config : Any
        Configuration object with weight attributes:
        LAMBDA_SEM, LAMBDA_LEX, LAMBDA_IO, LAMBDA_COOCCUR,
        LAMBDA_REPAIR, LAMBDA_LLM_JUDGE.
    """

    def __init__(self, trace_store: TraceStore, config: Any) -> None:
        self.trace_store = trace_store
        self.config = config
        self.semantic_signal = SemanticSignal()
        self.lexical_signal = LexicalSignal()
        self.io_signal = IOCompatibilitySignal()
        self.cooccur_signal = CooccurrenceSignal()
        self.repair_signal = RepairSignal()
        self.llm_judge_signal = LLMJudgeSignal()
        # Candidate pairs are scored concurrently, but provider RPM must be
        # enforced globally. The semaphore caps in-flight calls, while
        # `_wait_for_judge_slot()` spaces call starts across all workers.
        sem_limit = getattr(self.config, "JUDGE_SEMAPHORE", 1)
        self._judge_semaphore = asyncio.Semaphore(max(sem_limit, 1))
        self._judge_call_delay = getattr(self.config, "JUDGE_CALL_DELAY", 0.6)
        self._judge_rate_lock = asyncio.Lock()
        self._next_judge_call_at = 0.0
        logger.info(
            "Judge config: semaphore=%d, call_delay=%.2fs",
            sem_limit, self._judge_call_delay,
        )
        # name -> embedding vector, populated once per induce_candidates run.
        self._embedding_cache: dict[str, list[float]] = {}

    async def induce_candidates(
        self,
        nodes: list[SkillNode],
        existing_edges: list[SkillEdge],
        embedding_service: Any = None,
        llm_service: Any = None,
        max_candidates_per_node: int = 12,
    ) -> list[SkillEdge]:
        """Induce candidate edges with multi-signal association scores.

        Parameters
        ----------
        nodes : list[SkillNode]
            All skill nodes in the graph.
        existing_edges : list[SkillEdge]
            Current edges (included for re-evaluation).
        embedding_service : optional
            Async embedding service with an ``embed(text) -> list[float]`` method.
        llm_service : optional
            Async LLM service with a ``judge_edge(source, target) -> float`` method.
        max_candidates_per_node : int
            Maximum candidate neighbors per source node.

        Returns
        -------
        list[SkillEdge]
            Candidate edges with status="unverified" and association_score set.
        """
        if not nodes:
            return []

        # Build name -> node lookup
        node_map: dict[str, SkillNode] = {node.name: node for node in nodes}

        # Precompute node embeddings ONCE (cached by name) BEFORE generating
        # candidate pairs, so semantic recall can use them. Without a single
        # batched pass, the semantic signal would re-encode both endpoints of
        # every candidate edge — thousands of concurrent calls that trip the
        # provider RPM limit (429). One batched encode (~N calls) populates the
        # cache that both semantic recall and the semantic score read from.
        embedding_cache: dict[str, list[float]] = {}
        if embedding_service is not None:
            texts = [f"{n.name}: {n.description}" for n in nodes]
            try:
                vectors = await embedding_service.encode(texts)
                for n, vec in zip(nodes, vectors):
                    # encode() may return numpy arrays; coerce to plain lists so
                    # downstream truthiness checks (`if not vec`) and cosine math
                    # don't hit "truth value of an array is ambiguous".
                    embedding_cache[n.name] = (
                        vec.tolist() if hasattr(vec, "tolist") else list(vec)
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Node embedding precomputation failed (%s); semantic "
                    "signal will be skipped for this run.",
                    exc,
                )
        self._embedding_cache = embedding_cache

        # Phase 1: Generate high-recall candidate pairs (uses the cache for
        # semantic recall — pairs that are semantically related but share no
        # name tokens / IO schema would otherwise never enter the candidate pool).
        candidate_pairs = self._generate_candidate_pairs(
            nodes, existing_edges, max_per_node=max_candidates_per_node
        )

        if not candidate_pairs:
            return []

        # Precompute co-occurrence matrix and traces for batch efficiency
        skill_names = [n.name for n in nodes]
        cooccurrence_counts = self.trace_store.cooccurrence_matrix(skill_names)
        max_cooccur = max(cooccurrence_counts.values()) if cooccurrence_counts else 1
        recent_traces = self.trace_store.recent(n=200)


        candidate_edges: list[SkillEdge] = []

        # Resume support: load previously scored pairs from checkpoint file
        import json
        from pathlib import Path

        checkpoint_path = getattr(self, "_checkpoint_path", None)
        scored_pairs: set[tuple[str, str]] = set()
        if checkpoint_path and Path(checkpoint_path).exists():
            with open(checkpoint_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        pair = (rec["source"], rec["target"])
                        scored_pairs.add(pair)
                        candidate_edges.append(SkillEdge(
                            source=rec["source"],
                            target=rec["target"],
                            description=rec.get("description", ""),
                            type=rec.get("type", "semantic"),
                            weight=rec.get("weight", 0.0),
                            confidence=rec.get("confidence", 0.0),
                            association_score=rec.get("association_score", 0.0),
                            causal_score=0.0,
                            status="unverified",
                            uncertainty=1.0,
                        ))
                    except (json.JSONDecodeError, KeyError):
                        continue
            logger.info(
                "Resumed from checkpoint: %d pairs already scored", len(scored_pairs)
            )

        # Filter out already-scored pairs
        remaining_pairs = [
            p for p in candidate_pairs if p not in scored_pairs
        ]

        total_pairs = len(candidate_pairs)
        total_remaining = len(remaining_pairs)
        judge_calls = 0
        logger.info(
            "Scoring %d candidate pairs (%d resumed, %d remaining)",
            total_pairs, total_pairs - total_remaining, total_remaining,
        )

        checkpoint_file = None
        if checkpoint_path:
            checkpoint_file = open(checkpoint_path, "a")

        # ── Phase A: Pre-score ALL remaining pairs in parallel ─────────────
        # Computes sem/lex/io/cooccur/repair signals synchronously (cache hits).
        # LLM judge is deferred to Phase B so all judge calls can be issued
        # concurrently instead of being gated behind 20-pair batches.
        prescore_tasks = [
            self._prescore_pair(
                src_name, tgt_name, node_map,
                cooccurrence_counts, max_cooccur, recent_traces,
                embedding_service,
            )
            for src_name, tgt_name in remaining_pairs
        ]
        prescore_results = await asyncio.gather(*prescore_tasks, return_exceptions=True)

        prescored: list[dict] = []
        for (src_name, tgt_name), result in zip(remaining_pairs, prescore_results):
            if isinstance(result, Exception):
                logger.warning("Prescore failed for (%s, %s): %s", src_name, tgt_name, result)
                continue
            if result is not None:
                prescored.append(result)

        judge_needed = [p for p in prescored if p["needs_judge"]]
        direct = [p for p in prescored if not p["needs_judge"]]
        logger.info(
            "Phase A complete: %d pre-scored (%d need LLM judge, %d direct)",
            len(prescored), len(judge_needed), len(direct),
        )

        # ── Phase B.1: Finalize direct pairs (no LLM needed) ───────────────
        edges_to_save: list[tuple[SkillEdge, bool]] = []
        for pre in direct:
            edges_to_save.append(self._build_edge(pre, llm_judge_score=0.0, judge_called=False))

        # ── Write direct results first ──────────────────────────────────────
        import threading
        _write_lock = threading.Lock()
        done_total = [total_pairs - total_remaining]  # mutable for closure

        def _write_result(edge: SkillEdge, jc: bool) -> None:
            nonlocal judge_calls
            if jc:
                judge_calls += 1
            candidate_edges.append(edge)
            done_total[0] += 1
            if checkpoint_file:
                with _write_lock:
                    checkpoint_file.write(json.dumps({
                        "source": edge.source,
                        "target": edge.target,
                        "type": edge.type,
                        "weight": edge.weight,
                        "confidence": edge.confidence,
                        "association_score": edge.association_score,
                        "description": edge.description,
                    }) + "\n")
                    if done_total[0] % 10 == 0:
                        checkpoint_file.flush()
            if done_total[0] % 10 == 0:
                logger.info(
                    "Progress: %d/%d pairs scored (%.1f%%), judge_calls=%d",
                    done_total[0], total_pairs,
                    done_total[0] / total_pairs * 100 if total_pairs else 0,
                    judge_calls,
                )

        for pre in direct:
            edge, jc = self._build_edge(pre, llm_judge_score=0.0, judge_called=False)
            _write_result(edge, jc)

        # ── Phase B.2: Run LLM judge concurrently, write as each completes ─
        # asyncio.as_completed so progress is reported incrementally.
        if judge_needed:
            async def _judge_one(pre: dict) -> tuple:
                try:
                    return await self._judge_and_build(pre, llm_service)
                except Exception as exc:
                    logger.warning(
                        "Judge failed for (%s, %s): %s",
                        pre["src_name"], pre["tgt_name"], exc,
                    )
                    return self._build_edge(pre, llm_judge_score=0.0, judge_called=False)

            judge_tasks = [_judge_one(pre) for pre in judge_needed]
            for coro in asyncio.as_completed(judge_tasks):
                edge, jc = await coro
                _write_result(edge, jc)

        if checkpoint_file:
            checkpoint_file.flush()

        if checkpoint_file:
            checkpoint_file.close()

        return candidate_edges

    async def _prescore_pair(
        self,
        src_name: str,
        tgt_name: str,
        node_map: dict[str, SkillNode],
        cooccurrence_counts: dict[tuple[str, str], int],
        max_cooccur: int,
        recent_traces: list[Any],
        embedding_service: Any = None,
    ) -> dict | None:
        """Compute all non-LLM signals for a pair. Returns a dict with all
        intermediate state so `_judge_and_build` / `_build_edge` can finalise
        without recomputing anything."""
        source = node_map.get(src_name)
        target = node_map.get(tgt_name)
        if source is None or target is None:
            return None

        lex_score = self.lexical_signal.compute(source, target)
        io_score = self.io_signal.compute(source, target)
        cooccur_score = self.cooccur_signal.compute(src_name, tgt_name, self.trace_store)
        repair_score = self.repair_signal.compute(src_name, tgt_name, self.trace_store)
        structural_score = self._structural_compatibility_score(source, target)

        vec_i = self._embedding_cache.get(src_name)
        vec_j = self._embedding_cache.get(tgt_name)
        if vec_i is not None and vec_j is not None:
            sem_score = self.semantic_signal._cosine_similarity(vec_i, vec_j)
        else:
            sem_score = 0.0

        has_traces = bool(getattr(self.trace_store, "traces", None)) or bool(recent_traces)

        base_weights = {
            "sem": getattr(self.config, "LAMBDA_SEM", 0.25),
            "lex": getattr(self.config, "LAMBDA_LEX", 0.10),
            "io": getattr(self.config, "LAMBDA_IO", 0.25),
            "cooccur": getattr(self.config, "LAMBDA_COOCCUR", 0.15),
            "repair": getattr(self.config, "LAMBDA_REPAIR", 0.10),
            "structural": getattr(self.config, "LAMBDA_STRUCTURAL", 0.20),
            "llm_judge": getattr(self.config, "LAMBDA_LLM_JUDGE", 0.15),
        }

        def _active(signal_keys: list[str]) -> dict[str, float]:
            out = {}
            for k in signal_keys:
                if k in ("cooccur", "repair") and not has_traces:
                    continue
                out[k] = base_weights[k]
            return out

        preliminary_signals = {
            "sem": sem_score,
            "lex": lex_score,
            "io": io_score,
            "cooccur": cooccur_score,
            "repair": repair_score,
            "structural": structural_score,
        }
        preliminary_weights = _active(
            ["sem", "lex", "io", "cooccur", "repair", "structural"]
        )
        preliminary_signals = {
            k: v for k, v in preliminary_signals.items() if k in preliminary_weights
        }
        preliminary_a_ij = compute_association_score(preliminary_signals, preliminary_weights)

        llm_judge_gate = getattr(self.config, "LLM_JUDGE_GATE", 0.4)
        needs_judge = preliminary_a_ij > llm_judge_gate

        return {
            "src_name": src_name,
            "tgt_name": tgt_name,
            "source": source,
            "target": target,
            "sem": sem_score,
            "lex": lex_score,
            "io": io_score,
            "cooccur": cooccur_score,
            "repair": repair_score,
            "structural": structural_score,
            "base_weights": base_weights,
            "has_traces": has_traces,
            "preliminary_a_ij": preliminary_a_ij,
            "needs_judge": needs_judge,
        }

    async def _judge_and_build(self, pre: dict, llm_service: Any) -> tuple:
        """Run LLM judge (with semaphore rate-limit), then build the edge."""
        source = pre["source"]
        target = pre["target"]
        llm_judge_score, judge_called = await self._compute_llm_judge(
            source, target, llm_service
        )
        return self._build_edge(pre, llm_judge_score, judge_called)

    async def _wait_for_judge_slot(self) -> None:
        """Reserve the next globally rate-limited LLM judge start time."""
        delay = max(float(self._judge_call_delay), 0.0)
        if delay == 0.0:
            return

        async with self._judge_rate_lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            wait_seconds = self._next_judge_call_at - now
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
                now = loop.time()
            self._next_judge_call_at = max(self._next_judge_call_at, now) + delay

    async def _compute_llm_judge(
        self,
        source: SkillNode,
        target: SkillNode,
        llm_service: Any,
    ) -> tuple[float, bool]:
        """Compute one LLM judge score under concurrency and global RPM limits."""
        async with self._judge_semaphore:
            await self._wait_for_judge_slot()
            try:
                return await self.llm_judge_signal.compute(
                    source, target, llm_service
                ), True
            except RuntimeError:
                return 0.0, False

    def _build_edge(self, pre: dict, llm_judge_score: float, judge_called: bool) -> tuple:
        """Assemble a SkillEdge from pre-scored signals + optional LLM judge score.
        Pure computation — no async I/O."""
        base_weights = pre["base_weights"]
        has_traces = pre["has_traces"]

        def _active(signal_keys: list[str]) -> dict[str, float]:
            out = {}
            for k in signal_keys:
                if k in ("cooccur", "repair") and not has_traces:
                    continue
                out[k] = base_weights[k]
            return out

        signals = {
            "sem": pre["sem"],
            "lex": pre["lex"],
            "io": pre["io"],
            "cooccur": pre["cooccur"],
            "repair": pre["repair"],
            "structural": pre["structural"],
            "llm_judge": llm_judge_score,
        }
        active_keys = [
            "sem",
            "lex",
            "io",
            "cooccur",
            "repair",
            "structural",
            "llm_judge",
        ]
        weights = _active(active_keys)
        if not judge_called:
            weights.pop("llm_judge", None)
        signals = {k: v for k, v in signals.items() if k in weights}

        a_ij = compute_association_score(signals, weights)
        if pre["structural"] > 0.65:
            a_ij = max(a_ij, pre["structural"] * 0.7)
        relation_type = self._assign_relation_type(signals, a_ij)

        description = (
            f"Candidate edge: sem={pre['sem']:.2f} lex={pre['lex']:.2f} "
            f"io={pre['io']:.2f} cooccur={pre['cooccur']:.2f} "
            f"repair={pre['repair']:.2f} structural={pre['structural']:.2f} "
            f"llm={llm_judge_score:.2f}"
        )

        return SkillEdge(
            source=pre["src_name"],
            target=pre["tgt_name"],
            description=description,
            type=relation_type,
            weight=a_ij,
            confidence=a_ij,
            association_score=a_ij,
            causal_score=0.0,
            status="unverified",
            uncertainty=1.0,
        ), judge_called

    async def _score_pair(
        self,
        src_name: str,
        tgt_name: str,
        node_map: dict[str, SkillNode],
        cooccurrence_counts: dict[tuple[str, str], int],
        max_cooccur: int,
        recent_traces: list[Any],
        embedding_service: Any = None,
        llm_service: Any = None,
    ) -> SkillEdge | None:
        """Score a single candidate pair and return a SkillEdge or None."""
        source = node_map.get(src_name)
        target = node_map.get(tgt_name)
        if source is None or target is None:
            return None

        # Compute synchronous signals (correct method: .compute(), correct args)
        lex_score = self.lexical_signal.compute(source, target)
        io_score = self.io_signal.compute(source, target)
        cooccur_score = self.cooccur_signal.compute(
            src_name, tgt_name, self.trace_store
        )
        repair_score = self.repair_signal.compute(
            src_name, tgt_name, self.trace_store
        )
        structural_score = self._structural_compatibility_score(source, target)

        # Semantic signal: prefer the precomputed embedding cache (one encode
        # per node) to avoid re-encoding both endpoints of every candidate edge,
        # which would burst the provider RPM limit. Fall back to a live encode
        # only if a node is missing from the cache.
        vec_i = self._embedding_cache.get(src_name)
        vec_j = self._embedding_cache.get(tgt_name)
        if vec_i is not None and vec_j is not None:
            sem_score = self.semantic_signal._cosine_similarity(vec_i, vec_j)
        elif embedding_service is not None:
            sem_score = await self.semantic_signal.compute(
                source, target, embedding_service
            )
        else:
            sem_score = 0.0

        # Determine which trace-derived signals are actually active. In Phase 1
        # (candidate induction before any task execution) there are no traces, so
        # cooccur and repair are structurally 0. Including their weights in the
        # association-score denominator dilutes every genuine edge by ~25%. We
        # therefore drop inactive signals from BOTH the signal dict and the
        # weight dict so the score reflects only signals that could fire.
        has_traces = bool(getattr(self.trace_store, "traces", None)) or bool(
            recent_traces
        )

        base_weights = {
            "sem": getattr(self.config, "LAMBDA_SEM", 0.25),
            "lex": getattr(self.config, "LAMBDA_LEX", 0.10),
            "io": getattr(self.config, "LAMBDA_IO", 0.25),
            "cooccur": getattr(self.config, "LAMBDA_COOCCUR", 0.15),
            "repair": getattr(self.config, "LAMBDA_REPAIR", 0.10),
            "structural": getattr(self.config, "LAMBDA_STRUCTURAL", 0.20),
            "llm_judge": getattr(self.config, "LAMBDA_LLM_JUDGE", 0.15),
        }

        def _active(signal_keys: list[str]) -> dict[str, float]:
            """Weights for the given signals, dropping trace-signals when no traces exist."""
            out = {}
            for k in signal_keys:
                if k in ("cooccur", "repair") and not has_traces:
                    continue
                out[k] = base_weights[k]
            return out

        # Gate LLM judge: only invoke for top-priority pairs where the
        # preliminary association score (over active signals) exceeds the gate.
        # This avoids expensive LLM calls on low-quality pairs.
        preliminary_signals = {
            "sem": sem_score,
            "lex": lex_score,
            "io": io_score,
            "cooccur": cooccur_score,
            "repair": repair_score,
            "structural": structural_score,
        }
        preliminary_weights = _active(
            ["sem", "lex", "io", "cooccur", "repair", "structural"]
        )
        preliminary_signals = {
            k: v for k, v in preliminary_signals.items() if k in preliminary_weights
        }
        preliminary_a_ij = compute_association_score(
            preliminary_signals, preliminary_weights
        )

        llm_judge_gate = getattr(self.config, "LLM_JUDGE_GATE", 0.15)
        llm_judge_score = 0.0
        judge_called = False
        if preliminary_a_ij > llm_judge_gate and llm_service is not None:
            llm_judge_score, judge_called = await self._compute_llm_judge(
                source, target, llm_service
            )

        # Gather all signals
        signals = {
            "sem": sem_score,
            "lex": lex_score,
            "io": io_score,
            "cooccur": cooccur_score,
            "repair": repair_score,
            "structural": structural_score,
            "llm_judge": llm_judge_score,
        }

        # Active weights: drop dead trace-signals; drop llm_judge if it wasn't run.
        active_keys = [
            "sem",
            "lex",
            "io",
            "cooccur",
            "repair",
            "structural",
            "llm_judge",
        ]
        weights = _active(active_keys)
        if llm_service is None or llm_judge_score == 0.0:
            weights.pop("llm_judge", None)
        signals = {k: v for k, v in signals.items() if k in weights}

        a_ij = compute_association_score(signals, weights)
        if structural_score > 0.65:
            a_ij = max(a_ij, structural_score * 0.7)

        # Assign relation type based on signal heuristics
        relation_type = self._assign_relation_type(signals, a_ij)

        # Build description summarizing the signals
        description = (
            f"Candidate edge: sem={sem_score:.2f} lex={lex_score:.2f} "
            f"io={io_score:.2f} cooccur={cooccur_score:.2f} "
            f"repair={repair_score:.2f} structural={structural_score:.2f} "
            f"llm={llm_judge_score:.2f}"
        )

        return SkillEdge(
            source=src_name,
            target=tgt_name,
            description=description,
            type=relation_type,
            weight=a_ij,
            confidence=a_ij,
            association_score=a_ij,
            causal_score=0.0,
            status="unverified",
            uncertainty=1.0,
        ), judge_called

    def _generate_candidate_pairs(
        self,
        nodes: list[SkillNode],
        existing_edges: list[SkillEdge],
        max_per_node: int = 12,
    ) -> list[tuple[str, str]]:
        """Generate high-recall candidate pairs using multiple heuristics.

        Union of:
        - Lexical top-K per node (token overlap)
        - IO compatibility pairs (output->input matches)
        - Co-occurrence pairs from trace_store
        - Existing edges (for re-evaluation)

        Deduplicates and caps at max_per_node neighbors per source.

        Parameters
        ----------
        nodes : list[SkillNode]
            All skill nodes.
        existing_edges : list[SkillEdge]
            Current edges to include for re-evaluation.
        max_per_node : int
            Maximum candidate neighbors per source node.

        Returns
        -------
        list[tuple[str, str]]
            Deduplicated list of (source_name, target_name) candidate pairs.
        """
        if not nodes:
            return []

        # Track candidates per source for capping
        candidates_per_source: dict[str, set[str]] = defaultdict(set)
        all_pairs: set[tuple[str, str]] = set()

        node_map: dict[str, SkillNode] = {node.name: node for node in nodes}
        node_names = list(node_map.keys())

        # --- 1. Lexical top-K per node ---
        self._add_lexical_candidates(nodes, node_map, candidates_per_source, max_per_node)

        # --- 2. IO compatibility pairs ---
        self._add_io_candidates(nodes, candidates_per_source, max_per_node)

        # --- 3. Abstract capability-chain pairs ---
        self._add_structural_candidates(nodes, candidates_per_source, max_per_node)

        # --- 4. Co-occurrence pairs from trace_store ---
        self._add_cooccurrence_candidates(node_names, candidates_per_source, max_per_node)

        # --- 5. Semantic top-K per node (embedding cosine) ---
        # Recall pairs that are semantically related but share no name tokens or
        # IO schema — these would otherwise never enter the candidate pool. Uses
        # the precomputed embedding cache (no API calls).
        self._add_semantic_candidates(node_names, candidates_per_source, max_per_node)

        # --- 6. Existing edges (for re-evaluation) ---
        existing_pairs: set[tuple[str, str]] = set()
        for edge in existing_edges:
            if edge.source in node_map and edge.target in node_map:
                candidates_per_source[edge.source].add(edge.target)
                existing_pairs.add((edge.source, edge.target))

        # Flatten into deduplicated pair list. Each recall method self-budgets,
        # but their union can still overrun the requested per-source frontier.
        # Apply one deterministic, evidence-ranked cap at the end instead of
        # letting broad recall channels dominate the validation queue.
        final_cap = max(1, max_per_node)
        for source_name, targets in candidates_per_source.items():
            source = node_map.get(source_name)
            if source is None:
                continue
            ranked_targets = sorted(
                targets,
                key=lambda target_name: self._candidate_pair_priority(
                    source,
                    node_map.get(target_name),
                    (source_name, target_name) in existing_pairs,
                ),
                reverse=True,
            )
            for target_name in ranked_targets[:final_cap]:
                if source_name != target_name:
                    all_pairs.add((source_name, target_name))

        return list(all_pairs)

    def _candidate_pair_priority(
        self,
        source: SkillNode,
        target: SkillNode | None,
        is_existing: bool = False,
    ) -> tuple[float, float, float, float, float, float, str]:
        """Rank merged recall candidates before the final per-source cap."""
        if target is None:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "")

        vec_i = self._embedding_cache.get(source.name)
        vec_j = self._embedding_cache.get(target.name)
        sem_score = (
            self.semantic_signal._cosine_similarity(vec_i, vec_j)
            if vec_i is not None and vec_j is not None
            else 0.0
        )
        return (
            1.0 if is_existing else 0.0,
            self._structural_compatibility_score(source, target),
            self.io_signal.compute(source, target),
            sem_score,
            self.lexical_signal.compute(source, target),
            self._structural_pair_affinity(source, target),
            target.name,
        )

    def _add_lexical_candidates(
        self,
        nodes: list[SkillNode],
        node_map: dict[str, SkillNode],
        candidates_per_source: dict[str, set[str]],
        max_per_node: int,
    ) -> None:
        """Add top-K lexical overlap candidates per node."""
        import re

        def tokenize(text: str) -> set[str]:
            return set(re.findall(r"[a-z0-9]+", text.lower()))

        # Precompute token sets
        token_sets: dict[str, set[str]] = {}
        for node in nodes:
            text = (
                f"{node.name} {node.description} {node.one_line_capability} "
                f"{node.inputs} {node.outputs} {node.domain_tags}"
            )
            token_sets[node.name] = tokenize(text)

        # For each node, find top-K by Jaccard
        top_k = min(max_per_node // 2, 6)  # Use half budget for lexical
        for src_name, src_tokens in token_sets.items():
            if not src_tokens:
                continue
            scores: list[tuple[str, float]] = []
            for tgt_name, tgt_tokens in token_sets.items():
                if tgt_name == src_name or not tgt_tokens:
                    continue
                intersection = src_tokens & tgt_tokens
                union = src_tokens | tgt_tokens
                jaccard = len(intersection) / len(union) if union else 0.0
                if jaccard > 0.05:  # Minimum threshold
                    scores.append((tgt_name, jaccard))

            scores.sort(key=lambda x: x[1], reverse=True)
            for tgt_name, _ in scores[:top_k]:
                candidates_per_source[src_name].add(tgt_name)

    def _add_io_candidates(
        self,
        nodes: list[SkillNode],
        candidates_per_source: dict[str, set[str]],
        max_per_node: int,
    ) -> None:
        """Add pairs where source outputs match target inputs."""
        import re

        def tokenize(text: str) -> set[str]:
            return set(re.findall(r"[a-z0-9]+", text.lower()))

        # Build output -> source mapping
        output_index: dict[str, list[str]] = defaultdict(list)
        for node in nodes:
            output_tokens = tokenize(" ".join(node.output_types))
            for token in output_tokens:
                if len(token) > 2:  # Skip very short tokens
                    output_index[token].append(node.name)

        # For each node, find sources whose outputs match its inputs
        for node in nodes:
            input_tokens = tokenize(" ".join(node.input_types))
            precondition_tokens = tokenize(node.preconditions)
            lookup_tokens = input_tokens | precondition_tokens

            matched_sources: dict[str, int] = defaultdict(int)
            for token in lookup_tokens:
                if token in output_index:
                    for src_name in output_index[token]:
                        if src_name != node.name:
                            matched_sources[src_name] += 1

            # Add top matches as candidates (source -> target direction)
            sorted_sources = sorted(
                matched_sources.items(), key=lambda x: x[1], reverse=True
            )
            io_budget = max_per_node // 3
            for src_name, _ in sorted_sources[:io_budget]:
                candidates_per_source[src_name].add(node.name)

    def _add_structural_candidates(
        self,
        nodes: list[SkillNode],
        candidates_per_source: dict[str, set[str]],
        max_per_node: int,
    ) -> None:
        """Add candidate edges from abstract capability-chain roles.

        The role taxonomy is deliberately domain-agnostic: it models broad
        workflow phases (parse, context, locate, acquire, transform, deliver,
        verify) rather than mapping benchmark task words to specific skills.
        Retrieval later consumes only the validated graph topology.
        """
        role_index: dict[str, list[SkillNode]] = defaultdict(list)
        for node in nodes:
            for role in self._node_roles(node):
                role_index[role].append(node)

        role_edges = [
            ("task_parse", "context_scan"),
            ("task_parse", "target_locate"),
            ("context_scan", "target_locate"),
            ("target_locate", "acquire_inventory"),
            ("target_locate", "tool_device"),
            ("target_locate", "state_transform"),
            ("target_locate", "place_output"),
            ("acquire_inventory", "tool_device"),
            ("acquire_inventory", "state_transform"),
            ("acquire_inventory", "place_output"),
            ("tool_device", "state_transform"),
            ("tool_device", "verify_inspect"),
            ("state_transform", "place_output"),
            ("state_transform", "verify_inspect"),
            ("place_output", "verify_inspect"),
        ]

        structural_budget = max(1, max_per_node // 3)
        for source_role, target_role in role_edges:
            for source in role_index.get(source_role, []):
                added = 0
                budget = structural_budget
                if target_role == "verify_inspect":
                    budget = max(structural_budget, max_per_node // 2)
                scored_targets = []
                for target in role_index.get(target_role, []):
                    if source.name == target.name:
                        continue
                    affinity = self._structural_pair_affinity(source, target)
                    same_namespace = self._same_namespace(source.name, target.name)
                    min_affinity = 0.12 if same_namespace else 0.28
                    if affinity >= min_affinity:
                        scored_targets.append((target, affinity, same_namespace))

                targets = [
                    target
                    for target, _, _ in sorted(
                        scored_targets,
                        key=lambda item: (item[2], item[1], item[0].name),
                        reverse=True,
                    )
                ]
                for target in targets:
                    candidates_per_source[source.name].add(target.name)
                    added += 1
                    if added >= budget:
                        break

    @staticmethod
    def _same_namespace(source_name: str, target_name: str) -> bool:
        source_namespace = CandidateGraphInducer._skill_namespace(source_name)
        target_namespace = CandidateGraphInducer._skill_namespace(target_name)
        return bool(source_namespace and target_namespace and source_namespace == target_namespace)

    @staticmethod
    def _skill_namespace(name: str) -> str:
        prefix = (name or "").split("-", 1)[0]
        return prefix if len(prefix) >= 3 else ""

    def _add_cooccurrence_candidates(
        self,
        skill_names: list[str],
        candidates_per_source: dict[str, set[str]],
        max_per_node: int,
    ) -> None:
        """Add pairs that frequently co-occur in execution traces."""
        cooccurrence_counts = self.trace_store.cooccurrence_matrix(skill_names)
        if not cooccurrence_counts:
            return

        # Group by each skill and pick top co-occurring partners
        cooccur_per_skill: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for (skill_a, skill_b), count in cooccurrence_counts.items():
            cooccur_per_skill[skill_a].append((skill_b, count))
            cooccur_per_skill[skill_b].append((skill_a, count))

        cooccur_budget = max_per_node // 3
        for skill, partners in cooccur_per_skill.items():
            partners.sort(key=lambda x: x[1], reverse=True)
            for partner, _ in partners[:cooccur_budget]:
                candidates_per_source[skill].add(partner)

    def _add_semantic_candidates(
        self,
        skill_names: list[str],
        candidates_per_source: dict[str, set[str]],
        max_per_node: int,
    ) -> None:
        """Add top-K semantically-similar candidates per node via embedding cosine.

        Reads vectors from the precomputed ``self._embedding_cache`` (no API
        calls). This recalls genuinely related skills that share no name tokens
        or IO schema, which lexical/IO recall alone would miss. A minimum cosine
        threshold avoids flooding the pool with weak matches.
        """
        cache = self._embedding_cache
        if not cache:
            return

        import numpy as np

        # Only consider nodes with usable vectors. Normalize each row with a
        # scale-invariant norm so malformed or extreme provider responses cannot
        # overflow the full cosine matrix.
        names: list[str] = []
        normalized_rows: list[np.ndarray] = []
        expected_dim: int | None = None
        skipped = 0
        for name in skill_names:
            if name not in cache:
                continue
            vec = np.asarray(cache[name], dtype=np.float64)
            if vec.ndim != 1 or vec.size == 0:
                skipped += 1
                continue
            if expected_dim is None:
                expected_dim = int(vec.size)
            elif int(vec.size) != expected_dim:
                skipped += 1
                continue

            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
            max_abs = float(np.max(np.abs(vec))) if vec.size else 0.0
            if not np.isfinite(max_abs) or max_abs <= 0.0:
                skipped += 1
                continue
            scaled = vec / max_abs
            scaled_norm = float(np.linalg.norm(scaled))
            if not np.isfinite(scaled_norm) or scaled_norm <= 0.0:
                skipped += 1
                continue
            unit = scaled / scaled_norm
            unit = np.nan_to_num(unit, nan=0.0, posinf=0.0, neginf=0.0)
            np.clip(unit, -1.0, 1.0, out=unit)
            names.append(name)
            normalized_rows.append(unit)

        if len(names) < 2:
            if skipped:
                logger.warning("Skipped %d invalid semantic candidate vector(s)", skipped)
            return

        if skipped:
            logger.warning("Skipped %d invalid semantic candidate vector(s)", skipped)

        normalized = np.vstack(normalized_rows).astype(np.float64, copy=False)
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(normalized, -1.0, 1.0, out=normalized)
        # Full cosine similarity matrix (N x N). N is the skill-set size
        # (<=2000), so this is a cheap one-shot matmul, not per-edge work.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = normalized @ normalized.T
        sims = np.nan_to_num(sims, nan=-1.0, posinf=1.0, neginf=-1.0)
        np.clip(sims, -1.0, 1.0, out=sims)

        sem_budget = max(1, max_per_node // 3)
        sem_threshold = 0.55  # cosine; below this the pair is not "related"
        for i, src_name in enumerate(names):
            row = sims[i]
            # Rank partners by similarity, skipping self.
            order = np.argsort(row)[::-1]
            added = 0
            for j in order:
                if int(j) == i:
                    continue
                if row[j] < sem_threshold:
                    break  # sorted descending — nothing better remains
                candidates_per_source[src_name].add(names[int(j)])
                added += 1
                if added >= sem_budget:
                    break

    def _assign_relation_type(self, signals: dict[str, float], a_ij: float) -> str:
        """Assign a relation type based on signal heuristics.

        Priority:
          1. io_score > 0.6                                  → "prereq"
          2. llm_judge > 0.5 (LLM: removing source hurts target) → "prereq"
          3. structural > 0.65                                → "workflow"
          4. cooccur > 0.5 and io > 0.3                       → "enhance"
          4. sem > 0.7 and llm_judge > 0.3                    → "dependency"
          5. sem > 0.85                                        → "similar"
          6. sem > 0.7 and llm_judge < 0.3                    → "similar"
          7. sem > 0.6                                         → "alternative"
          8. cooccur > 0.4                                     → "cooccur"
          9. default                                           → "semantic"

        This fixes the root cause where all skills have empty I/O fields and
        no execution traces in Phase 1, causing every edge to fall through to
        sem_score > 0.7 → "similar". The LLM judge signal (which IS computed
        during Phase 1) now gets used to distinguish prerequisite/dependency
        edges from mere similarity.
        """
        io_score = signals.get("io", 0.0)
        cooccur_score = signals.get("cooccur", 0.0)
        sem_score = signals.get("sem", 0.0)
        llm_judge = signals.get("llm_judge", 0.0)
        structural_score = signals.get("structural", 0.0)

        if io_score > 0.6:
            return "prereq"

        if llm_judge > 0.5:
            return "prereq"

        if structural_score > 0.65:
            return "workflow"

        if cooccur_score > 0.5 and io_score > 0.3:
            return "enhance"

        if sem_score > 0.7 and llm_judge > 0.3:
            return "dependency"

        if sem_score > 0.85:
            return "similar"

        if sem_score > 0.7 and llm_judge < 0.3:
            return "similar"

        if sem_score > 0.6:
            return "alternative"

        if cooccur_score > 0.4:
            return "cooccur"

        return "semantic"

    @classmethod
    def _structural_compatibility_score(
        cls,
        source: SkillNode,
        target: SkillNode,
    ) -> float:
        source_roles = cls._node_roles(source)
        target_roles = cls._node_roles(target)
        if not source_roles or not target_roles:
            return 0.0

        pairs = cls._role_pairs(source_roles, target_roles)
        strong_edges = {
            ("task_parse", "context_scan"),
            ("task_parse", "target_locate"),
            ("context_scan", "target_locate"),
            ("target_locate", "acquire_inventory"),
            ("target_locate", "tool_device"),
            ("target_locate", "state_transform"),
            ("target_locate", "place_output"),
            ("acquire_inventory", "tool_device"),
            ("acquire_inventory", "state_transform"),
            ("acquire_inventory", "place_output"),
            ("tool_device", "state_transform"),
            ("state_transform", "place_output"),
        }
        support_edges = {
            ("tool_device", "verify_inspect"),
            ("state_transform", "verify_inspect"),
            ("place_output", "verify_inspect"),
        }
        base_score = 0.0
        if pairs & strong_edges:
            base_score = 0.78
        elif pairs & support_edges:
            base_score = 0.68
        if base_score == 0.0:
            return 0.0

        affinity = cls._structural_pair_affinity(source, target)
        same_namespace = cls._same_namespace(source.name, target.name)
        min_affinity = 0.12 if same_namespace else 0.28
        if affinity < min_affinity:
            return 0.0
        return base_score

    @staticmethod
    def _role_pairs(
        source_roles: set[str],
        target_roles: set[str],
    ) -> set[tuple[str, str]]:
        return {
            (source_role, target_role)
            for source_role in source_roles
            for target_role in target_roles
        }

    @classmethod
    def _node_roles(cls, node: SkillNode) -> set[str]:
        explicit_roles = cls._metadata_roles(node)
        if explicit_roles:
            return explicit_roles

        facets = cls._role_facets(node)
        identity = cls._sanitize_role_text(facets["identity"])
        contract = cls._sanitize_role_text(facets["contract"])
        tooling = cls._sanitize_role_text(facets["tooling"])
        text = " ".join((identity, contract, tooling))
        roles: set[str] = set()
        if cls._contains_any(
            text,
            {
                "parse",
                "parser",
                "planner",
                "plan",
                "decompose",
                "intent",
                "goal",
                "objective",
                "orchestrate",
                "sequence",
            },
        ):
            roles.add("task_parse")
        if cls._contains_any(
            text,
            {
                "context",
                "scan",
                "scanner",
                "observe",
                "observation",
                "explore",
                "survey",
                "navigation",
                "environment",
                "evidence",
                "snapshot",
            },
        ):
            roles.add("context_scan")
        if cls._contains_any(
            text,
            {
                "locate",
                "locator",
                "find",
                "finder",
                "search",
                "entity",
                "identifier",
                "select",
                "match",
            },
        ):
            roles.add("target_locate")
        if cls._contains_any(
            text,
            {
                "acquire",
                "collect",
                "gather",
                "fetch",
                "take",
                "load",
                "ingest",
                "resource",
                "inventory",
                "held",
            },
        ):
            roles.add("acquire_inventory")
        if cls._contains_any(
            text,
            {
                "deliver",
                "delivery",
                "output",
                "artifact",
                "export",
                "write",
                "save",
                "store",
                "publish",
                "sink",
                "destination",
                "deposit",
            },
        ):
            roles.add("place_output")
        if cls._contains_any(
            text,
            {
                "state",
                "transform",
                "modify",
                "change",
                "convert",
                "normalize",
                "update",
                "edit",
                "clean",
                "cool",
                "heat",
                "temperature",
            },
        ):
            roles.add("state_transform")
        if cls._contains_any(
            f"{tooling} {identity}",
            {
                "tool",
                "device",
                "operator",
                "appliance",
                "instrument",
                "equipment",
                "actuator",
                "interface",
                "api",
                "client",
                "service",
                "executor",
                "connector",
                "mcp",
            },
        ):
            roles.add("tool_device")
        if cls._contains_any(
            text,
            {
                "inspect",
                "inspector",
                "examine",
                "verify",
                "check",
                "validate",
                "test",
                "assert",
                "evaluate",
                "quality",
                "audit",
                "compare",
            },
        ):
            roles.add("verify_inspect")
        if (
            "verify_inspect" in roles
            and not roles
            & {
                "target_locate",
                "acquire_inventory",
                "place_output",
                "state_transform",
                "tool_device",
            }
        ):
            roles.discard("context_scan")
            roles.discard("task_parse")
        return roles

    @classmethod
    def _role_facets(cls, node: SkillNode) -> dict[str, str]:
        """Build abstract role evidence buckets from reusable skill metadata.

        This keeps structural candidate induction tied to workflow theory
        (identity, data contract, tools, state effects) instead of benchmark
        task strings. The token sets in `_node_roles` are broad role lexicons;
        query-specific retrieval rules should not be added here.
        """
        return {
            "identity": " ".join(
                (
                    node.name or "",
                    node.description or "",
                    node.one_line_capability or "",
                    node.domain_tags or "",
                )
            ).lower(),
            "contract": " ".join(
                (
                    node.inputs or "",
                    node.outputs or "",
                    node.preconditions or "",
                    node.effects or "",
                )
            ).lower(),
            "tooling": " ".join(
                (
                    node.tooling or "",
                    node.script_entrypoints or "",
                    node.compatibility or "",
                    node.allowed_tools or "",
                )
            ).lower(),
        }

    @classmethod
    def _sanitize_role_text(cls, text: str) -> str:
        """Remove common skill-document boilerplate before role inference."""
        import re

        sanitized = str(text or "").lower()
        boilerplate_patterns = [
            r"always\s+search\s+tools?\s+first(?:\s+for\s+current\s+schemas?)?",
            r"search\s+tools?\s+first(?:\s+for\s+current\s+schemas?)?",
            r"current\s+tool\s+schemas?",
            r"schema[-\s]?compliant",
        ]
        for pattern in boilerplate_patterns:
            sanitized = re.sub(pattern, " ", sanitized)
        return sanitized

    @classmethod
    def _structural_pair_affinity(cls, source: SkillNode, target: SkillNode) -> float:
        """Local affinity guard for abstract role-chain candidates.

        Role compatibility alone is intentionally not enough: broad workflow
        phases such as "locate" or "deliver" appear in many unrelated skills.
        This score keeps structural candidates local to a namespace or to
        reusable content overlap, avoiding a graph dominated by global phase
        cross-products.
        """
        source_tokens = cls._affinity_tokens(source)
        target_tokens = cls._affinity_tokens(target)
        if not source_tokens or not target_tokens:
            lexical = 0.0
        else:
            overlap = source_tokens & target_tokens
            lexical = len(overlap) / max((len(source_tokens) * len(target_tokens)) ** 0.5, 1.0)

        namespace_bonus = 0.45 if cls._same_namespace(source.name, target.name) else 0.0
        return min(1.0, namespace_bonus + lexical)

    @classmethod
    def _affinity_tokens(cls, node: SkillNode) -> set[str]:
        import re

        facets = cls._role_facets(node)
        text = cls._sanitize_role_text(" ".join(facets.values()))
        stopwords = {
            "about",
            "across",
            "active",
            "always",
            "and",
            "api",
            "automate",
            "automation",
            "available",
            "before",
            "capability",
            "client",
            "composio",
            "connected",
            "current",
            "data",
            "description",
            "execute",
            "first",
            "for",
            "from",
            "generic",
            "input",
            "inputs",
            "interface",
            "mcp",
            "operation",
            "operations",
            "output",
            "outputs",
            "requires",
            "result",
            "rube",
            "schema",
            "schemas",
            "service",
            "skill",
            "task",
            "tasks",
            "the",
            "this",
            "tool",
            "toolkit",
            "toolkits",
            "tools",
            "use",
            "using",
            "via",
            "with",
            "workflow",
        }
        return {
            token
            for token in re.findall(r"[a-z][a-z0-9_]{2,}", text)
            if token not in stopwords
        }

    @classmethod
    def _metadata_roles(cls, node: SkillNode) -> set[str]:
        """Read explicitly normalized abstract roles from structured metadata."""
        metadata = node.metadata if hasattr(node, "metadata") else {}
        context = node.context_profile if hasattr(node, "context_profile") else {}
        raw_roles = []
        for container in (metadata, context):
            if not isinstance(container, dict):
                continue
            for key in ("roles", "role", "workflow_roles", "capability_roles"):
                raw_roles.extend(cls._coerce_text_list(container.get(key)))

        allowed = {
            "task_parse",
            "context_scan",
            "target_locate",
            "acquire_inventory",
            "place_output",
            "state_transform",
            "tool_device",
            "verify_inspect",
        }
        normalized = {
            str(role).strip().lower().replace("-", "_").replace(" ", "_")
            for role in raw_roles
        }
        return {role for role in normalized if role in allowed}

    @classmethod
    def _coerce_text_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, set)):
            result: list[str] = []
            for item in value:
                result.extend(cls._coerce_text_list(item))
            return result
        return [str(value)]

    @classmethod
    def _node_text(cls, node: SkillNode) -> str:
        parts = [
            node.name or "",
            node.description or "",
            node.one_line_capability or "",
            node.inputs or "",
            node.outputs or "",
            node.domain_tags or "",
            node.tooling or "",
            node.preconditions or "",
            node.effects or "",
            cls._flatten_metadata_text(node.metadata),
            cls._flatten_metadata_text(node.context_profile),
        ]
        return " ".join(part for part in parts if part).lower()

    @classmethod
    def _flatten_metadata_text(cls, value: Any) -> str:
        if isinstance(value, dict):
            return " ".join(cls._flatten_metadata_text(item) for item in value.values())
        if isinstance(value, (list, tuple, set)):
            return " ".join(cls._flatten_metadata_text(item) for item in value)
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def _contains_any(text: str, needles: set[str]) -> bool:
        import re

        for needle in needles:
            if " " in needle:
                if needle in text:
                    return True
                continue
            if re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", text):
                return True
        return False
