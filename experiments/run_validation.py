#!/usr/bin/env python3
"""Phase 2: LLM-based counterfactual validation for CaSKG candidate edges.

Loads a candidate graph (caskg_state.json from Phase 1) and validates causal
edges via three counterfactual probe types -- Removal, Substitution, and
Reordering -- using an LLM-as-judge. Updates Bayesian Beta-Binomial posteriors
per edge and classifies edges as confirmed_causal, rejected_non_causal, or
unverified (needs more evidence).

Usage:
    # Validate the 500-skill graph (top 500 edges by association score)
    python experiments/run_validation.py \
        --workspace data/caskg_workspace/skills_500_v1 \
        --skills-dir data/skillsets/skills_500 \
        --probe-types removal,substitution,reordering \
        --threshold 0.05 \
        --max-edges 500 \
        --resume

    # Quick test: 10 edges, removal only
    python experiments/run_validation.py \
        --workspace data/caskg_workspace/skills_200_v1 \
        --skills-dir data/skillsets/skills_200 \
        --probe-types removal \
        --max-edges 10

    # Dry-run: no LLM calls, random outcomes
    python experiments/run_validation.py \
        --workspace data/caskg_workspace/skills_500_v1 \
        --skills-dir data/skillsets/skills_500 \
        --max-edges 20 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Deferred import: BayesianEdgeEstimator is resolved at runtime to avoid
# pulling in heavy transitive dependencies (fast_graphrag, etc.) at parse time.
_BayesianEdgeEstimator = None


def _get_bayesian_estimator_class():
    """Lazily import BayesianEdgeEstimator."""
    global _BayesianEdgeEstimator
    if _BayesianEdgeEstimator is None:
        try:
            from caskg.causal.interventions import BayesianEdgeEstimator
            _BayesianEdgeEstimator = BayesianEdgeEstimator
        except ImportError:
            # Fallback: inline implementation matching the interface
            _BayesianEdgeEstimator = _FallbackBayesianEstimator
    return _BayesianEdgeEstimator


class _FallbackBayesianEstimator:
    """Standalone Beta-distribution estimator (no external deps)."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        self.alpha = alpha
        self.beta = beta

    def update(self, positive: bool, magnitude: float = 1.0) -> None:
        if positive:
            self.alpha += magnitude
        else:
            self.beta += magnitude

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))


logger = logging.getLogger("caskg.validation")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROBE_REMOVAL = "removal"
PROBE_SUBSTITUTION = "substitution"
PROBE_REORDERING = "reordering"
ALL_PROBE_TYPES = (PROBE_REMOVAL, PROBE_SUBSTITUTION, PROBE_REORDERING)

# Bayesian classification thresholds
CONFIRM_THRESHOLD = 0.7   # P(causal) > 0.7 => confirmed_causal
REJECT_THRESHOLD = 0.3    # P(causal) < 0.3 => rejected_non_causal

STATUS_UNVERIFIED = "unverified"
STATUS_CONFIRMED = "confirmed_causal"
STATUS_REJECTED = "rejected_non_causal"
STATUS_DEFERRED_UNVALIDATED = "deferred_unvalidated"
STATUS_DEFERRED_UNCERTAIN = "deferred_uncertain"
STATUS_STABLE = "stable"
STATUS_PENDING_RETEST = "pending_retest"

SELECTABLE_STATUSES = {None, "", STATUS_UNVERIFIED}
DEFERRED_STATUSES = {STATUS_DEFERRED_UNVALIDATED, STATUS_DEFERRED_UNCERTAIN}

# Rate-limit defaults (RPM=60 => 1/s; keep a small buffer)
DEFAULT_CALL_DELAY = 1.05  # seconds between LLM calls
DEFAULT_BATCH_CONCURRENCY = 4  # number of edges validated in parallel
CHECKPOINT_INTERVAL = 25  # save progress every N edges

# LLM response parsing
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0  # exponential backoff base in seconds


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class ProbeResult:
    """Outcome of a single counterfactual probe."""
    edge_source: str
    edge_target: str
    probe_type: str
    causal_effect: float        # 0.0-1.0 from LLM
    reasoning: str
    raw_response: str = ""
    timestamp: str = ""
    model: str = ""
    error: str = ""


@dataclass
class EdgeProgress:
    """Tracks which probes have been run for a given edge."""
    source: str
    target: str
    completed_probes: list[str] = field(default_factory=list)
    probe_results: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Skill metadata loading
# ---------------------------------------------------------------------------
def load_skill_metadata(skills_dir: Path) -> dict[str, dict[str, str]]:
    """Load skill name, description, and content from SKILL.md files.

    Returns a mapping from skill name (directory name) to a dict with keys:
        name, description, content (full SKILL.md text).
    """
    skills: dict[str, dict[str, str]] = {}
    if not skills_dir.exists():
        logger.warning("Skills directory does not exist: %s", skills_dir)
        return skills

    for skill_dir in sorted(skills_dir.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue

        content = skill_md.read_text(encoding="utf-8", errors="replace")

        # Parse frontmatter for name and description
        name = skill_dir.name
        description = ""
        lines = content.split("\n")
        in_frontmatter = False
        for line in lines:
            stripped = line.strip()
            if stripped == "---":
                if in_frontmatter:
                    break  # end of frontmatter
                in_frontmatter = True
                continue
            if in_frontmatter:
                if stripped.startswith("name:"):
                    name = stripped[5:].strip().strip('"').strip("'")
                elif stripped.startswith("description:"):
                    description = stripped[12:].strip().strip('"').strip("'")

        skills[skill_dir.name] = {
            "name": name,
            "description": description,
            "content": content[:3000],  # cap content length for prompts
        }

    logger.info("Loaded metadata for %d skills from %s", len(skills), skills_dir)
    return skills


# ---------------------------------------------------------------------------
# caskg_state.json I/O
# ---------------------------------------------------------------------------
def load_caskg_state(workspace: Path) -> dict[str, Any]:
    """Load caskg_state.json from the workspace directory."""
    state_file = workspace / "caskg_state.json"
    if not state_file.exists():
        raise FileNotFoundError(
            f"No caskg_state.json found in {workspace}. "
            "Run Phase 1 (caskg-index) first."
        )
    with open(state_file, encoding="utf-8") as f:
        data = json.load(f)

    edges = data.get("edges", [])
    logger.info(
        "Loaded caskg_state.json: %d edges (episode=%s)",
        len(edges),
        data.get("episode", "?"),
    )
    return data


def save_caskg_state(workspace: Path, state: dict[str, Any]) -> None:
    """Atomically save caskg_state.json (write to .tmp then rename)."""
    state_file = workspace / "caskg_state.json"
    tmp_file = workspace / "caskg_state.json.tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    tmp_file.replace(state_file)
    logger.debug("Saved caskg_state.json (%d edges)", len(state.get("edges", [])))


def publish_frozen_graph(workspace: Path, state: dict[str, Any]) -> dict[str, int]:
    """Publish validated/live state edges into the retrieval igraph."""
    from caskg.causal.graph_publisher import publish_state_edges_to_workspace

    return asyncio.run(
        publish_state_edges_to_workspace(
            workspace,
            state,
            include_unverified=False,
        )
    )


def normalize_edge_statuses(edges: list[dict]) -> int:
    """Normalize missing legacy statuses to unverified.

    Older workspaces can contain JSON null for ``status``. During the active
    validation phase those edges are selectable, but the state file should never
    keep null statuses because frozen retrieval cannot distinguish "unknown"
    from "accidentally omitted" metadata.
    """
    changed = 0
    for edge in edges:
        status = edge.get("status")
        if status is None or status == "":
            edge["status"] = STATUS_UNVERIFIED
            changed += 1
    return changed


def finalize_validation_frontier(
    edges: list[dict],
    selected_keys: set[tuple[str, str]],
    *,
    finalize_unselected: bool,
    finalize_uncertain: bool,
) -> dict[str, int]:
    """Freeze unresolved edges after a validation pass.

    ``rejected_non_causal`` is reserved for explicit negative evidence. Edges
    that were not selected by the validation budget are instead marked
    ``deferred_unvalidated`` so Phase-3 retrieval can exclude them without
    pretending they were causally disproven. Selected edges whose posterior
    remains in the middle are marked ``deferred_uncertain`` when requested.
    """
    counts = {
        "normalized_null": normalize_edge_statuses(edges),
        "deferred_unvalidated": 0,
        "deferred_uncertain": 0,
    }

    for edge in edges:
        status = edge.get("status")
        if status != STATUS_UNVERIFIED:
            continue

        key = (edge.get("source", ""), edge.get("target", ""))
        interventions = int(edge.get("intervention_count") or 0)

        if key in selected_keys:
            if finalize_uncertain:
                if interventions > 0:
                    edge["status"] = STATUS_DEFERRED_UNCERTAIN
                    counts["deferred_uncertain"] += 1
                else:
                    edge["status"] = STATUS_DEFERRED_UNVALIDATED
                    counts["deferred_unvalidated"] += 1
        elif finalize_unselected:
            edge["status"] = STATUS_DEFERRED_UNVALIDATED
            counts["deferred_unvalidated"] += 1

    return counts


# ---------------------------------------------------------------------------
# Validation log (append-only JSONL)
# ---------------------------------------------------------------------------
class ValidationLog:
    """Append-only JSONL log for probe results."""

    def __init__(self, workspace: Path):
        self.path = workspace / "validation_log.jsonl"

    def append(self, result: ProbeResult) -> None:
        entry = asdict(result)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def load_completed(self) -> set[tuple[str, str, str]]:
        """Load (source, target, probe_type) tuples already completed."""
        completed: set[tuple[str, str, str]] = set()
        if not self.path.exists():
            return completed
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if not entry.get("error"):
                        completed.add((
                            entry["edge_source"],
                            entry["edge_target"],
                            entry["probe_type"],
                        ))
                except (json.JSONDecodeError, KeyError):
                    continue
        return completed

    def load_results(self) -> list[dict]:
        """Load all successful probe results as dicts."""
        results = []
        if not self.path.exists():
            return results
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if not entry.get("error"):
                        results.append(entry)
                except (json.JSONDecodeError, KeyError):
                    continue
        return results


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------
class LLMClient:
    """Thin wrapper around the OpenAI-compatible chat completions API."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        call_delay: float = DEFAULT_CALL_DELAY,
        dry_run: bool = False,
    ):
        self.model = model
        self.call_delay = call_delay
        self.dry_run = dry_run
        self._last_call_time: float = 0.0
        import threading
        self._rate_lock = threading.Lock()

        if not dry_run:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package is required. Install with: pip install openai"
                )
            # Strip provider prefix from model name for OpenAI client
            # e.g. "openai/MiniMax-M2.7" => "MiniMax-M2.7"
            self._api_model = model.split("/", 1)[-1] if "/" in model else model
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self._api_model = model
            self._client = None

    def _rate_limit_wait(self) -> None:
        """Enforce minimum interval between API calls."""
        if self.call_delay <= 0:
            return
        elapsed = time.time() - self._last_call_time
        if elapsed < self.call_delay:
            time.sleep(self.call_delay - elapsed)

    def call(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat completion request and return the response text.

        Handles rate limiting, retries with exponential backoff, and
        dry-run simulation.
        """
        if self.dry_run:
            # Simulate a plausible LLM response
            effect = round(random.uniform(0.0, 1.0), 2)
            return json.dumps({
                "causal_effect": effect,
                "reasoning": f"[DRY RUN] Simulated causal effect = {effect}",
            })

        for attempt in range(MAX_RETRIES):
            with self._rate_lock:
                self._rate_limit_wait()
                self._last_call_time = time.time()
            try:
                response = self._client.chat.completions.create(
                    model=self._api_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=800,
                )
                content = response.choices[0].message.content or ""
                return content.strip()
            except Exception as exc:
                wait_time = RETRY_BACKOFF_BASE ** (attempt + 1)
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1,
                    MAX_RETRIES,
                    exc,
                    wait_time,
                )
                time.sleep(wait_time)

        raise RuntimeError(
            f"LLM call failed after {MAX_RETRIES} retries"
        )


# ---------------------------------------------------------------------------
# Counterfactual prompt construction
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert in software engineering skill dependencies and causal reasoning.
You evaluate whether a causal dependency exists between two skills (tools, libraries,
frameworks, or automation capabilities).

A "causal dependency" means that Skill A genuinely enables, enhances, or is a
prerequisite for Skill B -- not just that they co-occur or are topically related.

You must respond with ONLY a JSON object (no markdown, no code fences) containing:
{
  "causal_effect": <float 0.0 to 1.0>,
  "reasoning": "<1-3 sentence explanation>"
}

Where causal_effect represents how strongly the evidence supports a genuine causal
link (1.0 = strong causal dependency, 0.0 = no causal relationship).
"""


def build_removal_prompt(
    source_name: str,
    source_desc: str,
    target_name: str,
    target_desc: str,
    source_content_snippet: str,
    target_content_snippet: str,
) -> str:
    """Build a REMOVAL counterfactual probe prompt."""
    return f"""\
COUNTERFACTUAL PROBE: REMOVAL

Consider these two skills:

SKILL A (potential cause): {source_name}
Description: {source_desc}
{f"Details: {source_content_snippet[:800]}" if source_content_snippet else ""}

SKILL B (potential effect): {target_name}
Description: {target_desc}
{f"Details: {target_content_snippet[:800]}" if target_content_snippet else ""}

COUNTERFACTUAL QUESTION:
If Skill A ("{source_name}") were completely removed from a developer's toolkit,
would Skill B ("{target_name}") still function and be effective independently?

Consider:
- Does Skill B depend on outputs, artifacts, or capabilities provided by Skill A?
- Could Skill B operate in isolation without any knowledge or setup from Skill A?
- Is Skill A a genuine prerequisite, or are they merely related by topic?

If removing A would significantly impair B, causal_effect should be HIGH (0.7-1.0).
If B works perfectly fine without A, causal_effect should be LOW (0.0-0.3).

Respond with ONLY the JSON object."""


def build_substitution_prompt(
    source_name: str,
    source_desc: str,
    target_name: str,
    target_desc: str,
    substitute_name: str,
    substitute_desc: str,
    source_content_snippet: str,
    target_content_snippet: str,
) -> str:
    """Build a SUBSTITUTION counterfactual probe prompt."""
    return f"""\
COUNTERFACTUAL PROBE: SUBSTITUTION

Consider these skills:

SKILL A (potential cause): {source_name}
Description: {source_desc}
{f"Details: {source_content_snippet[:600]}" if source_content_snippet else ""}

SKILL B (potential effect): {target_name}
Description: {target_desc}
{f"Details: {target_content_snippet[:600]}" if target_content_snippet else ""}

SUBSTITUTE SKILL D (unrelated replacement): {substitute_name}
Description: {substitute_desc}

COUNTERFACTUAL QUESTION:
If Skill A ("{source_name}") is replaced by Skill D ("{substitute_name}"),
does Skill B ("{target_name}") still work as effectively?

Consider:
- Is the connection between A and B based on specific capabilities that D lacks?
- Could any generic skill replace A in supporting B, or is A uniquely necessary?
- Does the replacement break a specific functional dependency?

If replacing A with the unrelated D significantly degrades B, causal_effect should be HIGH (0.7-1.0).
If B works just as well with D instead of A, causal_effect should be LOW (0.0-0.3).

Respond with ONLY the JSON object."""


def build_reordering_prompt(
    source_name: str,
    source_desc: str,
    target_name: str,
    target_desc: str,
    source_content_snippet: str,
    target_content_snippet: str,
) -> str:
    """Build a REORDERING counterfactual probe prompt."""
    return f"""\
COUNTERFACTUAL PROBE: REORDERING

Consider these two skills in a workflow:

SKILL A (currently first): {source_name}
Description: {source_desc}
{f"Details: {source_content_snippet[:800]}" if source_content_snippet else ""}

SKILL B (currently second): {target_name}
Description: {target_desc}
{f"Details: {target_content_snippet[:800]}" if target_content_snippet else ""}

Current order: A -> B (Skill A is applied before Skill B)

COUNTERFACTUAL QUESTION:
If the order is reversed to B -> A (Skill B applied before Skill A),
does the workflow still make logical sense and produce valid results?

Consider:
- Does Skill A produce outputs that Skill B consumes as inputs?
- Is there a logical temporal dependency (A must happen before B)?
- Would reversing the order cause errors, redundancy, or invalid states?

If reversing the order breaks the workflow, causal_effect should be HIGH (0.7-1.0).
If the order does not matter, causal_effect should be LOW (0.0-0.3).

Respond with ONLY the JSON object."""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
def parse_llm_response(raw: str) -> tuple[float, str]:
    """Parse the LLM response to extract causal_effect and reasoning.

    Returns (causal_effect, reasoning). On parse failure returns (0.5, error_msg).
    """
    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last line if they are fences
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try direct JSON parse
    try:
        data = json.loads(text)
        effect = float(data.get("causal_effect", 0.5))
        reasoning = str(data.get("reasoning", ""))
        return (max(0.0, min(1.0, effect)), reasoning)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: extract JSON object from mixed text
    json_match = re.search(r'\{[^{}]*"causal_effect"\s*:\s*[\d.]+[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            effect = float(data.get("causal_effect", 0.5))
            reasoning = str(data.get("reasoning", ""))
            return (max(0.0, min(1.0, effect)), reasoning)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Last resort: regex for the number
    effect_match = re.search(r'"?causal_effect"?\s*[:=]\s*([\d.]+)', text)
    if effect_match:
        try:
            effect = float(effect_match.group(1))
            return (max(0.0, min(1.0, effect)), f"[PARTIAL PARSE] {text[:200]}")
        except ValueError:
            pass

    logger.warning("Failed to parse LLM response: %s", text[:300])
    return (0.5, f"[PARSE FAILURE] {text[:200]}")


# ---------------------------------------------------------------------------
# Substitute skill selection
# ---------------------------------------------------------------------------
def pick_substitute_skill(
    source_name: str,
    target_name: str,
    all_skill_names: list[str],
    rng: random.Random,
) -> str | None:
    """Pick an unrelated skill to use as a substitute in substitution probes.

    Tries to pick a skill whose name shares minimal overlap with source/target.
    Returns None if no suitable substitute exists.
    """
    candidates = [
        s for s in all_skill_names
        if s != source_name and s != target_name
    ]
    if not candidates:
        return None

    # Prefer skills with low name-token overlap with source
    source_tokens = set(source_name.lower().replace("-", " ").replace("_", " ").split())
    scored = []
    for c in candidates:
        c_tokens = set(c.lower().replace("-", " ").replace("_", " ").split())
        overlap = len(source_tokens & c_tokens)
        scored.append((overlap, c))
    scored.sort(key=lambda x: x[0])

    # Pick from the least-overlapping quarter
    pool_size = max(1, len(scored) // 4)
    pool = [name for _, name in scored[:pool_size]]
    return rng.choice(pool)


# ---------------------------------------------------------------------------
# Core validation loop
# ---------------------------------------------------------------------------
def _skill_family(name: str) -> str:
    """Collapse a skill name to its family for deduplication.

    Tool wrappers often share suffixes such as `-automation`, `-integration`,
    or `-connector`. Deduplication should group siblings from the same provider,
    not collapse every wrapper in the ecosystem into one global family.
    """
    raw = str(name or "").strip()
    normalized = raw.lower().replace("_", "-")
    for suffix, tag in (
        ("-automation", "automation"),
        (" automation", "automation"),
        ("-integration", "integration"),
        (" integration", "integration"),
        ("-connector", "connector"),
        (" connector", "connector"),
    ):
        if not normalized.endswith(suffix):
            continue
        stem = normalized[: -len(suffix)].strip("- ")
        parts = [part for part in re.split(r"[-\s]+", stem) if part]
        provider = "-".join(parts[:2]) if parts else tag
        return f"{provider}:{tag}"
    return normalized or raw


def _skill_namespace(name: str) -> str:
    match = re.match(r"^([a-z][a-z0-9]{2,})-", (name or "").lower())
    return match.group(1) if match else ""


def _terminal_edge_scores(
    edges: list[dict],
    structural_types: set[str],
) -> dict[int, float]:
    """Score candidate edges whose targets behave like graph sinks.

    This is intentionally topology-only. It reserves validation coverage for
    tail-of-workflow nodes without inspecting skill descriptions for verifier,
    task, or benchmark-specific words.
    """
    incoming: dict[str, int] = {}
    outgoing: dict[str, int] = {}
    for item in edges:
        if item.get("type") not in structural_types:
            continue
        source = item.get("source", "")
        target = item.get("target", "")
        if not source or not target:
            continue
        outgoing[source] = outgoing.get(source, 0) + 1
        incoming[target] = incoming.get(target, 0) + 1

    scores: dict[int, float] = {}
    for item in edges:
        if item.get("type") not in structural_types:
            continue
        source = item.get("source", "")
        target = item.get("target", "")
        if not source or not target:
            continue
        target_in = incoming.get(target, 0)
        target_out = outgoing.get(target, 0)
        source_out = outgoing.get(source, 0)
        if target_in == 0 or source_out == 0:
            continue
        sinkness = target_in / (target_in + target_out + 1)
        bridge = min(source_out, 5) / 5
        if sinkness < 0.5:
            continue
        score = sinkness + 0.1 * bridge + item.get("association_score", 0.0)
        source_namespace = _skill_namespace(source)
        target_namespace = _skill_namespace(target)
        if source_namespace and target_namespace and source_namespace == target_namespace:
            score += 0.25
        elif source_namespace and target_namespace:
            score *= 0.65
        scores[id(item)] = score
    return scores


def _target_namespace(edge: dict) -> str:
    return _skill_namespace(edge.get("target", "")) or _skill_namespace(edge.get("source", ""))


def _edge_type_bucket(edge: dict) -> str:
    """Map raw induction edge types to retrieval/publishing families."""
    edge_type = str(edge.get("type") or "semantic").strip().lower()
    if edge_type in {"dependency", "prereq", "prerequisite", "data-flow", "data_flow"}:
        return "dependency"
    if edge_type in {"enhance", "workflow", "repair-support", "repair_support"}:
        return "workflow"
    if edge_type in {"similar", "semantic", "cooccur"}:
        return "semantic"
    if edge_type == "alternative":
        return "alternative"
    return edge_type or "semantic"


def _balanced_terminal_edges(
    edges: list[dict],
    terminal_scores: dict[int, float],
    quota: int,
) -> list[dict]:
    """Select terminal candidates with namespace and sink-target coverage."""
    if quota <= 0:
        return []

    buckets: dict[str, dict[str, list[dict]]] = {}
    for edge in edges:
        score = terminal_scores.get(id(edge))
        if score is None:
            continue
        namespace = _target_namespace(edge)
        target = edge.get("target", "")
        buckets.setdefault(namespace, {}).setdefault(target, []).append(edge)

    for targets in buckets.values():
        for target_edges in targets.values():
            target_edges.sort(key=lambda edge: terminal_scores[id(edge)], reverse=True)

    namespace_order = sorted(
        buckets,
        key=lambda namespace: max(
            terminal_scores[id(edge)]
            for edges_by_target in buckets[namespace].values()
            for edge in edges_by_target
        ),
        reverse=True,
    )
    target_order = {
        namespace: sorted(
            targets,
            key=lambda target: terminal_scores[id(targets[target][0])],
            reverse=True,
        )
        for namespace, targets in buckets.items()
    }

    selected: list[dict] = []
    selected_ids: set[int] = set()
    while len(selected) < quota:
        made_progress = False
        for namespace in namespace_order:
            targets = buckets.get(namespace, {})
            for target in list(target_order.get(namespace, [])):
                target_edges = targets.get(target, [])
                while target_edges and id(target_edges[0]) in selected_ids:
                    target_edges.pop(0)
                if not target_edges:
                    continue
                edge = target_edges.pop(0)
                selected.append(edge)
                selected_ids.add(id(edge))
                made_progress = True
                break
            if len(selected) >= quota:
                break
        if not made_progress:
            break

    return selected


def _balanced_type_coverage_edges(
    edges: list[dict],
    quota: int,
    priority_fn,
) -> list[dict]:
    """Select high-priority edges while covering relation-type families.

    Phase-1 induction can produce many high-scoring edges of one raw type. A
    validation frontier made only from that majority type publishes a narrow
    graph even when the candidate graph contains workflow and semantic support.
    """
    if quota <= 0:
        return []

    buckets: dict[str, dict[str, list[dict]]] = {}
    for edge in edges:
        type_bucket = _edge_type_bucket(edge)
        namespace = _target_namespace(edge)
        buckets.setdefault(type_bucket, {}).setdefault(namespace, []).append(edge)

    for namespaces in buckets.values():
        for bucket_edges in namespaces.values():
            bucket_edges.sort(key=priority_fn, reverse=True)

    type_order = sorted(
        buckets,
        key=lambda type_bucket: max(
            priority_fn(edge)
            for namespace_edges in buckets[type_bucket].values()
            for edge in namespace_edges
        ),
        reverse=True,
    )
    namespace_order = {
        type_bucket: sorted(
            namespaces,
            key=lambda namespace: priority_fn(namespaces[namespace][0]),
            reverse=True,
        )
        for type_bucket, namespaces in buckets.items()
    }

    selected: list[dict] = []
    selected_ids: set[int] = set()
    while len(selected) < quota:
        made_progress = False
        for type_bucket in type_order:
            namespaces = buckets.get(type_bucket, {})
            for namespace in list(namespace_order.get(type_bucket, [])):
                bucket_edges = namespaces.get(namespace, [])
                while bucket_edges and id(bucket_edges[0]) in selected_ids:
                    bucket_edges.pop(0)
                if not bucket_edges:
                    continue
                edge = bucket_edges.pop(0)
                selected.append(edge)
                selected_ids.add(id(edge))
                made_progress = True
                break
            if len(selected) >= quota:
                break
        if not made_progress:
            break

    return selected


def _new_node_coverage_edges(
    edges: list[dict],
    quota: int,
    priority_fn,
    covered_nodes: set[str],
) -> list[dict]:
    """Select edges that expand node coverage before local score saturation."""
    if quota <= 0:
        return []

    remaining = list(edges)
    selected: list[dict] = []
    while len(selected) < quota and remaining:
        remaining.sort(
            key=lambda edge: (
                int(bool(edge.get("source") and edge.get("source") not in covered_nodes))
                + int(bool(edge.get("target") and edge.get("target") not in covered_nodes)),
                priority_fn(edge),
            ),
            reverse=True,
        )
        edge = remaining.pop(0)
        source = edge.get("source", "")
        target = edge.get("target", "")
        new_nodes = {
            node
            for node in (source, target)
            if node and node not in covered_nodes
        }
        if not new_nodes:
            break
        selected.append(edge)
        covered_nodes.update(new_nodes)

    return selected


def select_edges_for_validation(
    edges: list[dict],
    threshold: float,
    max_edges: int,
    include_deferred: bool = False,
    skills: dict[str, dict[str, str]] | None = None,
) -> list[dict]:
    """Select candidate edges for validation with graph-level coverage.

    Pure top-N-by-association spends the whole budget on spurious lexical matches
    or one dense relation family. That can leave the published graph narrow even
    when the candidate graph contains broader structural evidence. This
    selection instead:

      1. Keeps only unresolved edges with association_score >= threshold.
      2. Reserves coverage for terminal validation/sink edges
         so tail-of-workflow skills are not starved by high-scoring internal
         workflow edges.
      3. Reserves coverage across relation-type families and namespaces.
      4. Reserves coverage for edges that add previously uncovered nodes.
      5. Deduplicates by (source-family, target-family), then fills remaining
         budget with the highest-scoring leftovers.
    """
    eligible = []
    selectable_statuses = set(SELECTABLE_STATUSES)
    if include_deferred:
        selectable_statuses.update(DEFERRED_STATUSES)
    for edge in edges:
        status = edge.get("status")
        if status not in selectable_statuses:
            continue
        score = edge.get("association_score")
        if score is None or score < threshold:
            continue
        if status in DEFERRED_STATUSES:
            edge["status"] = STATUS_UNVERIFIED
        eligible.append(edge)

    structural_types = {"dependency", "prereq", "enhance", "workflow", "data-flow"}

    def priority(edge: dict) -> tuple[int, int, float]:
        is_structural = 1 if (edge.get("type") in structural_types) else 0
        source_namespace = _skill_namespace(edge.get("source", ""))
        target_namespace = _skill_namespace(edge.get("target", ""))
        same_namespace = 1 if (
            source_namespace
            and target_namespace
            and source_namespace == target_namespace
        ) else 0
        return (is_structural, same_namespace, edge.get("association_score", 0.0))

    eligible.sort(key=priority, reverse=True)

    selected: list[dict] = []
    selected_ids: set[int] = set()
    family_counts: dict[tuple[str, str], int] = {}
    MAX_PER_FAMILY_PAIR = 2
    deferred: list[dict] = []

    def add_edge(edge: dict, *, enforce_family_cap: bool = True) -> bool:
        if max_edges > 0 and len(selected) >= max_edges:
            return False
        if id(edge) in selected_ids:
            return False
        fam = (_skill_family(edge["source"]), _skill_family(edge["target"]))
        if enforce_family_cap and family_counts.get(fam, 0) >= MAX_PER_FAMILY_PAIR:
            return False
        selected.append(edge)
        selected_ids.add(id(edge))
        family_counts[fam] = family_counts.get(fam, 0) + 1
        return True

    terminal_scores = _terminal_edge_scores(eligible, structural_types)
    terminal_quota = max(1, max_edges // 20) if max_edges > 0 and terminal_scores else 0

    if terminal_quota:
        terminal_edges = _balanced_terminal_edges(
            eligible,
            terminal_scores,
            terminal_quota,
        )
        for edge in terminal_edges:
            if len(selected) >= terminal_quota:
                break
            add_edge(edge)

    type_buckets = {_edge_type_bucket(edge) for edge in eligible}
    type_quota = 0
    if max_edges > 0 and len(type_buckets) > 1:
        type_quota = min(max_edges - len(selected), max(len(type_buckets), max_edges // 5))
    if type_quota > 0:
        coverage_edges = _balanced_type_coverage_edges(
            [edge for edge in eligible if id(edge) not in selected_ids],
            type_quota,
            priority,
        )
        target_count = len(selected) + type_quota
        for edge in coverage_edges:
            if len(selected) >= target_count:
                break
            add_edge(edge)

    node_quota = max_edges // 4 if max_edges > 0 else 0
    if node_quota > 0:
        covered_nodes = {
            node
            for edge in selected
            for node in (edge.get("source", ""), edge.get("target", ""))
            if node
        }
        node_edges = _new_node_coverage_edges(
            [edge for edge in eligible if id(edge) not in selected_ids],
            min(node_quota, max_edges - len(selected)),
            priority,
            covered_nodes,
        )
        for edge in node_edges:
            add_edge(edge)

    for edge in eligible:
        if id(edge) in selected_ids:
            continue
        if not add_edge(edge):
            deferred.append(edge)
        if max_edges > 0 and len(selected) >= max_edges:
            break

    # If budget remains, top up from deferred (highest score first already).
    if max_edges <= 0 or len(selected) < max_edges:
        for edge in deferred:
            if id(edge) in selected_ids:
                continue
            add_edge(edge, enforce_family_cap=False)
            if max_edges > 0 and len(selected) >= max_edges:
                break

    n_structural = sum(1 for e in selected if e.get("type") in structural_types)
    type_counts: dict[str, int] = {}
    touched_nodes = {
        node
        for edge in selected
        for node in (edge.get("source", ""), edge.get("target", ""))
        if node
    }
    for edge in selected:
        type_bucket = _edge_type_bucket(edge)
        type_counts[type_bucket] = type_counts.get(type_bucket, 0) + 1
    logger.info(
        (
            "Selected %d edges for validation "
            "(threshold=%.3f, max=%d, structural=%d, nodes=%d, types=%s)"
        ),
        len(selected),
        threshold,
        max_edges,
        n_structural,
        len(touched_nodes),
        type_counts,
    )
    return selected


def run_probe(
    llm: LLMClient,
    probe_type: str,
    edge: dict,
    skills: dict[str, dict[str, str]],
    all_skill_names: list[str],
    rng: random.Random,
) -> ProbeResult:
    """Execute a single counterfactual probe against the LLM.

    Constructs the appropriate prompt, calls the LLM, parses the response,
    and returns a ProbeResult.
    """
    source = edge["source"]
    target = edge["target"]

    source_meta = skills.get(source, {})
    target_meta = skills.get(target, {})
    source_desc = source_meta.get("description", source)
    target_desc = target_meta.get("description", target)
    source_content = source_meta.get("content", "")
    target_content = target_meta.get("content", "")

    try:
        if probe_type == PROBE_REMOVAL:
            user_prompt = build_removal_prompt(
                source, source_desc, target, target_desc,
                source_content, target_content,
            )
        elif probe_type == PROBE_SUBSTITUTION:
            sub_name = pick_substitute_skill(source, target, all_skill_names, rng)
            if sub_name is None:
                return ProbeResult(
                    edge_source=source,
                    edge_target=target,
                    probe_type=probe_type,
                    causal_effect=0.5,
                    reasoning="No suitable substitute skill found",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    model=llm.model,
                    error="no_substitute",
                )
            sub_meta = skills.get(sub_name, {})
            sub_desc = sub_meta.get("description", sub_name)
            user_prompt = build_substitution_prompt(
                source, source_desc, target, target_desc,
                sub_name, sub_desc, source_content, target_content,
            )
        elif probe_type == PROBE_REORDERING:
            user_prompt = build_reordering_prompt(
                source, source_desc, target, target_desc,
                source_content, target_content,
            )
        else:
            raise ValueError(f"Unknown probe type: {probe_type}")

        raw_response = llm.call(SYSTEM_PROMPT, user_prompt)
        causal_effect, reasoning = parse_llm_response(raw_response)

        return ProbeResult(
            edge_source=source,
            edge_target=target,
            probe_type=probe_type,
            causal_effect=causal_effect,
            reasoning=reasoning,
            raw_response=raw_response[:500],
            timestamp=datetime.now(timezone.utc).isoformat(),
            model=llm.model,
        )

    except Exception as exc:
        logger.error(
            "Probe failed for %s -> %s (%s): %s",
            source, target, probe_type, exc,
        )
        return ProbeResult(
            edge_source=source,
            edge_target=target,
            probe_type=probe_type,
            causal_effect=0.5,
            reasoning="",
            timestamp=datetime.now(timezone.utc).isoformat(),
            model=llm.model,
            error=str(exc),
        )


def update_edge_posterior(
    edge: dict,
    probe_results: list[ProbeResult],
) -> dict:
    """Update an edge's Bayesian posterior from probe results.

    Uses BayesianEdgeEstimator from caskg.causal.interventions.
    Applies weighted evidence from each probe's causal_effect score.

    Returns the updated edge dict (mutated in place).
    """
    alpha = edge.get("alpha_posterior") or 1.0
    beta = edge.get("beta_posterior") or 1.0
    intervention_count = edge.get("intervention_count") or 0

    EstimatorClass = _get_bayesian_estimator_class()
    estimator = EstimatorClass(alpha=alpha, beta=beta)

    for result in probe_results:
        if result.error:
            continue
        effect = result.causal_effect
        # Treat as magnitude-weighted evidence.
        # effect > 0.5 is positive causal evidence, < 0.5 is negative.
        magnitude = abs(effect - 0.5) * 2.0  # scale to 0-1
        magnitude = max(magnitude, 0.1)       # minimum evidence weight
        positive = effect > 0.5
        estimator.update(positive=positive, magnitude=magnitude)
        intervention_count += 1

    # Posterior classification
    p_causal = estimator.mean
    if p_causal > CONFIRM_THRESHOLD:
        new_status = "confirmed_causal"
    elif p_causal < REJECT_THRESHOLD:
        new_status = "rejected_non_causal"
    else:
        new_status = "unverified"

    edge["alpha_posterior"] = round(estimator.alpha, 6)
    edge["beta_posterior"] = round(estimator.beta, 6)
    edge["causal_score"] = round(p_causal, 6)
    edge["uncertainty"] = round(estimator.variance, 6)
    edge["intervention_count"] = intervention_count
    edge["status"] = new_status

    return edge


def run_validation(
    workspace: Path,
    skills_dir: Path,
    probe_types: list[str],
    threshold: float,
    max_edges: int,
    resume: bool,
    dry_run: bool,
    call_delay: float,
    batch_concurrency: int,
    seed: int | None,
    finalize_unselected: bool,
    finalize_uncertain: bool,
    include_deferred: bool,
) -> dict[str, Any]:
    """Main validation loop.

    Returns summary statistics dict.
    """
    rng = random.Random(seed)

    # Load state and skills
    state = load_caskg_state(workspace)
    edges = state["edges"]
    normalized_null = normalize_edge_statuses(edges)
    if normalized_null:
        logger.info("Normalized %d null edge statuses to unverified", normalized_null)
    skills = load_skill_metadata(skills_dir)
    all_skill_names = list(skills.keys())

    if not all_skill_names:
        logger.warning(
            "No skills loaded from %s. Prompts will lack skill descriptions.",
            skills_dir,
        )
        # Build skill name list from edges instead
        names_set: set[str] = set()
        for e in state["edges"]:
            names_set.add(e["source"])
            names_set.add(e["target"])
        all_skill_names = sorted(names_set)

    # LLM client
    api_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    base_url = (
        os.environ.get("BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    )
    model = os.environ.get("CASKG_LLM_MODEL", "openai/MiniMax-M2.7")

    if not api_key and not dry_run:
        raise RuntimeError(
            "API_KEY or OPENAI_API_KEY environment variable is not set. "
            "Set one of them or use --dry-run."
        )

    llm = LLMClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        call_delay=call_delay,
        dry_run=dry_run,
    )

    # Validation log for append-only persistence
    val_log = ValidationLog(workspace)

    # Resume support: load already-completed probes
    completed_probes: set[tuple[str, str, str]] = set()
    if resume:
        completed_probes = val_log.load_completed()
        if completed_probes:
            logger.info(
                "Resuming: %d probes already completed", len(completed_probes)
            )

    # Select edges to validate
    candidates = select_edges_for_validation(
        edges,
        threshold,
        max_edges,
        include_deferred=include_deferred,
        skills=skills,
    )
    selected_keys = {
        (edge.get("source", ""), edge.get("target", ""))
        for edge in candidates
    }

    if not candidates:
        logger.warning("No edges meet the selection criteria. Nothing to validate.")
        finalize_counts = finalize_validation_frontier(
            edges,
            selected_keys,
            finalize_unselected=finalize_unselected,
            finalize_uncertain=finalize_uncertain,
        )
        save_caskg_state(workspace, state)
        publish_stats = publish_frozen_graph(workspace, state)
        status_counts: dict[str, int] = {}
        for edge in edges:
            s = edge.get("status") or "null"
            status_counts[s] = status_counts.get(s, 0) + 1
        return {
            "total_edges_in_graph": len(edges),
            "edges_selected": 0,
            "edges_processed": 0,
            "total_probes_run": 0,
            "probes_skipped_resume": 0,
            "probes_failed": 0,
            "elapsed_seconds": 0.0,
            "probes_per_minute": 0,
            "edge_status_distribution": status_counts,
            "model": model,
            "probe_types": probe_types,
            "threshold": threshold,
            "dry_run": dry_run,
            "finalized_edges": finalize_counts,
            "published_graph": publish_stats,
        }

    # Build edge lookup for efficient update
    edge_index: dict[tuple[str, str], int] = {}
    for idx, edge in enumerate(edges):
        edge_index[(edge["source"], edge["target"])] = idx

    # Validation loop
    total_probes = 0
    skipped_probes = 0
    failed_probes = 0
    edges_processed = 0
    start_time = time.time()

    logger.info(
        "Starting validation: %d edges, probe types=%s, model=%s, dry_run=%s, batch_concurrency=%d",
        len(candidates),
        probe_types,
        model,
        dry_run,
        batch_concurrency,
    )

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Thread-safe lock for shared state updates
    _lock = threading.Lock()

    def _validate_edge(edge_idx_candidate):
        """Validate a single edge (all its probes). Thread-safe."""
        edge_idx, candidate = edge_idx_candidate
        source = candidate["source"]
        target = candidate["target"]
        edge_key = (source, target)

        edge_results: list[ProbeResult] = []
        local_probes = 0
        local_skipped = 0
        local_failed = 0

        for probe_type in probe_types:
            if (source, target, probe_type) in completed_probes:
                local_skipped += 1
                continue

            result = run_probe(
                llm, probe_type, candidate, skills, all_skill_names, rng,
            )
            edge_results.append(result)
            local_probes += 1

            if result.error:
                local_failed += 1

            with _lock:
                val_log.append(result)

        # Load prior results for resume
        if resume and edge_key in edge_index:
            prior_results = val_log.load_results()
            for pr in prior_results:
                if pr["edge_source"] == source and pr["edge_target"] == target:
                    already_included = any(
                        r.probe_type == pr["probe_type"] for r in edge_results
                    )
                    if not already_included:
                        edge_results.append(ProbeResult(
                            edge_source=pr["edge_source"],
                            edge_target=pr["edge_target"],
                            probe_type=pr["probe_type"],
                            causal_effect=pr["causal_effect"],
                            reasoning=pr.get("reasoning", ""),
                            raw_response=pr.get("raw_response", ""),
                            timestamp=pr.get("timestamp", ""),
                            model=pr.get("model", ""),
                        ))

        # Update posterior
        if edge_key in edge_index and edge_results:
            with _lock:
                actual_idx = edge_index[edge_key]
                edges[actual_idx] = update_edge_posterior(
                    edges[actual_idx], edge_results,
                )

        return local_probes, local_skipped, local_failed

    # Process edges in parallel batches
    work_items = list(enumerate(candidates))
    with ThreadPoolExecutor(max_workers=batch_concurrency) as executor:
        futures = {
            executor.submit(_validate_edge, item): item
            for item in work_items
        }
        for future in as_completed(futures):
            try:
                local_probes, local_skipped, local_failed = future.result()
            except Exception as exc:
                logger.warning("Edge validation failed: %s", exc)
                local_probes, local_skipped, local_failed = 0, 0, 0

            total_probes += local_probes
            skipped_probes += local_skipped
            failed_probes += local_failed
            edges_processed += 1

            if edges_processed % 10 == 0:
                elapsed = time.time() - start_time
                rate = total_probes / elapsed if elapsed > 0 else 0
                logger.info(
                    "Progress: edge %d/%d, probes=%d (%.1f/min), "
                    "skipped=%d, failed=%d",
                    edges_processed,
                    len(candidates),
                    total_probes,
                    rate * 60,
                    skipped_probes,
                    failed_probes,
                )

            # Periodic checkpoint
            if edges_processed % CHECKPOINT_INTERVAL == 0:
                with _lock:
                    save_caskg_state(workspace, state)
                logger.info("Checkpoint saved at edge %d/%d", edges_processed, len(candidates))

    finalize_counts = finalize_validation_frontier(
        edges,
        selected_keys,
        finalize_unselected=finalize_unselected,
        finalize_uncertain=finalize_uncertain,
    )

    # Final save
    save_caskg_state(workspace, state)
    publish_stats = publish_frozen_graph(workspace, state)

    elapsed = time.time() - start_time

    # Compute summary statistics
    status_counts: dict[str, int] = {}
    for edge in edges:
        s = edge.get("status") or "null"
        status_counts[s] = status_counts.get(s, 0) + 1

    summary = {
        "total_edges_in_graph": len(edges),
        "edges_selected": len(candidates),
        "edges_processed": edges_processed,
        "total_probes_run": total_probes,
        "probes_skipped_resume": skipped_probes,
        "probes_failed": failed_probes,
        "elapsed_seconds": round(elapsed, 1),
        "probes_per_minute": round(total_probes / elapsed * 60, 1) if elapsed > 0 else 0,
        "edge_status_distribution": status_counts,
        "model": model,
        "probe_types": probe_types,
        "threshold": threshold,
        "dry_run": dry_run,
        "finalized_edges": finalize_counts,
        "published_graph": publish_stats,
    }

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 2: LLM-based counterfactual validation for CaSKG "
            "candidate edges."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--workspace",
        type=str,
        required=True,
        help=(
            "Path to the CaSKG workspace directory containing "
            "caskg_state.json from Phase 1."
        ),
    )
    parser.add_argument(
        "--skills-dir",
        type=str,
        required=True,
        help="Path to the skillsets directory containing SKILL.md files.",
    )
    parser.add_argument(
        "--probe-types",
        type=str,
        default="removal,substitution,reordering",
        help=(
            "Comma-separated list of probe types to run. "
            "Options: removal, substitution, reordering. "
            "(default: removal,substitution,reordering)"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help=(
            "Minimum association_score for an edge to be selected for "
            "validation. (default: 0.05)"
        ),
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=500,
        help=(
            "Maximum number of edges to validate. Edges are selected in "
            "descending order of association_score. 0 means no limit. "
            "(default: 500)"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help=(
            "Resume from the last saved progress. Skips probes already "
            "recorded in validation_log.jsonl."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Simulate LLM responses with random values (no API calls).",
    )
    parser.add_argument(
        "--call-delay",
        type=float,
        default=DEFAULT_CALL_DELAY,
        help=(
            f"Seconds between LLM API calls for rate limiting. "
            f"(default: {DEFAULT_CALL_DELAY})"
        ),
    )
    parser.add_argument(
        "--batch-concurrency",
        type=int,
        default=DEFAULT_BATCH_CONCURRENCY,
        help=(
            f"Number of edges to validate in parallel. "
            f"(default: {DEFAULT_BATCH_CONCURRENCY})"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. (default: 42)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable debug-level logging.",
    )
    parser.add_argument(
        "--confirm-threshold",
        type=float,
        default=CONFIRM_THRESHOLD,
        help=f"P(causal) above this => confirmed_causal. (default: {CONFIRM_THRESHOLD})",
    )
    parser.add_argument(
        "--reject-threshold",
        type=float,
        default=REJECT_THRESHOLD,
        help=f"P(causal) below this => rejected_non_causal. (default: {REJECT_THRESHOLD})",
    )
    parser.add_argument(
        "--finalize-unselected",
        dest="finalize_unselected",
        action="store_true",
        default=True,
        help=(
            "Freeze validation-budget leftovers as deferred_unvalidated "
            "after the pass. This keeps untested candidates out of Phase-3 "
            "retrieval without treating them as rejected evidence. (default)"
        ),
    )
    parser.add_argument(
        "--no-finalize-unselected",
        dest="finalize_unselected",
        action="store_false",
        help="Leave unselected eligible edges as unverified for another active validation pass.",
    )
    parser.add_argument(
        "--finalize-uncertain",
        dest="finalize_uncertain",
        action="store_true",
        default=True,
        help=(
            "Freeze selected edges that remain posterior-uncertain as "
            "deferred_uncertain after probes. (default)"
        ),
    )
    parser.add_argument(
        "--no-finalize-uncertain",
        dest="finalize_uncertain",
        action="store_false",
        help="Leave selected but still uncertain edges as unverified.",
    )
    parser.add_argument(
        "--include-deferred",
        action="store_true",
        default=False,
        help=(
            "Re-open deferred_unvalidated/deferred_uncertain edges for another "
            "validation pass. Deferred edges are reset to unverified when selected."
        ),
    )

    return parser


def print_summary(summary: dict[str, Any]) -> None:
    """Print a formatted summary of the validation run."""
    print("\n" + "=" * 70)
    print("PHASE 2 VALIDATION SUMMARY")
    print("=" * 70)

    print(f"  Model:                 {summary['model']}")
    print(f"  Probe types:           {', '.join(summary['probe_types'])}")
    print(f"  Association threshold: {summary['threshold']}")
    print(f"  Dry run:               {summary['dry_run']}")
    print()
    print(f"  Total edges in graph:  {summary['total_edges_in_graph']}")
    print(f"  Edges selected:        {summary['edges_selected']}")
    print(f"  Edges processed:       {summary['edges_processed']}")
    print(f"  Total probes run:      {summary['total_probes_run']}")
    print(f"  Probes skipped (resume): {summary['probes_skipped_resume']}")
    print(f"  Probes failed:         {summary['probes_failed']}")
    print(f"  Elapsed time:          {summary['elapsed_seconds']:.1f}s")
    print(f"  Rate:                  {summary['probes_per_minute']:.1f} probes/min")
    print()
    print("  Edge status distribution:")
    for status, count in sorted(summary["edge_status_distribution"].items()):
        print(f"    {status:<25s} {count:>6d}")
    finalized = summary.get("finalized_edges")
    if finalized:
        print()
        print("  Finalized unresolved frontier:")
        for status, count in sorted(finalized.items()):
            print(f"    {status:<25s} {count:>6d}")
    published = summary.get("published_graph")
    if published:
        print()
        print("  Published retrieval graph:")
        for key, count in sorted(published.items()):
            print(f"    {key:<25s} {count:>6d}")

    print("=" * 70 + "\n")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Update module-level thresholds from CLI args
    global CONFIRM_THRESHOLD, REJECT_THRESHOLD
    CONFIRM_THRESHOLD = args.confirm_threshold
    REJECT_THRESHOLD = args.reject_threshold

    # Logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load .env if dotenv is available
    try:
        from dotenv import load_dotenv
        env_path = _PROJECT_ROOT / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            logger.info("Loaded environment from %s", env_path)
    except ImportError:
        logger.debug("python-dotenv not installed; using existing env vars")

    # Parse probe types
    probe_types = [p.strip().lower() for p in args.probe_types.split(",")]
    for pt in probe_types:
        if pt not in ALL_PROBE_TYPES:
            logger.error("Invalid probe type: %s. Must be one of %s", pt, ALL_PROBE_TYPES)
            return 1

    # Resolve paths (relative to project root if not absolute)
    workspace = Path(args.workspace)
    if not workspace.is_absolute():
        workspace = _PROJECT_ROOT / workspace

    skills_dir = Path(args.skills_dir)
    if not skills_dir.is_absolute():
        skills_dir = _PROJECT_ROOT / skills_dir

    logger.info("Workspace: %s", workspace)
    logger.info("Skills dir: %s", skills_dir)

    try:
        summary = run_validation(
            workspace=workspace,
            skills_dir=skills_dir,
            probe_types=probe_types,
            threshold=args.threshold,
            max_edges=args.max_edges,
            resume=args.resume,
            dry_run=args.dry_run,
            call_delay=args.call_delay,
            batch_concurrency=args.batch_concurrency,
            seed=args.seed,
            finalize_unselected=args.finalize_unselected,
            finalize_uncertain=args.finalize_uncertain,
            include_deferred=args.include_deferred,
        )
        print_summary(summary)

        # Save summary as JSON alongside the state
        summary_path = workspace / "validation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info("Summary saved to %s", summary_path)

        return 0

    except KeyboardInterrupt:
        logger.warning("Validation interrupted by user. Progress has been saved.")
        return 1
    except Exception as exc:
        logger.exception("Validation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
