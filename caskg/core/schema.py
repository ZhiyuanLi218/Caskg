from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Iterable, Optional

from fast_graphrag._types import BTNode, BTEdge, TSerializable
from pydantic import BaseModel, Field


class CausalEdgeStatus:
    """Valid status values for causal edges."""

    UNVERIFIED = "unverified"
    CONFIRMED_CAUSAL = "confirmed_causal"
    PENDING_RETEST = "pending_retest"
    REJECTED_NON_CAUSAL = "rejected_non_causal"
    DEFERRED_UNVALIDATED = "deferred_unvalidated"
    DEFERRED_UNCERTAIN = "deferred_uncertain"
    STABLE = "stable"
    DECAYED = "decayed"
    PRUNED = "pruned"

    ALL = (
        UNVERIFIED,
        CONFIRMED_CAUSAL,
        PENDING_RETEST,
        REJECTED_NON_CAUSAL,
        DEFERRED_UNVALIDATED,
        DEFERRED_UNCERTAIN,
        STABLE,
        DECAYED,
        PRUNED,
    )


def _split_multivalue(text: str) -> list[str]:
    if not text:
        return []
    return [part.strip() for part in text.split("\n") if part.strip()]


def _serialize_list(values: list[str]) -> str:
    return "\n".join(value.strip() for value in values if value and value.strip())


def _parse_json_list(payload: str, fallback: str = "") -> list[str]:
    if payload:
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]

    return _split_multivalue(fallback)


@dataclass
class SkillNode(BTNode, TSerializable):
    description: str = ""
    one_line_capability: str = ""
    inputs: str = ""
    outputs: str = ""
    input_schema_json: str = "[]"
    output_schema_json: str = "[]"
    domain_tags: str = ""
    tooling: str = ""
    example_tasks: str = ""
    script_entrypoints: str = ""
    compatibility: str = ""
    allowed_tools: str = ""
    source_path: str = ""
    rendered_snippet: str = ""
    raw_content: str = ""
    metadata_json: str = "{}"
    skill_id: str = ""
    type: str = "Skill"
    preconditions: str = ""
    effects: str = ""
    validators_json: str = "[]"
    failure_modes_json: str = "[]"
    context_profile_json: str = "{}"

    F_TO_CONTEXT = [
        "name",
        "description",
        "one_line_capability",
        "inputs",
        "outputs",
        "domain_tags",
        "tooling",
        "example_tasks",
        "script_entrypoints",
        "compatibility",
        "allowed_tools",
        "source_path",
        "rendered_snippet",
        "preconditions",
        "effects",
    ]

    @property
    def input_types(self) -> list[str]:
        return _parse_json_list(self.input_schema_json, self.inputs)

    @property
    def output_types(self) -> list[str]:
        return _parse_json_list(self.output_schema_json, self.outputs)

    @property
    def domain_tags_list(self) -> list[str]:
        return _split_multivalue(self.domain_tags)

    @property
    def tooling_list(self) -> list[str]:
        return _split_multivalue(self.tooling)

    @property
    def example_tasks_list(self) -> list[str]:
        return _split_multivalue(self.example_tasks)

    @property
    def script_entrypoints_list(self) -> list[str]:
        return _split_multivalue(self.script_entrypoints)

    @property
    def compatibility_list(self) -> list[str]:
        return _split_multivalue(self.compatibility)

    @property
    def allowed_tools_list(self) -> list[str]:
        return _split_multivalue(self.allowed_tools)

    @property
    def metadata(self) -> dict[str, Any]:
        try:
            value = json.loads(self.metadata_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @property
    def preconditions_list(self) -> list[str]:
        return _split_multivalue(self.preconditions)

    @property
    def effects_list(self) -> list[str]:
        return _split_multivalue(self.effects)

    @property
    def validators(self) -> list[str]:
        return _parse_json_list(self.validators_json)

    @property
    def failure_modes(self) -> list[str]:
        return _parse_json_list(self.failure_modes_json)

    @property
    def context_profile(self) -> dict[str, Any]:
        try:
            value = json.loads(self.context_profile_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    def to_str(self) -> str:
        parts = [f"[{self.type}] {self.name}"]
        if self.description:
            parts.append(f"[DESCRIPTION] {self.description}")
        if self.one_line_capability:
            parts.append(f"[CAPABILITY] {self.one_line_capability}")
        if self.inputs:
            parts.append(f"[INPUTS] {self.inputs}")
        if self.outputs:
            parts.append(f"[OUTPUTS] {self.outputs}")
        if self.domain_tags:
            parts.append(f"[DOMAIN_TAGS] {self.domain_tags}")
        if self.tooling:
            parts.append(f"[TOOLING] {self.tooling}")
        if self.example_tasks:
            parts.append(f"[EXAMPLE_TASKS] {self.example_tasks}")
        if self.script_entrypoints:
            parts.append(f"[SCRIPT_ENTRYPOINTS] {self.script_entrypoints}")
        if self.compatibility:
            parts.append(f"[COMPATIBILITY] {self.compatibility}")
        if self.allowed_tools:
            parts.append(f"[ALLOWED_TOOLS] {self.allowed_tools}")
        if self.rendered_snippet:
            parts.append(f"[SNIPPET] {self.rendered_snippet}")
        return "\n".join(parts)

    def render_for_agent(self, max_chars: int | None = None) -> str:
        content = self.raw_content or self.rendered_snippet or self.to_str()
        if max_chars is not None and max_chars > 0 and len(content) > max_chars:
            content = f"{content[: max_chars - 3].rstrip()}..."

        header = [
            f"## Skill: {self.name}",
            f"Source: {self.source_path or 'inline'}",
        ]
        return "\n".join(header + [content])

    @staticmethod
    def from_lists(
        *,
        name: str,
        description: str,
        one_line_capability: str = "",
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        domain_tags: list[str] | None = None,
        tooling: list[str] | None = None,
        example_tasks: list[str] | None = None,
        script_entrypoints: list[str] | None = None,
        compatibility: list[str] | None = None,
        allowed_tools: list[str] | None = None,
        source_path: str = "",
        rendered_snippet: str = "",
        raw_content: str = "",
        metadata: dict[str, Any] | None = None,
        skill_id: str = "",
        type: str = "Skill",
        preconditions: list[str] | None = None,
        effects: list[str] | None = None,
        validators: list[str] | None = None,
        failure_modes: list[str] | None = None,
        context_profile: dict[str, Any] | None = None,
    ) -> "SkillNode":
        input_values = inputs or []
        output_values = outputs or []
        domain_tag_values = domain_tags or []
        tooling_values = tooling or []
        example_task_values = example_tasks or []
        script_entrypoint_values = script_entrypoints or []
        compatibility_values = compatibility or []
        allowed_tool_values = allowed_tools or []
        metadata_value = metadata or {}
        precondition_values = preconditions or []
        effect_values = effects or []
        validator_values = validators or []
        failure_mode_values = failure_modes or []
        context_profile_value = context_profile or {}

        return SkillNode(
            name=name,
            description=description,
            one_line_capability=one_line_capability,
            inputs=_serialize_list(input_values),
            outputs=_serialize_list(output_values),
            input_schema_json=json.dumps(input_values),
            output_schema_json=json.dumps(output_values),
            domain_tags=_serialize_list(domain_tag_values),
            tooling=_serialize_list(tooling_values),
            example_tasks=_serialize_list(example_task_values),
            script_entrypoints=_serialize_list(script_entrypoint_values),
            compatibility=_serialize_list(compatibility_values),
            allowed_tools=_serialize_list(allowed_tool_values),
            source_path=source_path,
            rendered_snippet=rendered_snippet,
            raw_content=raw_content,
            metadata_json=json.dumps(metadata_value),
            skill_id=skill_id or source_path or name,
            type=type,
            preconditions=_serialize_list(precondition_values),
            effects=_serialize_list(effect_values),
            validators_json=json.dumps(validator_values),
            failure_modes_json=json.dumps(failure_mode_values),
            context_profile_json=json.dumps(context_profile_value),
        )

    @staticmethod
    def to_attrs(
        node: "SkillNode | None" = None,
        nodes: Iterable["SkillNode"] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if node is not None:
            return {
                "description": node.description,
                "one_line_capability": node.one_line_capability,
                "inputs": node.inputs,
                "outputs": node.outputs,
                "input_schema_json": node.input_schema_json,
                "output_schema_json": node.output_schema_json,
                "domain_tags": node.domain_tags,
                "tooling": node.tooling,
                "example_tasks": node.example_tasks,
                "script_entrypoints": node.script_entrypoints,
                "compatibility": node.compatibility,
                "allowed_tools": node.allowed_tools,
                "source_path": node.source_path,
                "rendered_snippet": node.rendered_snippet,
                "raw_content": node.raw_content,
                "metadata_json": node.metadata_json,
                "skill_id": node.skill_id,
                "type": node.type,
                "preconditions": node.preconditions,
                "effects": node.effects,
                "validators_json": node.validators_json,
                "failure_modes_json": node.failure_modes_json,
                "context_profile_json": node.context_profile_json,
            }
        if nodes is not None:
            nodes_list = list(nodes)
            return {
                "description": [item.description for item in nodes_list],
                "one_line_capability": [item.one_line_capability for item in nodes_list],
                "inputs": [item.inputs for item in nodes_list],
                "outputs": [item.outputs for item in nodes_list],
                "input_schema_json": [item.input_schema_json for item in nodes_list],
                "output_schema_json": [item.output_schema_json for item in nodes_list],
                "domain_tags": [item.domain_tags for item in nodes_list],
                "tooling": [item.tooling for item in nodes_list],
                "example_tasks": [item.example_tasks for item in nodes_list],
                "script_entrypoints": [item.script_entrypoints for item in nodes_list],
                "compatibility": [item.compatibility for item in nodes_list],
                "allowed_tools": [item.allowed_tools for item in nodes_list],
                "source_path": [item.source_path for item in nodes_list],
                "rendered_snippet": [item.rendered_snippet for item in nodes_list],
                "raw_content": [item.raw_content for item in nodes_list],
                "metadata_json": [item.metadata_json for item in nodes_list],
                "skill_id": [item.skill_id for item in nodes_list],
                "type": [item.type for item in nodes_list],
                "preconditions": [item.preconditions for item in nodes_list],
                "effects": [item.effects for item in nodes_list],
                "validators_json": [item.validators_json for item in nodes_list],
                "failure_modes_json": [item.failure_modes_json for item in nodes_list],
                "context_profile_json": [item.context_profile_json for item in nodes_list],
            }
        return {}


@dataclass
class SkillEdge(BTEdge, TSerializable):
    description: str = ""
    type: str = "dependency"
    weight: float = 1.0
    confidence: float = 1.0
    chunks: list[Any] = field(default_factory=list)
    causal_score: float = 0.0
    uncertainty: float = 1.0
    association_score: float = 0.0
    transportability: float = 0.0
    status: str = "unverified"
    context_condition: str = ""
    intervention_count: int = 0
    alpha_posterior: float = 1.0
    beta_posterior: float = 1.0
    last_validated_episode: int = 0

    F_TO_CONTEXT = [
        "source",
        "target",
        "description",
        "type",
        "weight",
        "confidence",
        "causal_score",
        "uncertainty",
        "association_score",
        "transportability",
        "status",
        "context_condition",
        "intervention_count",
    ]

    @staticmethod
    def to_attrs(
        edge: "SkillEdge | None" = None,
        edges: Iterable["SkillEdge"] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if edge is not None:
            return {
                "description": edge.description,
                "type": edge.type,
                "weight": edge.weight,
                "confidence": edge.confidence,
                "chunks": edge.chunks if edge.chunks is not None else [],
                "causal_score": edge.causal_score,
                "uncertainty": edge.uncertainty,
                "association_score": edge.association_score,
                "transportability": edge.transportability,
                "status": edge.status,
                "context_condition": edge.context_condition,
                "intervention_count": edge.intervention_count,
                "alpha_posterior": edge.alpha_posterior,
                "beta_posterior": edge.beta_posterior,
                "last_validated_episode": edge.last_validated_episode,
            }
        if edges is not None:
            edges_list = list(edges)
            return {
                "description": [item.description for item in edges_list],
                "type": [item.type for item in edges_list],
                "weight": [item.weight for item in edges_list],
                "confidence": [item.confidence for item in edges_list],
                "chunks": [item.chunks if item.chunks is not None else [] for item in edges_list],
                "causal_score": [item.causal_score for item in edges_list],
                "uncertainty": [item.uncertainty for item in edges_list],
                "association_score": [item.association_score for item in edges_list],
                "transportability": [item.transportability for item in edges_list],
                "status": [item.status for item in edges_list],
                "context_condition": [item.context_condition for item in edges_list],
                "intervention_count": [item.intervention_count for item in edges_list],
                "alpha_posterior": [item.alpha_posterior for item in edges_list],
                "beta_posterior": [item.beta_posterior for item in edges_list],
                "last_validated_episode": [item.last_validated_episode for item in edges_list],
            }
        return {}


class GOSSkill(BaseModel):
    name: str = Field(..., description="The unique name of the skill")
    description: str = Field(..., description="Detailed description of what the skill does")
    one_line_capability: str = ""
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    domain_tags: list[str] = Field(default_factory=list)
    tooling: list[str] = Field(default_factory=list)
    example_tasks: list[str] = Field(default_factory=list)
    script_entrypoints: list[str] = Field(default_factory=list)
    compatibility: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    source_path: str = ""
    rendered_snippet: str = ""
    raw_content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    skill_id: str = ""


class GOSRelation(BaseModel):
    source: str = Field(..., description="The name of the source skill")
    target: str = Field(..., description="The name of the target skill")
    description: str = Field(..., description="Why these two skills are related")
    type: str = Field(
        default="dependency",
        description="Type: 'dependency', 'workflow', 'semantic', or 'alternative'",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class GOSRelationList(BaseModel):
    relations: list[GOSRelation] = Field(default_factory=list)


class GOSGraph(BaseModel):
    nodes: list[GOSSkill] = Field(default_factory=list)
    edges: list[GOSRelation] = Field(default_factory=list)


class QuerySchema(BaseModel):
    goal: str = Field(default="", description="Concise statement of the task intent")
    task_name: str = Field(default="", description="Short task slug when obvious")
    domain: list[str] = Field(default_factory=list, description="Narrow technical domains")
    operations: list[str] = Field(default_factory=list, description="Concrete operations, algorithms, or APIs")
    artifacts: list[str] = Field(default_factory=list, description="Files, formats, interfaces, or concrete objects")
    constraints: list[str] = Field(default_factory=list, description="Acceptance constraints and invariants")
    keywords: list[str] = Field(default_factory=list, description="High-value retrieval terms or short phrases")

    def to_query_text(self) -> str:
        parts: list[str] = []
        if self.goal:
            parts.append(self.goal)
        for values in (self.domain, self.operations, self.artifacts, self.constraints, self.keywords):
            if values:
                parts.extend(values)
        if self.task_name:
            parts.append(self.task_name)
        return "\n".join(part.strip() for part in parts if part and part.strip())


class RetrievalBudget(BaseModel):
    seed_top_k: int
    seed_candidate_top_k_semantic: int
    seed_candidate_top_k_lexical: int
    top_n: int
    max_chars_per_skill: int
    max_context_chars: int
    ppr_damping: float


class SkillSeed(BaseModel):
    name: str
    source_path: str = ""
    seed_weight: float
    semantic_rank: int


class RetrievedSkill(BaseModel):
    name: str
    description: str
    source_path: str = ""
    one_line_capability: str = ""
    score: float
    rerank_score: float = 0.0
    semantic_rank: int | None = None
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    domain_tags: list[str] = Field(default_factory=list)
    tooling: list[str] = Field(default_factory=list)
    example_tasks: list[str] = Field(default_factory=list)
    script_entrypoints: list[str] = Field(default_factory=list)
    compatibility: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    rendered_snippet: str = ""
    payload: str = ""


class RetrievedRelation(BaseModel):
    source: str
    target: str
    description: str
    type: str
    weight: float
    confidence: float = 1.0


class SkillRetrievalResult(BaseModel):
    query: str
    rewritten_query: QuerySchema = Field(default_factory=QuerySchema)
    budget: RetrievalBudget
    seeds: list[SkillSeed] = Field(default_factory=list)
    skills: list[RetrievedSkill] = Field(default_factory=list)
    relations: list[RetrievedRelation] = Field(default_factory=list)
    rendered_context: str = ""
    summary: str = ""


class SkillSyncResult(BaseModel):
    requested_skill_count: int
    existing_skill_count: int
    final_skill_count: int
    reused_count: int = 0
    inserted_count: int = 0
    updated_count: int = 0
    inserted_skill_names: list[str] = Field(default_factory=list)
    updated_skill_names: list[str] = Field(default_factory=list)
    prebuilt_working_dir: str = ""


# ---------------------------------------------------------------------------
# CaSKG (Counterfactual-Causal Skill Graph) Models
# ---------------------------------------------------------------------------


class ContextProfile(BaseModel):
    """Describes the context in which a skill or edge operates."""

    task_types: list[str] = Field(default_factory=list, description="Task types where this applies")
    environments: list[str] = Field(default_factory=list, description="Execution environments")
    domain_constraints: list[str] = Field(default_factory=list, description="Domain-specific constraints")
    frequency_distribution: dict[str, float] = Field(
        default_factory=dict,
        description="Distribution of usage frequency across contexts",
    )


class InterventionResult(BaseModel):
    """Records the outcome of a causal intervention experiment."""

    edge_source: str = Field(..., description="Source skill of the edge under test")
    edge_target: str = Field(..., description="Target skill of the edge under test")
    intervention_type: str = Field(
        ..., description="Type of intervention (e.g. 'remove_skill', 'block_dependency', 'substitute')"
    )
    task_id: str = Field(..., description="Identifier of the task used for the intervention")
    outcome_with: float = Field(..., description="Performance score with the edge/skill present")
    outcome_without: float = Field(..., description="Performance score with the edge/skill removed")
    delta: float = Field(..., description="outcome_with - outcome_without")
    environment: str = Field(default="", description="Environment in which the intervention was run")
    timestamp: float = Field(default=0.0, description="Unix timestamp of the intervention")


class GraphHealthReport(BaseModel):
    """Summary of graph health after a maintenance pass."""

    decayed_edges: list[dict[str, Any]] = Field(
        default_factory=list, description="Edges whose causal score has decayed below threshold"
    )
    missing_edge_candidates: list[dict[str, Any]] = Field(
        default_factory=list, description="Potential edges not yet in the graph"
    )
    structural_anomalies: list[str] = Field(
        default_factory=list, description="Detected structural issues (cycles, orphans, etc.)"
    )
    variant_edges: list[dict[str, Any]] = Field(
        default_factory=list, description="Edges with high variance across contexts"
    )
    coverage_score: float = Field(
        default=0.0, description="Fraction of known skill pairs with validated causal status"
    )


class CausalRetrievalResult(BaseModel):
    """Extended retrieval result that includes causal reasoning information."""

    query: str = ""
    rewritten_query: QuerySchema = Field(default_factory=QuerySchema)
    budget: Optional[RetrievalBudget] = None
    seeds: list[SkillSeed] = Field(default_factory=list)
    skills: list[RetrievedSkill] = Field(default_factory=list)
    relations: list[RetrievedRelation] = Field(default_factory=list)
    rendered_context: str = ""
    summary: str = ""
    causal_paths: list[list[str]] = Field(
        default_factory=list,
        description="Ordered causal paths from prerequisite skills to goal skills",
    )
    bundle_sufficiency_score: float = Field(
        default=0.0,
        description="Score indicating whether the retrieved skill bundle is causally sufficient",
    )
    causal_conflicts: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Detected conflicts between causal edges in the retrieval set",
    )
