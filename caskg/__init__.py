__all__ = [
    "SkillGraphRAG",
    "SkillNode",
    "SkillEdge",
    "GOSSkill",
    "GOSGraph",
    "CausalGraphEngine",
    "CausalRetrievalResult",
    "GraphHealthReport",
    "InterventionResult",
]


def __getattr__(name: str):
    if name == "SkillGraphRAG":
        from .core.engine import SkillGraphRAG

        return SkillGraphRAG
    if name in {"SkillNode", "SkillEdge", "GOSSkill", "GOSGraph"}:
        from .core.schema import GOSGraph, GOSSkill, SkillEdge, SkillNode

        exports = {
            "SkillNode": SkillNode,
            "SkillEdge": SkillEdge,
            "GOSSkill": GOSSkill,
            "GOSGraph": GOSGraph,
        }
        return exports[name]
    if name == "CausalGraphEngine":
        from .causal.engine import CausalGraphEngine

        return CausalGraphEngine
    if name in {"CausalRetrievalResult", "GraphHealthReport", "InterventionResult"}:
        from .core.schema import CausalRetrievalResult, GraphHealthReport, InterventionResult

        exports = {
            "CausalRetrievalResult": CausalRetrievalResult,
            "GraphHealthReport": GraphHealthReport,
            "InterventionResult": InterventionResult,
        }
        return exports[name]
    raise AttributeError(name)
