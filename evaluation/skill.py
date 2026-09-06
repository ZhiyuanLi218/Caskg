import json
import os
import re
import asyncio
import inspect
from datetime import datetime, timezone
from pathlib import Path
import yaml
from typing import Any
import sys


_SKILLS_REF_SRC = str(Path(__file__).resolve().parent)
if _SKILLS_REF_SRC not in sys.path:
    sys.path.insert(0, _SKILLS_REF_SRC)

# Try to import CaSKG engine.  The evaluation-side candidate skill path should
# stay aligned with GoS: CaSKG differs by the workspace graph it builds, not by
# an extra task-time causal reranker.
try:
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from caskg.core.engine import SkillGraphRAG, build_default_embedding_service, build_default_llm_service
    from caskg.core.schema import QuerySchema
except ImportError:
    SkillGraphRAG = None
    build_default_embedding_service = None
    build_default_llm_service = None
    QuerySchema = None

try:
    from .utils import get_llm_response
    from .skills_ref import to_prompt as skills_ref_to_prompt
except ImportError:
    from utils import get_llm_response
    from skills_ref import to_prompt as skills_ref_to_prompt

class SkillModule:
    def __init__(self, **kwargs):
        self.skills_dir = Path(kwargs.get("skills_dir", "skills"))
        self.model = kwargs.get("model", "gpt-4o")
        self.mode = kwargs.get("mode", "caskg") # "all_full", "vector", "caskg", "none"
        self.caskg_workspace = kwargs.get("caskg_workspace", None)
        self.enable_alfworld_gating = bool(kwargs.get("enable_alfworld_gating", False))

        self.last_retrieval_result: Any = None
        self.last_retrieval_status = "NOT_RUN"
        self.last_retrieval_summary = ""
        self.last_retrieved_skill_names = []
        self.last_retrieval_query = ""
        self.runtime_skill_events = []
        self.runtime_skill_count = 0
        self.runtime_last_injection_step = -999
        self.action_schema_templates = list(kwargs.get("action_schema_templates") or [])

        self.metadata = self._load_metadata()

        # Initialize CaSKG retrieval engine if needed
        if self.mode in {"vector", "caskg"} and not self.caskg_workspace:
            raise ValueError(f"{self.mode} mode requires `caskg_workspace`.")

        if self.mode in {"vector", "caskg"} and SkillGraphRAG is None:
            raise ImportError("Failed to import `caskg.core.engine.SkillGraphRAG`; retrieval is unavailable.")

        if self.mode in {"vector", "caskg"} and SkillGraphRAG and self.caskg_workspace:
            caskg_workspace = str(Path(self.caskg_workspace).expanduser().resolve())
            self.caskg_workspace = caskg_workspace
            self.rag = SkillGraphRAG(
                working_dir=caskg_workspace,
                config=SkillGraphRAG.Config(
                    working_dir=caskg_workspace,
                    prebuilt_working_dir=caskg_workspace,
                    llm_service=build_default_llm_service() if build_default_llm_service else None,
                    embedding_service=build_default_embedding_service() if build_default_embedding_service else None,
                    # Evaluation code passes the task text directly as the retrieval query.
                    # Skip internal LLM rewrite to avoid schema-format drift.
                    enable_query_rewrite=False,
                )
            )

        else:
            self.rag = None

    def _log(self, message):
        print(f"[SkillModule] {message}")

    def set_action_schema(self, templates):
        self.action_schema_templates = [
            str(template).strip()
            for template in templates or []
            if str(template).strip()
        ]

    def _is_alfworld_task(self, task):
        task_lower = task.lower()
        return "your task is to:" in task_lower or "you are in the middle of a room" in task_lower

    def _extract_alfworld_goal(self, task):
        match = re.search(r"your task is to:\s*(.+)", task, re.IGNORECASE)
        if match:
            return match.group(1).splitlines()[0].strip().rstrip('.')
        return task.strip()

    def _infer_task_type(self, goal):
        goal_lower = goal.lower()
        if "look at" in goal_lower or "examine" in goal_lower:
            return "examine"
        if "find two" in goal_lower or "put two" in goal_lower:
            return "put_two"
        if "clean" in goal_lower:
            return "clean_and_place"
        if "cool" in goal_lower:
            return "cool_and_place"
        if "heat" in goal_lower or "hot " in goal_lower:
            return "heat_and_place"
        if "put" in goal_lower:
            return "put"
        return "other"

    def _extract_required_state(self, goal):
        goal_lower = goal.lower()
        if "clean" in goal_lower:
            return "clean"
        if "cool" in goal_lower:
            return "cool"
        if "heat" in goal_lower or "hot " in goal_lower:
            return "hot"
        return "none"

    def _extract_count(self, goal):
        goal_lower = goal.lower()
        if "find two" in goal_lower or "put two" in goal_lower:
            return "2"
        if re.search(r"\bsome\b", goal_lower):
            return "some"
        if re.search(r"\ban?\b", goal_lower):
            return "1"
        return "unspecified"

    def _extract_target_receptacle(self, goal):
        patterns = [
            r"\b(?:in|into|inside|on|onto|under)\s+([a-z0-9]+)",
            r"\bto\s+([a-z0-9]+)$",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, goal, re.IGNORECASE)
            if matches:
                return matches[-1].lower()
        return "unknown"

    def _extract_device(self, goal):
        devices = [
            "desklamp",
            "microwave",
            "fridge",
            "sinkbasin",
            "coffeemachine",
            "stoveburner",
            "cabinet",
            "drawer",
            "dresser",
            "garbagecan",
            "diningtable",
            "countertop",
            "desk",
            "sidetable",
            "table",
            "toilet",
        ]
        goal_lower = goal.lower()
        for device in devices:
            if device in goal_lower:
                return device
        return "none"

    def _extract_primary_object(self, goal):
        goal_lower = goal.lower().rstrip('.')
        patterns = [
            r"(?:look at|examine)\s+the\s+([a-z0-9]+)",
            r"(?:look at|examine)\s+([a-z0-9]+)",
            r"(?:put|find|clean|cool|heat)\s+(?:a|an|some|two)?\s*(?:clean|cool|hot|heated)?\s*([a-z0-9]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, goal_lower)
            if match:
                candidate = match.group(1).lower()
                if candidate not in {"clean", "cool", "hot", "heated"}:
                    return candidate
        tokens = re.findall(r"[a-z0-9]+", goal_lower)
        stop = {
            "put", "find", "clean", "cool", "heat", "hot", "heated", "look", "at", "examine",
            "the", "a", "an", "some", "two", "in", "into", "inside", "on", "onto",
            "under", "with", "and", "it", "them"
        }
        for token in tokens:
            if token not in stop:
                return token
        return "unknown"

    def _build_alfworld_structured_query(self, task):
        goal = self._extract_alfworld_goal(task)
        task_type = self._infer_task_type(goal)
        obj = self._extract_primary_object(goal)
        required_state = self._extract_required_state(goal)
        target_receptacle = self._extract_target_receptacle(goal)
        count = self._extract_count(goal)
        device = self._extract_device(goal)
        if QuerySchema is None:
            return (
                "environment=alfworld; "
                f"task_type={task_type}; "
                f"goal={goal}; "
                f"object={obj}; "
                f"required_state={required_state}; "
                f"target_receptacle={target_receptacle}; "
                f"count={count}; "
                f"device={device}; "
                "actions=navigate,take,move,open,close,heat,cool,clean,use,look; "
                "optimization=shortest_valid_action_sequence"
            )

        schema = QuerySchema(
            goal=goal,
            task_name=f"alfworld-{task_type}",
            domain=["alfworld", "household manipulation", "embodied task planning"],
            operations=[
                task_type,
                "navigate",
                "take",
                "move",
                "open",
                "close",
                "heat",
                "cool",
                "clean",
                "look",
            ],
            artifacts=[obj, target_receptacle, device],
            constraints=[
                f"required_state={required_state}",
                f"count={count}",
                "optimize for shortest valid action sequence",
            ],
            keywords=[
                "environment=alfworld",
                f"object={obj}",
                f"target_receptacle={target_receptacle}",
                f"device={device}",
            ],
        )
        return schema.to_query_text()

    def _build_targeted_retrieval_query(self, task):
        return task.strip()

    def _effective_top_k(self, task, requested_top_k):
        return requested_top_k

    def _skill_confident_enough(self, skill):
        rerank_score = float(getattr(skill, "rerank_score", 0.0) or 0.0)
        score = float(getattr(skill, "score", 0.0) or 0.0)
        semantic_rank = getattr(skill, "semantic_rank", None)

        if rerank_score >= 0.60:
            return True
        if rerank_score >= 0.45 and semantic_rank is not None and semantic_rank <= 2:
            return True
        if score >= 0.30 and semantic_rank is not None and semantic_rank <= 2:
            return True
        return False

    def _filter_skills_for_task(self, task, result, *, source_label="retrieval"):
        return self._extract_skill_payloads(result)

    @staticmethod
    def _positive_int_env(name, default):
        try:
            value = int(os.environ.get(name, "") or default)
        except (TypeError, ValueError):
            value = default
        return max(1, value)

    def _extract_skill_payloads(self, result):
        skill_names = [skill.name for skill in result.skills]
        skill_payloads = [skill.payload for skill in result.skills]
        return skill_payloads, skill_names

    def should_generate_procedure(self, task):
        return False

    def _reset_retrieval_state(self):
        self.last_retrieval_result = None
        self.last_retrieval_status = "NOT_RUN"
        self.last_retrieval_summary = ""
        self.last_retrieved_skill_names = []
        self.last_retrieval_query = ""
        self.runtime_skill_events = []
        self.runtime_skill_count = 0
        self.runtime_last_injection_step = -999

    def _set_retrieval_state(self, status, summary="", skill_names=None, result=None):
        self.last_retrieval_status = status
        self.last_retrieval_summary = summary or ""
        self.last_retrieved_skill_names = list(skill_names or [])
        self.last_retrieval_result = result

    def _all_metadata_entries(self):
        return [
            {
                "name": name,
                "description": data.get("description", ""),
                "skill_dir": data.get("skill_dir", ""),
            }
            for name, data in sorted(self.metadata.items())
        ]

    def _all_metadata_context(self):
        lines = []
        for item in self._all_metadata_entries():
            lines.append(f"- {item['name']}: {item['description']}")
        return "\n".join(lines)

    def _all_metadata_skill_bundle(self):
        metadata_context = self._all_metadata_context()
        if not metadata_context:
            return []
        return [
            "=== Full Skill Library Metadata ===\n"
            "The following is the full available skill library. Treat it as capability exposure, not as a pre-filtered retrieval result.\n\n"
            f"{metadata_context}"
        ]

    def get_all_full_exposure_messages(self):
        if self.mode != "all_full":
            return []

        skill_dirs = [Path(item["skill_dir"]) for item in self._all_metadata_entries() if item.get("skill_dir")]
        if not skill_dirs:
            return []

        prompt_block = skills_ref_to_prompt(skill_dirs)
        return [
            "The following block lists the full available skill library in Anthropic skills-ref format. "
            "This is not a pre-filtered retrieval result. Use it as a catalog of available capabilities. "
            "If a skill looks relevant, prefer reading only the few most relevant skills by exact name.\n\n"
            f"{prompt_block}"
        ]

    def get_all_full_exposure_message(self):
        messages = self.get_all_full_exposure_messages()
        if not messages:
            return ""
        return messages[0]

    def get_agent_skill_request_message(self):
        if self.mode == "none":
            return ""

        lines = [
            "Tool-style skill access is available in this run.",
            "Use it when you are blocked, the syntax is unclear, the retrieved skills look mismatched to the current blocker, or 1-2 actions already failed.",
            "Use skills conditionally, not by default: if the next required step is already obvious from the current state, act directly instead of retrieving.",
            "Prefer retrieval when the exact syntax is unclear, the task needs a multi-step procedure or tool setup, the current shortlist looks mismatched, or 1-2 recent actions failed.",
            "Prefer READ_SKILL when you already have a promising exact skill name. For unfamiliar procedures or interface details, do not guess twice in a row; retrieve first, then read the single best skill before continuing.",
            "Mirror the current task vocabulary in retrieval queries. Reuse the task's own artifact, operation, interface, constraint, and failure words instead of inventing a different domain.",
            "In a request turn, output exactly two lines: `Thought: ...` and `SkillRequest: ...`. Do not output an `Action:` line in the same turn.",
            "Skill responses are instructions or retrieved text only. Never treat them as runtime observations, never invent feedback lines inside your response, and continue using only real task feedback to decide actions.",
        ]

        if self.mode in {"caskg"}:
            lines.extend([
                "Available requests:",
                "- `SkillRequest: CASKG_RETRIEVE <short focused query>` to search CaSKG again. Prefer this first when you are blocked or the current shortlist looks noisy, generic, or off-task.",
                "- `SkillRequest: READ_SKILL <exact skill name>` to read one concrete skill after CaSKG has surfaced a promising candidate.",
                "Examples:",
                "- `Thought: I already have a good shortlist and need the exact instructions from one candidate.`",
                "  `SkillRequest: READ_SKILL <exact shortlisted skill name>`",
                "- `Thought: The current shortlist looks noisy. I need a narrower retrieval grounded in the current task.`",
                "  `SkillRequest: CASKG_RETRIEVE <goal artifact> <operation> <constraint> <failure signal>`",
                "- `Thought: I failed twice and need retrieval that mirrors the current blocker instead of guessing again.`",
                "  `SkillRequest: CASKG_RETRIEVE <task-specific words from the current blocker only>`",
                "Use skill requests sparingly, only when they directly help the next action. Prefer a two-step pattern: `CASKG_RETRIEVE` to shortlist candidates, then `READ_SKILL` for the single best candidate before guessing again.",
            ])
        elif self.mode == "vector":
            lines.extend([
                "Available requests:",
                "- `SkillRequest: VECTOR_RETRIEVE <short focused query>` to run vector-only retrieval again. This uses embedding similarity only, without graph propagation or lexical expansion.",
                "- `SkillRequest: READ_SKILL <exact skill name>` to read a known skill file. Use this only when you already know the exact skill you want.",
                "Examples:",
                "- `Thought: I already have a good shortlist and need the exact instructions from one candidate.`",
                "  `SkillRequest: READ_SKILL <exact shortlisted skill name>`",
                "- `Thought: The current shortlist looks noisy. I need a narrower vector retrieval grounded in the current task.`",
                "  `SkillRequest: VECTOR_RETRIEVE <goal artifact> <operation> <constraint> <failure signal>`",
                "- `Thought: I failed twice and need vector retrieval that mirrors the current blocker instead of guessing again.`",
                "  `SkillRequest: VECTOR_RETRIEVE <task-specific words from the current blocker only>`",
                "Use skill requests sparingly, only when they directly help the next action. In vector mode, prefer `VECTOR_RETRIEVE` before guessing again, and `READ_SKILL` only after a specific skill name looks relevant.",
            ])
        elif self.mode == "all_full":
            lines.extend([
                "Available requests:",
                "- `SkillRequest: READ_SKILL <exact skill name>` to read a known skill file.",
                "Examples:",
                "- `Thought: The full catalog already shows a likely match and I need its exact instructions.`",
                "  `SkillRequest: READ_SKILL <exact skill name already visible in the catalog>`",
                "Use skill requests sparingly, only when they directly help the next action. In all_full mode, do not attempt retrieval; read a specific skill only when the full catalog already reveals a directly relevant candidate.",
            ])

        else:
            return ""
        return "\n".join(lines)

    def _skill_catalog_entries(self, skill_names):
        entries = []
        for name in skill_names or []:
            meta = self.metadata.get(name, {})
            entries.append(
                {
                    "name": name,
                    "description": meta.get("description", ""),
                    "skill_dir": meta.get("skill_dir", ""),
                }
            )
        return entries

    def _format_retrieval_shortlist(self, header, query, skill_names, source_label):
        if not skill_names:
            return f"{header}\n\nNo relevant skills were retrieved."

        lines = [
            header,
            f"Query: {query}",
            f"Shortlisted {source_label} candidates:",
        ]
        for entry in self._skill_catalog_entries(skill_names[:3]):
            description = entry["description"] or "No description available."
            lines.append(f"- {entry['name']}: {description}")
            if entry["skill_dir"]:
                lines.append(f"  Source: {entry['skill_dir']}/SKILL.md")
        lines.extend([
            "Do not assume these summaries are enough to execute correctly.",
            "If one candidate looks directly relevant to the current blocker, issue `SkillRequest: READ_SKILL <exact skill name>` before trying another uncertain action.",
            "These retrieved summaries are not environment feedback. Do not invent observations or action feedback from a skill response.",
        ])
        return "\n".join(lines)

    def _load_metadata(self):
        """Load existing metadata from file."""
        metadata = {}
        if not self.skills_dir.exists():
            return metadata
            
        for skill_dir in self.skills_dir.iterdir():
            if skill_dir.is_dir():
                skill_md_path = skill_dir / "SKILL.md"
                if skill_md_path.exists():
                    try:
                        content = skill_md_path.read_text(encoding="utf-8")
                        if content.strip().startswith('---'):
                            parts = content.split('---', 2)
                            if len(parts) >= 3:
                                header_data = yaml.safe_load(parts[1])
                                if isinstance(header_data, dict) and header_data.get('name') and header_data.get('description'):
                                    metadata[header_data['name']] = {
                                        'description': header_data['description'],
                                        'skill_dir': str(skill_dir)
                                    }
                    except Exception as e:
                        print(f"[ERROR] Failed to parse SKILL.md for {skill_dir.name}: {e}")
        return metadata
    
    def retrieve_relevant_skills(self, task, top_k=15):
        self._reset_retrieval_state()
        effective_top_k = self._effective_top_k(task, top_k)
        retrieval_query = self._build_targeted_retrieval_query(task)
        self.last_retrieval_query = retrieval_query
        self._log(
            f"retrieve_relevant_skills start mode={self.mode} top_k={top_k} effective_top_k={effective_top_k} task_chars={len(task)} retrieval_query={retrieval_query!r}"
        )

        if self.mode == "none":
            self._log("mode=none, skipping retrieval")
            return []
            
        if self.mode in {"vector", "caskg"} and self.rag:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if self.mode == "caskg":
                self._log(f"starting CaSKG async_retrieve workspace={self.caskg_workspace}")
                result = loop.run_until_complete(
                    self.rag.async_retrieve(retrieval_query, top_n=effective_top_k)
                )
                skill_payloads, skill_names = self._filter_skills_for_task(
                    task,
                    result,
                    source_label="caskg",
                )
                status = "SKILL_HIT" if skill_names else "NO_SKILL_HIT"
                summary = result.summary
                if not skill_names:
                    summary = "CaSKG graph retrieval returned no matching skills."
                self._log(f"CaSKG async_retrieve finished status={status} n_skills={len(skill_names)}")
            else:
                self._log(f"starting vector async_retrieve workspace={self.caskg_workspace}")
                result = loop.run_until_complete(self.rag.async_retrieve_vector(retrieval_query, top_n=effective_top_k))
                skill_payloads, skill_names = self._filter_skills_for_task(
                    task,
                    result,
                    source_label="vector",
                )
                status = "SKILL_HIT" if skill_names else "NO_SKILL_HIT"
                summary = result.summary
                self._log(f"vector async_retrieve finished status={status} n_skills={len(skill_names)}")

            self._set_retrieval_state(
                status=status,
                summary=summary,
                skill_names=skill_names,
                result=result,
            )
            return skill_payloads

        if self.mode == "all_full":
            metadata_entries = self._all_metadata_entries()
            skill_names = [entry["name"] for entry in metadata_entries]
            status = "SKILL_HIT" if skill_names else "NO_SKILL_HIT"
            summary = (
                f"Exposed full skill metadata library in a single initial dialogue message ({len(skill_names)} skills). "
                "This matches the all-skills capability-exposure baseline rather than retrieval-time shortlisting."
            )
            self._set_retrieval_state(
                status=status,
                summary=summary,
                skill_names=skill_names,
                result={"skill_names": skill_names, "mode": "all_full"},
            )
            self._log(f"all_full exposure finished status={status} n_skills={len(skill_names)}")
            return []

        self._set_retrieval_state(status="NO_SKILL_HIT", summary="No retrieval configured for this mode.")
        return []

    def get_retrieval_guidance(self):
        if self.mode not in {"vector", "caskg"} or self.last_retrieval_result is None:
            return ""

        if self.last_retrieval_status != "SKILL_HIT" or not self.last_retrieved_skill_names:
            return ""

        summary_limit = self._positive_int_env(
            "CASKG_GUIDANCE_SUMMARY_TOP_K",
            len(self.last_retrieved_skill_names) if self.mode == "caskg" else 3,
        )
        top_skills = self.last_retrieved_skill_names[:summary_limit]
        if self.mode == "caskg":
            title = "CaSKG causal retrieval guidance:"
        else:
            title = "Vector-skills retrieval guidance:"
        content_parts = [
            title,
            f"Retrieval Status: {self.last_retrieval_status}",
        ]
        if top_skills:
            content_parts.append("Top retrieved skills: " + ", ".join(top_skills))
            skill_lines = ["Retrieved skill summaries:"]
            for entry in self._skill_catalog_entries(top_skills):
                description = entry["description"] or "No description available."
                skill_lines.append(f"- {entry['name']}: {description}")
            content_parts.append("\n".join(skill_lines))
            if self.mode == "caskg":
                detail_limit = self._positive_int_env(
                    "CASKG_INITIAL_SKILL_DETAIL_TOP_K",
                    4,
                )
                detail_chars = self._positive_int_env(
                    "CASKG_INITIAL_SKILL_DETAIL_CHARS",
                    900,
                )
                detail_names = self._priority_detail_skill_names(
                    self.last_retrieved_skill_names,
                    detail_limit,
                )
                detail_payloads = self._get_skill_contents(detail_names)
                if detail_payloads:
                    detail_lines = [
                        "Initial retrieved skill details:",
                        "Treat these as compact procedure guidance for the current task. Follow them when they match the current observation; do not wait to request the same skill again.",
                        self._action_schema_invariant(),
                    ]
                    for name, payload in zip(detail_names, detail_payloads):
                        detail_lines.append(
                            f"\n=== {name} ===\n"
                            f"{self._skill_payload_for_prompt(payload, detail_chars)}"
                        )
                    content_parts.append("\n".join(detail_lines))
        content_parts.append(
            "Use retrieval only as weak high-level guidance. Prioritize the shortest path from current observation to task completion."
        )
        content_parts.append(
            "Do not follow a rigid checklist when the current state already exposes the next required step."
        )
        content_parts.append(
            "If runtime feedback or the task score indicates completion, stop issuing new actions immediately."
        )
        content_parts.append(
            "Do not infer completion from your own summary, a skill response, or no-effect/invalid feedback; completion requires an explicit success signal from the environment or evaluator."
        )
        content_parts.extend(self._skill_response_invariants())
        if self.mode in {"caskg"}:
            content_parts.append(
                "If the current retrieved skills look mismatched to the blocker, or 1-2 actions already failed, issue `SkillRequest: CASKG_RETRIEVE <short focused query>`. Treat retrieval as a shortlist step and prefer `READ_SKILL` for the single best candidate before another uncertain action."
            )
        elif self.mode == "vector":
            content_parts.append(
                "If the current retrieved skills look mismatched to the blocker, or 1-2 actions already failed, issue `SkillRequest: VECTOR_RETRIEVE <short focused query>`. After vector retrieval surfaces a plausible exact skill name, prefer `READ_SKILL` for that single candidate before another uncertain action."
            )
        return "\n\n".join(part for part in content_parts if part)

    @staticmethod
    def _priority_detail_skill_names(skill_names, limit):
        names = list(dict.fromkeys(skill_names or []))
        return names[: max(1, int(limit or 1))]

    def _get_skill_contents(self, skill_names):
        skill_contents = []
        for name in skill_names:
            if name in self.metadata:
                skill_dir = Path(self.metadata[name]['skill_dir'])
                combined_text = f"=== Skill: {name} ===\n"
                for file_path in skill_dir.rglob('*'):
                    if file_path.is_file():
                        try:
                            content = file_path.read_text(encoding='utf-8')
                            combined_text += f"\n[File: {file_path.name}]\n{content}\n"
                        except: continue
                skill_contents.append(combined_text)
        return skill_contents

    def _parse_skill_request(self, response):
        if not isinstance(response, str):
            return None, ""

        patterns = [
            r"^SkillRequest:\s*(.+)$",
            r"^Action:\s*SkillRequest:\s*(.+)$",
        ]
        match = None
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                break
        if not match:
            return None, ""

        payload = match.group(1).strip()
        if not payload:
            return None, ""

        upper = payload.upper()
        if upper.startswith("READ_SKILL "):
            return "read_skill", payload[len("READ_SKILL "):].strip()
        if upper.startswith("CASKG_RETRIEVE "):
            return "caskg_retrieve", payload[len("CASKG_RETRIEVE "):].strip()
        if upper.startswith("VECTOR_RETRIEVE "):
            return "vector_retrieve", payload[len("VECTOR_RETRIEVE "):].strip()
        return None, payload

    def _record_runtime_skill_event(self, step, trigger, query, skill_names):
        self.runtime_skill_count += 1
        self.runtime_last_injection_step = step
        self.runtime_skill_events.append(
            {
                "step": step,
                "trigger": trigger,
                "query": query,
                "skill_names": list(skill_names or []),
            }
        )

    def _format_agent_skill_response(self, header, skill_names, skill_payloads):
        if not skill_payloads:
            return ""
        clipped_payloads = [
            self._skill_payload_for_prompt(payload, 2200)
            for payload in skill_payloads[:2]
        ]
        lines = [header, "Use this only if it directly improves the next action."]
        if skill_names:
            lines.append("Selected skills: " + ", ".join(skill_names[:2]))
        lines.extend(self._skill_response_invariants())
        return "\n\n".join(lines + clipped_payloads)

    def handle_agent_skill_request(self, task, response, current_step):
        request_type, payload = self._parse_skill_request(response)
        if not request_type:
            return ""

        if request_type == "read_skill":
            skill_name = payload
            skill_payloads = self._get_skill_contents([skill_name])[:1]
            if not skill_payloads:
                return (
                    f"Skill request could not be fulfilled: skill `{skill_name}` was not found. "
                    "Use an exact skill name from the available skill list or retrieval results."
                )
            self._record_runtime_skill_event(current_step, "agent_request:read_skill", skill_name, [skill_name])
            return self._format_agent_skill_response(
                f"Skill request fulfilled: READ_SKILL {skill_name}",
                [skill_name],
                skill_payloads,
            )

        if request_type == "caskg_retrieve":
            if self.mode not in {"caskg"} or not self.rag:
                return "Skill request could not be fulfilled: CASKG_RETRIEVE is only available in caskg mode."
            query = payload
            if not query:
                return "Skill request could not be fulfilled: empty CASKG_RETRIEVE query."
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.rag.async_retrieve(query, top_n=2))
            skill_payloads, skill_names = self._filter_skills_for_task(
                task,
                result,
                source_label="caskg",
            )
            skill_payloads = skill_payloads[:2]
            skill_names = skill_names[:2]
            if not skill_names:
                return f"Skill request fulfilled: CASKG_RETRIEVE {query}\n\nNo relevant skills were retrieved."
            self._record_runtime_skill_event(current_step, "agent_request:caskg_retrieve", query, skill_names)
            return self._format_retrieval_shortlist(
                f"Skill request fulfilled: CASKG_RETRIEVE {query}",
                query,
                skill_names,
                "CaSKG",
            )

        if request_type == "vector_retrieve":
            if self.mode != "vector" or not self.rag:
                return "Skill request could not be fulfilled: VECTOR_RETRIEVE is only available in vector mode."
            query = payload
            if not query:
                return "Skill request could not be fulfilled: empty VECTOR_RETRIEVE query."
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.rag.async_retrieve_vector(query, top_n=2))
            skill_payloads, skill_names = self._filter_skills_for_task(
                task,
                result,
                source_label="vector",
            )
            skill_payloads = skill_payloads[:2]
            skill_names = skill_names[:2]
            if not skill_names:
                return f"Skill request fulfilled: VECTOR_RETRIEVE {query}\n\nNo relevant skills were retrieved."
            self._record_runtime_skill_event(current_step, "agent_request:vector_retrieve", query, skill_names)
            return self._format_retrieval_shortlist(
                f"Skill request fulfilled: VECTOR_RETRIEVE {query}",
                query,
                skill_names,
                "vector",
            )

        return ""

    @staticmethod
    def _clip_text(text, max_chars=1800):
        if not text or len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    def _action_schema_invariant(self):
        schema_lines = ""
        if self.action_schema_templates:
            schema_lines = "\nAllowed action templates:\n" + "\n".join(
                f"- {template}" for template in self.action_schema_templates
            )
        return (
            "The active environment action schema is authoritative."
            f"{schema_lines}\n"
            "If a skill's command examples use a different verb or argument "
            "format, preserve only the workflow intent and express the next "
            "action with the currently allowed action templates. For templates "
            "that apply an operation as `{operation} {obj} with {recep}`, treat "
            "the object as the direct argument of the operation; do not invent "
            "extra intermediate steps unless real environment feedback says a "
            "prerequisite is missing."
        )

    def _skill_payload_for_prompt(self, payload, max_chars=1800):
        clipped = self._workflow_first_skill_payload(payload, max_chars)
        return self._align_payload_to_action_schema(clipped)

    def _align_payload_to_action_schema(self, text):
        return text

    def _skill_response_invariants(self):
        return [
            self._action_schema_invariant(),
            "No-effect or invalid-action feedback means the previous action failed; never treat failed feedback as task completion, confirmation, or success.",
            "Do not self-declare completion from your own reasoning or skill text; completion requires an explicit success signal from the environment or evaluator.",
            "A skill response is not environment feedback. Until real environment feedback reports completion, every normal environment turn must include exactly one `Action:` line.",
        ]

    @classmethod
    def _workflow_first_skill_payload(cls, text, max_chars=1800):
        """Compress retrieved skill text around executable workflow evidence.

        The prioritization is structural rather than task-specific: retain
        identity, instructions, workflow, action format, and recovery sections
        before examples or ancillary metadata.
        """
        if not text or len(text) <= max_chars:
            return text

        lines = text.splitlines()
        always_keep = []
        sections: list[tuple[int, list[str]]] = []
        current_heading = ""
        current: list[str] = []

        def section_priority(heading: str) -> int:
            normalized = heading.lower()
            if "workflow" in normalized or "procedure" in normalized:
                return 0
            if "action" in normalized or "format" in normalized or "command" in normalized:
                return 1
            if "error" in normalized or "recover" in normalized or "troubleshoot" in normalized:
                return 2
            if "instruction" in normalized:
                return 3
            if "example" in normalized:
                return 5
            return 4

        def flush_section() -> None:
            nonlocal current
            if current:
                sections.append((section_priority(current_heading), current))
                current = []

        in_frontmatter = False
        frontmatter_seen = False
        for line in lines:
            stripped = line.strip()
            if stripped == "---":
                if not frontmatter_seen:
                    frontmatter_seen = True
                    in_frontmatter = True
                elif in_frontmatter:
                    in_frontmatter = False
                continue

            if stripped.startswith("===") or stripped.startswith("[File:"):
                always_keep.append(line)
                continue

            if in_frontmatter:
                if stripped.startswith(("name:", "description:")):
                    always_keep.append(line)
                continue

            if stripped.startswith("#"):
                flush_section()
                current_heading = stripped
                current = [line]
                continue

            if current:
                current.append(line)
            elif stripped:
                always_keep.append(line)

        flush_section()

        ordered_sections = sorted(enumerate(sections), key=lambda item: (item[1][0], item[0]))
        kept_lines = list(always_keep)
        for _idx, (_priority, section_lines) in ordered_sections:
            candidate = "\n".join(kept_lines + [""] + section_lines).strip()
            if len(candidate) <= max_chars:
                kept_lines.extend([""] + section_lines)
                continue
            remaining = max_chars - len("\n".join(kept_lines)) - 8
            if remaining > 120:
                kept_lines.extend([""] + cls._clip_text("\n".join(section_lines), remaining).splitlines())
            break

        compressed = "\n".join(kept_lines).strip()
        if not compressed:
            return cls._clip_text(text, max_chars)
        return cls._clip_text(compressed, max_chars)

    @staticmethod
    def _recent_actions(messages, limit=2):
        actions = []
        for message in reversed(messages or []):
            if message.get("role") != "assistant":
                continue
            content = message.get("content", "")
            if not isinstance(content, str):
                continue
            match = re.search(r"Action:\s*(.+)", content, re.IGNORECASE)
            if match:
                actions.append(match.group(1).strip())
            if len(actions) >= limit:
                break
        actions.reverse()
        return actions

    def _runtime_trigger_reason(self, observation, current_step):
        observation_lower = (observation or "").lower()
        if current_step - self.runtime_last_injection_step < 3:
            return ""
        max_injections = self._positive_int_env("CASKG_RUNTIME_MAX_INJECTIONS", 4)
        if self.runtime_skill_count >= max_injections:
            return ""
        configured_markers = os.environ.get("CASKG_RUNTIME_FAILURE_MARKERS", "")
        failure_markers = [
            marker.strip().lower()
            for marker in configured_markers.split(",")
            if marker.strip()
        ] or [
            "no effect",
            "invalid",
            "failed",
            "failure",
            "you can't",
            "cannot",
            "can't",
            "not found",
            "don't see",
            "do not see",
        ]
        for marker in failure_markers:
            if marker in observation_lower:
                return f"runtime_failure:{marker}"
        feedback_tokens = re.findall(r"[a-z0-9]+", observation_lower)
        if 0 < len(feedback_tokens) <= 2:
            return "runtime_failure:low_information_feedback"
        return ""

    def _build_runtime_retrieval_query(self, task, messages, observation):
        base_query = self._build_targeted_retrieval_query(task)
        recent_actions = self._recent_actions(messages, limit=4)
        parts = [
            "runtime_repair_goal=recover the current execution chain after failed or low-information feedback",
        ]
        if recent_actions:
            parts.append("recent_actions=" + ", ".join(recent_actions))
        compact_observation = " ".join((observation or "").split())
        if compact_observation:
            parts.append("runtime_observation=" + compact_observation[:400])
        parts.append(
            "repair_needs=identify the missing prerequisite, select a non-redundant next action, and continue from the last successful state"
        )
        parts.append("base_task=" + base_query)
        return "\n".join(part for part in parts if part)

    def _format_runtime_skill_hint(self, skill_names, skill_payloads, trigger):
        if not skill_payloads:
            return ""
        clipped_payloads = [
            self._skill_payload_for_prompt(payload, 1600)
            for payload in skill_payloads[:2]
        ]
        header = [
            f"Additional runtime skill support was injected because: {trigger}.",
            "Use the following skill details only if they directly help recover and reach the shortest path to completion.",
            "Runtime recovery protocol: compare the last action with the real feedback, infer the missing prerequisite for that transition, and continue from the last successful state. Do not repeat the exact failed action unless an intervening action has changed its preconditions. If the current observation already exposes the needed next entity or destination, finish that causal chain before broad search.",
        ]
        if skill_names:
            header.append("Selected skills: " + ", ".join(skill_names[:2]))
        header.extend(self._skill_response_invariants())
        return "\n\n".join(header + clipped_payloads)

    def _known_runtime_skill_names(self):
        known = set(self.last_retrieved_skill_names or [])
        for event in self.runtime_skill_events or []:
            known.update(event.get("skill_names") or [])
        return known

    def _filter_novel_runtime_skills(self, skill_names, skill_payloads):
        known = self._known_runtime_skill_names()
        filtered_names = []
        filtered_payloads = []
        for name, payload in zip(skill_names or [], skill_payloads or []):
            if name in known:
                continue
            filtered_names.append(name)
            filtered_payloads.append(payload)
        return filtered_names, filtered_payloads

    @staticmethod
    def _runtime_repair_query_text(query):
        selected_lines = []
        for line in str(query or "").splitlines():
            lower = line.lower()
            if lower.startswith(
                (
                    "runtime_repair_goal=",
                    "recent_actions=",
                    "runtime_observation=",
                    "repair_needs=",
                )
            ):
                selected_lines.append(line)
        if selected_lines:
            return "\n".join(selected_lines)
        return str(query or "")[-500:]

    @staticmethod
    def _token_set(text):
        stopwords = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "that",
            "this",
            "into",
            "onto",
            "inside",
            "outside",
            "there",
            "here",
            "task",
            "your",
            "you",
            "are",
            "was",
            "were",
            "have",
            "has",
            "had",
            "action",
            "observation",
            "recent",
            "runtime",
        }
        return {
            token
            for token in re.findall(r"[a-z][a-z0-9_]{2,}", str(text or "").lower())
            if token not in stopwords
        }

    def _fallback_known_runtime_skill(self, skill_names, skill_payloads, trigger, query=""):
        if not str(trigger or "").startswith("runtime_failure"):
            return [], []

        query_tokens = self._token_set(self._runtime_repair_query_text(query))
        candidates = []
        for index, (name, payload) in enumerate(zip(skill_names or [], skill_payloads or [])):
            if not payload:
                continue
            payload_tokens = self._token_set(f"{name}\n{payload}")
            overlap = len(query_tokens & payload_tokens)
            candidates.append((overlap, -index, name, payload))
        if not candidates:
            return [], []
        _overlap, _neg_index, name, payload = max(candidates)
        return [name], [payload]

    def maybe_get_runtime_skill_hint(self, task, messages, observation, current_step):
        trigger = self._runtime_trigger_reason(observation, current_step)
        if not trigger:
            return ""

        dynamic_query = self._build_runtime_retrieval_query(task, messages, observation)
        skill_names = []
        skill_payloads = []

        if self.mode in {"vector", "caskg"} and self.rag:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            candidate_skill_names = []
            candidate_skill_payloads = []

            if self.mode in {"caskg"}:
                result = loop.run_until_complete(
                    self.rag.async_retrieve(dynamic_query, top_n=2)
                )
                skill_payloads, skill_names = self._filter_skills_for_task(
                    task,
                    result,
                    source_label="caskg",
                )
            else:
                result = loop.run_until_complete(self.rag.async_retrieve_vector(dynamic_query, top_n=2))
                skill_payloads, skill_names = self._filter_skills_for_task(
                    task,
                    result,
                    source_label="vector",
                )

            candidate_skill_names = list(skill_names)
            candidate_skill_payloads = list(skill_payloads)
            novel_names, novel_payloads = self._filter_novel_runtime_skills(
                candidate_skill_names,
                candidate_skill_payloads,
            )
            skill_names = novel_names[:2]
            skill_payloads = novel_payloads[:2]
            if not skill_payloads:
                skill_names, skill_payloads = self._fallback_known_runtime_skill(
                    candidate_skill_names,
                    candidate_skill_payloads,
                    trigger,
                    dynamic_query,
                )
                if skill_payloads:
                    trigger = f"{trigger}:known_skill_workflow_repair"
        if not skill_payloads:
            self.runtime_last_injection_step = current_step
            self._log(
                f"runtime skill injection skipped step={current_step} "
                f"trigger={trigger} reason=no_novel_skills"
            )
            return ""

        self._record_runtime_skill_event(current_step, trigger, dynamic_query, skill_names)
        self._log(
            f"runtime skill injection triggered step={current_step} trigger={trigger} n_skills={len(skill_names)}"
        )
        return self._format_runtime_skill_hint(skill_names, skill_payloads, trigger)

    def get_runtime_skill_events(self):
        return list(self.runtime_skill_events)

    def save_trace(
        self,
        task_id: str,
        skill_names: list[str],
        success: bool,
        steps: int,
        reward: float,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Append an evaluation trace to workspace/evaluation_traces.jsonl.

        Each line is a JSON object recording one task episode for later
        analysis and causal graph updates.

        Args:
            task_id: Unique identifier for the task episode.
            skill_names: Skills retrieved/used during the episode.
            success: Whether the task was completed successfully.
            steps: Number of agent steps taken.
            reward: Numerical reward signal (e.g. 0.0 or 1.0).
            extra: Optional dict of additional fields to merge into the trace.

        Returns:
            The path to the traces file.
        """
        trace = {
            "task_id": task_id,
            "skill_names": list(skill_names or []),
            "success": bool(success),
            "steps": int(steps),
            "reward": float(reward),
            "mode": self.mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Include retrieval metadata when available
        if self.last_retrieval_query:
            trace["retrieval_query"] = self.last_retrieval_query
        if self.last_retrieval_status:
            trace["retrieval_status"] = self.last_retrieval_status
        if self.runtime_skill_events:
            trace["runtime_skill_events"] = self.runtime_skill_events

        if extra:
            trace.update(extra)

        # Determine output path
        if self.caskg_workspace:
            traces_dir = Path(self.caskg_workspace)
        else:
            traces_dir = Path(".")
        traces_dir.mkdir(parents=True, exist_ok=True)
        traces_path = traces_dir / "evaluation_traces.jsonl"

        with open(traces_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

        self._log(f"trace saved to {traces_path}: task_id={task_id} success={success} reward={reward}")
        return str(traces_path)
