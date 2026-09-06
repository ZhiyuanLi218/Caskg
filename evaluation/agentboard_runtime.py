from __future__ import annotations

import re
import shutil
import subprocess
import sys
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.agentboard_protocol import AgentBoardEpisodeSpec


@dataclass
class ProgressTracker:
    patterns: tuple[str, ...]
    matched: list[bool] = field(init=False)

    def __post_init__(self) -> None:
        if not self.patterns:
            raise ValueError("AgentBoard progress tracking requires at least one subgoal.")
        self.matched = [False] * len(self.patterns)

    def update(self, observation: str) -> list[int]:
        newly_matched: list[int] = []
        for index, pattern in enumerate(self.patterns):
            if not self.matched[index] and re.search(pattern, observation):
                self.matched[index] = True
                newly_matched.append(index)
        return newly_matched

    @property
    def progress_rate(self) -> float:
        return sum(self.matched) / len(self.matched)

    @property
    def complete(self) -> bool:
        return all(self.matched)

    def to_record(self) -> dict[str, Any]:
        return {
            "matched": list(self.matched),
            "matched_count": sum(self.matched),
            "total_count": len(self.matched),
            "progress_rate": self.progress_rate,
        }


def shared_skill_context(rendered_context: str) -> str:
    context = rendered_context.strip() or "No relevant skills were retrieved."
    return (
        "Shared retrieval guidance format v1\n"
        "<retrieved_skill_context>\n"
        f"{context}\n"
        "</retrieved_skill_context>\n"
        "Treat this guidance as optional procedural help. The current ScienceWorld "
        "observation remains the source of truth.\n"
    )


def build_agentboard_messages(
    prompt: dict[str, str],
    spec: AgentBoardEpisodeSpec,
    memory: list[tuple[str, str]],
    rendered_skill_context: str,
    *,
    memory_size_records: int,
    max_prompt_chars: int,
) -> tuple[list[dict[str, str]], int]:
    query = prompt["instruction"]
    query += "\nHere are examples:\n"
    query += prompt["examples"] + "\n"
    query += shared_skill_context(rendered_skill_context)
    query += "You should perform actions to accomplish the goal: " + spec.goal + "\n"
    query += (
        "You should use the following commands for help when your action cannot be "
        "understood: check valid actions\n"
    )
    query += (
        "You should use the following commands for help when your action cannot be "
        "understood: inventory\n"
    )

    retained = list(memory[-memory_size_records:])
    removed = len(memory) - len(retained)
    while True:
        history = "\n".join(f"{kind}: {value}" for kind, value in retained)
        input_prompt = query + history + "\nAction: "
        messages = [
            {"role": "system", "content": prompt["system_msg"]},
            {"role": "user", "content": input_prompt},
        ]
        if sum(len(message["content"]) for message in messages) <= max_prompt_chars:
            return messages, removed
        if not retained:
            raise RuntimeError("Fixed AgentBoard prompt and skill context exceed the prompt budget.")
        retained = retained[1:]
        removed += 1


def abstract_action_space(env: Any) -> list[str]:
    actions = [str(action) for action in env.get_possible_actions() if "reset" not in str(action)]
    if "check valid actions" not in actions:
        actions.append("check valid actions")
    return actions


def concrete_action_space(env: Any) -> list[str]:
    actions = [
        str(value["action"] if isinstance(value, dict) else value)
        for value in env.get_valid_action_object_combinations_with_templates()
    ]
    if "check valid actions" not in actions:
        actions.append("check valid actions")
    return actions


def safe_close_scienceworld_env(env: Any) -> None:
    try:
        env.close()
    except BrokenPipeError:
        return
    except Exception as exc:
        print(
            f"Warning: failed to close ScienceWorldEnv cleanly: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )


def ensure_java_runtime() -> str:
    candidates: list[Path] = []
    java_on_path = shutil.which("java")
    if java_on_path:
        candidates.append(Path(java_on_path))
    if "JAVA_HOME" in os.environ:
        java_home = Path(os.environ["JAVA_HOME"])
        candidates.append(java_home / "bin" / "java.exe")
        candidates.append(java_home / "bin" / "java")
    prefix = Path(sys.prefix)
    candidates.extend(
        [
            prefix / "Library" / "bin" / "java.exe",
            prefix / "bin" / "java.exe",
            prefix / "bin" / "java",
        ]
    )

    checked: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in checked or not resolved.exists():
            continue
        checked.add(resolved)
        try:
            subprocess.run(
                [str(resolved), "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            continue

        java_bin = resolved.parent
        os.environ["PATH"] = str(java_bin) + os.pathsep + os.environ.get("PATH", "")
        if java_bin.name.lower() == "bin":
            os.environ["JAVA_HOME"] = str(java_bin.parent)
        return str(resolved)

    raise RuntimeError(
        "No usable Java runtime was found. Activate D:\\ScienceWorld\\envs\\evaluator "
        "or set JAVA_HOME before launching ScienceWorld."
    )


def summarize_agentboard_records(
    records: list[dict[str, Any]],
    *,
    planned_records: int,
) -> dict[str, Any]:
    valid_statuses = {"success", "step_limit", "model_failure"}
    valid = [record for record in records if record.get("status") in valid_statuses]
    infra = [record for record in records if record.get("status") == "infra_error"]

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    def subset_metrics(difficulty: str) -> dict[str, float | int | None]:
        subset = [record for record in valid if record["episode"]["difficulty"] == difficulty]
        return {
            "count": len(subset),
            "progress_rate": mean([float(record["progress_rate"]) for record in subset]),
            "success_rate": mean([1.0 if record.get("success") else 0.0 for record in subset]),
        }

    usage: dict[str, int] = {}
    for record in valid:
        for key, value in (record.get("usage") or {}).items():
            if isinstance(value, (int, float)):
                usage[str(key)] = usage.get(str(key), 0) + int(value)

    status_counts: dict[str, int] = {}
    for record in records:
        status = str(record.get("status", "missing"))
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "planned_records": planned_records,
        "written_records": len(records),
        "valid_records": len(valid),
        "infrastructure_error_count": len(infra),
        "missing_record_count": max(planned_records - len(records), 0),
        "formal_complete": len(records) == planned_records and not infra,
        "status_counts": status_counts,
        "progress_rate": mean([float(record["progress_rate"]) for record in valid]),
        "success_rate": mean([1.0 if record.get("success") else 0.0 for record in valid]),
        "easy": subset_metrics("easy"),
        "hard": subset_metrics("hard"),
        "grounding_accuracy": mean(
            [float(record.get("grounding_accuracy", 0.0)) for record in valid]
        ),
        "mean_agent_turns": mean([float(record.get("agent_turns", 0)) for record in valid]),
        "mean_wall_seconds": mean([float(record.get("wall_seconds", 0.0)) for record in valid]),
        "mean_retrieval_latency_seconds": mean(
            [float((record.get("retrieval") or {}).get("latency_seconds", 0.0)) for record in valid]
        ),
        "llm_calls": sum(int(record.get("llm_calls", 0)) for record in valid),
        "token_usage": usage,
        "infra_episode_ids": sorted(
            int(record["episode"]["episode_id"])
            for record in infra
            if isinstance(record.get("episode"), dict)
        ),
    }
