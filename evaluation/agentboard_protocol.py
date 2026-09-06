from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.protocol import sha256_file


class AgentBoardProtocolError(ValueError):
    pass


def _read_expected_checksum(path: str | Path) -> str:
    value = Path(path).resolve().read_text(encoding="ascii").strip().split()[0].lower()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise AgentBoardProtocolError(f"Invalid SHA-256 file: {path}")
    return value


@dataclass(frozen=True)
class AgentBoardProtocol:
    path: Path
    data: dict[str, Any]
    sha256: str

    @classmethod
    def load(cls, path: str | Path) -> "AgentBoardProtocol":
        resolved = Path(path).resolve()
        data = json.loads(resolved.read_text(encoding="utf-8"))
        protocol = cls(path=resolved, data=data, sha256=sha256_file(resolved))
        protocol.validate()
        return protocol

    @property
    def protocol_id(self) -> str:
        return str(self.data["protocol_id"])

    @property
    def benchmark(self) -> dict[str, Any]:
        return self.data["benchmark"]

    @property
    def selection(self) -> dict[str, Any]:
        return self.data["selection"]

    @property
    def assets(self) -> dict[str, Any]:
        return self.data["assets"]

    @property
    def scoring(self) -> dict[str, Any]:
        return self.data["scoring"]

    @property
    def retrieval(self) -> dict[str, Any]:
        return self.data["retrieval"]

    @property
    def agent(self) -> dict[str, Any]:
        return self.data["agent"]

    @property
    def reliability(self) -> dict[str, Any]:
        return self.data["reliability"]

    @property
    def execution(self) -> dict[str, Any]:
        return self.data.get("execution", {})

    def validate(self) -> None:
        if self.data.get("schema_version") != 1:
            raise AgentBoardProtocolError("Unsupported AgentBoard protocol schema version.")
        if self.protocol_id not in {
            "scienceworld-agentboard90-v1",
            "scienceworld-agentboard90-v2",
        }:
            raise AgentBoardProtocolError("Unexpected AgentBoard protocol id.")

        benchmark = self.benchmark
        required_benchmark_values = {
            "name": "ScienceWorld",
            "version": "1.2.3",
            "profile": "AgentBoard",
            "episode_count": 90,
            "task_type_count": 16,
            "simplification": "selfWateringFlowerPots,openContainers,openDoors,noElectricalAction",
            "agent_turn_limit": 30,
            "environment_step_limit": 30,
            "seed": 0,
        }
        for key, expected in required_benchmark_values.items():
            if benchmark.get(key) != expected:
                raise AgentBoardProtocolError(
                    f"AgentBoard protocol requires benchmark.{key}={expected!r}."
                )
        if benchmark.get("difficulty_counts") != {"easy": 34, "hard": 56}:
            raise AgentBoardProtocolError("AgentBoard difficulty counts must remain 34 easy / 56 hard.")
        if benchmark.get("source_variation_membership") != {"train": 88, "dev": 2, "test": 0}:
            raise AgentBoardProtocolError("AgentBoard source split membership changed.")

        selection = self.selection
        if selection.get("expected_episode_count") != 90:
            raise AgentBoardProtocolError("AgentBoard protocol requires 90 episodes.")
        if selection.get("attempts_per_episode") != 1:
            raise AgentBoardProtocolError("Paper-compatible AgentBoard runs require one attempt per episode.")
        if selection.get("difficulty_cutoff_subgoals") != 3:
            raise AgentBoardProtocolError("AgentBoard ScienceWorld difficulty cutoff must remain 3 subgoals.")

        for key in ("manifest_sha256", "prompt_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(self.assets.get(key, ""))):
                raise AgentBoardProtocolError(f"assets.{key} must be a lowercase SHA-256 digest.")

        scoring = self.scoring
        if scoring.get("primary") != "progress_rate":
            raise AgentBoardProtocolError("AgentBoard primary metric must be progress_rate.")
        if scoring.get("subgoal_match") != "python_regex_search_on_environment_observation":
            raise AgentBoardProtocolError("AgentBoard subgoal matching semantics changed.")
        if scoring.get("monotonic_subgoal_credit") is not True:
            raise AgentBoardProtocolError("AgentBoard subgoal credit must be monotonic.")

        retrieval = self.retrieval
        required_retrieval_values = {
            "trigger": "episode_start_once",
            "query_source": "exact_agentboard_modified_goal",
            "top_n": 8,
            "max_chars_per_skill": 2400,
            "max_context_chars": 12000,
            "query_rewrite": False,
        }
        for key, expected in required_retrieval_values.items():
            if retrieval.get(key) != expected:
                raise AgentBoardProtocolError(
                    f"AgentBoard protocol requires retrieval.{key}={expected!r}."
                )

        agent = self.agent
        if float(agent.get("temperature", -1)) != 0.0:
            raise AgentBoardProtocolError("AgentBoard model temperature must be 0.")
        if float(agent.get("top_p", -1)) != 1.0:
            raise AgentBoardProtocolError("AgentBoard model top_p must be 1.")
        if int(agent.get("memory_size_records", -1)) != 100:
            raise AgentBoardProtocolError("AgentBoard memory size must remain 100 records.")
        if agent.get("include_agentboard_one_shot") is not True:
            raise AgentBoardProtocolError("The AgentBoard one-shot example is required.")
        if agent.get("include_check_valid_actions") is not True:
            raise AgentBoardProtocolError("The AgentBoard check-valid-actions helper is required.")
        if agent.get("include_inventory_help") is not True:
            raise AgentBoardProtocolError("The AgentBoard inventory helper is required.")

        if self.protocol_id == "scienceworld-agentboard90-v2":
            if agent.get("action_selection") != "first_explicit_action_agentboard":
                raise AgentBoardProtocolError(
                    "AgentBoard v2 must execute the first explicit action in a model response."
                )
            required_execution_values = {
                "scheduler": "thread_pool_worker_local_resources_v1",
                "worker_resource_isolation": "one_retriever_and_chat_client_per_worker",
                "result_writes": "atomic_per_episode_main_thread_summary",
            }
            for key, expected in required_execution_values.items():
                if self.execution.get(key) != expected:
                    raise AgentBoardProtocolError(
                        f"AgentBoard v2 requires execution.{key}={expected!r}."
                    )


@dataclass(frozen=True)
class AgentBoardEpisodeSpec:
    episode_id: int
    task_name: str
    variation_idx: int
    goal: str
    difficulty: str
    subgoals: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_name": self.task_name,
            "variation_idx": self.variation_idx,
            "goal": self.goal,
            "difficulty": self.difficulty,
            "subgoals": list(self.subgoals),
        }


def load_agentboard_manifest(
    path: str | Path,
    protocol: AgentBoardProtocol,
    checksum_path: str | Path | None = None,
) -> list[AgentBoardEpisodeSpec]:
    resolved = Path(path).resolve()
    actual_checksum = sha256_file(resolved)
    protocol_checksum = str(protocol.assets["manifest_sha256"])
    if actual_checksum != protocol_checksum:
        raise AgentBoardProtocolError(
            f"AgentBoard manifest checksum mismatch: expected {protocol_checksum}, found {actual_checksum}."
        )
    if checksum_path is not None:
        expected = _read_expected_checksum(checksum_path)
        if actual_checksum != expected:
            raise AgentBoardProtocolError(
                f"AgentBoard checksum file expects {expected}, found {actual_checksum}."
            )

    rows = [
        json.loads(line)
        for line in resolved.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    specs: list[AgentBoardEpisodeSpec] = []
    for row in rows:
        additional = row.get("additional_info") or {}
        subgoals = row.get("subgoals")
        if row.get("task") != "scienceworld":
            raise AgentBoardProtocolError(f"Episode {row.get('id')} is not a ScienceWorld row.")
        if not isinstance(subgoals, list) or not subgoals:
            raise AgentBoardProtocolError(f"Episode {row.get('id')} has no subgoal list.")
        try:
            spec = AgentBoardEpisodeSpec(
                episode_id=int(row["id"]),
                task_name=str(additional["env_name"]),
                variation_idx=int(additional["var"]),
                goal=str(row["goal"]),
                difficulty=str(row["difficulty"]),
                subgoals=tuple(str(value) for value in subgoals),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AgentBoardProtocolError(f"Malformed AgentBoard row: {row!r}") from exc
        specs.append(spec)

    _validate_agentboard_specs(specs, protocol)
    return specs


def _validate_agentboard_specs(
    specs: list[AgentBoardEpisodeSpec], protocol: AgentBoardProtocol
) -> None:
    expected_count = int(protocol.selection["expected_episode_count"])
    if len(specs) != expected_count:
        raise AgentBoardProtocolError(
            f"AgentBoard manifest has {len(specs)} episodes; expected {expected_count}."
        )
    if [spec.episode_id for spec in specs] != list(range(expected_count)):
        raise AgentBoardProtocolError("AgentBoard episode ids must be exactly 0 through 89 in order.")
    if len({(spec.task_name, spec.variation_idx) for spec in specs}) != expected_count:
        raise AgentBoardProtocolError("AgentBoard task/variation pairs must be unique.")
    expected_task_types = int(protocol.benchmark["task_type_count"])
    if len({spec.task_name for spec in specs}) != expected_task_types:
        raise AgentBoardProtocolError(
            f"AgentBoard manifest must contain {expected_task_types} task types."
        )

    difficulty_counts = {"easy": 0, "hard": 0}
    cutoff = int(protocol.selection["difficulty_cutoff_subgoals"])
    for spec in specs:
        expected_difficulty = "hard" if len(spec.subgoals) > cutoff else "easy"
        if spec.difficulty != expected_difficulty:
            raise AgentBoardProtocolError(
                f"Episode {spec.episode_id} difficulty is {spec.difficulty}; "
                f"expected {expected_difficulty}."
            )
        difficulty_counts[spec.difficulty] += 1
        for pattern in spec.subgoals:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise AgentBoardProtocolError(
                    f"Episode {spec.episode_id} has an invalid subgoal regex: {pattern!r}."
                ) from exc
    if difficulty_counts != protocol.benchmark["difficulty_counts"]:
        raise AgentBoardProtocolError(
            f"AgentBoard difficulty counts are {difficulty_counts}; "
            f"expected {protocol.benchmark['difficulty_counts']}."
        )


def load_agentboard_prompt(
    path: str | Path,
    protocol: AgentBoardProtocol,
    checksum_path: str | Path | None = None,
) -> dict[str, str]:
    resolved = Path(path).resolve()
    actual_checksum = sha256_file(resolved)
    protocol_checksum = str(protocol.assets["prompt_sha256"])
    if actual_checksum != protocol_checksum:
        raise AgentBoardProtocolError(
            f"AgentBoard prompt checksum mismatch: expected {protocol_checksum}, found {actual_checksum}."
        )
    if checksum_path is not None:
        expected = _read_expected_checksum(checksum_path)
        if actual_checksum != expected:
            raise AgentBoardProtocolError(
                f"AgentBoard prompt checksum file expects {expected}, found {actual_checksum}."
            )

    value = json.loads(resolved.read_text(encoding="utf-8"))
    required = ("system_msg", "instruction", "examples")
    if any(not isinstance(value.get(key), str) or not value[key].strip() for key in required):
        raise AgentBoardProtocolError(
            "AgentBoard prompt must contain non-empty system_msg, instruction, and examples strings."
        )
    return {key: str(value[key]) for key in required}
