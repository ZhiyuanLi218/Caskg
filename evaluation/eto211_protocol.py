from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.protocol import sha256_file


class ETO211ProtocolError(ValueError):
    pass


def _expect(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise ETO211ProtocolError(
            f"ETO/SkillNet-211 protocol requires {label}={expected!r}; found {value!r}."
        )


@dataclass(frozen=True)
class ETO211Protocol:
    path: Path
    data: dict[str, Any]
    sha256: str

    @classmethod
    def load(cls, path: str | Path) -> "ETO211Protocol":
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
    def execution(self) -> dict[str, Any]:
        return self.data["execution"]

    @property
    def reliability(self) -> dict[str, Any]:
        return self.data["reliability"]

    def validate(self) -> None:
        _expect(self.data.get("schema_version"), 1, "schema_version")
        supported_protocols = {
            "scienceworld-eto-skillnet-unseen211-v1": {
                "top_n": 8,
                "formal_workers_per_method": 4,
                "paired_total_workers": 8,
                "method_schedule": "paired_parallel_by_model_then_sequential_models",
                "formal_model_order": [
                    "MiniMax-M2.7",
                    "gpt-5.4-mini",
                    "claude-opus-4-8",
                ],
            },
            "scienceworld-eto-skillnet-unseen211-top15-paired16-v1": {
                "top_n": 15,
                "formal_workers_per_method": 8,
                "paired_total_workers": 16,
                "method_schedule": "paired_parallel_by_model_then_next_benchmark",
                "formal_model_order": ["gpt-5.6-luna"],
            },
            "scienceworld-eto-skillnet-unseen211-top8-luna-paired8-v1": {
                "top_n": 8,
                "formal_workers_per_method": 4,
                "paired_total_workers": 8,
                "method_schedule": "paired_parallel_by_model_then_next_benchmark",
                "formal_model_order": ["gpt-5.6-luna"],
            },
        }
        variant = supported_protocols.get(self.protocol_id)
        if variant is None:
            raise ETO211ProtocolError(f"Unsupported protocol_id={self.protocol_id!r}.")

        required_benchmark = {
            "name": "ScienceWorld",
            "version": "1.2.3",
            "profile": "ETO-SkillNet-Unseen-211",
            "split": "test",
            "episode_count": 211,
            "task_type_count": 24,
            "simplification": "easy",
            "agent_turn_limit": 30,
            "environment_step_limit": 30,
        }
        for key, expected in required_benchmark.items():
            _expect(self.benchmark.get(key), expected, f"benchmark.{key}")

        required_selection = {
            "strategy": "published_order_first_up_to_10_official_test_variations_per_task",
            "expected_episode_count": 211,
            "max_variations_per_task": 10,
            "attempts_per_episode": 1,
        }
        for key, expected in required_selection.items():
            _expect(self.selection.get(key), expected, f"selection.{key}")

        excluded = self.selection.get("excluded_official_task_types")
        expected_excluded = [
            "inclined-plane-determine-angle",
            "inclined-plane-friction-named-surfaces",
            "inclined-plane-friction-unnamed-surfaces",
            "measure-melting-point-unknown-substance",
            "mendelian-genetics-known-plant",
            "mendelian-genetics-unknown-plant",
        ]
        _expect(excluded, expected_excluded, "selection.excluded_official_task_types")

        for key in (
            "source_indices_sha256",
            "manifest_sha256",
            "prompt_sha256",
        ):
            value = str(self.assets.get(key, ""))
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ETO211ProtocolError(f"assets.{key} must be a lowercase SHA256 digest.")

        required_scoring = {
            "primary": "episode_mean_best_official_score",
            "score_scale": "0_to_100",
            "episode_score": "maximum_info_score_observed",
            "full_success": "best_official_score_ge_100",
        }
        for key, expected in required_scoring.items():
            _expect(self.scoring.get(key), expected, f"scoring.{key}")

        required_retrieval = {
            "trigger": "episode_start_once",
            "query_source": "raw_task_description",
            "top_n": variant["top_n"],
            "max_chars_per_skill": 2400,
            "max_context_chars": 12000,
            "query_rewrite": False,
            "injection_format": "shared_skill_context_v1",
        }
        for key, expected in required_retrieval.items():
            _expect(self.retrieval.get(key), expected, f"retrieval.{key}")

        required_agent = {
            "temperature": 0.0,
            "max_completion_tokens": 1024,
            "max_prompt_chars": 100000,
            "max_action_parse_repairs": 0,
            "action_selection": "first_skillnet_action_regex",
            "include_initial_observation": False,
            "history_update": "assistant_response_then_environment_observation",
        }
        for key, expected in required_agent.items():
            _expect(self.agent.get(key), expected, f"agent.{key}")

        required_execution = {
            "scheduler": "thread_pool_worker_local_resources_v1",
            "worker_resource_isolation": "one_retriever_and_chat_client_per_worker",
            "result_writes": "atomic_per_episode_main_thread_summary",
            "router_policy": "strict",
            "formal_workers_per_method": variant["formal_workers_per_method"],
            "paired_total_workers": variant["paired_total_workers"],
            "paired_methods": ["gos", "caskg"],
            "method_schedule": variant["method_schedule"],
            "formal_model_order": variant["formal_model_order"],
        }
        for key, expected in required_execution.items():
            _expect(self.execution.get(key), expected, f"execution.{key}")

        required_reliability = {
            "request_attempts": 20,
            "request_timeout_seconds": 120,
            "retry_base_delay_seconds": 5,
            "retry_max_delay_seconds": 60,
            "episode_attempts": 5,
            "episode_retry_base_delay_seconds": 15,
            "episode_retry_max_delay_seconds": 60,
            "infrastructure_errors_excluded_from_model_score": True,
        }
        for key, expected in required_reliability.items():
            _expect(self.reliability.get(key), expected, f"reliability.{key}")


@dataclass(frozen=True)
class ETO211EpisodeSpec:
    episode_id: int
    protocol_id: str
    source_rank: int
    source_task_name: str
    task_name: str
    variation_idx: int
    split: str
    simplification: str
    environment_step_limit: int
    selection_digest: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ETO211EpisodeSpec":
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "protocol_id": self.protocol_id,
            "source_rank": self.source_rank,
            "source_task_name": self.source_task_name,
            "task_name": self.task_name,
            "variation_idx": self.variation_idx,
            "split": self.split,
            "simplification": self.simplification,
            "environment_step_limit": self.environment_step_limit,
            "selection_digest": self.selection_digest,
        }


def load_eto211_manifest(
    path: str | Path,
    protocol: ETO211Protocol,
    checksum_path: str | Path,
) -> list[ETO211EpisodeSpec]:
    resolved = Path(path).resolve()
    checksum_file = Path(checksum_path).resolve()
    expected_checksum = checksum_file.read_text(encoding="ascii").strip().split()[0].lower()
    actual_checksum = sha256_file(resolved)
    if actual_checksum != expected_checksum:
        raise ETO211ProtocolError(
            f"Manifest checksum mismatch: expected {expected_checksum}, found {actual_checksum}."
        )
    if actual_checksum != protocol.assets["manifest_sha256"]:
        raise ETO211ProtocolError("Manifest checksum does not match the locked protocol asset.")

    specs = [
        ETO211EpisodeSpec.from_dict(json.loads(line))
        for line in resolved.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _validate_specs(specs, protocol)
    return specs


def _validate_specs(specs: list[ETO211EpisodeSpec], protocol: ETO211Protocol) -> None:
    expected_count = int(protocol.selection["expected_episode_count"])
    if len(specs) != expected_count:
        raise ETO211ProtocolError(
            f"Manifest has {len(specs)} episodes; expected {expected_count}."
        )
    expected_ids = list(range(expected_count))
    if [spec.episode_id for spec in specs] != expected_ids:
        raise ETO211ProtocolError("Manifest episode ids must be contiguous and source ordered.")
    if [spec.source_rank for spec in specs] != expected_ids:
        raise ETO211ProtocolError("Manifest source ranks must be contiguous and source ordered.")

    task_names: set[str] = set()
    pairs: set[tuple[str, int]] = set()
    per_task: dict[str, int] = {}
    for spec in specs:
        if spec.protocol_id != protocol.protocol_id:
            raise ETO211ProtocolError(f"Protocol mismatch for episode {spec.episode_id}.")
        if spec.split != protocol.benchmark["split"]:
            raise ETO211ProtocolError(f"Split mismatch for episode {spec.episode_id}.")
        if spec.simplification != protocol.benchmark["simplification"]:
            raise ETO211ProtocolError(f"Simplification mismatch for episode {spec.episode_id}.")
        if spec.environment_step_limit != protocol.benchmark["environment_step_limit"]:
            raise ETO211ProtocolError(f"Step-limit mismatch for episode {spec.episode_id}.")
        pair = (spec.task_name, spec.variation_idx)
        if pair in pairs:
            raise ETO211ProtocolError(f"Duplicate task/variation pair: {pair}.")
        pairs.add(pair)
        task_names.add(spec.task_name)
        per_task[spec.task_name] = per_task.get(spec.task_name, 0) + 1

    if len(task_names) != int(protocol.benchmark["task_type_count"]):
        raise ETO211ProtocolError(
            f"Manifest covers {len(task_names)} task types; expected 24."
        )
    cap = int(protocol.selection["max_variations_per_task"])
    if any(count > cap for count in per_task.values()):
        raise ETO211ProtocolError("A manifest task exceeds the variation cap.")


def load_eto211_prompt(
    path: str | Path,
    protocol: ETO211Protocol,
    checksum_path: str | Path,
) -> dict[str, Any]:
    resolved = Path(path).resolve()
    expected_checksum = (
        Path(checksum_path).resolve().read_text(encoding="ascii").strip().split()[0].lower()
    )
    actual_checksum = sha256_file(resolved)
    if actual_checksum != expected_checksum:
        raise ETO211ProtocolError(
            f"Prompt checksum mismatch: expected {expected_checksum}, found {actual_checksum}."
        )
    if actual_checksum != protocol.assets["prompt_sha256"]:
        raise ETO211ProtocolError("Prompt checksum does not match the locked protocol asset.")
    prompt = json.loads(resolved.read_text(encoding="utf-8"))
    _expect(prompt.get("schema_version"), 1, "prompt.schema_version")
    _expect(prompt.get("prompt_id"), "skillnet-scienceworld-system-v1", "prompt.prompt_id")
    if not isinstance(prompt.get("system_prompt"), str) or not prompt["system_prompt"].strip():
        raise ETO211ProtocolError("Prompt must contain a non-empty system_prompt string.")
    return prompt
