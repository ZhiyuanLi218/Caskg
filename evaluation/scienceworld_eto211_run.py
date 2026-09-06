from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scienceworld import ScienceWorldEnv, __version__ as scienceworld_version
from scienceworld.utils import infer_task

from evaluation.agentboard_runtime import ensure_java_runtime, safe_close_scienceworld_env
from evaluation.eto211_protocol import (
    ETO211EpisodeSpec,
    ETO211Protocol,
    ETO211ProtocolError,
    load_eto211_manifest,
    load_eto211_prompt,
)
from evaluation.model_client import (
    ChatInfrastructureError,
    ChatResponse,
    OpenAICompatibleChatClient,
)
from evaluation.protocol import canonical_json_hash, sha256_file
from evaluation.retrievers import CaSKGRetriever, GoSRetriever
from evaluation.retrievers.base import NullRetriever, Retriever


ACTION_SELECTION_POLICY = "first_skillnet_action_regex"
SCHEDULER_POLICY = "thread_pool_worker_local_resources_v1"
PROFILE_DIR = "eto_skillnet_unseen211"
VALID_STATUSES = {"success", "environment_done", "step_limit", "model_failure"}
ACTION_PATTERN = re.compile(r"Action:\s*(.+)", re.IGNORECASE)
EVALUATOR_COMPONENTS = (
    "evaluation/scienceworld_eto211_run.py",
    "evaluation/eto211_protocol.py",
    "evaluation/agentboard_runtime.py",
    "evaluation/model_client.py",
    "evaluation/protocol.py",
    "evaluation/retrievers/__init__.py",
    "evaluation/retrievers/base.py",
    "evaluation/retrievers/caskg_adapter.py",
    "evaluation/retrievers/gos_adapter.py",
    "evaluation/retrievers/project_worker.py",
    "evaluation/retrievers/subprocess_adapter.py",
)
_WORKER_LOCAL = threading.local()


@dataclass
class UsageTotals:
    values: dict[str, int] = field(default_factory=dict)

    def add(self, response: ChatResponse) -> None:
        for key, value in response.usage.items():
            self.values[key] = self.values.get(key, 0) + value


@dataclass(frozen=True)
class EpisodeJob:
    spec: ETO211EpisodeSpec
    attempt_index: int
    result_path: Path


@dataclass
class WorkerResources:
    retriever: Retriever
    client: OpenAICompatibleChatClient


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_") or "model"


def _progress_bar(completed: int, total: int, width: int = 30) -> str:
    filled = width if total <= 0 else min(width, completed * width // total)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _evaluator_fingerprint(root: Path) -> str:
    component_hashes = {
        relative_path: sha256_file(root / relative_path)
        for relative_path in EVALUATOR_COMPONENTS
    }
    return canonical_json_hash({"schema_version": 1, "components": component_hashes})


def _build_retriever(args: argparse.Namespace, root: Path) -> Retriever:
    if args.retriever == "none":
        return NullRetriever()
    worker_path = root / "evaluation" / "retrievers" / "project_worker.py"
    if args.retriever == "gos":
        return GoSRetriever(
            args.retriever_config,
            worker_path=worker_path,
            timeout_seconds=args.retrieval_timeout,
        )
    return CaSKGRetriever(
        args.retriever_config,
        worker_path=worker_path,
        timeout_seconds=args.retrieval_timeout,
    )


def _shared_skill_context(rendered_context: str) -> str:
    context = rendered_context.strip() or "No relevant skills were retrieved."
    return (
        "Shared retrieval guidance format v1\n"
        "<retrieved_skill_context>\n"
        f"{context}\n"
        "</retrieved_skill_context>\n"
        "Treat this guidance as optional procedural help. The current ScienceWorld "
        "observation remains the source of truth."
    )


def _bounded_messages(
    static_messages: list[dict[str, str]],
    history: list[dict[str, str]],
    *,
    max_prompt_chars: int,
) -> tuple[list[dict[str, str]], int]:
    retained = list(history)
    removed = 0
    messages = static_messages + retained
    while retained and sum(len(item["content"]) for item in messages) > max_prompt_chars:
        drop_count = min(2, len(retained))
        retained = retained[drop_count:]
        removed += drop_count
        messages = static_messages + retained
    if sum(len(item["content"]) for item in messages) > max_prompt_chars:
        raise RuntimeError("Fixed task and retrieval context exceeds the prompt budget.")
    return messages, removed


def _parse_skillnet_action(response: str) -> str:
    match = ACTION_PATTERN.search(response or "")
    if not match:
        return ""
    return match.group(1).strip().strip('"\' *`')


def _response_record(response: ChatResponse) -> dict[str, Any]:
    return {
        "content": response.content,
        "response_id": response.response_id,
        "latency_seconds": response.latency_seconds,
        "usage": response.usage,
        "request_attempt_count": response.request_attempt_count,
        "request_retry_errors": list(response.retry_errors),
        "router_provider": response.router_provider,
        "router_policy": response.router_policy,
        "router_request_id": response.router_request_id,
    }


def _require_router_provider(response: ChatResponse, expected_provider: str) -> None:
    if expected_provider and response.router_provider != expected_provider:
        actual = response.router_provider or "<missing>"
        raise ChatInfrastructureError(
            f"Router provider mismatch: expected {expected_provider!r}, found {actual!r}."
        )


def _require_router_policy(response: ChatResponse, expected_policy: str) -> None:
    if expected_policy and response.router_policy != expected_policy:
        actual = response.router_policy or "<missing>"
        raise ChatInfrastructureError(
            f"Router policy mismatch: expected {expected_policy!r}, found {actual!r}."
        )


def _run_episode_once(
    spec: ETO211EpisodeSpec,
    protocol: ETO211Protocol,
    prompt: dict[str, Any],
    retriever: Retriever,
    client: OpenAICompatibleChatClient,
    *,
    expected_router_provider: str,
    expected_router_policy: str,
) -> dict[str, Any]:
    benchmark = protocol.benchmark
    retrieval_config = protocol.retrieval
    agent_config = protocol.agent
    env: ScienceWorldEnv | None = None
    started = time.perf_counter()
    trajectory: list[dict[str, Any]] = []
    usage = UsageTotals()
    llm_calls = 0
    prompt_history_removed = 0

    try:
        env = ScienceWorldEnv(envStepLimit=int(benchmark["environment_step_limit"]))
        env.load(
            spec.task_name,
            variationIdx=spec.variation_idx,
            simplificationStr=str(benchmark["simplification"]),
        )
        initial_observation, initial_info = env.reset()
        task_description = str(initial_info.get("taskDesc") or env.get_task_description())
        bundle = retriever.retrieve(
            task_description,
            top_n=int(retrieval_config["top_n"]),
            max_chars_per_skill=int(retrieval_config["max_chars_per_skill"]),
            max_context_chars=int(retrieval_config["max_context_chars"]),
        )
        if bundle.query != task_description:
            raise RuntimeError("Retriever did not preserve the exact raw task description.")
        if bundle.requested_top_n != int(retrieval_config["top_n"]):
            raise RuntimeError(
                "Retriever did not preserve the locked ScienceWorld retrieval top_n."
            )

        static_messages = [
            {"role": "system", "content": str(prompt["system_prompt"])},
            {"role": "system", "content": _shared_skill_context(bundle.rendered_context)},
            {"role": "user", "content": task_description},
        ]
        history: list[dict[str, str]] = []
        best_score = int(initial_info.get("score") or 0)
        final_score = best_score
        environment_done = False

        for turn_index in range(int(benchmark["agent_turn_limit"])):
            messages, removed = _bounded_messages(
                static_messages,
                history,
                max_prompt_chars=int(agent_config["max_prompt_chars"]),
            )
            prompt_history_removed += removed
            response = client.complete(messages)
            _require_router_provider(response, expected_router_provider)
            _require_router_policy(response, expected_router_policy)
            usage.add(response)
            llm_calls += 1
            action = _parse_skillnet_action(response.content)
            observation, reward, environment_done, info = env.step(action)
            final_score = int((info or {}).get("score") or 0)
            best_score = max(best_score, final_score)
            trajectory.append(
                {
                    "turn": turn_index + 1,
                    "action": action,
                    "action_parse_miss": not bool(action),
                    "model_response": _response_record(response),
                    "environment_observation": observation,
                    "environment_reward": reward,
                    "official_score": final_score,
                    "best_official_score": best_score,
                    "environment_done": bool(environment_done),
                }
            )
            history.extend(
                [
                    {"role": "assistant", "content": response.content},
                    {"role": "user", "content": f"Observation: {observation}"},
                ]
            )
            if environment_done:
                break

        success = best_score >= 100
        status = "success" if success else ("environment_done" if environment_done else "step_limit")
        return {
            "status": status,
            "success": success,
            "best_official_score": best_score,
            "normalized_score": best_score / 100.0,
            "final_official_score": final_score,
            "agent_turns": len(trajectory),
            "environment_steps": len(trajectory),
            "llm_calls": llm_calls,
            "usage": usage.values,
            "prompt_history_messages_removed": prompt_history_removed,
            "retrieval": bundle.to_record(),
            "task_description": task_description,
            "initial_observation_diagnostic": initial_observation,
            "initial_observation_in_model_prompt": False,
            "trajectory": trajectory,
            "wall_seconds": time.perf_counter() - started,
        }
    finally:
        if env is not None:
            safe_close_scienceworld_env(env)


def _result_path(
    output_root: Path,
    retriever_name: str,
    model: str,
    attempt_index: int,
    spec: ETO211EpisodeSpec,
) -> Path:
    return (
        output_root
        / PROFILE_DIR
        / retriever_name
        / _safe_model_name(model)
        / f"attempt_{attempt_index:02d}"
        / f"episode_{spec.episode_id:03d}.json"
    )


def _summary_path(output_root: Path, retriever_name: str, model: str) -> Path:
    return output_root / PROFILE_DIR / retriever_name / _safe_model_name(model) / "summary.json"


def _base_record(
    spec: ETO211EpisodeSpec,
    protocol: ETO211Protocol,
    manifest_sha256: str,
    prompt_sha256: str,
    retriever_name: str,
    model: str,
    attempt_index: int,
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "benchmark_profile": "ETO-SkillNet-Unseen-211",
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.sha256,
        "manifest_sha256": manifest_sha256,
        "source_indices_sha256": protocol.assets["source_indices_sha256"],
        "prompt_sha256": prompt_sha256,
        "scienceworld_version": scienceworld_version,
        "episode": spec.to_dict(),
        "retriever": retriever_name,
        "model": model,
        "attempt_index": attempt_index,
        "evaluator_fingerprint": run_metadata["evaluator_fingerprint"],
        "action_selection": run_metadata["action_selection"],
        "retriever_config_sha256": run_metadata["retriever_config_sha256"],
        "api_base": run_metadata["api_base"],
        "expected_router_provider": run_metadata["expected_router_provider"],
        "run_cohort": run_metadata["run_cohort"],
        "execution": dict(run_metadata["execution"]),
        "started_at_utc": _utc_now(),
    }


def _is_valid_existing_result(
    path: Path,
    *,
    protocol: ETO211Protocol,
    manifest_sha256: str,
    prompt_sha256: str,
    retriever_name: str,
    model: str,
    spec: ETO211EpisodeSpec,
    attempt_index: int,
    run_metadata: dict[str, Any],
) -> bool:
    if not path.exists():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if record.get("status") not in VALID_STATUSES:
        return False
    expected = {
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.sha256,
        "manifest_sha256": manifest_sha256,
        "source_indices_sha256": protocol.assets["source_indices_sha256"],
        "prompt_sha256": prompt_sha256,
        "retriever": retriever_name,
        "model": model,
        "attempt_index": attempt_index,
        "evaluator_fingerprint": run_metadata["evaluator_fingerprint"],
        "action_selection": run_metadata["action_selection"],
        "retriever_config_sha256": run_metadata["retriever_config_sha256"],
        "api_base": run_metadata["api_base"],
        "expected_router_provider": run_metadata["expected_router_provider"],
        "run_cohort": run_metadata["run_cohort"],
    }
    if any(record.get(key) != value for key, value in expected.items()):
        return False
    if record.get("execution") != run_metadata["execution"]:
        return False
    episode = record.get("episode") or {}
    return episode.get("episode_id") == spec.episode_id


def _resolved_api_base(args: argparse.Namespace) -> str:
    return args.api_base or os.environ.get("SCIENCEWORLD_ROUTER_BASE_URL", "")


def _make_client(args: argparse.Namespace, protocol: ETO211Protocol) -> OpenAICompatibleChatClient:
    api_base = _resolved_api_base(args)
    api_key = os.environ.get(args.api_key_env, "")
    if not api_base:
        raise ValueError("Set --api-base or SCIENCEWORLD_ROUTER_BASE_URL before a model run.")
    if not api_key:
        raise ValueError(f"Environment variable {args.api_key_env} is empty.")
    reliability = protocol.reliability
    agent = protocol.agent
    return OpenAICompatibleChatClient(
        api_base=api_base,
        api_key=api_key,
        model=args.model,
        temperature=float(agent["temperature"]),
        max_completion_tokens=int(agent["max_completion_tokens"]),
        timeout_seconds=float(reliability["request_timeout_seconds"]),
        attempts=int(reliability["request_attempts"]),
        retry_base_delay_seconds=float(reliability["retry_base_delay_seconds"]),
        retry_max_delay_seconds=float(reliability["retry_max_delay_seconds"]),
        router_provider=args.expected_router_provider,
        router_policy=str(protocol.execution["router_policy"]),
    )


def _build_run_metadata(
    args: argparse.Namespace,
    protocol: ETO211Protocol,
    root: Path,
) -> dict[str, Any]:
    if protocol.agent["action_selection"] != ACTION_SELECTION_POLICY:
        raise ETO211ProtocolError("Evaluator action parser does not match the locked protocol.")
    if protocol.execution["scheduler"] != SCHEDULER_POLICY:
        raise ETO211ProtocolError("Evaluator scheduler does not match the locked protocol.")
    return {
        "evaluator_fingerprint": _evaluator_fingerprint(root),
        "action_selection": ACTION_SELECTION_POLICY,
        "retriever_config_sha256": sha256_file(args.retriever_config),
        "api_base": _resolved_api_base(args),
        "expected_router_provider": args.expected_router_provider,
        "run_cohort": args.run_cohort,
        "execution": {
            "scheduler": SCHEDULER_POLICY,
            "max_workers": int(args.max_workers),
            "worker_resource_isolation": "one_retriever_and_chat_client_per_worker",
            "retrieval_timeout_seconds": float(args.retrieval_timeout),
            "router_policy": str(protocol.execution["router_policy"]),
        },
    }


def _initialize_worker(
    args: argparse.Namespace,
    root: Path,
    protocol: ETO211Protocol,
    retriever_registry: list[Retriever],
    registry_lock: threading.Lock,
) -> None:
    retriever = _build_retriever(args, root)
    try:
        client = _make_client(args, protocol)
    except Exception:
        retriever.close()
        raise
    _WORKER_LOCAL.resources = WorkerResources(retriever=retriever, client=client)
    with registry_lock:
        retriever_registry.append(retriever)


def _current_worker_resources() -> WorkerResources:
    resources = getattr(_WORKER_LOCAL, "resources", None)
    if not isinstance(resources, WorkerResources):
        raise RuntimeError("Worker resources were not initialized.")
    return resources


def _run_episode_job(
    job: EpisodeJob,
    *,
    protocol: ETO211Protocol,
    prompt: dict[str, Any],
    manifest_sha256: str,
    prompt_sha256: str,
    retriever_name: str,
    model: str,
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    resources = _current_worker_resources()
    base = _base_record(
        job.spec,
        protocol,
        manifest_sha256,
        prompt_sha256,
        retriever_name,
        model,
        job.attempt_index,
        run_metadata,
    )
    episode_errors: list[dict[str, Any]] = []
    result: dict[str, Any] | None = None
    episode_attempts = int(protocol.reliability["episode_attempts"])
    base_delay = float(protocol.reliability["episode_retry_base_delay_seconds"])
    max_delay = float(protocol.reliability["episode_retry_max_delay_seconds"])
    for episode_attempt in range(1, episode_attempts + 1):
        try:
            result = _run_episode_once(
                job.spec,
                protocol,
                prompt,
                resources.retriever,
                resources.client,
                expected_router_provider=str(run_metadata["expected_router_provider"]),
                expected_router_policy=str(run_metadata["execution"]["router_policy"]),
            )
            result["episode_attempt"] = episode_attempt
            break
        except Exception as exc:
            error_record: dict[str, Any] = {
                "episode_attempt": episode_attempt,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            try:
                resources.retriever.close()
            except Exception as reset_exc:
                error_record["retriever_reset_error"] = (
                    f"{type(reset_exc).__name__}: {reset_exc}"
                )
            episode_errors.append(error_record)
            if episode_attempt < episode_attempts:
                time.sleep(min(base_delay * episode_attempt, max_delay))

    if result is None:
        result = {
            "status": "infra_error",
            "success": False,
            "best_official_score": None,
            "normalized_score": None,
            "agent_turns": 0,
            "environment_steps": 0,
            "llm_calls": 0,
        }
    result["infrastructure_errors"] = episode_errors
    result["finished_at_utc"] = _utc_now()
    record = {**base, **result}
    _write_json_atomic(job.result_path, record)
    return record


def _scheduler_failure_record(
    job: EpisodeJob,
    exc: Exception,
    *,
    protocol: ETO211Protocol,
    manifest_sha256: str,
    prompt_sha256: str,
    retriever_name: str,
    model: str,
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    base = _base_record(
        job.spec,
        protocol,
        manifest_sha256,
        prompt_sha256,
        retriever_name,
        model,
        job.attempt_index,
        run_metadata,
    )
    result = {
        "status": "infra_error",
        "success": False,
        "best_official_score": None,
        "normalized_score": None,
        "agent_turns": 0,
        "environment_steps": 0,
        "llm_calls": 0,
        "infrastructure_errors": [
            {
                "episode_attempt": 0,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        ],
        "finished_at_utc": _utc_now(),
    }
    record = {**base, **result}
    _write_json_atomic(job.result_path, record)
    return record


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _summarize_records(records: list[dict[str, Any]], planned_records: int) -> dict[str, Any]:
    valid = [record for record in records if record.get("status") in VALID_STATUSES]
    infra = [record for record in records if record.get("status") == "infra_error"]
    scores = [float(record["best_official_score"]) for record in valid]
    steps = [float(record.get("environment_steps", 0)) for record in valid]
    wall_seconds = [float(record.get("wall_seconds", 0.0)) for record in valid]
    retrieval_latencies = [
        float((record.get("retrieval") or {}).get("latency_seconds", 0.0))
        for record in valid
        if record.get("retrieval")
    ]

    usage_totals: dict[str, int] = {}
    router_providers: Counter[str] = Counter()
    router_policies: Counter[str] = Counter()
    request_retry_count = 0
    for record in valid:
        for key, value in (record.get("usage") or {}).items():
            usage_totals[str(key)] = usage_totals.get(str(key), 0) + int(value)
        for turn in record.get("trajectory") or []:
            response = turn.get("model_response") or {}
            provider = str(response.get("router_provider") or "<missing>")
            router_providers[provider] += 1
            policy = str(response.get("router_policy") or "<missing>")
            router_policies[policy] += 1
            request_retry_count += max(0, int(response.get("request_attempt_count", 1)) - 1)

    per_task_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in valid:
        per_task_records[str((record.get("episode") or {}).get("task_name", ""))].append(record)
    per_task: dict[str, dict[str, Any]] = {}
    task_means: list[float] = []
    for task_name, task_records in sorted(per_task_records.items()):
        task_scores = [float(record["best_official_score"]) for record in task_records]
        task_mean = sum(task_scores) / len(task_scores)
        task_means.append(task_mean)
        per_task[task_name] = {
            "valid_episode_count": len(task_records),
            "mean_best_official_score": task_mean,
            "full_success_rate": sum(bool(record.get("success")) for record in task_records)
            / len(task_records),
            "mean_environment_steps": sum(
                float(record.get("environment_steps", 0)) for record in task_records
            )
            / len(task_records),
        }

    all_attempt_errors = [
        error
        for record in records
        for error in (record.get("infrastructure_errors") or [])
    ]
    error_types: Counter[str] = Counter()
    for error in all_attempt_errors:
        error_types[str(error.get("error_type", "UnknownError"))] += 1
    unresolved_error_types: Counter[str] = Counter()
    for record in infra:
        for error in record.get("infrastructure_errors") or []:
            unresolved_error_types[str(error.get("error_type", "UnknownError"))] += 1

    return {
        "planned_records": planned_records,
        "written_records": len(records),
        "valid_score_count": len(valid),
        "infrastructure_error_count": len(infra),
        "recovered_infrastructure_episode_count": sum(
            bool(record.get("infrastructure_errors")) for record in valid
        ),
        "infrastructure_attempt_error_count": len(all_attempt_errors),
        "status_counts": dict(sorted(Counter(str(r.get("status")) for r in records).items())),
        "episode_mean_best_official_score": _mean(scores),
        "episode_mean_normalized_score": (
            _mean(scores) / 100.0 if scores else None
        ),
        "macro_mean_task_score": _mean(task_means),
        "full_success_rate": (
            sum(bool(record.get("success")) for record in valid) / len(valid) if valid else None
        ),
        "mean_environment_steps": _mean(steps),
        "mean_wall_seconds": _mean(wall_seconds),
        "total_llm_calls": sum(int(record.get("llm_calls", 0)) for record in valid),
        "total_request_retries": request_retry_count,
        "usage_totals": usage_totals,
        "mean_retrieval_latency_seconds": _mean(retrieval_latencies),
        "retrieval_status_counts": dict(
            sorted(
                Counter(
                    str((record.get("retrieval") or {}).get("status", "<missing>"))
                    for record in valid
                ).items()
            )
        ),
        "router_provider_call_counts": dict(sorted(router_providers.items())),
        "router_policy_call_counts": dict(sorted(router_policies.items())),
        "infrastructure_attempt_error_types": dict(sorted(error_types.items())),
        "unresolved_infrastructure_error_types": dict(sorted(unresolved_error_types.items())),
        "per_task": per_task,
    }


def _write_summary(
    path: Path,
    *,
    records: list[dict[str, Any]],
    selected_count: int,
    full_manifest_count: int,
    protocol: ETO211Protocol,
    manifest_sha256: str,
    prompt_sha256: str,
    retriever_name: str,
    model: str,
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    metrics = _summarize_records(records, planned_records=selected_count)
    metrics["formal_complete"] = (
        selected_count == full_manifest_count
        and metrics["written_records"] == full_manifest_count
        and metrics["valid_score_count"] == full_manifest_count
        and metrics["infrastructure_error_count"] == 0
    )
    summary = {
        "schema_version": 1,
        "benchmark_profile": "ETO-SkillNet-Unseen-211",
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.sha256,
        "manifest_sha256": manifest_sha256,
        "source_indices_sha256": protocol.assets["source_indices_sha256"],
        "prompt_sha256": prompt_sha256,
        "retriever": retriever_name,
        "model": model,
        "evaluator_fingerprint": run_metadata["evaluator_fingerprint"],
        "action_selection": run_metadata["action_selection"],
        "retriever_config_sha256": run_metadata["retriever_config_sha256"],
        "api_base": run_metadata["api_base"],
        "expected_router_provider": run_metadata["expected_router_provider"],
        "run_cohort": run_metadata["run_cohort"],
        "execution": dict(run_metadata["execution"]),
        "selected_episode_count": selected_count,
        "full_manifest_episode_count": full_manifest_count,
        "generated_at_utc": _utc_now(),
        "metrics": metrics,
    }
    _write_json_atomic(path, summary)
    return summary


def _validate_environment(
    specs: list[ETO211EpisodeSpec],
    protocol: ETO211Protocol,
    root: Path,
) -> dict[str, Any]:
    source_path = root / str(protocol.assets["source_indices"])
    if sha256_file(source_path) != protocol.assets["source_indices_sha256"]:
        raise ETO211ProtocolError("Local source index checksum does not match the protocol.")
    source_rows = json.loads(source_path.read_text(encoding="utf-8"))
    expected_pairs = [
        (str(source_task), int(variation_idx)) for source_task, variation_idx in source_rows
    ]
    manifest_pairs = [(spec.source_task_name, spec.variation_idx) for spec in specs]
    if manifest_pairs != expected_pairs:
        raise ETO211ProtocolError("Manifest order/content differs from the pinned source indices.")
    for spec in specs:
        if infer_task(spec.source_task_name) != spec.task_name:
            raise ETO211ProtocolError(
                f"Canonical task mismatch in episode {spec.episode_id}: {spec.task_name}"
            )

    env = ScienceWorldEnv(envStepLimit=int(protocol.benchmark["environment_step_limit"]))
    try:
        official_tasks = set(env.get_task_names())
        by_task: dict[str, list[int]] = defaultdict(list)
        for spec in specs:
            by_task[spec.task_name].append(spec.variation_idx)
        excluded = sorted(official_tasks - set(by_task))
        if excluded != protocol.selection["excluded_official_task_types"]:
            raise ETO211ProtocolError(f"Official task exclusion mismatch: {excluded}")
        for task_name, selected in by_task.items():
            if task_name not in official_tasks:
                raise ETO211ProtocolError(f"Unknown official task in manifest: {task_name}")
            env.load(
                task_name,
                variationIdx=selected[0],
                simplificationStr=str(protocol.benchmark["simplification"]),
            )
            official_test = list(env.get_variations_test())
            if selected != official_test[: int(protocol.selection["max_variations_per_task"])]:
                raise ETO211ProtocolError(
                    f"Manifest is not the first up-to-10 official test variations for {task_name}."
                )
        return {
            "episode_count": len(specs),
            "task_type_count": len(by_task),
            "excluded_task_types": excluded,
            "official_test_episode_count": sum(
                len(values)
                for values in (
                    _official_test_variations(env, task_name, protocol) for task_name in official_tasks
                )
            ),
        }
    finally:
        safe_close_scienceworld_env(env)


def _official_test_variations(
    env: ScienceWorldEnv,
    task_name: str,
    protocol: ETO211Protocol,
) -> list[int]:
    env.load(
        task_name,
        variationIdx=0,
        simplificationStr=str(protocol.benchmark["simplification"]),
    )
    return list(env.get_variations_test())


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Paired ETO/SkillNet Unseen-211 ScienceWorld evaluator."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=root / "configs" / "protocol_eto_skillnet_unseen211_v1.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "manifests" / "scienceworld_eto_skillnet_unseen211_v1.jsonl",
    )
    parser.add_argument(
        "--manifest-checksum",
        type=Path,
        default=root / "manifests" / "scienceworld_eto_skillnet_unseen211_v1.sha256",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=root / "prompts" / "skillnet_scienceworld_system_v1.json",
    )
    parser.add_argument(
        "--prompt-checksum",
        type=Path,
        default=root / "prompts" / "skillnet_scienceworld_system_v1.sha256",
    )
    parser.add_argument(
        "--retriever-config",
        type=Path,
        default=root / "configs" / "retrievers_v1.json",
    )
    parser.add_argument("--retriever", choices=("none", "gos", "caskg"), default="none")
    parser.add_argument("--retrieval-timeout", type=float, default=180.0)
    parser.add_argument("--model", default="")
    parser.add_argument("--api-base", default="")
    parser.add_argument("--api-key-env", default="ROUTER_MASTER_KEY")
    parser.add_argument("--expected-router-provider", default="")
    parser.add_argument("--run-cohort", default="")
    parser.add_argument("--output-dir", type=Path, default=root / "results")
    parser.add_argument("--attempts", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--episode-id", action="append", type=int, default=[])
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--retrieval-preflight", action="store_true")
    args = parser.parse_args()

    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least 1.")
    java = ensure_java_runtime()
    protocol = ETO211Protocol.load(args.protocol)
    specs = load_eto211_manifest(args.manifest, protocol, args.manifest_checksum)
    prompt = load_eto211_prompt(args.prompt, protocol, args.prompt_checksum)
    manifest_sha256 = sha256_file(args.manifest)
    prompt_sha256 = sha256_file(args.prompt)
    environment = _validate_environment(specs, protocol, root)
    print(
        f"ETO/SkillNet Unseen-211 protocol OK: episodes={len(specs)}; task_types=24; "
        f"split=test; simplification=easy; max_steps=30; "
        f"official_test_total={environment['official_test_episode_count']}; java={java}",
        flush=True,
    )
    if args.validate_only:
        return 0

    selected = specs
    if args.episode_id:
        requested = set(args.episode_id)
        selected = [spec for spec in specs if spec.episode_id in requested]
        missing = requested - {spec.episode_id for spec in selected}
        if missing:
            raise ValueError(f"Unknown ETO/SkillNet episode ids: {sorted(missing)}")
    if args.max_episodes is not None:
        if args.max_episodes < 1:
            raise ValueError("--max-episodes must be at least 1.")
        selected = selected[: args.max_episodes]
    if not selected:
        raise ValueError("No ETO/SkillNet episodes were selected.")

    formal_episode_count = int(protocol.selection["expected_episode_count"])
    if len(selected) == formal_episode_count:
        formal_workers = int(protocol.execution["formal_workers_per_method"])
        if args.max_workers != formal_workers:
            raise ValueError(
                f"Formal ETO/SkillNet-211 runs require --max-workers {formal_workers}."
            )
        if not args.expected_router_provider:
            raise ValueError(
                "Formal ETO/SkillNet-211 runs require --expected-router-provider."
            )
        if not args.run_cohort:
            raise ValueError("Formal ETO/SkillNet-211 runs require --run-cohort.")

    attempts = args.attempts or int(protocol.selection["attempts_per_episode"])
    if attempts != int(protocol.selection["attempts_per_episode"]):
        raise ValueError("ETO/SkillNet-211 formal protocol requires exactly one attempt per episode.")

    if args.retrieval_preflight:
        if args.retriever == "none":
            raise ValueError("Retrieval preflight requires --retriever gos or caskg.")
        retriever = _build_retriever(args, root)
        env: ScienceWorldEnv | None = None
        try:
            spec = selected[0]
            env = ScienceWorldEnv(envStepLimit=int(protocol.benchmark["environment_step_limit"]))
            env.load(
                spec.task_name,
                variationIdx=spec.variation_idx,
                simplificationStr=str(protocol.benchmark["simplification"]),
            )
            _, info = env.reset()
            task_description = str(info.get("taskDesc") or env.get_task_description())
            bundle = retriever.retrieve(
                task_description,
                top_n=int(protocol.retrieval["top_n"]),
                max_chars_per_skill=int(protocol.retrieval["max_chars_per_skill"]),
                max_context_chars=int(protocol.retrieval["max_context_chars"]),
            )
            if bundle.query != task_description:
                raise RuntimeError("Retriever did not preserve the exact raw task description.")
            if bundle.requested_top_n != int(protocol.retrieval["top_n"]):
                raise RuntimeError(
                    "Retriever did not preserve the locked ScienceWorld retrieval top_n."
                )
            print(
                f"Retrieval OK: method={bundle.method}; status={bundle.status}; "
                f"requested_top_n={bundle.requested_top_n}; "
                f"skills={len(bundle.skills)}; query_exact={bundle.query == task_description}; "
                f"context_chars={len(bundle.rendered_context)}; "
                f"workspace_fingerprint={bundle.workspace_fingerprint}; "
                f"latency={bundle.latency_seconds:.3f}s",
                flush=True,
            )
            return 0
        finally:
            if env is not None:
                safe_close_scienceworld_env(env)
            retriever.close()

    if not args.model:
        raise ValueError("--model is required unless --validate-only is used.")
    run_metadata = _build_run_metadata(args, protocol, root)
    total = len(selected) * attempts
    completed_count = 0
    records: list[dict[str, Any]] = []
    pending_jobs: list[EpisodeJob] = []
    summary_path = _summary_path(args.output_dir, args.retriever, args.model)

    for attempt_index in range(1, attempts + 1):
        for spec in selected:
            result_path = _result_path(
                args.output_dir,
                args.retriever,
                args.model,
                attempt_index,
                spec,
            )
            if _is_valid_existing_result(
                result_path,
                protocol=protocol,
                manifest_sha256=manifest_sha256,
                prompt_sha256=prompt_sha256,
                retriever_name=args.retriever,
                model=args.model,
                spec=spec,
                attempt_index=attempt_index,
                run_metadata=run_metadata,
            ):
                records.append(json.loads(result_path.read_text(encoding="utf-8")))
                completed_count += 1
                print(
                    f"{_progress_bar(completed_count, total)} "
                    f"{completed_count}/{total} resume episode={spec.episode_id:03d}",
                    flush=True,
                )
            else:
                pending_jobs.append(
                    EpisodeJob(spec=spec, attempt_index=attempt_index, result_path=result_path)
                )

    print(
        f"Run start: profile=unseen211; method={args.retriever}; model={args.model}; "
        f"total={total}; resumed={completed_count}; pending={len(pending_jobs)}; "
        f"max_workers={args.max_workers}; evaluator={run_metadata['evaluator_fingerprint']}",
        flush=True,
    )

    retriever_registry: list[Retriever] = []
    registry_lock = threading.Lock()
    try:
        if pending_jobs:
            worker_count = min(args.max_workers, len(pending_jobs))
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix=f"scienceworld-eto211-{args.retriever}",
                initializer=_initialize_worker,
                initargs=(args, root, protocol, retriever_registry, registry_lock),
            ) as executor:
                future_to_job = {
                    executor.submit(
                        _run_episode_job,
                        job,
                        protocol=protocol,
                        prompt=prompt,
                        manifest_sha256=manifest_sha256,
                        prompt_sha256=prompt_sha256,
                        retriever_name=args.retriever,
                        model=args.model,
                        run_metadata=run_metadata,
                    ): job
                    for job in pending_jobs
                }
                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        record = future.result()
                    except Exception as exc:
                        record = _scheduler_failure_record(
                            job,
                            exc,
                            protocol=protocol,
                            manifest_sha256=manifest_sha256,
                            prompt_sha256=prompt_sha256,
                            retriever_name=args.retriever,
                            model=args.model,
                            run_metadata=run_metadata,
                        )
                    records.append(record)
                    records.sort(
                        key=lambda value: (
                            int(value.get("attempt_index", 0)),
                            int((value.get("episode") or {}).get("episode_id", -1)),
                        )
                    )
                    completed_count += 1
                    summary = _write_summary(
                        summary_path,
                        records=records,
                        selected_count=total,
                        full_manifest_count=len(specs) * attempts,
                        protocol=protocol,
                        manifest_sha256=manifest_sha256,
                        prompt_sha256=prompt_sha256,
                        retriever_name=args.retriever,
                        model=args.model,
                        run_metadata=run_metadata,
                    )
                    metrics = summary["metrics"]
                    mean_score = metrics["episode_mean_best_official_score"]
                    mean_text = "n/a" if mean_score is None else f"{mean_score:.2f}"
                    print(
                        f"{_progress_bar(completed_count, total)} "
                        f"{completed_count}/{total} episode={job.spec.episode_id:03d} "
                        f"status={record['status']} score={record.get('best_official_score')} "
                        f"mean={mean_text} infra={metrics['infrastructure_error_count']}",
                        flush=True,
                    )
    finally:
        for retriever in retriever_registry:
            retriever.close()

    records.sort(
        key=lambda value: (
            int(value.get("attempt_index", 0)),
            int((value.get("episode") or {}).get("episode_id", -1)),
        )
    )
    summary = _write_summary(
        summary_path,
        records=records,
        selected_count=total,
        full_manifest_count=len(specs) * attempts,
        protocol=protocol,
        manifest_sha256=manifest_sha256,
        prompt_sha256=prompt_sha256,
        retriever_name=args.retriever,
        model=args.model,
        run_metadata=run_metadata,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if summary["metrics"]["infrastructure_error_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
