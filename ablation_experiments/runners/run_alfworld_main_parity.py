"""Run the original CaSKG ALFWorld episode path with isolated result plumbing.

This adapter deliberately delegates every episode to evaluation.alfworld_run.
It changes only output placement, per-episode logging, and infrastructure-error
archiving so A0 retains the main experiment's prompt, query, dynamic skill
requests, runtime recovery, parser, and turn accounting.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _load_environment() -> tuple[str, str]:
    try:
        from dotenv import load_dotenv

        load_dotenv(REPOSITORY_ROOT / ".env", override=False)
    except ImportError:
        pass

    master_key = os.environ.get("ROUTER_MASTER_KEY", "").strip()
    router_env_value = os.environ.get("CASKG_ROUTER_ENV", "").strip()
    if not master_key and router_env_value:
        router_env = Path(router_env_value).expanduser().resolve()
        for raw_line in router_env.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "ROUTER_MASTER_KEY":
                master_key = value.strip().strip('"').strip("'")
                break
    if not master_key:
        raise RuntimeError("Set ROUTER_MASTER_KEY in the environment or .env file.")

    router_base = os.environ.get("CASKG_ROUTER_BASE", "http://127.0.0.1:4000/v1")
    os.environ["API_KEY"] = master_key
    os.environ["BASE_URL"] = router_base
    os.environ["OPENAI_API_KEY"] = master_key
    os.environ["OPENAI_BASE_URL"] = router_base
    return router_base, master_key


ROUTER_BASE, ROUTER_MASTER_KEY = _load_environment()

from openai import OpenAI  # noqa: E402
from tqdm import tqdm  # noqa: E402

from evaluation import alfworld_run as base  # noqa: E402
from ablation_experiments.adapters.causal_graph_views import sha256_file  # noqa: E402
from ablation_experiments.adapters.main_parity_views import PROTOCOL_ID  # noqa: E402


EXPECTED_PROVIDER = os.environ.get("CASKG_EXPECTED_CHAT_PROVIDER", "aigcbest_chat")
SESSION_ID = os.environ.get("CASKG_ROUTER_SESSION_ID", "c2-main-parity")
base.client = OpenAI(
    api_key=ROUTER_MASTER_KEY,
    base_url=ROUTER_BASE,
    default_headers={
        "X-Router-Policy": "strict",
        "X-Router-Provider": EXPECTED_PROVIDER,
        "X-Router-Session-ID": SESSION_ID,
    },
)


TERMINAL_INFRA_MARKERS = (
    "Error in LLM call:",
    "Error in game ",
    "InfrastructureRetrievalError",
    "Embedding request failed after",
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _task_identity(gamefile: str) -> str:
    """Return the stable ALFWorld task/trial identity, independent of root path."""
    path = Path(gamefile)
    return f"{path.parent.parent.name}/{path.parent.name}"


def _evaluator_game_order(gamefiles: list[str]) -> list[str]:
    """Match TextWorldBatchGymEnv's fixed first-cycle shuffle exactly."""

    order = list(gamefiles)
    np.random.RandomState(1234).shuffle(order)
    return order


def _load_task_manifest(path: Path) -> tuple[dict[int, dict[str, Any]], str]:
    payload = json.loads(path.read_text(encoding="ascii"))
    if payload.get("protocol_id") != PROTOCOL_ID or payload.get("benchmark_id") != "alfworld-id140":
        raise ValueError("Historical ALFWorld task manifest belongs to another protocol/benchmark.")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 140:
        raise ValueError("Historical ALFWorld task manifest must contain 140 records.")
    by_episode: dict[int, dict[str, Any]] = {}
    identities: set[str] = set()
    for record in records:
        episode_id = int(record["episode_id"])
        name = str(record["name"])
        query = str(record["query"])
        if episode_id in by_episode or name in identities:
            raise ValueError("Historical ALFWorld task manifest contains duplicate identities.")
        if _sha256_text(query) != str(record["query_sha256"]):
            raise ValueError(f"Historical ALFWorld query hash mismatch at episode {episode_id}.")
        by_episode[episode_id] = record
        identities.add(name)
    if sorted(by_episode) != list(range(140)):
        raise ValueError("Historical ALFWorld task IDs must be exactly 0..139.")
    checksum_path = path.with_suffix(".sha256")
    if not checksum_path.is_file():
        raise FileNotFoundError(checksum_path)
    expected = checksum_path.read_text(encoding="ascii").split()[0].lower()
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError("Historical ALFWorld task manifest checksum mismatch.")
    return by_episode, actual


def _read_result(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    required = {"name", "task_done", "reward", "steps", "messages", "retrieval_query"}
    if not isinstance(value, dict) or not required.issubset(value):
        return None
    if not isinstance(value["task_done"], bool) or not isinstance(value["messages"], list):
        return None
    return value


def _archive_infrastructure_result(
    result_path: Path,
    output_dir: Path,
    pass_index: int,
) -> None:
    if not result_path.exists():
        return
    destination = (
        output_dir
        / "infrastructure_history"
        / f"pass_{pass_index:02d}"
        / result_path.name
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(result_path), str(destination))


def _run_one(
    game_index: int,
    episode_id: int,
    worker_args: argparse.Namespace,
    config: dict[str, Any],
    split: str,
    output_dir: str,
    log_dir: str,
    pass_index: int,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    episode_log = Path(log_dir) / f"idx_{episode_id:03d}.log"
    episode_log.parent.mkdir(parents=True, exist_ok=True)
    # The production evaluator writes by current environment index. The
    # frozen manifest is keyed by historical episode ID, so normalize the
    # result filename before the summary reads it.
    raw_result_path = output_path / f"idx_{game_index}.json"
    result_path = output_path / f"idx_{episode_id}.json"
    with episode_log.open("w", encoding="utf-8", newline="\n") as handle:
        with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            result = base.eval_single_game(
                game_index,
                worker_args,
                config,
                split,
                str(output_path),
            )

    if raw_result_path.is_file() and raw_result_path != result_path:
        if result_path.exists():
            raise RuntimeError(
                f"Refusing to overwrite existing historical result: {result_path}"
            )
        raw_result_path.replace(result_path)

    log_text = episode_log.read_text(encoding="utf-8", errors="replace")
    terminal_infra = result is None or any(
        marker in log_text for marker in TERMINAL_INFRA_MARKERS
    )
    if terminal_infra:
        _archive_infrastructure_result(result_path, output_path, pass_index)
        return {
            "game_index": game_index,
            "episode_id": episode_id,
            "status": "infra_error",
            "log": str(episode_log),
        }
    if _read_result(result_path) is None:
        _archive_infrastructure_result(result_path, output_path, pass_index)
        return {
            "game_index": game_index,
            "episode_id": episode_id,
            "status": "invalid_result",
            "log": str(episode_log),
        }
    return {
        "game_index": game_index,
        "episode_id": episode_id,
        "status": "success" if bool(result["task_done"]) else "model_failure",
        "log": str(episode_log),
    }


def _parse_indices(raw_value: str) -> list[int] | None:
    if not raw_value.strip():
        return None
    values = [int(item.strip()) for item in raw_value.split(",") if item.strip()]
    if len(values) != len(set(values)):
        raise ValueError("--task-indices contains duplicates.")
    return values


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    temporary.replace(path)


def _summarize(
    output_dir: Path,
    selected_indices: list[int],
    *,
    variant: str,
    model: str,
    workspace: Path,
    max_workers: int,
    task_manifest: dict[int, dict[str, Any]] | None = None,
    task_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    missing: list[int] = []
    for index in selected_indices:
        record = _read_result(output_dir / f"idx_{index}.json")
        if record is None:
            missing.append(index)
        else:
            if task_manifest is not None:
                expected = task_manifest[index]
                if record["name"] != expected["name"] or record["retrieval_query"] != expected["query"]:
                    raise RuntimeError(
                        f"Historical ALFWorld task identity/query mismatch at episode {index}."
                    )
            records.append(record)
    success_count = sum(bool(record["task_done"]) for record in records)
    total_steps = sum(int(record["steps"]) for record in records)
    llm_calls = sum(
        int((record.get("token_usage") or {}).get("llm_calls", 0)) for record in records
    )
    usage_fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    usage = {
        field: sum(
            int((record.get("token_usage") or {}).get(field, 0)) for record in records
        )
        for field in usage_fields
    }
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "benchmark_profile": "ALFWorld-ID-140-main-protocol",
        "variant": variant,
        "model": model,
        "expected_router_provider": EXPECTED_PROVIDER,
        "workspace": str(workspace),
        "workspace_graph_sha256": sha256_file(workspace / "graph_igraph_data.pklz"),
        "base_evaluator_sha256": sha256_file(REPOSITORY_ROOT / "evaluation" / "alfworld_run.py"),
        "skill_module_sha256": sha256_file(REPOSITORY_ROOT / "evaluation" / "skill.py"),
        "system_prompt_sha256": _sha256_text(base.alfworld_system_prompt),
        "dynamic_skill_requests_enabled": True,
        "runtime_failure_recovery_enabled": True,
        "query_source": "raw_initial_observation_plus_goal",
        "top_n": 15,
        "max_steps": 30,
        "max_workers": max_workers,
        "selected_indices": selected_indices,
        "historical_task_manifest_sha256": task_manifest_sha256,
        "historical_task_identity_parity": task_manifest is not None,
        "metrics": {
            "planned_records": len(selected_indices),
            "valid_score_count": len(records),
            "missing_or_infrastructure_count": len(missing),
            "success_count": success_count,
            "success_rate": success_count / len(records) if records else 0.0,
            "mean_agent_turns": total_steps / len(records) if records else 0.0,
            "total_llm_calls": llm_calls,
            "usage_totals": usage,
            "formal_complete": len(selected_indices) == 140 and not missing,
        },
        "missing_indices": missing,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_atomic(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--skills-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="MiniMax-M2.7")
    parser.add_argument("--split", choices=("dev", "ood"), default="dev")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--task-indices", default="")
    parser.add_argument("--historical-task-manifest", type=Path)
    parser.add_argument("--pass-index", type=int, default=1)
    args = parser.parse_args()

    if args.max_workers < 1 or args.max_steps != 30:
        raise ValueError("Main-parity ALFWorld requires max_workers>=1 and max_steps=30.")
    if args.split != "dev":
        raise ValueError("This ablation is locked to ALFWorld ID-140 (--split dev).")
    workspace = args.workspace.resolve()
    skills_dir = args.skills_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not (workspace / "graph_igraph_data.pklz").is_file():
        raise FileNotFoundError(f"Invalid CaSKG workspace: {workspace}")
    if skills_dir.name != "skills_1000":
        raise ValueError("Main-parity ablation requires the skill1000 library.")

    config_path = REPOSITORY_ROOT / "evaluation" / "alfworld" / "base_config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = base.yaml.safe_load(handle)
    split = "eval_in_distribution"
    temporary_env = base.get_environment(config["env"]["type"])(config, train_eval=split)
    temporary_env = temporary_env.init_env(batch_size=1)
    gamefiles = list(temporary_env.gamefiles)
    game_count = len(gamefiles)
    temporary_env.close()
    if game_count != 140:
        raise RuntimeError(f"Expected ALFWorld ID-140, found {game_count} games.")

    task_manifest: dict[int, dict[str, Any]] | None = None
    task_manifest_sha256: str | None = None
    historical_to_game_index = {index: index for index in range(game_count)}
    if args.historical_task_manifest is not None:
        task_manifest, task_manifest_sha256 = _load_task_manifest(args.historical_task_manifest.resolve())
        current_by_identity = {
            _task_identity(gamefile): index
            for index, gamefile in enumerate(_evaluator_game_order(gamefiles))
        }
        expected_identities = {record["name"] for record in task_manifest.values()}
        current_identities = set(current_by_identity)
        if current_identities != expected_identities:
            missing = sorted(expected_identities - current_identities)[:3]
            extra = sorted(current_identities - expected_identities)[:3]
            raise RuntimeError(
                f"Historical ALFWorld task set mismatch (missing={missing}, extra={extra})."
            )
        historical_to_game_index = {
            episode_id: current_by_identity[record["name"]]
            for episode_id, record in task_manifest.items()
        }

    selected = _parse_indices(args.task_indices) or list(range(game_count))
    if args.max_games is not None:
        selected = selected[: args.max_games]
    if any(index < 0 or index >= game_count for index in selected):
        raise ValueError("Selected ALFWorld task index is out of range.")

    existing = {
        index
        for index in selected
        if _read_result(output_dir / f"idx_{index}.json") is not None
    }
    pending = [index for index in selected if index not in existing]
    worker_args = argparse.Namespace(
        use_skill=True,
        mode="caskg",
        caskg_workspace=str(workspace),
        skills_dir=str(skills_dir),
        max_steps=30,
        model=args.model,
        enable_alfworld_gating=False,
    )
    log_dir = output_dir / "episode_logs" / f"pass_{args.pass_index:02d}"
    statuses: Counter[str] = Counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                _run_one,
                historical_to_game_index[index],
                index,
                worker_args,
                config,
                split,
                str(output_dir),
                str(log_dir),
                args.pass_index,
            ): index
            for index in pending
        }
        progress = tqdm(total=len(futures), desc=f"{args.variant} ALFWorld ID-140")
        for future in concurrent.futures.as_completed(futures):
            outcome = future.result()
            statuses[str(outcome["status"])] += 1
            progress.set_postfix(dict(statuses))
            progress.update(1)
        progress.close()

    summary = _summarize(
        output_dir,
        selected,
        variant=args.variant,
        model=args.model,
        workspace=workspace,
        max_workers=args.max_workers,
        task_manifest=task_manifest,
        task_manifest_sha256=task_manifest_sha256,
    )
    print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False), flush=True)
    return 0 if not summary["missing_indices"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
