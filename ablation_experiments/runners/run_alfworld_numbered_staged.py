"""Run one numbered ALFWorld condition with collision-free per-episode staging.

The original numbered runner delegates to the frozen A0 evaluator.  This R5
adapter keeps that evaluator and retrieval surface unchanged, but prevents
parallel workers from sharing the evaluator's ``idx_<game_index>.json`` output
namespace.  Each episode gets a private staging directory; only after the
result passes validation is it atomically moved to the historical episode ID.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# WSL executes this file by absolute path; add the repository root before
# importing the sibling numbered runner package.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ablation_experiments.runners import run_alfworld_numbered as numbered


def _archive_staged_result(
    staged_result: Path,
    output_dir: Path,
    pass_index: int,
    episode_id: int,
) -> None:
    if not staged_result.exists():
        return
    destination = (
        output_dir
        / "infrastructure_history"
        / f"pass_{pass_index:02d}"
        / f"idx_{episode_id}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    staged_result.replace(destination)


def _staged_run_one(
    game_index: int,
    episode_id: int,
    worker_args,
    config,
    split: str,
    output_dir: str,
    log_dir: str,
    pass_index: int,
):
    output_path = Path(output_dir)
    episode_log = Path(log_dir) / f"idx_{episode_id:03d}.log"
    episode_log.parent.mkdir(parents=True, exist_ok=True)

    # A separate directory is essential: the frozen evaluator writes by
    # game_index, while the historical manifest is keyed by episode_id.  The
    # game-index permutation is shared by all workers, so a common directory
    # creates rename races under ProcessPoolExecutor.
    staging_dir = (
        output_path
        / ".worker_staging"
        / f"pass_{pass_index:02d}"
        / f"episode_{episode_id:03d}"
    )
    staging_dir.mkdir(parents=True, exist_ok=True)
    raw_result_path = staging_dir / f"idx_{game_index}.json"
    result_path = output_path / f"idx_{episode_id}.json"

    with episode_log.open("w", encoding="utf-8", newline="\n") as handle:
        import contextlib

        with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            result = numbered.runner.base.eval_single_game(
                game_index,
                worker_args,
                config,
                split,
                str(staging_dir),
            )

    log_text = episode_log.read_text(encoding="utf-8", errors="replace")
    terminal_infra = result is None or any(
        marker in log_text for marker in numbered.runner.TERMINAL_INFRA_MARKERS
    )
    if terminal_infra:
        _archive_staged_result(raw_result_path, output_path, pass_index, episode_id)
        shutil.rmtree(staging_dir, ignore_errors=True)
        return {
            "game_index": game_index,
            "episode_id": episode_id,
            "status": "infra_error",
            "log": str(episode_log),
        }

    if numbered.runner._read_result(raw_result_path) is None:
        _archive_staged_result(raw_result_path, output_path, pass_index, episode_id)
        shutil.rmtree(staging_dir, ignore_errors=True)
        return {
            "game_index": game_index,
            "episode_id": episode_id,
            "status": "invalid_result",
            "log": str(episode_log),
        }

    if result_path.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise RuntimeError(f"Refusing to overwrite existing historical result: {result_path}")
    raw_result_path.replace(result_path)
    shutil.rmtree(staging_dir, ignore_errors=True)
    return {
        "game_index": game_index,
        "episode_id": episode_id,
        "status": "success" if bool(result["task_done"]) else "model_failure",
        "log": str(episode_log),
    }


def _write_process_isolated_trace(*args, **kwargs):
    """Avoid a second shared-file race in evaluation trace output."""

    del args
    process_trace_dir = numbered.TRACE_DIR / f"worker_{os.getpid()}"
    process_trace_dir.mkdir(parents=True, exist_ok=True)
    kwargs["workspace"] = str(process_trace_dir)
    return numbered._production_save_evaluation_trace(**kwargs)


# ``numbered.main()`` delegates to ``run_alfworld_main_parity.main``.  That
# function resolves ``_run_one`` from its own module globals, so patching only
# the re-export on ``run_alfworld_numbered`` leaves the original shared-output
# worker active.  Patch both bindings and fail closed if either wiring changes.
numbered._run_one = _staged_run_one
numbered.runner._run_one = _staged_run_one
numbered._write_isolated_evaluation_trace = _write_process_isolated_trace

if numbered._run_one is not _staged_run_one or numbered.runner._run_one is not _staged_run_one:
    raise RuntimeError("Failed to install collision-free ALFWorld staged worker.")


if __name__ == "__main__":
    raise SystemExit(numbered.main())
