"""Run one numbered A1--A4 ALFWorld condition through the frozen A0 evaluator."""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ablation_experiments.adapters.causal_graph_views import sha256_file  # noqa: E402
from ablation_experiments.adapters.numbered_graph_views import (  # noqa: E402
    PROTOCOL_ID,
    VARIANTS,
)
from ablation_experiments.adapters.numbered_runtime import (  # noqa: E402
    InfrastructureRetrievalError,
    cache_stats,
    configure_numbered_engine,
)
from ablation_experiments.runners import run_alfworld_main_parity as runner  # noqa: E402


evaluator = runner.base
ProductionSkillModule = evaluator.SkillModule


def _argument_value(name: str) -> str:
    try:
        index = sys.argv.index(name)
    except ValueError as exc:
        raise ValueError(f"Missing required argument: {name}") from exc
    if index + 1 >= len(sys.argv):
        raise ValueError(f"Missing value for argument: {name}")
    return sys.argv[index + 1]


VARIANT = _argument_value("--variant")
if VARIANT not in VARIANTS:
    raise ValueError(f"Unsupported numbered ablation variant: {VARIANT}")
WORKSPACE = Path(_argument_value("--workspace")).expanduser().resolve()
EMBEDDING_CACHE = Path(
    os.environ["CASKG_ABLATION_EMBEDDING_CACHE"]
).expanduser().resolve()
RETRIEVAL_AUDIT_DIR = Path(
    os.environ["CASKG_ABLATION_RETRIEVAL_AUDIT_DIR"]
).expanduser().resolve()
TRACE_DIR = Path(os.environ["CASKG_ABLATION_TRACE_DIR"]).expanduser().resolve()


class NumberedSkillModule(ProductionSkillModule):
    """Production prompt/mode surface with only the retrieval engine wrapped."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.rag is None:
            raise RuntimeError("Numbered ALFWorld ablation requires a CaSKG engine.")
        configure_numbered_engine(
            self.rag,
            variant=VARIANT,
            workspace=WORKSPACE,
            embedding_cache=EMBEDDING_CACHE,
            audit_dir=RETRIEVAL_AUDIT_DIR,
        )


_production_eval_single_game = evaluator.eval_single_game
_production_save_evaluation_trace = evaluator._save_evaluation_trace


def _fail_closed_eval_single_game(*args, **kwargs):
    try:
        return _production_eval_single_game(*args, **kwargs)
    except InfrastructureRetrievalError as exc:
        game_index = args[0] if args else "unknown"
        print(f"Error in game {game_index}: {exc}", flush=True)
        return None


def _write_isolated_evaluation_trace(*args, **kwargs):
    """Keep evaluator traces out of the frozen graph-view workspace."""

    del args
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    kwargs["workspace"] = str(TRACE_DIR)
    return _production_save_evaluation_trace(**kwargs)


_production_summarize = runner._summarize


def _numbered_summarize(*args, **kwargs):
    summary = _production_summarize(*args, **kwargs)
    summary.update(
        {
            "protocol_id": PROTOCOL_ID,
            "numbered_ablation_variant": VARIANT,
            "retrieval_runtime_adapter_sha256": sha256_file(
                REPOSITORY_ROOT
                / "ablation_experiments"
                / "adapters"
                / "numbered_runtime.py"
            ),
            "embedding_cache": cache_stats(EMBEDDING_CACHE),
            "retrieval_audit_dir": str(RETRIEVAL_AUDIT_DIR),
            "evaluation_trace_dir": str(TRACE_DIR),
            "a1_vector_only_keeps_caskg_prompt_surface": VARIANT.startswith("a1-"),
        }
    )
    output_dir = Path(args[0]) if args else Path(kwargs["output_dir"])
    runner._write_json_atomic(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    evaluator.SkillModule = NumberedSkillModule
    evaluator.eval_single_game = _fail_closed_eval_single_game
    evaluator._save_evaluation_trace = _write_isolated_evaluation_trace
    runner.base.SkillModule = NumberedSkillModule
    runner.base.eval_single_game = _fail_closed_eval_single_game
    runner.base._save_evaluation_trace = _write_isolated_evaluation_trace
    runner.PROTOCOL_ID = PROTOCOL_ID
    runner._summarize = _numbered_summarize
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
