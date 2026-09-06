"""Build isolated Skill1000 A1--A4 workspaces without touching A0 assets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ablation_experiments.adapters.numbered_graph_views import (  # noqa: E402
    build_numbered_graph_views,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-workspace",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "data"
            / "caskg_workspace"
            / "skills_1000_v32_scaffold_publish_gospath"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "ablation_experiments"
            / "graph_views"
            / "generated"
            / "a1-a4-main-parity-v1"
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = build_numbered_graph_views(
        args.source_workspace,
        args.output_root,
        force=args.force,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
