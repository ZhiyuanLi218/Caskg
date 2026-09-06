"""Analyze concurrent API2 P0 against formal MiniMax A1--A4 results."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_experiments.adapters.numbered_graph_views import (  # noqa: E402
    A0_CONTROL_VARIANT,
    A1_VARIANT,
    A2_VARIANT,
    A3_VARIANT,
    A4_VARIANT,
    PROTOCOL_ID,
    VARIANTS,
)


TREATMENT_VARIANTS = (A1_VARIANT, A2_VARIANT, A3_VARIANT, A4_VARIANT)
VARIANT_LABELS = {
    A0_CONTROL_VARIANT: "P0 API2 full CaSKG control",
    A1_VARIANT: "A1 Vector-only / No-PPR",
    A2_VARIANT: "A2 Matched rewired graph",
    A3_VARIANT: "A3 Phase-1 weights",
    A4_VARIANT: "A4 Shuffled Phase-2 evidence",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _load_alfworld(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for index in range(140):
        record_path = path / f"idx_{index}.json"
        if not record_path.is_file():
            raise FileNotFoundError(record_path)
        record = _read_json(record_path)
        if not isinstance(record.get("task_done"), bool):
            raise ValueError(f"Invalid ALFWorld record: {record_path}")
        records[index] = record
    return records


def _load_scienceworld(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for record_path in path.rglob("episode_*.json"):
        record = _read_json(record_path)
        episode = record.get("episode") or {}
        episode_id = int(episode.get("episode_id", -1))
        if episode_id < 0 or episode_id in records:
            raise ValueError(f"Invalid/duplicate ScienceWorld episode: {record_path}")
        if record.get("best_official_score") is None:
            raise ValueError(f"ScienceWorld score is missing: {record_path}")
        records[episode_id] = record
    if set(records) != set(range(211)):
        missing = sorted(set(range(211)) - set(records))
        raise ValueError(f"ScienceWorld records are incomplete; missing={missing}")
    return records


def _bootstrap_mean_difference(
    differences: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    bootstrap = differences[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "mean_difference": float(differences.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "bootstrap_samples": samples,
    }


def _exact_mcnemar(p0: np.ndarray, ablation: np.ndarray) -> dict[str, Any]:
    p0_wins = int(np.sum((p0 == 1) & (ablation == 0)))
    ablation_wins = int(np.sum((p0 == 0) & (ablation == 1)))
    discordant = p0_wins + ablation_wins
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, value)
            for value in range(0, min(p0_wins, ablation_wins) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "p0_success_ablation_failure": p0_wins,
        "p0_failure_ablation_success": ablation_wins,
        "discordant_pairs": discordant,
        "exact_two_sided_p": p_value,
    }


def _sign_flip_p(
    differences: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> float:
    observed = abs(float(differences.mean()))
    if not np.any(differences):
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(samples, len(differences)))
    permuted = np.abs((signs * differences).mean(axis=1))
    return float((np.sum(permuted >= observed) + 1) / (samples + 1))


def _holm(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, p_value) in enumerate(ordered):
        candidate = min(1.0, (total - rank) * p_value)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _load_retrieval_audits(audit_root: Path) -> dict[str, list[dict[str, Any]]]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for variant in VARIANTS:
        records: list[dict[str, Any]] = []
        variant_root = audit_root / variant
        if variant_root.is_dir():
            for path in variant_root.rglob("retrieval_pid_*.jsonl"):
                for line in path.read_text(encoding="ascii").splitlines():
                    if line.strip():
                        records.append(json.loads(line))
        by_variant[variant] = records
    return by_variant


def _retrieval_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"retrieval_call_count": 0}
    return {
        "retrieval_call_count": len(records),
        "unique_query_count": len({record["query_sha256"] for record in records}),
        "mean_context_chars": _mean(
            float(record["rendered_context_chars"]) for record in records
        ),
        "ppr_damping_counts": dict(
            sorted(
                Counter(
                    str((record.get("budget") or {}).get("ppr_damping"))
                    for record in records
                ).items()
            )
        ),
        "mean_relation_count": _mean(float(record["relation_count"]) for record in records),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CaSKG Skill1000 MiniMax 编号消融结果",
        "",
        "## Material Passport",
        "",
        f"- Protocol: `{report['protocol_id']}`",
        "- Verification Status: ANALYZED",
        "- A0: 原始历史主实验，仅作参考，不参与主效应检验",
        "- P0: 与 A1–A4 同期、同 API2、同 evaluator 的完整 CaSKG 控制组",
        "",
        "## 主结果",
        "",
        "| 条件 | ALFWorld 成功率 | P0-消融 (pp) [95% CI] | McNemar p / Holm p | ScienceWorld R | P0-消融 [95% CI] | Sign-flip p / Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in TREATMENT_VARIANTS:
        alf = report["alfworld"][variant]
        sci = report["scienceworld"][variant]
        lines.append(
            "| {label} | {alf_rate:.2%} | {alf_diff:.2f} [{alf_lo:.2f}, {alf_hi:.2f}] | "
            "{alf_p:.4g} / {alf_holm:.4g} | {sci_score:.2f} | "
            "{sci_diff:.2f} [{sci_lo:.2f}, {sci_hi:.2f}] | {sci_p:.4g} / {sci_holm:.4g} |".format(
                label=VARIANT_LABELS[variant],
                alf_rate=alf["ablation_success_rate"],
                alf_diff=100 * alf["effect"]["mean_difference"],
                alf_lo=100 * alf["effect"]["ci95_lower"],
                alf_hi=100 * alf["effect"]["ci95_upper"],
                alf_p=alf["mcnemar"]["exact_two_sided_p"],
                alf_holm=alf["holm_adjusted_p"],
                sci_score=sci["ablation_mean_score"],
                sci_diff=sci["effect"]["mean_difference"],
                sci_lo=sci["effect"]["ci95_lower"],
                sci_hi=sci["effect"]["ci95_upper"],
                sci_p=sci["sign_flip_two_sided_p"],
                sci_holm=sci["holm_adjusted_p"],
            )
        )
    lines.extend(
        [
            "",
            "## 解释规则",
            "",
            "正的 `P0-消融` 表示完整 CaSKG 更优；置信区间跨 0 时，只能称为方向性证据，不能声称稳定贡献。所有正常完成的结果均保留，无论是否符合预期。",
            "",
            "## 检索运行审计",
            "",
            "```json",
            json.dumps(report["retrieval_runtime"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=(
            ROOT
            / "ablation_experiments"
            / "results"
            / "a1-a4-main-parity-v1"
            / "formal"
            / "minimax-m27-a1-a4-main-parity-v1"
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=8801)
    parser.add_argument(
        "--historical-a0-alfworld",
        type=Path,
        default=ROOT / "results" / "historical" / "alfworld" / "MiniMax-M2.7",
    )
    parser.add_argument(
        "--historical-a0-scienceworld",
        type=Path,
        default=ROOT / "results" / "historical" / "scienceworld" / "MiniMax-M2.7",
    )
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    output_root = run_root / "analysis"
    output_root.mkdir(parents=True, exist_ok=True)

    a0_alf = _load_alfworld(args.historical_a0_alfworld.resolve())
    a0_sci = _load_scienceworld(args.historical_a0_scienceworld.resolve())
    a0_alf_values = np.array(
        [int(a0_alf[index]["task_done"]) for index in range(140)], dtype=float
    )
    a0_sci_values = np.array(
        [float(a0_sci[index]["best_official_score"]) for index in range(211)],
        dtype=float,
    )
    a0_sci_success = (a0_sci_values >= 100.0).astype(int)

    p0_alf = _load_alfworld(run_root / "alfworld-id140" / A0_CONTROL_VARIANT)
    p0_sci = _load_scienceworld(
        run_root / "scienceworld-unseen211" / A0_CONTROL_VARIANT
    )
    p0_alf_values = np.array(
        [int(p0_alf[index]["task_done"]) for index in range(140)], dtype=float
    )
    p0_sci_values = np.array(
        [float(p0_sci[index]["best_official_score"]) for index in range(211)],
        dtype=float,
    )
    p0_sci_success = (p0_sci_values >= 100.0).astype(int)

    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "historical_a0": {
            "rerun": False,
            "alfworld_success_count": int(a0_alf_values.sum()),
            "alfworld_success_rate": float(a0_alf_values.mean()),
            "scienceworld_mean_score": float(a0_sci_values.mean()),
            "scienceworld_full_success_rate": float(a0_sci_success.mean()),
        },
        "p0_api2_full_control": {
            "alfworld_success_count": int(p0_alf_values.sum()),
            "alfworld_success_rate": float(p0_alf_values.mean()),
            "scienceworld_mean_score": float(p0_sci_values.mean()),
            "scienceworld_full_success_rate": float(p0_sci_success.mean()),
        },
        "alfworld": {},
        "scienceworld": {},
        "analysis": {
            "paired_bootstrap_samples": args.bootstrap_samples,
            "confidence_level": 0.95,
            "random_seed": args.seed,
            "multiplicity": "Holm across four P0-minus-ablation contrasts per benchmark",
        },
        "limitations": [
            "Historical A0 is reported descriptively and is not used for P0-minus-ablation inference.",
            "One stochastic run per condition estimates realized-run sensitivity, not model-population variance.",
            "P0-A1-A4 use the accepted current ScienceWorld evaluator fingerprint 6973cd44cf0a64bc829648250d54f1364b194212fcb5dd222148e279d17c023a.",
        ],
    }

    alf_raw_p: dict[str, float] = {}
    sci_raw_p: dict[str, float] = {}
    for offset, variant in enumerate(TREATMENT_VARIANTS):
        variant_alf = _load_alfworld(run_root / "alfworld-id140" / variant)
        variant_sci = _load_scienceworld(run_root / "scienceworld-unseen211" / variant)
        alf_values = np.array(
            [int(variant_alf[index]["task_done"]) for index in range(140)],
            dtype=float,
        )
        sci_values = np.array(
            [float(variant_sci[index]["best_official_score"]) for index in range(211)],
            dtype=float,
        )
        sci_success = (sci_values >= 100.0).astype(int)

        alf_difference = p0_alf_values - alf_values
        sci_difference = p0_sci_values - sci_values
        alf_effect = _bootstrap_mean_difference(
            alf_difference,
            samples=args.bootstrap_samples,
            seed=args.seed + offset,
        )
        sci_effect = _bootstrap_mean_difference(
            sci_difference,
            samples=args.bootstrap_samples,
            seed=args.seed + 100 + offset,
        )
        mcnemar = _exact_mcnemar(p0_alf_values.astype(int), alf_values.astype(int))
        sign_flip_p = _sign_flip_p(
            sci_difference,
            samples=args.bootstrap_samples,
            seed=args.seed + 200 + offset,
        )
        full_success_mcnemar = _exact_mcnemar(p0_sci_success, sci_success)
        paired_sd = float(np.std(sci_difference, ddof=1))
        report["alfworld"][variant] = {
            "label": VARIANT_LABELS[variant],
            "ablation_success_count": int(alf_values.sum()),
            "ablation_success_rate": float(alf_values.mean()),
            "effect": alf_effect,
            "mcnemar": mcnemar,
        }
        report["scienceworld"][variant] = {
            "label": VARIANT_LABELS[variant],
            "ablation_mean_score": float(sci_values.mean()),
            "ablation_full_success_rate": float(sci_success.mean()),
            "effect": sci_effect,
            "paired_standardized_effect_dz": (
                float(sci_difference.mean() / paired_sd) if paired_sd > 0 else 0.0
            ),
            "sign_flip_two_sided_p": sign_flip_p,
            "full_success_mcnemar": full_success_mcnemar,
        }
        alf_raw_p[variant] = float(mcnemar["exact_two_sided_p"])
        sci_raw_p[variant] = sign_flip_p

    alf_holm = _holm(alf_raw_p)
    sci_holm = _holm(sci_raw_p)
    for variant in TREATMENT_VARIANTS:
        report["alfworld"][variant]["holm_adjusted_p"] = alf_holm[variant]
        report["scienceworld"][variant]["holm_adjusted_p"] = sci_holm[variant]

    audits = _load_retrieval_audits(
        ROOT
        / "ablation_experiments"
        / "audits"
        / "runtime"
        / "a1-a4-main-parity-v1"
    )
    report["retrieval_runtime"] = {
        variant: _retrieval_summary(records) for variant, records in audits.items()
    }

    json_path = output_root / "numbered_ablation_analysis.json"
    markdown_path = output_root / "NUMBERED_ABLATION_RESULTS.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    markdown_path.write_text(_markdown(report), encoding="utf-8", newline="\n")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
