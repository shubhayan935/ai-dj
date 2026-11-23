"""Evaluation runner comparing DJ planner baselines across splits and conditioning modes."""

from __future__ import annotations

import json
from pathlib import Path

from ai_dj.data.schemas import SFTExample
from ai_dj.eval.baselines import (
    BPMProximityPlanner,
    DJPlanner,
    HeuristicDJPlanner,
    RandomPlanner,
)
from ai_dj.eval.metrics import evaluate_predictions


def load_sft_split(path: Path) -> list[SFTExample]:
    """Load SFTExample records from a JSONL split file."""
    examples: list[SFTExample] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(SFTExample.model_validate_json(line))
    return examples


def run_planner_eval(
    planner: DJPlanner,
    examples: list[SFTExample],
    *,
    include_dj: bool = True,
) -> dict:
    preds = [planner.predict(ex, include_dj=include_dj) for ex in examples]
    return evaluate_predictions(examples, preds)


def write_eval_markdown(results: dict, out_path: Path) -> Path:
    """Write a concise Markdown benchmark table for offline baselines."""
    lines: list[str] = [
        "# AI DJ Offline Evaluation Baselines",
        "",
        "Evaluated on DJ-disjoint held-out splits (20-track BPM-matched candidate pools).",
        "",
    ]
    for split_name, split_res in results["splits"].items():
        lines.extend(
            [
                f"## Split: `{split_name}` ({split_res['num_examples']:,} examples)",
                "",
                "| Planner | Mode | Recall@1 | Recall@5 | MRR | NDCG@10 | "
                "Overlap MAE | Within 8b | Style Acc |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in split_res["runs"]:
            m = row["metrics"]["overall"]
            lines.append(
                f"| `{row['planner']}` | `{row['mode']}` | {m['recall_at_1']:.3f} | "
                f"{m['recall_at_5']:.3f} | {m['mrr']:.3f} | {m['ndcg_at_10']:.3f} | "
                f"{m['overlap_beats_mae']:.1f} | {m['overlap_within_8_beats']:.3f} | "
                f"{m['style_accuracy']:.3f} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def run_baseline_suite(processed_dir: Path, splits: tuple[str, ...] = ("val", "test")) -> dict:
    """Run all deterministic baselines in with_dj and without_dj modes and save reports."""
    planners: list[DJPlanner] = [
        RandomPlanner(seed=42),
        BPMProximityPlanner(),
        HeuristicDJPlanner(),
    ]

    results: dict = {"splits": {}}
    for split_name in splits:
        split_path = processed_dir / f"sft_{split_name}.jsonl"
        examples = load_sft_split(split_path)
        runs: list[dict] = []
        for planner in planners:
            for mode, include_dj in (("with_dj", True), ("without_dj", False)):
                metrics = run_planner_eval(planner, examples, include_dj=include_dj)
                runs.append(
                    {
                        "planner": planner.name,
                        "mode": mode,
                        "metrics": metrics,
                    }
                )
        results["splits"][split_name] = {
            "num_examples": len(examples),
            "runs": runs,
        }

    (processed_dir / "eval_baselines.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    write_eval_markdown(results, processed_dir / "eval_baselines.md")
    return results
