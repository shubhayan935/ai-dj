"""Evaluate post-trained or zero-shot LLM predictions against Phase 1 DJ baselines."""

from __future__ import annotations

import json
import os
from pathlib import Path

from ai_dj.data.schemas import SFTExample
from ai_dj.eval.baselines import BPMProximityPlanner, HeuristicDJPlanner, RandomPlanner
from ai_dj.eval.metrics import PlannerPrediction, evaluate_predictions
from ai_dj.eval.runner import load_sft_split, run_planner_eval
from ai_dj.training.prompts import SYSTEM_PROMPT, format_user_prompt, parse_model_output


def evaluate_raw_predictions_jsonl(
    split_jsonl: Path,
    predictions_jsonl: Path,
    *,
    model_name: str = "posttrained_model",
    include_dj: bool = True,
    out_md: Path | None = None,
) -> dict:
    """Score a JSONL of `{example_id, raw_output}` completions against Phase 1 baselines."""
    all_examples = load_sft_split(split_jsonl)
    ex_by_id = {ex.example_id: ex for ex in all_examples}

    matched_examples: list[SFTExample] = []
    parsed_preds: list[PlannerPrediction] = []
    json_valid = 0
    valid_track_id = 0
    style_valid = 0

    with predictions_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ex_id = str(row["example_id"])
            ex = ex_by_id.get(ex_id)
            if ex is None:
                continue
            raw_output = str(row.get("raw_output", ""))
            pred, diag = parse_model_output(raw_output, ex.candidates)
            matched_examples.append(ex)
            parsed_preds.append(pred)
            if diag.json_valid:
                json_valid += 1
            if diag.valid_track_id:
                valid_track_id += 1
            if diag.style_valid:
                style_valid += 1

    if not matched_examples:
        raise ValueError(
            f"No prediction example_ids from {predictions_jsonl} matched {split_jsonl}"
        )

    n = len(matched_examples)
    model_metrics = evaluate_predictions(matched_examples, parsed_preds)
    schema_health = {
        "json_valid_rate": round(json_valid / n, 4),
        "valid_track_id_rate": round(valid_track_id / n, 4),
        "style_valid_rate": round(style_valid / n, 4),
    }

    baseline_runs: list[dict] = []
    for planner in (RandomPlanner(seed=42), BPMProximityPlanner(), HeuristicDJPlanner()):
        baseline_runs.append(
            {
                "planner": planner.name,
                "metrics": run_planner_eval(planner, matched_examples, include_dj=include_dj),
            }
        )

    report = {
        "model_name": model_name,
        "split_path": str(split_jsonl),
        "mode": "with_dj" if include_dj else "without_dj",
        "num_examples": n,
        "schema_health": schema_health,
        "model_metrics": model_metrics,
        "baselines": baseline_runs,
    }

    if out_md is not None:
        _write_comparison_markdown(report, out_md)
    return report


def sample_tinker_checkpoint(
    split_jsonl: Path,
    sampler_checkpoint: str,
    base_model: str,
    out_predictions_jsonl: Path,
    *,
    include_dj: bool = True,
    max_examples: int = 200,
    max_tokens: int = 160,
    temperature: float = 0.0,
) -> Path:
    """Sample completions from a trained Tinker LoRA checkpoint onto a JSONL file."""
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is not set; cannot sample from Tinker checkpoint.")

    try:
        import tinker
    except ImportError as exc:
        raise RuntimeError(
            "Install `tinker` (`uv pip install tinker`) to sample checkpoints."
        ) from exc

    examples = load_sft_split(split_jsonl)[:max_examples]
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(
        model_path=sampler_checkpoint
    )
    tokenizer = service_client.create_lora_training_client(
        base_model=base_model, rank=4
    ).result().get_tokenizer()

    out_predictions_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_predictions_jsonl.open("w", encoding="utf-8") as fh:
        for ex in examples:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": format_user_prompt(
                        ex.set_brief,
                        ex.history,
                        ex.current_track,
                        ex.candidates,
                        include_dj=include_dj,
                    ),
                },
            ]
            prompt_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            res = sampling_client.sample(
                prompt=tinker.ModelInput.from_ints(list(prompt_ids)),
                num_samples=1,
                sampling_params=tinker.SamplingParams(
                    max_tokens=max_tokens,
                    temperature=temperature,
                ),
            ).result()
            raw_text = tokenizer.decode(res.sequences[0].tokens, skip_special_tokens=True)
            fh.write(
                json.dumps({"example_id": ex.example_id, "raw_output": raw_text}) + "\n"
            )
    return out_predictions_jsonl


def _write_comparison_markdown(report: dict, out_path: Path) -> Path:
    lines: list[str] = [
        f"# Model Evaluation vs. Phase 1 Baselines (`{report['model_name']}`)",
        "",
        f"- Examples evaluated: **{report['num_examples']:,}** (`{report['mode']}`)",
        f"- Schema health: JSON valid **{report['schema_health']['json_valid_rate'] * 100:.1f}%**"
        f" · Valid track ID **{report['schema_health']['valid_track_id_rate'] * 100:.1f}%**"
        f" · Valid style **{report['schema_health']['style_valid_rate'] * 100:.1f}%**",
        "",
        "| Planner | Recall@1 | Recall@5 | MRR | NDCG@10 | Overlap MAE | Within 8b | Style Acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for b in report["baselines"]:
        m = b["metrics"]["overall"]
        lines.append(
            f"| `{b['planner']}` | {m['recall_at_1']:.3f} | {m['recall_at_5']:.3f} | "
            f"{m['mrr']:.3f} | {m['ndcg_at_10']:.3f} | {m['overlap_beats_mae']:.1f} | "
            f"{m['overlap_within_8_beats']:.3f} | {m['style_accuracy']:.3f} |"
        )
    mm = report["model_metrics"]["overall"]
    lines.append(
        f"| **`{report['model_name']}`** | **{mm['recall_at_1']:.3f}** | "
        f"**{mm['recall_at_5']:.3f}** | **{mm['mrr']:.3f}** | **{mm['ndcg_at_10']:.3f}** | "
        f"**{mm['overlap_beats_mae']:.1f}** | **{mm['overlap_within_8_beats']:.3f}** | "
        f"**{mm['style_accuracy']:.3f}** |"
    )
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
