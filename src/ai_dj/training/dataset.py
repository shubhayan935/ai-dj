"""Phase 3 SFT and DPO chat-dataset exporter with token and Tinker cost estimation."""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path

from ai_dj.data.schemas import DPOPair, QualityTier, SFTExample
from ai_dj.training.prompts import format_dpo_pair_messages, format_sft_messages

# Verified Tinker training prices (USD per 1M train tokens, 64K context)
TINKER_TRAIN_PRICE_PER_1M: dict[str, float] = {
    "thinkingmachines/Inkling-2-Mini": 0.75,
    "Qwen/Qwen3.6-35B-A3B": 1.177,
    "thinkingmachines/Inkling-2-Small": 1.50,
    "thinkingmachines/Inkling-Small": 1.73,
    "moonshotai/Kimi-K2.6": 4.84,
}


def approx_token_count(text: str) -> int:
    """Fast BPE-free token count estimate (~4 chars/token for English + JSON)."""
    return max(1, int(round(len(text) / 4.0)))


def _atomic_write_jsonl(rows: list[dict], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=out_path.parent,
        prefix=f".{out_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as fh:
        tmp_path = Path(fh.name)
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(out_path)
    return out_path


def _messages_tokens(messages: list[dict[str, str]]) -> int:
    return sum(approx_token_count(m["content"]) + 4 for m in messages)


def export_training_datasets(
    processed_dir: Path = Path("data/processed"),
    out_dir: Path = Path("data/processed/training"),
    *,
    train_tiers: tuple[QualityTier, ...] = ("gold", "silver"),
    max_dpo_pairs: int | None = None,
) -> dict:
    """Export Tinker/chat-ready SFT and DPO JSONL datasets + token/cost summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    allowed_tiers = set(train_tiers)

    split_stats: dict[str, dict] = {}
    for split_name in ("train", "val", "test"):
        src_path = processed_dir / f"sft_{split_name}.jsonl"
        if not src_path.exists():
            continue

        rows: list[dict] = []
        tier_counts: Counter[str] = Counter()
        prompt_tokens_total = 0
        completion_tokens_total = 0
        with_dj_count = 0

        with src_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                ex = SFTExample.model_validate_json(line)
                if split_name == "train" and ex.tier not in allowed_tiers:
                    continue
                msgs = format_sft_messages(ex, include_dj=True)
                p_tok = _messages_tokens(msgs[:2])
                c_tok = _messages_tokens(msgs[2:])
                prompt_tokens_total += p_tok
                completion_tokens_total += c_tok
                tier_counts[ex.tier] += 1
                if ex.set_brief.style_reference_dj:
                    with_dj_count += 1
                rows.append(
                    {
                        "example_id": ex.example_id,
                        "split": ex.split,
                        "tier": ex.tier,
                        "mix_id": ex.mix_id,
                        "target_track_id": ex.target_track_id,
                        "messages": msgs,
                    }
                )

        out_path = _atomic_write_jsonl(rows, out_dir / f"sft_{split_name}_chat.jsonl")
        total_tok = prompt_tokens_total + completion_tokens_total
        split_stats[split_name] = {
            "path": str(out_path),
            "examples": len(rows),
            "tier_counts": dict(tier_counts),
            "with_dj_pct": round(100.0 * with_dj_count / len(rows), 1) if rows else 0.0,
            "approx_prompt_tokens": prompt_tokens_total,
            "approx_completion_tokens": completion_tokens_total,
            "approx_total_tokens": total_tok,
            "avg_tokens_per_example": round(total_tok / len(rows), 1) if rows else 0.0,
        }

    # Export DPO preference pairs
    dpo_src = processed_dir / "dpo_train.jsonl"
    dpo_stats: dict = {"examples": 0, "approx_total_tokens": 0, "negative_types": {}}
    if dpo_src.exists():
        dpo_rows: list[dict] = []
        neg_counts: Counter[str] = Counter()
        dpo_tokens = 0
        with dpo_src.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                pair = DPOPair.model_validate_json(line)
                formatted = format_dpo_pair_messages(pair, include_dj=True)
                prompt_tok = _messages_tokens(formatted["prompt"])  # type: ignore[arg-type]
                chosen_tok = _messages_tokens(formatted["chosen"])  # type: ignore[arg-type]
                rejected_tok = _messages_tokens(formatted["rejected"])  # type: ignore[arg-type]
                # DPO runs forward_backward on BOTH (prompt+chosen) and (prompt+rejected)
                dpo_tokens += (2 * prompt_tok) + chosen_tok + rejected_tok
                neg_counts[pair.negative_type] += 1
                dpo_rows.append(formatted)
                if max_dpo_pairs is not None and len(dpo_rows) >= max_dpo_pairs:
                    break

        dpo_out = _atomic_write_jsonl(dpo_rows, out_dir / "dpo_train_chat.jsonl")
        dpo_stats = {
            "path": str(dpo_out),
            "examples": len(dpo_rows),
            "negative_types": dict(neg_counts),
            "approx_total_tokens": dpo_tokens,
            "avg_tokens_per_pair": round(dpo_tokens / len(dpo_rows), 1) if dpo_rows else 0.0,
        }

    sft_train_tok = split_stats.get("train", {}).get("approx_total_tokens", 0)
    dpo_train_tok = dpo_stats.get("approx_total_tokens", 0)

    # Estimate cost for recommended Phase 3 recipe: 2 epochs SFT + 1 epoch DPO
    cost_estimates: dict[str, dict[str, float]] = {}
    for model_id, price_per_1m in TINKER_TRAIN_PRICE_PER_1M.items():
        sft_2ep = round((2 * sft_train_tok / 1_000_000.0) * price_per_1m, 2)
        dpo_1ep = round((dpo_train_tok / 1_000_000.0) * price_per_1m, 2)
        cost_estimates[model_id] = {
            "price_per_1m_train_usd": price_per_1m,
            "sft_2_epochs_usd": sft_2ep,
            "dpo_1_epoch_usd": dpo_1ep,
            "total_phase3_usd": round(sft_2ep + dpo_1ep, 2),
        }

    summary = {
        "train_tiers": list(train_tiers),
        "sft_splits": split_stats,
        "dpo_train": dpo_stats,
        "tinker_cost_estimates_usd": cost_estimates,
    }
    (out_dir / "training_dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_summary_markdown(summary, out_dir / "training_dataset_summary.md")
    return summary


def _write_summary_markdown(summary: dict, out_path: Path) -> Path:
    lines: list[str] = [
        "# Phase 3 Training Dataset & Tinker Cost Summary",
        "",
        f"SFT training tiers: `{', '.join(summary['train_tiers'])}` "
        "(validation and test retain all tiers for apples-to-apples baseline comparison).",
        "",
        "## Exported Splits",
        "",
        "| Split | Examples | Avg Tokens | Approx Total Tokens | With-DJ % |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, st in summary["sft_splits"].items():
        lines.append(
            f"| `sft_{name}` | {st['examples']:,} | {st['avg_tokens_per_example']:.0f} | "
            f"{st['approx_total_tokens']:,} | {st['with_dj_pct']:.1f}% |"
        )
    dpo = summary["dpo_train"]
    if dpo.get("examples"):
        lines.append(
            f"| `dpo_train` (2× pass) | {dpo['examples']:,} | "
            f"{dpo['avg_tokens_per_pair']:.0f} | {dpo['approx_total_tokens']:,} | — |"
        )

    lines.extend(
        [
            "",
            "## Estimated Tinker LoRA Cost (2 Epochs SFT + 1 Epoch DPO)",
            "",
            "| Tinker Model ID | Train $/1M | SFT (2 ep) | DPO (1 ep) | Total Phase 3 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for model_id, c in summary["tinker_cost_estimates_usd"].items():
        lines.append(
            f"| `{model_id}` | ${c['price_per_1m_train_usd']:.3f} | "
            f"${c['sft_2_epochs_usd']:.2f} | ${c['dpo_1_epoch_usd']:.2f} | "
            f"**${c['total_phase3_usd']:.2f}** |"
        )
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
