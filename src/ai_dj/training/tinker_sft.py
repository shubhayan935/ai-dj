"""Tinker LoRA Supervised Fine-Tuning (SFT) trainer for the AI DJ planner."""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

from pydantic import BaseModel, Field

from ai_dj.training.dataset import TINKER_TRAIN_PRICE_PER_1M, approx_token_count


class SFTConfig(BaseModel):
    """Hyperparameters and paths for Stage 1 Tinker LoRA SFT."""

    base_model: str = "thinkingmachines/Inkling-2-Mini"
    renderer_name: str | None = None
    train_jsonl: Path = Path("data/processed/training/sft_train_chat.jsonl")
    val_jsonl: Path = Path("data/processed/training/sft_val_chat.jsonl")
    out_dir: Path = Path("data/processed/training/runs/sft_v1")
    save_name: str = "ai-dj-sft-v1"
    lora_rank: int = Field(default=32, ge=4, le=128)
    learning_rate: float = Field(default=2e-4, gt=0.0)
    min_lr_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    warmup_ratio: float = Field(default=0.05, ge=0.0, le=0.5)
    batch_size: int = Field(default=32, ge=1)
    epochs: int = Field(default=2, ge=1)
    max_length: int = Field(default=2048, ge=256)
    eval_every_steps: int = Field(default=50, ge=1)
    max_val_examples: int = Field(default=256, ge=1)
    max_steps: int | None = None
    seed: int = 42


def cosine_lr(
    step: int, total_steps: int, base_lr: float, warmup_ratio: float, min_ratio: float
) -> float:
    """Linear warmup followed by cosine decay to `base_lr * min_ratio`."""
    if total_steps <= 1:
        return base_lr
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    if step < warmup_steps:
        return base_lr * ((step + 1) / warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * cosine)


def load_chat_jsonl(path: Path) -> list[dict]:
    """Load chat-formatted SFT rows from a JSONL file."""
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_tokenizer_and_renderer(
    base_model: str,
    renderer_name: str | None = None,
    training_client=None,
):
    """Load the model's tokenizer and chat renderer (supports Inkling via tinker_cookbook)."""
    try:
        from tinker_cookbook import model_info
        from tinker_cookbook.renderers import get_renderer
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        tokenizer = get_tokenizer(base_model)
        rec_name = renderer_name or model_info.get_recommended_renderer_name(base_model)
        renderer = get_renderer(rec_name, tokenizer) if rec_name else None
        return tokenizer, renderer
    except ImportError:
        if "inkling" in base_model.lower():
            raise RuntimeError(
                f"Inkling model '{base_model}' requires `tinker-cookbook` and `tml-renderers`. "
                "Install them with `uv sync --extra train`."
            ) from None
        if training_client is not None:
            return training_client.get_tokenizer(), None
        raise


def _extract_token_ids(encoded) -> list[int]:
    """Extract a flat list of int token IDs from a BatchEncoding, dict, or sequence."""
    if hasattr(encoded, "get") and "input_ids" in encoded:
        encoded = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    ids = list(encoded)
    if ids and isinstance(ids[0], (list, tuple)):
        ids = list(ids[0])
    return [int(x) for x in ids]


def _build_sft_datum(messages: list[dict[str, str]], tokenizer, renderer, max_length: int):
    """Convert a chat conversation to a Tinker Datum with assistant-only loss weights."""
    import tinker

    if renderer is not None:
        from tinker_cookbook.renderers import TrainOnWhat
        from tinker_cookbook.supervised.data import conversation_to_datum

        return conversation_to_datum(
            messages,
            renderer,
            max_length=max_length,
            train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        )

    # Fallback when tinker_cookbook renderer is not explicitly configured:
    # use HuggingFace chat template on the tokenizer and mask prompt tokens to weight 0.
    prompt_ids = _extract_token_ids(
        tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True
        )
    )
    full_ids = _extract_token_ids(
        tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
    )[:max_length]
    if len(full_ids) < 2:
        raise ValueError("Tokenized conversation must contain at least 2 tokens.")

    input_tokens = full_ids[:-1]
    target_tokens = full_ids[1:]
    # Mask all prompt targets to 0.0; if the prompt itself fills/exceeds max_length,
    # zero_count == len(target_tokens) so prompt text never receives loss weight.
    zero_count = min(len(target_tokens), max(0, len(prompt_ids) - 1))
    weights = [0.0] * zero_count + [1.0] * (len(target_tokens) - zero_count)
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": target_tokens,
            "weights": weights,
        },
    )


def _extract_loss_value(fwdbwd_result) -> float:
    metrics = getattr(fwdbwd_result, "metrics", None)
    if isinstance(metrics, dict):
        for key in ("loss:sum", "loss", "cross_entropy"):
            if key in metrics:
                return float(metrics[key])
    return float("nan")


def run_tinker_sft(config: SFTConfig, *, dry_run: bool = False) -> dict:
    """Run Stage 1 LoRA SFT on Tinker (or validate full data/schedule locally in dry_run mode)."""
    train_rows = load_chat_jsonl(config.train_jsonl)
    val_rows = (
        load_chat_jsonl(config.val_jsonl)[: config.max_val_examples]
        if config.val_jsonl.exists()
        else []
    )
    if not train_rows:
        raise ValueError(f"No training examples found in {config.train_jsonl}")

    steps_per_epoch = math.ceil(len(train_rows) / config.batch_size)
    total_steps = steps_per_epoch * config.epochs
    if config.max_steps is not None:
        total_steps = min(total_steps, config.max_steps)

    approx_tokens_1ep = sum(
        sum(approx_token_count(m["content"]) + 4 for m in r["messages"]) for r in train_rows
    )
    approx_total_tokens = int(
        round(approx_tokens_1ep * (total_steps / max(1, steps_per_epoch)))
    )
    price_per_1m = TINKER_TRAIN_PRICE_PER_1M.get(config.base_model, 1.50)
    est_cost_usd = round((approx_total_tokens / 1_000_000.0) * price_per_1m, 2)

    config.out_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        schedule_preview = [
            {
                "step": s,
                "lr": round(
                    cosine_lr(
                        s,
                        total_steps,
                        config.learning_rate,
                        config.warmup_ratio,
                        config.min_lr_ratio,
                    ),
                    8,
                ),
            }
            for s in sorted({0, max(0, total_steps // 2), max(0, total_steps - 1)})
        ]
        summary = {
            "mode": "dry_run",
            "config": config.model_dump(mode="json"),
            "train_examples": len(train_rows),
            "val_examples": len(val_rows),
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "approx_total_tokens": approx_total_tokens,
            "estimated_tinker_cost_usd": est_cost_usd,
            "lr_schedule_preview": schedule_preview,
        }
        (config.out_dir / "sft_dry_run.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary

    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError(
            "TINKER_API_KEY is not set. Export TINKER_API_KEY before launching live Tinker SFT "
            "(or pass --dry-run to validate locally)."
        )

    try:
        import tinker
    except ImportError as exc:
        raise RuntimeError(
            "The `tinker` SDK is not installed. "
            "Install it with `uv pip install tinker tinker-cookbook`."
        ) from exc

    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=config.base_model,
        rank=config.lora_rank,
        seed=config.seed,
    ).result()
    tokenizer, renderer = load_tokenizer_and_renderer(
        config.base_model,
        renderer_name=config.renderer_name,
        training_client=training_client,
    )

    val_datums = [
        _build_sft_datum(r["messages"], tokenizer, renderer, config.max_length)
        for r in val_rows
    ]

    rng = random.Random(config.seed)
    history: list[dict] = []
    global_step = 0
    t0 = time.time()

    for epoch in range(config.epochs):
        indices = list(range(len(train_rows)))
        rng.shuffle(indices)
        for start in range(0, len(indices), config.batch_size):
            if global_step >= total_steps:
                break
            batch_idx = indices[start : start + config.batch_size]
            batch_datums = [
                _build_sft_datum(
                    train_rows[i]["messages"], tokenizer, renderer, config.max_length
                )
                for i in batch_idx
            ]
            lr = cosine_lr(
                global_step,
                total_steps,
                config.learning_rate,
                config.warmup_ratio,
                config.min_lr_ratio,
            )
            # Pipeline forward_backward + optim_step per Tinker best practices
            fwdbwd_fut = training_client.forward_backward(batch_datums, "cross_entropy")
            optim_fut = training_client.optim_step(tinker.AdamParams(learning_rate=lr))
            fwdbwd_res = fwdbwd_fut.result()
            optim_fut.result()

            train_loss = _extract_loss_value(fwdbwd_res)
            record: dict = {
                "step": global_step,
                "epoch": epoch + 1,
                "lr": lr,
                "train_loss": train_loss,
                "elapsed_sec": round(time.time() - t0, 1),
            }

            should_eval = (
                (global_step + 1) % config.eval_every_steps == 0
                or global_step + 1 == total_steps
            )
            if val_datums and should_eval:
                val_res = training_client.forward(val_datums, "cross_entropy").result()
                record["val_loss"] = _extract_loss_value(val_res)

            history.append(record)
            if (global_step + 1) % 10 == 0 or should_eval or global_step == 0:
                val_str = f" val_loss={record['val_loss']:.4f}" if "val_loss" in record else ""
                loss_str = f"{train_loss:.4f}" if train_loss is not None else "n/a"
                print(
                    f"[SFT step {global_step + 1}/{total_steps}] "
                    f"lr={lr:.2e} train_loss={loss_str}{val_str} "
                    f"elapsed={record['elapsed_sec']}s",
                    flush=True,
                )
            global_step += 1

    # Save weights for both downstream sampling/eval and Stage 2 DPO initialization
    sampler_path = training_client.save_weights_for_sampler(name=config.save_name).result().path
    state_path = training_client.save_state(name=f"{config.save_name}-state").result().path

    summary = {
        "mode": "live",
        "config": config.model_dump(mode="json"),
        "steps_completed": global_step,
        "sampler_checkpoint": sampler_path,
        "state_checkpoint": state_path,
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "history": history,
    }
    (config.out_dir / "sft_run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
