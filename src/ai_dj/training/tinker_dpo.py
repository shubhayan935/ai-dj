"""Tinker LoRA Direct Preference Optimization (DPO) trainer for AI DJ taste alignment."""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

from pydantic import BaseModel, Field

from ai_dj.training.dataset import TINKER_TRAIN_PRICE_PER_1M, approx_token_count
from ai_dj.training.tinker_sft import _build_sft_datum, cosine_lr, load_chat_jsonl


class DPOConfig(BaseModel):
    """Hyperparameters and paths for Stage 2 Tinker LoRA DPO."""

    base_model: str = "thinkingmachines/Inkling-2-Mini"
    sft_state_checkpoint: str | None = None
    renderer_name: str | None = None
    dpo_jsonl: Path = Path("data/processed/training/dpo_train_chat.jsonl")
    out_dir: Path = Path("data/processed/training/runs/dpo_v1")
    save_name: str = "ai-dj-dpo-v1"
    lora_rank: int = Field(default=32, ge=4, le=128)
    dpo_beta: float = Field(default=0.1, gt=0.0, le=2.0)
    learning_rate: float = Field(default=1e-5, gt=0.0)
    min_lr_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    warmup_ratio: float = Field(default=0.05, ge=0.0, le=0.5)
    batch_size: int = Field(default=16, ge=1)
    epochs: int = Field(default=1, ge=1)
    max_length: int = Field(default=2048, ge=256)
    max_steps: int | None = None
    seed: int = 42


def compute_dpo_metrics_np(
    chosen_logps: list[float],
    rejected_logps: list[float],
    ref_chosen_logps: list[float],
    ref_rejected_logps: list[float],
    beta: float = 0.1,
) -> dict[str, float]:
    """Compute scalar DPO loss, preference accuracy, and reward margin (numpy/math only)."""
    if not chosen_logps:
        return {"dpo_loss": 0.0, "accuracy": 0.0, "margin": 0.0}

    losses: list[float] = []
    margins: list[float] = []
    correct = 0
    for pi_c, pi_r, ref_c, ref_r in zip(
        chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, strict=True
    ):
        chosen_reward = beta * (pi_c - ref_c)
        rejected_reward = beta * (pi_r - ref_r)
        margin = chosen_reward - rejected_reward
        margins.append(margin)
        if margin > 0:
            correct += 1
        # Numerically stable -log(sigmoid(margin)) = log(1 + exp(-margin))
        if margin >= 0:
            loss = math.log1p(math.exp(-margin))
        else:
            loss = -margin + math.log1p(math.exp(margin))
        losses.append(loss)

    n = len(losses)
    return {
        "dpo_loss": round(sum(losses) / n, 4),
        "accuracy": round(correct / n, 4),
        "margin": round(sum(margins) / n, 4),
    }


def _dpo_pair_tokens(row: dict) -> int:
    p_tok = sum(approx_token_count(m["content"]) + 4 for m in row["prompt"])
    c_tok = sum(approx_token_count(m["content"]) + 4 for m in row["chosen"])
    r_tok = sum(approx_token_count(m["content"]) + 4 for m in row["rejected"])
    return (2 * p_tok) + c_tok + r_tok


def run_tinker_dpo(config: DPOConfig, *, dry_run: bool = False) -> dict:
    """Run Stage 2 LoRA DPO on Tinker (or validate full data/schedule locally in dry_run mode)."""
    pairs = load_chat_jsonl(config.dpo_jsonl)
    if not pairs:
        raise ValueError(f"No DPO pairs found in {config.dpo_jsonl}")

    steps_per_epoch = math.ceil(len(pairs) / config.batch_size)
    total_steps = steps_per_epoch * config.epochs
    if config.max_steps is not None:
        total_steps = min(total_steps, config.max_steps)

    approx_tokens_1ep = sum(_dpo_pair_tokens(r) for r in pairs)
    approx_total_tokens = int(
        round(approx_tokens_1ep * (total_steps / max(1, steps_per_epoch)))
    )
    price_per_1m = TINKER_TRAIN_PRICE_PER_1M.get(config.base_model, 1.50)
    est_cost_usd = round((approx_total_tokens / 1_000_000.0) * price_per_1m, 2)

    config.out_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        neg_breakdown: dict[str, int] = {}
        for row in pairs:
            nt = str(row.get("negative_type", "unknown"))
            neg_breakdown[nt] = neg_breakdown.get(nt, 0) + 1
        summary = {
            "mode": "dry_run",
            "config": config.model_dump(mode="json"),
            "dpo_pairs": len(pairs),
            "negative_type_counts": neg_breakdown,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "approx_total_tokens": approx_total_tokens,
            "estimated_tinker_cost_usd": est_cost_usd,
        }
        (config.out_dir / "dpo_dry_run.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary

    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError(
            "TINKER_API_KEY is not set. Export TINKER_API_KEY before launching live Tinker DPO "
            "(or pass --dry-run to validate locally)."
        )

    try:
        import tinker
        import torch
        from tinker_cookbook.preference.train_dpo import compute_dpo_loss
    except ImportError as exc:
        raise RuntimeError(
            "Tinker DPO requires `tinker`, `tinker-cookbook`, and `torch`. "
            "Install them with `uv pip install tinker tinker-cookbook torch`."
        ) from exc

    service_client = tinker.ServiceClient()
    if config.sft_state_checkpoint:
        training_client = service_client.create_training_client_from_state(
            config.sft_state_checkpoint
        ).result()
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.base_model,
            rank=config.lora_rank,
            seed=config.seed,
        ).result()

    # Frozen reference client anchored at initial (Stage 1 SFT) weights
    ref_sampler = training_client.save_weights_and_get_sampling_client(
        name=f"{config.save_name}-ref"
    ).result()
    tokenizer = training_client.get_tokenizer()

    renderer = None
    if config.renderer_name:
        from tinker_cookbook.renderers import get_renderer

        renderer = get_renderer(config.renderer_name, tokenizer)

    rng = random.Random(config.seed)
    history: list[dict] = []
    global_step = 0
    t0 = time.time()

    for epoch in range(config.epochs):
        indices = list(range(len(pairs)))
        rng.shuffle(indices)
        for start in range(0, len(indices), config.batch_size):
            if global_step >= total_steps:
                break
            batch_idx = indices[start : start + config.batch_size]
            interleaved_datums = []
            for i in batch_idx:
                row = pairs[i]
                chosen_conv = [*row["prompt"], *row["chosen"]]
                rejected_conv = [*row["prompt"], *row["rejected"]]
                interleaved_datums.append(
                    _build_sft_datum(chosen_conv, tokenizer, renderer, config.max_length)
                )
                interleaved_datums.append(
                    _build_sft_datum(rejected_conv, tokenizer, renderer, config.max_length)
                )

            # Reference logprobs (frozen Stage 1 policy)
            ref_logp_futs = [
                ref_sampler.compute_logprobs(d.model_input) for d in interleaved_datums
            ]
            ref_logps_raw = [f.result() for f in ref_logp_futs]
            ref_chosen = []
            ref_rejected = []
            for j in range(0, len(interleaved_datums), 2):
                w_c = torch.tensor(
                    interleaved_datums[j].loss_fn_inputs["weights"], dtype=torch.float32
                )
                w_r = torch.tensor(
                    interleaved_datums[j + 1].loss_fn_inputs["weights"], dtype=torch.float32
                )
                lp_c = torch.tensor(
                    [x or 0.0 for x in ref_logps_raw[j][1:]], dtype=torch.float32
                )
                lp_r = torch.tensor(
                    [x or 0.0 for x in ref_logps_raw[j + 1][1:]], dtype=torch.float32
                )
                ref_chosen.append((lp_c * w_c).sum())
                ref_rejected.append((lp_r * w_r).sum())

            def _dpo_loss_fn(data, logprobs_list, _ref_c=ref_chosen, _ref_r=ref_rejected):
                chosen_lp = []
                rejected_lp = []
                for j in range(0, len(data), 2):
                    w_c = torch.tensor(data[j].loss_fn_inputs["weights"], dtype=torch.float32)
                    w_r = torch.tensor(data[j + 1].loss_fn_inputs["weights"], dtype=torch.float32)
                    chosen_lp.append((logprobs_list[j] * w_c).sum())
                    rejected_lp.append((logprobs_list[j + 1] * w_r).sum())
                return compute_dpo_loss(
                    chosen_lp, rejected_lp, _ref_c, _ref_r, dpo_beta=config.dpo_beta
                )

            lr = cosine_lr(
                global_step,
                total_steps,
                config.learning_rate,
                config.warmup_ratio,
                config.min_lr_ratio,
            )
            fwdbwd_fut = training_client.forward_backward_custom(interleaved_datums, _dpo_loss_fn)
            optim_fut = training_client.optim_step(tinker.AdamParams(learning_rate=lr))
            fwdbwd_res = fwdbwd_fut.result()
            optim_fut.result()

            metrics = getattr(fwdbwd_res, "metrics", {}) or {}
            history.append(
                {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "lr": lr,
                    "metrics": {k: float(v) for k, v in metrics.items()},
                    "elapsed_sec": round(time.time() - t0, 1),
                }
            )
            global_step += 1

    sampler_path = training_client.save_weights_for_sampler(name=config.save_name).result().path
    summary = {
        "mode": "live",
        "config": config.model_dump(mode="json"),
        "steps_completed": global_step,
        "sampler_checkpoint": sampler_path,
        "history": history,
    }
    (config.out_dir / "dpo_run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
