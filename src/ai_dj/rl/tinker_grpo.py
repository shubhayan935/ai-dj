"""Phase 4: Tinker LoRA GRPO (importance_sampling) trainer with real DSP-verified audio rewards."""

from __future__ import annotations

import inspect
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from pydantic import BaseModel, Field

from ai_dj.audio.analyzer import AudioTrackCard, load_audio_mono
from ai_dj.audio.renderer import RenderSpec, render_transition
from ai_dj.audio.verifier import verify_transition
from ai_dj.rl.grpo_env import (
    GRPO_SYSTEM_PROMPT,
    AudioTransitionTask,
    GRPORewardBreakdown,
    build_audio_grpo_tasks,
    format_audio_grpo_prompt,
    load_electronic_track_cards,
    parse_creative_plan,
    score_creative_plan,
)
from ai_dj.training.tinker_sft import _extract_token_ids, load_tokenizer_and_renderer


class GRPOConfig(BaseModel):
    """Hyperparameters for Phase 4 audio-grounded Tinker LoRA GRPO."""

    cards_jsonl: Path = Path("data/processed/electronic_track_cards.jsonl")
    out_dir: Path = Path("data/processed/training_v2/runs/inkling2_mini_grpo_v1")
    base_model: str = "thinkingmachines/Inkling-2-Mini"
    sft_state_checkpoint: str | None = None
    renderer_name: str | None = None
    lora_rank: int = 32
    num_steps: int = 20
    prompts_per_step: int = 4
    group_size: int = 4
    crate_size: int = 8
    learning_rate: float = 2e-5
    temperature: float = 0.85
    max_tokens: int = 160
    render_sr: int = 22050
    render_workers: int = 6
    seed: int = 42
    save_name: str = "ai-dj-inkling2-mini-grpo-v1"
    min_local_stability: float = Field(default=0.70, ge=0.0, le=1.0)


def compute_group_advantages(rewards: list[float]) -> list[float]:
    """Compute zero-mean, unit-std GRPO advantages for one prompt group."""
    if not rewards:
        return []
    arr = np.asarray(rewards, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std < 1e-6:
        return [0.0] * len(rewards)
    return [float((r - mean) / (std + 1e-6)) for r in rewards]


def _decode_tokens(tokens: list[int], tokenizer, renderer) -> str:
    if renderer is not None and hasattr(renderer, "parse_response"):
        try:
            msg, _ = renderer.parse_response(tokens)
            content = msg.get("content", "")
            if isinstance(content, list):
                return "".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            return str(content)
        except Exception:
            pass
    return tokenizer.decode(tokens, skip_special_tokens=True)


def _build_prompt_input(messages: list[dict[str, str]], tokenizer, renderer, tinker_mod):
    if renderer is not None:
        if "effort" in inspect.signature(renderer.build_generation_prompt).parameters:
            return renderer.build_generation_prompt(messages, effort=0.0)
        return renderer.build_generation_prompt(messages)
    prompt_ids = _extract_token_ids(
        tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    )
    return tinker_mod.ModelInput.from_ints(prompt_ids)


def run_tinker_grpo(config: GRPOConfig, *, dry_run: bool = False) -> dict:
    """Run DSP-verified GRPO on Tinker using `importance_sampling` loss."""
    config.out_dir.mkdir(parents=True, exist_ok=True)
    cards = load_electronic_track_cards(
        config.cards_jsonl,
        min_local_stability=config.min_local_stability,
    )
    tasks = build_audio_grpo_tasks(
        cards,
        crate_size=config.crate_size,
        num_tasks=max(config.num_steps * config.prompts_per_step, 32),
        seed=config.seed,
    )

    if dry_run:
        sample_task = tasks[0]
        sample_prompt = format_audio_grpo_prompt(sample_task)
        best_cand = sample_task.candidates[0]
        mock_plan = json.dumps(
            {
                "next_track_id": best_cand.track_id,
                "transition_style": "bass_swap",
                "overlap_beats": 32,
                "cue_out_beat_prev": sample_task.deck_a.recommended_outro_beat or 64,
                "cue_in_beat_next": best_cand.recommended_intro_beat or 16,
                "bass_swap_beat_frac": 0.5,
            }
        )
        sample_rew = score_creative_plan(sample_task, mock_plan, render_audio=False)
        summary = {
            "mode": "dry_run",
            "club_stable_tracks": len(cards),
            "tasks_built": len(tasks),
            "num_steps": config.num_steps,
            "rollouts_per_step": config.prompts_per_step * config.group_size,
            "total_rollouts": config.num_steps * config.prompts_per_step * config.group_size,
            "sample_prompt_chars": len(sample_prompt),
            "sample_mock_reward": sample_rew.total_reward,
        }
        (config.out_dir / "grpo_dry_run.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary

    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is not set; cannot launch live Tinker GRPO.")

    try:
        import tinker
    except ImportError as exc:
        raise RuntimeError("Install `tinker` (`uv sync --extra train`) to run GRPO.") from exc

    service_client = tinker.ServiceClient()
    if config.sft_state_checkpoint:
        training_client = service_client.create_training_client_from_state(
            path=config.sft_state_checkpoint
        ).result()
    else:
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
    stop_seqs = renderer.get_stop_sequences() if renderer is not None else None
    sampling_params = tinker.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        stop=stop_seqs,
    )

    # Pre-decode mono audio for all club-stable cards once so GRPO inner-loop renders are fast
    audio_cache: dict[str, np.ndarray] = {
        c.track_id: load_audio_mono(Path(c.audio_path), sr=config.render_sr) for c in cards
    }

    rng = random.Random(config.seed)
    history: list[dict] = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=config.render_workers) as pool:
        for step in range(config.num_steps):
            step_tasks = rng.sample(tasks, min(config.prompts_per_step, len(tasks)))
            sc = training_client.save_weights_and_get_sampling_client()
            sampling_client = sc.result() if hasattr(sc, "result") else sc

            # 1. Sample G rollouts per prompt concurrently
            sample_jobs: list[tuple[AudioTransitionTask, list[int], object]] = []
            for task in step_tasks:
                messages = [
                    {"role": "system", "content": GRPO_SYSTEM_PROMPT},
                    {"role": "user", "content": format_audio_grpo_prompt(task)},
                ]
                model_input = _build_prompt_input(messages, tokenizer, renderer, tinker)
                prompt_tokens = list(model_input.to_ints())
                fut = sampling_client.sample(
                    prompt=model_input,
                    num_samples=config.group_size,
                    sampling_params=sampling_params,
                )
                sample_jobs.append((task, prompt_tokens, fut))

            # 2. Decode rollouts and render/verify audio rewards in parallel
            eval_jobs: list[tuple[int, list[int], list[int], list[float], object]] = []
            for group_idx, (task, prompt_tokens, fut) in enumerate(sample_jobs):
                res = fut.result()
                for seq in res.sequences:
                    comp_tokens = list(seq.tokens)
                    comp_logprobs = [float(lp) for lp in (seq.logprobs or [0.0] * len(comp_tokens))]
                    raw_text = _decode_tokens(comp_tokens, tokenizer, renderer)
                    rew_fut = pool.submit(
                        score_creative_plan,
                        task,
                        raw_text,
                        render_audio=True,
                        render_sr=config.render_sr,
                        audio_cache=audio_cache,
                    )
                    eval_jobs.append(
                        (group_idx, prompt_tokens, comp_tokens, comp_logprobs, rew_fut)
                    )

            # Collect rewards per group
            group_records: dict[
                int, list[tuple[list[int], list[int], list[float], GRPORewardBreakdown]]
            ] = {}
            all_rewards: list[float] = []
            all_qa: list[float] = []
            valid_count = 0
            style_counts: dict[str, int] = {}

            for group_idx, p_toks, c_toks, c_lps, rew_fut in eval_jobs:
                bd: GRPORewardBreakdown = rew_fut.result()
                group_records.setdefault(group_idx, []).append((p_toks, c_toks, c_lps, bd))
                all_rewards.append(bd.total_reward)
                all_qa.append(bd.dsp_qa_reward * 2.0)  # normalized 0..1 QA score
                if bd.schema_valid and bd.plan is not None:
                    valid_count += 1
                    st = bd.plan.transition_style
                    style_counts[st] = style_counts.get(st, 0) + 1

            # 3. Build importance_sampling datums with group-relative advantages
            datums = []
            for entries in group_records.values():
                rewards = [e[3].total_reward for e in entries]
                advs = compute_group_advantages(rewards)
                for (p_toks, c_toks, c_lps, _), adv in zip(entries, advs, strict=True):
                    if abs(adv) < 1e-6 or not c_toks:
                        continue
                    full_tokens = p_toks + c_toks
                    n_prompt_targets = max(0, len(p_toks) - 1)
                    n_comp = len(c_toks)
                    lp_vec = [0.0] * n_prompt_targets + c_lps[:n_comp]
                    adv_vec = [0.0] * n_prompt_targets + [adv] * n_comp
                    if len(lp_vec) != len(full_tokens) - 1:
                        continue
                    datums.append(
                        tinker.Datum(
                            model_input=tinker.ModelInput.from_ints(full_tokens[:-1]),
                            loss_fn_inputs={
                                "target_tokens": tinker.TensorData.from_numpy(
                                    np.asarray(full_tokens[1:], dtype=np.int64)
                                ),
                                "logprobs": tinker.TensorData.from_numpy(
                                    np.asarray(lp_vec, dtype=np.float32)
                                ),
                                "advantages": tinker.TensorData.from_numpy(
                                    np.asarray(adv_vec, dtype=np.float32)
                                ),
                            },
                        )
                    )

            updated = False
            if datums:
                fwdbwd_fut = training_client.forward_backward(datums, "importance_sampling")
                optim_fut = training_client.optim_step(
                    tinker.AdamParams(learning_rate=config.learning_rate)
                )
                fwdbwd_fut.result()
                optim_fut.result()
                updated = True

            rec = {
                "step": step + 1,
                "mean_reward": round(float(np.mean(all_rewards)), 4) if all_rewards else 0.0,
                "max_reward": round(float(np.max(all_rewards)), 4) if all_rewards else 0.0,
                "mean_dsp_qa": round(float(np.mean(all_qa)), 4) if all_qa else 0.0,
                "schema_valid_rate": round(valid_count / max(1, len(all_rewards)), 4),
                "datums_used": len(datums),
                "updated": updated,
                "styles": style_counts,
                "elapsed_sec": round(time.time() - t0, 1),
            }
            history.append(rec)
            print(
                f"[GRPO step {step + 1}/{config.num_steps}] "
                f"mean_rew={rec['mean_reward']:.3f} max_rew={rec['max_reward']:.3f} "
                f"qa={rec['mean_dsp_qa']:.3f} valid={rec['schema_valid_rate'] * 100:.0f}% "
                f"styles={style_counts} elapsed={rec['elapsed_sec']}s",
                flush=True,
            )

    sampler_path = training_client.save_weights_for_sampler(name=config.save_name).result().path
    state_path = training_client.save_state(name=f"{config.save_name}-state").result().path

    summary = {
        "mode": "live",
        "config": config.model_dump(mode="json"),
        "steps_completed": len(history),
        "sampler_checkpoint": sampler_path,
        "state_checkpoint": state_path,
        "initial_mean_reward": history[0]["mean_reward"] if history else None,
        "final_mean_reward": history[-1]["mean_reward"] if history else None,
        "history": history,
    }
    (config.out_dir / "grpo_run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def plan_and_render_electronic_set(
    cards_jsonl: Path,
    sampler_checkpoint: str,
    base_model: str,
    out_dir: Path,
    *,
    num_transitions: int = 3,
    temperature: float = 0.2,
    sr: int = 44100,
) -> dict:
    """Plan and render a 44.1kHz stereo electronic mini-set from a Tinker checkpoint."""
    import tinker

    out_dir.mkdir(parents=True, exist_ok=True)
    cards = load_electronic_track_cards(cards_jsonl)
    if len(cards) < num_transitions + 1:
        raise ValueError(f"Need at least {num_transitions + 1} stable electronic tracks.")

    service_client = tinker.ServiceClient()
    sc = service_client.create_sampling_client(model_path=sampler_checkpoint)
    sampling_client = sc.result() if hasattr(sc, "result") else sc
    tokenizer, renderer = load_tokenizer_and_renderer(base_model)
    stop_seqs = renderer.get_stop_sequences() if renderer is not None else None
    sampling_params = tinker.SamplingParams(
        max_tokens=160,
        temperature=temperature,
        stop=stop_seqs,
    )

    # Start from the track with the most club-compatible neighbors (closest BPM median)
    med_bpm = float(np.median([c.bpm for c in cards]))
    deck_a: AudioTrackCard = min(cards, key=lambda c: abs(c.bpm - med_bpm))
    used_ids = {deck_a.track_id}
    phases = ("opener", "build", "peak", "closer")

    transitions_log: list[dict] = []
    stitched_segments: list[np.ndarray] = []

    for idx in range(num_transitions):
        cands = [c for c in cards if c.track_id not in used_ids][:8]
        task = AudioTransitionTask(
            task_id=f"set-T{idx + 1}",
            deck_a=deck_a,
            candidates=cands,
            set_phase=phases[min(idx, len(phases) - 1)],
            target_genres=tuple(deck_a.genres[:3]) if deck_a.genres else ("electronic", "house"),
        )
        messages = [
            {"role": "system", "content": GRPO_SYSTEM_PROMPT},
            {"role": "user", "content": format_audio_grpo_prompt(task)},
        ]
        model_input = _build_prompt_input(messages, tokenizer, renderer, tinker)
        res = sampling_client.sample(
            prompt=model_input, num_samples=1, sampling_params=sampling_params
        ).result()
        raw_text = _decode_tokens(list(res.sequences[0].tokens), tokenizer, renderer)

        cand_by_id = {c.track_id: c for c in cands}
        plan, err = parse_creative_plan(raw_text, set(cand_by_id))
        if plan is None:
            raise RuntimeError(f"Model produced invalid plan on T{idx + 1}: {err} | {raw_text}")

        deck_b = cand_by_id[plan.next_track_id]
        used_ids.add(deck_b.track_id)

        a_max_cue = max(16, len(deck_a.beat_times) - plan.overlap_beats - 4)
        b_max_cue = max(0, len(deck_b.beat_times) - plan.overlap_beats - 4)
        spec = RenderSpec(
            cue_out_beat_a=min(max(16, plan.cue_out_beat_prev), a_max_cue),
            cue_in_beat_b=min(max(0, plan.cue_in_beat_next), b_max_cue),
            overlap_beats=plan.overlap_beats,
            transition_style=plan.transition_style,
            bass_swap_beat_frac=plan.bass_swap_beat_frac,
            context_beats=16,
        )
        wav_path = out_dir / f"T{idx + 1}_{deck_a.track_id}_to_{deck_b.track_id}.wav"
        render_res = render_transition(deck_a, deck_b, spec, out_wav=wav_path, sr=sr, stereo=True)
        qa = verify_transition(render_res, deck_a, deck_b, spec)

        audio_seg, _ = sf.read(str(wav_path), dtype="float32")
        stitched_segments.append(audio_seg)
        transitions_log.append(
            {
                "transition": f"T{idx + 1}",
                "track_a": deck_a.track_id,
                "track_b": deck_b.track_id,
                "genres_a": deck_a.genres,
                "genres_b": deck_b.genres,
                "bpm_a": deck_a.bpm,
                "bpm_b": deck_b.bpm,
                "key_a": deck_a.camelot,
                "key_b": deck_b.camelot,
                "style": plan.transition_style,
                "overlap_beats": plan.overlap_beats,
                "cue_out_beat_a": spec.cue_out_beat_a,
                "cue_in_beat_b": spec.cue_in_beat_b,
                "qa_reward": qa.verifiable_reward,
                "qa_flags": list(qa.defects_detected),
                "beat_drift_ms": qa.beat_drift_ms,
                "lufs_jump_db": qa.lufs_jump_db,
                "wav_path": str(wav_path),
            }
        )
        deck_a = deck_b

    # Stitch all transitions with 0.4s soft silence gap for clean listening showcase
    gap = np.zeros((int(0.4 * sr), 2), dtype=np.float32)
    full_audio = np.concatenate(
        [seg for pair in ((s, gap) for s in stitched_segments) for seg in pair][:-1],
        axis=0,
    )
    showcase_path = out_dir / "electronic_club_miniset_showcase.wav"
    sf.write(str(showcase_path), full_audio, sr, subtype="PCM_16")

    report = {
        "sampler_checkpoint": sampler_checkpoint,
        "showcase_wav": str(showcase_path),
        "showcase_duration_sec": round(len(full_audio) / sr, 1),
        "transitions": transitions_log,
    }
    (out_dir / "miniset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
