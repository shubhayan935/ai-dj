"""Phase 4: Tinker LoRA GRPO (importance_sampling) trainer with real DSP-verified audio rewards."""

from __future__ import annotations

import inspect
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
from pydantic import BaseModel, Field

from ai_dj.audio.analyzer import AudioTrackCard, analyze_track, load_audio_mono
from ai_dj.audio.renderer import RenderSpec, render_transition
from ai_dj.audio.verifier import verify_transition
from ai_dj.rl.audio_native import build_audio_native_messages
from ai_dj.rl.grpo_env import (
    GRPO_SYSTEM_PROMPT,
    AudioTransitionTask,
    GRPORewardBreakdown,
    build_audio_grpo_tasks,
    build_pair_mix_tasks,
    format_audio_grpo_prompt,
    load_electronic_track_cards,
    parse_creative_plan,
    score_creative_plan,
)
from ai_dj.training.tinker_sft import _extract_token_ids, load_tokenizer_and_renderer


class GRPOConfig(BaseModel):
    """Hyperparameters for Phase 4/5 audio-grounded Tinker LoRA GRPO."""

    cards_jsonl: Path = Path("data/processed/electronic_track_cards.jsonl")
    out_dir: Path = Path("data/processed/training_v2/runs/inkling2_mini_grpo_v1")
    base_model: str = "thinkingmachines/Inkling-2-Mini"
    sft_state_checkpoint: str | None = None
    renderer_name: str | None = None
    lora_rank: int = 32
    num_steps: int = 20
    prompts_per_step: int = 4
    group_size: int = Field(default=4, ge=2)
    crate_size: int = 8
    task_mode: Literal["crate", "pair", "mixed"] = "crate"
    audio_native: bool = False
    clip_sec: float = Field(default=4.0, gt=0.5, le=12.0)
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
    total_tasks_needed = max(config.num_steps * config.prompts_per_step, 32)
    if config.task_mode == "pair":
        tasks = build_pair_mix_tasks(cards, num_tasks=total_tasks_needed, seed=config.seed)
    elif config.task_mode == "mixed":
        n_pair = total_tasks_needed // 2
        n_crate = total_tasks_needed - n_pair
        tasks = build_pair_mix_tasks(cards, num_tasks=n_pair, seed=config.seed) + (
            build_audio_grpo_tasks(
                cards, crate_size=config.crate_size, num_tasks=n_crate, seed=config.seed + 1
            )
        )
    else:
        tasks = build_audio_grpo_tasks(
            cards,
            crate_size=config.crate_size,
            num_tasks=total_tasks_needed,
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
    audio_cache_16k: dict[str, np.ndarray] | None = (
        {c.track_id: load_audio_mono(Path(c.audio_path), sr=16000) for c in cards}
        if config.audio_native
        else None
    )

    rng = random.Random(config.seed)
    history: list[dict] = []
    traces_path = config.out_dir / "raw_traces.jsonl"
    if traces_path.exists() and traces_path.stat().st_size > 0:
        backup_ts = int(traces_path.stat().st_mtime)
        backup_path = config.out_dir / f"raw_traces.{backup_ts}.jsonl"
        suffix = 1
        while backup_path.exists():
            backup_path = config.out_dir / f"raw_traces.{backup_ts}_{suffix}.jsonl"
            suffix += 1
        traces_path.rename(backup_path)
    traces_path.write_text("", encoding="utf-8")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=config.render_workers) as pool:
        for step in range(config.num_steps):
            step_tasks = rng.sample(tasks, min(config.prompts_per_step, len(tasks)))
            sc = training_client.save_weights_and_get_sampling_client()
            sampling_client = sc.result() if hasattr(sc, "result") else sc

            # 1. Sample G rollouts per prompt concurrently
            sample_jobs: list[tuple[AudioTransitionTask, str, list[dict], object, object]] = []
            for task in step_tasks:
                prompt_text = format_audio_grpo_prompt(task)
                clip_meta: list[dict] = []
                if config.audio_native:
                    messages, clip_meta = build_audio_native_messages(
                        task,
                        clip_sec=config.clip_sec,
                        sr=16000,
                        audio_cache_16k=audio_cache_16k,
                    )
                else:
                    messages = [
                        {"role": "system", "content": GRPO_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt_text},
                    ]
                model_input = _build_prompt_input(messages, tokenizer, renderer, tinker)
                fut = sampling_client.sample(
                    prompt=model_input,
                    num_samples=config.group_size,
                    sampling_params=sampling_params,
                )
                sample_jobs.append((task, prompt_text, clip_meta, model_input, fut))

            # 2. Decode rollouts and render/verify audio rewards in parallel
            eval_jobs: list[
                tuple[
                    int,
                    AudioTransitionTask,
                    str,
                    list[dict],
                    object,
                    list[int],
                    list[float],
                    str,
                    object,
                ]
            ] = []
            for group_idx, (task, prompt_text, clip_meta, model_input, fut) in enumerate(
                sample_jobs
            ):
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
                        (
                            group_idx,
                            task,
                            prompt_text,
                            clip_meta,
                            model_input,
                            comp_tokens,
                            comp_logprobs,
                            raw_text,
                            rew_fut,
                        )
                    )

            # Collect rewards per group
            group_records: dict[int, list[tuple]] = {}
            all_rewards: list[float] = []
            all_qa: list[float] = []
            valid_count = 0
            style_counts: dict[str, int] = {}

            for (
                group_idx,
                task,
                prompt_text,
                clip_meta,
                model_input,
                c_toks,
                c_lps,
                raw_text,
                rew_fut,
            ) in eval_jobs:
                bd: GRPORewardBreakdown = rew_fut.result()
                group_records.setdefault(group_idx, []).append(
                    (task, prompt_text, clip_meta, model_input, c_toks, c_lps, raw_text, bd)
                )
                all_rewards.append(bd.total_reward)
                all_qa.append(bd.dsp_qa_reward * 2.0)  # normalized 0..1 QA score
                if bd.schema_valid and bd.plan is not None:
                    valid_count += 1
                    st = bd.plan.transition_style
                    style_counts[st] = style_counts.get(st, 0) + 1

            # 3. Build importance_sampling datums (supports text + DmelChunk audio inputs)
            datums = []
            step_traces: list[dict] = []
            for entries in group_records.values():
                rewards = [e[7].total_reward for e in entries]
                advs = compute_group_advantages(rewards)
                for r_idx, (
                    (task, prompt_text, clip_meta, prompt_mi, c_toks, c_lps, raw_text, bd),
                    adv,
                ) in enumerate(zip(entries, advs, strict=True)):
                    step_traces.append(
                        {
                            "step": step + 1,
                            "task_id": task.task_id,
                            "rollout_idx": r_idx,
                            "deck_a_id": task.deck_a.track_id,
                            "candidate_ids": [c.track_id for c in task.candidates],
                            "audio_native": config.audio_native,
                            "audio_clips": clip_meta,
                            "prompt_text": prompt_text,
                            "raw_completion": raw_text,
                            "parsed_plan": (
                                {
                                    "next_track_id": bd.plan.next_track_id,
                                    "transition_style": bd.plan.transition_style,
                                    "overlap_beats": bd.plan.overlap_beats,
                                    "cue_out_beat_prev": bd.plan.cue_out_beat_prev,
                                    "cue_in_beat_next": bd.plan.cue_in_beat_next,
                                    "bass_swap_beat_frac": bd.plan.bass_swap_beat_frac,
                                }
                                if bd.plan is not None
                                else None
                            ),
                            "schema_valid": bd.schema_valid,
                            "total_reward": bd.total_reward,
                            "compatibility_reward": bd.compatibility_reward,
                            "craft_reward": bd.craft_reward,
                            "dsp_qa_reward": bd.dsp_qa_reward,
                            "qa_flags": list(bd.qa_flags),
                            "error": bd.error,
                            "advantage": round(float(adv), 4),
                        }
                    )
                    if abs(adv) < 1e-6 or not c_toks:
                        continue
                    prompt_len = sum(int(ch.length) for ch in prompt_mi.chunks)
                    n_prompt_targets = max(0, prompt_len - 1)
                    n_comp = len(c_toks)
                    chunks = list(prompt_mi.chunks)
                    if n_comp > 1:
                        chunks.append(tinker.types.EncodedTextChunk(tokens=c_toks[:-1]))
                    datum_mi = tinker.ModelInput(chunks=chunks)
                    target_tokens = [0] * n_prompt_targets + c_toks
                    lp_vec = [0.0] * n_prompt_targets + c_lps[:n_comp]
                    adv_vec = [0.0] * n_prompt_targets + [adv] * n_comp
                    if len(target_tokens) != sum(int(ch.length) for ch in datum_mi.chunks):
                        continue
                    datums.append(
                        tinker.Datum(
                            model_input=datum_mi,
                            loss_fn_inputs={
                                "target_tokens": tinker.TensorData.from_numpy(
                                    np.asarray(target_tokens, dtype=np.int64)
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

            with traces_path.open("a", encoding="utf-8") as tfh:
                for tr in step_traces:
                    tfh.write(json.dumps(tr) + "\n")

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


def _safe_filename_part(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(raw)).strip("._-")
    return cleaned or "track"


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

        max_overlap = min(len(deck_a.beat_times) - 20, len(deck_b.beat_times) - 4)
        overlap = max(4, min(plan.overlap_beats, max(4, (max_overlap // 4) * 4)))
        a_max_cue = max(0, len(deck_a.beat_times) - overlap - 4)
        a_min_cue = min(16, a_max_cue)
        b_max_cue = max(0, len(deck_b.beat_times) - overlap - 4)
        spec = RenderSpec(
            cue_out_beat_a=min(max(a_min_cue, plan.cue_out_beat_prev), a_max_cue),
            cue_in_beat_b=min(max(0, plan.cue_in_beat_next), b_max_cue),
            overlap_beats=overlap,
            transition_style=plan.transition_style,
            bass_swap_beat_frac=plan.bass_swap_beat_frac,
            context_beats=16,
        )
        safe_a = _safe_filename_part(deck_a.track_id)
        safe_b = _safe_filename_part(deck_b.track_id)
        out_root = out_dir.resolve()
        wav_path = (out_root / f"T{idx + 1}_{safe_a}_to_{safe_b}.wav").resolve()
        if not wav_path.is_relative_to(out_root):
            raise ValueError(f"Unsafe output path for transition WAV: {wav_path}")
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
                "overlap_beats": spec.overlap_beats,
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


def mix_song_pair(
    song_a: AudioTrackCard | Path,
    song_b: AudioTrackCard | Path,
    sampler_checkpoint: str | None,
    base_model: str,
    out_dir: Path,
    *,
    audio_native: bool = False,
    clip_sec: float = 4.0,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    sr: int = 44100,
    label: str = "pair_mix",
) -> dict:
    """Analyze (if needed), plan, and render a 44.1kHz stereo mix between two full songs."""
    import tinker

    out_dir.mkdir(parents=True, exist_ok=True)
    card_a = song_a if isinstance(song_a, AudioTrackCard) else analyze_track(Path(song_a))
    card_b = song_b if isinstance(song_b, AudioTrackCard) else analyze_track(Path(song_b))

    service_client = tinker.ServiceClient()
    if sampler_checkpoint:
        sc = service_client.create_sampling_client(model_path=sampler_checkpoint)
    else:
        sc = service_client.create_sampling_client(base_model=base_model)
    sampling_client = sc.result() if hasattr(sc, "result") else sc
    tokenizer, renderer = load_tokenizer_and_renderer(base_model, training_client=sampling_client)
    stop_seqs = renderer.get_stop_sequences() if renderer is not None else None
    tok_budget = max_tokens if max_tokens is not None else (768 if audio_native else 200)
    sampling_params = tinker.SamplingParams(
        max_tokens=tok_budget, temperature=temperature, stop=stop_seqs
    )

    task = AudioTransitionTask(
        task_id=label,
        deck_a=card_a,
        candidates=[card_b],
        set_phase="peak",
        target_genres=tuple(card_a.genres[:3]) if card_a.genres else ("electronic", "house"),
    )
    prompt_text = format_audio_grpo_prompt(task)
    clip_meta: list[dict] = []
    if audio_native:
        messages, clip_meta = build_audio_native_messages(task, clip_sec=clip_sec, sr=16000)
    else:
        messages = [
            {"role": "system", "content": GRPO_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]

    model_input = _build_prompt_input(messages, tokenizer, renderer, tinker)
    res = sampling_client.sample(
        prompt=model_input, num_samples=1, sampling_params=sampling_params
    ).result()
    raw_text = _decode_tokens(list(res.sequences[0].tokens), tokenizer, renderer)
    bd = score_creative_plan(task, raw_text, render_audio=True, render_sr=22050)
    if bd.plan is None:
        return {
            "label": label,
            "model": base_model,
            "sampler_checkpoint": sampler_checkpoint,
            "audio_native": audio_native,
            "audio_clips": clip_meta,
            "prompt_text": prompt_text,
            "raw_completion": raw_text,
            "schema_valid": False,
            "error": bd.error,
        }

    plan = bd.plan
    max_overlap = min(len(card_a.beat_times) - 20, len(card_b.beat_times) - 4)
    overlap = max(4, min(plan.overlap_beats, max(4, (max_overlap // 4) * 4)))
    a_max_cue = max(0, len(card_a.beat_times) - overlap - 4)
    a_min_cue = min(16, a_max_cue)
    b_max_cue = max(0, len(card_b.beat_times) - overlap - 4)
    spec = RenderSpec(
        cue_out_beat_a=min(max(a_min_cue, plan.cue_out_beat_prev), a_max_cue),
        cue_in_beat_b=min(max(0, plan.cue_in_beat_next), b_max_cue),
        overlap_beats=overlap,
        transition_style=plan.transition_style,
        bass_swap_beat_frac=plan.bass_swap_beat_frac,
        context_beats=16,
    )
    safe_lbl = _safe_filename_part(label)
    safe_a = _safe_filename_part(card_a.track_id)
    safe_b = _safe_filename_part(card_b.track_id)
    out_root = out_dir.resolve()
    wav_path = (out_root / f"{safe_lbl}_{safe_a}_to_{safe_b}.wav").resolve()
    if not wav_path.is_relative_to(out_root):
        raise ValueError(f"Unsafe output path for pair WAV: {wav_path}")

    render_res = render_transition(
        card_a,
        card_b,
        spec,
        out_wav=wav_path,
        sr=sr,
        stereo=True,
        use_demucs=(spec.transition_style == "stem_strip_blend"),
    )
    qa = verify_transition(render_res, card_a, card_b, spec)
    from ai_dj.rl.grpo_env import _section_at_beat

    sec_a_obj = _section_at_beat(card_a, spec.cue_out_beat_a)
    sec_b_obj = _section_at_beat(card_b, spec.cue_in_beat_b)
    sec_a = sec_a_obj.label if sec_a_obj is not None else "unknown"
    sec_b = sec_b_obj.label if sec_b_obj is not None else "unknown"
    trace = {
        "label": label,
        "model": base_model,
        "sampler_checkpoint": sampler_checkpoint,
        "audio_native": audio_native,
        "audio_clips": clip_meta,
        "prompt_text": prompt_text,
        "raw_completion": raw_text,
        "schema_valid": True,
        "track_a": card_a.track_id,
        "track_b": card_b.track_id,
        "bpm_a": card_a.bpm,
        "bpm_b": card_b.bpm,
        "key_a": card_a.camelot,
        "key_b": card_b.camelot,
        "style": plan.transition_style,
        "overlap_beats": spec.overlap_beats,
        "cue_out_beat_a": spec.cue_out_beat_a,
        "cue_out_sec_a": round(card_a.beat_times[spec.cue_out_beat_a], 2),
        "cue_out_section_a": sec_a,
        "cue_in_beat_b": spec.cue_in_beat_b,
        "cue_in_sec_b": round(card_b.beat_times[spec.cue_in_beat_b], 2),
        "cue_in_section_b": sec_b,
        "bass_swap_beat_frac": spec.bass_swap_beat_frac,
        "total_reward": bd.total_reward,
        "compatibility_reward": bd.compatibility_reward,
        "craft_reward": bd.craft_reward,
        "qa_reward": qa.verifiable_reward,
        "qa_flags": list(qa.defects_detected),
        "beat_drift_ms": qa.beat_drift_ms,
        "lufs_jump_db": qa.lufs_jump_db,
        "wav_path": str(wav_path),
    }
    (out_root / f"{safe_lbl}_trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    return trace
