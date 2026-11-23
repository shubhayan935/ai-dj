"""Unit tests for Phase 4 audio-grounded GRPO environment, timbre similarity, and advantage math."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ai_dj.audio.analyzer import AudioTrackCard, timbre_cosine_similarity
from ai_dj.rl.grpo_env import (
    build_audio_grpo_tasks,
    format_audio_grpo_prompt,
    parse_creative_plan,
    score_creative_plan,
)
from ai_dj.rl.tinker_grpo import GRPOConfig, compute_group_advantages, run_tinker_grpo


def _fake_card(
    track_id: str,
    bpm: float,
    camelot: str,
    emb_seed: int,
    tmp_path: Path,
    vocal_ratio: float = 0.1,
) -> AudioTrackCard:
    wav_path = tmp_path / f"{track_id}.wav"
    wav_path.write_bytes(b"")
    rng = np.random.default_rng(emb_seed)
    vec = rng.normal(size=48)
    vec = vec / np.linalg.norm(vec)
    beat_sec = 60.0 / bpm
    beats = [round(i * beat_sec, 4) for i in range(160)]
    return AudioTrackCard(
        track_id=track_id,
        audio_path=str(wav_path),
        sample_rate=22050,
        duration_sec=beats[-1],
        bpm=bpm,
        bpm_stability=0.95,
        local_bpm_stability=0.92,
        beat_times=beats,
        phrase_boundaries_sec=beats[::32],
        key="A minor",
        camelot=camelot,
        key_confidence=0.85,
        integrated_lufs=-10.0,
        peak_dbfs=-0.5,
        energy_curve=[0.6] * 60,
        vocal_activity_mask=[False] * 60,
        recommended_intro_beat=16,
        recommended_outro_beat=96,
        genres=["electronic", "house"],
        timbre_embedding=[round(float(x), 5) for x in vec],
        mean_energy=0.6,
        vocal_ratio=vocal_ratio,
    )


def test_grpo_tasks_reward_and_dry_run(tmp_path: Path) -> None:
    cards = [
        _fake_card("t1", 124.0, "8A", 1, tmp_path),
        _fake_card("t2", 125.0, "8A", 1, tmp_path),  # identical timbre + exact key + tight BPM
        _fake_card("t3", 142.0, "2B", 2, tmp_path, vocal_ratio=0.6),  # distant key + big BPM gap
        _fake_card("t4", 126.0, "9A", 3, tmp_path),
    ]
    assert timbre_cosine_similarity(cards[0], cards[1]) > 0.99
    tasks = build_audio_grpo_tasks(cards, crate_size=3, num_tasks=4, seed=7)
    assert len(tasks) == 4
    prompt = format_audio_grpo_prompt(tasks[0])
    assert "timbre_sim=" in prompt
    assert "rec_intro_beat=" in prompt

    # Valid creative plan parses and scores higher for compatible pair than bad pair
    good_json = json.dumps(
        {
            "next_track_id": "t2",
            "transition_style": "bass_swap",
            "overlap_beats": 32,
            "cue_out_beat_prev": 96,
            "cue_in_beat_next": 16,
            "bass_swap_beat_frac": 0.5,
        }
    )
    bad_json = json.dumps(
        {
            "next_track_id": "t3",
            "transition_style": "long_blend",
            "overlap_beats": 64,
            "cue_out_beat_prev": 96,
            "cue_in_beat_next": 16,
            "bass_swap_beat_frac": 0.5,
        }
    )
    plan_good, err = parse_creative_plan(good_json, {"t2", "t3", "t4"})
    assert err is None and plan_good is not None
    rew_good = score_creative_plan(tasks[0], good_json, render_audio=False)
    rew_bad = score_creative_plan(tasks[0], bad_json, render_audio=False)
    assert rew_good.schema_valid and rew_bad.schema_valid
    assert rew_good.total_reward > rew_bad.total_reward

    # Invalid JSON earns 0.0
    rew_inv = score_creative_plan(tasks[0], "not json", render_audio=False)
    assert not rew_inv.schema_valid and rew_inv.total_reward == 0.0

    # Group advantages are zero-mean and 0 for constant groups
    advs = compute_group_advantages([0.2, 0.4, 0.8, 1.0])
    assert abs(sum(advs)) < 1e-6
    assert advs[-1] > advs[0]
    assert compute_group_advantages([0.5, 0.5, 0.5]) == [0.0, 0.0, 0.0]

    # Dry-run GRPO works end-to-end
    cards_jsonl = tmp_path / "cards.jsonl"
    cards_jsonl.write_text("\n".join(c.model_dump_json() for c in cards) + "\n", encoding="utf-8")
    res = run_tinker_grpo(
        GRPOConfig(cards_jsonl=cards_jsonl, out_dir=tmp_path / "grpo_out", num_steps=2),
        dry_run=True,
    )
    assert res["mode"] == "dry_run"
    assert res["club_stable_tracks"] == 4
