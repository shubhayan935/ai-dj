"""Unit tests for Phase 4 audio-grounded GRPO environment, timbre similarity, and advantage math."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ai_dj.audio.analyzer import (
    AudioTrackCard,
    TrackSection,
    segment_track_sections,
    timbre_cosine_similarity,
)
from ai_dj.rl.audio_native import build_audio_native_messages
from ai_dj.rl.grpo_env import (
    build_audio_grpo_tasks,
    build_pair_mix_tasks,
    format_audio_grpo_prompt,
    format_section_timeline,
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


def test_phase5_sections_pair_tasks_and_audio_native_messages(tmp_path: Path) -> None:
    import soundfile as sf

    sr = 16000
    t = np.linspace(0.0, 20.0, sr * 20, endpoint=False, dtype=np.float32)
    y = 0.25 * np.sin(2.0 * np.pi * 100.0 * t)
    beats = [round(i * 0.5, 4) for i in range(40)]  # 120 BPM
    energy = [0.2] * 5 + [0.8] * 10 + [0.3] * 5
    vocals = [False] * 10 + [True] * 5 + [False] * 5
    secs = segment_track_sections(y, sr, beats, energy, vocals)
    assert len(secs) >= 1
    assert isinstance(secs[0], TrackSection)
    assert secs[0].label == "intro"

    c1 = _fake_card("p1", 128.0, "8A", 10, tmp_path)
    c2 = _fake_card("p2", 128.0, "8A", 10, tmp_path)
    sf.write(c1.audio_path, y, sr)
    sf.write(c2.audio_path, y, sr)
    c1.sections = [
        TrackSection(
            index=0,
            label="intro",
            start_beat=0,
            end_beat=64,
            start_sec=0.0,
            end_sec=30.0,
            energy=0.3,
            bass_density=0.2,
            vocal_ratio=0.0,
            percussion_ratio=0.4,
        ),
        TrackSection(
            index=1,
            label="outro",
            start_beat=64,
            end_beat=128,
            start_sec=10.0,
            end_sec=20.0,
            energy=0.35,
            bass_density=0.25,
            vocal_ratio=0.0,
            percussion_ratio=0.3,
        ),
    ]
    c2.sections = list(c1.sections)
    tl = format_section_timeline(c1)
    assert "S0:intro@b0-64" in tl and "S1:outro@b64-128" in tl

    pair_tasks = build_pair_mix_tasks([c1, c2], num_tasks=2, seed=42)
    assert len(pair_tasks) == 2
    assert len(pair_tasks[0].candidates) == 1
    msgs, clips = build_audio_native_messages(pair_tasks[0], clip_sec=2.0, sr=sr)
    assert len(msgs) == 2 and msgs[1]["role"] == "user"
    audio_parts = [p for p in msgs[1]["content"] if p.get("type") == "input_audio"]
    assert len(audio_parts) == len(clips) and len(clips) >= 2
