"""Audio-native multimodal prompt builder (DmelChunk WAV clips) for Inkling-Small."""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
import soundfile as sf

from ai_dj.audio.analyzer import AudioTrackCard, TrackSection, load_audio_mono
from ai_dj.rl.grpo_env import AudioTransitionTask, format_audio_grpo_prompt

AUDIO_NATIVE_SYSTEM_PROMPT = (
    "You are an expert electronic club DJ planner who can HEAR raw audio cue clips "
    "from both songs alongside their full-song structural section timelines.\n"
    "Listen carefully to the attached audio snippets (Deck A exit moments and Deck B "
    "entry moments) to judge vocal density, kick/drum timbre, bass punch, synth harmony, "
    "and energy.\n"
    "IMPORTANT: Real club blends need full phrase overlaps (24, 32, 48, or 64 beats) "
    "and MUST cue early enough in Deck A (`cue_out_beat_prev + overlap_beats <= total_beats - 4`) "
    "so the transition breathes smoothly without abrupt cuts.\n"
    "Choose the exact right moment to mix out of Deck A and into Deck B, the creative DSP "
    "technique, AND the exact parametric stem-automation timing (fractions in 0.15..0.90):\n"
    '- "stem_strip_blend": 24-48b progressive 3-band drum-morph + 4-stem mix — BEST when the '
    "two songs have distinct drum/kick timbres, vocals, or distant keys (morphs hats >4 kHz first, "
    "then claps, then swaps <180 Hz kick+bass on the quantized downbeat).\n"
    '- "bass_swap": 24-48b 3-band EQ blend when drum timbres match closely and vocals are low.\n'
    '- "filter_sweep": 24-48b smooth HP/LP filter sweep across energy contrasts.\n'
    '- "long_blend": 32-64b hypnotic blend when BPM/key/timbre match closely and vocals are low.\n'
    '- "quick_cut": short 8-beat handoff ONLY when BPM gap >5% or key clash is unavoidable.\n'
    "Stem automation guidelines:\n"
    '- "hat_morph_end_frac": fraction when B\'s hi-hats finish replacing A\'s '
    "(typically 0.40-0.55).\n"
    '- "harmony_strip_a_end_frac": fraction when A\'s synths/chords reach 0% (use 0.35-0.45 for '
    "distant keys dist>=3 to create a clean drum-bridge; 0.60-0.80 for compatible keys).\n"
    '- "harmony_entry_b_start_frac": fraction when B\'s synths/chords enter (after '
    "harmony_strip_a_end_frac when keys are distant; 0.10-0.25 when keys match).\n"
    '- "vocal_entry_b_start_frac": fraction when B\'s vocals enter (after A\'s vocals clear, '
    "typically 0.50-0.65).\n"
    "CDJ phrasing & loop primitives (use to avoid plain volume crossfades):\n"
    '- "loop_a_beats": 0 (off) or 4, 8, 16 to loop a beat-exact phrase/hook of Deck A while '
    "bringing in Deck B's drums underneath.\n"
    '- "loop_a_tighten": true/false — halve Deck A\'s loop (16->8->4) in the final bars for '
    "tension before cutover.\n"
    '- "hold_a_stem": "none", "vocals" (keep A\'s vocal riding over B\'s beat), "harmony" '
    '(hold A\'s synth hook), or "loop_all" (hold A\'s looped hook until B\'s phrase downbeat).\n'
    '- "exit_fx": "none", "hpf_sweep", "echo_out", or "hpf_echo_out" (beat-synced echo tail '
    "on A's exit right into B's phrase downbeat).\n"
    "Respond with ONLY a single valid JSON object with keys in order: "
    '"next_track_id", "transition_style", "overlap_beats", '
    '"cue_out_beat_prev", "cue_in_beat_next", "bass_swap_beat_frac", '
    '"hat_morph_end_frac", "harmony_strip_a_end_frac", '
    '"harmony_entry_b_start_frac", "vocal_entry_b_start_frac", '
    '"loop_a_beats", "loop_a_tighten", "hold_a_stem", "exit_fx".'
)


def _select_candidate_moments(card: AudioTrackCard, *, role: str) -> list[tuple[str, int, float]]:
    """Pick 2 representative structural cue moments (label, beat, sec) to audition."""
    n_beats = len(card.beat_times)
    if n_beats < 16:
        return [("start", 0, 0.0)]

    if card.sections:
        if role == "out":
            # Prefer a late breakdown/groove AND the outro
            mid_cands = [
                s
                for s in card.sections
                if s.label in ("breakdown", "groove", "drop") and s.start_beat >= n_beats // 3
            ]
            out_cands = [s for s in card.sections if s.label == "outro"]
            chosen: list[TrackSection] = []
            if mid_cands:
                chosen.append(mid_cands[-1])
            if out_cands:
                chosen.append(out_cands[0])
            if not chosen:
                chosen = card.sections[-2:]
        else:
            # Prefer intro AND first buildup/groove/drop
            intro_cands = [s for s in card.sections if s.label == "intro"]
            body_cands = [s for s in card.sections if s.label in ("buildup", "groove", "drop")]
            chosen = []
            if intro_cands:
                chosen.append(intro_cands[0])
            if body_cands:
                chosen.append(body_cands[0])
            if not chosen:
                chosen = card.sections[:2]

        out: list[tuple[str, int, float]] = []
        seen_b: set[int] = set()
        for s in chosen[:2]:
            if s.start_beat not in seen_b:
                seen_b.add(s.start_beat)
                out.append((s.label, int(s.start_beat), float(s.start_sec)))
        if out:
            return out

    # Fallback when sections are absent
    if role == "out":
        b = card.recommended_outro_beat or max(32, n_beats - 64)
        b = min(b, n_beats - 1)
        return [("outro", b, float(card.beat_times[b]))]
    b = card.recommended_intro_beat or min(16, n_beats - 1)
    return [("intro", b, float(card.beat_times[b]))]


def extract_cue_wav_b64(
    card: AudioTrackCard,
    start_sec: float,
    *,
    duration_sec: float = 4.0,
    sr: int = 16000,
    audio_cache_16k: dict[str, np.ndarray] | None = None,
) -> str:
    """Slice a compact mono 16kHz WAV cue snippet and encode it as base64."""
    if audio_cache_16k is not None and card.track_id in audio_cache_16k:
        y = audio_cache_16k[card.track_id]
    else:
        y = load_audio_mono(card.audio_path, sr=sr)
        if audio_cache_16k is not None:
            audio_cache_16k[card.track_id] = y

    n_samples = int(round(duration_sec * sr))
    s0 = max(0, min(len(y) - 1, int(round(start_sec * sr))))
    clip = y[s0 : s0 + n_samples]
    if len(clip) < n_samples:
        clip = np.pad(clip, (0, n_samples - len(clip)))
    clip = np.clip(clip.astype(np.float32), -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, clip, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_audio_native_messages(
    task: AudioTransitionTask,
    *,
    clip_sec: float = 4.0,
    sr: int = 16000,
    audio_cache_16k: dict[str, np.ndarray] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build multimodal messages with native input_audio WAV clips for Inkling-Small.

    Returns (messages, clip_metadata) where clip_metadata records every auditioned clip
    for raw trace inspection.
    """
    prompt_text = format_audio_grpo_prompt(task)
    content: list[dict[str, Any]] = [
        {"type": "text", "text": prompt_text + "\n\n### Audio Cue Clips"}
    ]
    clip_meta: list[dict[str, Any]] = []

    # Deck A exit moments (up to 2 clips)
    for label, beat, sec in _select_candidate_moments(task.deck_a, role="out"):
        b64 = extract_cue_wav_b64(
            task.deck_a, sec, duration_sec=clip_sec, sr=sr, audio_cache_16k=audio_cache_16k
        )
        content.append(
            {
                "type": "text",
                "text": (
                    f"\nDeck A ({task.deck_a.track_id}) {label} clip at beat {beat} "
                    f"({sec:.1f}s, {clip_sec:.0f}s):"
                ),
            }
        )
        content.append({"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}})
        clip_meta.append(
            {
                "deck": "A",
                "track_id": task.deck_a.track_id,
                "section": label,
                "beat": beat,
                "start_sec": round(sec, 2),
                "duration_sec": clip_sec,
            }
        )

    # Candidate Deck B entry moments (2 clips if pair mode, else 1 clip for top 2 candidates)
    audition_cands = task.candidates[:1] if len(task.candidates) == 1 else task.candidates[:2]
    max_per_cand = 2 if len(task.candidates) == 1 else 1
    for cand in audition_cands:
        for label, beat, sec in _select_candidate_moments(cand, role="in")[:max_per_cand]:
            b64 = extract_cue_wav_b64(
                cand, sec, duration_sec=clip_sec, sr=sr, audio_cache_16k=audio_cache_16k
            )
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"\nDeck B ({cand.track_id}) {label} clip at beat {beat} "
                        f"({sec:.1f}s, {clip_sec:.0f}s):"
                    ),
                }
            )
            content.append({"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}})
            clip_meta.append(
                {
                    "deck": "B",
                    "track_id": cand.track_id,
                    "section": label,
                    "beat": beat,
                    "start_sec": round(sec, 2),
                    "duration_sec": clip_sec,
                }
            )

    content.append(
        {
            "type": "text",
            "text": (
                "\nBased on what you heard and the section timelines, return ONLY the JSON plan."
            ),
        }
    )
    messages = [
        {"role": "system", "content": AUDIO_NATIVE_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    return messages, clip_meta
