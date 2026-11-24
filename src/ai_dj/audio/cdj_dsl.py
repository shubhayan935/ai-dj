"""Composable beat-indexed CDJ action DSL for arbitrary simple-to-complex DJ transitions.

Every DJ transition — from a simple beat-1 drop swap or 16-beat bass swap to a complex
Deck B vocal/percussion tease, acapella bridge, 3-band progressive drum morph, reverb wash,
and beat-synced echo-out — compiles to a `CDJProgram` of independent Deck A / Deck B
per-stem beat keyframes, per-deck loops, and beat-windowed FX.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from scipy.signal import butter, sosfiltfilt

StemKey = Literal["drums_low", "drums_mid", "drums_high", "bass", "vocals", "other"]
FXStemTarget = Literal["all", "drums", "bass", "vocals", "other"]
FXType = Literal["hpf_sweep", "lpf_sweep", "echo_out", "reverb_wash", "vinyl_brake"]


class BeatKeyframe(BaseModel):
    """Gain value at a specific beat offset inside the transition overlap."""

    beat: float = Field(ge=0.0, le=128.0)
    gain: float = Field(ge=0.0, le=1.5)


class DeckLoop(BaseModel):
    """Beat-exact loop on Deck A (loop-out) or Deck B (early tease loop before release)."""

    start_beat: float = Field(default=0.0, ge=0.0, le=128.0)
    length_beats: int = Field(default=8, ge=1, le=32)
    end_beat: float | None = Field(default=None, ge=1.0, le=128.0)
    tighten: bool = False


class StemFX(BaseModel):
    """Beat-windowed DJ effect applied to a specific deck and stem."""

    deck: Literal["a", "b"]
    stem: FXStemTarget = "all"
    fx_type: FXType
    start_beat: float = Field(ge=0.0, le=128.0)
    end_beat: float = Field(ge=0.25, le=128.0)
    start_hz: float = Field(default=140.0, ge=30.0, le=16000.0)
    end_hz: float = Field(default=2400.0, ge=30.0, le=16000.0)
    feedback: float = Field(default=0.52, ge=0.1, le=0.85)
    delay_beats: float = Field(default=0.75, ge=0.125, le=2.0)


class DeckAutomation(BaseModel):
    """Independent per-stem keyframe curves and optional loop for one CDJ deck."""

    drums_low: list[BeatKeyframe] = Field(default_factory=list)
    drums_mid: list[BeatKeyframe] = Field(default_factory=list)
    drums_high: list[BeatKeyframe] = Field(default_factory=list)
    bass: list[BeatKeyframe] = Field(default_factory=list)
    vocals: list[BeatKeyframe] = Field(default_factory=list)
    other: list[BeatKeyframe] = Field(default_factory=list)
    loop: DeckLoop | None = None


class CDJProgram(BaseModel):
    """Complete composable DJ transition program executed on 4-stem + 3-band drums."""

    name: str = "custom"
    overlap_beats: int = Field(default=32, ge=4, le=128)
    deck_a: DeckAutomation
    deck_b: DeckAutomation
    fx: list[StemFX] = Field(default_factory=list)


def _eval_keyframes(
    keyframes: list[BeatKeyframe],
    n_samples: int,
    overlap_beats: float,
    *,
    default_gain: float,
) -> np.ndarray:
    """Evaluate smoothstep-interpolated per-stem gain curve across [0, overlap_beats]."""
    if not keyframes:
        return np.full(n_samples, default_gain, dtype=np.float64)
    pts = sorted(keyframes, key=lambda k: k.beat)
    beats = np.linspace(0.0, float(overlap_beats), n_samples, dtype=np.float64)
    out = np.empty(n_samples, dtype=np.float64)

    # Hold first/last values outside keyframe range
    out[beats <= pts[0].beat] = pts[0].gain
    out[beats >= pts[-1].beat] = pts[-1].gain
    for i in range(len(pts) - 1):
        b0, g0 = pts[i].beat, pts[i].gain
        b1, g1 = pts[i + 1].beat, pts[i + 1].gain
        mask = (beats >= b0) & (beats <= b1)
        if not np.any(mask):
            continue
        if b1 <= b0 + 1e-6:
            out[mask] = g1
        else:
            u = (beats[mask] - b0) / (b1 - b0)
            # Smooth cosine ease between keyframes (musical equal-power feel, zero zipper noise)
            s = 0.5 * (1.0 - np.cos(np.pi * u))
            out[mask] = g0 + (g1 - g0) * s
    return out


def apply_deck_loop(
    audio: np.ndarray,
    loop: DeckLoop,
    sr: int,
    beat_dur_sec: float,
) -> np.ndarray:
    """Apply a beat-exact Deck A or Deck B loop over [start_beat, end_beat) with 8ms crossfade."""
    n_ch, total_n = audio.shape
    beat_n = max(64, int(round(beat_dur_sec * sr)))
    s0 = max(0, min(total_n - 512, int(round(loop.start_beat * beat_n))))
    s1 = (
        total_n
        if loop.end_beat is None
        else max(s0 + 512, min(total_n, int(round(loop.end_beat * beat_n))))
    )
    loop_n = max(512, min(s1 - s0, int(round(loop.length_beats * beat_n))))
    if loop_n >= s1 - s0:
        return audio

    xfade = min(int(round(0.008 * sr)), loop_n // 8)
    out = audio.copy()
    region_len = s1 - s0
    src_region = audio[:, s0:s1]

    pos = 0
    cur_len = loop_n
    prev_len = loop_n
    while pos < region_len:
        if loop.tighten and pos >= int(0.49 * region_len) and cur_len >= 1024:
            target = loop_n // 2 if pos < int(0.74 * region_len) else max(512, loop_n // 4)
            cur_len = max(512, min(loop_n, target))
        chunk = src_region[:, :cur_len]
        take = min(cur_len, region_len - pos)
        if pos > 0 and xfade > 4 and take > xfade:
            fade = np.linspace(0.0, 1.0, xfade, dtype=np.float32)
            if prev_len + xfade <= src_region.shape[1]:
                prev_tail = src_region[:, prev_len : prev_len + xfade]
            else:
                prev_tail = out[:, max(s0, s0 + pos - xfade) : s0 + pos]
                if prev_tail.shape[1] < xfade:
                    prev_tail = np.pad(prev_tail, ((0, 0), (0, xfade - prev_tail.shape[1])))
            out[:, s0 + pos : s0 + pos + xfade] = prev_tail * (1.0 - fade) + chunk[:, :xfade] * fade
            out[:, s0 + pos + xfade : s0 + pos + take] = chunk[:, xfade:take]
        else:
            out[:, s0 + pos : s0 + pos + take] = chunk[:, :take]
        prev_len = cur_len
        pos += cur_len
    return out


def _apply_reverb_wash(
    sig: np.ndarray,
    sr: int,
    s0: int,
    s1: int,
    *,
    decay: float = 0.55,
) -> np.ndarray:
    """Lightweight stereo multi-tap club reverb wash over [s0, s1] with smooth wet ramp."""
    out = sig.copy()
    n = sig.shape[1]
    s0 = max(0, min(n - 1, s0))
    s1 = max(s0 + 1, min(n, s1))
    seg = sig[:, s0:s1].astype(np.float64)
    if seg.shape[1] < 256:
        return out
    # Band-limit reverb input (250 Hz - 6 kHz) so sub-bass stays tight
    nyq = 0.5 * sr
    sos = butter(2, [250.0 / nyq, min(6000.0 / nyq, 0.92)], btype="bandpass", output="sos")
    band = sosfiltfilt(sos, seg, axis=-1)

    # Schroeder-style multi-tap diffused delay tail extending up to 1.5s past s1
    tail_end = min(n, s1 + int(1.5 * sr))
    wet = np.zeros((sig.shape[0], tail_end - s0), dtype=np.float64)
    taps_ms = (37, 59, 89, 127, 179, 241, 317, 419)
    for idx, ms in enumerate(taps_ms):
        d = int(round((ms / 1000.0) * sr))
        g = (decay ** (idx * 0.45)) * 0.24
        if d >= wet.shape[1]:
            continue
        take = min(band.shape[1], wet.shape[1] - d)
        if wet.shape[0] == 2:
            pan = np.array([[1.0], [0.75]]) if idx % 2 == 0 else np.array([[0.75], [1.0]])
            wet[:, d : d + take] += g * pan * band[:, :take]
        else:
            wet[:, d : d + take] += g * band[:, :take]

    # Ramp wet in smoothly over [s0, s1], then let tail ring out
    ramp_len = s1 - s0
    env = np.ones(wet.shape[1], dtype=np.float64)
    env[:ramp_len] = np.linspace(0.0, 1.0, ramp_len, dtype=np.float64)
    if wet.shape[1] > ramp_len:
        env[ramp_len:] = np.linspace(1.0, 0.0, wet.shape[1] - ramp_len, dtype=np.float64)
    out[:, s0:tail_end] = (out[:, s0:tail_end] + (wet * env).astype(np.float32)).astype(np.float32)
    return out


def _apply_vinyl_brake(sig: np.ndarray, s0: int, s1: int) -> np.ndarray:
    """Turntable power-down pitch brake over [s0, s1], then silence after s1."""
    out = sig.copy()
    n = sig.shape[1]
    s0 = max(0, min(n - 2, s0))
    s1 = max(s0 + 2, min(n, s1))
    seg_len = s1 - s0
    # Quadratic phase slowdown (instantaneous speed goes 1.0 -> 0.0)
    u = np.linspace(0.0, 1.0, seg_len, dtype=np.float64)
    read_pos = s0 + (u - 0.5 * u**2) * seg_len
    idx_f = np.clip(read_pos, s0, s1 - 1.001)
    idx0 = idx_f.astype(np.int64)
    frac = (idx_f - idx0).astype(np.float32)
    fade = np.cos(0.5 * np.pi * u).astype(np.float32)
    for ch in range(sig.shape[0]):
        interp = sig[ch, idx0] * (1.0 - frac) + sig[ch, idx0 + 1] * frac
        out[ch, s0:s1] = interp * fade
    out[:, s1:] = 0.0
    return out


def execute_cdj_program(
    program: CDJProgram,
    stems_a: dict[str, np.ndarray],
    stems_b: dict[str, np.ndarray],
    sr: int,
    beat_dur_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Execute a composable `CDJProgram` on 4-stem Deck A and Deck B audio.

    Returns:
        (rendered_a, rendered_b, a_drums, b_drums, a_harm, b_harm)
    """
    from ai_dj.audio.renderer import (
        _apply_beat_synced_echo_out,
        _apply_filter_sweep,
        split_three_band,
    )

    ov_n = stems_a["drums"].shape[1]
    ov_beats = float(program.overlap_beats)
    beat_n = max(64, int(round(beat_dur_sec * sr)))

    # 3-band drum split (<180 Hz kick, 180-4000 Hz claps/snares, >4000 Hz hats/tops)
    a_dl, a_dm, a_dh = split_three_band(stems_a["drums"], sr, low_hz=180.0, high_hz=4000.0)
    b_dl, b_dm, b_dh = split_three_band(stems_b["drums"], sr, low_hz=180.0, high_hz=4000.0)

    def _render_deck(
        auto: DeckAutomation,
        stems: dict[str, np.ndarray],
        dl: np.ndarray,
        dm: np.ndarray,
        dh: np.ndarray,
        default_g: float,
    ) -> dict[str, np.ndarray]:
        g_dl = _eval_keyframes(auto.drums_low, ov_n, ov_beats, default_gain=default_g)
        g_dm = _eval_keyframes(auto.drums_mid, ov_n, ov_beats, default_gain=default_g)
        g_dh = _eval_keyframes(auto.drums_high, ov_n, ov_beats, default_gain=default_g)
        g_bs = _eval_keyframes(auto.bass, ov_n, ov_beats, default_gain=default_g)
        g_vc = _eval_keyframes(auto.vocals, ov_n, ov_beats, default_gain=default_g)
        g_ot = _eval_keyframes(auto.other, ov_n, ov_beats, default_gain=default_g)
        return {
            "drums": (dl * g_dl + dm * g_dm + dh * g_dh).astype(np.float32),
            "bass": (stems["bass"] * g_bs).astype(np.float32),
            "vocals": (stems["vocals"] * g_vc).astype(np.float32),
            "other": (stems["other"] * g_ot).astype(np.float32),
        }

    out_a = _render_deck(program.deck_a, stems_a, a_dl, a_dm, a_dh, default_g=1.0)
    out_b = _render_deck(program.deck_b, stems_b, b_dl, b_dm, b_dh, default_g=0.0)

    # Apply beat-windowed FX to targeted deck + stem(s)
    for fx in program.fx:
        target_dict = out_a if fx.deck == "a" else out_b
        stem_names = ("drums", "bass", "vocals", "other") if fx.stem == "all" else (fx.stem,)
        s0 = max(0, min(ov_n - 1, int(round(fx.start_beat * beat_n))))
        s1 = max(s0 + 1, min(ov_n, int(round(fx.end_beat * beat_n))))
        for sname in stem_names:
            sig = target_dict[sname]
            if fx.fx_type in ("hpf_sweep", "lpf_sweep"):
                btype: Literal["highpass", "lowpass"] = (
                    "highpass" if fx.fx_type == "hpf_sweep" else "lowpass"
                )
                seg_f = _apply_filter_sweep(
                    sig[:, s0:s1], sr, btype=btype, start_hz=fx.start_hz, end_hz=fx.end_hz
                ).astype(np.float32)
                updated = sig.copy()
                updated[:, s0:s1] = seg_f
                target_dict[sname] = updated
            elif fx.fx_type == "echo_out":
                cut_frac = float(fx.start_beat) / max(ov_beats, 1.0)
                target_dict[sname] = _apply_beat_synced_echo_out(
                    sig,
                    sr,
                    beat_dur_sec,
                    cut_frac,
                    feedback=fx.feedback,
                    delay_beats=fx.delay_beats,
                )
            elif fx.fx_type == "reverb_wash":
                target_dict[sname] = _apply_reverb_wash(sig, sr, s0, s1, decay=fx.feedback)
            elif fx.fx_type == "vinyl_brake":
                target_dict[sname] = _apply_vinyl_brake(sig, s0, s1)

    a_drums = out_a["drums"]
    b_drums = out_b["drums"]
    a_harm = (out_a["bass"] + out_a["vocals"] + out_a["other"]).astype(np.float32)
    b_harm = (out_b["bass"] + out_b["vocals"] + out_b["other"]).astype(np.float32)
    rendered_a = (a_drums + a_harm).astype(np.float32)
    rendered_b = (b_drums + b_harm).astype(np.float32)
    return rendered_a, rendered_b, a_drums, b_drums, a_harm, b_harm


# --- Example composable move builders (simple -> complex) ---


def build_drop_swap_dsl(overlap_beats: int = 16) -> CDJProgram:
    """Simple move: ride Deck A into a 4-beat HPF tension lift, then slam Deck B on beat 1."""
    b = float(overlap_beats)
    cut = b - 4.0
    kf_a_off = [
        BeatKeyframe(beat=0.0, gain=1.0),
        BeatKeyframe(beat=cut - 0.5, gain=1.0),
        BeatKeyframe(beat=cut, gain=0.0),
    ]
    kf_b_on = [
        BeatKeyframe(beat=0.0, gain=0.0),
        BeatKeyframe(beat=cut - 0.25, gain=0.0),
        BeatKeyframe(beat=cut, gain=1.0),
    ]
    return CDJProgram(
        name="phrase_drop_swap",
        overlap_beats=overlap_beats,
        deck_a=DeckAutomation(
            drums_low=kf_a_off,
            drums_mid=kf_a_off,
            drums_high=kf_a_off,
            bass=kf_a_off,
            vocals=kf_a_off,
            other=kf_a_off,
        ),
        deck_b=DeckAutomation(
            drums_low=kf_b_on,
            drums_mid=kf_b_on,
            drums_high=kf_b_on,
            bass=kf_b_on,
            vocals=kf_b_on,
            other=kf_b_on,
        ),
        fx=[
            StemFX(
                deck="a",
                stem="all",
                fx_type="hpf_sweep",
                start_beat=max(0.0, cut - 8.0),
                end_beat=cut,
                start_hz=80.0,
                end_hz=1800.0,
            ),
            StemFX(
                deck="a",
                stem="vocals",
                fx_type="echo_out",
                start_beat=cut,
                end_beat=b,
                feedback=0.50,
                delay_beats=0.75,
            ),
        ],
    )


def build_b_tease_acapella_bridge_dsl(overlap_beats: int = 32) -> CDJProgram:
    """Complex move: tease Deck B's 4-beat hat/percussion loop early under Deck A,
    strip A to an acapella+reverb bridge at beat 16, swap kick+bass at beat 16,
    then echo-out A's vocal right as B's full drop lands at beat 24."""
    b = float(overlap_beats)
    mid = b * 0.5
    cut = b * 0.75
    return CDJProgram(
        name="b_tease_acapella_bridge",
        overlap_beats=overlap_beats,
        deck_a=DeckAutomation(
            drums_high=[BeatKeyframe(beat=0.0, gain=1.0), BeatKeyframe(beat=mid - 2.0, gain=0.0)],
            drums_mid=[BeatKeyframe(beat=4.0, gain=1.0), BeatKeyframe(beat=mid, gain=0.0)],
            drums_low=[BeatKeyframe(beat=mid - 1.0, gain=1.0), BeatKeyframe(beat=mid, gain=0.0)],
            bass=[BeatKeyframe(beat=mid - 1.0, gain=1.0), BeatKeyframe(beat=mid, gain=0.0)],
            other=[BeatKeyframe(beat=4.0, gain=1.0), BeatKeyframe(beat=mid, gain=0.0)],
            # Hold A's vocal as an acapella riding over B's beat until beat 24!
            vocals=[
                BeatKeyframe(beat=0.0, gain=1.0),
                BeatKeyframe(beat=cut, gain=1.0),
                BeatKeyframe(beat=cut + 1.0, gain=0.0),
            ],
        ),
        deck_b=DeckAutomation(
            # Loop B's first 4 beats as an early percussion tease from beat 0..16, then release!
            loop=DeckLoop(start_beat=0.0, length_beats=4, end_beat=mid, tighten=False),
            drums_high=[
                BeatKeyframe(beat=0.0, gain=0.0),
                BeatKeyframe(beat=6.0, gain=0.85),
                BeatKeyframe(beat=mid, gain=1.0),
            ],
            drums_mid=[BeatKeyframe(beat=6.0, gain=0.0), BeatKeyframe(beat=mid, gain=1.0)],
            drums_low=[BeatKeyframe(beat=mid - 1.0, gain=0.0), BeatKeyframe(beat=mid, gain=1.0)],
            bass=[BeatKeyframe(beat=mid - 1.0, gain=0.0), BeatKeyframe(beat=mid, gain=1.0)],
            other=[BeatKeyframe(beat=mid, gain=0.0), BeatKeyframe(beat=cut, gain=1.0)],
            vocals=[BeatKeyframe(beat=cut, gain=0.0), BeatKeyframe(beat=b - 2.0, gain=1.0)],
        ),
        fx=[
            StemFX(
                deck="a",
                stem="other",
                fx_type="reverb_wash",
                start_beat=max(0.0, mid - 8.0),
                end_beat=mid,
                feedback=0.55,
            ),
            StemFX(
                deck="a",
                stem="vocals",
                fx_type="echo_out",
                start_beat=cut,
                end_beat=b,
                feedback=0.52,
                delay_beats=0.75,
            ),
        ],
    )
