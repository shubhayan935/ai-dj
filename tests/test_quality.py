from ai_dj.data.quality import (
    assign_quality_tiers,
    classify_transition_style,
    compute_tempo_stretch_pct,
    score_transition,
)
from ai_dj.data.sources import summarize_beat_array


def test_score_transition_passes_clean_example():
    q = score_transition(
        match_rate_prev=0.92,
        match_rate_next=0.88,
        cost_prev=0.12,
        cost_next=0.15,
        matched_beats_prev=400,
        matched_beats_next=380,
        overlap_beats=32.0,
        stretch_prev_pct=1.2,
        stretch_next_pct=-0.8,
        key_change_prev=0,
        key_change_next=0,
        bpm_prev=126.0,
        bpm_next=127.0,
        stability_prev=0.98,
        stability_next=0.97,
        mix_id_ratio=0.95,
        curated_series="Boiler Room",
        pair_repeat_count=3,
        artist_pair_repeat_count=5,
    )
    assert q.passed_hard_gates is True
    assert q.failure_reasons == []
    assert q.composite_score > 0.80


def test_score_transition_rejects_noisy_alignment():
    q = score_transition(
        match_rate_prev=0.20,
        match_rate_next=0.30,
        cost_prev=0.80,
        cost_next=0.70,
        matched_beats_prev=30,
        matched_beats_next=40,
        overlap_beats=450.0,
        stretch_prev_pct=25.0,
        stretch_next_pct=0.0,
        key_change_prev=2,
        key_change_next=3,
        bpm_prev=126.0,
        bpm_next=126.0,
        stability_prev=0.50,
        stability_next=0.95,
        mix_id_ratio=0.30,
        curated_series=None,
        pair_repeat_count=1,
        artist_pair_repeat_count=1,
    )
    assert q.passed_hard_gates is False
    assert q.tier == "rejected"
    assert "low_match_rate" in q.failure_reasons
    assert "overlap_too_long" in q.failure_reasons
    assert "excessive_tempo_stretch" in q.failure_reasons
    assert "unstable_beatgrid" in q.failure_reasons


def test_tempo_stretch_and_beat_summary():
    # Track is 100s originally, played in 95s -> sped up ~5.26%
    assert abs(compute_tempo_stretch_pct(95.0, 100.0) - 5.263) < 0.01
    # 120 BPM = 0.5s per beat
    beats = [0.5 * i for i in range(1, 201)]
    bpm, stab, dur, count = summarize_beat_array(beats)
    assert bpm == 120.0
    assert stab == 1.0
    assert dur == 100.0
    assert count == 200


def test_assign_quality_tiers_and_styles():
    assert classify_transition_style(4.0) == "quick_cut"
    assert classify_transition_style(16.0) == "short_blend"
    assert classify_transition_style(32.0) == "standard_blend"
    assert classify_transition_style(96.0) == "long_blend"
    assert classify_transition_style(160.0) == "extended_overlay"

    # Build 10 passing scores and check 20% gold / 50% silver / 30% bronze
    items = []
    for i in range(10):
        q = score_transition(
            match_rate_prev=0.5 + 0.04 * i,
            match_rate_next=0.5 + 0.04 * i,
            cost_prev=0.2,
            cost_next=0.2,
            matched_beats_prev=200,
            matched_beats_next=200,
            overlap_beats=32.0,
            stretch_prev_pct=0.0,
            stretch_next_pct=0.0,
            key_change_prev=0,
            key_change_next=0,
            bpm_prev=125.0,
            bpm_next=125.0,
            stability_prev=0.95,
            stability_next=0.95,
            mix_id_ratio=0.9,
            curated_series=None,
            pair_repeat_count=1,
            artist_pair_repeat_count=1,
        )
        items.append((i, q))
    tiers = assign_quality_tiers(items)
    assert list(tiers.values()).count("gold") == 2
    assert list(tiers.values()).count("silver") == 5
    assert list(tiers.values()).count("bronze") == 3
