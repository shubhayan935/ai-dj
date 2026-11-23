"""Unit tests for Phase 3 prompt formatting, dataset export, Tinker dry-runs, and model eval."""

from __future__ import annotations

import json
from pathlib import Path

from ai_dj.data.schemas import (
    CandidateTrackSummary,
    DPOPair,
    SetBrief,
    SFTExample,
    TransitionPlan,
)
from ai_dj.training.dataset import export_training_datasets
from ai_dj.training.eval_model import evaluate_raw_predictions_jsonl
from ai_dj.training.prompts import (
    format_dpo_pair_messages,
    format_sft_messages,
    format_target_completion,
    parse_model_output,
)
from ai_dj.training.tinker_dpo import DPOConfig, compute_dpo_metrics_np, run_tinker_dpo
from ai_dj.training.tinker_sft import SFTConfig, cosine_lr, run_tinker_sft


def _make_sample_example(example_id: str = "ex1", tier: str = "gold") -> SFTExample:
    cands = [
        CandidateTrackSummary(
            track_id=f"t{i}",
            artist=f"Artist {i}",
            title=f"Track {i}",
            bpm=124.0 + (i * 0.5),
            duration_sec=360.0,
            genres=["House", "Techno"],
        )
        for i in range(5)
    ]
    return SFTExample(
        example_id=example_id,
        split="train",
        tier=tier,  # type: ignore[arg-type]
        composite_score=0.88,
        mix_id="m1",
        dj_name="Ben UFO",
        set_brief=SetBrief(
            genres=["House", "Techno"],
            curated_series="Essential Mix",
            target_bpm_min=122.0,
            target_bpm_max=128.0,
            set_progress=0.5,
            set_phase="peak",
            style_reference_dj="Ben UFO",
        ),
        history=[cands[0]],
        current_track=cands[1],
        candidates=cands,
        target_track_id="t2",
        target_plan=TransitionPlan(
            track_cue_out_beat_prev=384.0,
            track_cue_in_beat_next=16.0,
            overlap_beats=48.0,
            overlap_bars=12.0,
            tempo_stretch_prev_pct=0.0,
            tempo_stretch_next_pct=0.4,
            transition_style="standard_blend",
        ),
    )


def test_prompt_formatting_respects_dj_toggle_and_roundtrips_json() -> None:
    ex = _make_sample_example()
    msgs_with = format_sft_messages(ex, include_dj=True)
    msgs_without = format_sft_messages(ex, include_dj=False)

    assert len(msgs_with) == 3
    assert "Style Reference DJ: Ben UFO" in msgs_with[1]["content"]
    assert "Style Reference DJ" not in msgs_without[1]["content"]

    pred, diag = parse_model_output(msgs_with[2]["content"], ex.candidates)
    assert diag.json_valid
    assert diag.valid_track_id
    assert diag.style_valid
    assert pred.ranked_track_ids == ["t2"]
    assert pred.predicted_overlap_beats == 48.0
    assert pred.predicted_style == "standard_blend"


def test_parse_model_output_handles_think_tags_fences_and_malformed() -> None:
    ex = _make_sample_example()
    fenced = (
        "<think>Choosing t3 for tight BPM match.</think>\n"
        "```json\n"
        '{"next_track_id":"t3","cue_out_beat_prev":320,"cue_in_beat_next":0,'
        '"overlap_beats":64,"transition_style":"long_blend","tempo_stretch_next_pct":-0.8}\n'
        "```"
    )
    pred, diag = parse_model_output(fenced, ex.candidates)
    assert diag.json_valid and diag.valid_track_id and diag.style_valid
    assert pred.ranked_track_ids == ["t3"]
    assert pred.predicted_style == "long_blend"

    bad_pred, bad_diag = parse_model_output("not json at all", ex.candidates)
    assert not bad_diag.json_valid
    assert bad_pred.ranked_track_ids == []


def test_export_and_dry_run_sft_dpo_and_eval(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()

    gold_ex = _make_sample_example("ex_gold", tier="gold")
    bronze_ex = _make_sample_example("ex_bronze", tier="bronze")
    val_ex = _make_sample_example("ex_val", tier="bronze").model_copy(update={"split": "val"})

    (processed / "sft_train.jsonl").write_text(
        gold_ex.model_dump_json() + "\n" + bronze_ex.model_dump_json() + "\n",
        encoding="utf-8",
    )
    (processed / "sft_val.jsonl").write_text(
        val_ex.model_dump_json() + "\n",
        encoding="utf-8",
    )

    pair = DPOPair(
        pair_id="p1",
        split="train",
        negative_type="off_phrase_plan",
        mix_id="m1",
        set_brief=gold_ex.set_brief,
        history=gold_ex.history,
        current_track=gold_ex.current_track,
        candidates=gold_ex.candidates,
        chosen_track_id=gold_ex.target_track_id,
        chosen_plan=gold_ex.target_plan,
        rejected_track_id=gold_ex.target_track_id,
        rejected_plan=gold_ex.target_plan.model_copy(
            update={"overlap_beats": 7.0, "overlap_bars": 1.75, "transition_style": "quick_cut"}
        ),
    )
    (processed / "dpo_train.jsonl").write_text(pair.model_dump_json() + "\n", encoding="utf-8")

    # Verify DPO pair formatting helper directly
    dpo_msg = format_dpo_pair_messages(pair)
    assert dpo_msg["negative_type"] == "off_phrase_plan"
    assert dpo_msg["chosen"][0]["content"] != dpo_msg["rejected"][0]["content"]

    train_out = tmp_path / "training"
    summary = export_training_datasets(
        processed_dir=processed,
        out_dir=train_out,
        train_tiers=("gold", "silver"),
    )
    # Bronze is excluded from train, retained in val
    assert summary["sft_splits"]["train"]["examples"] == 1
    assert summary["sft_splits"]["val"]["examples"] == 1
    assert summary["dpo_train"]["examples"] == 1
    assert (train_out / "training_dataset_summary.md").exists()

    sft_res = run_tinker_sft(
        SFTConfig(
            train_jsonl=train_out / "sft_train_chat.jsonl",
            val_jsonl=train_out / "sft_val_chat.jsonl",
            out_dir=train_out / "runs" / "sft",
            batch_size=1,
            epochs=2,
        ),
        dry_run=True,
    )
    assert sft_res["mode"] == "dry_run"
    assert sft_res["total_steps"] == 2

    dpo_res = run_tinker_dpo(
        DPOConfig(
            dpo_jsonl=train_out / "dpo_train_chat.jsonl",
            out_dir=train_out / "runs" / "dpo",
            batch_size=1,
            epochs=1,
        ),
        dry_run=True,
    )
    assert dpo_res["mode"] == "dry_run"
    assert dpo_res["dpo_pairs"] == 1

    # Verify prediction evaluator against perfect oracle completion
    preds_path = train_out / "oracle_preds.jsonl"
    preds_path.write_text(
        json.dumps(
            {
                "example_id": val_ex.example_id,
                "raw_output": format_target_completion(val_ex.target_track_id, val_ex.target_plan),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    eval_report = evaluate_raw_predictions_jsonl(
        split_jsonl=processed / "sft_val.jsonl",
        predictions_jsonl=preds_path,
        model_name="oracle_check",
        out_md=train_out / "eval.md",
    )
    assert eval_report["schema_health"]["json_valid_rate"] == 1.0
    assert eval_report["model_metrics"]["overall"]["recall_at_1"] == 1.0
    assert eval_report["model_metrics"]["overall"]["overlap_beats_mae"] == 0.0
    assert (train_out / "eval.md").exists()


def test_lr_schedule_and_dpo_metrics_math() -> None:
    lr0 = cosine_lr(0, 100, 2e-4, warmup_ratio=0.1, min_ratio=0.1)
    lr_peak = cosine_lr(9, 100, 2e-4, warmup_ratio=0.1, min_ratio=0.1)
    lr_end = cosine_lr(100, 100, 2e-4, warmup_ratio=0.1, min_ratio=0.1)
    assert 0.0 < lr0 < lr_peak
    assert abs(lr_peak - 2e-4) < 1e-9
    assert abs(lr_end - 2e-5) < 1e-9

    m = compute_dpo_metrics_np(
        chosen_logps=[-2.0, -1.5],
        rejected_logps=[-3.0, -4.0],
        ref_chosen_logps=[-2.5, -2.0],
        ref_rejected_logps=[-2.5, -3.0],
        beta=0.1,
    )
    assert m["accuracy"] == 1.0
    assert m["margin"] > 0.0
    assert 0.5 < m["dpo_loss"] < 0.7


def test_review_regressions_token_extraction_truncation_dpo_and_invalid_eval(
    tmp_path: Path,
) -> None:
    import math
    from types import SimpleNamespace

    from ai_dj.training.tinker_dpo import (
        _completion_mask_values,
        _full_sequence_for_logprobs,
        _resolve_dpo_model_name,
        _unwrap_tensor_values,
    )
    from ai_dj.training.tinker_sft import _extract_token_ids

    # BUG 0002: BatchEncoding-style dict returns token IDs, not key names
    assert _extract_token_ids({"input_ids": [10, 20, 30], "attention_mask": [1, 1, 1]}) == [
        10,
        20,
        30,
    ]
    assert _extract_token_ids({"input_ids": [[1, 2, 3]]}) == [1, 2, 3]

    # BUG 0003 & 0004: TensorData-like wrappers unwrap and full sequence appends last target
    class FakeTensorData:
        def __init__(self, vals):
            self._vals = vals

        def tolist(self):
            return list(self._vals)

    class FakeModelInput:
        def __init__(self, tokens):
            self.tokens = list(tokens)

        def append_int(self, tok: int):
            return FakeModelInput([*self.tokens, tok])

    fake_datum = SimpleNamespace(
        model_input=FakeModelInput([100, 101, 102]),
        loss_fn_inputs={
            "target_tokens": FakeTensorData([101, 102, 103]),
            "weights": FakeTensorData([0.0, 1.0, 1.0]),
        },
    )
    assert _unwrap_tensor_values(fake_datum.loss_fn_inputs["weights"]) == [0.0, 1.0, 1.0]
    full_seq = _full_sequence_for_logprobs(fake_datum)
    assert full_seq.tokens == [100, 101, 102, 103]

    # Normalized SFT weights (summing to 1.0) convert to binary 0/1 mask for DPO logprob sums
    norm_datum = SimpleNamespace(
        loss_fn_inputs={"weights": FakeTensorData([0.0, 0.25, 0.25, 0.5])}
    )
    assert _completion_mask_values(norm_datum) == [0.0, 1.0, 1.0, 1.0]

    # Resumed DPO resolves base model name from checkpoint info instead of default
    fake_client = SimpleNamespace(
        get_info=lambda: SimpleNamespace(
            model_name=None,
            model_data=SimpleNamespace(model_name="Qwen/Qwen3.6-35B-A3B"),
        )
    )
    assert _resolve_dpo_model_name(fake_client, "thinkingmachines/Inkling-2-Mini") == (
        "Qwen/Qwen3.6-35B-A3B"
    )

    # BUG 0006: Invalid non-JSON completions earn 0.0 style and within-8b credit
    ex = _make_sample_example("ex_bad")
    split_path = tmp_path / "sft_val.jsonl"
    split_path.write_text(ex.model_dump_json() + "\n", encoding="utf-8")
    preds_path = tmp_path / "bad_preds.jsonl"
    preds_path.write_text(
        json.dumps({"example_id": ex.example_id, "raw_output": "not json"}) + "\n",
        encoding="utf-8",
    )
    rep = evaluate_raw_predictions_jsonl(split_path, preds_path, model_name="bad_model")
    ov = rep["model_metrics"]["overall"]
    assert rep["schema_health"]["json_valid_rate"] == 0.0
    assert ov["recall_at_1"] == 0.0
    assert ov["style_accuracy"] == 0.0
    assert ov["overlap_within_8_beats"] == 0.0
    assert math.isnan(ov["overlap_beats_mae"])

