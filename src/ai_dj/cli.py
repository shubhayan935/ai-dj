"""Command-line interface for the AI DJ data pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from ai_dj.audio.analyzer import analyze_audio_pool
from ai_dj.audio.cc_pool import ingest_cc_audio_pool
from ai_dj.audio.defect_gym import run_defect_gym
from ai_dj.data.report import write_markdown_report
from ai_dj.data.sources import (
    fetch_djmix_metadata,
    fetch_track_beats_summary,
    fetch_transitions_scalar,
)
from ai_dj.data.splits import build_cleaned_dataset
from ai_dj.eval.runner import run_baseline_suite
from ai_dj.training.dataset import export_training_datasets
from ai_dj.training.eval_model import evaluate_raw_predictions_jsonl
from ai_dj.training.tinker_dpo import DPOConfig, run_tinker_dpo
from ai_dj.training.tinker_sft import SFTConfig, run_tinker_sft

app = typer.Typer(help="AI DJ data acquisition, cleaning, ranking, and dataset export CLI.")
console = Console()


@app.command()
def ingest(
    raw_dir: Annotated[Path, typer.Option(help="Directory to store raw downloaded data.")] = Path(
        "data/raw"
    ),
    force: Annotated[bool, typer.Option(help="Re-download even if cached files exist.")] = False,
) -> None:
    """Download MixesDB metadata, scalar transition alignments, and track BPM summaries."""
    console.print("[bold]1/3[/bold] Fetching djmix-dataset.json...")
    meta = fetch_djmix_metadata(raw_dir, force=force)
    console.print(f"    Saved {meta} ({meta.stat().st_size / 1e6:.1f} MB)")

    console.print("[bold]2/3[/bold] Streaming scalar columns from djmix/transitions (HF)...")
    trans = fetch_transitions_scalar(raw_dir, force=force)
    console.print(f"    Saved {trans} ({trans.stat().st_size / 1e6:.1f} MB)")

    console.print("[bold]3/3[/bold] Summarizing track beatgrids from djmix/beats-tracks (HF)...")
    beats = fetch_track_beats_summary(raw_dir, force=force)
    console.print(f"    Saved {beats} ({beats.stat().st_size / 1e6:.1f} MB)")


@app.command()
def build(
    raw_dir: Annotated[Path, typer.Option(help="Directory containing ingested raw files.")] = Path(
        "data/raw"
    ),
    out_dir: Annotated[Path, typer.Option(help="Output directory for clean datasets.")] = Path(
        "data/processed"
    ),
    candidate_pool_size: Annotated[
        int, typer.Option(help="Candidate tracks per SFT/DPO example.")
    ] = 20,
    history_window: Annotated[
        int, typer.Option(help="Number of previous tracks in set history.")
    ] = 3,
    dj_dropout_prob: Annotated[
        float, typer.Option(help="Probability of dropping style_reference_dj in train SetBrief.")
    ] = 0.50,
) -> None:
    """Clean, score, rank, split (DJ-disjoint), and export SFT/DPO JSONL + quality report."""
    console.print("[bold]Building cleaned, ranked, DJ-disjoint datasets...[/bold]")
    summary = build_cleaned_dataset(
        raw_dir=raw_dir,
        out_dir=out_dir,
        candidate_pool_size=candidate_pool_size,
        history_window=history_window,
        dj_dropout_prob=dj_dropout_prob,
    )
    md_path = write_markdown_report(summary, out_dir / "data_quality_report.md")

    table = Table(title="AI DJ Data Pipeline Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    table.add_row("Raw transitions", f"{summary['total_raw_transitions']:,}")
    table.add_row(
        "Clean transitions",
        f"{summary['total_clean_transitions']:,} ({summary['retention_rate_pct']}%)",
    )
    table.add_row(
        "Gold / Silver / Bronze",
        f"{summary['tier_counts'].get('gold', 0):,} / "
        f"{summary['tier_counts'].get('silver', 0):,} / "
        f"{summary['tier_counts'].get('bronze', 0):,}",
    )
    table.add_row(
        "SFT (train / val / test)",
        f"{summary['sft_split_counts'].get('train', 0):,} / "
        f"{summary['sft_split_counts'].get('val', 0):,} / "
        f"{summary['sft_split_counts'].get('test', 0):,}",
    )
    table.add_row("DPO train pairs", f"{summary['dpo_train_pairs']:,}")
    console.print(table)
    console.print(f"Report written to {md_path}")


@app.command("eval-baselines")
def eval_baselines(
    processed_dir: Annotated[
        Path, typer.Option(help="Directory containing sft_val.jsonl and sft_test.jsonl.")
    ] = Path("data/processed"),
) -> None:
    """Evaluate deterministic DJ baselines on DJ-disjoint val and test splits."""
    console.print("[bold]Running offline baseline evals (with_dj & without_dj)...[/bold]")
    results = run_baseline_suite(processed_dir=processed_dir)
    for split_name, split_res in results["splits"].items():
        table = Table(title=f"Offline Baselines — {split_name} ({split_res['num_examples']:,})")
        table.add_column("Planner", style="cyan")
        table.add_column("Mode", style="magenta")
        table.add_column("R@1", justify="right")
        table.add_column("R@5", justify="right")
        table.add_column("MRR", justify="right")
        table.add_column("NDCG@10", justify="right", style="green")
        table.add_column("Ov MAE", justify="right")
        table.add_column("Style Acc", justify="right")
        for row in split_res["runs"]:
            m = row["metrics"]["overall"]
            table.add_row(
                row["planner"],
                row["mode"],
                f"{m['recall_at_1']:.3f}",
                f"{m['recall_at_5']:.3f}",
                f"{m['mrr']:.3f}",
                f"{m['ndcg_at_10']:.3f}",
                f"{m['overlap_beats_mae']:.1f}",
                f"{m['style_accuracy']:.3f}",
            )
        console.print(table)
    console.print(f"Baseline report written to {processed_dir / 'eval_baselines.md'}")


@app.command("ingest-cc-audio")
def ingest_cc_audio(
    out_dir: Annotated[
        Path, typer.Option(help="Directory to store Creative Commons MP3 audio pool.")
    ] = Path("data/raw/cc_audio"),
    max_tracks: Annotated[
        int, typer.Option(help="Number of CC tracks to download from JamendoMaxCaps.")
    ] = 12,
    force: Annotated[bool, typer.Option(help="Re-download even if manifest exists.")] = False,
) -> None:
    """Download an open Creative Commons electronic audio pool for DSP rendering & RL."""
    console.print(f"[bold]Ingesting up to {max_tracks} CC tracks from JamendoMaxCaps...[/bold]")
    manifest = ingest_cc_audio_pool(out_dir=out_dir, max_tracks=max_tracks, force=force)
    console.print(f"Saved CC audio manifest to {manifest}")


@app.command("analyze-audio")
def analyze_audio(
    manifest_path: Annotated[Path, typer.Option(help="Path to CC audio manifest.json.")] = Path(
        "data/raw/cc_audio/manifest.json"
    ),
    out_jsonl: Annotated[Path, typer.Option(help="Output JSONL path for AudioTrackCards.")] = Path(
        "data/processed/cc_track_cards.jsonl"
    ),
) -> None:
    """Run deterministic MIR analysis (BPM, beatgrid, Camelot key, LUFS, vocal mask) on CC pool."""
    console.print(f"[bold]Analyzing CC audio pool from {manifest_path}...[/bold]")
    cards = analyze_audio_pool(manifest_path=manifest_path, out_jsonl=out_jsonl)
    table = Table(title=f"MIR AudioTrackCards ({len(cards)} tracks)")
    table.add_column("Track ID", style="cyan")
    table.add_column("Dur (s)", justify="right")
    table.add_column("BPM", justify="right", style="green")
    table.add_column("Stab", justify="right")
    table.add_column("Key / Camelot", style="magenta")
    table.add_column("LUFS", justify="right")
    for c in cards:
        table.add_row(
            c.track_id,
            f"{c.duration_sec:.1f}",
            f"{c.bpm:.1f}",
            f"{c.bpm_stability:.2f}",
            f"{c.key} ({c.camelot})",
            f"{c.integrated_lufs:.1f}",
        )
    console.print(table)
    console.print(f"Saved AudioTrackCards to {out_jsonl}")


@app.command("run-defect-gym")
def defect_gym(
    out_dir: Annotated[
        Path, typer.Option(help="Directory to write defect gym reports & sample WAV.")
    ] = Path("data/processed"),
    num_pairs: Annotated[
        int, typer.Option(help="Number of clean track pairs to test across 5 defect types.")
    ] = 8,
    cc_audio_dir: Annotated[
        Path, typer.Option(help="Optional CC audio pool dir for a real-audio smoke render.")
    ] = Path("data/raw/cc_audio"),
) -> None:
    """Calibrate QA verifier recall on injected audio defects (Phase 2 exit gate: >95% recall)."""
    console.print(f"[bold]Running Phase 2 Defect Gym ({num_pairs} pairs x 5 defects)...[/bold]")
    report = run_defect_gym(out_dir=out_dir, num_pairs=num_pairs, cc_audio_dir=cc_audio_dir)
    table = Table(title="Phase 2 Defect Gym Calibration")
    table.add_column("Defect Type", style="cyan")
    table.add_column("Trials", justify="right")
    table.add_column("Detected", justify="right")
    table.add_column("Recall", justify="right", style="green")
    for d, st in report["per_defect"].items():
        table.add_row(d, str(st["trials"]), str(st["detected"]), f"{st['recall'] * 100:.1f}%")
    console.print(table)
    console.print(
        f"Overall Recall: [bold green]{report['overall_defect_recall'] * 100:.1f}%[/bold green] "
        f"| Clean FP Rate: [bold]{report['clean_false_positive_rate'] * 100:.1f}%[/bold] "
        f"| Exit Gate: [bold]{'PASS' if report['passed_exit_gate'] else 'FAIL'}[/bold]"
    )
    console.print(f"Report written to {out_dir / 'defect_gym_report.md'}")


@app.command("export-training-data")
def export_training_data(
    processed_dir: Annotated[
        Path, typer.Option(help="Directory containing Phase 1 sft_*.jsonl and dpo_train.jsonl.")
    ] = Path("data/processed"),
    out_dir: Annotated[
        Path, typer.Option(help="Output directory for Tinker/chat SFT & DPO JSONLs.")
    ] = Path("data/processed/training"),
    include_bronze: Annotated[
        bool,
        typer.Option(
            help="Include bronze-tier transitions in SFT train (default: Gold+Silver only)."
        ),
    ] = False,
) -> None:
    """Export Tinker/chat-ready SFT and DPO JSONLs plus token & USD cost estimates."""
    tiers = ("gold", "silver", "bronze") if include_bronze else ("gold", "silver")
    tier_str = "+".join(tiers)
    console.print(f"[bold]Exporting Phase 3 SFT ({tier_str}) and DPO chat datasets...[/bold]")
    summary = export_training_datasets(
        processed_dir=processed_dir,
        out_dir=out_dir,
        train_tiers=tiers,
    )
    table = Table(title="Phase 3 Exported Training Datasets")
    table.add_column("Split", style="cyan")
    table.add_column("Examples", justify="right")
    table.add_column("Avg Tokens", justify="right")
    table.add_column("Approx Total Tokens", justify="right", style="green")
    for name, st in summary["sft_splits"].items():
        table.add_row(
            f"sft_{name}",
            f"{st['examples']:,}",
            f"{st['avg_tokens_per_example']:.0f}",
            f"{st['approx_total_tokens']:,}",
        )
    dpo = summary["dpo_train"]
    if dpo.get("examples"):
        table.add_row(
            "dpo_train (2x pass)",
            f"{dpo['examples']:,}",
            f"{dpo['avg_tokens_per_pair']:.0f}",
            f"{dpo['approx_total_tokens']:,}",
        )
    console.print(table)

    cost_table = Table(title="Estimated Tinker Cost (2 ep SFT + 1 ep DPO)")
    cost_table.add_column("Model", style="cyan")
    cost_table.add_column("SFT (2 ep)", justify="right")
    cost_table.add_column("DPO (1 ep)", justify="right")
    cost_table.add_column("Total", justify="right", style="green")
    for model_id, c in summary["tinker_cost_estimates_usd"].items():
        cost_table.add_row(
            model_id,
            f"${c['sft_2_epochs_usd']:.2f}",
            f"${c['dpo_1_epoch_usd']:.2f}",
            f"${c['total_phase3_usd']:.2f}",
        )
    console.print(cost_table)
    console.print(f"Summary written to {out_dir / 'training_dataset_summary.md'}")


@app.command("train-sft")
def train_sft(
    base_model: Annotated[
        str, typer.Option(help="Tinker base model ID.")
    ] = "thinkingmachines/Inkling-2-Mini",
    train_jsonl: Annotated[Path, typer.Option(help="Path to sft_train_chat.jsonl.")] = Path(
        "data/processed/training/sft_train_chat.jsonl"
    ),
    val_jsonl: Annotated[Path, typer.Option(help="Path to sft_val_chat.jsonl.")] = Path(
        "data/processed/training/sft_val_chat.jsonl"
    ),
    out_dir: Annotated[Path, typer.Option(help="Directory for SFT run logs and summary.")] = Path(
        "data/processed/training/runs/sft_v1"
    ),
    save_name: Annotated[str, typer.Option(help="Tinker checkpoint name.")] = "ai-dj-sft-v1",
    lora_rank: Annotated[int, typer.Option(help="LoRA rank.")] = 32,
    learning_rate: Annotated[float, typer.Option(help="Peak learning rate.")] = 2e-4,
    batch_size: Annotated[int, typer.Option(help="Batch size per step.")] = 32,
    epochs: Annotated[int, typer.Option(help="Training epochs.")] = 2,
    max_steps: Annotated[int | None, typer.Option(help="Optional step cap for smoke runs.")] = None,
    dry_run: Annotated[
        bool,
        typer.Option(help="Validate data, schedule, and cost locally without Tinker API calls."),
    ] = False,
) -> None:
    """Run Stage 1 Tinker LoRA SFT on Gold/Silver DJ transitions."""
    cfg = SFTConfig(
        base_model=base_model,
        train_jsonl=train_jsonl,
        val_jsonl=val_jsonl,
        out_dir=out_dir,
        save_name=save_name,
        lora_rank=lora_rank,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        max_steps=max_steps,
    )
    res = run_tinker_sft(cfg, dry_run=dry_run)
    console.print(f"[bold green]SFT ({res['mode']}) complete.[/bold green] Output in {out_dir}")


@app.command("train-dpo")
def train_dpo(
    base_model: Annotated[
        str, typer.Option(help="Tinker base model ID.")
    ] = "thinkingmachines/Inkling-2-Mini",
    sft_state_checkpoint: Annotated[
        str | None, typer.Option(help="Optional Tinker state checkpoint from Stage 1 SFT.")
    ] = None,
    dpo_jsonl: Annotated[Path, typer.Option(help="Path to dpo_train_chat.jsonl.")] = Path(
        "data/processed/training/dpo_train_chat.jsonl"
    ),
    out_dir: Annotated[Path, typer.Option(help="Directory for DPO run logs and summary.")] = Path(
        "data/processed/training/runs/dpo_v1"
    ),
    save_name: Annotated[str, typer.Option(help="Tinker checkpoint name.")] = "ai-dj-dpo-v1",
    dpo_beta: Annotated[float, typer.Option(help="DPO KL beta.")] = 0.1,
    learning_rate: Annotated[float, typer.Option(help="Peak learning rate.")] = 1e-5,
    batch_size: Annotated[int, typer.Option(help="Pairs per step.")] = 16,
    epochs: Annotated[int, typer.Option(help="DPO epochs.")] = 1,
    max_steps: Annotated[int | None, typer.Option(help="Optional step cap for smoke runs.")] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            help="Validate DPO data, schedule, and cost locally without Tinker API calls."
        ),
    ] = False,
) -> None:
    """Run Stage 2 Tinker LoRA DPO on taste & off-phrase preference pairs."""
    cfg = DPOConfig(
        base_model=base_model,
        sft_state_checkpoint=sft_state_checkpoint,
        dpo_jsonl=dpo_jsonl,
        out_dir=out_dir,
        save_name=save_name,
        dpo_beta=dpo_beta,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        max_steps=max_steps,
    )
    res = run_tinker_dpo(cfg, dry_run=dry_run)
    console.print(f"[bold green]DPO ({res['mode']}) complete.[/bold green] Output in {out_dir}")


@app.command("eval-predictions")
def eval_predictions(
    split_jsonl: Annotated[
        Path, typer.Option(help="Ground-truth sft_val.jsonl or sft_test.jsonl.")
    ] = Path("data/processed/sft_test.jsonl"),
    predictions_jsonl: Annotated[
        Path, typer.Option(help="JSONL of {example_id, raw_output} model completions.")
    ] = Path("data/processed/training/predictions.jsonl"),
    model_name: Annotated[str, typer.Option(help="Label for the evaluated model.")] = "sft_v1",
    without_dj: Annotated[bool, typer.Option(help="Evaluate in without_dj mode.")] = False,
    out_md: Annotated[Path, typer.Option(help="Output Markdown comparison report path.")] = Path(
        "data/processed/training/model_eval_comparison.md"
    ),
) -> None:
    """Score a model's raw JSONL completions head-to-head against Phase 1 baselines."""
    report = evaluate_raw_predictions_jsonl(
        split_jsonl=split_jsonl,
        predictions_jsonl=predictions_jsonl,
        model_name=model_name,
        include_dj=not without_dj,
        out_md=out_md,
    )
    m = report["model_metrics"]["overall"]
    sh = report["schema_health"]
    ov_mae = m["overlap_beats_mae"]
    ov_str = f"{ov_mae:.1f}" if ov_mae is not None else "n/a"
    console.print(
        f"[bold]{model_name}[/bold] (n={report['num_examples']:,}) | "
        f"JSON={sh['json_valid_rate'] * 100:.1f}% | "
        f"R@1={m['recall_at_1']:.3f} | OvMAE={ov_str} | "
        f"StyleAcc={m['style_accuracy']:.3f}"
    )
    console.print(f"Comparison written to {out_md}")


@app.command("ingest-electronic-pool")
def ingest_electronic_pool_cmd(
    out_dir: Annotated[Path, typer.Option()] = Path("data/raw/cc_electronic"),
    cards_out: Annotated[Path, typer.Option()] = Path(
        "data/processed/electronic_track_cards.jsonl"
    ),
    max_tracks: Annotated[int, typer.Option()] = 24,
    max_shards: Annotated[int, typer.Option()] = 8,
    force: Annotated[bool, typer.Option()] = False,
) -> None:
    """Ingest MTG-Jamendo genre-filtered electronic CC tracks and extract MIR + timbre cards."""
    from ai_dj.audio.analyzer import analyze_audio_pool
    from ai_dj.audio.cc_pool import ingest_electronic_cc_pool

    manifest = ingest_electronic_cc_pool(
        out_dir=out_dir, max_tracks=max_tracks, max_shards=max_shards, force=force
    )
    cards = analyze_audio_pool(manifest, cards_out)
    stable = [c for c in cards if c.local_bpm_stability >= 0.70 and 105.0 <= c.bpm <= 155.0]
    console.print(
        f"Ingested & analyzed [bold]{len(cards)}[/bold] verified electronic tracks "
        f"([green]{len(stable)} club-stable 105–155 BPM[/green]) -> {cards_out}"
    )


@app.command("train-grpo")
def train_grpo_cmd(
    cards_jsonl: Annotated[Path, typer.Option()] = Path(
        "data/processed/electronic_track_cards.jsonl"
    ),
    out_dir: Annotated[Path, typer.Option()] = Path(
        "data/processed/training_v2/runs/inkling2_mini_grpo_v1"
    ),
    base_model: Annotated[str, typer.Option()] = "thinkingmachines/Inkling-2-Mini",
    sft_state_checkpoint: Annotated[str | None, typer.Option()] = None,
    num_steps: Annotated[int, typer.Option()] = 20,
    prompts_per_step: Annotated[int, typer.Option()] = 4,
    group_size: Annotated[int, typer.Option()] = 4,
    learning_rate: Annotated[float, typer.Option()] = 2e-5,
    save_name: Annotated[str, typer.Option()] = "ai-dj-inkling2-mini-grpo-v1",
    dry_run: Annotated[
        bool, typer.Option(help="Validate GRPO tasks & reward without Tinker.")
    ] = False,
) -> None:
    """Phase 4: Run DSP-verified Tinker LoRA GRPO on verified electronic CC tracks."""
    from ai_dj.rl.tinker_grpo import GRPOConfig, run_tinker_grpo

    cfg = GRPOConfig(
        cards_jsonl=cards_jsonl,
        out_dir=out_dir,
        base_model=base_model,
        sft_state_checkpoint=sft_state_checkpoint,
        num_steps=num_steps,
        prompts_per_step=prompts_per_step,
        group_size=group_size,
        learning_rate=learning_rate,
        save_name=save_name,
    )
    res = run_tinker_grpo(cfg, dry_run=dry_run)
    console.print(f"GRPO ({res['mode']}) complete. Output in [bold]{out_dir}[/bold]")


@app.command("render-model-set")
def render_model_set_cmd(
    sampler_checkpoint: Annotated[
        str, typer.Option(help="Tinker sampler_weights checkpoint path.")
    ],
    cards_jsonl: Annotated[Path, typer.Option()] = Path(
        "data/processed/electronic_track_cards.jsonl"
    ),
    base_model: Annotated[str, typer.Option()] = "thinkingmachines/Inkling-2-Mini",
    out_dir: Annotated[Path, typer.Option()] = Path("data/processed/electronic_grpo_miniset"),
    num_transitions: Annotated[int, typer.Option()] = 3,
) -> None:
    """Have a trained Tinker checkpoint plan and render a 44.1kHz stereo electronic mini-set."""
    from ai_dj.rl.tinker_grpo import plan_and_render_electronic_set

    report = plan_and_render_electronic_set(
        cards_jsonl=cards_jsonl,
        sampler_checkpoint=sampler_checkpoint,
        base_model=base_model,
        out_dir=out_dir,
        num_transitions=num_transitions,
    )
    n_tr = len(report["transitions"])
    console.print(
        f"Rendered {n_tr} transitions -> [bold]{report['showcase_wav']}[/bold] "
        f"({report['showcase_duration_sec']}s)"
    )


@app.command("run-all")
def run_all(
    raw_dir: Annotated[Path, typer.Option()] = Path("data/raw"),
    out_dir: Annotated[Path, typer.Option()] = Path("data/processed"),
    force: Annotated[bool, typer.Option()] = False,
) -> None:
    """Run ingest followed by build end-to-end."""
    ingest(raw_dir=raw_dir, force=force)
    build(raw_dir=raw_dir, out_dir=out_dir)


if __name__ == "__main__":
    app()
