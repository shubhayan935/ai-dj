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

app = typer.Typer(help="AI DJ data acquisition, cleaning, ranking, and dataset export CLI.")
console = Console()


@app.command()
def ingest(
    raw_dir: Annotated[
        Path, typer.Option(help="Directory to store raw downloaded data.")
    ] = Path("data/raw"),
    force: Annotated[
        bool, typer.Option(help="Re-download even if cached files exist.")
    ] = False,
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
    raw_dir: Annotated[
        Path, typer.Option(help="Directory containing ingested raw files.")
    ] = Path("data/raw"),
    out_dir: Annotated[
        Path, typer.Option(help="Output directory for clean datasets.")
    ] = Path("data/processed"),
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
    manifest_path: Annotated[
        Path, typer.Option(help="Path to CC audio manifest.json.")
    ] = Path("data/raw/cc_audio/manifest.json"),
    out_jsonl: Annotated[
        Path, typer.Option(help="Output JSONL path for AudioTrackCards.")
    ] = Path("data/processed/cc_track_cards.jsonl"),
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
