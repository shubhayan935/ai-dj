"""Command-line interface for the AI DJ data pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

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
