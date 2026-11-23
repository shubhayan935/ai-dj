"""Markdown report generator for data quality, filtering funnel, and split diagnostics."""

from __future__ import annotations

from pathlib import Path


def write_markdown_report(summary: dict, out_path: Path) -> Path:
    """Render data_quality_report.json into a clean human-readable Markdown report."""
    lines: list[str] = [
        "# AI DJ Data Quality & Ranking Report",
        "",
        "## Corpus Funnel",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Raw Mixes Loaded | {summary['total_mixes_loaded']:,} |",
        f"| Unique Tracks Indexed | {summary['total_tracks_indexed']:,} |",
        f"| Raw Aligned Transitions | {summary['total_raw_transitions']:,} |",
        f"| Clean Transitions (Passed Hard Gates) | {summary['total_clean_transitions']:,} ({summary['retention_rate_pct']}%) |",
        f"| Gold / Silver / Bronze | {summary['tier_counts'].get('gold', 0):,} / {summary['tier_counts'].get('silver', 0):,} / {summary['tier_counts'].get('bronze', 0):,} |",
        f"| DPO Train Pairs (Taste + Execution) | {summary['dpo_train_pairs']:,} |",
        "",
        "## Hard-Gate Failure Breakdown (Non-Exclusive)",
        "",
        "| Failure Reason | Count |",
        "| --- | ---: |",
    ]
    for reason, count in summary["failure_reason_counts"].items():
        lines.append(f"| `{reason}` | {count:,} |")

    lines += [
        "",
        "## DJ-Disjoint Splits & Leakage Check",
        "",
        "| Split | SFT Examples | Unique DJs |",
        "| --- | ---: | ---: |",
    ]
    for sp in ("train", "val", "test"):
        lines.append(
            f"| `{sp}` | {summary['sft_split_counts'].get(sp, 0):,} | {summary['unique_djs_per_split'].get(sp, 0):,} |"
        )
    leak = summary["dj_leakage_check"]
    lines += [
        "",
        f"DJ leakage overlaps: train∩val={leak['train_val_overlap']}, "
        f"train∩test={leak['train_test_overlap']}, val∩test={leak['val_test_overlap']}.",
        "",
        "## Clean Corpus Distributions",
        "",
        "| Feature | p10 | p50 | p90 | Mean |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for feat, stats in summary["distributions"].items():
        if stats:
            lines.append(
                f"| `{feat}` | {stats['p10']} | {stats['p50']} | {stats['p90']} | {stats['mean']} |"
            )

    lines += [
        "",
        "## Transition Styles",
        "",
        "| Style | Count |",
        "| --- | ---: |",
    ]
    for style, count in summary["transition_style_counts"].items():
        lines.append(f"| `{style}` | {count:,} |")

    lines += [
        "",
        "## Top 10 Gold Transitions",
        "",
        "| Score | DJ | Prev Track | Next Track | Overlap (beats) | Style |",
        "| ---: | --- | --- | --- | ---: | --- |",
    ]
    for item in summary["top_10_gold_transitions"]:
        lines.append(
            f"| {item['score']:.4f} | {item['dj']} | {item['prev']} | {item['next']} | "
            f"{item['overlap_beats']} | `{item['style']}` |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
