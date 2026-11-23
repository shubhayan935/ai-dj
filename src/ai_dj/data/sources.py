"""Data acquisition for real DJ mix corpora (MixesDB / mir-aidj / Hugging Face)."""

from __future__ import annotations

from pathlib import Path

import duckdb
import httpx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DJMIX_METADATA_URL = (
    "https://raw.githubusercontent.com/mir-aidj/djmix-dataset/main/dataset/djmix-dataset.json"
)
HF_TRANSITIONS_API = "https://huggingface.co/api/datasets/djmix/transitions/tree/main/data"
HF_TRANSITIONS_BASE = "https://huggingface.co/datasets/djmix/transitions/resolve/main"
HF_BEATS_TRACKS_API = "https://huggingface.co/api/datasets/djmix/beats-tracks/tree/main/data"
HF_BEATS_TRACKS_BASE = "https://huggingface.co/datasets/djmix/beats-tracks/resolve/main"

# All scalar columns from djmix/transitions (excludes huge DTW warping path arrays wp_*).
TRANSITION_SCALAR_COLUMNS = [
    "tran_id",
    "mix_id",
    "i_tran",
    "i_track_prev",
    "i_track_next",
    "track_id_prev",
    "track_id_next",
    "match_rate_prev",
    "match_rate_next",
    "matched_beats_prev",
    "matched_beats_next",
    "overlap_wpts",
    "overlap_beats",
    "tran_wpts",
    "extra_wpts_prev",
    "extra_wpts_next",
    "extra_beats_prev",
    "extra_beats_next",
    "last_wpt_prev",
    "last_wpt_next",
    "total_wpt_prev",
    "total_wpt_next",
    "matched_time_mix_prev",
    "matched_time_mix_next",
    "matched_time_track_prev",
    "matched_time_track_next",
    "timestamp_prev",
    "timestamp_next",
    "key_change_prev",
    "key_change_next",
    "mix_cue_in_beat_prev",
    "mix_cue_in_beat_next",
    "mix_cue_out_beat_prev",
    "mix_cue_out_beat_next",
    "track_cue_in_beat_prev",
    "track_cue_in_beat_next",
    "track_cue_out_beat_prev",
    "track_cue_out_beat_next",
    "mix_cue_in_time_prev",
    "mix_cue_in_time_next",
    "mix_cue_out_time_prev",
    "mix_cue_out_time_next",
    "track_cue_in_time_prev",
    "track_cue_in_time_next",
    "track_cue_out_time_prev",
    "track_cue_out_time_next",
    "cost_prev",
    "cost_next",
]


def _list_hf_parquet_urls(api_url: str, base_url: str) -> list[str]:
    resp = httpx.get(api_url, timeout=60.0, follow_redirects=True)
    resp.raise_for_status()
    files = resp.json()
    return [
        f"{base_url}/{entry['path']}"
        for entry in sorted(files, key=lambda x: x["path"])
        if entry["path"].endswith(".parquet")
    ]


def fetch_djmix_metadata(raw_dir: Path, force: bool = False) -> Path:
    """Download djmix-dataset.json (5,040 MixesDB mixes with tracklists and tags)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / "djmix-dataset.json"
    if out_path.exists() and not force:
        return out_path

    with httpx.stream("GET", DJMIX_METADATA_URL, timeout=120.0, follow_redirects=True) as resp:
        resp.raise_for_status()
        with out_path.open("wb") as fh:
            for chunk in resp.iter_bytes(chunk_size=1 << 20):
                fh.write(chunk)
    return out_path


def fetch_transitions_scalar(raw_dir: Path, force: bool = False) -> Path:
    """Stream scalar columns from all djmix/transitions shards on Hugging Face via DuckDB."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / "transitions_scalar.parquet"
    if out_path.exists() and not force:
        return out_path

    urls = _list_hf_parquet_urls(HF_TRANSITIONS_API, HF_TRANSITIONS_BASE)
    cols_sql = ", ".join(TRANSITION_SCALAR_COLUMNS)
    urls_sql = ", ".join(f"'{u}'" for u in urls)

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(
        f"""
        COPY (
            SELECT {cols_sql}
            FROM read_parquet([{urls_sql}])
        ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    con.close()
    return out_path


def summarize_beat_array(
    beats: list[float] | np.ndarray,
) -> tuple[float | None, float | None, float | None, int]:
    """Compute (bpm, bpm_stability, duration_sec, beat_count) from a track's beat timestamps."""
    arr = np.asarray(beats, dtype=np.float64)
    n = int(arr.size)
    if n < 8:
        return None, None, float(arr[-1]) if n > 0 else None, n

    ibis = np.diff(arr)
    valid = ibis[(ibis > 0.2) & (ibis < 2.0)]
    if valid.size < 4:
        return None, None, float(arr[-1]), n

    median_ibi = float(np.median(valid))
    if n >= 16:
        # 8-beat spans cancel 10ms frame quantization (e.g. 128.0 vs 127.66 BPM).
        spans = (arr[8:] - arr[:-8]) / 8.0
        valid_spans = spans[(spans > 0.2) & (spans < 2.0)]
        if valid_spans.size >= 4:
            median_ibi = float(np.median(valid_spans))
    bpm = 60.0 / median_ibi
    q75, q25 = np.percentile(valid, [75, 25])
    iqr_ratio = float((q75 - q25) / median_ibi)
    # 1.0 = rock-solid beatgrid; drops toward 0 as IBI jitter exceeds 25% of beat length.
    stability = float(np.clip(1.0 - (iqr_ratio / 0.25), 0.0, 1.0))
    duration_sec = float(arr[-1])
    return round(bpm, 2), round(stability, 4), round(duration_sec, 2), n


def fetch_track_beats_summary(raw_dir: Path, force: bool = False) -> Path:
    """Download djmix/beats-tracks and summarize per-track BPM, stability, and duration."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / "tracks_bpm.parquet"
    if out_path.exists() and not force:
        return out_path

    urls = _list_hf_parquet_urls(HF_BEATS_TRACKS_API, HF_BEATS_TRACKS_BASE)
    local_beats = raw_dir / "_beats_tracks_raw.parquet"
    if not local_beats.exists() or force:
        with httpx.stream("GET", urls[0], timeout=300.0, follow_redirects=True) as resp:
            resp.raise_for_status()
            with local_beats.open("wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=1 << 20):
                    fh.write(chunk)

    pf = pq.ParquetFile(local_beats)
    track_ids: list[str] = []
    bpms: list[float | None] = []
    stabilities: list[float | None] = []
    durations: list[float | None] = []
    beat_counts: list[int] = []

    for batch in pf.iter_batches(batch_size=2048, columns=["track_id", "beats"]):
        tids = batch.column("track_id").to_pylist()
        beats_col = batch.column("beats").to_pylist()
        for tid, beats in zip(tids, beats_col, strict=True):
            bpm, stab, dur, count = summarize_beat_array(beats or [])
            track_ids.append(tid)
            bpms.append(bpm)
            stabilities.append(stab)
            durations.append(dur)
            beat_counts.append(count)

    table = pa.table(
        {
            "track_id": pa.array(track_ids, type=pa.string()),
            "bpm": pa.array(bpms, type=pa.float64()),
            "bpm_stability": pa.array(stabilities, type=pa.float64()),
            "duration_sec": pa.array(durations, type=pa.float64()),
            "beat_count": pa.array(beat_counts, type=pa.int32()),
        }
    )
    pq.write_table(table, out_path, compression="zstd")
    local_beats.unlink(missing_ok=True)
    return out_path
